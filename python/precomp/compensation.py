"""Springback compensation by displacement adjustment.

Displacement adjustment (DA; Gan and Wagoner, 2004) iterates

    c_(k+1) = c_k - alpha (f(c_k) - t),

with t the target, c_k the commanded surface and f a predictor of the formed
surface - a finite-element simulation, a learned surrogate, a combination, or
the measured part itself for a one-step shop-floor update. For a linear
springback f(c) = c + G c with spectral radius rho(G) < 1 and alpha = 1, the
error on the adjusted region contracts as e_(k+1) = -G e_k: geometric
convergence at rate rho(G).

Every update here is conditioned the same way:

* the error is measured vertically (z_formed - z_target at each target node)
  or along the target normal (`signed_deviation`), in which case the
  commanded surface is moved along the target normals and resampled;
* optional Gaussian smoothing of the update at a length scale [m], normalised
  over the adjusted region so the flange does not leak into it;
* the hold region (by default the target's flange) is kept at the target
  height, z = 0: the tool never presses there;
* the commanded surface is kept at z <= 0 (SPIF only pushes the sheet down),
  or below `upper_bound`, the highest command the setup's tools can realise
  node by node: a support that pushes the sheet up from below (the DSIF rim
  pass, `precomp.fea.support.command_upper_bound`) lets the command rise
  above the sheet plane in the band it sweeps, and nowhere else;
* a formability projection: the smallest surface above the commanded one
  whose slope nowhere exceeds `max_wall_angle_deg` (`limit_wall_angle`). It
  only ever raises the surface - it never cuts deeper than asked. It is the
  forming tool's limit and applies to the command below the sheet plane; a
  command above it (the support's) is kept as it is;
* optionally (`tool_radius`), the tool's reach: the command is raised to
  the lowest surface a ball of that radius pressing from above forms
  (`precomp.toolpath.tool_reach_surface`, the closing by the ball). A pit
  narrower than the tool, which DA digs where the formed part keeps coming
  out too shallow, is never formed; without the clip the next update digs
  it deeper still, and a model trained on the command learns depth that
  never reached the sheet. The tool path is the same with and without the
  clip (the drop-cutter surface does not change).

All heights are in metres, angles in degrees where named `_deg`.

Where this fits: "Compensation": the DA loop, run with the FE model for the FE-DA baselines
    (FEAPredictor) and with the surrogate under ml/compensate.py.
Called from: springback_fine_benchmark.py:run_chain, ml/compensate.py:surrogate_compensate,
    ml/generate.py:compensated_commanded.
Calls into: metrology (masks, metrics), fea.runner (simulate), toolpath (tool_reach_surface).
Start reading at: displacement_adjustment, then FEAPredictor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

import numpy as np
from scipy import ndimage
from scipy.interpolate import griddata

from ._util import PathLike, PrecompError
from .geometry.heightmap import HeightMap
from .geometry.parts import MAX_WALL_ANGLE_DEG
from .metrology import (flange_mask, metrics, part_mask, signed_deviation, vertical_deviation)

#: A predictor maps a commanded surface to the formed surface on the same grid.
Predictor = Callable[[HeightMap], HeightMap]


# ---------------------------------------------------------------------------
# Formability projection and update conditioning
# ---------------------------------------------------------------------------
def limit_wall_angle(hm: HeightMap, max_wall_angle_deg: float = MAX_WALL_ANGLE_DEG,
                     radius_cells: int = 4, max_rounds: int = 8) -> HeightMap:
    """The smallest surface z' >= z whose slope nowhere exceeds the limit.

    z'(x) = max over y of [z(y) - s d(x, y)], computed by iterated grey-scale
    dilation with the cone ``-s |d|`` over the offsets |d| <= `radius_cells`
    grid spacings until nothing changes (d is then the shortest path through
    those offsets, within about 1 % of the Euclidean distance). Steep regions
    are filled from their upper edge; slopes already within the limit are
    untouched; the surface only rises.

    The guarantee is stated in the measure the rest of the package uses:
    the central-difference wall angle (`HeightMap.wall_angle`) of the result
    is at most `max_wall_angle_deg`. A surface of facets can show a
    central-difference slope above the cone slope s: the cone bounds each of
    |z_x| and |z_y| by s, so their combination by at most sqrt(2) s. The
    cone slope therefore starts at tan(limit); if the result exceeds the
    limit, s is reduced once by the measured excess and then bisected, for
    up to `max_rounds` rounds, between the largest failing slope and
    tan(limit) / sqrt(2), which always satisfies it. (Reducing by the excess
    alone stalled where the steepest central difference combines an x and a
    y difference the cone does not bind yet: the measure did not move while
    s shrank by 0.07 % per round.) Metadata records the nodes raised, the
    largest raise [m] and the cone slope used.
    """
    if not 0 < max_wall_angle_deg < 90:
        raise ValueError("max_wall_angle_deg must lie in (0, 90)")
    limit = math.tan(math.radians(max_wall_angle_deg))
    r = int(radius_cells)
    if r < 2:
        raise ValueError("radius_cells must be >= 2 (the central differences span two cells)")
    k = np.arange(-r, r + 1)
    DX, DY = np.meshgrid(k, k, indexing="xy")
    footprint = np.hypot(DX, DY) <= r + 1e-9
    dist = np.hypot(DX, DY) * hm.grid.h

    def dilate(s: float) -> Tuple[np.ndarray, float]:
        structure = np.where(footprint, -s * dist, 0.0)
        zz = hm.z.copy()
        while True:
            znew = ndimage.grey_dilation(zz, footprint=footprint, structure=structure,
                                         mode="nearest")
            if np.array_equal(znew, zz):
                break
            zz = znew
        gy, gx = np.gradient(zz, hm.grid.h)
        return zz, float(np.hypot(gx, gy).max())

    def ok(m: float) -> bool:
        return m <= limit * (1.0 + 1e-12)

    slope = limit
    z, measured = dilate(slope)
    if not ok(measured):
        hi = slope
        lo = limit / math.sqrt(2.0) * (1.0 - 1e-9)
        z_lo, m_lo = dilate(lo)
        if not ok(m_lo):
            raise PrecompError(f"limit_wall_angle: the central-difference slope exceeds "
                               f"{max_wall_angle_deg} deg even with the cone slope "
                               f"tan(limit) / sqrt(2)")
        trial = slope * limit / measured * (1.0 - 1e-6)
        for _ in range(max_rounds):
            if not lo < trial < hi:
                trial = 0.5 * (lo + hi)
            z_t, m_t = dilate(trial)
            if ok(m_t):
                lo, z_lo = trial, z_t
            else:
                hi = trial
            if hi - lo <= 1e-3 * lo:
                break
            trial = 0.5 * (lo + hi)
        slope, z = lo, z_lo
    raised = z - hm.z
    meta = dict(hm.metadata)
    meta["wall_angle_limit"] = {"max_wall_angle_deg": float(max_wall_angle_deg),
                                "cone_slope": float(slope),
                                "nodes_raised": int(np.sum(raised > 0)),
                                "max_raise_m": float(raised.max())}
    return HeightMap(hm.grid, z, hm.mask, meta)


def smooth_update(update: np.ndarray, region: np.ndarray, sigma: float, h: float) -> np.ndarray:
    """Gaussian smoothing at `sigma` [m] normalised over `region`: the
    region-weighted average ``S(u w) / S(w)`` (w the region indicator),
    zero outside the region, so values outside do not leak in."""
    w = region.astype(float)
    num = ndimage.gaussian_filter(update * w, sigma / h, mode="nearest")
    den = ndimage.gaussian_filter(w, sigma / h, mode="nearest")
    out = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
    return np.where(region, out, 0.0)


UpperBound = Union[float, np.ndarray, Callable[[np.ndarray], np.ndarray]]


def _upper(upper_bound: Optional[UpperBound], shape: Tuple[int, int],
           z: Optional[np.ndarray] = None) -> Union[float, np.ndarray]:
    if upper_bound is None:
        return 0.0
    if callable(upper_bound):
        upper_bound = upper_bound(np.zeros(shape) if z is None else z)
    ub = np.asarray(upper_bound, dtype=float)
    if ub.ndim and ub.shape != shape:
        raise ValueError(f"upper_bound has shape {ub.shape}, the grid needs {shape}")
    if not np.all(np.isfinite(ub)) or np.any(ub < 0.0):
        raise ValueError("upper_bound must be finite and >= 0 (the sheet plane is always "
                         "reachable: the tool need not touch)")
    return ub


def _condition(commanded: HeightMap, z_new: np.ndarray, hold: np.ndarray, target: HeightMap,
               max_wall_angle_deg: Optional[float],
               upper_bound: Optional[UpperBound] = None,
               tool_radius: Optional[float] = None) -> HeightMap:
    z = np.where(hold, target.z, z_new)
    z = np.minimum(z, _upper(upper_bound, z.shape, z))
    hm = commanded.with_z(z)
    if max_wall_angle_deg is not None:
        if not (z > 0.0).any():
            hm = limit_wall_angle(hm, max_wall_angle_deg)
        else:
            # the forming tool's limit, on its part of the command (below the
            # plane); the part above it is the support's and stays as it is
            below = limit_wall_angle(hm.with_z(np.minimum(z, 0.0)), max_wall_angle_deg)
            hm = below.with_z(np.where(z > 0.0, z, below.z))
    if tool_radius is not None:
        from .toolpath import tool_reach_surface

        # the forming tool's reach, on its part of the command (as above)
        z = hm.z
        reach = tool_reach_surface(hm.with_z(np.minimum(z, 0.0)), tool_radius)
        hm = reach.with_z(np.where(z > 0.0, z, reach.z))
    return hm


def _apply_update(commanded: HeightMap, target: HeightMap, error: HeightMap, adjust: np.ndarray,
                  alpha: float, direction: str, smoothing: Optional[float]) -> np.ndarray:
    """New commanded heights (before conditioning) from an error field."""
    region = adjust & error.mask
    upd = np.where(region, -alpha * error.z, 0.0)
    if smoothing:
        upd = smooth_update(upd, region, smoothing, target.grid.h)
    if direction == "vertical":
        return commanded.z + upd
    # Normal: move the commanded surface along the target normals, resample.
    n = target.normals()
    X, Y = target.grid.mesh()
    P = np.stack([X + upd * n[..., 0], Y + upd * n[..., 1], commanded.z + upd * n[..., 2]], -1)
    pts = P.reshape(-1, 3)
    z = griddata(pts[:, :2], pts[:, 2], (X, Y), method="linear")
    ok = np.isfinite(z)
    return np.where(ok, z, commanded.z)


def error_field(formed: HeightMap, target: HeightMap, direction: str = "vertical") -> HeightMap:
    """The DA error on the target grid [m]: vertical z_f - z_t, or the signed
    normal deviation (positive on the tool side)."""
    if direction == "vertical":
        return vertical_deviation(formed, target)
    if direction == "normal":
        return signed_deviation(formed, target)
    raise ValueError("direction must be 'vertical' or 'normal'")


# ---------------------------------------------------------------------------
# Displacement adjustment
# ---------------------------------------------------------------------------
@dataclass
class DAResult:
    """Outcome of `displacement_adjustment`.

    commanded : the commanded surface whose prediction was best (lowest RMS
        error over the adjusted region) - a verified shape.
    formed : its predicted formed surface.
    best_iteration : index (0-based) of that prediction in `history`.
    proposed : the update computed from the last prediction, not yet
        predicted itself (use it to continue, or verify it).
    history : per prediction: iteration, the error metrics over the adjusted
        region (rms, mae, max_abs, p95_abs, bias, std, n [m]), the RMS of the
        update it produced, and the commanded depth and wall angle.
    converged : the RMS error reached `tolerance`.
    """

    target: HeightMap
    commanded: HeightMap
    formed: HeightMap
    best_iteration: int
    proposed: HeightMap
    history: List[Dict[str, Any]] = field(default_factory=list)
    converged: bool = False

    @property
    def rms_history(self) -> List[float]:
        return [h["error"]["rms"] for h in self.history]


def displacement_adjustment(target: HeightMap, predictor: Predictor, *, iterations: int = 5,
                            alpha: float = 1.0, direction: str = "vertical",
                            smoothing: Optional[float] = None,
                            hold_mask: Optional[np.ndarray] = None,
                            adjust_mask: Optional[np.ndarray] = None,
                            max_wall_angle_deg: Optional[float] = MAX_WALL_ANGLE_DEG,
                            upper_bound: Optional[UpperBound] = None,
                            tool_radius: Optional[float] = None,
                            tolerance: Optional[float] = None,
                            initial: Optional[HeightMap] = None,
                            callback: Optional[Callable[[int, HeightMap, HeightMap, HeightMap],
                                                        None]] = None) -> DAResult:
    """Displacement adjustment of the commanded surface towards `target`.

    Parameters
    ----------
    target : the desired tool-side surface [m].
    predictor : commanded HeightMap -> formed HeightMap (resampled onto the
        target grid when on another grid).
    iterations : number of predictions (the last one's update is returned
        as `proposed`).
    alpha : relaxation factor (1 = full correction).
    direction : "vertical" or "normal" (see the module docstring).
    smoothing : Gaussian length scale [m] of the update, or None.
    hold_mask : nodes held at the target height (default: the target flange,
        z >= -1 um). adjust_mask : nodes updated and measured (default: the
        part, z < -1 um).
    max_wall_angle_deg : formability limit, or None for no projection.
    upper_bound : the highest command per node [m]: (ny, nx), a number
        >= 0, or a function of the proposed heights returning (ny, nx);
        None keeps the command at the sheet plane, z <= 0 (a tool pressing
        from above). A support that pushes up gives it with
        `precomp.fea.support.command_upper_bound(setup, target)` (and its
        masks with `compensation_masks`).
    tool_radius : clip every update to the reach of a ball tool of this
        radius [m] (`setup.tool_radius`; see the module docstring), or None.
    tolerance : stop when the RMS error over the adjusted region is <= this [m].
    initial : the first commanded surface (default: the target).
    callback : called as callback(k, commanded, formed, error) after each
        prediction.
    """
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if not alpha > 0:
        raise ValueError("alpha must be > 0")
    # Regions: hold (default the flange) stays at the target; adjust (default the part) moves
    hold = flange_mask(target) if hold_mask is None else np.asarray(hold_mask, bool)
    adjust = part_mask(target) if adjust_mask is None else np.asarray(adjust_mask, bool)
    adjust = adjust & ~hold
    if not adjust.any():
        raise ValueError("the adjusted region is empty (a flat target?)")
    c = (target if initial is None else initial).copy()
    if not c.grid.matches(target.grid):
        c = c.resample(target.grid)
    history: List[Dict[str, Any]] = []
    best = None
    converged = False
    proposed = c
    for k in range(iterations):
        # Predict the formed surface of command c and measure its error on the adjusted region
        f = predictor(c)
        if not f.grid.matches(target.grid):
            f = f.resample(target.grid)
        err = error_field(f, target, direction)
        m = metrics(err, adjust)
        entry: Dict[str, Any] = {"iteration": k, "error": m,
                                 "commanded_depth_m": c.depth,
                                 "commanded_max_wall_angle_deg":
                                     float(np.degrees(c.wall_angle()[adjust].max()))}
        # Keep the best command so far (lowest RMS error): it is the one returned
        if best is None or m["rms"] < best[0]:
            best = (m["rms"], k, c, f)
        if callback is not None:
            callback(k, c, f, err)
        # DA update c - alpha * error, then conditioning (hold, upper bound, wall angle, reach)
        z_new = _apply_update(c, target, err, adjust, alpha, direction, smoothing)
        proposed = _condition(c, z_new, hold, target, max_wall_angle_deg, upper_bound,
                              tool_radius)
        entry["update_rms_m"] = float(np.sqrt(np.mean((proposed.z - c.z)[adjust] ** 2)))
        history.append(entry)
        if tolerance is not None and m["rms"] <= tolerance:
            converged = True
            break
        c = proposed
    _, kbest, cbest, fbest = best
    cbest = cbest.copy()
    cbest.metadata["compensation"] = {"method": "displacement_adjustment", "alpha": alpha,
                                      "direction": direction, "iteration": kbest,
                                      "smoothing_m": smoothing, "tool_radius_m": tool_radius}
    return DAResult(target, cbest, fbest, kbest, proposed, history, converged)


# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------
@runtime_checkable
class FieldModel(Protocol):
    """What a learned model must provide to act as a predictor.

    ``predict_deviation(commanded, setup) -> (mean, std)``: the vertical
    deviation dz = z_formed - z_commanded [m] on the commanded grid, as
    (ny, nx) arrays; `std` [m] may be None when the model has no uncertainty.
    `precomp.ml.DeviationSurrogate` implements it.
    """

    def predict_deviation(self, commanded: HeightMap, setup: Any
                          ) -> Tuple[np.ndarray, Optional[np.ndarray]]: ...


class FEAPredictor:
    """Predict the formed surface by a sparlab_form simulation (cached).

    Each call runs (or fetches from the cache in `work_dir`) the simulation of
    the commanded surface with `setup` and returns the formed surface of
    `step` (default the last: after release) on the commanded grid. The
    results of all calls are kept in `results`. `target` is the part the
    fixture is made for: a setup with a support (`FormingSetup.support`)
    builds the backing plate's opening and the rim pass band from its
    outline, the same for every commanded iterate (without it, from each
    command's own outline, which moves as DA changes the rim).
    """

    def __init__(self, setup: Any, work_dir: PathLike, *, step: Union[int, str] = -1,
                 cache: bool = True, target: Optional[HeightMap] = None):
        self.setup = setup
        self.work_dir = Path(work_dir)
        self.step = step
        self.cache = cache
        self.target = target
        self.results: List[Any] = []

    def __call__(self, commanded: HeightMap) -> HeightMap:
        from .fea.runner import simulate

        res = simulate(self.setup, commanded, self.work_dir, cache=self.cache,
                       target=self.target)
        self.results.append(res)
        return res.formed_surface(self.step, grid=commanded.grid)


class SurrogatePredictor:
    """Predict the formed surface as commanded + a learned vertical deviation.

    `model` follows `FieldModel`. `__call__` returns the mean prediction;
    `predict_with_uncertainty` also returns the model's std [m] (or None).
    """

    def __init__(self, model: FieldModel, setup: Any):
        if not hasattr(model, "predict_deviation"):
            raise TypeError("the model must implement predict_deviation(commanded, setup)")
        self.model = model
        self.setup = setup

    def predict_with_uncertainty(self, commanded: HeightMap
                                 ) -> Tuple[HeightMap, Optional[np.ndarray]]:
        mean, std = self.model.predict_deviation(commanded, self.setup)
        mean = np.asarray(mean, dtype=float)
        if mean.shape != commanded.grid.shape:
            raise PrecompError(f"the model returned a deviation of shape {mean.shape}, "
                               f"expected {commanded.grid.shape}")
        formed = commanded.with_z(commanded.z + mean,
                                  metadata={"source": "surrogate",
                                            "model": type(self.model).__name__})
        return formed, None if std is None else np.asarray(std, dtype=float)

    def __call__(self, commanded: HeightMap) -> HeightMap:
        return self.predict_with_uncertainty(commanded)[0]
