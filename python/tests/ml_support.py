"""Test-only ML helpers: what the tests need that the workflow does not.

ProxySimulator - a fast analytic springback stand-in, NOT physics, labelled
    "proxy - not physics" wherever it appears. The ML tests train and
    compensate on its data in seconds instead of hours of SparLab runs.
simulate_samples - `precomp.ml.generate` into a temporary data set.
grouped_split, family_split, index_frame - the tests' train / calibration /
    test splits by whole part or family.
partition_coverage - repeated split-conformal calibration over random
    calibration / test partitions of held-out parts (how the tests check that
    ConformalCalibrator's intervals are valid).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import ndimage

from precomp._util import PrecompError
from precomp.geometry.heightmap import HeightMap
from precomp.ml.dataset import Dataset, Sample
from precomp.ml.features import as_setup
from precomp.ml.generate import DesignPoint, SimOutcome, generate
from precomp.ml.uncertainty import ConformalCalibrator, min_parts

#: Version of the proxy's formulas; part of every proxy sample's fidelity tag.
PROXY_VERSION = "proxy-v1"
PROXY_LABEL = "proxy - not physics"


# ---------------------------------------------------------------------------
# Proxy simulator
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProxyParams:
    """Coefficients of the proxy deviation (SI; lengths in m unless noted).

    rim_amplitude : under-forming near the rim at the reference material and
        thickness for a vertical wall; rim_decay : its decay length inside the
        part [tool radii]; rim_depth_scale : depth over which it builds up.
    pillow_amplitude, pillow_length : the bulge of flat bases and how fast it
        grows with the distance from the base edge; flat_angle_deg : wall
        angle below which a region counts as flat.
    curvature : unclamping curvature [1/m] of the whole sheet for a part
        `curvature_depth` deep of the reference material and thickness.
    strain_ref : flow stress at plastic strain 0.2 / E of the reference
        material (AA5754-O); thickness_ref, tool_radius_ref, step_down_ref,
        friction_ref : the reference process; friction_gain : relative change
        of the rim term per unit of friction.
    smoothing : Gaussian smoothing of the local terms [tool radii].
    noise, noise_seed : optional white noise [m] (e.g. to mimic a scan).
    """

    rim_amplitude: float = 8e-4
    rim_decay: float = 2.5
    rim_depth_scale: float = 0.01
    pillow_amplitude: float = 5e-4
    pillow_length: float = 0.012
    flat_angle_deg: float = 12.0
    curvature: float = 0.1
    curvature_depth: float = 0.025
    strain_ref: float = 3.8e-3
    thickness_ref: float = 1e-3
    tool_radius_ref: float = 5e-3
    step_down_ref: float = 5e-4
    friction_ref: float = 0.1
    friction_gain: float = 1.0
    smoothing: float = 0.5
    noise: float = 0.0
    noise_seed: int = 0


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class ProxySimulator:
    """Analytic springback stand-in. NOT physics-validated.

    dz = S_R(rim + pillow) + global (+ noise), with, for commanded depth d,
    wall angle theta and signed rim distance s (positive inside):

    * rim: under-forming (dz > 0, the part too shallow) near the rim,
      A sin(theta_2.5R) exp(-s / (rim_decay R)) inside and a short bending
      zone exp(-(s / R)^2) outside, A growing with the depth and with
      (sigma_f(0.2) / E)^0.75 (t_ref / t)^0.5 (1 + gain (mu - mu_ref));
    * pillow: flat bases bulge up by P (1 - exp(-(e / pillow_length)^2)),
      e the distance inside the flat region from its edge, P growing with
      (sigma_f(0.2) / E)^0.5, sqrt(R / R_ref), sqrt(step_down / step_ref);
    * global: after a 3-2-1 release the whole sheet bends,
      dz = (kx (a^2 - x^2) + ky (a^2 - y^2)) / 2 with a the support position
      (zero at the supports), kx + ky = 2 kappa, kappa proportional to depth,
      sigma_f(0.2) / E and t_ref / t, split between x and y by the depth field's
      second moments; with "clamped_only" a quarter of it inside the window.

    It has the global + local structure the models must capture, depends on
    the commanded shape (so displacement adjustment on it is non-trivial) and
    on every process and material descriptor that matters to it. Every
    sample, metric and plot made from it is labelled "proxy - not physics".
    It knows single-point forming only: as a simulator or a model it refuses
    a setup with support from below (`FormingSetup.support`) rather than give
    single-point numbers under the supported setup's name.

    `run` (the `Simulator` protocol) also builds the setup's tool path of
    every job, as the deck builder does, and fails the job without one;
    `check_toolpath=False` skips that (the deviation does not depend on it).
    """

    source = "proxy"
    label = PROXY_LABEL
    #: as a model: what its numbers are
    data_source = PROXY_LABEL

    def __init__(self, params: Optional[ProxyParams] = None, *, check_toolpath: bool = True,
                 **changes: Any):
        p = params or ProxyParams()
        self.params = dataclasses.replace(p, **changes) if changes else p
        self.check_toolpath = bool(check_toolpath)

    @property
    def fidelity(self) -> str:
        return PROXY_VERSION

    def shifted(self, **changes: Any) -> "ProxySimulator":
        """A proxy with some coefficients changed (e.g. a 'real' process that
        differs from the simulated one, for transfer-learning tests)."""
        return ProxySimulator(dataclasses.replace(self.params, **changes),
                              check_toolpath=self.check_toolpath)

    def describe(self) -> Dict[str, Any]:
        return {"name": "ProxySimulator", "version": PROXY_VERSION, "label": PROXY_LABEL,
                "params": dataclasses.asdict(self.params)}

    def __getstate__(self) -> Dict[str, Any]:
        return {"params": dataclasses.asdict(self.params),
                "check_toolpath": self.check_toolpath}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.params = ProxyParams(**state["params"])
        self.check_toolpath = bool(state.get("check_toolpath", True))

    def deviation(self, commanded: HeightMap, setup: Any) -> np.ndarray:
        """(ny, nx) dz [m] of forming `commanded` with `setup` (single-point
        forming; PrecompError for a setup with a support)."""
        p = self.params
        s = as_setup(setup)
        if s.support != "none":
            raise PrecompError(
                f"the proxy has no model of support from below (support {s.support!r}); its "
                "dz would be single-point forming's under the supported setup's name - "
                "simulate with sparlab_form (SparlabSimulator)")
        m = s.material
        g = commanded.grid
        h = g.h
        R = s.tool_radius
        depth = np.maximum(-commanded.z, 0.0)
        part = commanded.mask & (commanded.z < -1e-6)
        if not part.any():
            return np.zeros(g.shape)
        D = float(depth[part].max())
        strain = float(m.flow_stress(0.2)) / m.youngs_modulus / p.strain_ref
        thick = p.thickness_ref / s.thickness
        X, Y = g.mesh()
        # rim under-forming
        d_in = ndimage.distance_transform_edt(part) * h - 0.5 * h
        d_out = ndimage.distance_transform_edt(~part) * h - 0.5 * h
        theta_rim = commanded.wall_angle(2.5 * R)
        amp = (p.rim_amplitude * strain ** 0.75 * thick ** 0.5
               * max(0.0, 1.0 + p.friction_gain * (s.friction - p.friction_ref))
               * (1.0 - math.exp(-D / p.rim_depth_scale)))
        shape = np.where(part, np.exp(-d_in / (p.rim_decay * R)), np.exp(-(d_out / R) ** 2))
        rim = amp * np.sin(theta_rim) * shape
        # pillow on flat bases
        theta = commanded.wall_angle(R)
        flat_limit = math.radians(p.flat_angle_deg)
        flat = part & (theta < flat_limit)
        pillow = np.zeros(g.shape)
        if flat.any():
            e = ndimage.distance_transform_edt(flat) * h
            pamp = (p.pillow_amplitude * strain ** 0.5 * math.sqrt(R / p.tool_radius_ref)
                    * math.sqrt(s.step_down / p.step_down_ref) * thick ** 0.25
                    * min(D / 0.01, 1.0))
            pillow = np.where(flat, pamp * (1.0 - np.exp(-(e / p.pillow_length) ** 2))
                              * _smoothstep((flat_limit - theta) / flat_limit), 0.0)
        local = ndimage.gaussian_filter(rim + pillow, p.smoothing * R / h, mode="nearest")
        # global bending on unclamping
        kappa = p.curvature * strain * (D / p.curvature_depth) * thick
        w = np.where(part, depth, 0.0)
        ixx, iyy = float((w * X * X).sum()), float((w * Y * Y).sum())
        kx = 2.0 * kappa * ixx / (ixx + iyy)
        ky = 2.0 * kappa - kx
        if s.release == "321":
            a = 0.5 * s.meshed_blank_size - 0.5 * s.clamp_margin
            glob = 0.5 * (kx * (a * a - X * X) + ky * (a * a - Y * Y))
        else:
            e_ = s.free_half_width
            inside = np.clip(e_ - np.maximum(np.abs(X), np.abs(Y)), 0.0, None)
            glob = 0.25 * 0.5 * kappa * inside * (2 * e_ - inside)
        dz = local + glob
        if p.noise > 0:
            rng = np.random.default_rng(p.noise_seed)
            dz = dz + rng.normal(0.0, p.noise, g.shape)
        return dz

    def formed(self, commanded: HeightMap, setup: Any) -> HeightMap:
        """commanded + dz, labelled as a proxy result in its metadata."""
        dz = self.deviation(commanded, setup)
        return commanded.with_z(commanded.z + dz,
                                metadata={"source": "proxy", "label": PROXY_LABEL,
                                          "version": PROXY_VERSION})

    # protocols: a FieldModel, a Simulator
    def predict_deviation(self, commanded: HeightMap, setup: Any
                          ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return self.deviation(commanded, setup), None

    def __call__(self, commanded: HeightMap, setup: Any) -> HeightMap:
        return self.formed(commanded, setup)

    def run(self, jobs: Sequence[Tuple[Any, ...]]) -> List[SimOutcome]:
        out = []
        for job in jobs:
            setup, commanded = job[0], job[1]
            try:
                if self.check_toolpath:          # a deck needs one; so does time_frac
                    from precomp.fea.deck import make_toolpath
                    try:
                        make_toolpath(as_setup(setup), commanded)
                    except (ValueError, PrecompError) as exc:
                        raise PrecompError(f"no {as_setup(setup).toolpath_style} tool path: "
                                           f"{exc}") from exc
                out.append(SimOutcome(True, self.formed(commanded, setup), None,
                                      {"source": "proxy", "fidelity": self.fidelity,
                                       "simulator": self.describe()}))
            except Exception as exc:  # recorded, never dropped
                out.append(SimOutcome(False, None, f"{type(exc).__name__}: {exc}",
                                      {"source": "proxy", "fidelity": self.fidelity}))
        return out


def simulate_samples(points: Sequence[DesignPoint], simulator: Any, *, created_at: str,
                     kinds: Sequence[str] = ("uncompensated",), seed: int = 0) -> List[Sample]:
    """`generate` into a temporary data set, for small experiments and tests:
    the samples in memory, in the data set's order. Raises PrecompError
    listing the failures if any job failed (use `generate` to keep going)."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        ds = Dataset.create(tmp, created_at=created_at)
        rep = generate(ds, points, simulator, created_at=created_at, kinds=kinds, seed=seed)
        if rep.failed:
            raise PrecompError(f"{len(rep.failed)} job(s) failed: "
                               + "; ".join(f"{f['sample_id']}: {f['reason']}"
                                           for f in rep.failed[:5]))
        return ds.samples()


# ---------------------------------------------------------------------------
# Grouped splits
# ---------------------------------------------------------------------------
SPLIT_KEYS = {"sample": "sample_id", "part": "part_id", "family": "family"}


def index_frame(samples: Union[Dataset, pd.DataFrame, Sequence[Sample]]) -> pd.DataFrame:
    """The index (sample_id, part_id, family, ...) of a data set, a frame or samples."""
    if isinstance(samples, Dataset):
        return samples.index()
    if isinstance(samples, pd.DataFrame):
        return samples
    return pd.DataFrame([s.index_entry() for s in samples])


def _groups(index: pd.DataFrame, by: str) -> np.ndarray:
    if by not in SPLIT_KEYS:
        raise ValueError(f"by must be one of {sorted(SPLIT_KEYS)}, got {by!r}")
    return index[SPLIT_KEYS[by]].astype(str).to_numpy()


def grouped_split(samples: Union[Dataset, pd.DataFrame, Sequence[Sample]], *,
                  test_fraction: float, seed: int, by: str = "part"
                  ) -> Tuple[List[str], List[str]]:
    """(train ids, test ids): whole groups (`by` "part", "sample" or "family")
    go to the test side until it holds `test_fraction` of the groups
    (at least one, never all)."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must lie in (0, 1)")
    idx = index_frame(samples)
    g = _groups(idx, by)
    uniq = np.unique(g)
    if uniq.size < 2:
        raise ValueError(f"need at least two {by} groups to split, found {uniq.size}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    n_test = int(min(max(1, round(test_fraction * uniq.size)), uniq.size - 1))
    test_groups = set(perm[:n_test])
    ids = idx["sample_id"].astype(str).to_numpy()
    test = [i for i, gg in zip(ids, g) if gg in test_groups]
    train = [i for i, gg in zip(ids, g) if gg not in test_groups]
    return train, test


def family_split(samples: Union[Dataset, pd.DataFrame, Sequence[Sample]],
                 holdout: Union[str, Sequence[str]]) -> Tuple[List[str], List[str]]:
    """Leave-family-out: (ids of the other families, ids of `holdout`)."""
    idx = index_frame(samples)
    hold = {holdout} if isinstance(holdout, str) else set(holdout)
    fam = idx["family"].astype(str).to_numpy()
    missing = hold - set(fam)
    if missing:
        raise ValueError(f"no sample of the families {sorted(missing)}")
    ids = idx["sample_id"].astype(str).to_numpy()
    return ([i for i, f in zip(ids, fam) if f not in hold],
            [i for i, f in zip(ids, fam) if f in hold])


# ---------------------------------------------------------------------------
# Conformal coverage
# ---------------------------------------------------------------------------
def partition_coverage(mu: np.ndarray, std: np.ndarray, y: np.ndarray, groups: np.ndarray, *,
                       level: float, n_partitions: int = 50, calibration_fraction: float = 0.5,
                       seed: int = 0, **calibrator_kwargs: Any) -> Dict[str, float]:
    """Coverage of repeated split-conformal calibration over whole parts.

    The held-out parts (`groups`, part ids) are split at random
    `n_partitions` times into calibration and test parts; each time a
    calibrator is fitted on the calibration parts and measured on the test
    parts: the mean over test parts of each part's coverage (what the
    guarantee is about: its expectation is >= level), the share of test
    parts covered at >= level, the worst part, and the mean width - also for
    the constant-width baseline calibrated on the same parts, so a std that
    carries no information shows as a width ratio near or above 1.
    """
    mu, std, y = (np.asarray(a, dtype=float).ravel() for a in (mu, std, y))
    groups = np.asarray(groups).astype(str).ravel()
    uniq = np.unique(groups)
    if uniq.size < 4:
        raise ValueError("partition_coverage needs at least four parts")
    rng = np.random.default_rng(seed)
    n_cal = int(min(max(2, round(calibration_fraction * uniq.size)), uniq.size - 2))
    if n_cal < min_parts(level):
        raise PrecompError(f"{n_cal} calibration parts per partition cannot support level "
                           f"{level:g} (needs {min_parts(level)}); raise calibration_fraction "
                           "or hold out more parts")
    rows = []
    for _ in range(int(n_partitions)):
        cal = np.isin(groups, rng.permutation(uniq)[:n_cal])
        c = ConformalCalibrator(**calibrator_kwargs).fit(mu[cal], std[cal], y[cal],
                                                         groups[cal])
        lo, hi = c.intervals(mu[~cal], std[~cal], level)
        q0 = c.baseline_halfwidth(level)
        yt, mt, gt = y[~cal], mu[~cal], groups[~cal]
        per = pd.Series((yt >= lo) & (yt <= hi)).groupby(gt).mean()
        per0 = pd.Series(np.abs(yt - mt) <= q0).groupby(gt).mean()
        rows.append((per.mean(), (per >= level).mean(), per.min(), float(np.mean(hi - lo)),
                     per0.mean(), 2.0 * q0))
    r = np.array(rows)
    return {"level": float(level), "mean": float(r[:, 0].mean()), "std": float(r[:, 0].std()),
            "min": float(r[:, 0].min()), "max": float(r[:, 0].max()),
            "share_parts_at_level": float(r[:, 1].mean()),
            "worst_part_median": float(np.median(r[:, 2])),
            "mean_width_m": float(r[:, 3].mean()),
            "baseline_mean": float(r[:, 4].mean()), "baseline_width_m": float(r[:, 5].mean()),
            "width_ratio": float(r[:, 3].mean() / r[:, 5].mean()),
            "n_partitions": int(n_partitions), "n_parts": int(uniq.size),
            "n_calibration_parts": n_cal}
