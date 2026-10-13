"""Compensation on a learned surrogate, and its honest end-to-end check.

`surrogate_compensate` runs displacement adjustment (`precomp.compensation`)
with the surrogate as the predictor of the formed surface,

    formed(c) = c + dz_hat(c),

one iteration at a time so that it can stop when the PREDICTED deviation from
the target is within tolerance or stops improving (stagnation) - or, as an
option, within the model's own interval half-width. It returns the
compensated commanded surface (the best predicted iterate), the predicted
formed surface, the predicted residual deviation from the target with its
conformal interval, the out-of-distribution report of the compensated shape,
and the history - all predictions, labelled with the data source the model
learned from.

The envelope is enforced, not only reported: a target outside the model's
training envelope is refused, and every iterate is assessed before the model
is evaluated on it - the first one outside ends the loop ("left_envelope"),
keeping the best iterate inside - unless `allow_out_of_envelope`, in which
case the verdicts are recorded in the history and the result.

`surrogate_optimize` goes further than DA's pointwise update: starting from
the DA command it searches a smooth, low-dimensional correction field (depth
level x angle) that minimises the predicted deviation over the part, under
the same constraints and the envelope, with a penalty on the model's std.

A prediction is not a result: the workflow simulates every compensated shape
with sparlab_form before reporting it (python/scripts/combined_rim_ml.py).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .._util import PathLike, PrecompError, to_jsonable, write_json
from ..compensation import SurrogatePredictor as _BaseSurrogatePredictor
from ..compensation import _condition, displacement_adjustment, smooth_update
from ..geometry.heightmap import HeightMap
from ..geometry.parts import MAX_WALL_ANGLE_DEG
from ..metrology import flange_mask, metrics, part_mask, vertical_deviation
from .features import as_setup


class SurrogatePredictor(_BaseSurrogatePredictor):
    """`precomp.compensation.SurrogatePredictor` that keeps the std map of its
    last prediction (`last_std`, [m]) and counts its calls."""

    def __init__(self, model: Any, setup: Any):
        super().__init__(model, setup)
        self.last_std: Optional[np.ndarray] = None
        self.calls = 0

    def __call__(self, commanded: HeightMap) -> HeightMap:
        formed, std = self.predict_with_uncertainty(commanded)
        self.last_std = std
        self.calls += 1
        return formed


@dataclass
class SurrogateCompensation:
    """The outcome of `surrogate_compensate` (all [m]; predictions, not results).

    compensated : the commanded surface to form. predicted : its predicted
    formed surface. residual : predicted - target (vertical) on the target
    grid. residual_interval : (lower, upper) of the residual at `level`, or
    None without calibration. std : the model's std map at `compensated`.
    ood : the envelope report of `compensated` (and `ood_target` of the
    uncompensated target). history : per iteration, the predicted error over
    the part, the update size, the envelope verdict and the interval
    half-width. stopped : "tolerance", "within_interval", "stagnation",
    "left_envelope" or "iterations" (`surrogate_compensate`), "optimizer"
    (`surrogate_optimize`). data_source : what the model learned
    from.
    verification : a record of a check simulation, when the caller made one
        (saved with the result; None otherwise).
    allowed_out_of_envelope : whether the envelope was overridden.
    """

    target: HeightMap
    compensated: HeightMap
    predicted: HeightMap
    residual: HeightMap
    residual_interval: Optional[Tuple[HeightMap, HeightMap]]
    std: Optional[np.ndarray]
    level: float
    ood: Optional[Dict[str, Any]]
    ood_target: Optional[Dict[str, Any]]
    history: List[Dict[str, Any]]
    stopped: str
    best_iteration: int
    data_source: str
    verification: Optional[Dict[str, Any]] = None
    allowed_out_of_envelope: bool = False

    @property
    def in_envelope(self) -> Optional[bool]:
        """Whether the target and the compensated shape are both inside the
        envelope (None without one)."""
        if self.ood is None or self.ood_target is None:
            return None
        return bool(self.ood["in_envelope"] and self.ood_target["in_envelope"])

    def predicted_metrics(self, tolerance: Optional[float] = None) -> Dict[str, Any]:
        """Metrics of the predicted residual over the part, labelled as a
        prediction, with the mean half-width of its interval (the model's
        own uncertainty, which a predicted residual near 0 does not remove)."""
        pm = part_mask(self.target)
        m = metrics(self.residual, pm, tolerance)
        out = {"quantity": "predicted residual (surrogate)",
               "model_data_source": self.data_source, **m}
        if self.residual_interval is not None:
            lo, hi = self.residual_interval
            out["interval_level"] = self.level
            out["interval_halfwidth_mean_m"] = float(np.mean(0.5 * (hi.z - lo.z)[pm]))
        return out

    def save(self, out_dir: PathLike) -> Path:
        """compensated.npz, predicted.npz, residual.npz (+ bounds), compensation.json."""
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        self.compensated.save(d / "compensated.npz")
        self.predicted.save(d / "predicted.npz")
        self.residual.save(d / "residual.npz")
        if self.residual_interval is not None:
            self.residual_interval[0].save(d / "residual_lower.npz")
            self.residual_interval[1].save(d / "residual_upper.npz")
        write_json(d / "compensation.json", to_jsonable({
            "method": self.compensated.metadata.get("compensation", {}).get(
                "method", "surrogate displacement adjustment"), "stopped": self.stopped,
            "best_iteration": self.best_iteration, "level": self.level,
            "model_data_source": self.data_source,
            "history_quantity": "predictions of the surrogate (model data source: "
                                f"{self.data_source}), not simulations",
            "history": self.history,
            "predicted_residual": self.predicted_metrics(), "in_envelope": self.in_envelope,
            "allowed_out_of_envelope": self.allowed_out_of_envelope, "ood": self.ood,
            "ood_target": self.ood_target, "verification": self.verification}))
        return d


def surrogate_compensate(target: HeightMap, setup: Any, surrogate: Any, *,
                         iterations: int = 8, alpha: float = 1.0,
                         tolerance: Optional[float] = None, stagnation: float = 0.02,
                         smoothing: Optional[float] = None, direction: str = "vertical",
                         level: float = 0.9,
                         max_wall_angle_deg: Optional[float] = MAX_WALL_ANGLE_DEG,
                         allow_out_of_envelope: bool = False,
                         interval_stop: bool = False,
                         upper_bound: Any = None,
                         hold_mask: Optional[np.ndarray] = None,
                         adjust_mask: Optional[np.ndarray] = None,
                         tool_radius: Optional[float] = None) -> SurrogateCompensation:
    """Displacement adjustment of `target` on `surrogate` (see the module docstring).

    iterations : at most this many surrogate predictions.
    tolerance : stop when the predicted RMS error over the part is <= this [m].
    stagnation : stop when an iteration improves the predicted RMS error by
        less than this fraction (0.02 = 2 %), or makes it worse.
    alpha, smoothing [m], direction, max_wall_angle_deg : as for
        `precomp.compensation.displacement_adjustment`.
    level : coverage of the residual interval (needs a calibrated surrogate;
        without one the interval is None and the result says so).
    allow_out_of_envelope : compensate a target outside the training
        envelope, and keep iterating when an iterate leaves it (PrecompError
        for such a target, and a stop at such an iterate, otherwise). Without
        an envelope there is nothing to enforce.
    interval_stop : with a calibrated surrogate, stop ("within_interval")
        once the predicted RMS error over the part is within the RMS
        half-width of the model's interval at `level` at that iterate - the
        model cannot resolve less. It saves about two thirds of the
        predictions (2 instead of 6), but whether it helps is unclear: on
        proxy data the verified deviation was 4 % smaller on the 23 held-out
        targets of the tests (0.054 against 0.056 mm) and 9 % larger on the
        11 of the proxy demonstration (0.049 against 0.045 mm). Off by default.
    upper_bound, hold_mask, adjust_mask, tool_radius : as for
        `displacement_adjustment` (a support's command bound and masks,
        `precomp.fea.support`; the tool's reach). None: the defaults.
    """
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if not 0 <= stagnation < 1:
        raise ValueError("stagnation must lie in [0, 1)")
    setup = as_setup(setup)
    has_env = getattr(surrogate, "ood", None) is not None and hasattr(surrogate, "assess")
    ood_t = dict(surrogate.assess(target, setup)) if has_env else None
    if ood_t is not None and not ood_t["in_envelope"] and not allow_out_of_envelope:
        raise PrecompError(
            "the target is outside the model's training envelope (reasons: "
            f"{', '.join(ood_t['reasons'][:5]) or 'fraction of nodes out'}; part score "
            f"{ood_t['part_score']:.3g}): the surrogate would extrapolate. Train on such parts, "
            "or pass allow_out_of_envelope=True (--allow-out-of-envelope) to compensate "
            "anyway")
    pred = SurrogatePredictor(surrogate, setup)
    cal = getattr(surrogate, "calibrator", None)
    can_interval = (hasattr(surrogate, "predict_interval") and cal is not None
                    and cal.fitted and cal.supports(level))
    pm = part_mask(target)
    c = target.copy()
    history: List[Dict[str, Any]] = []
    best: Optional[Tuple[float, int, HeightMap, HeightMap, Optional[np.ndarray]]] = None
    stopped = "iterations"
    prev = None
    verdicts: Dict[int, Dict[str, Any]] = {}
    for k in range(iterations):
        a = ood_t if k == 0 else (dict(surrogate.assess(c, setup)) if has_env else None)
        if a is not None:
            verdicts[k] = a
            if not a["in_envelope"] and k > 0 and not allow_out_of_envelope:
                history.append({"iteration": k, "predicted": False, "in_envelope": False,
                                "ood_part_score": a["part_score"],
                                "ood_reasons": a["reasons"][:5],
                                "note": "iterate outside the training envelope: not evaluated"})
                stopped = "left_envelope"
                break
        da = displacement_adjustment(target, pred, iterations=1, alpha=alpha,
                                     direction=direction, smoothing=smoothing,
                                     max_wall_angle_deg=max_wall_angle_deg, initial=c,
                                     upper_bound=upper_bound, hold_mask=hold_mask,
                                     adjust_mask=adjust_mask, tool_radius=tool_radius)
        h = dict(da.history[0])
        h["iteration"] = k
        h["predicted"] = True
        if a is not None:
            h["in_envelope"] = bool(a["in_envelope"])
            h["ood_part_score"] = a["part_score"]
        rms = h["error"]["rms"]
        history.append(h)
        if best is None or rms < best[0]:
            best = (rms, k, da.commanded, da.formed, pred.last_std)
        if tolerance is not None and rms <= tolerance:
            stopped = "tolerance"
            break
        if interval_stop and can_interval:
            lo, hi = surrogate.predict_interval(da.commanded, setup, level)
            hw = float(np.sqrt(np.mean((0.5 * (hi - lo))[pm] ** 2)))
            h["interval_halfwidth_rms_m"] = hw
            if rms <= hw:
                stopped = "within_interval"
                break
        if prev is not None and rms > (1.0 - stagnation) * prev:
            stopped = "stagnation"
            break
        prev = rms
        c = da.proposed
    assert best is not None
    _, kbest, comp, formed, std = best
    comp = comp.copy()
    comp.metadata["compensation"] = {"method": "surrogate displacement adjustment",
                                     "iteration": kbest, "alpha": alpha,
                                     "model_data_source": getattr(surrogate, "data_source",
                                                                  None)}
    residual = vertical_deviation(formed, target)
    residual.metadata.update(quantity="predicted residual (surrogate)",
                             model_data_source=getattr(surrogate, "data_source", None))
    interval = None
    if can_interval:
        lo, hi = surrogate.predict_interval(comp, setup, level)
        interval = (residual.with_z(comp.z + lo - target.z),
                    residual.with_z(comp.z + hi - target.z))
    ood = verdicts.get(kbest) if has_env else None
    return SurrogateCompensation(target, comp, formed, residual, interval, std, level, ood,
                                 ood_t, history, stopped, kbest,
                                 str(getattr(surrogate, "data_source", "unknown")),
                                 allowed_out_of_envelope=bool(allow_out_of_envelope))


#: Knots of the hat functions of `correction_basis` in the normalised depth
#: level s = -z_target / depth (0 at the sheet plane, 1 at the bottom): dense
#: in the upper band next to the clamp, where the surrogate's DA leaves most
#: of its error.
DEFAULT_LEVEL_KNOTS: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.35, 0.6, 1.0)


def correction_basis(target: HeightMap, region: np.ndarray,
                     level_knots: Sequence[float] = DEFAULT_LEVEL_KNOTS,
                     angular_orders: Sequence[int] = (2, 4),
                     smoothing: Optional[float] = 1e-3) -> np.ndarray:
    """The smooth correction fields `surrogate_optimize` searches over, as an
    (n, ny, nx) array (each field's largest magnitude 1, zero outside
    `region`): hat functions (piecewise-linear B-splines) in the normalised
    depth level s = -z_target / depth at `level_knots`, times the angular
    Fourier modes 1, cos(m theta), sin(m theta) for m in `angular_orders`
    (theta about the region's centroid), each smoothed in plan at
    `smoothing` [m] over the region (`smooth_update`). n = len(level_knots)
    * (1 + 2 len(angular_orders)).

    The smoothing matters: on a steep wall a band of depth levels is a
    ring a node or two wide in plan, and a learned surrogate answers such a
    groove far outside anything it was trained on (on the fine-mesh MLP a
    25 um groove at the rim predicted a 0.85 mm residual, std 2.4 mm)."""
    region = np.asarray(region, bool)
    if not region.any():
        raise ValueError("the correction region is empty")
    knots = np.asarray(sorted(float(k) for k in level_knots))
    if knots.size < 2 or knots[0] < 0 or knots[-1] > 1 or np.any(np.diff(knots) <= 0):
        raise ValueError("level_knots must be >= 2 distinct values in [0, 1]")
    depth = float(-target.z[region].min())
    s = np.clip(-target.z / max(depth, 1e-12), 0.0, 1.0)
    X, Y = target.grid.mesh()
    th = np.arctan2(Y - Y[region].mean(), X - X[region].mean())
    modes = [np.ones_like(th)]
    for m in angular_orders:
        modes += [np.cos(m * th), np.sin(m * th)]
    out = []
    for i in range(knots.size):
        hat = np.interp(s, knots, np.eye(knots.size)[i])
        for mode in modes:
            f = np.where(region, hat * mode, 0.0)
            if smoothing:
                f = smooth_update(f, region, smoothing, target.grid.h)
            peak = float(np.abs(f).max())
            if peak > 1e-12:
                out.append(f / peak)
    return np.asarray(out)


def surrogate_optimize(target: HeightMap, setup: Any, surrogate: Any, *,
                       start: Any = None,
                       level_knots: Sequence[float] = DEFAULT_LEVEL_KNOTS,
                       angular_orders: Sequence[int] = (2, 4),
                       basis_smoothing: Optional[float] = 1e-3,
                       std_weight: float = 0.25, reg_weight: float = 0.05,
                       fd_step: float = 5e-5, central: bool = True,
                       max_correction: float = 1e-3,
                       max_step: float = 3e-4,
                       dampings: Sequence[float] = (1e-3, 1e-2, 1e-1, 1.0),
                       max_iterations: int = 8, rel_tol: float = 1e-3,
                       max_time_s: Optional[float] = 300.0,
                       level: float = 0.9,
                       max_wall_angle_deg: Optional[float] = MAX_WALL_ANGLE_DEG,
                       allow_out_of_envelope: bool = False,
                       upper_bound: Any = None,
                       hold_mask: Optional[np.ndarray] = None,
                       adjust_mask: Optional[np.ndarray] = None,
                       tool_radius: Optional[float] = None,
                       da_kwargs: Optional[Dict[str, Any]] = None) -> SurrogateCompensation:
    """Search the commanded surface whose PREDICTED formed surface is closest
    to `target`, starting from a surrogate DA result: a prediction, not a
    result (verify it with a simulation).

    DA corrects each node by its own predicted error, so it cannot trade a
    change in one place for an effect in another (a deeper wall that pulls
    the rim down, say). Here the command is

        c(p) = condition(start + sum_k p_k B_k),

    B the smooth fields of `correction_basis` (hat functions in the depth
    level - dense next to the rim - times angular Fourier modes, smoothed
    in plan at `basis_smoothing` [m]; ~30 parameters [m]) and `condition` DA's own (`precomp.compensation`: hold
    mask kept at the target, `upper_bound`, the wall-angle limit, the
    tool's reach), and p minimises

        J(p) = mean_part(r^2) + std_weight mean_part(std^2) + reg_weight mean(p^2),

    r the predicted vertical residual and std the model's std at c(p), over
    the part (`part_mask(target)`, as `SurrogateCompensation.predicted_metrics`).
    The std term keeps the search where the model is sure of itself
    instead of exploiting its errors. `std_weight` (default 0.25, >= 0; 0
    ignores the std) sets how strongly: an under-dispersed std (an ensemble
    whose spread is below its actual error, e.g. a transfer model fitted on
    a few parts) lets the search exploit model error, which a larger weight
    (1.0: the std counts as much as the residual) resists. The weights are
    recorded in the command's metadata. J is a sum of squares: a
    Levenberg-Marquardt iteration with a finite-difference Jacobian (step
    `fd_step` [m] per parameter, central unless not `central`: one
    evaluation batch of 2n (n) predictions), then a full and a half step
    per damping in `dampings` (relative to the mean diagonal of J^T J),
    each limited to `max_step` [m] in the command and |p| <=
    `max_correction` [m]. The difference probes are candidates too: a
    learned surrogate is rough at the scale of tens of microns (on the
    fine-mesh MLP most smooth 25 um perturbations of the DA command made
    the predicted RMS worse in both directions), where a Gauss-Newton
    step alone stalls. The best candidate that
    lowers J by more than `rel_tol` (relative) and whose command is inside
    the training envelope is accepted (one outside is rejected, unless
    `allow_out_of_envelope`). It stops after `max_iterations`, at no
    accepted step, or at `max_time_s` (not starting a batch the last one's
    duration says would end after it; checked between predictions).

    start : the starting command - a HeightMap, a SurrogateCompensation, or
        None to run `surrogate_compensate` with `da_kwargs` and the
        constraint keywords here.
    Returns a SurrogateCompensation with stopped = "optimizer" (the reason
    in `history[-1]["stop_reason"]` and the command's metadata), the best
    accepted iterate (the start if none), and a history entry per
    evaluation batch (the predicted RMS of its accepted or best candidate
    in "error").
    """
    t_start = time.perf_counter()
    if not (np.isfinite(std_weight) and std_weight >= 0.0):
        raise ValueError(f"std_weight must be finite and >= 0, got {std_weight!r}")
    if not (np.isfinite(reg_weight) and reg_weight >= 0.0):
        raise ValueError(f"reg_weight must be finite and >= 0, got {reg_weight!r}")
    setup = as_setup(setup)
    constraints = dict(max_wall_angle_deg=max_wall_angle_deg, upper_bound=upper_bound,
                       hold_mask=hold_mask, adjust_mask=adjust_mask, tool_radius=tool_radius)
    started_from = "surrogate DA" if start is None else "given"
    if start is None:
        start = surrogate_compensate(target, setup, surrogate, level=level,
                                     allow_out_of_envelope=allow_out_of_envelope,
                                     **constraints, **(da_kwargs or {}))
    c0 = start.compensated if isinstance(start, SurrogateCompensation) else start
    if not c0.grid.matches(target.grid):
        c0 = c0.resample(target.grid)
    has_env = getattr(surrogate, "ood", None) is not None and hasattr(surrogate, "assess")
    ood_t = dict(surrogate.assess(target, setup)) if has_env else None
    if ood_t is not None and not ood_t["in_envelope"] and not allow_out_of_envelope:
        raise PrecompError(
            "the target is outside the model's training envelope (reasons: "
            f"{', '.join(ood_t['reasons'][:5]) or 'fraction of nodes out'}): the surrogate "
            "would extrapolate; pass allow_out_of_envelope=True to optimise anyway")
    hold = flange_mask(target) if hold_mask is None else np.asarray(hold_mask, bool)
    adjust = part_mask(target) if adjust_mask is None else np.asarray(adjust_mask, bool)
    region = adjust & ~hold
    B = correction_basis(target, region, level_knots, angular_orders, basis_smoothing)
    n = B.shape[0]
    pred = SurrogatePredictor(surrogate, setup)
    pm = part_mask(target)
    sel: Optional[np.ndarray] = None

    def evaluate(p: np.ndarray) -> Dict[str, Any]:
        nonlocal sel
        z_new = c0.z + np.tensordot(p, B, axes=1)
        cmd = _condition(c0, z_new, hold, target, max_wall_angle_deg, upper_bound,
                         tool_radius)
        formed, std = pred.predict_with_uncertainty(cmd)
        res = vertical_deviation(formed, target)
        if sel is None:
            sel = pm & res.mask
        r = np.nan_to_num(res.z[sel])
        sd = np.zeros(0) if std is None else np.nan_to_num(np.asarray(std)[sel])
        N = float(r.size)
        vec = np.concatenate([r / np.sqrt(N), np.sqrt(std_weight / N) * sd,
                              np.sqrt(reg_weight / n) * p])
        return {"p": p, "cmd": cmd, "formed": formed, "std": std, "res": res, "vec": vec,
                "J": float(vec @ vec), "rms": float(np.sqrt(np.mean(r ** 2))),
                "std_rms": float(np.sqrt(np.mean(sd ** 2))) if sd.size else None}

    def out_of_time() -> bool:
        return max_time_s is not None and time.perf_counter() - t_start > max_time_s

    def entry(k: int, kind: str, e: Dict[str, Any], evals: int, **extra) -> Dict[str, Any]:
        return {"iteration": k, "batch": kind, "evaluations": evals, "predicted": True,
                "error": metrics(e["res"], sel), "objective": e["J"],
                "std_rms_m": e["std_rms"], "params_rms_m": float(np.sqrt(np.mean(e["p"] ** 2))),
                "elapsed_s": time.perf_counter() - t_start, **extra}

    cur = evaluate(np.zeros(n))
    a0 = dict(surrogate.assess(cur["cmd"], setup)) if has_env else None
    history: List[Dict[str, Any]] = [entry(0, "start", cur, 1, accepted=True,
                                           in_envelope=None if a0 is None
                                           else bool(a0["in_envelope"]))]
    ood_best = a0
    reason = "max_iterations"
    last_batch_s = 0.0
    for k in range(1, max_iterations + 1):
        t_batch = time.perf_counter()
        if out_of_time() or (max_time_s is not None and
                             t_batch - t_start + last_batch_s > max_time_s):
            reason = "time"     # (the next batch would not finish in time)
            break
        # finite-difference Jacobian of the residual vector; the probes are
        # candidates too (a coordinate search where the model is too rough
        # for the Gauss-Newton step)
        Jm = np.empty((cur["vec"].size, n))
        evals, done = 0, 0
        cands: List[Dict[str, Any]] = []
        for j in range(n):
            if out_of_time():
                break
            pj = cur["p"].copy()
            pj[j] += fd_step
            ep = evaluate(pj)
            ep["damping"] = "probe"
            cands.append(ep)
            if central:
                pj = cur["p"].copy()
                pj[j] -= fd_step
                em = evaluate(pj)
                em["damping"] = "probe"
                cands.append(em)
                Jm[:, j] = (ep["vec"] - em["vec"]) / (2.0 * fd_step)
                evals += 2
            else:
                Jm[:, j] = (ep["vec"] - cur["vec"]) / fd_step
                evals += 1
            done += 1
        if done < n:
            reason = "time"
            history.append({"iteration": k, "batch": "jacobian (incomplete)",
                            "evaluations": evals, "predicted": True,
                            "elapsed_s": time.perf_counter() - t_start})
            break
        A = Jm.T @ Jm
        g = Jm.T @ cur["vec"]
        scale = float(np.mean(np.diag(A))) or 1.0
        for mu in dampings:
            d = -np.linalg.solve(A + mu * scale * np.eye(n), g)
            big = float(np.abs(np.tensordot(d, B, axes=1)).max())
            if big > max_step:
                d *= max_step / big
            for frac in (1.0, 0.5):
                if out_of_time():
                    break
                p = np.clip(cur["p"] + frac * d, -max_correction, max_correction)
                e = evaluate(p)
                evals += 1
                e["damping"] = mu if frac == 1.0 else f"{mu:g} (half step)"
                cands.append(e)
        cands.sort(key=lambda e: e["J"])
        accepted, verdict = None, None
        for e in cands:
            if e["J"] >= (1.0 - rel_tol) * cur["J"]:
                break
            a = dict(surrogate.assess(e["cmd"], setup)) if has_env else None
            if a is not None and not a["in_envelope"] and not allow_out_of_envelope:
                continue
            accepted, verdict = e, a
            break
        shown = accepted or cands[0]
        history.append(entry(k, "jacobian + steps", shown, evals,
                             accepted=accepted is not None, damping=shown["damping"],
                             in_envelope=None if verdict is None
                             else bool(verdict["in_envelope"]),
                             step_max_abs_m=float(np.abs(shown["cmd"].z - cur["cmd"].z).max())))
        last_batch_s = time.perf_counter() - t_batch
        if accepted is None:
            reason = "no_improvement"
            break
        cur, ood_best = accepted, verdict
    history[-1]["stop_reason"] = reason
    best_k = max(i for i, h in enumerate(history) if h.get("accepted"))
    comp = cur["cmd"].copy()
    comp.metadata["compensation"] = {
        "method": "surrogate optimisation (Levenberg-Marquardt on a smooth basis)",
        "start": started_from, "parameters": n,
        "params_m": [float(x) for x in cur["p"]], "stop_reason": reason,
        "std_weight": float(std_weight), "reg_weight": float(reg_weight),
        "predictions": pred.calls, "time_s": time.perf_counter() - t_start,
        "model_data_source": getattr(surrogate, "data_source", None)}
    residual = cur["res"]
    residual.metadata.update(quantity="predicted residual (surrogate)",
                             model_data_source=getattr(surrogate, "data_source", None))
    cal = getattr(surrogate, "calibrator", None)
    interval = None
    if (hasattr(surrogate, "predict_interval") and cal is not None and cal.fitted
            and cal.supports(level)):
        lo, hi = surrogate.predict_interval(comp, setup, level)
        interval = (residual.with_z(comp.z + lo - target.z),
                    residual.with_z(comp.z + hi - target.z))
    return SurrogateCompensation(target, comp, cur["formed"], residual, interval, cur["std"],
                                 level, ood_best, ood_t, history, "optimizer", best_k,
                                 str(getattr(surrogate, "data_source", "unknown")),
                                 allowed_out_of_envelope=bool(allow_out_of_envelope))


__all__ = ["SurrogatePredictor", "SurrogateCompensation", "surrogate_compensate",
           "surrogate_optimize", "correction_basis"]
