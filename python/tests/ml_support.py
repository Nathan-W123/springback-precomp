"""Test-only ML helpers: things the tests need that the workflow does not.

partition_coverage - repeated split-conformal calibration over random
    calibration / test partitions of held-out parts (how the tests check that
    ConformalCalibrator's intervals are valid).
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

from precomp._util import PrecompError
from precomp.ml.uncertainty import ConformalCalibrator, min_parts


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
