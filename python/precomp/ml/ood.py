"""The training envelope: is a commanded part like the parts a model was trained on?

Two levels, two distances each, on standardised features:

* point level - every node's feature vector (local, placement, global,
  process, material); part level - the part's descriptor vector (global,
  process, material features);
* Mahalanobis distance under a Ledoit-Wolf shrunk covariance, and the mean
  distance to the k nearest training vectors.

Thresholds are quantiles (99 % by default) of the distances of HELD-OUT
training parts: for the points, the parts are split into folds by part id and
each fold is scored against the others; for the parts, leave-one-part-out. So
an in-distribution part scores like a new part from the training distribution,
not like a training point scored against itself. The variants of one design
point (uncompensated, perturbed, compensated) share a part id and are held out
together.

`assess` returns `in_envelope` (part-level score <= 1 and at most
`max_fraction_out` of the part's nodes beyond a point threshold - that share
is itself the 99 % quantile over the held-out training parts, at least 5 %),
`part_score` (the larger of the two part distances over their thresholds),
`fraction_points_out`, and `reasons`: the features most responsible - those
outside their training range first, then the largest contributions to the
Mahalanobis distance.

Where this fits: the training envelope of "Training"; "Compensation" and "Evaluation" ask it
    whether a commanded part is inside (through DeviationSurrogate.assess).
Called from: ml/surrogate.py:fit_envelope (fit), ml/surrogate.py:DeviationSurrogate.assess
    (assess_features), ml/surrogate.py:DeviationSurrogate.from_state (from_state).
Calls into: sklearn.covariance (LedoitWolf), numpy, pandas.
Start reading at: OODEnvelope.fit, then OODEnvelope.assess_features.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .._util import PrecompError

#: Largest |z| kept after standardisation (a feature that was constant in
#: training and differs now would otherwise overflow the distances).
Z_CAP = 1e6


def _scaler(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    floor = np.maximum(1e-6 * np.maximum(np.abs(mu), np.abs(X).max(axis=0)), 1e-12)
    return mu, np.maximum(sd, floor)


class _Distances:
    """Mahalanobis (Ledoit-Wolf) and kNN distances to a reference set of
    standardised vectors."""

    def __init__(self, Z: np.ndarray, k: int):
        from sklearn.covariance import LedoitWolf

        if len(Z) < 2:
            raise ValueError("the envelope needs at least two reference vectors")
        lw = LedoitWolf().fit(Z)
        self.location = lw.location_
        self.precision = lw.precision_
        self.reference = Z
        self.k = int(min(k, len(Z)))
        self._ref_sq = np.einsum("ij,ij->i", Z, Z)

    def mahalanobis(self, Z: np.ndarray) -> np.ndarray:
        d = Z - self.location
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, self.precision, d), 0.0))

    def knn(self, Z: np.ndarray, chunk: int = 2048) -> np.ndarray:
        """Mean distance to the k nearest reference vectors (brute force in
        chunks: kd-trees gain nothing in ~50 dimensions)."""
        out = np.empty(len(Z))
        for i in range(0, len(Z), chunk):
            q = Z[i:i + chunk]
            d2 = (np.einsum("ij,ij->i", q, q)[:, None] + self._ref_sq[None, :]
                  - 2.0 * q @ self.reference.T)
            part = np.partition(np.maximum(d2, 0.0), self.k - 1, axis=1)[:, :self.k]
            out[i:i + chunk] = np.sqrt(part).mean(axis=1)
        return out

    def state(self) -> Dict[str, Any]:
        return {"location": self.location, "precision": self.precision,
                "reference": self.reference, "k": self.k}

    @classmethod
    def from_state(cls, s: Mapping[str, Any]) -> "_Distances":
        obj = cls.__new__(cls)
        obj.location = np.asarray(s["location"])
        obj.precision = np.asarray(s["precision"])
        obj.reference = np.asarray(s["reference"])
        obj.k = int(s["k"])
        obj._ref_sq = np.einsum("ij,ij->i", obj.reference, obj.reference)
        return obj


class OODEnvelope:
    """Point- and part-level training envelope (see the module docstring).

    quantile : threshold quantile of held-out training distances (0.99).
    k : neighbours of the kNN distance (points); parts use min(k, 3).
    max_reference : training points kept as the kNN reference and for the
        covariance (a seeded random subset).
    n_folds : for the point thresholds the training parts are split into
        this many folds by part id and each fold is scored against the others.
    min_fraction_out : floor of `max_fraction_out`, the largest share of a
        part's nodes beyond a point threshold for the part to count as inside
        the envelope; at fit it is set to the `quantile` of that share over the
        held-out training parts, never below the floor.
    """

    def __init__(self, *, quantile: float = 0.99, k: int = 5, max_reference: int = 5000,
                 min_fraction_out: float = 0.05, n_folds: int = 5, seed: int = 0):
        if not 0.5 < quantile < 1:
            raise ValueError("quantile must lie in (0.5, 1)")
        if int(n_folds) < 2:
            raise ValueError("n_folds must be >= 2")
        self.quantile = float(quantile)
        self.k = int(k)
        self.max_reference = int(max_reference)
        self.min_fraction_out = float(min_fraction_out)
        self.max_fraction_out = float(min_fraction_out)
        self.n_folds = int(n_folds)
        self.seed = int(seed)
        self.point_names: List[str] = []
        self.part_names: List[str] = []
        self.p_mean = self.p_std = self.d_mean = self.d_std = None
        self.p_min = self.p_max = self.d_min = self.d_max = None
        self.points: Optional[_Distances] = None
        self.parts: Optional[_Distances] = None
        self.thresholds: Dict[str, float] = {}
        self.n_train_parts = 0
        self.n_train_points = 0

    # -- fitting --------------------------------------------------------------
    def _z(self, X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        return np.clip((np.asarray(X, float) - mean) / std, -Z_CAP, Z_CAP)

    def fit(self, point_X: np.ndarray, point_groups: np.ndarray, part_D: np.ndarray,
            part_groups: np.ndarray, *, point_names: Sequence[str],
            part_names: Sequence[str]) -> "OODEnvelope":
        """point_X (n, F) features of training nodes with their part ids
        `point_groups`; part_D (m, P) descriptors of the training samples with
        their part ids `part_groups`."""
        point_X = np.asarray(point_X, float)
        part_D = np.asarray(part_D, float)
        point_groups = np.asarray(point_groups).astype(str)
        part_groups = np.asarray(part_groups).astype(str)
        if point_X.shape[1] != len(point_names) or part_D.shape[1] != len(part_names):
            raise ValueError("feature names do not match the arrays")
        if not (np.all(np.isfinite(point_X)) and np.all(np.isfinite(part_D))):
            raise ValueError("the envelope's training features contain NaN or infinity")
        uniq = np.unique(part_groups)
        if uniq.size < 3:
            raise ValueError("the envelope needs at least three training parts")
        rng = np.random.default_rng(self.seed)
        self.point_names, self.part_names = list(point_names), list(part_names)
        self.n_train_parts = int(uniq.size)
        self.n_train_points = int(len(point_X))
        # points: every part is scored once, against the other folds' parts
        self.p_mean, self.p_std = _scaler(point_X)
        self.p_min, self.p_max = np.quantile(point_X, [0.005, 0.995], axis=0)
        Zp = self._z(point_X, self.p_mean, self.p_std)
        pg = np.unique(point_groups)
        folds = np.array_split(rng.permutation(pg), int(min(self.n_folds, pg.size)))
        md_l, kn_l, grp_l = [], [], []
        for f in folds:
            in_f = np.isin(point_groups, f)
            dist = _Distances(Zp[self._subset(np.flatnonzero(~in_f), rng)], self.k)
            probe = self._subset(np.flatnonzero(in_f), rng)
            md_l.append(dist.mahalanobis(Zp[probe]))
            kn_l.append(dist.knn(Zp[probe]))
            grp_l.append(point_groups[probe])
        md, kn, grp = np.concatenate(md_l), np.concatenate(kn_l), np.concatenate(grp_l)
        self.thresholds["point_mahalanobis"] = float(np.quantile(md, self.quantile))
        self.thresholds["point_knn"] = float(np.quantile(kn, self.quantile))
        out = (md > self.thresholds["point_mahalanobis"]) | (kn > self.thresholds["point_knn"])
        frac = pd.Series(out).groupby(grp).mean().to_numpy()
        self.max_fraction_out = float(max(self.min_fraction_out,
                                          np.quantile(frac, self.quantile)))
        self.points = _Distances(Zp[self._subset(np.arange(len(Zp)), rng)], self.k)
        # parts: leave one part out
        self.d_mean, self.d_std = _scaler(part_D)
        self.d_min, self.d_max = part_D.min(axis=0), part_D.max(axis=0)
        Zd = self._z(part_D, self.d_mean, self.d_std)
        kp = min(self.k, 3)
        md_l, kn_l = [], []
        for g in uniq:
            out = part_groups == g
            if (~out).sum() < 2:
                continue
            d = _Distances(Zd[~out], kp)
            md_l.append(d.mahalanobis(Zd[out]))
            kn_l.append(d.knn(Zd[out]))
        md_all, kn_all = np.concatenate(md_l), np.concatenate(kn_l)
        self.thresholds["part_mahalanobis"] = float(np.quantile(md_all, self.quantile))
        self.thresholds["part_knn"] = float(np.quantile(kn_all, self.quantile))
        self.parts = _Distances(Zd, kp)
        return self

    def _subset(self, rows: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rows.size > self.max_reference:
            return np.sort(rng.choice(rows, size=self.max_reference, replace=False))
        return rows

    # -- scoring --------------------------------------------------------------
    def _require(self) -> None:
        if self.points is None or self.parts is None:
            raise PrecompError("the OOD envelope is not fitted")

    def point_scores(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(Mahalanobis, kNN) distances of feature rows (standardised units)."""
        self._require()
        Z = self._z(X, self.p_mean, self.p_std)
        return self.points.mahalanobis(Z), self.points.knn(Z)

    def part_scores(self, D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self._require()
        Z = self._z(np.atleast_2d(D), self.d_mean, self.d_std)
        return self.parts.mahalanobis(Z), self.parts.knn(Z)

    def points_out(self, X: np.ndarray) -> np.ndarray:
        md, kn = self.point_scores(X)
        return (md > self.thresholds["point_mahalanobis"]) | (kn > self.thresholds["point_knn"])

    def _reasons(self, values: np.ndarray, names: List[str], lo: np.ndarray, hi: np.ndarray,
                 mean: np.ndarray, std: np.ndarray, precision: np.ndarray,
                 location: np.ndarray, top: int = 5) -> List[Dict[str, Any]]:
        v = np.atleast_2d(values)
        excess = np.maximum(np.maximum(v - hi, lo - v), 0.0) / std
        ex = excess.mean(axis=0)
        order = [i for i in np.argsort(-ex) if ex[i] > 0][:top]
        out = [{"feature": names[i], "why": "outside the training range",
                "value": float(np.median(v[:, i])), "train_low": float(lo[i]),
                "train_high": float(hi[i]), "excess_std": float(ex[i])} for i in order]
        if len(out) < top:
            z = np.clip((v - mean) / std, -Z_CAP, Z_CAP) - location
            contrib = (z * (z @ precision)).mean(axis=0)
            for i in np.argsort(-contrib):
                if len(out) >= top or contrib[i] <= 0:
                    break
                if names[i] in {o["feature"] for o in out}:
                    continue
                out.append({"feature": names[i], "why": "Mahalanobis contribution",
                            "value": float(np.median(v[:, i])),
                            "contribution": float(contrib[i])})
        return out

    def assess_features(self, point_X: np.ndarray, part_d: np.ndarray) -> Dict[str, Any]:
        """The assessment of one part from its node features and descriptor."""
        self._require()
        point_X = np.asarray(point_X, float)
        part_d = np.asarray(part_d, float).ravel()
        if point_X.shape[1] != len(self.point_names) or part_d.size != len(self.part_names):
            raise ValueError("feature dimensions differ from the envelope's")
        md, kn = self.part_scores(part_d)
        t = self.thresholds
        part_score = float(max(md[0] / t["part_mahalanobis"], kn[0] / t["part_knn"]))
        out_pts = self.points_out(point_X)
        frac = float(out_pts.mean()) if len(out_pts) else 0.0
        inside = bool(part_score <= 1.0 and frac <= self.max_fraction_out)
        details: List[Dict[str, Any]] = []
        if part_score > 1.0:
            details += [dict(d, level="part") for d in self._reasons(
                part_d, self.part_names, self.d_min, self.d_max, self.d_mean, self.d_std,
                self.parts.precision, self.parts.location)]
        if frac > self.max_fraction_out:
            details += [dict(d, level="point") for d in self._reasons(
                point_X[out_pts], self.point_names, self.p_min, self.p_max, self.p_mean,
                self.p_std, self.points.precision, self.points.location)]
        reasons: List[str] = []
        for d in details:
            if d["feature"] not in reasons:
                reasons.append(d["feature"])
        return {"in_envelope": inside, "part_score": part_score,
                "part_mahalanobis": float(md[0]), "part_knn": float(kn[0]),
                "fraction_points_out": frac, "n_points": int(len(point_X)),
                "reasons": reasons, "reason_details": details,
                "thresholds": dict(self.thresholds),
                "max_fraction_out": self.max_fraction_out}

    def summary(self) -> Dict[str, Any]:
        return {"quantile": self.quantile, "k": self.k, "thresholds": dict(self.thresholds),
                "n_train_parts": self.n_train_parts, "n_train_points": self.n_train_points,
                "max_fraction_out": self.max_fraction_out,
                "n_point_features": len(self.point_names),
                "n_part_features": len(self.part_names)}

    def to_state(self) -> Dict[str, Any]:
        self._require()
        return {"params": {"quantile": self.quantile, "k": self.k,
                           "max_reference": self.max_reference,
                           "min_fraction_out": self.min_fraction_out,
                           "n_folds": self.n_folds, "seed": self.seed},
                "max_fraction_out": self.max_fraction_out,
                "point_names": self.point_names, "part_names": self.part_names,
                "p_mean": self.p_mean, "p_std": self.p_std, "p_min": self.p_min,
                "p_max": self.p_max, "d_mean": self.d_mean, "d_std": self.d_std,
                "d_min": self.d_min, "d_max": self.d_max, "points": self.points.state(),
                "parts": self.parts.state(), "thresholds": dict(self.thresholds),
                "n_train_parts": self.n_train_parts, "n_train_points": self.n_train_points}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "OODEnvelope":
        e = cls(**state["params"])
        e.point_names, e.part_names = list(state["point_names"]), list(state["part_names"])
        for key in ("p_mean", "p_std", "p_min", "p_max", "d_mean", "d_std", "d_min", "d_max"):
            setattr(e, key, np.asarray(state[key]))
        e.points = _Distances.from_state(state["points"])
        e.parts = _Distances.from_state(state["parts"])
        e.thresholds = dict(state["thresholds"])
        e.max_fraction_out = float(state["max_fraction_out"])
        e.n_train_parts = int(state["n_train_parts"])
        e.n_train_points = int(state["n_train_points"])
        return e


__all__ = ["OODEnvelope"]
