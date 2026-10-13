"""Deviation models: dz [m] from features, with an uncertainty.

The protocol, `DeviationModel` (a point model): ``fit(X, y, groups)`` on a
feature table (rows = grid nodes, `groups` = sample ids), ``predict(X) ->
(mean, std)`` in metres, ``feature_names``, ``to_state()`` /
``from_state(state)``. The implementations are `MLPEnsemble` (the five-network
ensemble the workflow trains) and `TransferModel` (a frozen base plus a
correction fitted on a few parts of a new process).

`precomp.ml.surrogate.DeviationSurrogate` turns a model into what the
compensation code consumes (``predict_deviation(commanded, setup)``).

Determinism: every model takes a `seed`; bootstrap draws, initialisations,
shuffles and dropout masks derive from it, so a fit on the same table with the
same seed gives the same model (on the same library versions and hardware).
Threads: `n_threads` caps OpenMP/BLAS (threadpoolctl) and torch threads during
fit and predict; None uses $PRECOMP_ML_THREADS, else the libraries' defaults.

The models import torch lazily and raise ImportError with the install command
when it is missing; they run on the CPU.
"""

from __future__ import annotations

import contextlib
import copy
import math
import os
from typing import (Any, Dict, Iterator, List, Mapping, Optional, Protocol, Sequence, Tuple,
                    runtime_checkable)

import numpy as np

from .._util import PrecompError

THREADS_ENV = "PRECOMP_ML_THREADS"


# ---------------------------------------------------------------------------
# Protocols and helpers
# ---------------------------------------------------------------------------
@runtime_checkable
class DeviationModel(Protocol):
    """A point model: rows of features -> dz [m] with a standard deviation [m]."""

    kind: str
    feature_names: List[str]

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, *,
            feature_names: Optional[Sequence[str]] = None) -> "DeviationModel": ...

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: ...

    def to_state(self) -> Dict[str, Any]: ...


def resolve_threads(n_threads: Optional[int]) -> Optional[int]:
    if n_threads is not None:
        if int(n_threads) < 1:
            raise ValueError("n_threads must be >= 1")
        return int(n_threads)
    env = os.environ.get(THREADS_ENV)
    if env:
        try:
            return max(1, int(env))
        except ValueError as exc:
            raise ValueError(f"${THREADS_ENV} must be an integer, got {env!r}") from exc
    return None


@contextlib.contextmanager
def thread_limit(n_threads: Optional[int]) -> Iterator[None]:
    """Cap OpenMP/BLAS threads (threadpoolctl) and torch's intra-op threads."""
    n = resolve_threads(n_threads)
    if n is None:
        yield
        return
    from threadpoolctl import threadpool_limits

    torch_prev = None
    try:
        import sys
        if "torch" in sys.modules:
            import torch
            torch_prev = torch.get_num_threads()
            torch.set_num_threads(n)
    except Exception:  # torch is optional; a failure to cap it is not fatal
        torch_prev = None
    try:
        with threadpool_limits(limits=n):
            yield
    finally:
        if torch_prev is not None:
            import torch
            torch.set_num_threads(torch_prev)


def _check_table(X: np.ndarray, y: Optional[np.ndarray] = None,
                 groups: Optional[np.ndarray] = None) -> Tuple[np.ndarray, ...]:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (rows, features), got shape {X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinity")
    out: List[np.ndarray] = [X]
    if y is not None:
        y = np.asarray(y, dtype=float).ravel()
        if y.shape[0] != X.shape[0]:
            raise ValueError(f"y has {y.shape[0]} rows, X has {X.shape[0]}")
        if not np.all(np.isfinite(y)):
            raise ValueError("y contains NaN or infinity")
        out.append(y)
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape[0] != X.shape[0]:
            raise ValueError(f"groups has {groups.shape[0]} rows, X has {X.shape[0]}")
        out.append(groups)
    return tuple(out)


def _check_names(model: Any, X: np.ndarray) -> None:
    names = getattr(model, "feature_names", None)
    if names and X.shape[1] != len(names):
        raise ValueError(f"{type(model).__name__} was fitted on {len(names)} features, "
                         f"X has {X.shape[1]} columns")


def _require_torch(what: str):
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(f"{what} needs torch: pip install -e '.[torch]'") from exc
    return torch


#: Default target scale: dz is learned divided by the elastic unloading
#: curvature 3 sigma_f(0.2) / (E t) of the sheet, the classical springback scale.
DEFAULT_TARGET_SCALE = "elastic_curvature"


def _target_scale(model: Any, X: np.ndarray) -> np.ndarray:
    """Per-row divisor of the target: the feature column `target_scale`
    (ones when it is None). ValueError when the column is absent or not > 0."""
    name = model.params.get("target_scale")
    if name is None:
        return np.ones(len(X))
    if name not in model.feature_names:
        raise ValueError(f"target_scale {name!r} is not among the feature names; pass "
                         "feature_names to fit, or target_scale=None")
    s = X[:, model.feature_names.index(name)]
    if not np.all(s > 0):
        raise ValueError(f"target_scale column {name!r} must be > 0 on every row")
    return s


def _group_bootstrap_rows(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row indices of a bootstrap resample of the groups (whole samples, drawn
    with replacement; a group drawn twice contributes its rows twice)."""
    uniq, inv = np.unique(groups, return_inverse=True)
    counts = np.bincount(rng.integers(0, uniq.size, size=uniq.size), minlength=uniq.size)
    return np.repeat(np.arange(len(groups)), counts[inv])


def _grouped_holdout(groups: np.ndarray, fraction: float, rng: np.random.Generator
                     ) -> Tuple[np.ndarray, np.ndarray]:
    uniq = np.unique(groups)
    if uniq.size < 2 or fraction <= 0:
        rows = np.arange(len(groups))
        return rows, rows[:0]
    n_val = int(min(max(1, round(fraction * uniq.size)), uniq.size - 1))
    val = set(rng.permutation(uniq)[:n_val])
    is_val = np.array([g in val for g in groups])
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


# ---------------------------------------------------------------------------
# Deep ensemble of MLPs (torch)
# ---------------------------------------------------------------------------
LOGVAR_RANGE = (-14.0, 6.0)


def _standardiser(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    tiny = 1e-12 * np.maximum(np.abs(mu), 1.0)
    sd = np.where(sd > tiny, sd, 1.0)                  # constant columns pass through centred
    return mu, sd


def _state_to_numpy(sd: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy().copy() for k, v in sd.items()}


def _state_from_numpy(sd: Mapping[str, Any]):
    import torch
    return {k: torch.as_tensor(np.asarray(v)) for k, v in sd.items()}


class MLPEnsemble:
    """A deep ensemble of small MLPs with a Gaussian negative-log-likelihood head.

    Each member maps standardised features to (mean, log-variance) of the
    standardised target and is trained with Adam on the Gaussian NLL (after
    `mse_warmup` epochs of squared error on the mean), with early stopping on a
    grouped validation split (whole samples) and its own seed. The prediction
    is the mixture: mean of the means, variance = mean of (variance + mean^2)
    - mean^2. `target_scale` names a feature column the target is divided by
    before fitting (and the prediction multiplied by): by default the elastic
    unloading curvature 3 sigma_f(0.2) / (E t), which takes the leading
    material and thickness dependence of springback out of what the networks
    must learn from a few dozen parts. Feature standardisation, the target
    scale and the target standardisation are part of the state.
    CPU only.
    """

    kind = "point"

    def __init__(self, n_members: int = 5, *, hidden: Sequence[int] = (128, 128),
                 epochs: int = 40, batch_size: int = 1024, lr: float = 2e-3,
                 weight_decay: float = 1e-5, patience: int = 6, val_fraction: float = 0.2,
                 mse_warmup: int = 3, bootstrap: bool = False,
                 target_scale: Optional[str] = DEFAULT_TARGET_SCALE, seed: int = 0,
                 n_threads: Optional[int] = None):
        if n_members < 1:
            raise ValueError("n_members must be >= 1")
        self.params = dict(n_members=int(n_members), hidden=[int(h) for h in hidden],
                           epochs=int(epochs), batch_size=int(batch_size), lr=float(lr),
                           weight_decay=float(weight_decay), patience=int(patience),
                           val_fraction=float(val_fraction), mse_warmup=int(mse_warmup),
                           bootstrap=bool(bootstrap), target_scale=target_scale,
                           seed=int(seed))
        self.n_threads = n_threads
        self.feature_names: List[str] = []
        self.x_mean = self.x_std = None
        self.y_mean = 0.0
        self.y_scale = 1.0
        self.states: List[Dict[str, np.ndarray]] = []
        self.history: List[Dict[str, Any]] = []
        self._nets: Optional[list] = None

    def _net(self, n_in: int):
        import torch.nn as nn
        layers: List[Any] = []
        prev = n_in
        for h in self.params["hidden"]:
            layers += [nn.Linear(prev, h), nn.SiLU()]
            prev = h
        layers.append(nn.Linear(prev, 2))
        return nn.Sequential(*layers)

    @staticmethod
    def _nll(out, t, mse: bool):
        import torch
        mu = out[:, 0]
        if mse:
            return torch.mean((mu - t) ** 2)
        lv = out[:, 1].clamp(*LOGVAR_RANGE)
        return torch.mean(0.5 * (lv + (t - mu) ** 2 * torch.exp(-lv)))

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, *,
            feature_names: Optional[Sequence[str]] = None) -> "MLPEnsemble":
        torch = _require_torch("MLPEnsemble")
        X, y, groups = _check_table(X, y, groups)
        if len(y) < 2:
            raise ValueError("MLPEnsemble.fit needs at least two rows")
        self.feature_names = list(feature_names) if feature_names is not None else \
            [f"x{i}" for i in range(X.shape[1])]
        self.x_mean, self.x_std = _standardiser(X)
        yn = y / _target_scale(self, X)
        self.y_mean = float(yn.mean())
        self.y_scale = float(yn.std()) or 1.0
        Xs = ((X - self.x_mean) / self.x_std).astype(np.float32)
        ts = ((yn - self.y_mean) / self.y_scale).astype(np.float32)
        p = self.params
        self.states, self.history = [], []
        with thread_limit(self.n_threads):
            for m in range(p["n_members"]):
                seed = p["seed"] * 1009 + m
                rng = np.random.default_rng(seed)
                tr, va = _grouped_holdout(groups, p["val_fraction"], rng)
                if p["bootstrap"]:
                    tr = tr[_group_bootstrap_rows(groups[tr], rng)]
                torch.manual_seed(seed)
                net = self._net(X.shape[1])
                opt = torch.optim.Adam(net.parameters(), lr=p["lr"],
                                       weight_decay=p["weight_decay"])
                gen = torch.Generator().manual_seed(seed)
                Xt, Tt = torch.from_numpy(Xs[tr]), torch.from_numpy(ts[tr])
                Xv, Tv = torch.from_numpy(Xs[va]), torch.from_numpy(ts[va])
                best, best_state, bad = math.inf, None, 0
                epochs_run = 0
                for epoch in range(p["epochs"]):
                    mse = epoch < p["mse_warmup"]
                    net.train()
                    perm = torch.randperm(len(Tt), generator=gen)
                    for i in range(0, len(Tt), p["batch_size"]):
                        b = perm[i:i + p["batch_size"]]
                        opt.zero_grad()
                        loss = self._nll(net(Xt[b]), Tt[b], mse)
                        loss.backward()
                        opt.step()
                    epochs_run = epoch + 1
                    if mse:
                        continue
                    net.eval()
                    with torch.no_grad():
                        vl = float(self._nll(net(Xv), Tv, False)) if len(Tv) else float(
                            self._nll(net(Xt), Tt, False))
                    if vl < best - 1e-4:
                        best, best_state, bad = vl, copy.deepcopy(net.state_dict()), 0
                    else:
                        bad += 1
                        if bad >= p["patience"]:
                            break
                if best_state is None:
                    best_state = net.state_dict()
                self.states.append(_state_to_numpy(best_state))
                self.history.append({"member": m, "epochs": epochs_run, "best_val_nll": best,
                                     "n_train_rows": int(len(tr)), "n_val_rows": int(len(va))})
        self._nets = None
        return self

    def _networks(self):
        if not self.states:
            raise PrecompError("MLPEnsemble is not fitted")
        if self._nets is None:
            _require_torch("MLPEnsemble")
            nets = []
            for sd in self.states:
                net = self._net(len(self.feature_names))
                net.load_state_dict(_state_from_numpy(sd))
                net.eval()
                nets.append(net)
            self._nets = nets
        return self._nets

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        import torch
        (X,) = _check_table(X)
        _check_names(self, X)
        nets = self._networks()
        Xs = torch.from_numpy(((X - self.x_mean) / self.x_std).astype(np.float32))
        mus, vars_ = [], []
        with thread_limit(self.n_threads), torch.no_grad():
            for net in nets:
                out = torch.cat([net(Xs[i:i + 65536]) for i in range(0, len(Xs), 65536)]) \
                    if len(Xs) else torch.zeros((0, 2))
                mus.append(out[:, 0].double().numpy())
                vars_.append(np.exp(out[:, 1].clamp(*LOGVAR_RANGE).double().numpy()))
        M = np.stack(mus)
        V = np.stack(vars_)
        mean = M.mean(axis=0)
        var = np.maximum((V + M * M).mean(axis=0) - mean * mean, 0.0)
        s = _target_scale(self, X)
        return (self.y_mean + self.y_scale * mean) * s, self.y_scale * np.sqrt(var) * s

    def to_state(self) -> Dict[str, Any]:
        if not self.states:
            raise PrecompError("MLPEnsemble is not fitted")
        return {"class": "MLPEnsemble", "params": dict(self.params),
                "feature_names": list(self.feature_names), "x_mean": self.x_mean,
                "x_std": self.x_std, "y_mean": self.y_mean, "y_scale": self.y_scale,
                "history": list(self.history),
                "torch_state_dicts": {f"member{m}": sd for m, sd in enumerate(self.states)}}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "MLPEnsemble":
        m = cls(**state["params"])
        m.feature_names = list(state["feature_names"])
        m.x_mean, m.x_std = np.asarray(state["x_mean"]), np.asarray(state["x_std"])
        m.y_mean, m.y_scale = float(state["y_mean"]), float(state["y_scale"])
        m.history = list(state.get("history", []))
        sds = state["torch_state_dicts"]
        m.states = [dict(sds[f"member{i}"]) for i in range(len(sds))]
        return m


# ---------------------------------------------------------------------------
# Transfer to a new process from a few parts
# ---------------------------------------------------------------------------
class TransferModel:
    """A frozen simulation-trained base plus a correction fitted on a few real parts.

    A handful of real parts supports only a low-dimensional correction. (An
    earlier default regressed the residual on the ~60 standardised features:
    from one to three real parts it made the base two to five times worse on
    proxy data, also when the "real" data equalled the simulation.)

    correction : "linear" (the only one) - y = a + (1 + b) mu, mu the base
      mean: a relative change of the springback magnitude, b ~ N(0,
      prior_gain^2), and an offset a ~ N(0, (prior_offset s)^2), s the RMS
      base prediction over the real parts. It is a Bayesian regression with a
      FIXED prior in which every real part weighs one observation (its nodes
      are not independent: they share weight 1), the noise variance estimated
      from the part-weighted residuals; the predictive std combines the base
      std and the parameter uncertainty in quadrature.

    Validation: a correction is kept only if its leave-one-real-part-out
    error - fitted without a part, scored on it, for every part - is below
    the base's on the same parts. Otherwise, and always with fewer than two
    real parts (nothing to validate on), the model predicts exactly as the
    base; `accepted` and `validation` say which and why. `n_real` is the
    number of real parts (groups) seen; with none the model IS the base.

    On proxy data (with the gradient-boosting base the package had then, ten
    held-out parts, 1-4 "real" parts in three draws each): with no shift, or
    50 um noise, the validated correction was never accepted (error = the
    base's); with the rim and curvature 30-40 % stronger (base 0.18 mm) it
    cut the error to 0.09-0.15 mm from two parts on; with one part nothing
    changes. Two real parts of ONE family can validate that family's offset
    and then shift every family (tests/test_ml_models.py): take the real
    parts from different families.
    """

    kind = "point"
    CORRECTIONS = ("linear",)

    def __init__(self, base: DeviationModel, correction: str = "linear", *, seed: int = 0,
                 prior_gain: float = 0.5, prior_offset: float = 0.5, min_parts: int = 2):
        if correction not in self.CORRECTIONS:
            raise ValueError(f"correction must be one of {self.CORRECTIONS}, got {correction!r}")
        if not (prior_gain > 0 and prior_offset > 0):
            raise ValueError("prior_gain and prior_offset must be > 0")
        if int(min_parts) < 2:
            raise ValueError("min_parts must be >= 2 (one part cannot be validated)")
        self.base = base
        self.correction = correction
        self.seed = int(seed)
        self.prior_gain = float(prior_gain)
        self.prior_offset = float(prior_offset)
        self.min_parts = int(min_parts)
        self.n_real = 0
        self.real_groups: List[str] = []
        self.accepted = False
        self.validation: Dict[str, Any] = {"reason": "no real parts"}
        self.theta: Optional[np.ndarray] = None
        self.cov: Optional[np.ndarray] = None

    @property
    def feature_names(self) -> List[str]:
        return list(getattr(self.base, "feature_names", []))

    # -- the correction -------------------------------------------------------
    @staticmethod
    def _design(mu: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones_like(mu), mu])

    def _fit_linear(self, mu: np.ndarray, y: np.ndarray, groups: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray]:
        uniq, inv = np.unique(groups, return_inverse=True)
        k = uniq.size
        w = 1.0 / np.bincount(inv)[inv]                      # each part weighs 1
        r = y - mu
        s = float(np.sqrt(np.sum(w * mu * mu) / k)) or 1.0
        Phi = self._design(mu)
        prec = np.array([1.0 / (self.prior_offset * s) ** 2, 1.0 / self.prior_gain ** 2])
        Lam = np.diag(prec)
        PtW = (Phi * w[:, None]).T
        s2 = max(float(np.sum(w * r * r)) / k, (1e-3 * s) ** 2)
        for _ in range(5):                                   # noise variance and theta
            A = PtW @ Phi / s2 + Lam
            theta = np.linalg.solve(A, PtW @ r / s2)
            s2 = max(float(np.sum(w * (r - Phi @ theta) ** 2)) / k, (1e-3 * s) ** 2)
        A = PtW @ Phi / s2 + Lam
        theta = np.linalg.solve(A, PtW @ r / s2)
        return theta, np.linalg.inv(A)

    def _apply(self, fitted: Any, mu: np.ndarray, sd: np.ndarray
               ) -> Tuple[np.ndarray, np.ndarray]:
        theta, cov = fitted
        Phi = self._design(mu)
        var = np.einsum("ij,jk,ik->i", Phi, cov, Phi)
        return mu + Phi @ theta, np.sqrt(sd * sd + np.maximum(var, 0.0))

    # -- protocol -------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, *,
            feature_names: Optional[Sequence[str]] = None) -> "TransferModel":
        """`groups`: the real PART of every row (the unit of validation)."""
        y = np.asarray(y, dtype=float).ravel()
        self.theta = self.cov = None
        self.accepted = False
        if len(y) == 0:
            self.n_real, self.real_groups = 0, []
            self.validation = {"reason": "no real parts"}
            return self
        X, y, groups = _check_table(X, y, groups)
        _check_names(self.base, X)
        groups = np.asarray(groups).astype(str)
        self.real_groups = sorted(np.unique(groups).tolist())
        self.n_real = len(self.real_groups)
        mu, sd = self.base.predict(X)
        rms = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))  # noqa: E731
        if self.n_real < self.min_parts:
            self.validation = {"reason": f"{self.n_real} real part(s): at least "
                               f"{self.min_parts} are needed to validate a correction, so "
                               "the model is the base", "n_parts": self.n_real}
            return self
        per = []
        for g in self.real_groups:
            out = groups == g
            fitted = self._fit_linear(mu[~out], y[~out], groups[~out])
            c, _ = self._apply(fitted, mu[out], sd[out])
            per.append({"part": g, "base_rms_m": rms(mu[out], y[out]),
                        "corrected_rms_m": rms(c, y[out])})
        b = float(np.mean([p["base_rms_m"] for p in per]))
        c = float(np.mean([p["corrected_rms_m"] for p in per]))
        self.accepted = c < b
        self.validation = {"reason": "leave-one-real-part-out error "
                           + ("below" if self.accepted else "not below") + " the base's",
                           "n_parts": self.n_real, "lopo_base_rms_m": b,
                           "lopo_corrected_rms_m": c, "per_part": per}
        if self.accepted:
            self.theta, self.cov = self._fit_linear(mu, y, groups)
            self.validation["theta"] = [float(t) for t in self.theta]
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.accepted:
            return self.base.predict(X)
        (X,) = _check_table(X)
        mu, sd = self.base.predict(X)
        return self._apply((self.theta, self.cov), mu, sd)

    def to_state(self) -> Dict[str, Any]:
        return {"class": "TransferModel", "base": self.base.to_state(),
                "correction": self.correction, "seed": self.seed,
                "prior_gain": self.prior_gain, "prior_offset": self.prior_offset,
                "min_parts": self.min_parts, "n_real": self.n_real,
                "real_groups": list(self.real_groups), "accepted": self.accepted,
                "validation": self.validation, "theta": self.theta, "cov": self.cov}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "TransferModel":
        m = cls(model_from_state(state["base"]), state["correction"], seed=state["seed"],
                prior_gain=state["prior_gain"], prior_offset=state["prior_offset"],
                min_parts=state["min_parts"])
        m.n_real = int(state["n_real"])
        m.real_groups = list(state["real_groups"])
        m.accepted = bool(state["accepted"])
        m.validation = dict(state["validation"])
        m.theta = None if state["theta"] is None else np.asarray(state["theta"])
        m.cov = None if state["cov"] is None else np.asarray(state["cov"])
        return m


MODEL_CLASSES = {"MLPEnsemble": MLPEnsemble, "TransferModel": TransferModel}


def model_from_state(state: Mapping[str, Any]):
    """Rebuild any model of this module from its `to_state()`."""
    cls = MODEL_CLASSES.get(state.get("class"))
    if cls is None:
        raise PrecompError(f"unknown model class {state.get('class')!r}; known: "
                           f"{sorted(MODEL_CLASSES)}")
    return cls.from_state(state)


__all__ = ["DeviationModel", "MLPEnsemble", "TransferModel", "model_from_state", "thread_limit",
           "MODEL_CLASSES", "THREADS_ENV"]
