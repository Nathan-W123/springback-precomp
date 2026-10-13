"""The deployable surrogate: features + model + conformal intervals + envelope.

`DeviationSurrogate` wraps a trained point model with the feature
configuration it was trained under, its conformal calibrator and its OOD
envelope, and states the data source it learned from. It implements what
`precomp.compensation` and `precomp.ml.compensate` expect of a model:

* ``predict_deviation(commanded, setup) -> (mean, std)`` - dz [m] at every
  node of the commanded grid;
* ``predict_interval(commanded, setup, level) -> (lower, upper)`` - conformal
  bounds of dz [m];
* ``assess(commanded, setup) -> dict`` - the envelope report, including the
  setup fields no feature describes (release, tool path style, mesh, ...):
  a query whose value was never seen in training is outside the envelope.

What it predicts (`target`, "dz") is the TOTAL vertical deviation of the
formed surface from the commanded one - also for a ResidualModel, whose
prior is evaluated inside the model - never a residual over a simulation, so
it must not be added to one.

A point model is trained on the nodes of `domain` ("part" by default); it is
evaluated at every node so the formed surface is defined everywhere, but only
the domain is in-distribution - the flange outside it is an extrapolation used
for display (displacement adjustment holds the flange and measures the error
on the part only).

`train_surrogate` does the whole fit: table, model, envelope, calibration on
held-out whole parts.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .._util import PrecompError, call_with_target, canonical_json
from ..geometry.heightmap import HeightMap
from .dataset import Sample, build_table, source_label
from .features import (DEFAULT_CONFIG, PRIOR_FEATURE, FeatureConfig, FeatureMaps, as_setup,
                       feature_maps, global_features)
from .models import ResidualModel, TransferModel, model_from_state
from .ood import OODEnvelope
from .uncertainty import ConformalCalibrator

SURROGATE_FORMAT = "precomp.ml.DeviationSurrogate/1"

#: FormingSetup fields the features describe (process and material features);
#: the envelope's descriptors cover them.
FEATURE_SETUP_FIELDS = ("material", "thickness", "tool_radius", "step_down", "friction")
#: Fields that are labels, not physics.
LABEL_SETUP_FIELDS = ("name",)
#: Process fields whose training ranges the envelope records.
PROCESS_FIELDS = ("tool_radius", "step_down", "thickness", "friction")


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)


def explicit_support_settings(settings: Mapping[str, Any], support: str) -> Dict[str, Any]:
    """`support_settings` without the keys that only spell out their default
    (`precomp.fea.setup.SUPPORT_SETTINGS` of `support`): {} and
    {"clearance": 1e-3} are the same backing plate."""
    from ..fea.setup import SUPPORT_SETTINGS

    defaults = SUPPORT_SETTINGS.get(support, {})
    return {k: v for k, v in dict(settings or {}).items()
            if not (k in defaults and defaults[k] is not None
                    and canonical_json(v) == canonical_json(defaults[k]))}


def _value_key(key: str, value: Any, doc: Mapping[str, Any]) -> str:
    """The canonical text a setup field's value is compared by: a support
    setting that spells out its default counts as the default."""
    if key == "support_settings":
        value = explicit_support_settings(value, doc.get("support", "none"))
    return canonical_json(value)


def setup_envelope(setups: Sequence[Any]) -> Dict[str, Any]:
    """What the training setups held: for every physics field of
    `FormingSetup` that no feature describes (blank, clamp, mesh, element,
    tool path style / spacing / direction, contact, increment, release,
    kinematics, solver, support and its settings - these without the keys
    that spell out a default, `explicit_support_settings`) its range
    (numbers) or its set of values; plus the ranges of the process fields
    and the material names, for reference."""
    docs = [as_setup(x).physics_dict() for x in setups]
    if not docs:
        raise ValueError("no setups")
    for d in docs:
        d["support_settings"] = explicit_support_settings(d["support_settings"], d["support"])
    fields: Dict[str, Any] = {}
    for key in docs[0]:
        if key in FEATURE_SETUP_FIELDS or key in LABEL_SETUP_FIELDS:
            continue
        vals = [d[key] for d in docs]
        if all(_is_number(v) for v in vals):          # ints stay ints (layers)
            fields[key] = {"min": min(vals), "max": max(vals)}
        else:
            uniq = {canonical_json(v): v for v in vals}
            fields[key] = {"values": [uniq[k] for k in sorted(uniq)]}
    process = {k: {"min": min(d[k] for d in docs), "max": max(d[k] for d in docs)}
               for k in PROCESS_FIELDS}
    materials = sorted({str(d["material"].get("name")) for d in docs})
    return {"fields": fields, "process": process, "materials": materials}


def _setup_default(key: str) -> Any:
    """The default of a FormingSetup field (a fresh copy of a factory's)."""
    from ..fea.setup import FormingSetup

    for f in dataclasses.fields(FormingSetup):
        if f.name == key:
            if f.default is not dataclasses.MISSING:
                return f.default
            if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                return f.default_factory()                     # type: ignore[misc]
    raise KeyError(key)


def setup_mismatch(envelope: Optional[Mapping[str, Any]], setup: Any) -> List[Dict[str, Any]]:
    """The fields of `setup` outside `envelope` (see `setup_envelope`):
    [{field, value, trained}], empty when every field was seen in training.

    A physics field the envelope does not record at all - the model was
    trained before the field existed (e.g. `support`) - was held at the
    field's default in every training run, so a setup is inside only with
    that default (the entry then says so in `note`)."""
    if not envelope:
        return []
    doc = as_setup(setup).physics_dict()
    fields = envelope.get("fields", {})
    out = []
    for key, rule in fields.items():
        v = doc.get(key)
        if "values" in rule:
            if _value_key(key, v, doc) not in {_value_key(key, x, doc) for x in rule["values"]}:
                out.append({"field": key, "value": v, "trained": rule["values"]})
        else:
            lo, hi = rule["min"], rule["max"]
            tol = 1e-9 * max(abs(lo), abs(hi), 1e-300)
            if not (_is_number(v) and lo - tol <= float(v) <= hi + tol):
                out.append({"field": key, "value": v, "trained": [lo, hi]})
    for key, v in doc.items():
        if key in fields or key in FEATURE_SETUP_FIELDS or key in LABEL_SETUP_FIELDS:
            continue
        default = _setup_default(key)
        if canonical_json(v) != canonical_json(default):
            out.append({"field": key, "value": v, "trained": [default],
                        "note": "not recorded by the model (it predates the field): "
                                "trained at the default"})
    return out


def seen_ids(training: Mapping[str, Any]) -> Tuple[set, set]:
    """(sample ids, part ids) a surrogate was trained, corrected or
    calibrated on, from its training record (a transfer surrogate's includes
    its base's and its real data)."""
    samples: set = set()
    parts: set = set()
    for key in ("sample_ids", "calibration_sample_ids"):
        samples |= set(training.get(key) or [])
    for key in ("part_ids", "calibration_part_ids"):
        parts |= set(training.get(key) or [])
    tr = training.get("transfer") or {}
    for key in ("real_sample_ids", "calibration_sample_ids"):
        samples |= set(tr.get(key) or [])
    for key in ("real_part_ids", "calibration_part_ids"):
        parts |= set(tr.get(key) or [])
    if isinstance(training.get("base"), Mapping):
        a, b = seen_ids(training["base"])
        samples |= a
        parts |= b
    return samples, parts


def describe_prior(prior: Any) -> Dict[str, Any]:
    """What a ResidualModel's prior is, for the manifest: its own
    `describe()` when it has one, with its data-source label."""
    d = dict(prior.describe()) if hasattr(prior, "describe") else {}
    d.setdefault("name", type(prior).__name__)
    label = getattr(prior, "data_source", None) or getattr(prior, "label", None)
    if label is None:
        label = "SparLab simulation" if type(prior).__name__ == "FEAPrior" else "unknown"
    d["data_source"] = label
    return d


def _key(commanded: HeightMap, setup: Any, target: Optional[HeightMap] = None) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(commanded.z).tobytes())
    h.update(np.ascontiguousarray(commanded.mask).tobytes())
    h.update(canonical_json(commanded.grid.to_dict()).encode())
    h.update(canonical_json(as_setup(setup).physics_dict()).encode())
    if target is not None and as_setup(setup).support != "none":
        # the fixture a support is made for changes what the FE prior simulates
        h.update(b"target")
        h.update(np.ascontiguousarray(target.z).tobytes())
        h.update(np.ascontiguousarray(target.mask).tobytes())
        h.update(canonical_json(target.grid.to_dict()).encode())
    return h.hexdigest()


def model_prior(model: Any) -> Any:
    """The physics prior a model needs as its last column, or None."""
    if isinstance(model, ResidualModel):
        return model.prior
    if isinstance(model, TransferModel):
        return model_prior(model.base)
    return None


class DeviationSurrogate:
    """A trained deviation model ready for prediction (see the module docstring).

    model : a fitted point model (`models.DeviationModel`).
    features : the FeatureConfig it was trained with.
    data_source : the label of its training data ("SparLab simulation",
        "proxy - not physics", "scan" or "mixed: ...").
    training : provenance of the fit (sample ids, counts, families, seed ...).
    calibrator, ood : optional fitted ConformalCalibrator / OODEnvelope.
    domain : the training mask ("part" or "all").
    """

    #: What the model predicts: the total vertical deviation dz of the formed
    #: surface from the commanded one (see the module docstring).
    target = "dz"

    def __init__(self, model: Any, features: FeatureConfig = DEFAULT_CONFIG, *,
                 data_source: str, training: Optional[Dict[str, Any]] = None,
                 calibrator: Optional[ConformalCalibrator] = None,
                 ood: Optional[OODEnvelope] = None, domain: str = "part"):
        if getattr(model, "kind", None) != "point":
            raise TypeError("model must be a point model of precomp.ml.models")
        self.model = model
        self.features = features
        self.data_source = data_source
        self.training = dict(training or {})
        self.calibrator = calibrator
        self.ood = ood
        self.domain = domain
        #: the manifest when loaded from a bundle (`registry.load_model`)
        self.manifest: Optional[Dict[str, Any]] = None
        self._cache: "OrderedDict[str, Any]" = OrderedDict()

    # -- state -----------------------------------------------------------------
    @property
    def kind(self) -> str:
        return self.model.kind

    @property
    def feature_names(self) -> List[str]:
        names = self.features.names
        if model_prior(self.model) is not None:
            names = names + [PRIOR_FEATURE]
        return names

    @property
    def model_class(self) -> str:
        return type(self.model).__name__

    def to_state(self) -> Dict[str, Any]:
        return {"format": SURROGATE_FORMAT, "model": self.model.to_state(),
                "features": self.features.to_dict(), "data_source": self.data_source,
                "training": self.training, "domain": self.domain,
                "calibrator": None if self.calibrator is None else self.calibrator.to_state(),
                "ood": None if self.ood is None else self.ood.to_state()}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "DeviationSurrogate":
        if state.get("format") != SURROGATE_FORMAT:
            raise PrecompError(f"not a surrogate state (format {state.get('format')!r})")
        return cls(model_from_state(state["model"]), FeatureConfig.from_dict(state["features"]),
                   data_source=state["data_source"], training=state.get("training"),
                   calibrator=None if state.get("calibrator") is None
                   else ConformalCalibrator.from_state(state["calibrator"]),
                   ood=None if state.get("ood") is None
                   else OODEnvelope.from_state(state["ood"]),
                   domain=state.get("domain", "part"))

    # -- evaluation ------------------------------------------------------------
    #: Cached entries per kind: feature maps are large (ny x nx x F), the
    #: predictions and envelope reports small. The key is a hash of the
    #: commanded surface and the setup's physics.
    CACHE_SIZES = {"maps": 4, "pred": 128, "assess": 128}

    def _cached(self, key: str, make):
        kind = key.split(":", 1)[0]
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        val = make()
        self._cache[key] = val
        keys = [k for k in self._cache if k.startswith(kind + ":")]
        for k in keys[:max(0, len(keys) - self.CACHE_SIZES.get(kind, 8))]:
            del self._cache[k]
        return val

    @property
    def prior(self) -> Any:
        """The physics prior evaluated inside the model (ResidualModel), or None."""
        return model_prior(self.model)

    def feature_maps(self, commanded: HeightMap, setup: Any) -> FeatureMaps:
        return self._cached("maps:" + _key(commanded, setup),
                            lambda: feature_maps(commanded, setup, None, self.features))

    def _point_matrix(self, commanded: HeightMap, setup: Any,
                      target: Optional[HeightMap] = None) -> np.ndarray:
        X = self.feature_maps(commanded, setup).gather(None)
        prior = model_prior(self.model)
        if prior is not None:
            p = np.asarray(call_with_target(prior.prior_deviation, commanded, as_setup(setup),
                                            target=target), float)
            if p.shape != commanded.grid.shape or not np.all(np.isfinite(p)):
                raise PrecompError("the model's prior returned a field of the wrong shape or "
                                   "non-finite values")
            X = np.column_stack([X, p.ravel()])
        return X

    def predict_deviation(self, commanded: HeightMap, setup: Any,
                          target: Optional[HeightMap] = None
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """(mean, std) of dz [m], (ny, nx), at every node of the commanded grid.
        `target`: the part a support's fixture is made for, which a model's
        FE prior simulates with (`models.FEAPrior`; None: the commanded
        surface's outline)."""
        def make():
            X = self._point_matrix(commanded, setup, target)
            mu, sd = self.model.predict(X)
            shape = commanded.grid.shape
            return np.asarray(mu, float).reshape(shape), np.asarray(sd, float).reshape(shape)
        mu, sd = self._cached("pred:" + _key(commanded, setup, target), make)
        return mu.copy(), sd.copy()

    def predict_interval(self, commanded: HeightMap, setup: Any, level: float = 0.9,
                         target: Optional[HeightMap] = None
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """Conformal (lower, upper) bounds of dz [m] at coverage `level`
        (`target` as for `predict_deviation`)."""
        if self.calibrator is None or not self.calibrator.fitted:
            raise PrecompError("this surrogate has no conformal calibration; train it with "
                               "calibration samples")
        mu, sd = self.predict_deviation(commanded, setup, target)
        region = self.feature_maps(commanded, setup).region if self.calibrator.mondrian \
            else None
        return self.calibrator.intervals(mu, sd, level, region)

    def assess(self, commanded: Any, setup: Any = None, *, max_points: int = 1500
               ) -> Dict[str, Any]:
        """The envelope report of a commanded part - a HeightMap with its
        setup, or a `Sample` - (see `ood.OODEnvelope`) on at most `max_points`
        part nodes (evenly spread), with the data source the model learned
        from. A setup field that no feature describes and whose value the
        training data never had (`setup_mismatch`, e.g. another release or
        tool path style) puts the part outside the envelope too; the reasons
        then name it first ("setup.<field>")."""
        if self.ood is None:
            raise PrecompError("this surrogate has no OOD envelope")
        if isinstance(commanded, Sample):
            commanded, setup = commanded.commanded, commanded.setup
        if setup is None:
            raise ValueError("assess needs the setup with a commanded surface")

        def make() -> Dict[str, Any]:
            fm = self.feature_maps(commanded, setup)
            nodes = np.flatnonzero(fm.part.ravel())
            if nodes.size > max_points:
                nodes = nodes[np.linspace(0, nodes.size - 1, max_points).round().astype(int)]
            d, _ = global_features(commanded, setup, self.features)
            rep = self.ood.assess_features(fm.gather(nodes), d)
            mism = setup_mismatch(self.training.get("setup"), setup)
            rep["setup_mismatch"] = mism
            if mism:
                rep["in_envelope"] = False
                rep["reasons"] = [f"setup.{m['field']}" for m in mism] + rep["reasons"]
            rep["model_data_source"] = self.data_source
            return rep
        import copy
        return copy.deepcopy(self._cached(f"assess:{max_points}:" + _key(commanded, setup),
                                          make))

    def describe(self) -> Dict[str, Any]:
        return {"model_class": self.model_class, "kind": self.kind, "target": self.target,
                "data_source": self.data_source, "domain": self.domain,
                "prior": None if self.prior is None else describe_prior(self.prior),
                "features": self.features.to_dict(), "training": self.training,
                "calibration": None if self.calibrator is None else self.calibrator.summary(),
                "ood": None if self.ood is None else self.ood.summary()}


def fit_envelope(samples: Sequence[Sample], features: FeatureConfig, *,
                 points_per_sample: int = 400, seed: int = 0, n_jobs: int = 1,
                 points: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 **kwargs: Any) -> OODEnvelope:
    """An OOD envelope from the training samples (their part nodes and
    descriptors, grouped by part id). `points` = (X, part ids) of part nodes
    already featurised (e.g. the training table) saves featurising again."""
    if points is None:
        table = build_table(samples, features, points_per_sample=points_per_sample,
                            mask="part", rng=seed + 101, stratify=True, n_jobs=n_jobs)
        points = (table.X, table.part_id)
    X, groups = points
    if X.shape[1] != len(features.names):
        raise ValueError("the envelope's node features must follow the feature schema")
    D, names = [], None
    for s in samples:
        d, names = global_features(s.commanded, s.setup, features)
        D.append(d)
    return OODEnvelope(seed=seed, **kwargs).fit(
        X, groups, np.stack(D), np.array([s.part_id for s in samples]),
        point_names=features.names, part_names=names)


def calibrate(surrogate: DeviationSurrogate, samples: Sequence[Sample], *,
              points_per_sample: Optional[int] = None, mondrian: bool = False,
              seed: int = 0) -> ConformalCalibrator:
    """Fit a ConformalCalibrator on held-out samples: every part node (or a
    uniform subsample of `points_per_sample`) of each sample, grouped by PART
    id - the variants of one part are one calibration unit."""
    mus, sds, ys, gs, rs = [], [], [], [], []
    rng = np.random.default_rng(seed)
    for s in samples:
        try:
            mu, sd = surrogate.predict_deviation(s.commanded, s.setup, s.target)
            fm = surrogate.feature_maps(s.commanded, s.setup)
        except (ValueError, PrecompError) as exc:
            raise PrecompError(f"sample {s.sample_id}: {exc}") from exc
        sel = np.flatnonzero((fm.part & s.valid).ravel())
        if points_per_sample is not None and sel.size > points_per_sample:
            sel = np.sort(rng.choice(sel, size=points_per_sample, replace=False))
        mus.append(mu.ravel()[sel])
        sds.append(sd.ravel()[sel])
        ys.append(s.dz.ravel()[sel])
        gs.append(np.full(sel.size, s.part_id, dtype=object))   # the part is the unit
        rs.append(fm.region.ravel()[sel])
    cal = ConformalCalibrator(mondrian=mondrian)
    return cal.fit(np.concatenate(mus), np.concatenate(sds), np.concatenate(ys),
                   np.concatenate(gs), region=np.concatenate(rs),
                   data_source=source_label(s.source for s in samples))


def _ids_hash(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def train_surrogate(model: Any, train: Sequence[Sample], *,
                    calibration: Optional[Sequence[Sample]] = None,
                    features: FeatureConfig = DEFAULT_CONFIG,
                    points_per_sample: Optional[int] = 2000, mask: str = "part",
                    seed: int = 0, envelope: bool = True, mondrian: bool = False,
                    calibration_points: Optional[int] = None, n_jobs: int = 1,
                    group_by: str = "sample") -> DeviationSurrogate:
    """Fit `model` on `train`, then the envelope, then calibrate on `calibration`.

    The model gets a stratified table (`points_per_sample` per sample, the
    physics prior's column appended for residual models). `calibration` must hold parts that are not
    in `train` (checked by part id); without it the surrogate has no
    intervals. The training provenance - sample and part ids (and a hash),
    families, sources, the data-source label, the setup fields seen
    (`setup_envelope`), the prior of a residual model, seed and sizes - is
    kept in `surrogate.training`.

    group_by : the groups the model's bootstrap (GBMEnsemble) or
        early-stopping split (MLPEnsemble) draws: "sample" (default) or
        "part". The variants of a part are near duplicates, so with "sample"
        a bootstrap member sees about 95 % of the parts and the ensemble
        spread is far smaller than the error on a new part - it is a relative
        scale that conformal calibration turns into intervals, not an error
        estimate. "part" gives a larger spread but, on proxy data, an 18 %
        larger held-out error and 90 % intervals almost twice as wide after
        calibration (docs/precomp_ml.md, "Models").
    """
    train = list(train)
    if not train:
        raise ValueError("no training samples")
    if group_by not in ("sample", "part"):
        raise ValueError("group_by must be 'sample' or 'part'")
    cal = list(calibration or [])
    overlap = {s.part_id for s in train} & {s.part_id for s in cal}
    if overlap:
        raise PrecompError(f"{len(overlap)} part(s) are in both the training and the "
                           f"calibration set, e.g. {sorted(overlap)[:3]}; calibrate on held-out "
                           "parts")
    label = source_label(s.source for s in train)
    info: Dict[str, Any] = {
        "sample_ids": sorted(s.sample_id for s in train),
        "sample_ids_sha256": _ids_hash([s.sample_id for s in train]),
        "part_ids": sorted({s.part_id for s in train}),
        "n_samples": len(train), "n_parts": len({s.part_id for s in train}),
        "families": sorted({s.family for s in train}),
        "sources": sorted({s.source for s in train}),
        "fidelities": sorted({s.fidelity for s in train}), "data_source": label,
        "sparlab_versions": sorted({str(s.provenance["sparlab_version"]) for s in train
                                    if s.provenance.get("sparlab_version")}),
        "seed": int(seed), "mask": mask, "points_per_sample": points_per_sample,
        "group_by": group_by, "grid_spacings": sorted({float(s.grid.h) for s in train}),
        "calibration_sample_ids": sorted(s.sample_id for s in cal),
        "calibration_part_ids": sorted({s.part_id for s in cal}),
        "setup": setup_envelope([s.setup for s in train])}
    if model_prior(model) is not None:
        info["prior"] = describe_prior(model_prior(model))
    table = build_table(train, features, points_per_sample=points_per_sample, mask=mask,
                        rng=seed, stratify=True, prior=model_prior(model), n_jobs=n_jobs)
    model.fit(table.X, table.y, table.groups if group_by == "sample" else table.part_id,
              feature_names=table.feature_names)
    info["n_rows"] = len(table)
    sur = DeviationSurrogate(model, features, data_source=label, training=info, domain=mask)
    if envelope:
        reuse = None
        if mask == "part":
            # the training table's part nodes, without a prior column, thinned
            # to at most 400 per sample (the envelope keeps 5000 anyway)
            ncol = len(features.names)
            rng = np.random.default_rng(seed + 101)
            keep = np.concatenate([rng.permutation(np.flatnonzero(table.groups == g))[:400]
                                   for g in np.unique(table.groups)])
            reuse = (table.X[np.sort(keep), :ncol], table.part_id[np.sort(keep)])
        sur.ood = fit_envelope(train, features, seed=seed, n_jobs=n_jobs, points=reuse)
    if cal:
        sur.calibrator = calibrate(sur, cal, points_per_sample=calibration_points,
                                   mondrian=mondrian, seed=seed)
    return sur


def transfer_surrogate(base: DeviationSurrogate, real: Sequence[Sample], *,
                       calibration: Optional[Sequence[Sample]] = None,
                       correction: str = "linear",
                       points_per_sample: Optional[int] = 2000, seed: int = 0,
                       mondrian: bool = False, **correction_kwargs: Any) -> DeviationSurrogate:
    """A `TransferModel` surrogate: `base`'s model frozen, corrected on the few
    `real` samples (scans, or a higher-fidelity simulation), grouped by part.

    The correction is kept only when it beats the base on held-out real parts
    (leave one real part out; see `TransferModel`) - with fewer than two real
    parts the surrogate predicts as the base, and says so in
    `training["transfer"]["validation"]`. The feature envelope stays the
    base's (the correction does not widen what the base has seen); the setup
    fields it accepts are the base's and the real parts'. Intervals are
    calibrated only on `calibration` - held-out real parts; without them the
    transfer surrogate has no intervals rather than intervals calibrated on
    another data source. The training record lists the base's and the real
    samples, parts and families (so evaluation refuses both), and the real
    calibration ids.
    """
    if base.kind != "point":
        raise TypeError("transfer learning is implemented for point models")
    real = list(real)
    cal = list(calibration or [])
    overlap = {s.part_id for s in real} & {s.part_id for s in cal}
    if overlap:
        raise PrecompError(f"calibration parts {sorted(overlap)[:3]} were used for the "
                           "correction")
    base_seen_s, base_seen_p = seen_ids(base.training)
    reused = sorted({s.part_id for s in cal if s.part_id in base_seen_p or
                     s.sample_id in base_seen_s})
    if reused:
        raise PrecompError(f"calibration parts {reused[:3]} were used to train or calibrate "
                           "the base model")
    model = TransferModel(base.model, correction, seed=seed, **correction_kwargs)
    if real:
        table = build_table(real, base.features, points_per_sample=points_per_sample,
                            mask=base.domain, rng=seed, prior=model_prior(base.model))
        model.fit(table.X, table.y, table.part_id, feature_names=table.feature_names)
    info = dict(base.training)
    info["base"] = {k: base.training.get(k) for k in ("sample_ids_sha256", "n_samples",
                                                       "n_parts", "families", "data_source")}
    real_s = sorted(s.sample_id for s in real)
    real_p = sorted({s.part_id for s in real})
    info["sample_ids"] = sorted(set(base.training.get("sample_ids", [])) | set(real_s))
    info["sample_ids_sha256"] = _ids_hash(info["sample_ids"])
    info["part_ids"] = sorted(set(base.training.get("part_ids", [])) | set(real_p))
    info["families"] = sorted(set(base.training.get("families", []))
                              | {s.family for s in real})
    info["sources"] = sorted(set(base.training.get("sources", [])) | {s.source for s in real})
    info["calibration_sample_ids"] = sorted(set(base.training.get("calibration_sample_ids", []))
                                            | {s.sample_id for s in cal})
    info["calibration_part_ids"] = sorted(set(base.training.get("calibration_part_ids", []))
                                          | {s.part_id for s in cal})
    if real:
        env = dict(base.training.get("setup") or {})
        if env:
            info["setup"] = setup_envelope_union(env, setup_envelope([s.setup for s in real]))
    info["transfer"] = {"correction": correction, "n_real": len(real_s),
                        "n_real_parts": len(real_p), "real_sample_ids": real_s,
                        "real_part_ids": real_p,
                        "real_families": sorted({s.family for s in real}),
                        "real_sources": sorted({s.source for s in real}),
                        "calibration_sample_ids": sorted(s.sample_id for s in cal),
                        "calibration_part_ids": sorted({s.part_id for s in cal}),
                        "accepted": model.accepted, "validation": model.validation}
    sur = DeviationSurrogate(model, base.features, data_source=source_label(info["sources"]),
                             training=info, ood=base.ood, domain=base.domain)
    if cal:
        sur.calibrator = calibrate(sur, cal, mondrian=mondrian, seed=seed)
    return sur


def setup_envelope_union(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
    """The setup envelope covering both `a` and `b` (see `setup_envelope`)."""
    fields: Dict[str, Any] = {}
    for key in set(a.get("fields", {})) | set(b.get("fields", {})):
        ra, rb = a.get("fields", {}).get(key), b.get("fields", {}).get(key)
        if ra is None or rb is None:
            # an envelope without the field predates it: its runs had the default
            known = ra or rb
            try:
                default = _setup_default(key)
            except KeyError:
                fields[key] = known
                continue
            if "values" in known:
                vals = {canonical_json(v): v for v in known["values"] + [default]}
                fields[key] = {"values": [vals[k] for k in sorted(vals)]}
            elif _is_number(default):
                fields[key] = {"min": min(known["min"], default),
                               "max": max(known["max"], default)}
            else:
                fields[key] = known
        elif "values" in ra or "values" in rb:
            vals = {canonical_json(v): v for v in ra.get("values", []) + rb.get("values", [])}
            fields[key] = {"values": [vals[k] for k in sorted(vals)]}
        else:
            fields[key] = {"min": min(ra["min"], rb["min"]), "max": max(ra["max"], rb["max"])}
    process = {}
    for key in set(a.get("process", {})) | set(b.get("process", {})):
        ra, rb = a.get("process", {}).get(key), b.get("process", {}).get(key)
        if ra is None or rb is None:
            process[key] = ra or rb
        else:
            process[key] = {"min": min(ra["min"], rb["min"]), "max": max(ra["max"], rb["max"])}
    return {"fields": fields, "process": process,
            "materials": sorted(set(a.get("materials", [])) | set(b.get("materials", [])))}


__all__ = ["DeviationSurrogate", "train_surrogate", "transfer_surrogate", "calibrate",
           "fit_envelope", "model_prior", "setup_envelope", "setup_mismatch", "seen_ids",
           "describe_prior", "explicit_support_settings", "SURROGATE_FORMAT"]
