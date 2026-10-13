"""Model bundles on disk: save, load, and refuse what does not match.

A bundle is a directory

    manifest.json     what the model is and where it came from (below)
    model.joblib      the surrogate's state (scikit-learn estimators, arrays)
    torch/*.pt        torch state dicts of the neural models (tensors only)

manifest.json holds: format, model_class, kind, feature_schema_version,
feature_schema_hash, feature_names and their hash, the feature configuration,
the data source of the training data ("SparLab simulation", "proxy - not
physics", "scan" or "mixed: ..."), the training record (sample ids or their
hash and count, parts, families, sources, fidelities, SparLab versions, seed,
sizes), metrics (each block carrying its own data source), the conformal
quantiles, the OOD thresholds, package versions, the git revision of the
checkout, created_at (given by the caller, never read from the clock here),
free notes, and the SHA-256 of every file.

`load_model` refuses a bundle whose feature schema version, feature names
or schema hash differ from what this code computes - a model must never be
fed features it was not trained on -, a manifest without the hashes of
model.joblib and of every torch file the state refers to, a file that does
not match its recorded hash, and a state whose model class, kind, domain or
data source differ from the manifest's. The manifest is written last, after
it has been checked to be valid JSON, so an interrupted save leaves no
loadable bundle. model.joblib is a pickle: load bundles you trust only.

Where this fits: the end of "Training": a trained surrogate is saved as a bundle, and the
    scripts load it instead of retraining when its training samples match.
Called from: springback_fine_ml.py:train_or_load (load_model, save_model),
    combined_rim_ml.py:transfer_model (load_model, save_model).
Calls into: ml.surrogate (DeviationSurrogate state), ml.features (schema checks), joblib, torch.
Start reading at: save_model, then load_model.
"""

from __future__ import annotations

import hashlib
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from .. import __version__
from .._util import PathLike, PrecompError, canonical_json, read_json
from .features import FEATURE_SCHEMA_VERSION, FeatureConfig, names_hash
from .surrogate import DeviationSurrogate

MODEL_FORMAT = "precomp.ml.model/1"
MANIFEST_FILE = "manifest.json"
STATE_FILE = "model.joblib"
TORCH_DIR = "torch"
TORCH_KEY = "torch_state_dicts"
TORCH_FILES_KEY = "__torch_files__"

#: Sample ids are listed in the manifest up to this many (always hashed).
MAX_LISTED_IDS = 5000


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def package_versions() -> Dict[str, Optional[str]]:
    """Versions of Python and the libraries a bundle depends on."""
    out: Dict[str, Optional[str]] = {"python": platform.python_version(), "precomp": __version__}
    for mod in ("numpy", "scipy", "sklearn", "pandas", "joblib", "torch"):
        try:
            m = sys.modules.get(mod) or __import__(mod)
            out[mod] = str(getattr(m, "__version__", "unknown"))
        except ImportError:
            out[mod] = None
    return out


def git_sha() -> Optional[str]:
    """HEAD of the checkout this package runs from, or None when not a git tree."""
    from ..fea.setup import repository_root

    root = repository_root()
    if root is None:
        return None
    try:
        proc = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    sha = proc.stdout.strip()
    if proc.returncode != 0 or not sha:
        return None
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain",
                            "--untracked-files=no"], capture_output=True, text=True, timeout=10)
    return sha + ("-dirty" if dirty.returncode == 0 and dirty.stdout.strip() else "")


def _extract_torch(state: Any, prefix: str, files: Dict[str, Dict[str, np.ndarray]]) -> Any:
    """Replace every `torch_state_dicts` block by file references (recursively)."""
    if isinstance(state, dict):
        out = {}
        for k, v in state.items():
            if k == TORCH_KEY and isinstance(v, dict):
                refs = {}
                for name, sd in v.items():
                    rel = f"{TORCH_DIR}/{prefix or 'model'}__{name}.pt"
                    files[rel] = sd
                    refs[name] = rel
                out[TORCH_FILES_KEY] = refs
            else:
                out[k] = _extract_torch(v, f"{prefix}.{k}" if prefix else str(k), files)
        return out
    return state


def _torch_refs(state: Any) -> list:
    """Every torch file a (loaded) state refers to."""
    if isinstance(state, dict):
        out = []
        for k, v in state.items():
            out += list(v.values()) if k == TORCH_FILES_KEY else _torch_refs(v)
        return out
    return []


def _restore_torch(state: Any, directory: Path) -> Any:
    if isinstance(state, dict):
        out = {}
        for k, v in state.items():
            if k == TORCH_FILES_KEY:
                try:
                    import torch
                except ImportError as exc:
                    raise PrecompError(
                        f"{directory} holds a neural model ({', '.join(sorted(v.values()))}); "
                        "loading it needs torch: pip install -e '.[torch]'") from exc
                out[TORCH_KEY] = {name: {key: t.detach().cpu().numpy() for key, t in
                                         torch.load(directory / rel, weights_only=True,
                                                    map_location="cpu").items()}
                                  for name, rel in v.items()}
            else:
                out[k] = _restore_torch(v, directory)
        return out
    return state


def _finite_or_none(value: Any) -> Any:
    """Metrics with every non-finite number (a metric that could not be
    computed, e.g. coverage without calibration) replaced by None: JSON has no
    NaN."""
    if isinstance(value, Mapping):
        return {k: _finite_or_none(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_or_none(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def build_manifest(model: DeviationSurrogate, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """The manifest document of `model` (without the file hashes)."""
    if not isinstance(manifest.get("created_at"), str) or not manifest["created_at"]:
        raise ValueError("manifest['created_at'] must be a non-empty string supplied by the "
                         "caller")
    metrics = _finite_or_none(manifest.get("metrics") or {})
    for name, block in metrics.items():
        if isinstance(block, Mapping) and "data_source" not in block:
            raise ValueError(f"metrics block {name!r} does not state its data_source")
    cfg = model.features
    names = model.feature_names
    tr = dict(model.training)
    ids = tr.pop("sample_ids", [])
    cal_ids = tr.pop("calibration_sample_ids", [])
    training = dict(tr)
    if len(ids) <= MAX_LISTED_IDS:
        training["sample_ids"] = list(ids)
    if len(cal_ids) <= MAX_LISTED_IDS:
        training["calibration_sample_ids"] = list(cal_ids)
    doc = {"format": MODEL_FORMAT, "model_class": model.model_class, "kind": model.kind,
           "target": model.target,
           "feature_schema_version": FEATURE_SCHEMA_VERSION,
           "feature_schema_hash": cfg.schema_hash(), "feature_config": cfg.to_dict(),
           "feature_names": list(names), "feature_names_sha256": names_hash(names),
           "data_source": model.data_source, "domain": model.domain, "training": training,
           "metrics": metrics,
           "conformal": None if model.calibrator is None else model.calibrator.summary(),
           "ood": None if model.ood is None else model.ood.summary(),
           "package_versions": package_versions(),
           "sparlab_version": manifest.get("sparlab_version",
                                           tr.get("sparlab_versions") or None),
           "git_sha": manifest.get("git_sha", git_sha()),
           "created_at": manifest["created_at"], "notes": manifest.get("notes", "")}
    extra = {k: v for k, v in manifest.items() if k not in doc and k != "metrics"}
    doc.update(extra)
    return doc


def save_model(model: DeviationSurrogate, directory: PathLike, manifest: Mapping[str, Any], *,
               overwrite: bool = False) -> Path:
    """Write the bundle of `model` into `directory` and return it.

    `manifest` supplies at least `created_at`; optional `metrics` (each block
    a dict with its `data_source`), `notes`, `sparlab_version`, `git_sha` and
    any further keys, kept verbatim. An existing non-empty directory is
    refused unless `overwrite`.
    """
    if not isinstance(model, DeviationSurrogate):
        raise TypeError("save_model expects a DeviationSurrogate (see precomp.ml.surrogate)")
    d = Path(directory)
    if d.exists() and any(d.iterdir()) and not overwrite:
        raise PrecompError(f"{d} is not empty; pass overwrite=True to replace the bundle")
    # everything that can fail on the content fails before a file is touched
    doc = build_manifest(model, manifest)
    canonical_json(doc)                       # PrecompError on a non-finite number
    files: Dict[str, Dict[str, np.ndarray]] = {}
    state = _extract_torch(model.to_state(), "", files)
    if d.exists():
        (d / MANIFEST_FILE).unlink(missing_ok=True)     # the bundle is invalid from here
        (d / STATE_FILE).unlink(missing_ok=True)
        shutil.rmtree(d / TORCH_DIR, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(state, d / STATE_FILE, compress=3)
    if files:
        import torch
        (d / TORCH_DIR).mkdir(exist_ok=True)
        for rel, sd in files.items():
            torch.save({k: torch.as_tensor(np.asarray(v)) for k, v in sd.items()}, d / rel)
    doc["files"] = {rel: _sha256_file(d / rel) for rel in [STATE_FILE] + sorted(files)}
    text = canonical_json(doc)
    tmp = d / f".{MANIFEST_FILE}.tmp"
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(d / MANIFEST_FILE)             # the manifest last: the bundle is complete
    return d


def read_manifest(directory: PathLike) -> Dict[str, Any]:
    d = Path(directory)
    doc = read_json(d / MANIFEST_FILE)
    if doc.get("format") != MODEL_FORMAT:
        raise PrecompError(f"{d / MANIFEST_FILE}: format {doc.get('format')!r}, expected "
                           f"{MODEL_FORMAT!r}")
    return doc


def check_schema(doc: Mapping[str, Any], where: str = "the model") -> FeatureConfig:
    """The bundle's FeatureConfig; PrecompError unless this code computes the
    same features (schema version, names and schema hash)."""
    v = doc.get("feature_schema_version")
    if v != FEATURE_SCHEMA_VERSION:
        raise PrecompError(
            f"{where} was trained with feature schema version {v!r}, but this precomp "
            f"computes version {FEATURE_SCHEMA_VERSION!r}; the features differ, so the model "
            "cannot be used. Retrain it, or use the precomp version it was trained with.")
    cfg = FeatureConfig.from_dict(doc["feature_config"])
    if cfg.schema_hash() != doc.get("feature_schema_hash"):
        raise PrecompError(f"{where}: the feature schema hash differs from the one this code "
                           "computes for the same configuration (feature definitions changed "
                           "without a version bump?); refusing to load")
    names = list(doc.get("feature_names") or [])
    if names_hash(names) != doc.get("feature_names_sha256"):
        raise PrecompError(f"{where}: feature_names do not match feature_names_sha256")
    return cfg


def load_model(directory: PathLike) -> DeviationSurrogate:
    """Load a bundle written by `save_model` (see the module docstring for
    what is checked). Returns a DeviationSurrogate, which implements
    `predict_deviation`, `predict_interval` and `assess`."""
    d = Path(directory)
    if not d.is_dir():
        raise PrecompError(f"{d} is not a model directory")
    doc = read_manifest(d)
    cfg = check_schema(doc, str(d))
    # verify every file against its SHA-256 in the manifest before unpickling anything
    files = doc.get("files")
    if not isinstance(files, Mapping) or STATE_FILE not in files:
        raise PrecompError(f"{d / MANIFEST_FILE} does not record the SHA-256 of {STATE_FILE}; "
                           "the bundle cannot be verified, refusing to load it")
    for rel, digest in files.items():
        path = d / rel
        if not path.is_file():
            raise PrecompError(f"{path} is missing from the bundle")
        if _sha256_file(path) != digest:
            raise PrecompError(f"{path} does not match its SHA-256 in the manifest; the bundle "
                               "was modified or is incomplete")
    import joblib
    raw = joblib.load(d / STATE_FILE)
    unlisted = sorted(set(_torch_refs(raw)) - set(files))
    if unlisted:
        raise PrecompError(f"{d}: the state refers to {unlisted}, which the manifest does not "
                           "hash; refusing to load unverified weights")
    model = DeviationSurrogate.from_state(_restore_torch(raw, d))
    if model.features != cfg or model.feature_names != list(doc["feature_names"]):
        raise PrecompError(f"{d}: the state's features differ from the manifest's")
    for key, value in (("model_class", model.model_class), ("kind", model.kind),
                       ("domain", model.domain), ("data_source", model.data_source)):
        if doc.get(key) != value:
            raise PrecompError(f"{d}: the state's {key} {value!r} differs from the manifest's "
                               f"{doc.get(key)!r}")
    model.manifest = doc  # type: ignore[attr-defined]
    return model


__all__ = ["save_model", "load_model", "read_manifest", "check_schema", "build_manifest",
           "package_versions", "git_sha", "MODEL_FORMAT"]
