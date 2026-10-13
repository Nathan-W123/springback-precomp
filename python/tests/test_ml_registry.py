"""precomp.ml.registry: bundles round-trip and refuse what does not match."""

import json

import numpy as np
import pytest

from conftest import ML_CREATED_AT
from precomp._util import PrecompError
from precomp.ml import load_model, read_manifest, save_model
from precomp.ml.features import FEATURE_SCHEMA_VERSION


def test_a_bundle_round_trips_and_says_where_it_came_from(tmp_path, mlp_surrogate, proxy_split):
    metrics = {"held_out": {"err_rms_mean_m": 1e-4, "data_source": "proxy - not physics"}}
    d = save_model(mlp_surrogate, tmp_path / "m",
                   {"created_at": ML_CREATED_AT, "metrics": metrics, "notes": "test"})
    m = load_model(d)
    s = proxy_split["test"][0]
    a, b = mlp_surrogate.predict_deviation(s.commanded, s.setup), m.predict_deviation(
        s.commanded, s.setup)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    lo1, hi1 = mlp_surrogate.predict_interval(s.commanded, s.setup, 0.9)
    lo2, hi2 = m.predict_interval(s.commanded, s.setup, 0.9)
    assert np.array_equal(lo1, lo2) and np.array_equal(hi1, hi2)
    assert m.assess(s.commanded, s.setup)["part_score"] == pytest.approx(
        mlp_surrogate.assess(s.commanded, s.setup)["part_score"], rel=1e-12)
    doc = read_manifest(d)
    assert doc["model_class"] == "MLPEnsemble" and doc["created_at"] == ML_CREATED_AT
    assert doc["data_source"] == "proxy - not physics"
    assert doc["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert doc["training"]["n_samples"] == len(proxy_split["train"])
    assert len(doc["training"]["sample_ids_sha256"]) == 64
    assert doc["conformal"]["quantiles"]["0.9"] > 0 and doc["ood"]["thresholds"]
    assert doc["package_versions"]["sklearn"] and "model.joblib" in doc["files"]
    assert doc["metrics"] == metrics and doc["notes"] == "test"
    with pytest.raises(PrecompError, match="not empty"):
        save_model(mlp_surrogate, d, {"created_at": ML_CREATED_AT})
    with pytest.raises(ValueError, match="created_at"):
        save_model(mlp_surrogate, tmp_path / "n", {})
    with pytest.raises(ValueError, match="data_source"):
        save_model(mlp_surrogate, tmp_path / "n", {"created_at": ML_CREATED_AT,
                                                     "metrics": {"x": {"rms": 1.0}}})


def _tamper(path, **changes):
    doc = json.loads(path.read_text())
    doc.update(changes)
    path.write_text(json.dumps(doc))


def test_loading_refuses_another_schema_or_modified_files(tmp_path, mlp_surrogate):
    d = save_model(mlp_surrogate, tmp_path / "m", {"created_at": ML_CREATED_AT})
    man = d / "manifest.json"
    original = man.read_text()
    _tamper(man, feature_schema_version="0")
    with pytest.raises(PrecompError, match="feature schema version '0'"):
        load_model(d)
    man.write_text(original)
    doc = json.loads(original)
    _tamper(man, feature_names=doc["feature_names"][::-1])
    with pytest.raises(PrecompError, match="feature_names"):
        load_model(d)
    man.write_text(original)
    cfg = dict(doc["feature_config"], rings=3)
    _tamper(man, feature_config=cfg)
    with pytest.raises(PrecompError, match="schema hash"):
        load_model(d)
    man.write_text(original)
    with open(d / "model.joblib", "ab") as handle:
        handle.write(b"\0")
    with pytest.raises(PrecompError, match="SHA-256"):
        load_model(d)


def test_torch_weights_are_stored_as_state_dict_files(tmp_path, proxy_split, ml_features):
    pytest.importorskip("torch")
    from precomp.ml import MLPEnsemble, train_surrogate

    sur = train_surrogate(MLPEnsemble(2, hidden=(16,), epochs=3), proxy_split["train"][:12],
                          features=ml_features, points_per_sample=100, envelope=False)
    d = save_model(sur, tmp_path / "mlp", {"created_at": ML_CREATED_AT})
    pts = sorted(p.name for p in (d / "torch").iterdir())
    assert pts == ["model__member0.pt", "model__member1.pt"]
    assert set(read_manifest(d)["files"]) == {"model.joblib", "torch/model__member0.pt",
                                              "torch/model__member1.pt"}
    s = proxy_split["test"][0]
    assert np.array_equal(load_model(d).predict_deviation(s.commanded, s.setup)[0],
                          sur.predict_deviation(s.commanded, s.setup)[0])
    # weights the manifest does not hash are refused
    man = d / "manifest.json"
    doc = json.loads(man.read_text())
    doc["files"].pop("torch/model__member1.pt")
    man.write_text(json.dumps(doc))
    with pytest.raises(PrecompError, match="does not hash"):
        load_model(d)


def test_a_neural_bundle_without_torch_says_what_is_missing(tmp_path, proxy_split,
                                                             ml_features, monkeypatch):
    pytest.importorskip("torch")
    import sys

    from precomp.ml import MLPEnsemble, train_surrogate

    sur = train_surrogate(MLPEnsemble(1, hidden=(8,), epochs=2), proxy_split["train"][:6],
                          features=ml_features, points_per_sample=50, envelope=False)
    d = save_model(sur, tmp_path / "mlp", {"created_at": ML_CREATED_AT})
    monkeypatch.setitem(sys.modules, "torch", None)          # import torch -> ImportError
    with pytest.raises(PrecompError, match=r"needs torch: pip install -e '\.\[torch\]'"):
        load_model(d)


def test_loading_refuses_unverifiable_or_relabelled_bundles(tmp_path, mlp_surrogate):
    d = save_model(mlp_surrogate, tmp_path / "m", {"created_at": ML_CREATED_AT})
    man = d / "manifest.json"
    original = man.read_text()
    doc = json.loads(original)
    assert doc["target"] == "dz"
    no_files = {k: v for k, v in doc.items() if k != "files"}
    man.write_text(json.dumps(no_files))
    with pytest.raises(PrecompError, match="SHA-256 of model.joblib"):
        load_model(d)
    for key, value in (("data_source", "SparLab simulation"), ("model_class", "TransferModel"),
                       ("domain", "all")):
        man.write_text(original)
        _tamper(man, **{key: value})
        with pytest.raises(PrecompError, match=key):
            load_model(d)
    man.write_text(original)
    assert load_model(d).data_source == "proxy - not physics"


def test_saving_checks_everything_before_touching_a_file(tmp_path, mlp_surrogate):
    d = save_model(mlp_surrogate, tmp_path / "m", {"created_at": ML_CREATED_AT})
    files = lambda: {str(p.relative_to(d)): p.read_bytes()  # noqa: E731
                     for p in d.rglob("*") if p.is_file()}
    before = files()
    with pytest.raises(ValueError, match="data_source"):
        save_model(mlp_surrogate, d, {"created_at": ML_CREATED_AT,
                                      "metrics": {"x": {"rms": 1.0}}}, overwrite=True)
    assert files() == before                                         # the old bundle stands
    # a metric that could not be computed is written as null, not NaN
    d2 = save_model(mlp_surrogate, tmp_path / "n", {"created_at": ML_CREATED_AT, "metrics": {
        "held_out": {"coverage_mean": float("nan"), "data_source": "proxy - not physics"}}})
    assert read_manifest(d2)["metrics"]["held_out"]["coverage_mean"] is None
