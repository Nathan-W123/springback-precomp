"""precomp.ml.dataset: samples on disk, feature tables and grouped splits."""

import json

import numpy as np
import pandas as pd
import pytest

from conftest import ML_CREATED_AT
from precomp._util import PrecompError
from precomp.fea import FormingSetup
from precomp.geometry import Grid, TruncatedCone
from precomp.materials import get_material
from precomp.ml.dataset import (Dataset, Sample, build_table, family_split, grouped_split,
                                index_frame, source_label)
from precomp.ml.features import FeatureConfig
from precomp.toolpath import spiral_toolpath


def _sample(sid="cone-a", source="proxy", **kw):
    setup = FormingSetup(get_material("AA5754-O"))
    part = TruncatedCone(0.04, 45.0, 0.02, 0.004, 0.004)
    cmd = part.heightmap(Grid.centered(setup.meshed_blank_size, 4e-3))
    formed = cmd.with_z(0.95 * cmd.z, metadata={"source": source})
    prov = {"created_at": ML_CREATED_AT, **kw.pop("provenance", {})}
    return Sample(sid, cmd, formed, setup.to_dict(), source, "test-v1", part=part.to_dict(),
                  target=cmd, provenance=prov, **kw)


def test_dataset_round_trip_keeps_every_array_and_record(tmp_path):
    ds = Dataset.create(tmp_path / "d", created_at=ML_CREATED_AT, description="t")
    s = _sample(toolpath=None)
    s.toolpath = spiral_toolpath(s.commanded, 0.005, 2e-3, 3e-3)
    ds.append(s)
    with pytest.raises(PrecompError, match="already"):
        ds.append(s)
    r = Dataset.open(tmp_path / "d").load("cone-a")
    assert np.array_equal(r.commanded.z, s.commanded.z) and np.array_equal(r.formed.z, s.formed.z)
    assert np.array_equal(r.commanded.mask, s.commanded.mask)
    assert np.array_equal(r.target.z, s.target.z)
    assert np.array_equal(r.toolpath.points, s.toolpath.points)
    assert r.setup == json.loads(json.dumps(s.setup)) and r.part == s.part
    assert r.provenance["created_at"] == ML_CREATED_AT and r.family == "truncated_cone"
    np.testing.assert_allclose(r.dz, -0.05 * s.commanded.z)
    doc = json.loads((tmp_path / "d" / "samples" / "cone-a.json").read_text())
    assert doc["data_source"] == "proxy - not physics"
    # overwrite appends a new index line; the latest wins
    s2 = _sample(kind="perturbed")
    ds.append(s2, overwrite=True)
    assert len(ds) == 1 and ds.index().loc[0, "kind"] == "perturbed"
    ds.append(_sample("cone-b", part_id="P7"))
    assert ds.filter(kind="perturbed") == ["cone-a"]
    assert ds.filter(part_id=["P7"]) == ["cone-b"]
    with pytest.raises(KeyError):
        ds.filter(colour="red")
    ds.record_failure({"sample_id": "cone-c", "reason": "exit code 3"})
    ds.record_failure({"sample_id": "cone-b", "reason": "old failure, since succeeded"})
    f = ds.failures()
    assert list(f["sample_id"]) == ["cone-c"] and f.loc[0, "reason"] == "exit code 3"


def test_samples_refuse_missing_provenance_and_bad_labels():
    with pytest.raises(ValueError, match="created_at"):
        Sample("x", _sample().commanded, _sample().formed, _sample().setup, "proxy", "v1")
    with pytest.raises(ValueError, match="sparlab_version"):
        _sample(source="sim")
    with pytest.raises(ValueError, match="source"):
        _sample(source="guess")
    with pytest.raises(ValueError, match="letters"):
        _sample("bad/id")
    ok = _sample(source="sim", provenance={"sparlab_version": "1.0", "deck_hash": "ab"})
    assert ok.data_source == "SparLab simulation"
    assert source_label(["proxy", "sim"]) == "mixed: proxy - not physics + SparLab simulation"


def test_tables_share_the_points_among_regions_and_are_reproducible():
    s = _sample()
    cfg = FeatureConfig(time_source="depth")
    t = build_table([s], cfg, points_per_sample=300, rng=4)
    counts = pd.Series(t.region).value_counts()
    assert set(counts.index) == {1, 2, 3} and counts.sum() == 300
    # equal shares of 100; a region with fewer nodes than its share (the base,
    # then the wall, on this coarse grid) gives all it has and the rest goes
    # to the others
    avail = pd.Series(build_table([s], cfg, points_per_sample=None).region).value_counts()
    small = sorted(avail.index, key=lambda r: avail[r])
    assert avail[small[0]] < 100 and counts[small[0]] == avail[small[0]]
    left = 300 - avail[small[0]]
    assert counts[small[1]] == min(avail[small[1]], left // 2)
    assert counts[small[2]] == 300 - counts[small[0]] - counts[small[1]]
    assert counts[small[2]] < avail[small[2]]
    X, y, groups = t                                             # unpacks as the spec says
    assert X.shape == (300, len(cfg.names)) and set(groups) == {"cone-a"}
    t2 = build_table([s], cfg, points_per_sample=300, rng=4)
    assert np.array_equal(t.node, t2.node) and np.array_equal(t.X, t2.X)
    uni = build_table([s], cfg, points_per_sample=300, rng=4, stratify=False)
    ucounts = pd.Series(uni.region).value_counts()
    assert ucounts[small[0]] < counts[small[0]]                  # a uniform draw thins it
    full = build_table([s], cfg, points_per_sample=None, mask="all")
    assert len(full) == s.grid.nx * s.grid.ny and t.data_source == "proxy - not physics"
    np.testing.assert_allclose(y, s.dz.ravel()[t.node])
    # a sample that cannot be featurised is named
    coarse = _sample("cone-coarse")
    st = FormingSetup.from_dict(coarse.setup).replace(tool_radius=3e-3)
    coarse.setup = st.to_dict()
    with pytest.raises(PrecompError, match="sample cone-coarse: grid spacing"):
        build_table([s, coarse], cfg, points_per_sample=50)


def test_grouped_splits_never_put_a_part_on_both_sides(proxy_samples):
    idx = index_frame(proxy_samples)
    col = {"part": "part_id", "sample": "sample_id", "family": "family"}
    key = lambda by: dict(zip(idx["sample_id"], idx[col[by]]))  # noqa: E731
    for by in ("part", "sample", "family"):
        tr, te = grouped_split(idx, test_fraction=0.3, seed=5, by=by)
        assert set(tr).isdisjoint(te) and len(tr) + len(te) == len(idx)
        k = key(by)
        assert {k[i] for i in tr}.isdisjoint({k[i] for i in te})
    rest, fam = family_split(idx, "dome")
    assert set(idx.set_index("sample_id").loc[fam, "family"]) == {"dome"}
    assert "dome" not in set(idx.set_index("sample_id").loc[rest, "family"])
