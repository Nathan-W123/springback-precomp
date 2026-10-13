"""precomp.ml.features: the versioned point-feature schema."""

import time

import numpy as np
import pytest

from precomp.fea import FormingSetup
from precomp.geometry import Grid, HeightMap, Pyramid, TruncatedCone, families
from precomp.materials import get_material
from precomp.ml.features import (FEATURE_SCHEMA_VERSION, KINDS, REGION_NAMES, FeatureConfig,
                                 feature_maps, global_features, region_labels)
from precomp.metrology import part_mask

SETUP = FormingSetup(get_material("AA5754-O"))
DEPTH_TIME = FeatureConfig(time_source="depth")


def point_features(commanded, setup, toolpath=None, *, config):
    """Feature matrix X (every node, row-major) and the names, as the
    training table gathers them."""
    fm = feature_maps(commanded, setup, toolpath, config)
    return fm.gather(None, commanded.grid), list(fm.names)


def test_every_feature_is_finite_documented_and_in_schema_order(rng):
    cfg = DEPTH_TIME
    specs = cfg.specs()
    assert [s.name for s in specs] == cfg.names and len(set(cfg.names)) == len(cfg.names)
    assert all(s.unit and s.description and s.kind in KINDS for s in specs)
    grid = Grid.centered(SETUP.meshed_blank_size, 3e-3)
    for name, cls in sorted(families().items()):
        part = cls.sample(rng)
        hm = part.heightmap(grid)
        X, names = point_features(hm, SETUP, config=cfg)
        assert names == cfg.names
        assert X.shape == (grid.nx * grid.ny, len(names))
        assert np.all(np.isfinite(X)), name
        d, dnames = global_features(hm, SETUP, cfg)
        assert dnames == cfg.names_of_kind("global", "process", "material")
        # a part-level descriptor has the same value at every node
        for j, n in enumerate(names):
            if n in dnames:
                assert np.all(X[:, j] == d[dnames.index(n)])


def test_schema_hash_tracks_the_configuration_and_version():
    a, b = FeatureConfig(), FeatureConfig(time_source="depth")
    assert FEATURE_SCHEMA_VERSION == "2"
    assert "flow_stress_20_over_E" in a.names and "yield_over_E" not in a.names
    assert a.schema_hash() == FeatureConfig().schema_hash()
    assert a.schema_hash() != b.schema_hash()
    assert FeatureConfig.from_dict(b.to_dict()) == b
    with pytest.raises(ValueError, match="time_source"):
        FeatureConfig(time_source="clock")


def test_local_features_do_not_change_when_the_part_moves_in_plane():
    """Moving the whole part by whole grid spacings on a flat blank changes
    only the placement features (and none of the local ones) at the
    corresponding nodes."""
    part = Pyramid(0.045, 0.035, 50.0, 0.02, 0.03, 0.004, 0.004)
    h = 2e-3
    grid = Grid.centered(0.2, h)
    X, Y = grid.mesh()
    di, dj = 7, -4                                            # nodes in x and y
    base = HeightMap(grid, part.height(X, Y))
    moved = HeightMap(grid, part.height(X - di * h, Y - dj * h))
    cfg = DEPTH_TIME
    fa = feature_maps(base, SETUP, config=cfg)
    fb = feature_maps(moved, SETUP, config=cfg)
    local = [i for i, s in enumerate(cfg.specs()) if s.kind in ("local", "global", "process",
                                                                   "material")]
    # compare away from the grid border (the widest kernel reaches 4 sigma)
    m = 30
    a = fa.values[m:-m, m:-m][:, :, local]
    b = fb.values[m + dj:grid.ny - m + dj, m + di:grid.nx - m + di][:, :, local]
    scale = np.maximum(np.abs(a).max(axis=(0, 1)), 1e-30)
    assert np.max(np.abs(a - b) / scale) < 1e-9
    placement = [i for i, s in enumerate(cfg.specs()) if s.kind == "placement"]
    pa = fa.values[m:-m, m:-m][:, :, placement]
    pb = fb.values[m + dj:grid.ny - m + dj, m + di:grid.nx - m + di][:, :, placement]
    assert np.abs(pa - pb).max() > 1e-3                        # these do move
    assert np.array_equal(fa.region[m:-m, m:-m],
                          fb.region[m + dj:grid.ny - m + dj, m + di:grid.nx - m + di])


def test_time_frac_of_a_spiral_path_moves_with_the_part():
    """With the tool path as time source, time_frac is rebuilt from the path of
    the moved part: the spiral gives the same values."""
    part = Pyramid(0.045, 0.035, 50.0, 0.02, 0.03, 0.004, 0.004)
    h = 2e-3
    grid = Grid.centered(0.2, h)
    X, Y = grid.mesh()
    di, dj = 7, -4
    base = HeightMap(grid, part.height(X, Y))
    moved = HeightMap(grid, part.height(X - di * h, Y - dj * h))
    cfg = FeatureConfig()
    j = cfg.names.index("time_frac")
    m = 30
    a = feature_maps(base, SETUP, config=cfg).values[m:-m, m:-m, j]
    b = feature_maps(moved, SETUP, config=cfg).values[m + dj:grid.ny - m + dj,
                                                      m + di:grid.nx - m + di, j]
    assert np.abs(a - b).max() <= 1e-9


def test_ring_extrema_equal_the_footprint_filters_exactly(rng):
    """The run-decomposed ring min/max (linear in the ring radius) equals
    scipy's footprint filters bit for bit, borders included."""
    from scipy import ndimage

    from precomp.ml.features import _footprint_extrema, _ring_footprints

    for trial in range(4):
        z = rng.normal(size=tuple(rng.integers(20, 70, 2)))
        r = rng.uniform(4.0, 11.0)
        fps = _ring_footprints([r * k / 4 for k in range(1, 5)])
        for fp, (lo, hi) in zip(fps, _footprint_extrema(z, fps)):
            assert np.array_equal(lo, ndimage.minimum_filter(z, footprint=fp, mode="nearest"))
            assert np.array_equal(hi, ndimage.maximum_filter(z, footprint=fp, mode="nearest"))
    # a fine grid and a large tool: 32 cells to the last ring (16.8 s before)
    part = TruncatedCone(0.035, 50.0, 0.02, 0.005, 0.005)
    hm = part.heightmap(Grid(-0.05, -0.05, 200, 200, 5e-4))
    t = time.perf_counter()
    X, _ = point_features(hm, SETUP.replace(tool_radius=8e-3), config=DEPTH_TIME)
    assert np.all(np.isfinite(X)) and time.perf_counter() - t < 4.0


def test_a_200_by_200_grid_featurises_within_two_seconds():
    part = TruncatedCone(0.05, 50.0, 0.025, 0.005, 0.005)
    grid = Grid(-0.1, -0.1, 200, 200, 0.2 / 199)
    hm = part.heightmap(grid)
    from precomp.fea import make_toolpath
    path = make_toolpath(SETUP, hm)
    best = np.inf
    for _ in range(2):
        t = time.perf_counter()
        X, _ = point_features(hm, SETUP, path, config=FeatureConfig())
        best = min(best, time.perf_counter() - t)
    assert X.shape[0] == 40000 and np.all(np.isfinite(X))
    assert best < 2.0, best


def test_time_frac_follows_the_tool_path():
    """The pseudo-time where the tool passes nearest rises with depth on a
    spiral path: ~0 at the rim, ~1 on the floor."""
    part = TruncatedCone(0.04, 45.0, 0.02, 0.004, 0.004)
    hm = part.heightmap(Grid.centered(0.2, 2e-3))
    X, names = point_features(hm, SETUP, config=FeatureConfig())
    t = X[:, names.index("time_frac")]
    d = X[:, names.index("depth_frac")]
    sel = part_mask(hm).ravel()
    assert t.min() >= 0 and t.max() <= 1
    assert np.corrcoef(t[sel], d[sel])[0, 1] > 0.95
    assert np.median(t[sel & (d > 0.99)]) > 0.9 and np.median(t[sel & (d < 0.05)]) < 0.15


def test_point_selections_agree_and_regions_cover_the_part():
    part = TruncatedCone(0.04, 45.0, 0.02, 0.004, 0.004)
    hm = part.heightmap(Grid.centered(0.2, 2e-3))
    fm = feature_maps(hm, SETUP, config=DEPTH_TIME)
    mask = part_mask(hm)
    rows = fm.gather(mask)
    idx = np.flatnonzero(mask.ravel())
    assert np.array_equal(rows, fm.gather(idx))
    Xg, Yg = hm.grid.mesh()
    xy = np.column_stack([Xg.ravel()[idx[:50]], Yg.ravel()[idx[:50]]])
    np.testing.assert_allclose(fm.gather(xy, hm.grid), rows[:50], rtol=1e-12, atol=1e-15)
    reg = region_labels(hm, SETUP, DEPTH_TIME)
    assert np.array_equal(reg, fm.region)
    assert set(np.unique(reg[mask])) == {1, 2, 3} and np.all(reg[~mask] == 0)
    assert REGION_NAMES[1] == "rim"
    with pytest.raises(ValueError, match="no part"):
        feature_maps(HeightMap(hm.grid, np.zeros(hm.grid.shape)), SETUP, config=DEPTH_TIME)
    with pytest.raises(ValueError, match="coarser"):
        feature_maps(part.heightmap(Grid.centered(0.2, 8e-3)), SETUP, config=DEPTH_TIME)
