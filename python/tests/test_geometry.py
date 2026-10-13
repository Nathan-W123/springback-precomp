"""HeightMap and STL input/output against exact answers."""

import numpy as np
import pytest

from precomp import PrecompError
from precomp.geometry import Grid, HeightMap, zeros


def test_grid_centered_layout():
    g = Grid.centered(0.1, 1e-3)
    assert (g.nx, g.ny) == (101, 101)
    assert g.extent == pytest.approx((-0.05, 0.05, -0.05, 0.05), abs=1e-15)
    X, Y = g.mesh()
    assert X.shape == g.shape and X[0, 1] > X[0, 0] and Y[1, 0] > Y[0, 0]
    assert g.matches(Grid.from_dict(g.to_dict()))
    assert not g.matches(Grid.centered(0.1, 0.5e-3))


def test_bilinear_interpolation_is_exact_for_bilinear_fields(rng):
    g = Grid(-0.02, -0.03, 41, 61, 1e-3)
    X, Y = g.mesh()
    f = lambda x, y: 0.3 - 0.2 * x + 0.7 * y + 5.0 * x * y  # noqa: E731
    hm = HeightMap(g, f(X, Y))
    x = rng.uniform(g.x0, g.xmax, 500)
    y = rng.uniform(g.y0, g.ymax, 500)
    assert np.abs(hm.interpolate(x, y) - f(x, y)).max() < 1e-15
    assert np.isnan(hm.interpolate(np.array([1.0]), np.array([0.0]))[0])


def test_masked_interpolation_refuses_invalid_stencils():
    g = Grid.centered(0.01, 1e-3)
    mask = np.ones(g.shape, bool)
    mask[5, 5] = False
    hm = HeightMap(g, np.zeros(g.shape), mask)
    x, y = g.x[5], g.y[5]
    assert np.isnan(hm.interpolate(np.array([x + 0.5e-3]), np.array([y]))[0])
    assert hm.interpolate(np.array([x + 1.5e-3]), np.array([y]))[0] == 0.0
    assert hm.interpolate(np.array([x + 0.5e-3]), np.array([y]), masked=False)[0] == 0.0


def test_plane_gradient_normals_and_wall_angle():
    g = Grid.centered(0.05, 1e-3)
    X, Y = g.mesh()
    p, q = 0.4, -0.3
    hm = HeightMap(g, p * X + q * Y)
    gx, gy = hm.gradient()
    assert np.allclose(gx, p, atol=1e-13) and np.allclose(gy, q, atol=1e-13)
    n = hm.normals()
    assert np.allclose(np.linalg.norm(n, axis=-1), 1.0)
    assert np.allclose(n[..., :], np.array([-p, -q, 1.0]) / np.sqrt(1 + p * p + q * q))
    assert np.allclose(hm.wall_angle(), np.arctan(np.hypot(p, q)))
    H, K = hm.curvature()
    assert np.abs(H).max() < 1e-10 and np.abs(K).max() < 1e-10


def test_curvature_of_a_spherical_bowl():
    R = 0.05
    g = Grid.centered(0.04, 2e-4)
    X, Y = g.mesh()
    hm = HeightMap(g, R - np.sqrt(R * R - X * X - Y * Y))   # bottom of a sphere, opening up
    H, K = hm.curvature()
    c = (g.ny // 2, g.nx // 2)
    assert H[c] == pytest.approx(1 / R, rel=1e-4)
    assert K[c] == pytest.approx(1 / R ** 2, rel=1e-4)
    inner = (X ** 2 + Y ** 2) < 0.015 ** 2
    assert np.allclose(H[inner], 1 / R, rtol=1e-3) and np.allclose(K[inner], 1 / R ** 2, rtol=2e-3)


def test_smoothed_derivatives_are_exact_for_quadratics_far_from_zero():
    """Gaussian derivatives at a length scale must not see the height itself:
    a quadratic 15 mm below the sheet plane has exact smoothed derivatives
    (scipy's order-2 Gaussian filter does not sum to zero and fails this)."""
    g = Grid.centered(0.04, 5e-4)
    X, Y = g.mesh()
    hm = HeightMap(g, -0.015 + 0.3 * X + 3.0 * X ** 2 - 2.0 * Y ** 2 + 0.5 * X * Y)
    inner = (np.abs(X) < 0.012) & (np.abs(Y) < 0.012)          # away from the border
    gx, gy = hm.gradient(sigma=1e-3)
    assert np.allclose(gx[inner], (0.3 + 6.0 * X + 0.5 * Y)[inner], rtol=0, atol=1e-9)
    assert np.allclose(gy[inner], (-4.0 * Y + 0.5 * X)[inner], rtol=0, atol=1e-9)
    H, K = hm.curvature(sigma=1e-3)
    p, q = 0.3 + 6.0 * X + 0.5 * Y, -4.0 * Y + 0.5 * X
    w = 1 + p * p + q * q
    K_exact = (6.0 * -4.0 - 0.25) / w ** 2
    assert np.allclose(K[inner], K_exact[inner], rtol=1e-9)


def test_npz_round_trip_keeps_mask_and_metadata(tmp_path):
    g = Grid(0.0, 0.0, 7, 5, 2e-3)
    z = -np.arange(35.0).reshape(5, 7) * 1e-4
    mask = np.ones((5, 7), bool)
    mask[0, 0] = False
    hm = HeightMap(g, z, mask, {"part": {"family": "x"}, "note": "é"})
    back = HeightMap.load(hm.save(tmp_path / "m.npz"))
    assert back.grid == g
    assert np.array_equal(back.z, z) and np.array_equal(back.mask, mask)
    assert back.metadata == hm.metadata


def test_from_points_is_exact_for_a_plane_and_masks_outside(rng):
    g = Grid.centered(0.04, 1e-3)
    xy = rng.uniform(-0.015, 0.015, (4000, 2))
    pts = np.column_stack([xy, 0.1 * xy[:, 0] - 0.2 * xy[:, 1] - 0.003])
    hm = HeightMap.from_points(pts, g)
    X, Y = g.mesh()
    exact = 0.1 * X - 0.2 * Y - 0.003
    assert np.abs(hm.z - exact)[hm.mask].max() < 1e-15
    assert not hm.mask[np.abs(X) > 0.016].any()
    holes = pts[np.hypot(pts[:, 0], pts[:, 1]) > 0.005]
    gapped = HeightMap.from_points(holes, g, max_gap=1.5e-3)
    assert not gapped.mask[g.ny // 2, g.nx // 2]


def test_resample_matches_and_marks_outside(small_cone):
    fine = small_cone.heightmap(Grid.centered(0.08, 0.5e-3))
    coarse_grid = Grid.centered(0.1, 1e-3)
    out = fine.resample(coarse_grid)
    X, Y = coarse_grid.mesh()
    inside = (np.abs(X) <= 0.04 + 1e-12) & (np.abs(Y) <= 0.04 + 1e-12)
    assert np.array_equal(out.mask, inside)
    assert np.abs(out.z - small_cone.height(X, Y))[inside].max() < 5e-6


def test_invalid_heightmaps_are_refused():
    g = Grid.centered(0.01, 1e-3)
    with pytest.raises(ValueError):
        HeightMap(g, np.zeros((3, 3)))
    z = np.zeros(g.shape)
    z[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        HeightMap(g, z)
    with pytest.raises(PrecompError):
        HeightMap.load("does-not-exist.npz")
    assert zeros(g).depth == 0.0
