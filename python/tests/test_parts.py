"""Part families: exact geometry, formability, C1 continuity, JSON round trips."""

import json
import math

import numpy as np
import pytest

from precomp.geometry import (MAX_WALL_ANGLE_DEG, Dome, EllipticCone, FilletedProfile, Grid,
                              Pyramid, TruncatedCone, ellipse_distance, families, part_from_dict)


def test_cone_wall_slope_equals_the_wall_angle():
    cone = TruncatedCone(top_radius=0.05, wall_angle_deg=50.0, depth=0.02,
                         top_fillet=0.005, bottom_fillet=0.005)
    prof = cone.profile
    g = Grid.centered(0.12, 2.5e-4)
    hm = cone.heightmap(g)
    X, Y = g.mesh()
    r = np.hypot(X, Y)
    w = cone.top_radius - r
    straight = (w > prof.tangent_lengths[0] + 1e-3) & \
        (w < prof.w[1] - prof.tangent_lengths[1] - 1e-3)
    angle = np.degrees(hm.wall_angle())[straight]
    assert straight.sum() > 1000
    assert np.abs(angle - 50.0).max() < 5e-3                     # central differences, h = 0.25 mm
    # the profile's own slope is exact
    assert np.allclose(np.degrees(np.arctan(np.abs(prof.slope(w[straight])))), 50.0,
                       atol=1e-12)
    assert hm.depth == pytest.approx(0.02, abs=1e-12)
    assert np.all(hm.z[r > cone.footprint_radius()] == 0.0)


def test_filleted_profile_is_c1():
    prof = FilletedProfile([(0.0, 0.0), (0.01, -0.012), (0.02, -0.012), (0.03, -0.02)],
                           [0.004, 0.003, 0.002, 0.005])
    w = np.linspace(-0.01, 0.05, 600001)
    z = prof(w)
    s = prof.slope(w)
    dw = w[1] - w[0]
    assert np.abs(np.diff(z)).max() <= np.abs(s).max() * dw * (1 + 1e-9)   # continuous
    assert np.abs(np.diff(s)).max() < 1e-3                                  # slope continuous
    fd = np.gradient(z, dw)
    assert np.abs(fd - s)[1:-1].max() < 1e-4                               # slope is dz/dw
    # the arcs have the prescribed radii: curvature 1/r inside each fillet
    for arc in prof.arcs:
        a, b, cw, cz, R, up = arc
        mid = 0.5 * (a + b)
        pts = np.array([a + 0.25 * (b - a), mid, a + 0.75 * (b - a)])
        assert np.allclose(np.hypot(pts - cw, prof(pts) - cz), R, rtol=1e-12)


def test_overlapping_fillets_and_bad_parameters_are_refused():
    with pytest.raises(ValueError, match="overlap"):
        TruncatedCone(0.05, 60.0, 0.0007, 0.005, 0.005)
    with pytest.raises(ValueError, match="wall_angle_deg"):
        TruncatedCone(0.05, 70.0, 0.02)
    with pytest.raises(ValueError, match="floor radius"):
        TruncatedCone(0.01, 30.0, 0.02)
    with pytest.raises(ValueError, match="crease"):
        Pyramid(0.05, 0.05, 45.0, 0.02, corner_radius=0.01)
    with pytest.raises(ValueError, match="curvature"):
        EllipticCone(0.07, 0.03, 40.0, 0.02)
    with pytest.raises(ValueError, match="exceeds"):
        Dome(0.03, 0.025, 0.005)


def test_ellipse_distance_is_exact():
    a, b = 0.06, 0.035
    t = np.linspace(0, 2 * np.pi, 400001)
    curve = np.column_stack([a * np.cos(t), b * np.sin(t)])
    rng = np.random.default_rng(1)
    pts = rng.uniform(-0.08, 0.08, (200, 2))
    pts[:4, 1] = 0.0
    pts[4:8, 0] = 0.0
    d = ellipse_distance(pts[:, 0], pts[:, 1], a, b)
    brute = np.array([np.hypot(*(curve - p).T).min() for p in pts])
    inside = (pts[:, 0] / a) ** 2 + (pts[:, 1] / b) ** 2 < 1
    assert np.all((d > 0) == inside)
    assert np.abs(np.abs(d) - brute).max() < 1e-9     # limited by the brute-force sampling
    assert ellipse_distance(np.array([0.0]), np.array([0.0]), a, b)[0] == pytest.approx(b)


def test_elliptic_cone_has_a_constant_wall_angle():
    part = EllipticCone(0.06, 0.045, 55.0, 0.012, 0.003, 0.003)
    g = Grid.centered(0.14, 2.5e-4)
    hm = part.heightmap(g)
    X, Y = g.mesh()
    w = ellipse_distance(X, Y, 0.06, 0.045)
    prof = part.profile
    straight = (w > prof.tangent_lengths[0] + 1e-3) & (w < prof.w[1] - prof.tangent_lengths[1]
                                                       - 1e-3)
    assert np.abs(np.degrees(hm.wall_angle()[straight]) - 55.0).max() < 0.02


def test_dome_depth_opening_and_steepest_wall():
    d = Dome(opening_radius=0.05, depth=0.02, rim_fillet=0.008)
    assert d.height(np.array([0.0]), np.array([0.0]))[0] == pytest.approx(-0.02, abs=1e-15)
    assert d.height(np.array([0.05]), np.array([0.0]))[0] == 0.0
    expected = math.degrees(math.asin(2 * 0.05 * 0.02 / (0.05 ** 2 + 0.02 ** 2)))
    assert d.max_wall_angle_deg() == pytest.approx(expected)
    assert Part_numeric_angle(d) == pytest.approx(expected, abs=0.02)
    e = Dome(0.05, 0.015, 0.008, aspect=0.8)
    assert Part_numeric_angle(e) == pytest.approx(e.max_wall_angle_deg(), abs=0.02)


def Part_numeric_angle(part):
    from precomp.geometry import Part
    return Part.max_wall_angle_deg(part)


@pytest.mark.parametrize("family", sorted(families()))
def test_sampled_parts_are_valid_formable_and_serialisable(family):
    cls = families()[family]
    rng = np.random.default_rng(7)
    g = Grid.centered(0.18, 5e-4)
    for _ in range(2):
        part = cls.sample(rng)
        hm = part.heightmap(g)
        assert np.all(hm.z <= 0.0)
        X, Y = g.mesh()
        outside = np.hypot(X, Y) > part.footprint_radius() + 1e-9
        assert np.all(hm.z[outside] == 0.0)                        # flange at z = 0
        assert np.all(hm.z[0] == 0.0) and np.all(hm.z[:, -1] == 0.0)
        assert np.degrees(hm.wall_angle().max()) <= MAX_WALL_ANGLE_DEG + 0.3
        assert part.max_depth() == pytest.approx(hm.depth, rel=2e-2)
        doc = json.loads(json.dumps(part.to_dict()))
        assert part_from_dict(doc) == part
        for key, (lo, hi) in cls.bounds.items():
            assert lo <= getattr(part, key) <= hi


def test_sampling_is_deterministic_by_seed():
    a = [Pyramid.sample(np.random.default_rng(5)) for _ in range(2)]
    assert a[0] == a[1]
    assert Pyramid.sample(np.random.default_rng(5)) != Pyramid.sample(np.random.default_rng(6))


def test_heightmap_refuses_a_grid_that_cuts_the_part(small_cone):
    with pytest.raises(ValueError, match="footprint"):
        small_cone.heightmap(Grid.centered(0.06, 1e-3))
    with pytest.raises(ValueError, match="unknown part family"):
        part_from_dict({"family": "sphere"})
    with pytest.raises(ValueError, match="unknown parameters"):
        part_from_dict({"family": "truncated_cone", "top_radius": 0.05, "wall_angle_deg": 45,
                        "depth": 0.01, "colour": "red"})
