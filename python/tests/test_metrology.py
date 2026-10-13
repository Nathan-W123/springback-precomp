"""Metrology: robust ICP, signed normal deviation, point-cloud readers, metrics."""


import numpy as np
import pytest

from precomp.geometry import EllipticCone, Grid, HeightMap, TruncatedCone
from precomp.metrology import (align, flange_mask, metrics, rotation_angle_deg,
                               rotation_matrix, signed_deviation, vertical_deviation, wall_mask)


def _scan(part, rng, n=20000, half=0.06):
    xy = rng.uniform(-half, half, (n, 2))
    return np.column_stack([xy, part.height(xy[:, 0], xy[:, 1])])


def _corrupt(pts, rng, R, t, noise=1e-5, outliers=0.02):
    scan = pts @ R.T + t + rng.normal(0.0, noise, pts.shape)
    k = int(outliers * len(pts))
    idx = rng.choice(len(pts), k, replace=False)
    scan[idx, 2] += rng.uniform(-0.01, 0.01, k)         # gross outliers, up to 10 mm
    return scan


def test_icp_recovers_a_rigid_transform_despite_noise_and_outliers():
    rng = np.random.default_rng(3)
    part = EllipticCone(0.055, 0.04, 45.0, 0.015, 0.006, 0.006)
    target = part.heightmap(Grid.centered(0.13, 5e-4))
    pts = _scan(part, rng)
    Rt = rotation_matrix(np.radians([2.0, -1.5, 3.0]))
    tt = np.array([0.003, -0.002, 0.0015])
    scan = _corrupt(pts, rng, Rt, tt)
    al = align(scan, target, "rigid", initial="auto")
    assert al.converged and al.rank == 6
    moved = al.apply(pts @ Rt.T + tt)                     # the true points, re-aligned
    rms = np.sqrt(np.mean(np.sum((moved - pts) ** 2, axis=1)))
    assert rms < 1e-5
    assert rotation_angle_deg(al.rotation @ Rt) < 0.01
    assert al.rms_inliers < 2e-5                          # the noise level, outliers rejected
    assert 0.97 * len(pts) < al.n_inliers < 0.99 * len(pts)


def test_symmetric_and_flange_only_alignments_report_what_is_unobservable():
    rng = np.random.default_rng(4)
    cone = TruncatedCone(0.045, 45.0, 0.02, 0.008, 0.008)
    target = cone.heightmap(Grid.centered(0.13, 5e-4))
    pts = _scan(cone, rng)
    Rt = rotation_matrix(np.radians([1.0, -0.5, 0.0]))
    tt = np.array([0.001, 0.0, -0.002])
    scan = _corrupt(pts, rng, Rt, tt, outliers=0.0)
    al = align(scan, target, "rigid", initial="identity")
    assert al.rank == 5                                   # rotation about the cone axis is free
    assert np.abs(al.apply(pts @ Rt.T + tt)[:, 2] - pts[:, 2]).max() < 1e-4
    fl = align(scan, target, "rigid", fixture_mask=flange_mask(target), initial="identity")
    assert fl.rank == 3                                   # a flat flange fixes z, rx, ry only
    on_flange = np.hypot(pts[:, 0], pts[:, 1]) > cone.footprint_radius()
    dz = fl.apply(pts @ Rt.T + tt)[on_flange, 2] - pts[on_flange, 2]
    assert np.abs(dz).max() < 1e-6
    tz = align(pts + [0.0, 0.0, 0.0023], target, "translation_z")
    assert tz.rank == 1 and tz.translation[2] == pytest.approx(-0.0023, abs=1e-9)
    none = align(pts, target, "none")
    assert np.array_equal(none.rotation, np.eye(3)) and none.rms < 1e-6


def _offset_cone(cone, d, grid):
    """The exact surface at distance d along the unit normal of `cone`."""
    prof = cone.profile
    r = np.linspace(0.0, 0.2, 400001)
    w = cone.top_radius - r
    gw = prof(w)
    slope = -prof.slope(w)                     # dz/dr
    norm = np.sqrt(1.0 + slope ** 2)
    ro = r - d * slope / norm
    zo = gw + d / norm
    X, Y = grid.mesh()
    return HeightMap(grid, np.interp(np.hypot(X, Y), ro, zo)), (ro, zo)


@pytest.mark.parametrize("d", [3e-4, -3e-4])
def test_signed_deviation_of_a_normal_offset_is_the_offset(d):
    cone = TruncatedCone(0.035, 50.0, 0.015, 0.005, 0.005)
    grid = Grid.centered(0.1, 5e-4)
    target = cone.heightmap(grid)
    formed, (ro, zo) = _offset_cone(cone, d, grid)
    dev = signed_deviation(formed, target)
    walls = wall_mask(target, 5.0) & dev.mask
    assert dev.metadata["unconverged"] == 0
    # The only error is the bilinear interpolation of the formed field,
    # h^2 / 8 times its second derivative across a cell:
    # * on the straight wall, the circumferential curvature, tan(alpha) / r,
    #   largest at the foot of the wall;
    h = grid.h
    prof = cone.profile
    X, Y = grid.mesh()
    w = cone.top_radius - np.hypot(X, Y)
    straight = walls & (w > prof.tangent_lengths[0]) & (w < prof.w[1] - prof.tangent_lengths[1])
    r_foot = cone.top_radius - prof.w[1] + prof.tangent_lengths[1]
    bound = 1.2 * h ** 2 / 8 * np.tan(np.radians(50.0)) / r_foot
    assert straight.sum() > 3000
    assert np.abs(dev.z[straight] - d).max() < bound                    # 1.5 um
    # * in the fillets, the meridian curvature 1 / rho seen by a height field
    #   at wall angle theta, 1 / (rho cos^3 theta), projected on the normal.
    rho = 0.005 - abs(d)
    bound_fillet = 1.2 * h ** 2 / (8 * rho * np.cos(np.radians(50.0)) ** 2)
    assert np.abs(dev.z[walls] - d).max() < bound_fillet                # 20 um
    # the same from a dense point cloud of the offset surface
    rng = np.random.default_rng(1)
    rr = np.sqrt(rng.uniform(0, 0.045 ** 2, 60000))
    th = rng.uniform(0, 2 * np.pi, 60000)
    cloud = np.column_stack([rr * np.cos(th), rr * np.sin(th), np.interp(rr, ro, zo)])
    devc = signed_deviation(cloud, target)
    # a local plane through 8 neighbours misses a surface of curvature kappa
    # by about kappa s^2 / 2 at point spacing s (0.33 mm here): 2 um on the
    # straight wall, up to 10 um where the neighbourhood reaches into a
    # fillet, tens of um inside the fillets (hence method "grid" for scans)
    sel = straight & devc.mask
    spacing = devc.metadata["cloud_spacing_m"]
    assert np.abs(devc.z[sel] - d).max() < 1e-5
    # The circumferential curvature is one-signed, so the planar fit is
    # biased: by kappa r8^2 / 4 with r8 the radius holding 8 neighbours.
    kappa = np.tan(np.radians(50.0)) / r_foot
    r8 = np.sqrt(8.0 / (len(cloud) / (np.pi * 0.045 ** 2)) / np.pi)
    assert np.abs(np.median(devc.z[sel]) - d) < kappa * r8 ** 2 / 4       # 3.2 um
    assert 1e-4 < spacing < 3e-4                                  # nearest-neighbour distance


def test_vertical_deviation_and_metrics_are_exact():
    g = Grid.centered(0.02, 1e-3)
    t = HeightMap(g, np.zeros(g.shape))
    f = HeightMap(g, np.full(g.shape, -2e-4))
    assert np.allclose(vertical_deviation(f, t).z, -2e-4)
    m = metrics(np.array([3e-4, -4e-4, np.nan]), tolerance=3.5e-4)
    assert m["n"] == 2 and m["rms"] == pytest.approx(np.sqrt(12.5) * 1e-4)
    assert m["mae"] == pytest.approx(3.5e-4) and m["bias"] == pytest.approx(-0.5e-4)
    assert m["max_abs"] == pytest.approx(4e-4) and m["pct_within_tol"] == 50.0
    with pytest.raises(ValueError):
        metrics(np.array([np.nan]))
