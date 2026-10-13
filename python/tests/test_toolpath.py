"""Drop cutter and tool paths: no gouging, contact, closed loops, monotone spiral."""

import numpy as np
import pandas as pd
import pytest

from precomp.geometry import Freeform, Grid, HeightMap, TruncatedCone, zeros
from precomp.toolpath import (AIR, Toolpath, contour_toolpath, dsif_support_path,
                              dsif_support_points, lift_at_points, lift_heights, max_gouge,
                              rim_pass_path, signed_outline_distance, spherical_structure,
                              spiral_toolpath, swept_ball_top, tool_center_surface)

R = 0.005


def test_drop_cutter_on_a_plane_is_exactly_R_above_it():
    g = Grid.centered(0.05, 3e-4)
    for level in (0.0, -0.0123):
        cz = tool_center_surface(zeros(g).with_z(np.full(g.shape, level)), R)
        assert np.array_equal(cz.z, np.full(g.shape, level + R))


def test_spherical_structure_is_the_ball_cap():
    fp, st = spherical_structure(R, 1e-3)
    assert fp.shape == (11, 11) and fp[5, 5] and fp[5, 0] and not fp[0, 0]
    assert st[5, 5] == R and st[5, 0] == pytest.approx(0.0, abs=1e-12)


def test_drop_cutter_never_gouges_a_cone_and_touches_it_everywhere():
    """For tool centres over the walls, fillets and floor of a cone, the
    distance from the centre to the exact (analytic) surface is never below
    R - tol and equals R at the grid-node contact: the ball touches without
    gouging. tol bounds what a grid of spacing h can hide between nodes,
    about h^2 / (8 R cos^3 theta) = 6 um here (h = 0.25 mm, theta = 50 deg)."""
    cone = TruncatedCone(0.03, 50.0, 0.015, 0.004, 0.004)
    g = Grid.centered(0.1, 2.5e-4)
    cz = tool_center_surface(cone.heightmap(g), R)
    X, Y = g.mesh()
    rng = np.random.default_rng(3)
    cand = np.flatnonzero((np.hypot(X, Y) < cone.footprint_radius() + R).ravel())
    pick = rng.choice(cand, 120, replace=False)
    hf = g.h / 4.0                                  # dense offsets include the nodes
    k = np.arange(-int(1.05 * R / hf), int(1.05 * R / hf) + 1) * hf
    DX, DY = np.meshgrid(k, k, indexing="xy")
    tol = 1e-5
    dmin = []
    for idx in pick:
        c = np.array([X.ravel()[idx], Y.ravel()[idx], cz.z.ravel()[idx]])
        sx, sy = c[0] + DX, c[1] + DY
        sz = cone.height(sx, sy)
        dmin.append(np.sqrt((sx - c[0]) ** 2 + (sy - c[1]) ** 2 + (sz - c[2]) ** 2).min())
    dmin = np.array(dmin)
    assert dmin.min() >= R - tol                     # never penetrates
    assert np.abs(dmin - R).max() <= tol              # touches at every centre
    assert np.sum(np.abs(dmin - R) < 1e-12) > 10      # exactly R where the contact is a node


def _loops(path: Toolpath):
    """The closed ring of each level, in order: the in-contact run of a level
    is its link from the previous level followed by the ring, which starts at
    the first occurrence of its closing point."""
    out = []
    lv = path.level
    start = 0
    for i in range(1, len(lv) + 1):
        if i == len(lv) or lv[i] != lv[start]:
            if lv[start] != AIR:
                run = path.points[start:i]
                j = int(np.flatnonzero(np.all(run == run[-1], axis=1))[0])
                out.append((lv[start], run[j:]))
            start = i
    return out


def test_contour_toolpath_levels_are_closed_equally_spaced_ccw_loops(small_cone):
    g = Grid.centered(0.1, 5e-4)
    target = small_cone.heightmap(g)
    step, spacing = 1e-3, 1e-3
    path = contour_toolpath(target, R, step, spacing)
    heights = path.metadata["level_heights"]
    assert heights[:5] == pytest.approx([R - (k + 1) * step for k in range(5)])
    cz = tool_center_surface(target, R)
    assert heights[-1] == pytest.approx(cz.z.min() + 1e-3 * step)
    assert max_gouge(path, cz) == 0.0 and path.metadata["max_gouge_m"] == 0.0
    loops = [pts for lv, pts in _loops(path) if len(pts) > 10]
    assert len(loops) == len(heights)
    for ring in loops:
        closed = ring[-1] - ring[0]
        assert np.linalg.norm(closed) < 1e-12                          # closed loop
        seg = np.linalg.norm(np.diff(ring[:, :2], axis=0), axis=1)
        assert seg.max() <= spacing * (1 + 1e-9) and seg.min() > 0.5 * spacing
        x, y = ring[:-1, 0], ring[:-1, 1]
        area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
        assert area > 0                                               # counter-clockwise
    # consecutive loops start close to each other (aligned starts)
    starts = np.array([pts[0, :2] for pts in loops])
    assert np.linalg.norm(np.diff(starts, axis=0), axis=1).max() < 3 * step / np.tan(0.5) + spacing
    # alternate direction flips every other level
    alt = contour_toolpath(target, R, step, spacing, alternate=True)
    signs = []
    for pts in [p for lv, p in _loops(alt) if len(p) > 10][:4]:
        x, y = pts[:-1, 0], pts[:-1, 1]
        signs.append(np.sign(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))
    assert signs == [1, -1, 1, -1]


def test_spiral_descends_monotonically_and_touches_the_surface(small_cone):
    g = Grid.centered(0.1, 5e-4)
    target = small_cone.heightmap(g)
    path = spiral_toolpath(target, R, 1e-3, 1e-3)
    contact = path.level != AIR
    z = path.points[contact, 2]
    assert np.all(np.diff(z) <= 0.0)                                   # never climbs
    assert z[0] == R                                                   # tip on the sheet
    assert z[-1] == pytest.approx(path.metadata["level_heights"][-1])
    cz = tool_center_surface(target, R)
    gap = cz.interpolate(path.points[contact, 0], path.points[contact, 1]) - z
    assert gap.max() <= 1e-12                                          # never gouges
    first_rev = path.level[contact] == 1
    assert np.abs(gap[~first_rev]).max() < 1e-9                        # touches after rev. 1
    assert path.points[0, 2] == pytest.approx(R + 2e-3)                # approach from above
    assert path.points[-1, 2] == pytest.approx(R + 2e-3)               # and retract


def test_two_pockets_give_two_loops_per_level_and_air_moves():
    part = Freeform(((-0.03, 0.0, 0.012, 0.008), (0.03, 0.0, 0.012, 0.008)), 0.05, 0.06)
    g = Grid.centered(0.14, 5e-4)
    target = part.heightmap(g)
    path = contour_toolpath(target, R, 1e-3, 1e-3)
    assert max(path.metadata["loops_per_level"]) == 2
    air = np.flatnonzero(path.level == AIR)
    assert len(air) > 2                                                # traverses between pockets
    assert np.all(path.points[air, 2] == pytest.approx(R + 2e-3))
    with pytest.raises(ValueError, match="one loop per level"):
        spiral_toolpath(target, R, 1e-3, 1e-3)


def test_the_trajectory_file_holds_the_path_exactly(tmp_path, small_cone):
    target = small_cone.heightmap(Grid.centered(0.1, 5e-4))
    path = spiral_toolpath(target, R, 1e-3, 1e-3)
    csv = path.to_sparlab_csv(tmp_path / "toolpath.csv")
    frame = pd.read_csv(csv, float_precision="round_trip")
    assert list(frame.columns) == ["t", "x", "y", "z"]
    t = frame["t"].to_numpy()
    assert t[0] == 0.0 and t[-1] == 1.0 and np.all(np.diff(t) > 0)
    assert np.array_equal(frame[["x", "y", "z"]].to_numpy(), path.points)
    summ = path.summary(feed=0.02)
    assert summ["time_s"] == pytest.approx(path.total_length / 0.02)


def test_dsif_support_moves_with_the_tool_opposite_it_and_never_inside_the_sheet():
    """The support ball of DSIF: one centre per point of the forming path
    (the same pseudo-time); opposite the tool along the surface normal with
    the sine-law gap where the sheet is already formed; never inside the
    in-process sheet (the first revolution starts on a flat sheet); below
    the whole part in the air."""
    t, R2 = 1e-3, 0.004
    target = TruncatedCone(0.012, 40.0, 0.006, 0.002, 0.002).heightmap(Grid.centered(0.06, 2.5e-4))
    path = spiral_toolpath(target, R, 1e-3, 1e-3)
    pts = dsif_support_points(path, target, t, R2)
    assert pts.shape == path.points.shape
    air = path.level == AIR
    contact = ~air
    # in the air: below the deepest point, clear by the clearance
    assert np.all(pts[air, 2] + R2 <= -target.depth - t - 2e-3 + 1e-12)
    # never inside the in-process sheet (the surface above the tip's height,
    # flat at it below; the underside t lower), nor the finished one (to the
    # 1 um the last level keeps above the floor) ...
    xy = pts[contact, :2].T
    now = lift_heights(target, R2, *xy, offset=t, floor=path.points[contact, 2] - R)
    assert np.all(pts[contact, 2] <= now + 1e-12)
    assert np.all(pts[contact, 2] <= lift_heights(target, R2, *xy, offset=t) + 2e-6)
    # ... nor inside the flat sheet at the first contact
    first = np.flatnonzero(contact)[0]
    assert pts[first, 2] + R2 <= -t + 1e-9
    # where the sheet is formed (the last revolutions) the balls hold the
    # sine-law thickness t cos(theta) between them, along the normal
    d = np.linalg.norm(path.points - pts, axis=1)
    cz = tool_center_surface(target, R)
    nz = cz.sample(cz.normals(), path.points[:, 0], path.points[:, 1])[:, 2]
    late = contact & (path.points[:, 2] - R < -0.004)
    assert late.sum() > 50
    gap = d[late] - R - R2
    assert np.median(np.abs(gap - t * nz[late])) < 1e-6
    assert np.all(gap >= t * nz[late] - 1e-9)          # never squeezed more than asked
    # squeeze moves it towards the tool by squeeze t_n (where it is opposite)
    sq = dsif_support_points(path, target, t, R2, squeeze=0.2)
    d2 = np.linalg.norm(path.points - sq, axis=1)
    opposite = np.abs(gap - t * nz[late]) < 1e-6
    assert opposite.sum() > 0.5 * late.sum()
    assert np.allclose((d[late] - d2[late])[opposite], 0.2 * t * nz[late][opposite], atol=2e-6)
    # the Toolpath wrapper keeps the points (its own time is not the tool's)
    wrapped = dsif_support_path(path, target, thickness=t, support_radius=R2)
    assert wrapped.metadata["synchronised_with_primary"] is False
    assert np.all(wrapped.points[:, 2] < path.points[:len(wrapped.points), 2].max())
    with pytest.raises(ValueError):
        dsif_support_points(path, target, t, R2, thickness_law="thin")


def test_rim_pass_traces_the_band_from_below_outside_in():
    t, R2 = 1e-3, 0.004
    target = TruncatedCone(0.012, 40.0, 0.004, 0.002, 0.002).heightmap(Grid.centered(0.06, 2.5e-4))
    rim = rim_pass_path(target, target, R2, t, inside=2e-3, outside=1e-3, band_spacing=1e-3,
                        spacing=1e-3)
    assert rim.metadata["band_levels_m"] == pytest.approx([1e-3, 0.0, -1e-3, -2e-3])
    assert rim.level[0] == AIR and rim.level[-1] == AIR
    on = rim.level != AIR
    # every loop point touches the commanded underside from below at a node
    # and has no node of it inside the ball (checked node by node)
    X, Y = target.grid.mesh()
    under = target.z - t
    loops = rim.level[on]
    clearance = []
    for cx, cy, cz in rim.points[on]:
        d2 = (X - cx) ** 2 + (Y - cy) ** 2
        near = d2 <= R2 * R2
        clearance.append((under[near] - (cz + np.sqrt(R2 * R2 - d2[near]))).min())
    clearance = np.array(clearance)
    assert clearance.min() >= -1e-12              # no node inside any ball ...
    assert np.all(np.abs(clearance[loops == 1]) < 1e-12)   # ... and the loops touch
    assert np.median(np.abs(clearance)) < 1e-12    # (links between loops may stay lower)
    b = lift_at_points(target.with_z(under), R2, rim.points[on, 0], rim.points[on, 1])
    assert np.all(rim.points[on, 2] <= b + 1e-15)
    first = rim.points[on][loops == 1]
    # the lattice lift of the interpolated surface (lift_heights) agrees to a few um
    assert np.median(np.abs(first[:, 2] - lift_heights(target, R2, *first[:, :2].T,
                                                       offset=t))) < 5e-6
    # outside in: the loops shrink, around the part's axis
    radii = [np.median(np.hypot(*rim.points[on][loops == k][:, :2].T)) for k in (1, 2, 3, 4)]
    assert all(a > b_ for a, b_ in zip(radii, radii[1:]))
    sd = signed_outline_distance(target)
    edge = np.median(np.abs(sd.interpolate(first[:, 0], first[:, 1]) - 1e-3))
    assert edge < 2.5e-4
    # a raised command lifts the ball with it
    raised = target.with_z(np.where(sd.z < 0, target.z + 5e-4, target.z))
    up = rim_pass_path(raised, target, R2, t, inside=2e-3, outside=1e-3, band_spacing=1e-3,
                       spacing=1e-3)
    assert up.points[up.level == 3, 2].mean() > rim.points[rim.level == 3, 2].mean() + 1e-4


def test_swept_ball_top_is_the_upper_envelope_of_the_balls_along_the_path():
    g = Grid.centered(0.02, 2.5e-4)
    X, Y = g.mesh()
    Rb = 2e-3
    # one position: the ball's cap over its footprint, -inf beyond it
    top = swept_ball_top(np.array([[1e-3, -5e-4, -3e-3]]), Rb, g)
    d2 = (X - 1e-3) ** 2 + (Y + 5e-4) ** 2
    inside = d2 <= Rb * Rb
    assert np.allclose(top[inside], -3e-3 + np.sqrt(Rb * Rb - d2[inside]), atol=1e-15)
    assert np.all(np.isneginf(top[~inside]))
    # a straight move: the half cylinder over it (to max_step^2 / (8 R)), and
    # the balls at both ends
    a, b = np.array([-4e-3, 0.0, -2e-3]), np.array([4e-3, 0.0, -2e-3])
    top = swept_ball_top(np.vstack([a, b]), Rb, g)
    mid = (np.abs(X) <= 4e-3) & (np.abs(Y) < Rb)
    exact = -2e-3 + np.sqrt(Rb * Rb - Y[mid] ** 2)
    assert np.all(top[mid] <= exact + 1e-15)
    assert np.all(top[mid] >= exact - (0.25 * g.h) ** 2 / (8 * Rb) - 1e-15)
    assert np.all(np.isneginf(top[np.abs(Y) > Rb + 1e-12]))
    # a rising move reaches its highest end
    up = swept_ball_top(np.vstack([a, b + [0.0, 0.0, 1e-3]]), Rb, g)
    assert up.max() == pytest.approx(-1e-3 + Rb)
    with pytest.raises(ValueError):
        swept_ball_top(np.zeros((0, 3)), Rb, g)


def test_lift_at_points_is_exact_over_nodes_and_stable_under_its_own_sweep():
    """The ball at the lift height of a point touches the underside at a
    node and holds no node inside it; cutting the underside down to what a
    set of such balls sweeps (a surface that stood higher) leaves every one
    of them where it was - the property that makes the rim pass reach the
    command it was made for (precomp.fea.support.command_upper_bound)."""
    g = Grid.centered(0.02, 2.5e-4)
    X, Y = g.mesh()
    Rb = 1.5e-3
    under = HeightMap(g, 1e-3 * np.exp(-(X ** 2 + Y ** 2) / (2 * 0.003 ** 2)) - 1e-3)
    on_node = lift_at_points(under, Rb, [0.0], [0.0])
    assert on_node[0] == pytest.approx(under.z[g.shape[0] // 2, g.shape[1] // 2] - Rb, abs=1e-15)
    flat = HeightMap(g, np.full(g.shape, -1e-3))
    p = np.array([1.1e-4, -3.7e-4])
    b = lift_at_points(flat, Rb, [p[0]], [p[1]])[0]
    near = (X - p[0]) ** 2 + (Y - p[1]) ** 2 <= Rb * Rb
    assert np.isclose(b + np.sqrt(Rb * Rb - ((X - p[0]) ** 2 + (Y - p[1]) ** 2)[near]),
                      -1e-3, atol=1e-15).any()                           # touches a node
    assert np.isinf(lift_at_points(flat, Rb, [1.0], [1.0])[0])           # no node within R
    # a ring of balls under a raised underside, then the underside cut to their sweep
    ang = np.linspace(0.0, 2 * np.pi, 60)
    xy = np.column_stack([3e-3 * np.cos(ang), 3e-3 * np.sin(ang)])
    raised = under.with_z(under.z + 4e-4 * (np.hypot(X, Y) < 3.5e-3))
    z0 = lift_at_points(raised, Rb, *xy.T)
    top = swept_ball_top(np.column_stack([xy, z0]), Rb, g)
    cut = raised.with_z(np.minimum(raised.z, np.where(np.isfinite(top), top, raised.z)))
    assert (cut.z < raised.z - 1e-5).any()                               # it did cut
    assert np.allclose(lift_at_points(cut, Rb, *xy.T), z0, rtol=0.0, atol=1e-15)


def test_flat_target_is_refused():
    with pytest.raises(Exception, match="flat"):
        contour_toolpath(zeros(Grid.centered(0.05, 1e-3)), R, 1e-3)
    with pytest.raises(ValueError):
        Toolpath(np.zeros((1, 3)), np.zeros(1), R)
    with pytest.raises(ValueError):
        HeightMap(Grid.centered(0.05, 1e-3), np.zeros((2, 2)))


@pytest.mark.parametrize("make", [spiral_toolpath, contour_toolpath])
def test_the_deepest_level_reaches_a_dimpled_floor(make):
    """Over a flat floor the last level traces the floor's outline. A dimple
    at the deepest point (a compensated floor, a dome's pole) makes that
    contour - and a regular level just above it - too short to follow; the
    level is then raised only as far as needed, not dropped (a DA command
    was formed up to a step-down short of its floor)."""
    g = Grid.centered(0.04, 2.5e-4)
    cone = TruncatedCone(0.009, 40.0, 0.003, 0.001, 0.002).heightmap(g)
    X, Y = g.mesh()
    for dimple in (0.4e-3, 1.01e-3):   # the last level alone; it and a regular one
        target = cone.with_z(cone.z - dimple * np.exp(-(X ** 2 + Y ** 2) / (2 * 0.8e-3 ** 2)))
        cz = tool_center_surface(target, 0.004)
        path = make(target, 0.004, 1e-3, 1e-3)
        reach = float(cz.z.min())
        assert path.metadata["deepest_level_raised_m"] > 0.0
        # within the depth of the dimple's shortest followable loop (3.1 mm
        # round here), not a step-down (before: 0.4 and 1.0 mm short)
        assert path.points[path.level != AIR, 2].min() - reach < 0.2e-3
        assert max_gouge(path, cz) < 1e-9
    # a flat floor keeps its level just above the floor
    path = make(cone, 0.004, 1e-3, 1e-3)
    assert path.metadata["deepest_level_raised_m"] == 0.0
    assert path.points[:, 2].min() - float(tool_center_surface(cone, 0.004).z.min()) < 1e-5


def test_the_spiral_stays_in_the_pocket_round_a_small_off_centre_last_loop():
    """The last revolution blends from the previous loop towards a small loop
    round an off-centre dimple; a line from the loop that misses it crossed
    the surface only on the far side of the pocket or at the grid's edge
    (r = 20 mm here, a DA command of the fine-mesh benchmark refused for
    leaving the window). A crossing next to the small loop wins."""
    from precomp.geometry.parts import EllipticCone

    g = Grid.centered(0.04, 2.5e-4)
    X, Y = g.mesh()
    base = EllipticCone(semi_axis_x=0.0099, semi_axis_y=0.00698, wall_angle_deg=39.14,
                        depth=0.0021, top_fillet=0.0019, bottom_fillet=0.0015).heightmap(g)
    target = base.with_z(base.z - 0.1e-3 * np.exp(-((X + 3e-3) ** 2 + Y ** 2)
                                                  / (2 * 0.7e-3 ** 2)))
    path = spiral_toolpath(target, 0.004, 1e-3, 1e-3)
    assert path.metadata["deepest_level_raised_m"] > 0.0
    assert np.abs(path.points[path.level != AIR, :2]).max() < 0.012
    assert max_gouge(path, tool_center_surface(target, 0.004)) < 1e-9


def test_a_line_that_grazes_the_last_loop_keeps_its_nearest_point():
    """A line from the previous loop that only touches the last level at its
    nearest point B (the surface comes down to z there and rises again) has
    no crossing next to B; the next one, on the far side of the pocket, put
    the fine-mesh benchmark's pyramid-s2026-0000 FE-DA-2 command 20 mm out.
    B, on the surface at z, wins."""
    from precomp.toolpath import _blend_to_height

    g = Grid.centered(0.02, 2.5e-4)
    X, Y = g.mesh()
    z = np.where(X > 0.0, -1e-4, 50.0 * (X + 0.003) ** 2)
    cz = zeros(g).with_z(z)
    A = np.array([[-0.004, 0.0], [-0.004, 0.001]])
    B = np.array([[-0.003, 0.0], [-0.003, 0.001]])
    xy = _blend_to_height(cz, A, B, np.zeros(2))
    np.testing.assert_allclose(xy, B, atol=1e-12)


def test_a_row_inside_the_surface_without_a_crossing_walks_to_its_loop():
    """A line from A that never crosses z (the surface comes down to z only
    at B) kept A, which may lie above z: the tool centre then sat inside the
    surface. It now takes B, on its loop at z."""
    from precomp.toolpath import _blend_to_height

    g = Grid.centered(0.02, 2.5e-4)
    X, Y = g.mesh()
    cz = zeros(g).with_z(50.0 * (X + 0.003) ** 2)
    A = np.array([[-0.004, 0.0], [-0.004, 0.001]])
    B = np.array([[-0.003, 0.0], [-0.003, 0.001]])
    xy = _blend_to_height(cz, A, B, np.zeros(2))
    np.testing.assert_allclose(xy, B, atol=1e-12)
    # with a pit next to B the row stops where the surface reaches z
    cz2 = cz.with_z(cz.z - 2e-6)
    xy2 = _blend_to_height(cz2, A, B, np.zeros(2))
    gap = cz2.interpolate(xy2[:, 0], xy2[:, 1], masked=False)
    assert np.all(gap <= 0.0) and np.all(gap > -1e-7)
    assert np.all(np.abs(xy2[:, 0] + 0.003) < 0.25e-3)


def test_the_last_revolution_never_gouges_a_raised_spot_of_the_floor():
    """A DA command raises a spot of a steep cone's floor off centre. The
    last revolution starts at the previous loop, where no line towards the
    last loop crossed the deepest level: those points kept the previous
    loop's position, up to a step-down below the drop-cutter surface (the
    fine-mesh benchmark: 13 of 48 DA decks, 0.04-0.70 mm)."""
    g = Grid.centered(0.04, 2.5e-4)
    X, Y = g.mesh()
    cone = TruncatedCone(0.0081, 54.0, 0.003, 0.0015, 0.0015).heightmap(g)
    bump = 0.7e-3 * np.exp(-((X - 3.5e-3) ** 2 + Y ** 2) / (2 * 0.8e-3 ** 2))
    target = cone.with_z(np.minimum(cone.z + bump * (cone.z < -0.003 + 1e-4), 0.0))
    for make in (spiral_toolpath, contour_toolpath):
        path = make(target, 0.004, 1e-3, 1e-3)
        cz = tool_center_surface(target, 0.004)
        assert path.metadata["max_gouge_m"] == max_gouge(path, cz) < 1e-6
        assert path.points[:, 2].min() - float(cz.z.min()) < 1e-5
