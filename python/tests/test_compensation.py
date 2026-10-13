"""Displacement adjustment on synthetic linear springback operators with known inverses."""

import math

import numpy as np
import pytest

from scipy import ndimage

from precomp.compensation import (SurrogatePredictor, displacement_adjustment, limit_wall_angle,
                                  smooth_update)
from precomp.geometry import Grid, HeightMap, Pyramid, TruncatedCone
from precomp.metrology import flange_mask, part_mask


def linear_springback_predictor(factor: float, smoothing: float = 0.0):
    """A closed-form stand-in predictor: formed = c + (factor - 1) S(c), with
    S Gaussian smoothing at `smoothing` [m] (identity when 0). For
    0 < factor < 2 the operator G = (factor - 1) S has spectral radius
    |factor - 1| < 1, so DA's convergence on it is known."""
    def predict(c: HeightMap) -> HeightMap:
        sc = ndimage.gaussian_filter(c.z, smoothing / c.grid.h, mode="nearest") \
            if smoothing > 0 else c.z
        return c.with_z(c.z + (factor - 1.0) * sc)
    return predict


@pytest.fixture
def target():
    return TruncatedCone(0.035, 45.0, 0.015, 0.005, 0.005).heightmap(Grid.centered(0.1, 1e-3))


def test_da_converges_geometrically_at_the_spectral_radius(target):
    """formed = c + G c with G = -0.2 S (S a Gaussian smoothing, ||S|| <= 1):
    on the adjusted region e_(k+1) = -G_PP e_k, so the error falls by at
    most rho(G) = 0.2 per iteration with alpha = 1."""
    pred = linear_springback_predictor(0.8, smoothing=0.004)
    res = displacement_adjustment(target, pred, iterations=7, alpha=1.0,
                                  max_wall_angle_deg=None)
    rms = np.array(res.rms_history)
    ratios = rms[1:] / rms[:-1]
    assert np.all(ratios <= 0.2 + 1e-9)
    assert np.all(ratios > 0.15)                       # and not faster: G is that strong
    assert rms[-1] < 1e-4 * rms[0]
    assert res.best_iteration == len(rms) - 1
    assert res.commanded.depth > target.depth          # over-bent to beat springback
    # a half step converges more slowly: e_(k+1) = (I - alpha (I + G)) e_k
    half = displacement_adjustment(target, pred, iterations=4, alpha=0.5,
                                   max_wall_angle_deg=None)
    r = np.array(half.rms_history)
    assert np.all(r[1:] / r[:-1] < 0.61) and np.all(r[1:] / r[:-1] > 0.5)


def test_da_holds_the_flange_and_stays_below_the_sheet(target):
    seen = []

    def spy(k, c, f, err):
        seen.append(c.z.copy())

    pred = linear_springback_predictor(0.7, smoothing=0.01)   # spills onto the flange
    res = displacement_adjustment(target, pred, iterations=4, smoothing=0.002, callback=spy)
    fl = flange_mask(target)
    for z in seen + [res.proposed.z]:
        assert np.array_equal(z[fl], target.z[fl])
        assert z.max() <= 0.0
    assert res.rms_history[-1] < 0.2 * res.rms_history[0]


def test_formability_projection_only_raises_and_bounds_the_wall_angle():
    part = Pyramid(0.045, 0.04, 60.0, 0.02, 0.04, 0.003, 0.003)
    hm = part.heightmap(Grid.centered(0.12, 1e-3))
    steep = hm.with_z(1.8 * hm.z)
    assert np.degrees(steep.wall_angle().max()) > 70
    out = limit_wall_angle(steep, 65.0)
    assert np.all(out.z >= steep.z)
    assert np.degrees(out.wall_angle().max()) <= 65.0 + 1e-9
    assert out.metadata["wall_angle_limit"]["nodes_raised"] > 0
    same = limit_wall_angle(hm, 65.0)
    assert np.array_equal(same.z, hm.z)                 # formable input is untouched
    res = displacement_adjustment(hm, linear_springback_predictor(0.6), iterations=3)
    assert np.degrees(res.proposed.wall_angle().max()) <= 65.0 + 1e-9


def test_smoothing_does_not_leak_across_the_region_boundary():
    g = Grid.centered(0.02, 1e-3)
    region = np.zeros(g.shape, bool)
    region[:, :10] = True
    u = np.where(region, 1.0, 50.0)
    s = smooth_update(u, region, 3e-3, g.h)
    assert np.allclose(s[region], 1.0) and np.all(s[~region] == 0.0)


def test_normal_direction_da_converges_too(target):
    pred = linear_springback_predictor(0.85)
    res = displacement_adjustment(target, pred, iterations=4, direction="normal",
                                  max_wall_angle_deg=None)
    r = res.rms_history
    assert r[-1] < 0.01 * r[0]


class _ConstantModel:
    """A stand-in learned model: dz = -0.1 z_commanded, std 1e-5 m."""

    def predict_deviation(self, commanded, setup):
        return -0.1 * commanded.z, np.full(commanded.grid.shape, 1e-5)


def test_the_surrogate_predictor(target):
    sp = SurrogatePredictor(_ConstantModel(), setup=None)
    formed, std = sp.predict_with_uncertainty(target)
    assert np.allclose(formed.z, 0.9 * target.z) and np.all(std == 1e-5)
    res = displacement_adjustment(target, sp, iterations=4, max_wall_angle_deg=None)
    assert res.rms_history[-1] < 1e-3 * res.rms_history[0]
    with pytest.raises(TypeError):
        SurrogatePredictor(object(), None)


def test_limit_wall_angle_no_longer_stalls_on_combined_differences():
    """The central-difference slope combines an x and a y difference the cone
    need not bind; reducing the cone slope by the measured excess alone moved
    it by 0.07 % per round on this perturbed pyramid (found by the data
    generation) and gave up after 8 rounds. Bisection towards tan / sqrt(2)
    settles it."""
    from scipy import ndimage

    t = Pyramid(0.054947587200440476, 0.06160189188085497, 62.49617939814925,
                0.029419167179614306, 0.044168543145060545, 0.0033550517037510873,
                0.004863982778042555).heightmap(Grid(-0.1005, -0.1005, 68, 68, 3e-3))
    rng = np.random.default_rng(np.random.SeedSequence([0, 30, 1]))
    amp, corr = rng.uniform(2e-4, 1e-3), rng.uniform(0.008, 0.025)
    f = ndimage.gaussian_filter(rng.standard_normal(t.grid.shape), corr / 3e-3, mode="nearest")
    part = part_mask(t)
    w = np.clip(ndimage.distance_transform_edt(part) * 3e-3 / (2 * 0.007928765352815389), 0, 1)
    f = f * w * w * (3 - 2 * w)
    z = np.where(part, np.minimum(t.z + amp * f / np.sqrt(np.mean(f[part] ** 2)), 0.0), t.z)
    assert np.degrees(t.with_z(z).wall_angle().max()) > 65.0
    out = limit_wall_angle(t.with_z(z), 65.0)
    assert np.degrees(out.wall_angle().max()) <= 65.0 + 1e-9 and np.all(out.z >= z)
    info = out.metadata["wall_angle_limit"]
    assert math.tan(math.radians(65)) / math.sqrt(2) < info["cone_slope"] < 2.14
    assert 0 < info["nodes_raised"] < 10


def test_tool_reach_is_the_closing_by_the_ball_and_keeps_the_tool_path():
    from precomp.toolpath import spiral_toolpath, tool_center_surface, tool_reach_surface

    g = Grid.centered(0.04, 2.5e-4)
    X, Y = g.mesh()
    cone = TruncatedCone(0.009, 40.0, 0.003, 0.001, 0.004).heightmap(g)
    # a pit 1 mm wide (sigma 0.5 mm) and 0.5 mm deep in the 5 mm floor, off centre
    pit = cone.with_z(cone.z - 0.5e-3 * np.exp(-((X - 1.5e-3) ** 2 + Y ** 2)
                                              / (2 * 0.5e-3 ** 2)))
    r = tool_reach_surface(pit, 0.004)
    assert np.all(r.z >= pit.z) and np.all(r.z <= 0.0)
    assert r.metadata["tool_reach"]["max_raise_m"] > 0.3e-3     # the pit is not formed
    # a bowl curved five times less than the ball: the ball fits everywhere
    bowl = cone.with_z(np.minimum(0.0, -3e-3 + (X ** 2 + Y ** 2) / (2 * 0.02)))
    # (to the grid: the ball meets a node between nodes, h^2 / 8R = 2 um)
    np.testing.assert_allclose(tool_reach_surface(bowl, 0.004).z, bowl.z, atol=3e-6)
    # a floor fillet as large as the tool, but curved round the axis as well:
    # the ball does not quite fit (40 um here); with a 1 mm fillet, 0.24 mm
    raise_4 = tool_reach_surface(cone, 0.004).metadata["tool_reach"]["max_raise_m"]
    sharp = TruncatedCone(0.009, 40.0, 0.003, 0.001, 0.001).heightmap(g)
    raise_1 = tool_reach_surface(sharp, 0.004).metadata["tool_reach"]["max_raise_m"]
    assert 0.02e-3 < raise_4 < 0.06e-3 and 0.2e-3 < raise_1 < 0.3e-3
    # idempotent, and the same drop-cutter surface and tool path
    np.testing.assert_allclose(tool_reach_surface(r, 0.004).z, r.z, atol=1e-12)
    np.testing.assert_allclose(tool_center_surface(r, 0.004).z,
                               tool_center_surface(pit, 0.004).z, atol=1e-12)
    a, b = spiral_toolpath(pit, 0.004, 1e-3, 1e-3), spiral_toolpath(r, 0.004, 1e-3, 1e-3)
    np.testing.assert_allclose(a.points, b.points, atol=1e-12)


def test_da_clipped_to_the_tool_reach_stops_digging_what_the_tool_cannot_form():
    """The formed part is what the tool reaches, shallower by a constant
    springback; the target has a pit narrower than the tool. Unclipped, DA
    digs the pit deeper every step (the formed part never follows); clipped,
    the command stays formable and the rest of the part converges as
    before."""
    from precomp.toolpath import tool_reach_surface

    g = Grid.centered(0.04, 2.5e-4)
    X, Y = g.mesh()
    cone = TruncatedCone(0.009, 40.0, 0.003, 0.001, 0.004).heightmap(g)
    target = cone.with_z(cone.z - 0.5e-3 * np.exp(-((X - 1.5e-3) ** 2 + Y ** 2)
                                                 / (2 * 0.5e-3 ** 2)))
    R = 0.004

    def form(c):
        r = tool_reach_surface(c, R)
        return r.with_z(np.minimum(0.0, 0.9 * r.z))

    kw = dict(iterations=4, max_wall_angle_deg=None)
    free = displacement_adjustment(target, form, **kw)
    clip = displacement_adjustment(target, form, tool_radius=R, **kw)
    d_free = [h["commanded_depth_m"] for h in free.history]
    d_clip = [h["commanded_depth_m"] for h in clip.history]
    assert d_free[-1] - d_free[1] > 0.5e-3                   # keeps digging
    assert max(d_clip) < d_free[-1] - 0.5e-3
    assert np.allclose(tool_reach_surface(clip.proposed, R).z, clip.proposed.z, atol=1e-12)
    assert clip.proposed.metadata["tool_reach"]["tool_radius"] == R
    assert clip.commanded.metadata["compensation"]["tool_radius_m"] == R
    # where the ball fits, the clip changes nothing: the same error away from the pit
    away = part_mask(target) & (np.hypot(X - 1.5e-3, Y) > 4e-3)
    assert abs(free.history[1]["error"]["rms"] - clip.history[1]["error"]["rms"]) < 0.2e-3
    np.testing.assert_allclose(free.history[0]["error"]["rms"], clip.history[0]["error"]["rms"])
    assert np.abs((free.proposed.z - clip.proposed.z)[away]).max() < 1e-3
