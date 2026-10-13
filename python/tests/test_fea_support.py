"""Rim support: the backing plate, the DSIF support tool and its rim pass in
the deck, the setup field, the commands a support can realise, and what a
trained model makes of a setup with another support.

Every run here is the test double (fake_sparlab_form.py); the real solver
runs these decks in test_integration_sparlab.py.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from precomp import PrecompError
from precomp._util import canonical_json, read_json, sha256_bytes
from precomp.compensation import FEAPredictor, displacement_adjustment
from precomp.fea import FormingSetup, build_deck, deck_document, deck_hash, simulate
from precomp.fea.support import (COMMAND_TOLERANCE, PLATE_TOOL, SUPPORT_PATH_FILE, SUPPORT_TOOL,
                                 bottom_node_grid, check_command, command_upper_bound,
                                 compensation_masks, outline_distance, part_distance,
                                 plate_nodes, rim_band)
from precomp.geometry import Grid, HeightMap, TruncatedCone
from precomp.geometry.parts import part_from_dict
from precomp.materials import get_material
from precomp.metrology import flange_mask, part_mask
from precomp.toolpath import AIR, reach_from_below, signed_outline_distance

T = 1e-3


@pytest.fixture
def base(fake_solver):
    """A 40 mm blank with 2 mm elements (21 x 21 nodes per layer), the
    springback benchmark's process, run by the fake solver."""
    return FormingSetup(get_material("AA5754-O"), blank_size=0.04, clamp_margin=0.005,
                        element_size=2e-3, layers=2, thickness=T, tool_radius=4e-3,
                        step_down=1e-3, toolpath_spacing=1e-3, executable=str(fake_solver))


@pytest.fixture
def cone():
    return TruncatedCone(0.007, 40.0, 0.003, 0.001, 0.002).heightmap(Grid.centered(0.04, 2.5e-4))


def forming(deck_dir):
    return read_json(deck_dir / "deck.json")["forming"]


# ---------------------------------------------------------------------------
# The setup field
# ---------------------------------------------------------------------------
def test_support_is_a_setup_field_with_validated_settings(base):
    assert base.support == "none" and base.support_settings == {}
    assert base.physics_dict()["support"] == "none"
    plate = base.replace(support="backing_plate", support_settings={"clearance": 2e-3})
    back = FormingSetup.from_dict(json.loads(json.dumps(plate.to_dict())))
    assert back == plate and back.physics_dict()["support_settings"] == {"clearance": 2e-3}
    r = plate.resolved_support()
    assert r == {"clearance": 2e-3, "friction": base.friction, "penalty": 10.0}
    d = base.replace(support="dsif").resolved_support()
    assert d["radius"] == base.tool_radius and d["squeeze"] == 0.0 and d["rim_pass"] is False
    assert d["thickness_law"] == "sine"
    # an old setup document (no support fields) reads as "none"
    old = base.to_dict()
    del old["support"], old["support_settings"]
    assert FormingSetup.from_dict(old) == base
    for bad, match in [({"support": "die"}, "support must be one of"),
                       ({"support_settings": {"clearance": 1e-3}}, "not settings of support"),
                       ({"support": "backing_plate", "support_settings": {"squeeze": 0.1}},
                        "not settings"),
                       ({"support": "backing_plate", "support_settings": {"clearance": -1.0}},
                        "clearance"),
                       ({"support": "dsif", "support_settings": {"squeeze": 1.0}}, "squeeze"),
                       ({"support": "dsif", "support_settings": {"rim_pass": 1}}, "rim_pass"),
                       ({"support": "dsif", "support_settings": {"thickness_law": "x"}},
                        "thickness_law"),
                       ({"support": "dsif", "support_settings": {"radius": 0.0}}, "radius")]:
        with pytest.raises(ValueError, match=match):
            base.replace(**bad)


# ---------------------------------------------------------------------------
# No support: the deck and its hash are what they were
# ---------------------------------------------------------------------------
def test_without_support_the_deck_and_its_hash_are_unchanged(tmp_path, base, cone):
    d = build_deck(base, cone, tmp_path / "d", target=cone)
    fm = forming(d)
    assert [t["name"] for t in fm["tools"]] == ["tool"]
    assert [s["name"] for s in fm["steps"]] == ["form", "unload", "release"]
    assert not (d / SUPPORT_PATH_FILE).exists()
    assert "support" not in read_json(d / "precomp_deck.json")
    # the content hash is the one of before: deck, toolpath.csv, version
    doc = read_json(d / "deck.json")
    old = sha256_bytes(canonical_json(doc).encode(), (d / "toolpath.csv").read_bytes(), b"v")
    assert deck_hash(d, "v") == old
    assert deck_document(base) == deck_document(base, support=None)
    with pytest.raises(PrecompError, match="plan"):
        deck_document(base.replace(support="dsif"))


# ---------------------------------------------------------------------------
# Backing plate
# ---------------------------------------------------------------------------
def test_the_backing_plate_carries_the_bottom_faces_outside_the_opening(tmp_path, base, cone):
    setup = base.replace(support="backing_plate", support_settings={"clearance": 1e-3})
    d = build_deck(setup, cone, tmp_path / "plate", target=cone)
    fm = forming(d)
    assert [t["name"] for t in fm["tools"]] == ["tool", PLATE_TOOL]
    plate = fm["tools"][1]
    assert plate["shape"] == "plane" and plate["normal"] == [0.0, 0.0, 1.0]
    assert plate["trajectory"]["points"] == [[0.0, 0.0, -T], [0.0, 0.0, -T]]
    assert plate["friction"] == setup.friction and plate["penalty"] == 10.0
    steps = {s["name"]: s for s in fm["steps"]}
    assert steps["form"]["tools"] == ["tool", PLATE_TOOL]
    assert steps["unload"]["tools"] == [] and steps["release"]["tools"] == []
    ids = plate["surface"]["node_ids"]
    grid_ids, X, Y = bottom_node_grid(setup)
    n = setup.elements_per_side
    assert grid_ids.max() == (n + 1) ** 2 - 1 and set(ids) <= set(grid_ids.ravel())
    listed = np.isin(grid_ids, ids)
    # the clearance is measured to the outline (signed_outline_distance: half
    # way between the last part node and the first flange node), not to the
    # part's nodes, which lie up to a grid spacing inside it
    dist = outline_distance(cone, X, Y)
    sd = signed_outline_distance(cone)
    assert np.allclose(dist, sd.interpolate(X, Y), atol=1e-12)       # the mesh on grid nodes
    assert np.allclose(dist[dist > 0], part_distance(cone, X, Y)[dist > 0] - 0.5 * cone.grid.h)
    assert np.all(dist[listed] > 1e-3)                       # outside the opening ...
    # ... and every node outside it whose cell is wholly outside is listed
    out = dist > 1e-3
    cell = out[:-1, :-1] & out[1:, :-1] & out[:-1, 1:] & out[1:, 1:]
    assert listed[:-1, :-1][cell].all() and listed[1:, 1:][cell].all()
    assert not listed[dist < 1e-3].any()
    info = read_json(d / "precomp_deck.json")["support"]
    assert info["plate"]["realised_clearance_m"] == pytest.approx(dist[listed].min())
    assert info["plate"]["realised_clearance_m"] >= 1e-3
    assert info["plate"]["realised_clearance_m"] < 1e-3 + setup.element_size * np.sqrt(2)
    assert info["outline_from"] == "target"
    # a wider clearance carries fewer faces
    assert len(plate_nodes(setup, cone, 3e-3)[0]) < len(ids)


PYRAMID = {"family": "pyramid", "half_width_x": 0.007445503572002053,
           "half_width_y": 0.008327501218765975, "corner_radius": 0.006246182421222329,
           "wall_angle_deg": 34.65761963278055, "depth": 0.003382830085232854,
           "top_fillet": 0.001299202023074031, "bottom_fillet": 0.0013530378779396416}


@pytest.mark.parametrize("clearance", [0.0, 1e-3, 2.5e-3])
def test_the_plate_opening_follows_a_non_circular_outline(tmp_path, base, clearance):
    """On a rounded-rectangle pyramid (pyramid-s2026-0000 of the springback
    benchmark, 14.9 x 16.7 mm) the plate carries exactly the bottom faces
    whose corners all lie farther than the clearance outside the outline:
    none nearer, none missed, and the faces sparlab_form's selection takes
    (every node of a face listed) are exactly those."""
    setup = base.replace(support="backing_plate", support_settings={"clearance": clearance})
    target = part_from_dict(PYRAMID).heightmap(Grid.centered(setup.meshed_blank_size, 2.5e-4))
    ids, info = plate_nodes(setup, target)
    grid_ids, X, Y = bottom_node_grid(setup)
    listed = np.isin(grid_ids, ids)
    dist = outline_distance(target, X, Y)
    out = dist > clearance
    cell = out[:-1, :-1] & out[1:, :-1] & out[:-1, 1:] & out[1:, 1:]
    selected = listed[:-1, :-1] & listed[1:, :-1] & listed[:-1, 1:] & listed[1:, 1:]
    assert np.array_equal(selected, cell) and cell.sum() == info["carried_faces"]
    assert np.all(dist[listed] > clearance) and info["realised_clearance_m"] > clearance
    # every carried node belongs to a carried face (no stray node)
    corner = np.zeros_like(listed)
    for a, b in ((slice(None, -1), slice(None, -1)), (slice(1, None), slice(None, -1)),
                 (slice(None, -1), slice(1, None)), (slice(1, None), slice(1, None))):
        corner[a, b] |= cell
    assert np.array_equal(corner, listed)
    # the opening is not a circle: along the pyramid's short axis (x) the
    # plate comes nearer the centre than along its long axis (y)
    on_x = listed & (np.abs(Y) < 1e-9)
    on_y = listed & (np.abs(X) < 1e-9)
    assert np.abs(X[on_x]).min() <= np.abs(Y[on_y]).min()
    d = build_deck(setup, target, tmp_path / "pyr", target=target)
    assert forming(d)["tools"][1]["surface"]["node_ids"] == ids


def test_the_plate_follows_the_target_not_the_compensated_command(tmp_path, base, cone):
    """Fixed hardware: a command whose rim DA raised to the sheet plane
    (outline shrunk) keeps the plate made for the target."""
    setup = base.replace(support="backing_plate", support_settings={"clearance": 1e-3})
    sd = signed_outline_distance(cone).z
    clipped = cone.with_z(np.where(sd > -1.5e-3, 0.0, cone.z))
    a = forming(build_deck(setup, clipped, tmp_path / "a", target=cone))["tools"][1]
    b = forming(build_deck(setup, cone, tmp_path / "b", target=cone))["tools"][1]
    c = forming(build_deck(setup, clipped, tmp_path / "c"))["tools"][1]
    assert a["surface"] == b["surface"]
    assert len(c["surface"]["node_ids"]) > len(b["surface"]["node_ids"])


def test_the_plate_node_ids_follow_the_structured_numbering(tmp_path, base, cone, counter):
    """The ids the deck lists are, in the mesh the solver writes, bottom
    nodes outside the opening (the numbering of StructuredMesh.hpp)."""
    setup = base.replace(support="backing_plate")
    res = simulate(setup, cone, tmp_path / "w", target=cone)
    mesh = json.loads((res.directory / "mesh.json").read_text())
    nodes = np.asarray(mesh["nodes_m"])
    ids = res.summary and read_json(res.directory / "config.json")["forming"]["tools"][1][
        "surface"]["node_ids"]
    assert np.allclose(nodes[ids, 2], -T)
    assert np.all(outline_distance(cone, nodes[ids, 0], nodes[ids, 1]) > 1e-3)
    assert res.step_names == ["form", "unload", "release"]
    f = res.forming_forces()
    assert set(f["tool"]) == {"tool", PLATE_TOOL}


# ---------------------------------------------------------------------------
# DSIF
# ---------------------------------------------------------------------------
def test_dsif_drives_both_tools_on_one_pseudo_time(tmp_path, base, cone):
    setup = base.replace(support="dsif")
    d = build_deck(setup, cone, tmp_path / "dsif", target=cone)
    fm = forming(d)
    assert [t["name"] for t in fm["tools"]] == ["tool", SUPPORT_TOOL]
    sup = fm["tools"][1]
    assert sup["shape"] == "sphere" and sup["radius"] == setup.tool_radius
    assert sup["trajectory"] == {"file": SUPPORT_PATH_FILE}
    box = sup["surface"]["box"]
    assert box["zmax"] == -T and fm["tools"][0]["surface"]["box"]["zmin"] == 0.0  # disjoint
    steps = {s["name"]: s for s in fm["steps"]}
    assert steps["form"]["tools"] == ["tool", SUPPORT_TOOL]
    assert steps["unload"]["tools"] == []                           # both removed
    tool = pd.read_csv(d / "toolpath.csv")
    support = pd.read_csv(d / SUPPORT_PATH_FILE)
    # synchronised: every knot of the forming tool is one of the support's
    # (which has more where it swings round the tool, `SUPPORT_KNOT_STEP`)
    assert np.isin(tool["t"].to_numpy(), support["t"].to_numpy()).all()
    # the support is under the sheet all along: its top never above the flat
    # sheet's underside (no command rises above the plane here)
    assert np.all(support["z"] + setup.tool_radius <= -T + 1e-12)
    # the hash covers the support's trajectory
    h = deck_hash(d, "v")
    d2 = build_deck(setup.replace(support_settings={"squeeze": 0.1}), cone, tmp_path / "sq",
                    target=cone)
    assert read_json(d2 / "deck.json") == read_json(d / "deck.json")
    assert deck_hash(d2, "v") != h


def test_the_dsif_support_never_pinches_the_sheet_between_its_knots(base):
    """At a corner the support swings round the forming tool; both tools move
    in straight lines between knots, so without knots in between it cuts the
    chord towards the tool. The fine-mesh benchmark's pyramid-s2026-0001 had
    the balls 0.1 mm apart (its DA command: overlapping), which stalled the
    solver."""
    from precomp.fea.deck import forming_surface, make_toolpath
    from precomp.fea.support import SUPPORT_KNOT_STEP, dsif_trajectory
    from precomp.geometry import Pyramid

    setup = base.replace(support="dsif")
    target = Pyramid(half_width_x=0.008118, half_width_y=0.006990, wall_angle_deg=41.21,
                     depth=0.00258, corner_radius=0.004221, top_fillet=0.001807,
                     bottom_fillet=0.001854).heightmap(Grid.centered(0.04, 2.5e-4))
    path = make_toolpath(setup, target)
    t, pts, _, info = dsif_trajectory(setup, forming_surface(target), path, target, target)
    assert info["support_knots_added"] > 0
    assert np.isin(path.t, t).all()
    tt = np.linspace(0.0, 1.0, 200001)
    tool = np.stack([np.interp(tt, path.t, path.points[:, k]) for k in range(3)], 1)
    sup = np.stack([np.interp(tt, t, pts[:, k]) for k in range(3)], 1)
    gap = np.linalg.norm(tool - sup, axis=1) - setup.tool_radius - info["support_radius_m"]
    # the sine-law thickness on its 41 deg walls, less the chord's sag
    assert gap.min() > T * np.cos(np.radians(41.21)) - 0.05e-3
    # in contact, the support moves about SUPPORT_KNOT_STEP at most between knots
    contact = path.level[np.searchsorted(path.t, t)] != AIR
    step = np.linalg.norm(np.diff(pts, axis=0), axis=1)[contact[:-1] & contact[1:]]
    assert step.max() <= 1.5 * SUPPORT_KNOT_STEP


def test_the_dsif_rim_pass_runs_after_unload_and_is_removed_before_the_release(tmp_path, base,
                                                                               cone, counter):
    setup = base.replace(support="dsif", support_settings={"rim_pass": True})
    d = build_deck(setup, cone, tmp_path / "rim", target=cone)
    steps = forming(d)["steps"]
    assert [s["name"] for s in steps] == ["form", "unload", "rim_pass", "rim_unload", "release"]
    rim = steps[2]
    assert rim["type"] == "form" and rim["tools"] == [SUPPORT_TOOL]
    assert rim["time"][0] == 2.0 and 2.0 < rim["time"][1] <= 3.0
    assert steps[3]["tools"] == [] and steps[4]["tools"] == []
    sup = pd.read_csv(d / SUPPORT_PATH_FILE)
    assert np.all(np.diff(sup["t"]) > 0) and sup["t"].iloc[-1] == pytest.approx(3.0)
    tool = pd.read_csv(d / "toolpath.csv")
    form = sup["t"].to_numpy()[sup["t"].to_numpy() <= 1.0]
    assert np.isin(tool["t"].to_numpy(), form).all()
    band = sup[(sup["t"] >= 2.0) & (sup["t"] <= rim["time"][1])]
    assert band["z"].max() + setup.tool_radius <= -T + 1e-9   # pushes up to the underside
    info = read_json(d / "precomp_deck.json")["support"]["dsif"]["rim_pass"]
    assert info["band_levels_m"] == pytest.approx([1e-3, 0.0, -1e-3, -2e-3])
    res = simulate(setup, cone, tmp_path / "w", target=cone)
    assert res.step_names == ["form", "unload", "rim_pass", "rim_unload", "release"]
    f = res.forming_forces()
    assert set(f[f["step"] == 3]["tool"]) == {SUPPORT_TOOL}


def test_a_rim_pass_reaching_the_clamp_is_refused(tmp_path, base, cone):
    wide = base.replace(support="dsif", support_settings={"rim_pass": True,
                                                           "rim_outside": 8e-3})
    with pytest.raises(PrecompError, match="rim pass reaches"):
        build_deck(wide, cone, tmp_path / "d", target=cone)


# ---------------------------------------------------------------------------
# Commands above the sheet plane
# ---------------------------------------------------------------------------
def test_only_the_rim_pass_realises_a_command_above_the_sheet_plane(tmp_path, base, cone):
    rim = base.replace(support="dsif", support_settings={"rim_pass": True,
                                                          "rim_max_raise": 1e-3,
                                                          "radius": 1.5e-3})
    band, strip = rim_band(rim, cone)
    sd = signed_outline_distance(cone).z
    assert np.array_equal(band, part_mask(cone) & (sd >= -2e-3))
    assert np.array_equal(strip, flange_mask(cone) & (sd <= 1e-3))
    for s in (base, base.replace(support="backing_plate"), base.replace(support="dsif")):
        assert not command_upper_bound(s, cone)(cone.z + 1e-3).any()
        assert not rim_band(s, cone)[0].any()
    # a bump over band and strip: seen from below the rim is a corner, which
    # the ball cannot fill - the bound cuts the command to what it reaches
    bump = 4e-4 * np.clip(1.0 - np.abs(sd + 0.5e-3) / 1.5e-3, 0.0, None)
    asked = cone.with_z(cone.z + bump)
    bound = command_upper_bound(rim, cone)
    ub = bound(asked.z)
    assert np.all(ub[~(band | strip)] == 0.0) and ub.max() <= 1e-3
    raised = asked.with_z(np.minimum(asked.z, ub))
    assert raised.z.max() > 1.5e-4 and (raised.z < asked.z - 1e-5).any()
    assert np.allclose(np.minimum(raised.z, bound(raised.z)), raised.z)   # a fixed point
    check_command(rim, raised, cone)                               # realisable
    with pytest.raises(PrecompError, match="too high"):
        check_command(rim, asked, cone)
    with pytest.raises(PrecompError, match="pass the target"):
        check_command(rim, raised, None)
    for s in (base, base.replace(support="backing_plate"), base.replace(support="dsif")):
        with pytest.raises(PrecompError, match="rises above what the tools can realise"):
            build_deck(s, raised, tmp_path / s.support, target=cone)
    with pytest.raises(PrecompError, match="too high"):             # above rim_max_raise
        check_command(rim, cone.with_z(np.where(band, 1.5e-3, cone.z)), cone)
    # a raised ring narrower than the ball: out of its reach from below
    narrow = cone.with_z(np.where(band & (sd > -0.5e-3), 5e-4, cone.z))
    with pytest.raises(PrecompError, match="too high"):
        check_command(rim.replace(support_settings={"rim_pass": True, "radius": 4e-3}),
                      narrow, cone)
    # the forming tool's path is made for the part below the plane only; the
    # rim pass follows the raised command
    d = build_deck(rim, raised, tmp_path / "ok", target=cone)
    d0 = build_deck(rim, raised.with_z(np.minimum(raised.z, 0.0)), tmp_path / "ok0",
                    target=cone)
    assert (d / "toolpath.csv").read_bytes() == (d0 / "toolpath.csv").read_bytes()
    up, flat = (pd.read_csv(x / SUPPORT_PATH_FILE) for x in (d, d0))
    assert up["z"][up["t"] >= 2].max() > flat["z"][flat["t"] >= 2].max() + 1e-4


def test_reach_from_below_is_the_underside_where_the_ball_fits():
    g = Grid.centered(0.02, 2.5e-4)
    X, Y = g.mesh()
    wide = HeightMap(g, 1e-3 * np.exp(-(X ** 2 + Y ** 2) / (2 * 0.004 ** 2)))
    assert np.allclose(reach_from_below(wide, 1e-3).z, wide.z, atol=1e-6)
    narrow = HeightMap(g, np.where(np.abs(X) < 5e-4, 1e-3, 0.0))
    r = reach_from_below(narrow, 4e-3).z
    assert np.all(r <= narrow.z + 1e-15)
    assert r[np.abs(X) < 2.5e-4].max() < 1e-4                     # the slot is out of reach
    assert np.allclose(r[np.abs(X) > 3e-3], 0.0)


def _sagging_predictor(target):
    """formed = commanded - a sag of up to 1 mm around the outline (a
    stand-in for the simulated rim sag)."""
    sd = signed_outline_distance(target).z
    sag = 1e-3 * np.exp(-((sd + 5e-4) / 1.5e-3) ** 2)

    def predict(c):
        return c.with_z(c.z - sag)
    return predict


def test_displacement_adjustment_rises_above_the_plane_only_where_the_support_can(base, cone):
    rim = base.replace(support="dsif", support_settings={"rim_pass": True,
                                                          "rim_max_raise": 2e-3,
                                                          "radius": 1.5e-3})
    pred = _sagging_predictor(cone)
    plain = displacement_adjustment(cone, pred, iterations=1)
    assert plain.proposed.z.max() <= 0.0                         # SPIF: z <= 0
    bound = command_upper_bound(rim, cone)
    hold, adjust = compensation_masks(rim, cone)
    band, strip = rim_band(rim, cone)
    assert np.array_equal(adjust, part_mask(cone) | strip)
    assert np.array_equal(hold, flange_mask(cone) & ~strip)
    up = displacement_adjustment(cone, pred, iterations=1, upper_bound=bound, hold_mask=hold,
                                 adjust_mask=adjust)
    z = up.proposed.z
    assert z.max() > 3e-4                                        # the rim is raised ...
    assert np.all(z <= bound(z) + 1e-6)                          # ... as far as realisable
    assert np.all(z[~(band | strip)] <= 0.0)
    assert np.all(z[hold] == 0.0)                                # the rest of the flange held
    check_command(rim, up.proposed, cone)
    # with a big ball the same update is cut to what the ball reaches
    big = rim.replace(support_settings={"rim_pass": True, "radius": 4e-3,
                                        "rim_max_raise": 2e-3})
    cut = displacement_adjustment(cone, pred, iterations=1,
                                  upper_bound=command_upper_bound(big, cone),
                                  hold_mask=hold, adjust_mask=adjust).proposed.z
    assert cut[band | strip].max() < z[band | strip].max()
    check_command(big, up.proposed.with_z(cut), cone)
    with pytest.raises(ValueError, match="upper_bound"):
        displacement_adjustment(cone, pred, iterations=1, upper_bound=-1.0)


def _written_rim_pass_reach(deck_dir, radius, grid):
    """What the rim pass written into the deck pushes the sheet's top to,
    worked out here from support_path.csv alone: the ball moved in straight
    lines between the knots of the "rim_pass" step (sampled every 20 um),
    its top over every node, plus the sheet thickness."""
    step = next(s for s in forming(deck_dir)["steps"] if s["name"] == "rim_pass")
    sp = pd.read_csv(deck_dir / SUPPORT_PATH_FILE)
    knots = sp[(sp["t"] >= step["time"][0]) & (sp["t"] <= step["time"][1])]
    knots = knots[["x", "y", "z"]].to_numpy()
    X, Y = grid.mesh()
    top = np.full(X.shape, -np.inf)
    for a, b in zip(knots[:-1], knots[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / 2e-5)))
        for c in a + (np.arange(n + 1) / n)[:, None] * (b - a):
            win = (np.abs(X - c[0]) <= radius) & (np.abs(Y - c[1]) <= radius)
            d2 = (X[win] - c[0]) ** 2 + (Y[win] - c[1]) ** 2
            cap = np.where(d2 <= radius ** 2, c[2] + np.sqrt(np.clip(radius ** 2 - d2, 0, None)),
                           -np.inf)
            top[win] = np.maximum(top[win], cap)
    return top + T


@pytest.mark.parametrize("radius", [1.5e-3, 2e-3, 4e-3])
def test_the_written_rim_pass_reaches_every_command_it_accepts(tmp_path, base, cone, radius):
    """The bound lets a command rise above the sheet plane only as far as
    the rim pass the deck then writes for that command pushes the sheet: DA
    against the stand-in sag, cut by the bound, is accepted, and every raised
    node lies within the tolerance of the top the written ball sweeps
    (worked out independently from support_path.csv). Its loops follow the
    command, so each cut is where the ball, not a guess of it, stops: a raise
    between two loops, next to the held flange or in the rim's corner seen
    from below is cut; the rest is kept."""
    rim = base.replace(support="dsif", support_settings={"rim_pass": True,
                                                          "rim_max_raise": 2e-3,
                                                          "radius": radius})
    hold, adjust = compensation_masks(rim, cone)
    bound = command_upper_bound(rim, cone)
    up = displacement_adjustment(cone, _sagging_predictor(cone), iterations=1,
                                 upper_bound=bound, hold_mask=hold,
                                 adjust_mask=adjust).proposed
    free = displacement_adjustment(cone, _sagging_predictor(cone), iterations=1,
                                   upper_bound=2e-3, hold_mask=hold,
                                   adjust_mask=adjust).proposed
    raised = up.z > COMMAND_TOLERANCE
    assert free.z.max() > 5e-4 and (up.z < free.z - 1e-4).any()     # the bound cut ...
    d = build_deck(rim, up, tmp_path / "d", target=cone)            # ... and it is accepted
    reach = _written_rim_pass_reach(d, radius, cone.grid)
    assert np.all(up.z[raised] <= reach[raised] + COMMAND_TOLERANCE)
    # the bound is a fixed point: the command it allows is the one it returns
    assert np.array_equal(np.minimum(up.z, bound(up.z)), up.z)
    if radius < 4e-3:
        assert up.z.max() > 3e-4 and raised.sum() > 1000
    else:                        # the validation's ball: the rim's corner is out of its reach
        assert up.z.max() < 5e-5
    # what DA asked for without the bound is out of reach, and refused
    with pytest.raises(PrecompError, match="too high"):
        build_deck(rim, free, tmp_path / "free", target=cone)


def test_a_command_within_a_micrometre_of_the_plane_is_taken_as_it_is(tmp_path, base, cone):
    """1 um (COMMAND_TOLERANCE) is the one tolerance above the sheet plane:
    a command no higher is accepted with every support, with or without a
    target, and the forming path is made for it as it is (as before
    supports existed); a higher one needs a rim pass, and without one it is
    refused for what it is - not for a missing target."""
    rim = base.replace(support="dsif", support_settings={"rim_pass": True})
    setups = (base, base.replace(support="backing_plate"), base.replace(support="dsif"), rim)
    noisy = cone.with_z(np.where(cone.z >= 0.0, 1e-8, cone.z))       # 10 nm over the flange
    for i, s in enumerate(setups):
        for target in (None, cone):
            build_deck(s, noisy, tmp_path / f"{i}-{target is None}", target=target)
    from precomp.fea.deck import forming_surface, make_toolpath
    assert forming_surface(noisy) is noisy
    d = tmp_path / "0-True"
    assert pd.read_csv(d / "toolpath.csv")["z"].to_numpy() == pytest.approx(
        make_toolpath(base, noisy).points[:, 2], abs=1e-15)
    high = cone.with_z(np.where(cone.z >= 0.0, 1e-5, cone.z))        # 10 um
    for s in setups[:3]:
        for target in (None, cone):
            with pytest.raises(PrecompError, match="nothing pushes the sheet up") as err:
                check_command(s, high, target)
            assert "target" not in str(err.value).split("(")[0]
    with pytest.raises(PrecompError, match="pass the target"):
        check_command(rim, high, None)
    with pytest.raises(PrecompError, match="too high"):             # the flange beyond the band
        check_command(rim, high, cone)


def test_fea_prediction_forms_every_iterate_on_the_targets_fixture(tmp_path, base, cone,
                                                                   counter):
    setup = base.replace(support="dsif", support_settings={"rim_pass": True})
    pred = FEAPredictor(setup, tmp_path / "w", target=cone)
    hold, adjust = compensation_masks(setup, cone)
    da = displacement_adjustment(cone, pred, iterations=2, hold_mask=hold, adjust_mask=adjust,
                                 upper_bound=command_upper_bound(setup, cone))
    assert len(pred.results) == 2
    for res in pred.results:
        prov = read_json(res.directory.parent / "precomp_deck.json")
        assert prov["support"]["outline_from"] == "target"
    assert da.history[0]["error"]["rms"] > 0


def _runs(work):
    """(outline_from, plate node ids) of every run in a cache directory."""
    out = []
    for prov in sorted(Path(work).rglob("precomp_deck.json")):
        deck = read_json(prov.parent / "deck.json")
        out.append((read_json(prov)["support"]["outline_from"],
                    deck["forming"]["tools"][1]["surface"]["node_ids"]))
    return out


def test_fe_da_forms_on_the_targets_plate(tmp_path, base, cone, counter):
    """A compensated command (its rim raised to the plane: a smaller outline)
    is formed on the plate made for the target when the FE-DA predictor is
    given the target. Without the target the command's own, different plate
    is used."""
    setup = base.replace(support="backing_plate")
    sd = signed_outline_distance(cone).z
    comp = cone.with_z(np.where(sd > -1.5e-3, 0.0, cone.z))
    plate = plate_nodes(setup, cone)[0]
    assert plate_nodes(setup, comp)[0] != plate                    # the outlines differ
    FEAPredictor(setup, tmp_path / "api", target=cone)(comp)
    assert _runs(tmp_path / "api") == [("target", plate)]
    FEAPredictor(setup, tmp_path / "own")(comp)
    assert _runs(tmp_path / "own") == [("commanded", plate_nodes(setup, comp)[0])]


# ---------------------------------------------------------------------------
# A trained model and another support
# ---------------------------------------------------------------------------
def test_a_model_refuses_a_setup_with_a_support_it_was_not_trained_on(base):
    from precomp.ml.surrogate import setup_envelope, setup_envelope_union, setup_mismatch

    env = setup_envelope([base, base.replace(step_down=2e-3)])
    assert env["fields"]["support"] == {"values": ["none"]}
    assert setup_mismatch(env, base) == []
    dsif = base.replace(support="dsif")
    fields = [m["field"] for m in setup_mismatch(env, dsif)]
    assert fields == ["support"]
    plate = base.replace(support="backing_plate")
    assert {m["field"] for m in setup_mismatch(setup_envelope([plate]), plate.replace(
        support_settings={"clearance": 2e-3}))} == {"support_settings"}
    # a setting that spells out its default is that default, either way round
    spelt = plate.replace(support_settings={"clearance": 1e-3})
    assert setup_mismatch(setup_envelope([plate]), spelt) == []
    assert setup_mismatch(setup_envelope([spelt]), plate) == []
    assert setup_envelope([spelt])["fields"]["support_settings"] == {"values": [{}]}
    assert setup_mismatch(setup_envelope([dsif]), dsif.replace(
        support_settings={"squeeze": 0.0, "rim_pass": False, "thickness_law": "sine"})) == []
    assert setup_mismatch(setup_envelope([dsif]), dsif.replace(
        support_settings={"squeeze": 0.1}))[0]["field"] == "support_settings"
    # a model trained before the field existed was trained without support
    old = json.loads(json.dumps(env))
    del old["fields"]["support"], old["fields"]["support_settings"]
    assert setup_mismatch(old, base) == []
    miss = setup_mismatch(old, dsif)
    assert [m["field"] for m in miss] == ["support"] and "predates" in miss[0]["note"]
    union = setup_envelope_union(old, setup_envelope([dsif]))
    assert union["fields"]["support"] == {"values": ["dsif", "none"]}


def test_the_proxy_refuses_a_setup_with_support(base, cone):
    """The proxy knows single-point forming only; it does not give its
    numbers under a supported setup's name."""
    from ml_support import ProxySimulator

    proxy = ProxySimulator()
    assert proxy.deviation(cone, base).shape == cone.grid.shape
    for s in (base.replace(support="backing_plate"), base.replace(support="dsif")):
        with pytest.raises(PrecompError, match="no model of support"):
            proxy.deviation(cone, s)
        [out] = proxy.run([(s, cone, cone)])
        assert not out.ok and "no model of support" in out.error
        with pytest.raises(PrecompError, match="no model of support"):
            proxy.predict_deviation(cone, s)


def test_a_scan_update_adjusts_the_strip_the_rim_pass_sweeps(tmp_path, base, cone):
    """update_from_scan takes the masks and the bound of displacement
    adjustment, so a measured part gives the command FE-DA would: the swept
    flange strip adjusted, the rim raised where the rim pass reaches."""
    from precomp.compensation import update_from_scan

    rim = base.replace(support="dsif", support_settings={"rim_pass": True, "radius": 1.5e-3})
    pred = _sagging_predictor(cone)
    hold, adjust = compensation_masks(rim, cone)
    bound = command_upper_bound(rim, cone)
    da = displacement_adjustment(cone, pred, iterations=1, hold_mask=hold, adjust_mask=adjust,
                                 upper_bound=bound)
    scan = pred(cone)
    new, _ = update_from_scan(cone, scan, cone, hold_mask=hold, adjust_mask=adjust,
                              upper_bound=bound)
    assert np.array_equal(new.z, da.proposed.z) and new.z.max() > 3e-4
    held, _ = update_from_scan(cone, scan, cone, upper_bound=bound)   # the whole flange held
    fl = flange_mask(cone)
    assert np.array_equal(held.z[fl], cone.z[fl]) and held.z.max() < 0.1 * new.z.max()
