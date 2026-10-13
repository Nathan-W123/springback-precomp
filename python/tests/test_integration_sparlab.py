"""Integration with the real sparlab_form, skipped when it has not been built.

* Every deck variant precomp writes passes `sparlab_form --strict-config`
  (the run is stopped once the deck is parsed and the analysis built).
* The test double (fake_sparlab_form.py) writes what sparlab_form writes:
  the same files, CSV columns and summary keys, on the same deck.
* A tiny SPIF runs end to end through `precomp.fea.simulate` and its result
  is physically sane: depth, the sign of the springback, the sign and the
  order of magnitude of the tool force. (The physics itself is verified by
  the C++ suite, `tests/test_forming.cpp`.)
* The small cone of the original contract test.

Each simulation takes seconds to two minutes on one thread.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import REPO_ROOT
from precomp.fea import EXECUTABLE_ENV, FormingSetup, build_deck, load_result, simulate
from precomp.geometry import Grid, TruncatedCone
from precomp.materials import get_material
from precomp.toolpath import spiral_toolpath

EXE = os.environ.get(EXECUTABLE_ENV) or str(REPO_ROOT / "build" / "bin" / "sparlab_form")

pytestmark = pytest.mark.skipif(not (os.path.isfile(EXE) and os.access(EXE, os.X_OK)),
                                reason=f"{EXE} is not built")


def deck_accepted(deck_dir: Path, out: Path, timeout: float = 120.0) -> None:
    """Start sparlab_form --strict-config on `deck_dir/deck.json` and stop it
    once mesh.json exists: it writes that after parsing the deck strictly and
    building the model and the analysis (tools and their surfaces, every
    step's constraints against the rigid-body motions, the step windows), each
    of which exits with 2 on a refusal. Fails with the solver's message."""
    proc = subprocess.Popen([EXE, "--config", str(deck_dir / "deck.json"), "--output", str(out),
                             "--strict-config", "--threads", "1"], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    start = time.monotonic()
    try:
        while proc.poll() is None and not (out / "mesh.json").is_file():
            if time.monotonic() - start > timeout:
                pytest.fail(f"sparlab_form took over {timeout} s to start {deck_dir}")
            time.sleep(0.05)
    finally:
        if proc.poll() is None:
            proc.kill()
        log = proc.communicate()[0]
    assert proc.returncode in (0, -9), f"{deck_dir}: exit {proc.returncode}\n{log}"
    assert (out / "config.json").is_file() and (out / "mesh.json").is_file(), log


def test_every_deck_precomp_writes_passes_strict_config(tmp_path):
    base = FormingSetup(get_material("AA5754-O"), blank_size=0.04, clamp_margin=0.005,
                        element_size=2.5e-3, layers=2, step_down=1e-3, toolpath_spacing=2e-3,
                        executable=EXE)
    af = get_material("DC04").replace(af_C=8e9, af_gamma=60.0, kinematic_hardening_modulus=1e8)
    variants = {
        "hill48_log": base,                                     # the defaults
        "von_mises": base.replace(material=get_material("Ti-6Al-4V-annealed")),
        "chaboche": base.replace(material=af),
        "finite": base.replace(kinematics="finite"),
        "small_strain": base.replace(kinematics="small_strain"),
        "contact_solver": base.replace(
            contact={"penalty": 20.0, "tangential_penalty": 0.5},
            solver={"max_iterations": 40, "residual_tolerance": 1e-7,
                    "displacement_tolerance": 1e-7, "line_search": True, "max_cuts": 8,
                    "max_increments": 10000, "friction_tangent": "symmetric",
                    "solver": "eigen", "mean_dilatation": "all"}),
        "clamped_only": base.replace(release="clamped_only"),
        "frictionless": base.replace(friction=0.0),
        "incompatible_modes_tp5": base.replace(layers=1, element_formulation="incompatible_modes",
                                               thickness_points=5),
        "standard_tp5": base.replace(thickness_points=5),
        "fine_dsif": FormingSetup.preset("springback_fine", executable=EXE, support="dsif"),
        "backing_plate": base.replace(support="backing_plate",
                                      support_settings={"clearance": 1e-3, "friction": 0.05}),
        "dsif": base.replace(support="dsif", support_settings={"squeeze": 0.1}),
        "dsif_rim_pass": base.replace(support="dsif", support_settings={"rim_pass": True}),
        "dsif_clamped_only": base.replace(support="dsif", release="clamped_only",
                                          support_settings={"rim_pass": True}),
    }
    target = TruncatedCone(0.01, 45.0, 0.002, 0.002, 0.002).heightmap(Grid.centered(0.04, 5e-4))
    for name, setup in variants.items():
        deck = build_deck(setup, target, tmp_path / name, target=target)
        deck_accepted(deck, tmp_path / f"{name}_out")
    # and a key it does not read is refused, naming it
    doc = json.loads((tmp_path / "hill48_log" / "deck.json").read_text())
    doc["forming"]["tools"][0]["contact"] = {}
    (tmp_path / "hill48_log" / "deck.json").write_text(json.dumps(doc))
    proc = subprocess.run([EXE, "--config", str(tmp_path / "hill48_log" / "deck.json"),
                           "--output", str(tmp_path / "refused"), "--strict-config"],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 2 and "forming.tools[0].contact" in proc.stderr


# ---------------------------------------------------------------------------
# A tiny SPIF: 20 x 20 x 1 mm blank, 8 x 8 x 2 Hex8, a 4 mm tool, one
# revolution of a spiral contour down to 1 mm, unload, release onto 3-2-1
# ---------------------------------------------------------------------------
TINY_DEPTH = 1e-3


@pytest.fixture(scope="module")
def tiny_spif(tmp_path_factory):
    """(setup, target, result) of the tiny SPIF, run once for the module.

    The path is one revolution of a spiral that descends to the full depth
    and ends there, in contact (no closing loop at the bottom), so the
    "unload" step removes a tool that is pressing on the sheet."""
    setup = FormingSetup(get_material("AA5754-O"), blank_size=0.02, clamp_margin=2.5e-3,
                         element_size=2.5e-3, layers=2, thickness=1e-3, tool_radius=4e-3,
                         step_down=1e-3, toolpath_style="spiral", toolpath_spacing=1e-3,
                         max_tool_travel=5e-4, friction=0.1, executable=EXE, timeout=600.0)
    target = TruncatedCone(0.006, 45.0, TINY_DEPTH, 1e-3, 1e-3).heightmap(
        Grid.centered(0.02, 2.5e-4))
    path = spiral_toolpath(target, setup.tool_radius, setup.step_down,
                           setup.toolpath_spacing, final_loop=False)
    work = tmp_path_factory.mktemp("tiny_spif")
    start = time.perf_counter()
    res = simulate(setup, target, work, toolpath=path)
    res.provenance["wall_s"] = time.perf_counter() - start
    return setup, target, res


def test_tiny_spif_is_physically_sane(tiny_spif):
    setup, target, res = tiny_spif
    assert res.provenance["wall_s"] < 60.0
    assert res.completed and res.termination == "completed every step"
    assert res.step_names == ["form", "unload", "release"]
    assert res.summary["analysis"]["kinematics"] == "finite_logarithmic"
    assert res.summary["mesh"]["nodes"] == 9 * 9 * 3
    steps = {s["name"]: s for s in res.summary["steps"]}

    # Depth: the formed tool-side surface reaches about the tool tip's 1 mm
    # (penalty contact, elastic recovery and the coarse mesh within 30 %),
    # in every step; the flange stays put.
    X, Y = target.grid.mesh()
    clamp = np.maximum(np.abs(X), np.abs(Y)) >= setup.free_half_width - 1e-9
    formed = {name: res.formed_surface(name, grid=target.grid) for name in res.step_names}
    for name, f in formed.items():
        assert f.mask[~clamp].all()          # (the blank's edge may move inside the grid)
        assert 0.7 * TINY_DEPTH < f.depth < 1.3 * TINY_DEPTH, (name, f.depth)
    assert np.abs(formed["form"].z[clamp]).max() < 1e-9            # clamped
    assert np.abs(formed["release"].z[clamp]).max() < 0.1 * TINY_DEPTH

    # Tool force: the force the sheet exerts on the tool - upwards, every
    # increment, and of the order of a kilonewton for 1 mm AA5754-O under a
    # 4 mm tool; mostly vertical.
    forces = res.forming_forces("tool")
    assert (forces["step"] == 1).all()                             # the form step only
    touching = forces[forces["active_nodes"] > 0]
    assert len(touching) > 0.5 * len(forces)
    assert (forces["fz"] >= 0).all() and (touching["fz"] > 0).all()
    assert 100.0 < touching["fz"].max() < 5000.0
    assert (np.hypot(touching["fx"], touching["fy"]) < touching["fz"]).all()
    assert (touching["max_penetration_m"] < 1e-5).all()            # penalty contact
    last = forces.iloc[-1]
    assert last["active_nodes"] > 0 and last["fz"] > 100.0          # in contact at the end

    # Springback sign: removing the pressing tool ("unload", clamp held) lets
    # the sheet under it come back up towards the tool; the tool's force is
    # the imbalance the step ramps out.
    assert steps["unload"]["start_imbalance_N"] > 0.5 * last["fz"] / np.sqrt(3)
    form, unload = res.step("form"), res.step("unload")
    top = np.abs(form.reference[:, 2]) < 1e-12
    xy = form.reference[top, :2]
    near = np.argsort(np.hypot(xy[:, 0] - last["cx"], xy[:, 1] - last["cy"]))[:3]
    rise = unload.current[top][near, 2] - form.current[top][near, 2]
    assert (rise > 1e-6).all() and (rise < 0.2 * TINY_DEPTH).all(), rise
    # The release onto 3-2-1 supports leaves no reaction: the part is free.
    assert steps["release"]["reaction_norm_N"] < 1e-6 * steps["release"]["reference_force_N"]
    assert 0.0 < steps["release"]["max_displacement_change_m"] < 0.2 * TINY_DEPTH


def _key_tree(doc, dynamic=("factor_",)):
    """The key structure of a JSON document: dicts by key (keys starting with
    one of `dynamic` - the factorisations that ran - dropped), lists of dicts
    by the union of their items' keys, leaves by type."""
    if isinstance(doc, dict):
        return {k: _key_tree(v) for k, v in doc.items() if not k.startswith(dynamic)}
    if isinstance(doc, list) and doc and all(isinstance(x, dict) for x in doc):
        merged = {}
        for item in doc:
            merged.update(_key_tree(item))
        return [merged]
    if isinstance(doc, (int, float)) and not isinstance(doc, bool):
        return "number"
    return type(doc).__name__


def test_the_test_double_writes_what_sparlab_form_writes(tiny_spif, tmp_path):
    """fake_sparlab_form.py on the tiny SPIF's deck: the same files in the
    same order, the same CSV columns, the same summary and mesh.json keys, the
    same mesh and step windows (the numbers are the double's own)."""
    _, _, real = tiny_spif
    deck_dir = real.directory.parent
    fake_out = tmp_path / "fake"
    script = Path(__file__).with_name("fake_sparlab_form.py")
    proc = subprocess.run([sys.executable, str(script), "--config", str(deck_dir / "deck.json"),
                           "--output", str(fake_out), "--strict-config"],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    fake = load_result(fake_out)
    rs, fs = real.summary, fake.summary
    assert fs["files"] == rs["files"]
    assert sorted(p.name for p in fake_out.iterdir()) == sorted(
        p.name for p in real.directory.iterdir())
    for name in rs["files"]:
        if name.endswith(".csv"):
            assert (pd.read_csv(fake_out / name, nrows=0).columns.tolist()
                    == pd.read_csv(real.directory / name, nrows=0).columns.tolist()), name
    assert _key_tree(fs) == _key_tree(rs)
    for key in ("case", "completed", "termination"):
        assert fs[key] == rs[key]
    assert fs["mesh"]["source"] == rs["mesh"]["source"]
    assert [s["files_stem"] for s in fs["steps"]] == [s["files_stem"] for s in rs["steps"]]
    assert [s["termination"] for s in fs["steps"]] == [s["termination"] for s in rs["steps"]]
    assert np.allclose([(s["t_begin_s"], s["t_end_s"]) for s in fs["steps"]],
                       [(s["t_begin_s"], s["t_end_s"]) for s in rs["steps"]])
    mr = json.loads((real.directory / "mesh.json").read_text())
    mf = json.loads((fake_out / "mesh.json").read_text())
    assert set(mf) == set(mr) and mf["elements"] == mr["elements"]
    assert np.allclose(mf["nodes_m"], mr["nodes_m"], rtol=0, atol=1e-15)
    assert np.allclose(fake.step(0).reference, real.step(0).reference, rtol=0, atol=1e-15)
    rf, ff = real.forming_forces(), fake.forming_forces()
    assert rf.columns.tolist() == ff.columns.tolist()
    assert (ff["fz"] >= 0).all() and ff["step"].min() == rf["step"].min() == 1


def test_small_cone_runs_through_sparlab_form(tmp_path):
    setup = FormingSetup(get_material("AA5754-O"), blank_size=0.08, clamp_margin=0.01,
                         element_size=4e-3, layers=1, step_down=2e-3, toolpath_spacing=4e-3,
                         max_tool_travel=2e-3, executable=EXE, timeout=1800.0)
    target = TruncatedCone(0.02, 40.0, 0.004, 0.003, 0.003).heightmap(Grid.centered(0.08, 1e-3))
    res = simulate(setup, target, tmp_path / "w")
    assert res.step_names == ["form", "unload", "release"] and res.completed
    formed = res.formed_surface(grid=target.grid)
    assert formed.mask.mean() > 0.9
    assert 0.0 < formed.depth < 2 * target.depth
    f = res.forming_forces()
    assert np.all(np.isfinite(f["f"])) and (f["fz"] >= 0).all()


# ---------------------------------------------------------------------------
# Support from below: the tiny SPIF's part on a backing plate and with a DSIF
# support tool, each run end to end
# ---------------------------------------------------------------------------
def _tiny_supported(tmp_path, support, settings):
    setup = FormingSetup(get_material("AA5754-O"), blank_size=0.024, clamp_margin=2.5e-3,
                         element_size=2.0e-3, layers=2, thickness=1e-3, tool_radius=3e-3,
                         step_down=1e-3, toolpath_style="spiral", toolpath_spacing=1e-3,
                         max_tool_travel=5e-4, friction=0.1, executable=EXE, timeout=900.0,
                         support=support, support_settings=settings)
    target = TruncatedCone(0.005, 45.0, TINY_DEPTH, 1e-3, 1e-3).heightmap(
        Grid.centered(0.024, 2.5e-4))
    path = spiral_toolpath(target, setup.tool_radius, setup.step_down,
                           setup.toolpath_spacing, final_loop=False)
    start = time.perf_counter()
    res = simulate(setup, target, tmp_path / "w", toolpath=path, target=target)
    return setup, target, res, time.perf_counter() - start


def test_a_backing_plate_holds_the_sheet_up_and_comes_off_before_the_release(tmp_path):
    setup, target, res, wall = _tiny_supported(tmp_path, "backing_plate", {"clearance": 5e-4})
    assert wall < 180.0
    assert res.completed and res.step_names == ["form", "unload", "release"]
    f = res.forming_forces()
    plate = f[f["tool"] == "plate"]
    tool = f[f["tool"] == "tool"]
    assert (plate["step"] == 1).all() and len(plate) == len(tool)   # "form" only
    # the sheet pushes the plate down (the plate holds it up), never pulls it
    assert (plate["fz"] <= 1e-9).all() and plate["fz"].min() < -1.0
    assert (plate["max_penetration_m"] < 1e-5).all()
    # the plate carries only nodes outside the opening, and none sinks below it
    deck = json.loads((res.directory / "config.json").read_text())
    ids = deck["forming"]["tools"][1]["surface"]["node_ids"]
    form = res.step("form")
    assert np.allclose(form.reference[ids, 2], -setup.thickness)
    assert form.current[ids, 2].min() > -setup.thickness - 1e-5
    # in the solver's own mesh the ids are the bottom nodes precomp meant (the
    # structured numbering), each farther than the clearance outside the outline
    from precomp.fea.support import bottom_node_grid, outline_distance, plate_nodes

    assert ids == plate_nodes(setup, target)[0]
    _, X, Y = bottom_node_grid(setup)
    assert np.allclose(form.reference[ids, 0], X.ravel()[ids], rtol=0.0, atol=1e-12)
    assert np.allclose(form.reference[ids, 1], Y.ravel()[ids], rtol=0.0, atol=1e-12)
    assert outline_distance(target, *form.reference[ids, :2].T).min() > 5e-4
    steps = {s["name"]: s for s in res.summary["steps"]}
    assert steps["release"]["reaction_norm_N"] < 1e-6 * steps["release"]["reference_force_N"]


def test_a_dsif_support_pushes_up_opposite_the_tool(tmp_path):
    setup, target, res, wall = _tiny_supported(tmp_path, "dsif", {"rim_pass": True})
    assert wall < 300.0
    assert res.completed
    assert res.step_names == ["form", "unload", "rim_pass", "rim_unload", "release"]
    f = res.forming_forces()
    tool, sup = f[f["tool"] == "tool"], f[f["tool"] == "support"]
    assert set(tool["step"]) == {1} and set(sup["step"]) == {1, 3}
    # the support touches the sheet (on this 1 mm deep part it sits under
    # the still flat sheet beside the tool most of the time: a light touch)
    both = sup[(sup["step"] == 1) & (sup["active_nodes"] > 0)]
    assert len(both) > 0 and both["fz"].min() < -1.0
    assert (sup["fz"] <= 1e-9).all() and (tool["fz"] >= -1e-9).all()   # squeezed from both sides
    assert (f["max_penetration_m"] < 1e-5).all()
    rim = sup[sup["step"] == 3]
    assert (rim["active_nodes"] > 0).any() and rim["fz"].min() < -1.0   # the rim pass pushes up
    steps = {s["name"]: s for s in res.summary["steps"]}
    assert steps["release"]["reaction_norm_N"] < 1e-6 * steps["release"]["reference_force_N"]
    formed = res.formed_surface(grid=target.grid)
    assert 0.5 * TINY_DEPTH < formed.depth < 1.5 * TINY_DEPTH
