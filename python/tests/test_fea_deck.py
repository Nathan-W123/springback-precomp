"""The sparlab_form deck: structure, supports, trajectory file and content hash."""

import json

import numpy as np
import pandas as pd
import pytest

from precomp import PrecompError
from precomp.fea import (EXECUTABLE_ENV, FormingSetup, build_deck, deck_document, deck_hash,
                         support_321)
from precomp.fea.deck import check_toolpath, make_toolpath
from precomp.geometry import Grid, TruncatedCone
from precomp.materials import get_material


def test_deck_document_matches_the_contract(small_setup):
    """The keys of docs/forming.md, section 2 (the real executable checks them
    with --strict-config in test_integration_sparlab.py)."""
    doc = deck_document(small_setup)
    assert set(doc) == {"name", "description", "units", "mesh", "material", "model", "forming"}
    m = doc["mesh"]
    assert m["type"] == "structured_hex"
    assert (m["nx"], m["ny"], m["nz"]) == (48, 48, 1)
    assert m["lx"] == pytest.approx(0.12) and m["x0"] == pytest.approx(-0.06)
    assert m["z0"] == -small_setup.thickness and m["lz"] == small_setup.thickness
    assert doc["material"] == small_setup.material.to_sparlab()
    fm = doc["forming"]
    assert fm["kinematics"] == "finite_logarithmic"            # the large-strain default
    assert set(fm) == {"kinematics", "tools", "steps", "output"}
    tool = fm["tools"][0]
    assert set(tool) == {"name", "shape", "radius", "surface", "friction", "trajectory"}
    assert tool["shape"] == "sphere" and tool["radius"] == small_setup.tool_radius
    assert tool["trajectory"] == {"file": "toolpath.csv"}
    assert tool["surface"]["box"]["zmin"] == 0.0
    steps = fm["steps"]
    assert [s["name"] for s in steps] == ["form", "unload", "release"]
    assert [s["type"] for s in steps] == ["form", "release", "release"]
    assert steps[0]["tools"] == ["tool"] and steps[1]["tools"] == []
    assert steps[0]["time"] == [0.0, 1.0]
    assert steps[0]["max_tool_travel"] == small_setup.max_tool_travel
    for s in steps:
        assert "increments" not in s
        for bc in s["boundary_conditions"]:
            assert bc["mode"] == "hold" and "value" not in bc
    assert steps[0]["boundary_conditions"][0]["fix"] == ["x", "y", "z"]
    json.dumps(doc, allow_nan=False)


def test_contact_and_solver_settings_go_where_sparlab_form_reads_them(small_setup):
    s = small_setup.replace(contact={"penalty": 20.0, "tangential_penalty": 0.5},
                            solver={"max_iterations": 40, "residual_tolerance": 1e-7,
                                    "friction_tangent": "symmetric", "solver": "eigen"},
                            kinematics="finite")
    fm = deck_document(s)["forming"]
    tool = fm["tools"][0]
    assert tool["penalty"] == 20.0 and tool["tangential_penalty"] == 0.5
    assert fm["newton"] == {"max_iterations": 40, "residual_tolerance": 1e-7}
    assert fm["friction_tangent"] == "symmetric" and fm["solver"] == "eigen"
    assert fm["kinematics"] == "finite"
    with pytest.raises(ValueError, match="contact: unknown keys"):
        small_setup.replace(contact={"stiffness": 1e12})
    with pytest.raises(ValueError, match="solver: unknown keys"):
        small_setup.replace(solver={"steps": 10})
    with pytest.raises(ValueError, match="penalty"):
        small_setup.replace(contact={"penalty": 0.0})
    with pytest.raises(ValueError, match="kinematics"):
        small_setup.replace(kinematics="neo_hookean")


def test_321_support_removes_exactly_the_six_rigid_modes(small_setup):
    bcs = support_321(small_setup)
    assert sum(len(b["fix"]) for b in bcs) == 6
    p = [np.array(b["region"]["nearest_node"]) for b in bcs]
    assert bcs[0]["fix"] == ["x", "y", "z"] and bcs[1]["fix"] == ["y", "z"] and \
        bcs[2]["fix"] == ["z"]
    # the y-fixed pair lies along x (stops the rotation about z), the three z-fixed
    # points are not collinear (stop the rotations about x and y)
    assert p[1][1] == p[0][1] and p[1][0] > p[0][0]
    assert abs(np.cross(p[1] - p[0], p[2] - p[0])[2]) > 0
    clamped = FormingSetup(small_setup.material, blank_size=0.12, clamp_margin=0.01,
                           element_size=2.5e-3, release="clamped_only")
    assert [s["name"] for s in deck_document(clamped)["forming"]["steps"]] == ["form", "unload"]


def test_build_deck_writes_the_files_and_the_hash_tracks_the_physics(tmp_path, small_setup,
                                                                       small_cone):
    cmd = small_cone.heightmap(Grid.centered(0.12, 1e-3))
    d1 = build_deck(small_setup, cmd, tmp_path / "a")
    for name in ("deck.json", "toolpath.csv", "commanded.npz", "precomp_deck.json"):
        assert (d1 / name).is_file()
    traj = pd.read_csv(d1 / "toolpath.csv")
    assert list(traj.columns) == ["t", "x", "y", "z"]
    assert np.all(np.diff(traj["t"]) > 0) and traj["t"].iloc[-1] == 1.0
    h1 = deck_hash(d1, "v1")
    d2 = build_deck(small_setup.replace(timeout=5.0, threads=4), cmd, tmp_path / "b")
    assert deck_hash(d2, "v1") == h1                   # execution settings do not count
    assert deck_hash(d1, "v2") != h1                   # the solver build does
    d3 = build_deck(small_setup.replace(friction=0.2), cmd, tmp_path / "c")
    assert deck_hash(d3, "v1") != h1                   # the physics does
    d4 = build_deck(small_setup, cmd.with_z(cmd.z * 1.01), tmp_path / "d")
    assert deck_hash(d4, "v1") != h1                   # and so does the commanded shape


def test_a_path_reaching_the_clamp_is_refused(small_setup):
    big = TruncatedCone(0.05, 45.0, 0.01, 0.003, 0.003)
    cmd = big.heightmap(Grid.centered(0.14, 1e-3))
    path = make_toolpath(small_setup, cmd)
    with pytest.raises(PrecompError, match="unclamped"):
        check_toolpath(small_setup, path)


def test_setup_round_trip_validation_and_executable_resolution(tmp_path, monkeypatch):
    s = FormingSetup("DC04", thickness=0.8e-3, contact={"penalty": 20.0})
    back = FormingSetup.from_dict(json.loads(json.dumps(s.to_dict())))
    assert back == s and back.material == get_material("DC04")
    assert "executable" not in s.physics_dict()
    with pytest.raises(ValueError):
        FormingSetup("DC04", element="tet10")
    with pytest.raises(ValueError):
        FormingSetup("DC04", clamp_margin=1e-3, element_size=2.5e-3)
    with pytest.raises(ValueError, match="unknown keys"):
        FormingSetup.from_dict({"material": "DC04", "colour": 1})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(EXECUTABLE_ENV, str(tmp_path / "nope"))
    with pytest.raises(PrecompError, match="not found"):
        s.resolved_executable()
    exe = tmp_path / "sf"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    monkeypatch.setenv(EXECUTABLE_ENV, str(exe))
    assert s.resolved_executable() == exe.resolve()
    assert s.replace(executable=str(exe)).resolved_executable() == exe.resolve()


def test_element_formulation_and_thickness_points_go_into_the_model_block(small_setup):
    """The Hex8 formulation and the points through the thickness are written
    into `model` only when set, so a setup at the defaults keeps its deck (and
    its content hash) of before these fields existed."""
    assert deck_document(small_setup)["model"] == {"stress_state": "three_dimensional"}
    s = small_setup.replace(layers=1, element_formulation="incompatible_modes",
                            thickness_points=5)
    assert deck_document(s)["model"] == {
        "stress_state": "three_dimensional", "element_formulation": "incompatible_modes",
        "integration": {"thickness_points": 5, "thickness_direction": "z"}}
    assert deck_document(small_setup.replace(thickness_points=7))["model"]["integration"] == \
        {"thickness_points": 7, "thickness_direction": "z"}
    for bad, match in [({"element_formulation": "reduced"}, "element_formulation"),
                       ({"element": "tet4"}, "element must be 'hex8'"),
                       ({"toolpath_style": "contour"}, "toolpath_style must be 'spiral'"),
                       ({"thickness_points": 8}, "thickness_points"),
                       ({"thickness_points": 2.0}, "thickness_points"),
                       ({"element_formulation": "incompatible_modes", "thickness_points": 1},
                        "thickness_points 1"),
                       ({"element_formulation": "incompatible_modes",
                         "solver": {"mean_dilatation": "all"}}, "mean_dilatation")]:
        with pytest.raises(ValueError, match=match):
            small_setup.replace(**bad)


def test_element_formulation_and_thickness_points_change_the_content_hash(tmp_path, small_setup,
                                                                          small_cone):
    s0 = small_setup.replace(layers=1)
    cmd = small_cone.heightmap(Grid.centered(0.12, 1e-3))
    hashes = set()
    for name, s in {"std": s0, "im": s0.replace(element_formulation="incompatible_modes"),
                    "im5": s0.replace(element_formulation="incompatible_modes",
                                      thickness_points=5),
                    "im7": s0.replace(element_formulation="incompatible_modes",
                                      thickness_points=7)}.items():
        hashes.add(deck_hash(build_deck(s, cmd, tmp_path / name), "sparlab 1"))
    assert len(hashes) == 4
    d = FormingSetup.from_dict(s0.replace(element_formulation="incompatible_modes",
                                          thickness_points=5).to_dict())
    assert (d.element_formulation, d.thickness_points) == ("incompatible_modes", 5)
    # a setup recorded before the fields existed reads back at the defaults
    old = s0.to_dict()
    del old["element_formulation"], old["thickness_points"]
    assert FormingSetup.from_dict(old) == s0


def test_presets_are_the_benchmark_and_the_audit_setups():
    """`springback_2mm` is benchmarks/springback's solver setup field for
    field; `springback_fine` the physics audit's recommended 0.833 mm IM5 mesh
    on the same process (its convergence deck h0.833_L1_im_tp5)."""
    from pathlib import Path

    from precomp.fea import PRESETS

    repo = Path(__file__).resolve().parents[2]
    old = json.loads((repo / "benchmarks" / "springback" / "config.json").read_text())
    s2 = FormingSetup.preset("springback_2mm")
    assert s2 == FormingSetup.from_dict(old["solver"]["setup"])
    fine = FormingSetup.preset("springback_fine")
    assert fine.elements_per_side == 48 and fine.meshed_blank_size == pytest.approx(0.04)
    assert (fine.layers, fine.element_formulation, fine.thickness_points) == \
        (1, "incompatible_modes", 5)
    assert fine.contact == {}                   # penalty 10, the default
    assert fine.replace(element_size=2e-3, layers=2, element_formulation="standard",
                        thickness_points=0) == s2
    bulk = FormingSetup.preset("springback_fine_bulk")
    assert bulk == fine.replace(contact={"penalty": 3.0})
    audit = repo / "benchmarks/physics_audit/convergence/cases/h0.833_L1_im_tp5/deck/deck.json"
    if audit.is_file():
        doc = json.loads(audit.read_text())
        mine = deck_document(fine)
        assert mine["mesh"] == pytest.approx(doc["mesh"])
        assert mine["model"] == doc["model"]
    # overrides go on top; a preset is never changed by its use
    d = FormingSetup.preset("springback_fine", support="dsif", threads=2)
    assert d.support == "dsif" and d.threads == 2
    assert "support" not in PRESETS["springback_fine"]
    with pytest.raises(ValueError, match="unknown setup preset"):
        FormingSetup.preset("fine")
