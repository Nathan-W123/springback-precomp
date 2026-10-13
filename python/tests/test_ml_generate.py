"""precomp.ml.generate: design of experiments, variants, simulators, generation."""

import numpy as np
import pytest

from conftest import ML_CREATED_AT, run_count
from ml_support import PROXY_LABEL, ProxySimulator
from precomp._util import PrecompError
from precomp.fea import FormingSetup
from precomp.geometry import Grid, Pyramid, TruncatedCone
from precomp.materials import get_material
from precomp.metrology import flange_mask, part_mask
from precomp.ml.dataset import Dataset
from precomp.ml.generate import (DesignSpace, PerturbationSpec, SimOutcome, SparlabSimulator,
                                 design_points, generate, perturbed_commanded)


def test_the_design_is_deterministic_valid_and_within_bounds():
    space = DesignSpace(families=("truncated_cone", "pyramid", "freeform"))
    a = design_points(space, 4, seed=3)
    b = design_points(space, 4, seed=3)
    assert [p.describe() for p in a] == [p.describe() for p in b]
    assert [p.point_id for p in a] != [p.point_id for p in design_points(space, 4, seed=4)]
    for p in a:
        p.setup.check_part_fits(p.part.footprint_radius())
        assert p.part.max_wall_angle_deg() <= 65.0 + 1e-9
        for key, (lo, hi) in space.process.items():
            assert lo <= getattr(p.setup, key) <= hi
        if p.family == "truncated_cone":
            for key, (lo, hi) in space.bounds_of(p.family).items():
                assert lo <= getattr(p.part, key) <= hi
        assert p.setup.material.name in space.materials
    with pytest.raises(ValueError, match="valid design points"):
        design_points(DesignSpace(families=("truncated_cone",),
                                  base_setup={"blank_size": 0.08, "clamp_margin": 0.01}),
                      1, seed=0, max_draws=64)


def test_the_proxy_has_rim_pillow_and_global_structure_and_says_what_it_is():
    setup = FormingSetup(get_material("AA5754-O"))
    grid = Grid.centered(setup.meshed_blank_size, 2e-3)
    cone = TruncatedCone(0.05, 50.0, 0.025, 0.005, 0.005).heightmap(grid)
    px = ProxySimulator()
    dz = px.deviation(cone, setup)
    j = grid.ny // 2
    row, x = dz[j], grid.x
    rim = np.argmin(np.abs(np.abs(x) - 0.05))
    assert row[rim] > row[np.argmin(np.abs(x - 0.035))]         # under-forming at the rim
    assert row[grid.nx // 2] > row[np.argmin(np.abs(x - 0.02))]  # pillow on the floor
    assert row[0] < 0.3 * row[grid.nx // 2]                      # global bending, zero at supports
    clamped = px.deviation(cone, setup.replace(release="clamped_only"))
    assert abs(clamped[j, 0]) < 1e-12 and clamped[j, grid.nx // 2] < dz[j, grid.nx // 2]
    soft = px.deviation(cone, setup.replace(material=get_material("DC04")))
    assert np.abs(soft).mean() < np.abs(dz).mean()               # lower sigma/E, less springback
    f = px.formed(cone, setup)
    assert f.metadata["label"] == PROXY_LABEL == "proxy - not physics"
    assert px.run([(setup, cone)])[0].provenance["source"] == "proxy"


def test_perturbed_commands_stay_formable_and_hold_the_flange():
    setup = FormingSetup(get_material("AA5754-O"))
    grid = Grid.centered(setup.meshed_blank_size, 3e-3)
    for part in (TruncatedCone(0.05, 64.0, 0.03, 0.004, 0.004),
                 Pyramid(0.055, 0.06, 62.5, 0.0294, 0.044, 0.0034, 0.0049)):
        t = part.heightmap(grid)
        for seed in range(4):
            c, info = perturbed_commanded(t, setup, np.random.default_rng(seed),
                                          PerturbationSpec())
            assert np.degrees(c.wall_angle().max()) <= 65.0 + 1e-9
            fl = flange_mask(t)
            assert np.array_equal(c.z[fl], t.z[fl]) and c.z.max() <= 0
            d = (c.z - t.z)[part_mask(t)]
            assert 0.05e-3 < np.sqrt(np.mean(d ** 2)) < 1.5e-3, info


class _Flaky:
    """A simulator that fails every job whose commanded surface is deeper than a limit."""

    source = "proxy"
    fidelity = "flaky"

    def __init__(self, max_depth):
        self.proxy = ProxySimulator(check_toolpath=False)
        self.max_depth = max_depth
        self.jobs = 0

    def run(self, jobs):
        out = []
        for setup, cmd in jobs:
            self.jobs += 1
            if cmd.depth > self.max_depth:
                out.append(SimOutcome(False, None, "solver diverged: too deep\nstep 2 inc 7",
                                      {"source": "proxy", "fidelity": "flaky"}))
            else:
                out.extend(self.proxy.run([(setup, cmd)]))
        return out


def test_generation_resumes_and_records_every_failure(tmp_path):
    space = DesignSpace(families=("truncated_cone", "dome"), grid_spacing=4e-3,
                        materials=("AA5754-O",), process={})
    points = design_points(space, 3, seed=1)
    depths = sorted(p.part.max_depth() for p in points)
    sim = _Flaky(max_depth=depths[3] + 1e-6)          # the two deepest targets fail
    ds = Dataset.create(tmp_path / "d", created_at=ML_CREATED_AT)
    rep = generate(ds, points, sim, created_at=ML_CREATED_AT, seed=1)
    assert rep.requested == 18
    failed = {f["sample_id"]: f["reason"] for f in rep.failed}
    assert len(rep.created) + len(failed) == 18
    deep = [p for p in points if p.part.max_depth() > depths[3] + 1e-6]
    for p in deep:
        assert failed[f"{p.point_id}-unco"] == "solver diverged: too deep"
        assert failed[f"{p.point_id}-comp"].startswith("dependency failed")
    assert set(ds.failures()["sample_id"]) == set(failed)
    s = ds.load(rep.created[0])
    assert s.source == "proxy" and s.provenance["created_at"] == ML_CREATED_AT
    assert s.provenance["design"]["seed"] == 1 and s.target is not None
    # resume: nothing that succeeded runs again; failed jobs are retried
    jobs = sim.jobs
    rep2 = generate(ds, points, sim, created_at=ML_CREATED_AT, seed=1)
    assert sorted(rep2.skipped_existing) == sorted(rep.created)
    assert sim.jobs - jobs == sum(1 for r in failed.values()
                                  if not r.startswith("dependency failed"))
    assert {f["sample_id"] for f in rep2.failed} == set(failed)


def test_generation_through_sparlab_form_records_the_run(tmp_path, fake_solver, counter):
    base = FormingSetup(get_material("AA5754-O"), blank_size=0.12, clamp_margin=0.01,
                        element_size=2.5e-3, layers=1, step_down=1e-3, toolpath_spacing=2e-3,
                        executable=str(fake_solver))
    doc = base.to_dict()
    doc.pop("material")
    space = DesignSpace(families=("truncated_cone",), grid_spacing=2.5e-3,
                        materials=("AA5754-O",), process={}, base_setup=doc,
                        part_bounds={"truncated_cone": {"top_radius": (0.02, 0.028),
                                                        "depth": (0.008, 0.012),
                                                        "wall_angle_deg": (40.0, 50.0),
                                                        "top_fillet": (0.003, 0.004),
                                                        "bottom_fillet": (0.003, 0.004)}})
    points = design_points(space, 2, seed=0)
    ds = Dataset.create(tmp_path / "d", created_at=ML_CREATED_AT)
    sim = SparlabSimulator(tmp_path / "runs", max_workers=1, executor="serial")
    rep = generate(ds, points, sim, created_at=ML_CREATED_AT,
                   kinds=("uncompensated", "compensated"))
    assert not rep.failed and len(rep.created) == 4 and run_count(counter) == 4
    for sid in rep.created:
        s = ds.load(sid)
        assert s.source == "sim" and s.data_source == "SparLab simulation"
        assert "test double" in s.provenance["sparlab_version"]
        assert len(s.provenance["deck_hash"]) == 64 and s.fidelity.startswith("sparlab:hex8")
        # the test double forms 0.9 x the commanded surface at the nodes
        np.testing.assert_allclose(s.formed.z, 0.9 * s.commanded.z, atol=1e-12)
    comp = ds.load(f"{points[0].point_id}-comp")
    t = comp.target
    sel = part_mask(t)
    np.testing.assert_allclose(comp.commanded.z[sel], 1.1 * t.z[sel], atol=1e-9)
    generate(ds, points, sim, created_at=ML_CREATED_AT, kinds=("uncompensated", "compensated"))
    assert run_count(counter) == 4                              # resumed, nothing re-run


def test_a_design_point_id_names_one_part_whatever_else_is_drawn(tmp_path):
    """Families are seeded by name and perturbations by point id: a point is
    the same with other families beside it and with more points per family,
    and a resumed run that would put another part under an id is refused."""
    kw = dict(grid_spacing=4e-3, materials=("AA5754-O",), process={})
    both = {p.point_id: p for p in design_points(
        DesignSpace(families=("truncated_cone", "dome"), **kw), 2, seed=1)}
    alone = {p.point_id: p for p in design_points(DesignSpace(families=("dome",), **kw), 3,
                                                  seed=1)}
    shared = sorted(set(both) & set(alone))
    assert shared == ["dome-s1-0000", "dome-s1-0001"]
    assert all(both[i].describe() == alone[i].describe() for i in shared)
    sim = ProxySimulator(check_toolpath=False)
    a = Dataset.create(tmp_path / "a", created_at=ML_CREATED_AT)
    b = Dataset.create(tmp_path / "b", created_at=ML_CREATED_AT)
    generate(a, list(both.values()), sim, created_at=ML_CREATED_AT, kinds=("perturbed",),
             seed=1)
    generate(b, [alone["dome-s1-0001"]], sim, created_at=ML_CREATED_AT, kinds=("perturbed",),
             seed=1)
    assert np.array_equal(a.load("dome-s1-0001-pert").commanded.z,
                          b.load("dome-s1-0001-pert").commanded.z)
    # the same ids from another design (another thickness): refused, nothing run
    other = design_points(DesignSpace(families=("dome",), base_setup={"thickness": 1.2e-3},
                                      **kw), 2, seed=1)
    n = len(a)
    with pytest.raises(PrecompError, match="another design point .*setup"):
        generate(a, other, sim, created_at=ML_CREATED_AT, kinds=("perturbed",), seed=1)
    assert len(a) == n


def test_the_proxy_fails_a_job_without_a_tool_path_as_the_deck_would(tmp_path):
    kw = dict(families=("pyramid",), grid_spacing=4e-3, materials=("AA5754-O",), process={})
    points = design_points(DesignSpace(**kw), 1, seed=1)
    ds = Dataset.create(tmp_path / "d", created_at=ML_CREATED_AT)
    rep = generate(ds, points, ProxySimulator(), created_at=ML_CREATED_AT,
                   kinds=("uncompensated", "perturbed"), seed=1)
    assert rep.created == ["pyramid-s1-0000-unco"]
    assert rep.failed[0]["sample_id"] == "pyramid-s1-0000-pert"
    assert "no spiral tool path" in rep.failed[0]["reason"]
    contour = design_points(DesignSpace(base_setup={"toolpath_style": "contour"}, **kw), 1,
                            seed=1)
    rep = generate(Dataset.create(tmp_path / "e", created_at=ML_CREATED_AT), contour,
                   ProxySimulator(), created_at=ML_CREATED_AT, kinds=("perturbed",), seed=1)
    assert not rep.failed


def test_the_element_formulation_and_thickness_points_are_recorded_physics():
    """A model's setup envelope records the Hex8 formulation and the points
    through the thickness like any physics field: a model trained on the 2 mm
    benchmark mesh refuses the fine incompatible-mode setup, and one trained
    before the fields existed was trained at their defaults."""
    import json

    from precomp.ml.generate import SparlabSimulator
    from precomp.ml.surrogate import setup_envelope, setup_mismatch

    coarse = FormingSetup.preset("springback_2mm")
    fine = FormingSetup.preset("springback_fine")
    env = setup_envelope([coarse])
    assert env["fields"]["element_formulation"] == {"values": ["standard"]}
    assert env["fields"]["thickness_points"] == {"min": 0, "max": 0}
    assert {m["field"] for m in setup_mismatch(env, fine)} == {
        "element_size", "layers", "element_formulation", "thickness_points"}
    assert setup_mismatch(setup_envelope([fine]), fine) == []
    assert [m["field"] for m in setup_mismatch(setup_envelope([fine]),
                                               FormingSetup.preset("springback_fine_bulk"))] \
        == ["contact"]
    old = json.loads(json.dumps(env))
    del old["fields"]["element_formulation"], old["fields"]["thickness_points"]
    assert setup_mismatch(old, coarse) == []
    miss = setup_mismatch(old, coarse.replace(thickness_points=5))
    assert [m["field"] for m in miss] == ["thickness_points"] and "predates" in miss[0]["note"]
    # the simulator's fidelity label tells the meshes apart (and is unchanged
    # for the standard element)
    assert SparlabSimulator.fidelity_of(coarse) == \
        "sparlab:hex8:2mm:2L:finite_logarithmic:321"
    assert SparlabSimulator.fidelity_of(fine) == \
        "sparlab:hex8-im-tp5:0.833333mm:1L:finite_logarithmic:321"


def test_the_compensated_variant_takes_da_options(tmp_path):
    """`da_options` reaches the DA step of the compensated variant: here the
    tool's reach, which the stored command then satisfies."""
    from precomp.toolpath import tool_reach_surface

    space = DesignSpace(families=("truncated_cone",), grid_spacing=1e-3,
                        materials=("AA5754-O",), process={})
    points = design_points(space, 2, seed=3)
    seen = []

    def opts(p):
        seen.append(p.point_id)
        return {"tool_radius": p.setup.tool_radius}

    ds = Dataset.create(tmp_path / "d", created_at=ML_CREATED_AT)
    rep = generate(ds, points, ProxySimulator(), created_at=ML_CREATED_AT, seed=3,
                   kinds=("uncompensated", "compensated"), da_options=opts)
    assert not rep.failed and sorted(seen) == sorted(p.point_id for p in points)
    for p in points:
        c = ds.load(f"{p.point_id}-comp").commanded
        assert c.metadata["commanded"]["tool_reach_radius_m"] == p.setup.tool_radius
        np.testing.assert_allclose(tool_reach_surface(c, p.setup.tool_radius).z, c.z,
                                   atol=1e-9)
