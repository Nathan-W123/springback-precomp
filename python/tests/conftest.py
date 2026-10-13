"""Shared fixtures of the precomp test suite.

The package is imported from the source tree (python/), so the tests run
without installing it. `fake_solver` is an executable wrapper around
`fake_sparlab_form.py`, a test double that writes contract-conforming result
directories; the real sparlab_form is only used by test_integration_sparlab.py,
which is skipped when build/bin/sparlab_form is absent.

The precomp.ml tests share one proxy data set and one trained surrogate
(session fixtures below). Every number they produce comes from the
ProxySimulator - an analytic stand-in, NOT physics. Threads are capped at two
(the machine is shared).
"""

import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

PYTHON_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PYTHON_DIR.parent
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from precomp.geometry import Grid, TruncatedCone  # noqa: E402


@pytest.fixture(scope="session")
def fake_solver(tmp_path_factory) -> Path:
    """Path of an executable that behaves like sparlab_form (see fake_sparlab_form.py)."""
    d = tmp_path_factory.mktemp("fake_bin")
    exe = d / "sparlab_form"
    script = Path(__file__).with_name("fake_sparlab_form.py")
    exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return exe


@pytest.fixture
def counter(tmp_path, monkeypatch) -> Path:
    """A file the fake solver appends one line to per run."""
    path = tmp_path / "runs.log"
    monkeypatch.setenv("FAKE_SPARLAB_COUNTER", str(path))
    monkeypatch.delenv("FAKE_SPARLAB_FAIL", raising=False)
    return path


def run_count(path: Path) -> int:
    return len(path.read_text().splitlines()) if path.exists() else 0


@pytest.fixture
def small_setup(fake_solver):
    """A 0.12 m blank with 2.5 mm elements, run by the fake solver."""
    from precomp.fea import FormingSetup
    from precomp.materials import get_material

    return FormingSetup(get_material("AA5754-O"), blank_size=0.12, clamp_margin=0.01,
                        element_size=2.5e-3, layers=1, step_down=1e-3,
                        toolpath_spacing=2e-3, executable=str(fake_solver))


@pytest.fixture
def small_cone():
    return TruncatedCone(0.03, 45.0, 0.012, 0.004, 0.004)


@pytest.fixture
def node_grid() -> Grid:
    """The node grid of `small_setup`'s mesh (spacing = element size)."""
    return Grid.centered(0.12, 2.5e-3)


@pytest.fixture
def rng():
    return np.random.default_rng(20240917)


# ---------------------------------------------------------------------------
# precomp.ml: one proxy data set and one surrogate for the whole session
# ---------------------------------------------------------------------------
ML_CREATED_AT = "2026-09-29T00:00:00+00:00"
#: The family no ML fixture model is trained on (out-of-distribution tests).
#: Freeform, because its dents give it part descriptors no other family has:
#: the envelope detects new DESCRIPTORS, not new family names, and a family
#: that resembles the others (dome, saddle, two_level) is not flagged - see
#: test_the_envelope_across_every_held_out_family.
ML_HELD_OUT_FAMILY = "freeform"


@pytest.fixture(scope="session")
def ml_threads():
    """Cap scikit-learn/OpenMP and torch at two threads."""
    os.environ["PRECOMP_ML_THREADS"] = "2"
    try:
        import torch
        torch.set_num_threads(2)
    except ImportError:
        pass
    return 2


@pytest.fixture(scope="session")
def ml_features():
    from precomp.ml import FeatureConfig
    return FeatureConfig(time_source="depth")


@pytest.fixture(scope="session")
def proxy_samples(ml_threads):
    """7 parts of every family (AA5754-O, the default process, contour tool
    paths), three commanded variants each, on a 4 mm grid, formed by the
    ProxySimulator. (Every job has a contour path; the proxy's own check of
    that, which costs 10 s here, is tested in test_ml_generate.)"""
    from ml_support import ProxySimulator, simulate_samples
    from precomp.ml import DesignSpace, design_points
    space = DesignSpace(grid_spacing=4e-3, materials=("AA5754-O",), process={},
                        base_setup={"toolpath_style": "contour"})
    points = design_points(space, 7, seed=11)
    return simulate_samples(points, ProxySimulator(check_toolpath=False),
                            created_at=ML_CREATED_AT,
                            kinds=("uncompensated", "perturbed", "compensated"))


@pytest.fixture(scope="session")
def proxy_split(proxy_samples):
    """Grouped (by part) split of the in-distribution families: train (~45 %),
    calibration and test (the rest, halved); plus the held-out family."""
    from ml_support import family_split, grouped_split, index_frame

    byid = {s.sample_id: s for s in proxy_samples}
    rest, fam = family_split(proxy_samples, ML_HELD_OUT_FAMILY)
    train, pool = grouped_split(index_frame([byid[i] for i in rest]), test_fraction=0.55,
                                seed=0)
    cal, test = grouped_split(index_frame([byid[i] for i in pool]), test_fraction=0.5, seed=1)
    pick = lambda ids: [byid[i] for i in ids]  # noqa: E731
    return {"train": pick(train), "calibration": pick(cal), "test": pick(test),
            "pool": pick(pool), "family": pick(fam)}


@pytest.fixture(scope="session")
def mlp_surrogate(proxy_split, ml_features):
    """A calibrated five-member MLPEnsemble surrogate (the workflow's model,
    smaller) with its envelope, trained without the held-out family."""
    pytest.importorskip("torch")
    from precomp.ml import MLPEnsemble, train_surrogate
    return train_surrogate(MLPEnsemble(5, hidden=(64, 64), epochs=40, seed=0),
                           proxy_split["train"], calibration=proxy_split["calibration"],
                           features=ml_features, points_per_sample=250, seed=0)
