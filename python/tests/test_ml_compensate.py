"""Surrogate compensation, verified by the ProxySimulator (NOT physics) and,
for the plumbing of the SparLab path, by the test-double solver."""

import json

import numpy as np
import pytest

from conftest import run_count
from precomp._util import PrecompError
from precomp.fea import FormingSetup
from precomp.geometry import Grid, TruncatedCone
from precomp.materials import get_material
from precomp.ml import (ProxySimulator, surrogate_compensate, verify_with_fea,
                        verify_with_simulator)


def test_surrogate_compensation_cuts_the_verified_deviation_by_a_large_factor(
        gbm_surrogate, proxy_split):
    """On held-out parts: DA on the surrogate, then the compensated shape
    formed by the proxy. The deviation from the target falls several-fold
    against forming the target itself (both measured by the proxy)."""
    parts = [s for s in proxy_split["test"] if s.kind == "uncompensated"
             and gbm_surrogate.assess(s.target, s.setup)["in_envelope"]][:3]
    factors = []
    for s in parts:
        res = surrogate_compensate(s.target, s.setup, gbm_surrogate, iterations=6)
        assert res.stopped in ("within_interval", "stagnation", "iterations", "tolerance")
        assert res.data_source == "proxy - not physics"
        assert res.compensated.depth > s.target.depth         # over-bent against springback
        lo, hi = res.residual_interval
        assert np.all(lo.z <= res.residual.z + 1e-15) and np.all(res.residual.z <= hi.z + 1e-15)
        assert res.ood["in_envelope"] and res.in_envelope
        assert all(h["in_envelope"] for h in res.history)
        pm = res.predicted_metrics()
        assert pm["interval_halfwidth_mean_m"] > 0 and pm["interval_level"] == 0.9
        v = verify_with_simulator(res, s.setup, ProxySimulator())
        assert v["data_source"] == "proxy - not physics" and res.verification is v
        factors.append(v["improvement_factor"])
    assert min(factors) > 2.5 and np.median(factors) > 4.0, factors


def test_stopping_rules(tmp_path, gbm_surrogate, proxy_split):
    s = next(x for x in proxy_split["test"] if gbm_surrogate.assess(x.target, x.setup)[
        "in_envelope"])
    res = surrogate_compensate(s.target, s.setup, gbm_surrogate, iterations=5, tolerance=1.0)
    assert res.stopped == "tolerance" and len(res.history) == 1
    res = surrogate_compensate(s.target, s.setup, gbm_surrogate, iterations=6,
                               interval_stop=True)
    assert res.stopped == "within_interval"                  # the model cannot resolve less
    h = res.history[-1]
    assert h["error"]["rms"] <= h["interval_halfwidth_rms_m"]
    res = surrogate_compensate(s.target, s.setup, gbm_surrogate, iterations=3, stagnation=0.0)
    assert len(res.history) <= 3 and res.best_iteration < len(res.history)
    out = res.save(tmp_path / "c")
    assert (out / "compensated.npz").is_file() and (out / "residual_upper.npz").is_file()
    doc = json.loads((out / "compensation.json").read_text())
    assert doc["model_data_source"] == "proxy - not physics" and doc["in_envelope"] is True
    assert "not simulations" in doc["history_quantity"]
    assert doc["predicted_residual"]["quantity"] == "predicted residual (surrogate)"


def test_the_envelope_is_enforced(gbm_surrogate, proxy_split):
    """A target outside the envelope is refused unless allowed; an iterate
    that leaves it ends the loop at the best iterate inside."""
    s = next(x for x in proxy_split["test"] if gbm_surrogate.assess(x.target, x.setup)[
        "in_envelope"])
    thick = s.forming_setup().replace(thickness=2.5e-3)
    with pytest.raises(PrecompError, match="outside the model's training envelope"):
        surrogate_compensate(s.target, thick, gbm_surrogate, iterations=2)
    res = surrogate_compensate(s.target, thick, gbm_surrogate, iterations=2,
                               allow_out_of_envelope=True)
    assert res.allowed_out_of_envelope and res.in_envelope is False
    assert "thickness" in res.ood_target["reasons"]

    class Shrinking:
        """The surrogate, with an envelope that ends after the first iterate."""

        def __init__(self, sur):
            self.sur, self.calls = sur, 0
            self.ood, self.calibrator, self.data_source = sur.ood, None, sur.data_source

        def predict_deviation(self, c, st):
            return self.sur.predict_deviation(c, st)

        def assess(self, c, st):
            self.calls += 1
            a = dict(self.sur.assess(c, st))
            a["in_envelope"] = self.calls == 1
            return a

    res = surrogate_compensate(s.target, s.setup, Shrinking(gbm_surrogate), iterations=4)
    assert res.stopped == "left_envelope" and res.best_iteration == 0
    assert len(res.history) == 2 and res.history[-1]["predicted"] is False


def test_verification_through_sparlab_form(tmp_path, gbm_surrogate, fake_solver, counter):
    """The plumbing of the SparLab check, with the test double (formed = 0.9 x
    commanded): the record names the solver build and the deck."""
    setup = FormingSetup(get_material("AA5754-O"), executable=str(fake_solver), layers=1,
                         element_size=4e-3, clamp_margin=0.02, step_down=1.5e-3,
                         toolpath_spacing=3e-3)
    target = TruncatedCone(0.045, 45.0, 0.015, 0.005, 0.005).heightmap(
        Grid.centered(setup.meshed_blank_size, 4e-3))
    # the double's setup (one layer, 4 mm elements, 1.5 mm steps) is not the
    # training setup: refused, unless allowed - this test is about the plumbing
    with pytest.raises(PrecompError, match="setup.layers"):
        surrogate_compensate(target, setup, gbm_surrogate, iterations=3)
    res = surrogate_compensate(target, setup, gbm_surrogate, iterations=3,
                               allow_out_of_envelope=True)
    v = verify_with_fea(res, setup, tmp_path / "runs", max_workers=1)
    assert v["data_source"] == "SparLab simulation" and "test double" in v["sparlab_version"]
    assert len(v["deck_hash"]) == 64 and run_count(counter) == 2
    # the double springs back 10 % of the depth, the proxy-trained model
    # predicted otherwise: the check exposes the mismatch instead of echoing
    # the prediction
    assert v["achieved_part_rms_m"] > 5 * v["predicted_part_rms_m"]
    assert v["uncompensated"]["vertical_deviation"]["part"]["rms"] > 0


def test_surrogate_compensation_passes_the_conditioning_on(gbm_surrogate, proxy_split):
    """The support's bound and masks and the tool's reach reach every DA step."""
    from precomp.toolpath import tool_reach_surface

    s = next(x for x in proxy_split["test"] if gbm_surrogate.assess(x.target, x.setup)[
        "in_envelope"])
    R = s.forming_setup().tool_radius
    res = surrogate_compensate(s.target, s.setup, gbm_surrogate, iterations=3,
                               stagnation=0.0, tool_radius=R, upper_bound=0.0)
    assert res.best_iteration > 0          # a compensated iterate, not the target
    c = res.compensated
    np.testing.assert_allclose(tool_reach_surface(c, R).z, c.z, atol=1e-9)
    hold = np.ones(s.target.grid.shape, dtype=bool)
    with pytest.raises(ValueError, match="adjusted region is empty"):
        surrogate_compensate(s.target, s.setup, gbm_surrogate, iterations=2, hold_mask=hold)


def test_the_optimiser_does_not_predict_worse_than_da(gbm_surrogate, proxy_split):
    """`surrogate_optimize` from a surrogate DA command (one DA update): its
    predicted RMS deviation over the part is at most DA's (the start is a
    candidate, every accepted step lowers the objective) - here clearly
    below it - the command keeps DA's constraints
    (the flange held, at or below the sheet plane, the wall-angle limit) and
    stays inside the envelope. Predictions of a proxy-trained model only."""
    from precomp.geometry.parts import MAX_WALL_ANGLE_DEG
    from precomp.metrology import flange_mask
    from precomp.ml import correction_basis, surrogate_optimize

    s = next(x for x in proxy_split["test"] if x.kind == "uncompensated"
             and gbm_surrogate.assess(x.target, x.setup)["in_envelope"])
    # one DA update (two predictions): a start the optimiser can improve on
    da = surrogate_compensate(s.target, s.setup, gbm_surrogate, iterations=2,
                              stagnation=0.0)
    opt = surrogate_optimize(s.target, s.setup, gbm_surrogate, start=da,
                             level_knots=(0.0, 0.25, 0.6, 1.0), angular_orders=(2,),
                             max_iterations=3, max_time_s=60.0)
    assert opt.stopped == "optimizer" and opt.data_source == "proxy - not physics"
    rms_da = da.predicted_metrics()["rms"]
    rms_opt = opt.predicted_metrics()["rms"]
    assert rms_opt <= rms_da * (1 + 1e-9), (rms_opt, rms_da)
    assert rms_opt < 0.9 * rms_da, (rms_opt, rms_da)    # here it improves on it
    assert opt.history[0]["batch"] == "start"
    assert abs(opt.history[0]["error"]["rms"] - rms_da) < 1e-12
    assert "stop_reason" in opt.history[-1] and opt.history[opt.best_iteration]["accepted"]
    assert opt.in_envelope
    c = opt.compensated
    fl = flange_mask(s.target)
    assert np.allclose(c.z[fl], s.target.z[fl]) and c.z.max() <= 1e-12
    assert np.degrees(c.wall_angle().max()) <= MAX_WALL_ANGLE_DEG + 1e-6
    assert c.metadata["compensation"]["parameters"] == 12
    B = correction_basis(s.target, s.target.z < -1e-6, (0.0, 0.25, 0.6, 1.0), (2,))
    assert B.shape[0] == 12 and np.allclose(np.abs(B).max(axis=(1, 2)), 1.0)
    assert np.all(B[:, s.target.z >= -1e-6] == 0.0)


def test_the_optimiser_std_weight_keeps_it_where_the_model_is_sure(gbm_surrogate, proxy_split):
    """`std_weight` (default 0.25, backward compatible) weighs the model's std
    in the objective. A wrapped surrogate whose std grows with the distance
    from the start command (a model sure only of what it has seen): without
    the std term the optimiser moves and lowers the predicted RMS; with a
    strong weight it stays much closer to the start (smaller std), trading
    predicted RMS for confidence. The weights are recorded; a negative one is
    refused. Predictions of a proxy-trained model only."""
    import inspect

    from precomp.ml import surrogate_optimize

    assert inspect.signature(surrogate_optimize).parameters["std_weight"].default == 0.25
    s = next(x for x in proxy_split["test"] if x.kind == "uncompensated"
             and gbm_surrogate.assess(x.target, x.setup)["in_envelope"])
    da = surrogate_compensate(s.target, s.setup, gbm_surrogate, iterations=2,
                              stagnation=0.0)
    c0 = da.compensated

    class Unsure:
        """gbm_surrogate, with std = its own + 3 x |command - start|."""

        def __getattr__(self, name):
            return getattr(gbm_surrogate, name)

        def predict_deviation(self, commanded, setup):
            mu, sd = gbm_surrogate.predict_deviation(commanded, setup)
            return mu, np.sqrt(np.asarray(sd) ** 2 + (3.0 * (commanded.z - c0.z)) ** 2)

    kw = dict(start=da, level_knots=(0.0, 0.25, 0.6, 1.0), angular_orders=(2,),
              max_iterations=3, max_time_s=60.0)
    free = surrogate_optimize(s.target, s.setup, Unsure(), std_weight=0.0, **kw)
    sure = surrogate_optimize(s.target, s.setup, Unsure(), std_weight=4.0, **kw)
    move = lambda r: float(np.abs(r.compensated.z - c0.z).max())  # noqa: E731
    assert free.predicted_metrics()["rms"] < 0.9 * da.predicted_metrics()["rms"]
    assert move(sure) < 0.5 * move(free), (move(sure), move(free))
    pm = s.target.z < -1e-6
    sd_rms = lambda r: float(np.sqrt(np.mean(r.std[pm] ** 2)))  # noqa: E731
    assert sd_rms(sure) < sd_rms(free), (sd_rms(sure), sd_rms(free))
    assert free.compensated.metadata["compensation"]["std_weight"] == 0.0
    assert sure.compensated.metadata["compensation"]["std_weight"] == 4.0
    with pytest.raises(ValueError, match="std_weight"):
        surrogate_optimize(s.target, s.setup, gbm_surrogate, std_weight=-1.0, **kw)
