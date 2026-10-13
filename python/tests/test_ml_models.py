"""precomp.ml models, conformal intervals and the envelope, on ProxySimulator data.

Every number here is measured on proxy data - an analytic stand-in, NOT
physics. The shared surrogate (conftest `gbm_surrogate`) is trained on the
in-distribution families without the held-out one.
"""

import numpy as np
import pytest

from conftest import ML_CREATED_AT
from precomp._util import PrecompError
from precomp.materials import get_material
from precomp.ml import (GBMEnsemble, ProxySimulator, ResidualModel, build_table,
                        evaluate_surrogate, simulate_samples, train_surrogate, transfer_surrogate)
from precomp.ml.generate import DesignSpace, design_points
from precomp.ml.surrogate import calibrate
from ml_support import partition_coverage
from precomp.ml.uncertainty import ConformalCalibrator, min_parts


def _held_out_arrays(sur, samples):
    mus, sds, ys, gs = [], [], [], []
    for s in samples:
        mu, sd = sur.predict_deviation(s.commanded, s.setup)
        sel = sur.feature_maps(s.commanded, s.setup).part & s.valid
        mus.append(mu[sel])
        sds.append(sd[sel])
        ys.append(s.dz[sel])
        gs.append(np.full(sel.sum(), s.part_id, dtype=object))
    return [np.concatenate(v) for v in (mus, sds, ys, gs)]


def test_gbm_ensemble_learns_the_proxy_on_held_out_parts(gbm_surrogate, proxy_split):
    rep = evaluate_surrogate(gbm_surrogate, proxy_split["test"], level=0.9, levels=(0.9,),
                             assess=False)
    pp = rep.per_part
    assert set(pp["data_source"]) == {"proxy - not physics"}
    assert pp["part_id"].nunique() >= 10
    # dz is ~0.5-1 mm; the held-out error a small fraction of it
    assert pp["dz_rms"].mean() > 4e-4
    assert pp["rel_rms"].mean() < 0.2, pp[["sample_id", "rel_rms"]]
    assert pp["err_rms"].mean() < 1.2e-4
    assert np.all(pp["std_mean_m"] > 0)


def test_conformal_coverage_is_valid_on_held_out_parts(gbm_surrogate, proxy_split):
    """Split-conformal calibration over whole parts: over random calibration /
    test partitions of held-out parts, the mean coverage of a test part is at
    least nominal (the guarantee, minus Monte-Carlo slack) and at most the
    (k + 1) / k bound of k calibration parts. Validity holds for ANY std -
    so it cannot show that the std is useful; the constant-width baseline
    and a randomly permuted std can."""
    mu, sd, y, g = _held_out_arrays(gbm_surrogate, proxy_split["pool"])
    rng = np.random.default_rng(0)
    for level in (0.8, 0.9):
        pc = partition_coverage(mu, sd, y, g, level=level, n_partitions=60, seed=3)
        k = pc["n_calibration_parts"]
        assert pc["n_parts"] >= 20 and k >= min_parts(level)
        assert level - 0.03 <= pc["mean"] <= level * (k + 1) / k + 0.03, pc
        assert pc["baseline_mean"] >= level - 0.03                  # the baseline is valid too
    # a std without information: the same values permuted over the nodes
    real = partition_coverage(mu, sd, y, g, level=0.9, n_partitions=60, seed=3)
    perm = partition_coverage(mu, rng.permutation(sd), y, g, level=0.9, n_partitions=60, seed=3)
    const = partition_coverage(mu, np.full_like(sd, sd.mean()), y, g, level=0.9,
                               n_partitions=60, seed=3)
    assert const["width_ratio"] == pytest.approx(1.0)
    assert perm["mean"] >= 0.87 and perm["width_ratio"] > 1.0      # valid, but wider
    # the ensemble std locates the errors better than chance - but on this
    # proxy data it does NOT beat a constant width (ratio > 1, reported in
    # every calibration table): it moves width to the worst parts
    assert real["mean_width_m"] < 0.95 * perm["mean_width_m"], (real, perm)
    assert real["worst_part_median"] > const["worst_part_median"], (real, const)


def test_the_finite_sample_correction_counts_parts_not_nodes(gbm_surrogate, proxy_split):
    # one node per part: q is the ceil((k + 1) level)-th smallest score of k
    cal = ConformalCalibrator(eps=1.0).fit(np.zeros(9), np.zeros(9), np.arange(1.0, 10.0),
                                           np.arange(9))
    assert cal.quantile(0.8) == pytest.approx(8.0)           # ceil(10 x 0.8) = 8
    assert cal.quantile(0.9) == pytest.approx(9.0)           # k = 9 = min_parts(0.9)
    with pytest.raises(PrecompError, match="cannot support level 0.95"):
        cal.quantile(0.95)                                    # needs 19 parts
    # 8 parts of 10 000 nodes each: 80 000 nodes do not make 90 % supportable
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(8), 10_000)
    big = ConformalCalibrator().fit(np.zeros(g.size), np.ones(g.size), rng.normal(size=g.size),
                                    g)
    assert big.n_parts == 8 and big.n_nodes == 80_000 and not big.supports(0.9)
    with pytest.raises(PrecompError, match="needs at least 9 held-out parts"):
        big.intervals(np.zeros(3), np.ones(3), 0.9)
    assert big.summary()["quantiles"]["0.9"] is None and big.summary()["quantiles"]["0.8"]
    # the shipped path groups by PART: the variants of one part count once
    c = gbm_surrogate.calibrator
    assert c.n_parts == len({s.part_id for s in proxy_split["calibration"]}) \
        == len(proxy_split["calibration"]) // 3
    one_part = [s for s in proxy_split["calibration"]
                if s.part_id == proxy_split["calibration"][0].part_id]
    assert len(one_part) == 3
    with pytest.raises(ValueError, match="at least two held-out parts"):
        calibrate(gbm_surrogate, one_part)
    rep = evaluate_surrogate(gbm_surrogate, proxy_split["test"], level=0.9, assess=False)
    row = rep.calibration.set_index("level").loc[0.9]
    assert row["n_parts"] == len({s.part_id for s in proxy_split["test"]}) and row["supported"]
    assert row["calibration_parts"] == c.n_parts
    assert 0 <= row["part_coverage_min"] <= row["part_coverage_median"] <= 1
    assert row["width_ratio"] == pytest.approx(row["mean_width_m"] / row["baseline_width_m"])
    assert not rep.calibration.set_index("level").loc[0.95, "supported"]


def test_the_envelope_flags_new_descriptors_and_setups(gbm_surrogate, proxy_split):
    """The envelope flags parts whose DESCRIPTORS are new - here the held-out
    freeform family, whose dents no other family has. A family that resembles
    the others (dome, saddle, two_level) is not flagged: see the
    leave-each-family-out table in benchmarks/ml_proxy. A setup field no
    feature describes (release, tool path, mesh) is checked against the
    training setups."""
    known = [gbm_surrogate.assess(s.commanded, s.setup) for s in proxy_split["test"]]
    unseen = [gbm_surrogate.assess(s.commanded, s.setup) for s in proxy_split["family"]]
    flag_known = np.mean([not a["in_envelope"] for a in known])
    flag_unseen = np.mean([not a["in_envelope"] for a in unseen])
    assert flag_known <= 0.25 and flag_unseen >= flag_known + 0.5, (flag_known, flag_unseen)
    assert np.median([a["part_score"] for a in unseen]) > np.median(
        [a["part_score"] for a in known])
    out = [a for a in unseen if not a["in_envelope"]]
    assert all(a["reasons"] for a in out)                     # it says why
    assert known[0]["model_data_source"] == "proxy - not physics"
    s = proxy_split["test"][0]
    assert gbm_surrogate.assess(s)["part_score"] == known[0]["part_score"]   # a Sample too
    assert all(a["setup_mismatch"] == [] for a in known)
    st = s.forming_setup()
    for change, reason in ((dict(release="clamped_only"), "setup.release"),
                           (dict(toolpath_style="spiral"), "setup.toolpath_style"),
                           (dict(layers=1), "setup.layers"),
                           (dict(material=get_material("DC04")), "youngs_modulus"),
                           (dict(thickness=2.5e-3), "thickness")):
        a = gbm_surrogate.assess(s.commanded, st.replace(**change))
        assert not a["in_envelope"] and reason in a["reasons"], (change, a["reasons"])
    rel = gbm_surrogate.assess(s.commanded, st.replace(release="clamped_only"))
    assert rel["setup_mismatch"] == [{"field": "release", "value": "clamped_only",
                                      "trained": ["321"]}]


@pytest.fixture(scope="module")
def shifted_samples(ml_threads):
    """A 'real' process that differs from the simulated one: rim under-forming
    +40 %, unclamping curvature +30 % (still proxy data)."""
    real = ProxySimulator().shifted(rim_amplitude=1.12e-3, curvature=0.13)
    space = DesignSpace(grid_spacing=4e-3, materials=("AA5754-O",), process={},
                        families=("truncated_cone", "pyramid", "dome", "two_level"))
    pts = design_points(space, 3, seed=29)
    return simulate_samples(pts, real, created_at=ML_CREATED_AT, kinds=("uncompensated",))


def _err(model, table):
    return float(np.sqrt(np.mean((model.predict(table.X)[0] - table.y) ** 2)))


def test_transfer_model_equals_its_base_until_a_correction_is_validated(
        gbm_surrogate, shifted_samples, ml_features):
    base = gbm_surrogate
    t0 = transfer_surrogate(base, [])
    s = shifted_samples[0]
    a, b = base.predict_deviation(s.commanded, s.setup), t0.predict_deviation(s.commanded,
                                                                              s.setup)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert t0.model.n_real == 0 and not t0.model.accepted
    # one real part: nothing to validate a correction on, so none is applied
    t1 = transfer_surrogate(base, shifted_samples[:1], points_per_sample=400)
    assert t1.model.n_real == 1 and not t1.model.accepted
    assert "at least 2" in t1.training["transfer"]["validation"]["reason"]
    assert np.array_equal(t1.predict_deviation(s.commanded, s.setup)[0], a[0])
    # a real process 30-40 % stronger than the simulated one, four real parts
    real, test = shifted_samples[:4], shifted_samples[4:]
    t4 = transfer_surrogate(base, real, points_per_sample=400, seed=0)
    tr = t4.training["transfer"]
    assert t4.model.n_real == 4 and tr["n_real"] == 4 and tr["n_real_parts"] == 4
    assert t4.model.accepted and tr["validation"]["lopo_corrected_rms_m"] < \
        tr["validation"]["lopo_base_rms_m"]
    tb = build_table(test, ml_features, points_per_sample=None)
    err_base, err_tr = _err(base.model, tb), _err(t4.model, tb)
    assert err_tr < 0.7 * err_base, (err_tr, err_base)
    assert np.all(t4.model.predict(tb.X)[1] >= base.model.predict(tb.X)[1] - 1e-15)
    with pytest.raises(Exception, match="no conformal"):
        t4.predict_interval(s.commanded, s.setup, 0.9)        # not calibrated on real data
    # the provenance names the real data: evaluation refuses it, --family knows it
    assert set(t4.training["part_ids"]) >= {x.part_id for x in real}
    assert set(t4.training["families"]) >= {x.family for x in real}
    with pytest.raises(PrecompError, match="trained, corrected or calibrated"):
        evaluate_surrogate(t4, real[:1], assess=False)


def test_transfer_does_not_make_the_base_worse_when_real_equals_simulation(
        gbm_surrogate, proxy_split, ml_features):
    """No shift: the "real" parts are more proxy parts. (The former default,
    a Bayesian ridge on every feature, made the base 2-5 times worse here.)"""
    base = gbm_surrogate
    real = [s for s in proxy_split["test"] if s.kind == "uncompensated"]
    held = [s for s in proxy_split["test"] if s.part_id not in {r.part_id for r in real[:3]}]
    tb = build_table(held, ml_features, points_per_sample=500, rng=2)
    for n in (2, 3):
        t = transfer_surrogate(base, real[:n], points_per_sample=400, seed=0)
        assert _err(t.model, tb) <= 1.05 * _err(base.model, tb), t.training["transfer"]


def test_residual_model_learns_what_its_prior_misses(proxy_split, shifted_samples, ml_features):
    """A proxy prior of the wrong strength plus a GBM on the residual: the
    hybrid path. On the shifted data it beats the prior alone by far."""
    prior = ProxySimulator()
    model = ResidualModel(prior, GBMEnsemble(2, max_iter=60, target_scale=None))
    sur = train_surrogate(model, shifted_samples[:8], features=ml_features,
                          points_per_sample=300, envelope=False)
    assert sur.feature_names[-1] == "prior_dz"
    err_prior, err_hyb = [], []
    for s in shifted_samples[8:]:
        sel = s.commanded.z < -1e-6
        mu, _ = sur.predict_deviation(s.commanded, s.setup)
        err_hyb.append(np.sqrt(np.mean((mu - s.dz)[sel] ** 2)))
        err_prior.append(np.sqrt(np.mean((prior.deviation(s.commanded, s.setup) - s.dz)[sel]
                                         ** 2)))
    assert np.mean(err_hyb) < 0.5 * np.mean(err_prior)


def test_gbm_is_deterministic_and_validates_its_input():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 3))
    y = X[:, 0] + 0.1 * rng.normal(size=600)
    g = np.repeat(np.arange(6), 100)
    a = GBMEnsemble(3, max_iter=30, target_scale=None, seed=5).fit(X, y, g)
    b = GBMEnsemble(3, max_iter=30, target_scale=None, seed=5).fit(X, y, g)
    assert np.array_equal(a.predict(X)[0], b.predict(X)[0])
    with pytest.raises(ValueError, match="target_scale"):
        GBMEnsemble(2, max_iter=5).fit(X, y, g)               # no elastic_curvature column
    with pytest.raises(ValueError, match="NaN"):
        a.predict(np.full((2, 3), np.nan))
    with pytest.raises(ValueError, match="3 features"):
        a.predict(X[:, :2])
    q = GBMEnsemble(2, max_iter=30, target_scale=None, quantiles=(0.05, 0.95)).fit(X, y, g)
    band = q.predict_quantiles(X)
    assert np.mean((y >= band[0.05]) & (y <= band[0.95])) > 0.8


def test_mlp_ensemble_fits_and_is_reproducible(proxy_split, ml_features):
    pytest.importorskip("torch")
    from precomp.ml import MLPEnsemble

    tr = build_table(proxy_split["train"][:24], ml_features, points_per_sample=200, rng=0)
    te = build_table(proxy_split["test"][:6], ml_features, points_per_sample=200, rng=1)
    kw = dict(hidden=(32, 32), epochs=8, batch_size=512, seed=3)
    m = MLPEnsemble(2, **kw).fit(tr.X, tr.y, tr.groups, feature_names=tr.feature_names)
    mu, sd = m.predict(te.X)
    assert np.all(np.isfinite(mu)) and np.all(sd > 0)
    err = np.sqrt(np.mean((mu - te.y) ** 2))
    assert err < 0.5 * np.sqrt(np.mean((te.y - tr.y.mean()) ** 2))
    m2 = MLPEnsemble(2, **kw).fit(tr.X, tr.y, tr.groups, feature_names=tr.feature_names)
    assert np.array_equal(m2.predict(te.X)[0], mu)
