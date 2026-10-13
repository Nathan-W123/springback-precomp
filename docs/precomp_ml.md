# precomp.ml: learned springback prediction and surrogate pre-compensation

`precomp.ml` predicts the surface a commanded tool path will form, inverts
that prediction (displacement adjustment and an optimiser) so the first part
lands near the target, and says when the prediction should not be trusted. It
is built on `precomp` ([precomp.md](precomp.md)) and uses its height maps,
part families, tool paths, SparLab bridge, metrology and compensation as they
are.

**Where the numbers come from.** Every metric carries its data source:
`SparLab simulation`, `scan` or `proxy - not physics` (`SOURCE_LABELS`). The
[results](#results-on-sparlab-simulations) are `sparlab_form` simulations with
nominal material data, not experiments; no model here has seen a scanned part.
The unit tests train on `ProxySimulator` (`python/tests/ml_support.py`), an
analytic stand-in that is **not physics** and not part of the package.

## The learning problem

The target is the vertical deviation after springback,
dz(x, y) = z_formed - z_commanded [m], on the grid of the commanded surface, as
a function of the **commanded** surface, the process and the material. During
compensation the model is queried at compensated shapes, deeper and steeper
than the target, so the data hold compensated commands as well as targets.
Displacement adjustment inverts formed(c) = c + dz_hat(c).

## Assumptions

* Height fields z <= 0 of the tool side, centred on the blank; the grid spacing
  must not exceed the ring width (one tool radius by default).
* **One material and one machine setup per model.** The release, tool path,
  mesh, element formulation, increment, contact, solver settings and support
  enter through **no** feature; a model records the values its training setups
  had and flags any other (`setup.<field>` in the envelope's reasons).
* `time_frac` uses the setup's own spiral (`precomp.fea.make_toolpath`), so a
  command with several pockets on one level can be neither featurised nor
  simulated.

## Modules

| Module | Public API |
|--------|------------|
| `features` | `FEATURE_SCHEMA_VERSION` ("2"), `FeatureSpec`, `FeatureConfig(scales, rings, ring_max, time_source, part_eps, rim_width)` with `.specs()`, `.names`, `.schema_hash()`; `feature_maps(commanded, setup, toolpath=None, config) -> FeatureMaps` (`.values` (ny, nx, F), `.gather(points)`, `.region`); `global_features`, `region_labels` |
| `dataset` | `Sample`, `Dataset` (`create(root, created_at=...)`, `append`, `load`, `samples`, `ids`, `index`, `filter`, `record_failure`, `failures`, `to_table`), `build_table(...) -> Table` (unpacks as `X, y, groups`), `SOURCE_LABELS`, `source_label` |
| `generate` | `DesignSpace`, `DesignPoint`, `design_points(space, n_per_family, seed)`, `PerturbationSpec`, `perturbed_commanded`, `compensated_commanded`, `generate(dataset, points, simulator, *, created_at, kinds, seed, compensator, da_options, store_toolpath) -> GenerationReport`, `SparlabSimulator(work_dir, max_workers=...)`, the `Simulator` protocol, `sim_job` |
| `models` | `MLPEnsemble`, `TransferModel(base, correction="linear", prior_gain, prior_offset, min_parts)`, `model_from_state`, `thread_limit`; the `DeviationModel` protocol (`fit(X, y, groups)`, `predict(X) -> (mean, std)`, `feature_names`, `to_state`) |
| `uncertainty` | `ConformalCalibrator(eps, eps_fraction, mondrian)`: `fit(mu, std, y, groups, region=...)`, `quantile(level)`, `supports(level)`, `intervals(mu, std, level)`, `baseline_intervals(mu, level)`; `min_parts(level)` |
| `ood` | `OODEnvelope(quantile, k, ...)`: `fit(point_X, point_groups, part_D, part_groups, ...)`, `assess_features(point_X, part_d)` |
| `surrogate` | `DeviationSurrogate` (`target = "dz"`, `predict_deviation`, `predict_interval`, `assess`, `describe`); `train_surrogate(model, train, calibration=..., features, points_per_sample, mask, seed, group_by)`, `transfer_surrogate(base, real, calibration=..., correction="linear")`, `calibrate`, `fit_envelope`, `setup_envelope`, `setup_mismatch`, `seen_ids` |
| `registry` | `save_model(model, directory, manifest)`, `load_model(directory)`, `read_manifest`, `check_schema` |
| `evaluate` | `evaluate_surrogate(surrogate, samples, tolerance, level) -> EvaluationReport` (`per_part`, `per_region`, `calibration`, `ood`, `summary()`, `to_csv()`) |
| `compensate` | `SurrogatePredictor(model, setup)`, `surrogate_compensate(...) -> SurrogateCompensation`, `correction_basis`, `surrogate_optimize(...)` |

`DeviationSurrogate` implements the `FieldModel` protocol of
`precomp.compensation` (plus `predict_interval` and `assess`). It predicts the
**total** deviation dz, never a residual over a simulation.

## Features

Per grid node, schema version "2", 57 features in five kinds
(`FeatureConfig().specs()` gives each with its unit and description):

| Kind | Features | Units |
|------|----------|-------|
| local (invariant when the part moves in x-y) | `depth`, `depth_frac`; `wall_angle_s1..3` (Gaussian-smoothed at 1, 2.5, 6 tool radii); `mean_curv_s1..3` = H sigma and `gauss_curv_s1..3` = K sigma^2 (dimensionless); `rim_dist` (signed distance to the rim, + inside); `radial_norm` (distance to the part centroid / sqrt(A/pi)); `polar_sin`, `polar_cos` (regularised by the tool radius R: dy / sqrt(r^2 + R^2)); `time_frac` (pseudo-time of the in-contact path point nearest the node offset by R along its normal; or the depth fraction with `time_source="depth"`); `ring{1..4}_{mean,min,max}` (neighbour heights in four annuli out to 4 R, relative to the node) | m, -, rad |
| placement | `clamp_dist` (L-infinity distance to the clamped frame), `blank_radius` (distance to the blank centre / free half-width) | m, - |
| global (per part) | `max_depth`, `area`, `volume`, `mean_wall_angle`, `max_wall_angle`, `perimeter_over_area`, `aspect`, `depth_gyration_major/minor` | m, m^2, m^3, rad, 1/m, - |
| process | `tool_radius`, `step_down`, `thickness`, `friction`, `elastic_curvature` = 3 sigma_f(0.2) / (E t), the unloading curvature of a fully plastic bend | m, -, 1/m |
| material | `youngs_modulus`, `poisson_ratio`, `yield_stress`, `flow_stress_20`, `flow_stress_50`, `hardening_modulus`, `voce_Q`, `voce_rate`, `r0`, `r45`, `r90` (1 when isotropic), `backstress_sat`, `flow_stress_20_over_E` | Pa, - |

sigma_f(0.2) is the flow stress at 20 % plastic strain, not the 0.2 % proof
stress (version 2 renamed `yield_over_E`; version-1 bundles are refused).
`FeatureConfig.schema_hash()` fingerprints version, names, units and
configuration; every feature is checked finite. The ring extrema equal
`scipy.ndimage`'s footprint filters bit for bit; a 200 x 200 grid at 1 mm
featurises in about 0.2 s (0.4 s with its tool path; the test requires < 2 s).
Moving a pyramid by (7, -4) grid spacings leaves every non-placement feature,
`time_frac` included, equal to 1e-9 of its range (`test_ml_features`). Region
codes (`region_labels`): 0 flange, 1 rim (part nodes within 2 tool radii of the
boundary), 2 wall (wall angle > 5 deg), 3 base.

## Data

A `Sample` holds the commanded and formed surfaces on one grid, the target, the
setup and part as JSON, the variant kind, a `part_id` shared by the variants of
one design point, the source, a fidelity tag (`SparlabSimulator.fidelity_of`)
and provenance (`created_at` from the caller; for simulations the SparLab
version, deck hash, runtime, cache hit). A `Dataset` directory holds
`dataset.json`, `index.jsonl` (append-only), `samples/<id>.npz`,
`samples/<id>.json` and `failures.jsonl`. `build_table` / `Dataset.to_table`
featurise each sample and draw `points_per_sample` nodes of the mask (`"part"`
or `"all"`) shared equally among the regions present; rows carry sample id
(`groups`), part id, family, region and source.

**Splits** are by design point, so the variants of a part are never on both
sides; `train_surrogate` refuses a calibration part that is also a training
part. The workflow's `split_of` (`python/scripts/common.py`) makes the first
`test_per_family` indices of a family TEST, every `calibration_every`-th of the
rest CALIBRATION, the others TRAIN (2 and 4 in `benchmarks/springback/config.json`).

**Design and variants.** `design_points` draws, per family, a scrambled Sobol
sequence (seeded by the seed and a hash of the family's **name**) over the
family's bounds, the process ranges (default tool radius 4-8 mm, step-down
0.2-1 mm, thickness 0.6-1.5 mm, friction 0.05-0.2) and the material; refused
draws are skipped, so a point id (`dome-s2026-0003`) names the same part
whatever else is drawn. The workflow fixes process and material (AA5754-O).
Variants: `uncompensated` (commanded = target); `perturbed` (a smooth random
field of RMS 0.2-1 mm, correlation length 8-25 mm, zero on the flange,
projected onto the formable set); `compensated` (one DA step from the
simulated uncompensated part, with per-point `da_options` such as a support's
masks and bound and the tool's reach, or a `compensator`). The workflow uses
`uncompensated` and `compensated`.

**Generation** (`generate`, through `SparlabSimulator` and
`precomp.fea.simulate_many`) is resumable (a stored sample is skipped once
checked to be the same part, setup physics and grid) and records every failure
with its reason; jobs of a setup with a support carry the target (`sim_job`).

## Models

Models are deterministic for a seed, cap their threads at `n_threads` or
`$PRECOMP_ML_THREADS`, and learn dz divided by the `elastic_curvature` feature,
the first-order springback scale (`target_scale`; `None` learns dz as is).

* **`MLPEnsemble`** (torch, CPU): `n_members` MLPs (`hidden`, SiLU) with a
  (mean, log-variance) head, trained on the Gaussian NLL after `mse_warmup`
  epochs of squared error, Adam, early stopping on a validation split of whole
  samples (whole parts with `train_surrogate(group_by="part")`), one seed per
  member; the prediction is the mixture's mean and variance. The workflow
  trains `MLPEnsemble(5, hidden=(128, 128), epochs=40, seed=2026)`. Its spread
  is a relative scale for conformal calibration, not an error estimate.
* **`TransferModel(base, correction="linear")`**: the base frozen, plus
  y = a + (1 + b) mu, a Bayesian regression with a fixed prior (b ~ N(0,
  `prior_gain`^2), a ~ N(0, (`prior_offset` s)^2), s the RMS base prediction,
  both 0.5) in which every real part weighs one observation; the parameter
  uncertainty adds to the base std in quadrature. The correction is kept only
  if its leave-one-real-part-out error beats the base's; otherwise, and always
  with fewer than `min_parts` = 2 real parts, the model is the base
  (`accepted`, `validation`, `n_real`). Take the real parts from different
  families: two of one family validate each other's family offset.
  `transfer_surrogate` keeps the base's feature envelope, widens the setup
  envelope to the real setups, records the real parts (evaluation refuses
  them), and gives intervals only when calibrated on held-out real parts the
  base never saw. On the rim-pass runs of `combined_rim_ml.py` (SparLab) it
  cut the plate-trained MLP's dz error from 0.486 to 0.100 mm RMS
  (leave-one-part-out, 8 fit parts) and from 0.526 to 0.124 mm on 4 parts held
  out from the correction (training parts of the base, not unseen geometry).

## Uncertainty: conformal intervals

`ConformalCalibrator` is fitted on held-out **whole parts** and counts parts,
not nodes (the errors of a part are one smooth field; `calibrate` groups the
variants of a design point by part id):

    score r = |y - mu| / (std + eps),   eps = 0.1 median(std) of the calibration set,
    q = the smallest t with  sum_j F_j(t) >= level (k + 1),
    interval = mu -+ q (std + eps),

F_j the empirical distribution of the scores of calibration part j (each part
weighs one), k the number of calibration parts. **The guarantee:** for a new
part exchangeable with the calibration parts, the **expected share of its nodes
inside the interval is at least the level** (with the new part's F included the
threshold would be symmetric in all k + 1 parts; leaving it out can only raise
q). A level needs k >= level / (1 - level) parts, 9 at 90 %, 19 at 95 %
(`min_parts`); otherwise it is an error and tables report it as not supported.
When parts behave alike the expected coverage approaches level (k + 1) / k
(97.5 % at 90 % with 12 parts).

**What it is not.** A guarantee for any one part (a part whose error is a
large offset can be covered far less: the evaluation reports the per-part
distribution), nor evidence that the std is useful, since any std gives valid
intervals. The calibrator therefore also calibrates a constant-width interval
mu -+ q0; a width ratio below 1 means the std narrows the intervals.
`mondrian=True` takes one quantile per region (rim, wall, base).

## The training envelope

`OODEnvelope` measures, on standardised features, the Mahalanobis distance
(Ledoit-Wolf covariance) and the mean distance to the nearest training vectors
(5 for nodes, 3 for parts), for every node's feature vector and for the part's
descriptor vector (global, process, material). Thresholds are 99 % quantiles
over **held-out training parts** (nodes: 5 folds by part id; parts:
leave-one-part-out), so an in-distribution part scores like a new one. A part
is inside when its part score (the larger distance over its threshold) is
<= 1 and at most `max_fraction_out` of its nodes (at least 5 %) exceed a node
threshold. `assess` returns `in_envelope`, `part_score`,
`fraction_points_out`, the thresholds and `reasons` (features outside their
training range first, then the largest Mahalanobis contributions).

**Setup fields.** Every physics field of `FormingSetup` that no feature
describes (blank, clamp, mesh, element formulation, thickness points, tool path
settings, contact, increment, release, kinematics, solver, `support`,
`support_settings`) is recorded by `setup_envelope`; a query outside is out of
the envelope with `setup.<field>` first among the reasons and the details in
`setup_mismatch`. A support setting that spells out its default counts as the
default; a field the model never recorded admits only its default. The
envelope detects new **descriptors**, not new family names.

## Model bundles

`save_model(surrogate, dir, manifest)` writes `manifest.json`, `model.joblib`
(the state) and `torch/<name>.pt` (state dicts). The manifest, written last,
holds the model class, `target` ("dz"), the feature schema (version, hash,
configuration, names and their SHA-256), the data source, the training record
(sample and part ids, calibration parts, families, fidelities, SparLab versions,
setup envelope, a transfer model's real parts and validation, seed), metrics
(each with its `data_source`), the conformal quantiles at 50-99 %, the OOD
thresholds, package versions, the git revision, `created_at` and the SHA-256
of every file. `load_model` refuses another schema, feature names that do not
match their hash, a missing or wrong file hash, and a state whose model class,
kind, domain or data source differ from the manifest's. **`model.joblib` is a
pickle: load bundles you trust only.**

## Evaluation

`evaluate_surrogate(surrogate, samples, tolerance=2e-4, level=0.9)` refuses a
sample whose sample id **or part id** trained, corrected or calibrated the
model (`seen_ids`) and scores dz on the part's nodes: `per_part` (error rms,
mae, max_abs, p95_abs, bias, share within tolerance [m], coverage, width,
envelope verdict), `per_region`, `calibration` (per level: supported, mean
per-part and pooled coverage, worst / 10th-percentile / median part, the
constant-width baseline) and `ood`, each with a `data_source` column.

## Compensation on the surrogate

`surrogate_compensate(target, setup, surrogate)` runs `displacement_adjustment`
one iteration at a time on formed(c) = c + dz_hat(c), with the same
conditioning (`upper_bound`, `hold_mask`, `adjust_mask`, `tool_radius`, the
65 deg limit), and keeps the best predicted iterate. It stops at `tolerance`,
when an iteration improves the predicted RMS by less than `stagnation` (2 %),
after `iterations` (8), or, with `interval_stop=True` (off by default), within
the interval half-width. `SurrogateCompensation` holds the command, the
predicted formed surface, the predicted residual with its conformal interval,
the envelope reports, the history and `stopped`; `predicted_metrics()` labels
it all as a prediction. **The envelope is enforced:** a target outside is
refused and the first iterate outside ends the loop (`left_envelope`), unless
`allow_out_of_envelope=True`.

`surrogate_optimize(target, setup, surrogate, start=...)` trades changes in one
place for effects in another, which DA's pointwise update cannot. The command
is c(p) = condition(start + sum_k p_k B_k), B the 30 smooth fields of
`correction_basis` (hat functions in the normalised depth level at knots 0,
0.1, 0.2, 0.35, 0.6, 1.0 times the angular modes 1, cos/sin 2 theta, cos/sin
4 theta, smoothed at 1 mm), and p minimises

    J(p) = mean_part(r^2) + std_weight mean_part(std^2) + reg_weight mean(p^2)

(r the predicted residual; `std_weight` 0.25 keeps the search where the model
is sure; `reg_weight` 0.05) by Levenberg-Marquardt with a central
finite-difference Jacobian (50 um), command steps <= 0.3 mm and |p| <= 1 mm,
the probes being candidates too; a candidate must lower J by more than
`rel_tol` and stay inside the envelope. It stops after `max_iterations` (8), at
no accepted step or at `max_time_s` (`stopped = "optimizer"`, the reason in
`history[-1]["stop_reason"]`). **A prediction is not a result:** the workflow
forms every compensated shape once with `sparlab_form`.

## How the workflow uses this

Both scripts run from the repository root (`OPENBLAS_NUM_THREADS=1 python3
python/scripts/<name>.py`) and resume through the run cache, the data set and
the saved bundles; the FE baselines come from `springback_fine_benchmark.py`.

* `springback_fine_ml.py --out DIR --work DIR --stages N [N ...] [--workers N]
  [--threads N] [--data-only] [--executable PATH]`: per stage (N design points
  per family of `benchmarks/springback/config.json`), the target and one FE-DA
  step of every training and calibration part on a backing plate at
  `springback_fine_bulk` (`generate`, `SparlabSimulator`); `train_surrogate`,
  `save_model`; the ML first shot (`surrogate_compensate`), verified by one run
  at `springback_fine`; `evaluate_surrogate`. Outputs in `<out>/stage_n<N>/`.
* `combined_rim_ml.py {all,one,select,status,report} [--max-time S]`: `select`
  picks 12 training parts; `all` simulates them with DSIF and the rim pass,
  fits `transfer_surrogate(..., correction="linear")` of the stage n26 MLP on 8
  of them, and for `$COMBINED_PART` (default `pyramid-s2026-0001`) runs
  `surrogate_compensate` on the deep region, `surrogate_optimize` (`--max-time`,
  default 240 s) and one verifying run; `one` repeats that for another part.

In Python, with the rim pass:

```python
from precomp.fea import FormingSetup, simulate
from precomp.fea.support import command_upper_bound, compensation_masks
from precomp.geometry import Grid, TruncatedCone
from precomp.ml import load_model, surrogate_compensate, surrogate_optimize

setup = FormingSetup.preset("springback_fine", support="dsif", support_settings={"rim_pass": True})
target = TruncatedCone(0.007, 40.0, 0.003, 0.001, 0.002).heightmap(
    Grid.centered(setup.meshed_blank_size, 2.5e-4))
model = load_model("models/transfer")                  # checks schema and file hashes
hold, adjust = compensation_masks(setup, target)
kw = dict(upper_bound=command_upper_bound(setup, target), hold_mask=hold,
          adjust_mask=adjust, tool_radius=setup.tool_radius)
da = surrogate_compensate(target, setup, model, **kw)  # refused outside the envelope
opt = surrogate_optimize(target, setup, model, start=da, max_time_s=240.0, **kw)
res = simulate(setup, opt.compensated, "work/runs", target=target)  # the verifying run
```

## Results on SparLab simulations

**SparLab simulations (`sparlab_form`, AA5754-O with nominal handbook data),
not experiments.** The ML first shot (surrogate DA on the target, then one
simulation of the compensated shape) against the target formed as it is and FE
displacement adjustment (FE-DA-k: k + 1 runs per part), on the same 8 held-out
test parts (two per family). Vertical RMS deviation over the part, mean over
the 8 parts [mm]:

| benchmark | uncompensated | FE-DA-1 | FE-DA-2 | ML-MLP | dz error (MLP) |
|---|--:|--:|--:|--:|--:|
| 2 mm mesh, single-point, stage n18 (study at commit `7d571e6`) | 0.731 | 0.623 | 0.602 | 0.626 | 0.106 |
| fine mesh, backing plate, stage n18 (48 train / 16 calibration parts) | 0.300 | 0.263 | 0.272 | 0.250 | 0.052 |
| fine mesh, backing plate, stage n26 (72 train / 24 calibration parts) | 0.300 | 0.263 | 0.272 | 0.248 | 0.047 |

On the plate the ML first shot is about as good as one FE-DA step with one FE
run fewer per part: 0.012 mm better on average at n18 (6 of 8 parts), 0.015 mm
at n26 (7 of 8), with a paired standard deviation (0.016-0.017 mm) that 8 parts
do not resolve. The rim sag (about -0.35 mm) dominates what remains. Training
data at penalty 3 (128 samples, about 43 CPU-h, at n18), tests at 10; the 90 %
intervals cover 88 % of the test nodes; at n18 one test target was outside the
envelope (override). The committed stage tables (`stage_n18/`,
`stage_n26/`) and the benchmark README also show an ML-GBM row from the run that
made them; that model has since been removed from the package.

**Combined** (`combined_rim_ml.py`, the 8 test parts of stage n26, penalty
10): DSIF with the rim pass holds the clamp-side band, the transfer MLP with
surrogate DA and the optimiser compensates the deeper part, one verifying run
per part. Mean vertical RMS 0.189 mm (plate: ML-MLP 0.248, FE-DA-1 0.263,
uncompensated 0.300 mm), better than ML-MLP on 7 of 8 parts; upper band 0.129
(ML-MLP 0.367) mm, deep region 0.213 (0.115) mm. The surrogate predicted
0.139 mm: it still over-promises. It is a different, two-tool process with 12
extra training runs.

Details, per-part tables and limitations:
[springback_fine_ml/README.md](../benchmarks/springback_fine_ml/README.md),
[springback_fine_ml/combined_rim/README.md](../benchmarks/springback_fine_ml/combined_rim/README.md),
[springback_fine/README.md](../benchmarks/springback_fine/README.md).

## Limitations

* **Simulation, not experiment.** SparLab with nominal AA5754-O data, 8 test
  parts from four families; no model has been trained on or checked against
  scanned parts.
* **Small data.** A few dozen parts; the part is the unit of evidence. The
  conformal guarantee needs >= 9 calibration parts at 90 %, is conservative
  with a dozen, and holds on average over parts, not for each part.
* **Global effects** (the unclamping springback of the whole sheet) reach the
  point model only through part descriptors and placement features; its
  prediction on the flange is an extrapolation (DA holds the flange).
* **The envelope** uses 99 % quantiles over a few dozen held-out parts: some
  in-distribution targets are refused until overridden; a new family is
  flagged only if its descriptors differ; the reasons are indicative.
* **Data mismatch.** The workflow's model is trained at penalty 3 (no feature)
  and verified at 10, on data with one DA step per part, so later surrogate
  iterates lie further from the data.
* **Transfer** is a gain and an offset of the base prediction: it cannot fix an
  error the base does not predict, needs two real parts, and a
  leave-one-part-out check on a few parts is itself noisy.
* **Predictions over-promise**; only a simulation (or a part) says how good a
  shot is. **Pickles:** load only trusted bundles. **Tool path:** `time_frac`
  rebuilds the setup's path, which the schema version does not track; record
  paths with `generate(..., store_toolpath=True)` when that matters.
