# springback-precomp: first-time-right incremental sheet forming

**Predict how an incrementally formed sheet-metal part will spring back, and
form a corrected shape instead, so the first part lands closer to the target.**

In incremental sheet forming a tool pushes a clamped sheet into shape, level
by level. When the tool leaves and the clamp is released, the part **springs
back** and comes out shallower and bent. This repository replaces the
form-scan-correct loop with computation:

* **`sparlab_form`** (C++): a finite-element forming simulator. Rigid tools
  follow a tool path in frictional penalty contact with a Hex8 sheet (Hill48
  or J2 plasticity, isotropic / Prager / Chaboche hardening, finite strain
  in the logarithmic strain); a step sequence forms the part, removes the
  tools and releases the clamp onto 3-2-1 supports.
* **`precomp`** (Python, `python/precomp`): part geometry, tool paths,
  forming decks and a run cache, deviation metrics, and displacement
  adjustment (DA) compensation.
* **`precomp.ml`**: an ensemble of five MLPs trained on `sparlab_form`
  simulations, with conformal intervals and a training-range check, a
  transfer model that adapts it to a new process from a few runs, and an
  optimiser that searches the corrected shape on the surrogate.

**Status: an independent project. Every number below is a SparLab
simulation with nominal material data. Nothing has been compared with a
formed and scanned part.**

## Headline result (simulation)

![Shape error after springback on 8 held-out test parts](docs/figures/headline_results.png)

8 held-out test parts (two each of dome, truncated cone, elliptic cone,
pyramid; 15-20 mm footprint, 2-4 mm deep, 1 mm AA5754-O), fine-mesh
`sparlab_form` model, one forming run per part unless stated:

| Method | Mean vertical RMS error | Simulations per part | Source |
|---|--:|--:|---|
| No correction (backing plate) | 0.300 mm | 1 | [`springback_fine/`](benchmarks/springback_fine/README.md) |
| FE displacement adjustment, one round (FE-DA-1) | 0.263 mm | 2 | [`springback_fine/`](benchmarks/springback_fine/README.md) |
| FE displacement adjustment, two rounds (FE-DA-2) | 0.272 mm | 3 | [`springback_fine/`](benchmarks/springback_fine/README.md) |
| Two-tool forming, no correction | 0.273 mm | 1 | [`springback_fine/`](benchmarks/springback_fine/README.md) |
| ML first shot (MLP ensemble, backing plate) | 0.248 mm | 1 | [`springback_fine_ml/`](benchmarks/springback_fine_ml/README.md), stage n26 |
| **Two-tool forming + edge pass + ML + optimiser** | **0.189 mm** | **1** | [`combined_rim/`](benchmarks/springback_fine_ml/combined_rim/README.md) |

* 37 % less error than no correction, and better than a round of FE
  displacement adjustment with half the simulations. It beats the ML first
  shot on 7 of 8 parts.
* Most of the remaining error used to sit in the band next to the clamp
  (0.45 mm RMS), which correcting the commanded shape cannot reach.
  Two-tool forming with an edge pass holds that band (0.13 mm); the
  surrogate and optimiser correct the deeper region.
* The surrogate learned the two-tool process from 8 extra simulations (a
  linear transfer correction on the frozen model, not a retrain; 4 more
  runs were held out to score it).
* Honest gaps: the surrogate over-promises (predicted 0.139 mm against
  0.189 mm simulated); one part got slightly worse than the ML first shot;
  the interior error rose (0.15 to 0.21 mm) while the edge error fell; and
  8 parts is a small sample.

**How the 8-part result was assembled.** `combined_rim_ml.py all` ran once,
then `combined_rim_ml.py one` for each of the 8 test parts
([`run_all8.sh`](benchmarks/springback_fine_ml/combined_rim/run_all8.sh)).
The 8-part table (`combined_rim/all8_results.csv` and the "All 8 test parts"
section of its README) was assembled by hand from the per-part
verification records in commit `3a4556c`; no script in the repository
writes it, and `combined_rim_ml.py report` writes a one-part table only.
The figure above was drawn from that CSV outside the repository's scripts.
The committed stage tables of `springback_fine_ml/` also still contain an
`ML-GBM` row: they were written by the run that made them, when a
gradient-boosting ensemble was trained alongside the MLPs. That model has
since been removed (see [History](#history)).

## How it fits together

```
 design: 4 part families, Sobol draws          (precomp.ml.generate, precomp.geometry)
        |
        v
 target height map -> spiral tool path -> forming deck -> sparlab_form -> formed + released shape
        (precomp.toolpath)        (precomp.fea.deck)    (C++)          (precomp.fea.results)
        |                                                                     |
        |                    training samples: commanded, formed, setup <-----+
        v                                          |
 precomp.ml: 57 features per node -> 5-MLP ensemble -> conformal intervals, training envelope
        |                                          |
        |            transfer_surrogate: linear correction from a few two-tool runs
        v                                          v
 surrogate DA (surrogate_compensate) -> surrogate_optimize (about 30 smooth basis shapes)
        |
        v
 compensated command -> one verifying sparlab_form run -> deviation by region (precomp.metrology)
```

| Stage | Code |
|---|---|
| Design of the parts | `ml/generate.py` (`DesignSpace`, `design_points`); `geometry/parts.py` (truncated cone, pyramid, dome, elliptic cone); settings in [`benchmarks/springback/config.json`](benchmarks/springback/config.json) |
| Simulations | `ml/generate.py` `SparlabSimulator` -> `fea/setup.py` -> `toolpath.py` (spiral, two-tool support path, edge pass) -> `fea/deck.py` -> `fea/runner.py` (content-addressed run cache) -> `sparlab_form` -> `fea/results.py` |
| Features and data | `ml/features.py` (57 per node), `ml/dataset.py` |
| Training | `ml/surrogate.py` `train_surrogate` -> `ml/models.py` `MLPEnsemble`; `ml/uncertainty.py` (conformal calibration over whole parts), `ml/ood.py` (training envelope), `ml/registry.py` (bundles) |
| Transfer to two-tool + edge pass | `ml/surrogate.py` `transfer_surrogate`, `ml/models.py` `TransferModel` |
| Compensation | `ml/compensate.py` (`surrogate_compensate`, `surrogate_optimize`), `compensation.py` (DA, `FEAPredictor`), `fea/support.py` (masks and bounds of the supports) |
| Evaluation | `metrology.py`, `ml/evaluate.py`, `python/scripts/common.py` (`deviation` by region) |

## Repository map

| Path | What it is |
|---|---|
| `src/`, `include/`, `apps/sparlab_form.cpp` | the C++ forming simulator ([`docs/forming.md`](docs/forming.md)) |
| `tests/` | C++ unit tests of the kept engine code (ctest) |
| `configs/forming/` | small forming decks: the CI smoke case and examples |
| `python/precomp/` | the Python package ([`docs/precomp.md`](docs/precomp.md), [`docs/precomp_ml.md`](docs/precomp_ml.md)) |
| `python/tests/` | pytest suite, including integration tests against the real `sparlab_form` |
| `python/scripts/springback_fine_benchmark.py` | baselines: uncompensated and FE-DA runs of the 8 test parts |
| `python/scripts/springback_fine_ml.py` | training data, the 5-MLP surrogate, the ML first shot |
| `python/scripts/combined_rim_ml.py` | the headline method: transfer to two-tool + edge pass, DA + optimiser, verification |
| `python/scripts/common.py` | helpers shared by the three scripts |
| `benchmarks/springback/` | the design configuration (families, bounds, seed) the other benchmarks read |
| `benchmarks/springback_fine/` | baseline results (`cases.csv`, `summary.csv`, `tables.md`) |
| `benchmarks/springback_fine_ml/` | ML results; `combined_rim/` holds the headline method's results |
| `benchmarks/physics_audit/` | validation of the simulator: mesh convergence, material and process sensitivity, published experiments, CalculiX cross-check |
| `docs/writeup/` | the write-up as sent (describes the repository at commit `7d571e6`) |

## Quick start

```bash
# 1. Build sparlab_form (Eigen 3.4 and a C++17 compiler; SuiteSparse is
#    optional and speeds up the forming analysis: apt-get install libsuitesparse-dev)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure

# 2. Install precomp with the MLP ensemble (CPU torch) and run its tests
#    (the sparlab_form integration tests run because build/bin/sparlab_form exists)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[dev,torch]'
python3 -m pytest python/tests -q

# 3. The forming smoke case on its own (about 75 s with SuiteSparse)
./build/bin/sparlab_form --config configs/forming/spif_smoke.json --output results/spif_smoke
```

### Reproducing the results

Each fine-mesh forming simulation takes about 45-90 minutes on one CPU core,
so the full chain is days of CPU time. Every script is resumable: runs go
through a content-addressed cache in `--work` (git-ignored), and an
identical deck never runs twice.

```bash
export OPENBLAS_NUM_THREADS=1 PYTHONPATH=python
# baselines (uncompensated, FE-DA-1, FE-DA-2) of the 8 test parts, and FE-DA-1
# on the fixed decks the ML runs use
python3 python/scripts/springback_fine_benchmark.py \
    --out benchmarks/springback_fine --work benchmarks/springback_fine/work
python3 python/scripts/springback_fine_benchmark.py --strategies backing_plate --fe-runs 2 \
    --tool-reach --out benchmarks/springback_fine_ml/fe_da_fixed --work benchmarks/springback_fine_ml/work
# training data, the MLP surrogate and the ML first shot (stages = parts per family)
python3 python/scripts/springback_fine_ml.py \
    --out benchmarks/springback_fine_ml --work benchmarks/springback_fine_ml/work --stages 18 26
# the headline method on one part, then on all 8 (see run_all8.sh)
python3 python/scripts/combined_rim_ml.py all
bash benchmarks/springback_fine_ml/combined_rim/run_all8.sh
```

The scripts' docstrings give the full protocol (splits, setups, stopping
rules). The trained models and run caches are not committed.

## Validation summary

From [`benchmarks/physics_audit`](benchmarks/physics_audit/README.md)
(simulation studies; every claim there was re-tested by an adversarial
"challenge" pass):

* **Solver.** Where SparLab and CalculiX solve the same problem, the released
  shapes agree to 2 um.
* **Mesh.** The coarse 2 mm mesh errs by 0.08-0.12 mm RMS; the fine mesh used
  for the headline (0.833 mm incompatible-mode Hex8, 5 points through the
  thickness) is converged to about 0.02-0.03 mm at 7-11 times the cost.
* **Material.** The library AA5754-O card contributes 0.02-0.05 mm.
* **What is not modelled.** The real cell uses two robots; robot compliance,
  the full-sheet fixture, trimming and heat treatment are not modelled. Each
  of the first three is worth 0.2-0.8 mm on these parts, so about 0.3-1 mm
  on a real part is the defensible estimate today, not a measurement.

## Limitations

* **No experiment.** No simulated or compensated shape has been compared with
  a formed and scanned part. The first step with real data would be one
  scanned part: check the simulation, then fit the transfer model to scans.
* **Small, simple parts.** 15-20 mm footprint, height-field shapes from four
  families, one material (AA5754-O, nominal handbook data), one spherical
  tool.
* **Small data.** About 100 simulated parts per model; 8 test parts.
* **The ML is only as good as the simulations**, and the optimiser exploits
  model error (it predicts better results than the simulations deliver).
* Details: [`docs/precomp.md`](docs/precomp.md#limitations),
  [`docs/precomp_ml.md`](docs/precomp_ml.md#limitations),
  [`docs/forming.md`](docs/forming.md) and
  [`docs/limitations.md`](docs/limitations.md).

## Documentation

| Document | Contents |
|----------|----------|
| [`docs/precomp.md`](docs/precomp.md) | the Python package: conventions, geometry, materials, tool paths, the forming deck and run cache, metrology, compensation, supports, verification |
| [`docs/precomp_ml.md`](docs/precomp_ml.md) | the learning problem, features, data, the MLP ensemble, transfer, conformal intervals, the training envelope, bundles, surrogate compensation and optimisation, results |
| [`docs/forming.md`](docs/forming.md) | the forming analysis `sparlab_form`: tools, contact and friction, steps, release, linear algebra, deck and output contract, verification, cost |

## History

This repository was trimmed on 2026-10-13 to the code the workflow above
uses. Everything removed - the general-purpose SparLab engine (static, modal,
buckling and thermal analyses, topology optimisation, other element types),
the U-Net and gradient-boosting models, the follow-up experiments on top of
the headline method, the `precomp` command line, the proxy-data
demonstration and the engine figures - is in the history at commit
[`7d571e6`](https://github.com/Nathan-W123/springback-precomp/tree/7d571e63af6c899717d49f95f4d7842c9fd7c682).

## Licence

MIT. See [`LICENSE`](LICENSE).
