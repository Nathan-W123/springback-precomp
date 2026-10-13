# Combined compensation on pyramid-s2026-0001: DSIF rim pass + transfer model + optimiser

**SparLab simulation** (sparlab_form). **One test part: not statistically meaningful.** Predictions are the surrogate's (the stage_n26 MLP, or that MLP with the transfer correction), not results.

All simulated rows at penalty 10 (`springback_fine` preset). Deviation formed - target, vertical, in mm; upper band = the top 1 mm of depth (next to the clamp), deep = the rest of the part.

| case | kind | FE runs | vertical RMS | upper-band RMS | upper-band bias | deep RMS | deep bias | note |
|---|---|--:|--:|--:|--:|--:|--:|---|
| plate uncompensated | simulated | 1 | 0.308 | 0.478 | -0.464 | 0.130 | -0.012 |  |
| ML-MLP DA (plate) | predicted (MLP) | 0 | 0.268 | 0.264 | -0.218 | 0.271 | 0.212 |  |
| ML-MLP DA (plate) | simulated | 1 | 0.268 | 0.359 | -0.329 | 0.196 | 0.143 |  |
| MLP + optimiser (plate) | predicted (MLP) | 0 | 0.228 | 0.285 | -0.245 | 0.186 | 0.086 |  |
| MLP + optimiser (plate) | simulated | 1 | 0.259 | 0.390 | -0.365 | 0.131 | 0.041 |  |
| DSIF + rim pass uncompensated | simulated | 1 | 0.362 | 0.137 | 0.069 | 0.444 | 0.432 |  |
| DSIF + rim pass uncompensated | predicted (transfer) | 0 | 0.340 | 0.094 | 0.036 | 0.421 | 0.408 |  |
| combined: rim pass + transfer DA | predicted (transfer) | 0 | 0.218 | 0.096 | -0.009 | 0.265 | 0.191 |  |
| combined: rim pass + transfer DA + optimiser | predicted (transfer) | 0 | 0.185 | 0.102 | -0.025 | 0.220 | 0.087 |  |
| combined: rim pass + transfer DA + optimiser | simulated | 1 | 0.216 | 0.143 | 0.066 | 0.250 | 0.096 |  |

FE runs: SparLab runs of this part for that row (the plate rows' ML models were trained on separate data runs; the combined row also needed the 12 rim-pass training runs below).

Combined: surrogate DA (stagnation, 2 s), then the optimiser (5 evaluation batches, 89 s, stop 'no_improvement', in envelope True, largest change from the DA command 0.217 mm). Adjusted: the deep region (2273 nodes, the part minus the 1640-node band the rim pass sweeps); the band and the flange are held at the target. Envelope override: False.

## Base vs transfer model on the rim-pass data

dz prediction error (RMS over the part, mean over parts) of the stage_n26 MLP (base, backing plate, penalty 3) and of the base + linear TransferModel correction, on the rim-pass runs (penalty 10):

| evaluation | parts | base [mm] | transfer [mm] |
|---|--:|--:|--:|
| leave-one-part-out over the fit parts (the model used) | 8 | 0.486 | 0.100 |
| leave-one-part-out, all 12 parts (separate fit) | 12 | 0.499 | 0.104 |
| held out from the correction | 4 | 0.526 | 0.124 |

(dz RMS of the held-out runs: 0.408 mm.) Correction (offset a [m], gain b; y = a + (1 + b) mu): [0.00042010037702198586, -0.18603209592800898]. the 12 parts are TRAIN parts of the base MLP (with the backing plate at penalty 3): 'held out' means held out from the transfer correction, not unseen geometry.

## The 12 training parts (TRAIN split of stage_n26, 3 per family)

Picked per family among the TRAIN parts whose penalty-3 data run took at most the family median (cost cap), as the 3 farthest apart in standardised (footprint, depth, wall angle); the one nearest their centroid is held out from the correction. Uncompensated target, DSIF + rim pass, penalty 10.

| part | role | footprint | depth | wall angle | status | runtime [s] | vertical RMS | upper-band bias | deep bias |
|---|---|--:|--:|--:|---|--:|--:|--:|--:|
| dome-s2026-0010 | fit | 19.60 | 2.42 | 33.7 | ok | 2321.373 | 0.294 | 0.165 | 0.363 |
| dome-s2026-0011 | fit | 15.36 | 3.89 | 54.2 | ok | 2914.066 | 0.389 | -0.074 | 0.461 |
| dome-s2026-0018 | holdout | 15.00 | 2.16 | 39.7 | ok | 1589.775 | 0.305 | 0.100 | 0.401 |
| elliptic_cone-s2026-0008 | fit | 19.07 | 2.27 | 32.4 | ok | 2774.331 | 0.237 | 0.076 | 0.286 |
| elliptic_cone-s2026-0015 | fit | 17.85 | 3.34 | 44.4 | ok | 4013.499 | 0.324 | 0.039 | 0.387 |
| elliptic_cone-s2026-0016 | holdout | 18.11 | 2.03 | 45.5 | ok | 2918.432 | 0.545 | 0.149 | 0.651 |
| pyramid-s2026-0006 | fit | 17.76 | 2.30 | 30.2 | ok | 2814.170 | 0.184 | 0.021 | 0.223 |
| pyramid-s2026-0019 | holdout | 17.91 | 2.05 | 45.6 | ok | 2890.640 | 0.552 | 0.140 | 0.667 |
| pyramid-s2026-0023 | fit | 16.64 | 3.21 | 42.1 | ok | 4206.950 | 0.336 | 0.040 | 0.399 |
| truncated_cone-s2026-0008 | fit | 15.17 | 3.78 | 52.4 | ok | 3877.293 | 0.531 | 0.051 | 0.652 |
| truncated_cone-s2026-0010 | holdout | 17.06 | 3.21 | 31.9 | ok | 2750.509 | 0.229 | 0.064 | 0.275 |
| truncated_cone-s2026-0015 | fit | 16.34 | 2.18 | 50.2 | ok | 4033.888 | 0.392 | 0.024 | 0.493 |

Made by `python/scripts/combined_rim_ml.py` (rim-pass setup = `edge_setup(True)`: benchmarks/springback_fine's dsif strategy with the edge pass). Data set: benchmarks/springback_fine_ml/work/data_rim (kept apart from the plate data).

## All 8 test parts (run_all8.sh)

**SparLab simulation**, penalty 10, the 8 held-out test parts of stage_n26. Vertical RMS of formed - target [mm]. Combined = DSIF + rim pass, transfer MLP (fitted on the 8 rim-pass runs above), surrogate DA + optimiser, one verifying run per part. The other columns are the backing-plate results of stage_n26 (`stage_n26/headline.csv`). All 8 runs completed; no envelope override. Per-part rows with the band/deep split: `all8_results.csv`.

| part | uncompensated | FE-DA-1 | ML-MLP | combined |
|---|--:|--:|--:|--:|
| dome-s2026-0000 | 0.307 | 0.213 | 0.211 | 0.071 |
| dome-s2026-0001 | 0.290 | 0.223 | 0.217 | 0.145 |
| elliptic_cone-s2026-0000 | 0.278 | 0.252 | 0.258 | 0.202 |
| elliptic_cone-s2026-0001 | 0.305 | 0.266 | 0.258 | 0.205 |
| pyramid-s2026-0000 | 0.290 | 0.213 | 0.196 | 0.103 |
| pyramid-s2026-0001 | 0.308 | 0.299 | 0.268 | 0.216 |
| truncated_cone-s2026-0000 | 0.280 | 0.251 | 0.234 | 0.205 |
| truncated_cone-s2026-0001 | 0.344 | 0.386 | 0.344 | 0.365 |
| **mean** | **0.300** | **0.263** | **0.248** | **0.189** |

Mean upper-band RMS: 0.445 / 0.356 / 0.367 / 0.129 mm; mean deep RMS: 0.148 / 0.162 / 0.115 / 0.213 mm. Combined beats ML-MLP on 7 of 8 parts (worse on truncated_cone-s2026-0001). The surrogate predicted a mean of 0.139 mm for the combined commands (simulated 0.189): it still over-promises.
