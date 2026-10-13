# Part design

`config.json` defines the parts every benchmark here uses:

- the four families (truncated cone, pyramid, dome, elliptic cone) and their parameter bounds;
- the Sobol seed (2026) and the 0.25 mm height-map grid;
- the split: per family, indices 0-1 are TEST (the 8 held-out parts), then every 4th is CALIBRATION, and the rest TRAIN;
- `solver.setup`, the base forming setup (`springback_2mm` preset, AA5754-O).

It is read by
[`springback_fine_benchmark.py`](../../python/scripts/springback_fine_benchmark.py),
[`springback_fine_ml.py`](../../python/scripts/springback_fine_ml.py),
[`combined_rim_ml.py`](../../python/scripts/combined_rim_ml.py) and the
physics audit.

The 2 mm-mesh study that first used this design is at tag `pre-cleanup`.
