# Adversarial challenge of the five physics-audit lenses

> **Cleanup note.** The BM3 runs behind finding C6 and recommendation 7 used the explicit forming step
> (`form_explicit`), which was removed in the cleanup; at commit 7d571e6. Their committed outputs in
> `../validation/bm3/` are the record; re-running them needs a build of that commit.

Reviewer lens: I tried to break the claims that decide the verdict. For each claim I opened the
evidence, recomputed the numbers from raw outputs where I could, and reran two cheap cases
(two single-thread runs at a time, as the rules require). Nothing here is committed.

Files in this directory:

| file | what it is |
|---|---|
| `analyze_challenge.py` | measures the reruns and recomputes the 2 mm deltas from the lenses' raw `output/` dirs, using the convergence lens' own metric code (`convergence/analyze.py: case_row`) |
| `results.csv` | output of the above |
| `make_cases.py`, `run_case.sh`, `cases/` | the two reruns: TC on the 0.833 mm 1-layer IM5 mesh (`convergence/cases/h0.833_L1_im_tp5`) with the material lens' `kin_shutov` card (`kin_h0833`) and with the process lens' DSIF support tool (`dsif_h0833`) |
| `robot_fixedpoint.json` | fixed-point residual of the process lens' robot-compliance iteration (made by `robot_fixedpoint.py`, removed in the cleanup; at commit 7d571e6) |
| `bauschinger_check.py/.json` | uniaxial tension→compression response of the `kin_shutov` card vs the library card |

Parts: TC = truncated_cone-s2026-0000. Unless said otherwise, "effect" means the RMS (max) of the change in the released
surface over the part, on the target grid, in mm.

## Verdict in one paragraph

The solver is right and the material is a small error. The 2 mm mesh is a real but moderate error. The open
question is what process the model represents, and no lens measured that against the real cell. Ranked by effect on the
predicted final shape (TC unless noted):

| # | effect | size (mm RMS) | confidence |
|--:|---|--:|---|
| 1 | one-sided model vs a two-sided cell (strategy unknown) | 0.17-0.79 | computed; strategy assumed |
| 2 | robot compliance, if not compensated by the cell | 0.08-0.41 | computed; stiffness assumed |
| 3 | fixture / free span (40 mm blank vs real frame) | 0.22-0.28 per change | computed |
| 4 | 2 mm mesh | 0.08-0.12 | computed and converged, upheld |
| 5 | measurement state / alignment (reporting only) | ≥0.2 | computed, one flaw |
| 6 | material card (incl. Bauschinger) | 0.02-0.05 | computed; converged rerun gives 0.022 for the largest item |
| 7 | solver vs CalculiX | 0.002 | upheld |

"Almost exactly" is achievable for the numerics (about 0.02 mm at 7-11x cost) and for the material (about 0.02-0.05 mm).
It is **not** achievable for the real part until rows 1-3 are pinned down with cell data. Today about 0.3-1 mm on a real
part can be defended; it is not a measured number.

## My two reruns (0.833 mm, 1 incompatible-mode layer, 5 points; the convergence lens' near-converged setup)

`results.csv`; base = `convergence/cases/h0.833_L1_im_tp5`. Both runs finished with no failed increments left
(`cases/*/DONE`: 2051 s and 2472 s on one thread). The 2 mm rows were recomputed from the lenses' raw outputs with the
same code, and they reproduce the lenses' numbers exactly (0.016/0.050, 0.034/0.056, 0.696/1.119).

| change | mesh | released-shape effect RMS (max) | rim-band effect RMS | rim-sag change | springback change | deep bias |
|---|---|--:|--:|--:|--:|--:|
| Bauschinger (`kin_shutov`) | 2 mm std (material lens) | 0.016 (0.050) | 0.017 | -0.013 | -12 % | |
| Bauschinger | 2 mm IM5 (material lens) | 0.034 (0.056) | 0.023 | +0.023 | -31 % | |
| **Bauschinger** | **0.833 mm IM5 (challenge)** | **0.022 (0.041)** | **0.010** | **+0.009** | **-22 %** | |
| DSIF support at target underside | 2 mm std (process lens) | 0.696 (1.119) | 0.955 | +0.951 | +41 % | +0.256 |
| **DSIF support** | **0.833 mm IM5 (challenge)** | **0.786 (1.211)** | **1.077** | **+1.074 (sag -1.05 -> +0.02)** | **+96 %** | **+0.314** |

DSIF contact at 0.833 mm: the support touches 3.5 nodes on average (1.3 at 2 mm). Mean forces are unchanged
(support 578 vs 566 N, primary 713 vs 712 N), and peaks drop 18-25 % (1188 vs 1576 N; 1368 vs 1675 N).

## Claims challenged

### C1 Convergence: "2 mm is not converged; error 0.08-0.12 mm RMS; 0.833 mm IM5 gets about 0.02-0.03". UPHELD, with two corrections
* I recomputed it from `results.csv` and `results_c1.csv`. Benchmark vs 0.625 mm reference: TC 0.099 (0.242), part 2
  0.080 (0.256). Rim sag -1.151 vs -1.038 (TC) and -1.122 vs -1.043 (part 2). Richardson order is about 2 on the IM
  family (Table 3). The reference itself is resolved: IM 1 vs 2 layers changes it by 0.008, and 5 vs 7 points by 0.003.
* Correction 1: about 0.03 mm RMS of the 2 mm error is surface reconstruction (linear facets on 2 mm nodes; the lens' own
  bicubic test), not mechanics. It still counts against the benchmark metric.
* Correction 2: the force error of the 2 mm mesh is part-dependent. Mean Fz is +7 % on TC (566 vs 528 N) but **+24 % on
  part 2** (548 vs 440 N, `results_c1.csv`). Fz is also not monotone in h there (456 / 462 / 440 N at 1 / 0.833 / 0.625 mm).
  The lens quotes only the 7 %, and the process lens carries that 7 % into robot compensation.
* Consequence the lenses missed: **every sensitivity study at 2 mm has about ±30-60 % relative error.** My reruns moved the
  DSIF effect by +13 % and the Bauschinger effect by -36 % (vs 2 mm IM5) or +38 % (vs 2 mm std). Springback changes were
  off by a factor of 2 (DSIF +41 % -> +96 %).

### C2 Crosscode: "SparLab = CalculiX to 2.3 µm", and "1 mm mesh cuts shape error to 0.0045 mm". First UPHELD, second REFUTED
* `punch/plate_compare_plate_h2_im_mu0.1_drag3.csv`: IM Hex8 vs C3D8I gives release_dz 0.62 µm RMS / 2.25 µm max, and
  plunge force 1216 vs 1214 N. The test uses J2, isotropic hardening and one tool, so it verifies the code. It does not
  check Hill48, Chaboche, two tools or the fixture.
* The "0.0045 mm at 1 mm" figure is measured against `plate_h1_std_nz4`, which has the **same 1 mm in-plane size**. It
  shows layer convergence, not mesh convergence. On the real spiral parts the convergence lens measured 1 mm std at
  **0.035 (0.083)** from the converged reference, 8x more.
* The 3D test's 2 mm error (0.038) understates the real-part error (0.099) by 2.6x: a 3 mm plunge plus a 3 mm drag is not
  a multi-pass spiral. **Recommendation 1 of crosscode (1 mm std) is weaker than the convergence lens' 0.833 mm IM5**
  (0.014 vs 0.035 RMS from the reference at similar cost: 1586 vs 1601 s).

### C3 Process: "one tool vs the real two-robot cell is the biggest gap, 0.65-0.81 mm". UPHELD in magnitude, WEAKENED as a statement about the real cell
* It is robust to the mesh: 0.786 at 0.833 mm vs 0.696 at 2 mm, and the 1-2-node contact caveat is resolved (3.5 nodes).
* But this is a best case: a position-controlled support placed exactly at the CAD underside, i.e. a moving die. With
  0.2 / 0.5 mm clearance the lens' own runs give 0.475 / 0.172 (`process/results.csv`). The real strategy is not public,
  so the honest range is **0.17-0.79 mm**, and the number to quote is that range.
* What the lens did not report:
  * The support **doubles the springback** (0.132 -> 0.259 mm RMS, +96 % at 0.833 mm).
  * It moves the error from the rim to the floor: deep bias goes from -0.20 to **+0.31** (too shallow), and rim sag
    overshoots to +0.02.
  * So in a two-sided cell springback, and with it kinematic hardening and E, becomes a larger share of the error than
    the one-sided benchmark suggests.
  * The process lens' 0.28 RMS-vs-target is not a "better model". It is a different process with its own uncompensated
    error.

### C4 Process: "robot compliance 0.41 mm RMS". Computation UPHELD, headline WEAKENED to 0.08-0.41
* The fixed point is sound. `robot_fixedpoint.json` gives the mean |Fz| mismatch between input and output forces of
  `rob_iso1_it5`: -1.6 / -2.0 / -3.6 % over path fractions 0-0.5 / 0.5-0.8 / 0.8-1. The pointwise residual is 17-28 %
  (contact noise, smoothed over 3 mm).
* Literature check: Bharti et al. 2024, Table 5 (`validation/literature/sr2024_roboforming.txt`, read in full) gives a
  robot-compliance resultant mean deviation of 0.908 mm (cone) and 1.048 mm (VW cone), and a z-compliance kc33 of about
  2.5 µm/N. The process lens' citation is correct. Context: AA1050, about 380 N, horizontal sheet, 100 mm deep cones.
  The same paper reports the VJM model **mispredicting** by 0.8-1.0 mm. A compliance model without measurement is
  therefore not trustworthy, and the IRB-VJM numbers below are equally uncertain.
* 1 µm/N is an assumption. The lens' own IRB 7600 VJM with a vertical sheet gives 0.076 mm, and a horizontal sheet gives
  0.220. The 2 mm-mesh forces are 7-24 % (mean) too high (C1).
* Open-loop only: the company runs closed-loop force and scan correction. **Range 0.08-0.41 mm, unmeasured.** It is still
  ≥ the whole springback of these parts, so it belongs in the model or in the cell's calibration.

### C5 Experiment: "kinematic hardening / pitch are worth about ±0.3 mm of the benchmark rim sag". REFUTED (about 30x too large)
* Direct measurement at the converged mesh: kinematic hardening changes the rim sag by **+0.009 mm** (rim-band field
  0.010 RMS). At 2 mm it changes it by -0.013 (std) or +0.023 (IM5).
* Pitch: halving the step-down (`process/results.csv`, `stepdown05`) moves the whole shape 0.032 RMS.
* Why the inference fails: the rim sag is formed in, not sprung back. Loaded rim bias is -1.03 vs released -1.15 (TC),
  and convergence Table 2 shows the formed-shape change carrying the field. A 30 % change in a 0.12-0.15 mm springback
  is 0.04 mm at most, not 0.3.

### C6 Experiment: "engine 1.4 mm RMS from NUMISHEET BM3; no published model does much better; the 1.6 mm offset is experimental; forces 25-60 % high". WEAKENED
* The replica is far from converged:
  * the pitch is 4 mm vs the experiment's 0.5 mm (8x);
  * the elements are 2.5 x 4.1 mm under a 6.33 mm-radius tool, where the convergence lens' rule is ≤ 1.3 mm;
  * it is explicit at 20 m/s, with von Mises instead of Yld91;
  * 4 -> 2 mm pitch alone moved the floor RMS from 0.95 to 0.44 mm and springback by -32 % (`bm3/regions.csv`,
    `bm3/springback.csv`).
* So 1.4 mm describes this replica, not the engine.
* The narrower measured cone is real and documented: the paper itself says "the diameter of the cone is overestimated
  both by the CAD geometry and by the numerical simulation" (`numisheet2014_bm3_*.txt`, about line 482). Calling it an
  experimental error is inference. It could equally be a path-definition convention (tool centre vs contact) that both
  models share.
* "Rim bend predicted without bias" was measured at unconverged settings. On the benchmark, 2 mm-class meshes over-sag
  the rim by 0.08-0.15 mm (C1), so the agreement may come from cancelling errors.
* The "+25-60 % force" figure comes from AA7075 with the wrong yield locus on that coarse mesh. It should not be
  transferred to the AA5754 benchmark, where mesh alone moves the force -21 % (IM 2 mm) to +24 % (std 2 mm, part 2).

### C7 Material: "material share 0.02-0.05 mm; Bauschinger is the largest material effect and grows as the model gets more accurate". Share UPHELD; "grows" REFUTED; card is an upper bound
* Recomputed: `sensitivity.csv` matches my recomputation from raw outputs.
* Converged rerun: 0.022 (0.041), springback -22 %. That is below the 2 mm IM5 value (0.034, -31 %), so the 2 mm IM5
  overstated it 1.6x. It is still the largest material item, and it still does not touch the rim sag (+0.009).
* The card is aggressive (`bauschinger_check.json`, uniaxial):
  * the 0.2 %-offset reverse yield is 0.31 / 0.42 / 0.50 / 0.56 of the forward flow stress after 2 / 5 / 10 / 20 %
    prestrain (library card: 1.00);
  * after 2 % prestrain the card re-yields **while still in tension (+16 MPa)**;
  * the initial elastic limit is 29.5 MPa, against 100 MPa in the library.
* The lens' own source [1] reports "a very weak Bauschinger effect" in reversed shear. Treat `kin_shutov` as an upper
  bound and expect ≤ 0.02 mm. Literature search for AA5754-O tension-compression data was inconclusive (not verified).
* "Shape is independent of absolute strength" is true but trivial (rate-independent homogeneity, `scaled090` exact to
  1e-13). It holds only for a rigid robot. With compliance (C4), a 10 % strength error is about 10 % of the deflection:
  ≈0.05 mm at 1 µm/N.

### C8 Process: "measurement state / alignment changes the reported error by up to 0.2 mm (3-DOF 0.526, 6-DOF 0.659)". Magnitude UPHELD as a lower bound; 6-DOF number INVALID
* A 6-DOF best fit cannot score worse than its own 3-DOF subset under the same objective. `measure_states.csv` shows why
  it does: the part-only ICP rotates the parts by **8.9-16.2°** (`icp6_part_rot_deg`), which is a failed registration.
  The whole-surface ICP minimises point-to-plane distance over the flange too, so it is a different objective.
* Report 3-DOF or a proper least-squares 6-DOF fit. The reporting swing is ≥ 0.2 mm (0.73 -> ≤ 0.53).

### Not challenged in depth (spot-checked only)
* Process fixture and scale runs (`clamp8`, `blank60`, `scale2`): same 2 mm mesh, so ±30 % relative per C1.
* Path-offset 0.22 mm: a constant 0.3 mm z offset is the worst case for RMS. A calibrated robot's error varies along the
  path and is partly removed by best fit.
* Literature values: ESAFORM 2024 Swift K = 408.52 MPa, ε0 = 0.001, n = 0.25 and r = 0.6539 / 0.8799 / 0.6840 are
  verified in `validation/literature/esaform2024.txt` (Table 9 and text). The library curve is within -10 / +9 % of it
  (my check: -10 % at εp = 0.05, -3.5 % at 0.2, 0 % at 0.5).

## Recommendations (cost)
1. **Get the cell's real process before tuning the model further** (no compute; needs the company): the two-robot strategy
   (support, squeeze, force or position control), logged tool paths and forces for 3 parts, scans clamped and released.
   Rows 1-3 of the table are worth 0.2-0.8 mm each and none can be settled in simulation. About 1 week of cell time.
2. **Default mesh: 0.833 mm, 1 incompatible-mode layer, 5 points** (convergence lens rec. 1; 7-11x cost). Use penalty 3 for
   bulk data (4-7x). Redo the key sensitivity studies (DSIF clearance, compliance, fixture) on it: about 40 min per run,
   10 runs ≈ 7 CPU-h. Their 2 mm magnitudes carry ±30-60 %.
3. **If the cell is two-sided, build the non-gouging support path generator** (2-3 days) and a force-controlled support
   (1-2 weeks). Then springback doubles in importance, so do rec. 5.
4. **Measure each robot's stiffness** (laser tracker plus known loads, 1-2 days per robot) and add the compliance fixed point
   (works today: 3-5 runs per part). Do not trust VJM alone: Bharti et al. found 0.8-1.0 mm VJM error.
5. **Material**: one tension-compression or cyclic-shear test on the production coil (EUR 3-5k) to replace the
   upper-bound `kin_shutov` card. Expected payoff is ≤ 0.02 mm on these parts. It is low priority unless parts get larger
   or the cell is two-sided.
6. **Fix the 6-DOF alignment** in the process lens' measurement (least squares with a rotation sanity check; hours) and
   report the shop's measurement state.
7. **Do not quote the BM3 1.4 mm as engine accuracy.** A converged replica needs about 18 h explicit at 0.5 mm pitch,
   or validate on DB4ISF / in-house parts instead.
