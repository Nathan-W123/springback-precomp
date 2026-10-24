# Physics audit of the forming/springback simulation: decision document

Question asked: is our physics "extremely sound", and does it simulate robotic incremental sheet forming
(two robots, one on each side of a clamped sheet, then release/trim, springback) "almost exactly"?

Five lenses audited the engine, and an adversarial challenge then tested their claims:
[`convergence/`](convergence/README.md), [`material/`](material/README.md), [`process/`](process/README.md),
[`validation/`](validation/README.md) (published experiments), [`calculix/`](calculix/README.md) (cross-code), and
[`challenge/`](challenge/README.md). This document states only claims that survived the challenge. Claims the
challenge weakened carry their caveat. Claims it refuted are listed in section 6 so that nobody reuses them.

Abbreviations:
- TC = `truncated_cone-s2026-0000`, the 38.6° cone used as the main test part.
- Part 2 = `truncated_cone-s2026-0001`, a 54° cone.
- IM5 = incompatible-mode Hex8 with 5 integration points through the thickness.
- "Effect" = RMS (worst point in brackets) of the change in the released surface over the part, in mm.

## 1. Verdict

The solver is correct. Where SparLab and CalculiX solve the same problem, their released shapes agree to 2 µm.
The material card is a small error, 0.02–0.05 mm. The benchmark's 2 mm mesh is a real error of 0.08–0.12 mm RMS
(up to 0.26 mm at the worst point), about as large as the springback it is meant to predict. A finer mesh fixes this
at 7–11× the cost per run.

The physics we do model is sound. We do not yet model the process the company actually runs. Three gaps are each worth
0.2–0.8 mm on these 14–22 mm parts, and the largest effects grow with part size:
- **One tool vs two robots.** We simulate one tool pressing from above. The company's cell uses two robots, one on
  each side of the sheet. Effect: 0.17–0.79 mm, depending on a strategy that is not public.
- **Robot compliance.** We treat the robot as rigid. Effect: 0.08–0.41 mm, because the robot stiffness has not been
  measured.
- **Fixture.** We use a small blank with a 5 mm clamp; the real cell holds the whole sheet in a frame. Effect:
  0.22–0.28 mm per change.

Trimming, heat treatment and the cell's closed-loop corrections are not modelled at all. No lens compared the engine
with a part formed in the company's cell.

Today, about 0.3–1 mm on a real part is all we can defend, and even that is an estimate, not a measurement. Our own
published-experiment replica is not converged enough to measure it (section 3).

What "almost exactly" would take:
- **Numerics: possible now.** About 0.02–0.03 mm at 7–11× the run cost.
- **Material: within reach.** 0.02–0.05 mm now; about 0.01 mm with coupon tests costing EUR 3–6k.
- **Real part: needs the company's cell.** We need its two-robot strategy, logged paths and forces, the robots'
  measured stiffness, and scans of a few parts both clamped and released. With these, 0.1–0.3 mm RMS is a realistic
  target at this part size (process-lens estimate, inferred). The robots' calibrated accuracy (0.1–0.3 mm) and the
  scanner's 0.5 mm point accuracy (from the company's patent, verified) put a floor under any measured agreement.

## 2. Error budget (effect on the predicted released shape, mm)

How to read the table:
- The sizes are for the benchmark's 14–22 mm AA5754-O parts.
- Unless marked otherwise, sensitivities were computed on the 2 mm mesh. The challenge found that such sensitivities
  carry about ±30–60 % relative error ([`challenge/README.md`](challenge/README.md) C1).
- **"Survived"** says what the challenge concluded about the claim.

| # | Source | Effect, mm RMS (max) | Confidence / survived | Evidence |
|--:|---|---|---|---|
| **Process gaps (not modelled today)** | | | | |
| P1 | One tool from above vs a two-sided cell | **0.17–0.79** (≤1.2). The upper end needs a support exactly at the target's underside; 0.2 / 0.5 mm clearance gives 0.48 / 0.17. At 0.833 mm IM5 the effect is 0.786 (1.211). The support doubles springback (0.132 → 0.259) and moves the error from the rim to the floor (+0.31 too shallow). | Size **upheld** and mesh-robust. As a statement about the real cell it is **weakened**: the company's strategy is unknown. | [`process/results.csv`](process/results.csv) (`dsif_ng*`), [`challenge/results.csv`](challenge/results.csv) (`dsif_h0833`) |
| P2 | Robot compliance (the model assumes a rigid robot) | **0.08–0.41** (0.59), and the part comes out up to 0.54 mm shallower. 0.41 assumes 1 µm/N. A modelled IRB 7600 with a vertical sheet gives 0.08, with a horizontal sheet 0.22. The stylus bends another 0.07–0.16 sideways. A published study measured 0.91–1.05 mm from compliance (Bharti 2024). | The fixed-point calculation is **upheld** (forces agree to 1.6–3.6 % mean). The headline is **weakened** to a range: stiffness unmeasured, and the cell's closed-loop correction is ignored. | [`process/robot_deflection.csv`](process/robot_deflection.csv), [`process/robot_compliance.json`](process/robot_compliance.json), [`challenge/robot_fixedpoint.json`](challenge/robot_fixedpoint.json) (the robot-compliance scripts were removed in the cleanup; at commit 7d571e6; these files are the committed record) |
| P3 | Fixture / free span: our 40 mm blank with a 5 mm clamp vs the real frame with "skirting" | **0.22–0.28 per change**. The rim sag grows with the free span: 0.84 / 1.15 / 1.55 for a 24 / 30 / 50 mm window. | Medium (computed at 2 mm, ±30 %) | `process/runs/*/{clamp8,blank60}` |
| P4 | Part scale (the company makes parts up to 3.7 m) | Doubling part and frame raises the deviation from 0.77 to 1.45 RMS and the rim sag to 2.34. Absolute errors do not transfer from these 14–22 mm parts. | Medium (one run) | `process/runs/truncated_cone-s2026-0000/scale2` |
| P5 | Robot path accuracy (0.1–0.3 mm after calibration) | ≤0.22 (0.29 in depth). This is the worst case, a constant 0.3 mm z offset. | Medium | `process/runs/truncated_cone-s2026-0000/offset_z03` |
| P6 | Trimming | Not computable: the engine has no element removal and no stress-tensor output. Inferred to be of the order of the release springback: 0.1–0.2 here, millimetres on large parts. | Low (inference) | [`process/README.md`](process/README.md) |
| P7 | Post-form heat treatment (stress relief / temper) | Not computable (no thermal or relaxation model). Unknown size. | none | [`process/README.md`](process/README.md) |
| P8 | Closed-loop path correction in the cell (force, displacement and in-process scans) | Not quantified. An open-loop simulation of the nominal path predicts a part the cell never makes. | Unquantified | [`process/README.md`](process/README.md) gap table |
| P9 | Gravity | 1.4 µm on our blank. At product scale: 0.3–1.9 mm for a sheet clamped horizontally; 70–280 mm for a 0.6–1.5 m sheet resting on 3 points. | Medium (CalculiX) | [`process/gravity/`](process/gravity/) |
| P10 | Tool radius 6 mm vs 4 mm / step-down 0.5 mm vs 1 mm / H32 temper vs O | 0.14 / 0.032 / 0.07 (the H32 temper doubles springback) | Medium | `process/runs/truncated_cone-s2026-0000/{tool6,stepdown05,mat_5052H32}` |
| **Discretisation** | | | | |
| D1 | In-plane element size (2 mm standard Hex8, the benchmark mesh) | **0.08–0.12 (0.11–0.26)**. The RMS deviation from target is overstated by 0.06–0.11 and the rim sag by 0.09–0.15. About 0.03 of this is linear surface reconstruction in metrology, not mechanics. The springback is under-predicted by 8–9 %. | High, **upheld** (2 parts) | [`convergence/README.md`](convergence/README.md) Tables 1, 4–6; `convergence/diffs*.csv` |
| D2 | The 2 mm result looks close only because errors cancel: standard-Hex8 locking offsets coarse-mesh error. The locking-free element at 2 mm is worse (0.11–0.15). | (part of D1) | High | `convergence/results*.csv` |
| D3 | Forces on the 2 mm mesh | Mean force +7 % (TC) or +24 % (part 2), so robot-compensation inputs are wrong by up to 0.2 mm at 1 µm/N | High, corrected by the challenge | `convergence/results_c1.csv`, challenge C1 |
| D4 | Layers, thickness points, tool travel per increment, Newton tolerance, contact penalty 10 → 30, release increments | ≤0.008 (≤0.027) each: converged | High | [`convergence/README.md`](convergence/README.md), [`calculix/README.md`](calculix/README.md) |
| D5 | Penalty 3 instead of 10 | 0.005–0.009 (about 0.02), always towards more sag; halves the cost | High | `convergence/results.csv` |
| D6 | Explicit vs implicit | 0.017–0.019 on the shape, but 13–20 % less springback, and the gap grows with refinement | Medium | `convergence/cases/h0.625_L2_std_x05` (the explicit step was removed in the cleanup; at commit 7d571e6; the committed outputs are the record) |
| **Material parameters (library card vs the published AA5754-O spread)** | | | | |
| M1 | Yield-locus shape. r-based Hill48 gives a biaxial-to-uniaxial yield ratio of 0.95; AA5754-O is measured at 1.0–1.1. | 0.027–0.042 (≤0.066); springback −6 to −20 % | Medium, upheld | `material/sensitivity.csv` (`hill_sb110`, `vm`) |
| M2 | Hardening curve beyond tensile-test strains (27–46 % of elements exceed plastic strain 0.19) | 0.007–0.016 (≤0.030) | Medium, upheld | `material/sensitivity.csv` (`hard_*`) |
| M3 | E ±10 %, initial yield ±15 MPa, measured r-value sets, 5 % thinner sheet | ≤0.011 each | High, upheld | `material/sensitivity.csv` |
| M4 | Absolute strength level | 0 for shape: scaling every stress by 0.9 reproduces the shape to 1e-13 mm. It acts only through force, i.e. through robot deflection (about 0.05 mm per 10 % at 1 µm/N). | High | `material/results.jsonl` (`scaled090`) |
| **Missing material physics** | | | | |
| M5 | Bauschinger effect (no kinematic hardening) | **0.022 (0.041)** at 0.833 mm IM5; rim sag +0.009; springback −22 %. The `kin_shutov` card is an upper bound: it reverse-yields at 0.31–0.56 of the forward stress, and its source calls AA5754-O's Bauschinger effect "very weak". Expect ≤0.02. Springback is twice as large in a two-sided cell, so the effect matters more there. | Medium. "Grows with model accuracy" was **refuted**. | `challenge/results.csv` (`kin_h0833`), [`challenge/bauschinger_check.json`](challenge/bauschinger_check.json) |
| M6 | Drop of E with plastic strain (−15 % at saturation, emulated) | 0.005–0.010; springback ≤2 % | Medium | `material/sensitivity.csv` |
| M7 | Strain rate (0.7–10 /s) and adiabatic heating (≤35 K) | ≤0.011; springback ≤5 % | Medium (rate sensitivity bound assumed) | `material/rate_temperature.json` |
| M8 | Lüders / serrated-flow banding | Affects surface quality, not shape. Cannot be modelled. | Low | [`material/README.md`](material/README.md) |
| **Contact / friction** | | | | |
| C1 | Friction 0.05–0.2 (benchmark 0.1) | ≤0.003 | High | `material/sensitivity.csv` |
| C2 | Linear-element point contact vs quadratic surface contact | 6 % lower force; released floor 0.03 deeper (plane-strain strip) | Medium | `calculix/strip_reference/*.csv` |
| C3 | Lower tool on a 2 mm mesh (touches 1.3 nodes) | Resolved at 0.833 mm (3.5 nodes): the DSIF effect moves +13 % | High | challenge C3 |
| **Solver** | | | | |
| S1 | SparLab vs CalculiX on the identical problem (J2, isotropic hardening, one tool) | 0.6 µm RMS / 2.3 µm max; force within 0.7 % | High, **upheld**. Hill48, Chaboche and two-tool cases are not cross-checked. | `calculix/punch/plate_compare_plate_h2_*_drag3.csv` |
| S2 | Logarithmic-additive vs multiplicative plasticity | ≤0.3 % in stress (tension); −2.2 % at shear γ = 1. Negligible at plastic strain ≤0.4. | High | `calculix/homogeneous/results_inc400.csv` |
| **Measurement / alignment** | | | | |
| A1 | What gets measured: released on 3-2-1 supports (benchmark) vs scanned in the cell and best-fitted to CAD (shop) | ≥0.2 on the reported number with no physics change (0.73 released → 0.53 with a 3-DOF best fit). The 6-DOF figure (0.659) is invalid: the registration failed, rotating the parts 9–16°. | Upheld as a lower bound | `process/measure_states.csv`, challenge C8 |
| A2 | Scanner point accuracy | 0.5 per point (patent example) | Verified source; it is a noise floor, not a bias | `process/README.md` [S3] |

Quadrature roll-up (inference):
- Engine and material on the benchmark mesh: about 0.09–0.13.
- Engine and material on the recommended mesh: about 0.03–0.06.
- The process gaps P1–P3 combined: about 0.35–0.9, dominated by P1, the strategy question.

## 3. Validation

### 3a. Against published experiments ([`validation/`](validation/README.md))

**Case.** NUMISHEET 2014 Benchmark 3 (Neto et al. 2016, https://doi.org/10.1007/s00170-015-7954-9): single-sided
incremental forming of a 45° truncated cone in AA7075-O, 1.63 mm sheet, 44 mm deep, 12.66 mm tool. The measured
profiles and axial-force history were digitised from the PDF's vector graphics.

**What our replica gave:**
- Final profile: 1.35–1.45 mm normal RMS, clamped. The published DD3IMP model, on the same data, is 1.18/1.38 mm.
- Both models share a radial offset of about 1.6 mm. The paper itself says the CAD and the simulation overestimate
  the cone's diameter. Calling it experimental error is inference; it could equally be a path-definition convention
  that both models share.
- Steady axial force: 25–60 % above the measured 1,525 N. Kinematic hardening gives the lower end of that range.

**Why it is not an accuracy figure (challenge C6, weakened).** The replica is far from converged:
- toolpath pitch of 4 mm vs the experiment's 0.5 mm;
- elements of 2.5 × 4.1 mm under a 6.33 mm-radius tool;
- von Mises instead of Yld91.

Halving the pitch alone moved the floor RMS from 0.95 to 0.44 mm. **Neither the 1.4 mm nor the 25–60 % force figure
may be applied to the AA5754 benchmark.** A converged replica would need about 18 h of explicit time on one thread. (The replica's runs used the explicit step, which was removed in the cleanup; at commit 7d571e6; `validation/bm3/` holds their committed outputs.)

**Material data found.** The ESAFORM 2024 benchmark (https://doi.org/10.1007/s12289-025-01928-1, open access)
reports measured AA5754 data:
- Swift hardening K = 408.5 MPa, ε0 = 0.001, n = 0.25. Our library curve is within −10/+9 % of it.
- r = 0.654/0.880/0.684 measured, against our 0.75/0.70/0.80.

Its 400 mm parts reach about 3 mm maximum deviation, and its simulations needed 180 cores.

**Conclusion.** No public experiment validates the engine to better than about 1 mm. A replicable robotic
double-sided dataset exists: DB4ISF, DC04 steel, https://zenodo.org/records/10000815. It is the cheapest next public
check.

### 3b. Against CalculiX 2.21 ([`calculix/`](calculix/README.md))

**Code agreement: upheld.**

| Test | SparLab vs CalculiX |
|---|---|
| 3D plate: benchmark blank, clamp, 4 mm ball, friction 0.1, 3-2-1 supports; 3 mm press-in plus 3 mm drag; IM Hex8 vs C3D8I | Press-in force 1,216 vs 1,214 N. Vertical drag force within 0.9 %. Friction drag force 3.6 % apart (CalculiX uses a faceted tool). Formed / unloaded / released shapes within 1.7 / 1.9 / 2.3 µm. |
| Full-integration pair | Force within 0.006 %, shape within 3.5 µm |
| Plane-strain strip | Within 0.4 µm |
| Large-strain plasticity (homogeneous) | Uniaxial to log strain 0.5 and back: within 0.3 %. Shear: within 0.5 % to γ = 0.4; −2.2 % at γ = 1. |

**Element and mesh checks:**
- **Locking.** The plain full-integration Hex8 (`mean_dilatation: none`) locks: 0.50 mm off on the strip and 0.14 mm
  on the 3D test. The default `auto` avoids this.
- **2 mm mesh vs a converged quadratic reference (strip).** 0.080 mm RMS off; 0.02 mm at 1 mm elements.
- **Claim refuted by the challenge.** The lens said a 1 mm mesh cuts the 3D shape error to 0.0045 mm. Its reference
  had the same 1 mm in-plane size, so that figure only shows the layers are converged. On the real spiral parts, 1 mm
  standard Hex8 is still 0.035 (0.083) from the converged result.
- **Not checkable.** CalculiX has no Hill48 or Chaboche, so those parts of the model are verified only by SparLab's
  own tests.

## 4. Prioritized fix list

### 4a. Can do now (no company data)

| # | Change | Expected error reduction | Cost |
|--:|---|---|---|
| 1 | **Make the 0.833 mm IM5 mesh the default** (section 5). Add `element_formulation` and `thickness_points` to `precomp/fea/setup.py`, `deck.py` and the physics hash. | Shape error 0.08–0.12 → 0.02–0.03 RMS; rim-sag bias 0.09–0.15 → ≤0.055; force bias 7–24 % → a few % | A few lines of code. Runs cost 7–11× (4–7× with penalty 3). |
| 2 | **Re-run the springback benchmark** on that mesh (section 5) | Removes the +0.06–0.11 bias in the reported 0.73 mm and in every FE-DA/ML training label | About 90–115 CPU-h (50–65 h with penalty 3), vs 13.8 h before |
| 3 | **Re-run the key process sensitivities on that mesh**: support clearance, compliance, fixture/free span | Their 2 mm magnitudes carry ±30–60 % | About 10 runs, about 7 CPU-h |
| 4 | **Measure the way the shop does**: scan clamped/in-cell, 3-DOF or a sanity-checked least-squares 6-DOF best fit to CAD; drop the invalid 6-DOF numbers | ≥0.2 mm on the reported figure (a reporting fix, not physics) | Hours |
| 5 | **Fit the metrology surface with bicubic splines** instead of linear facets | About 0.03 RMS of measurement artefact | Free |
| 6 | **Non-gouging support-tool path generator** for two-sided runs. `precomp`'s `dsif_support_path` starts the lower tool 1.6–2 mm inside the sheet. | Makes P1 simulable at all (0.17–0.79) | 2–3 days |
| 7 | **Robot-compliance fixed point in precomp**: repeat the run with the tool offset by the predicted deflection until forces settle. The scripts work today (`process/proclens.py`). | 0.08–0.41 once stiffness is known | 3–5 runs per part; needs fixes 1 and 9 for force accuracy |
| 8 | **Material library: replace r-based Hill48** with von Mises, or stress-ratio Hill48 fitted to ESAFORM 2024 / Iadicola biaxial data; take the hardening curve from ESAFORM 2024 Swift | ≤0.04 | Hours, plus retraining the surrogates |
| 9 | **Gravity as a body load**; element removal plus stress output for trimming | Gravity 0.3–1.9 mm at product scale; trimming 0.1–0.2 here (inferred) | Gravity 1–2 days; trimming 1–2 weeks |
| 10 | **Force- or spring-controlled tool in sparlab_form** (compliant robot, force-controlled support) | Enables P1/P2 without fixed-point iteration | 1–2 weeks |
| 11 | **Element refinement along the tool path, or a solid-shell element** (+ a multithreaded explicit solver; the one this audit used was removed in the cleanup; at commit 7d571e6). A uniform 0.833 mm mesh on a 500 mm sheet is about 360k elements. | Makes production-scale parts (0.1–3.7 m) affordable at converged accuracy | 2–4 weeks |
| 12 | **Per-family mesh check** at 0.833 / 0.625 mm for pyramids, domes and elliptic cones | Confirms fix 1 beyond cones | About 1.5 CPU-h per family |

### 4b. Needs the company (coupons, cell data, scans)

| # | Data | Expected error reduction | Cost |
|--:|---|---|---|
| A | **The two-robot strategy**: support, squeeze or forming from both sides; force or position control; clearance | Pins P1, the largest gap: 0.17–0.79 → a single computed value | A conversation, plus the logged paths |
| B | **Validation set**: 3 parts (a cone and a pyramid about 100 mm across, plus one product-like part) formed one-sided and two-sided, with logged paths and forces, one laser-tracker-measured tool path, and scans at 4 states (formed, unloaded clamped, released, trimmed) | Turns the 0.3–1 mm estimate into a measured number, and closes P2, P3, P6 and P8 | About 1 week of cell time plus scanning |
| C | **Robot stiffness per robot**: laser tracker plus known loads. Do not rely on modelled (VJM) stiffness: Bharti 2024 found it off by 0.8–1.0 mm. | P2: 0.08–0.41 → about ±0.05 | 1–2 days per robot |
| D | **Coupons from the production coil**: tensile 0/45/90° with extensometer, width gauge and unload-reload loops; one hydraulic bulge test; one tension-compression or cyclic-shear test; at 1e-2 to 1e-1 /s | Material 0.02–0.05 → about 0.01 (estimated). Replaces the upper-bound Bauschinger card. Matters more when the cell is two-sided (springback ×2) and on large parts. | About 2 lab days, EUR 3–6k |
| E | **The frame and skirting geometry**, the trim line, and the heat-treatment schedule | Pins P3, P6 and P7 (0.2–0.3 per change here, mm-level at scale) | Drawings; trimming and heat treatment need fix 9 and an empirical per-alloy correction |

## 5. Recommended production simulation setup

### Deck settings

| Item | Setting | Why |
|---|---|---|
| Kinematics | `finite_logarithmic` | Agrees with CalculiX's multiplicative model to 0.3 % at these strains |
| Element | Hex8 `incompatible_modes` (`model.element_formulation`) | Locking-free. Standard Hex8 at 2 mm is only right through cancelling errors. Never use `mean_dilatation: none`. |
| Mesh | In-plane size 40/48 = 0.833 mm, so that it divides the blank and the 5 mm clamp margin; 1 layer; `model.integration.thickness_points: 5`. Rule for other geometries: about 0.2 × tool radius along the path. | Released-shape error 0.021–0.033 RMS; central worst-point estimate about 0.035 (pessimistic 0.08) |
| Increments | `max_tool_travel` 1 mm; 10 release increments; Newton tolerance 1e-6 | Converged to ≤0.008 |
| Contact | Penalty scale 10 for validation and compensation runs; 3 for bulk surrogate data (+0.005–0.009, halves cost) | Measured |
| Solver | Implicit quasi-static, SuiteSparse direct solver | At this size explicit costs 1.5–2.8× more and gives 13–20 % less springback (the explicit step was removed in the cleanup; at commit 7d571e6) |
| Material, now | AA5754-O, linear+Voce isotropic hardening matched to the ESAFORM 2024 Swift curve; von Mises or stress-ratio Hill48 (not r-based); no kinematic hardening until coupon data exist | The library card is within the published spread; r-based Hill48 contradicts the measured biaxial ratio |
| Material, after coupons | Hill48 by stress ratio (Yld2000-2d only if the bulge test demands it); Voce fitted to the bulge curve; 2 Chaboche backstresses from the tension-compression test | Fix D |
| Process | Two tools when the cell is two-sided (fixes A and 6); robot-compliance fixed point (fixes C and 7); real frame and free span (fix E); report the shop's measurement state (fix 4) | Gaps P1–P3, A1 |

### Per-run cost

Measured on one thread at a machine load of about 4; timings ±50 %.

| Setup | Cost per run | vs benchmark |
|---|---|---|
| Benchmark (2 mm) | 178–215 s | 1× |
| Recommended (0.833 mm IM5), penalty 10 | 1,586–2,051 s (~30 min) | 7–11× |
| Recommended, penalty 3 | 798–1,172 s | 4–7× |
| Recommended, with a two-sided support tool | 2,472 s | |
| Robot-compliance fixed point | 3–5 runs, ≈ 1.5–3 CPU-h per part | |

Production-size parts (0.1–3.7 m) are not affordable on this element (fix 11).

### Must `benchmarks/springback` be re-run?

**Yes.** Its numbers were produced on the 2 mm mesh:
- Its 0.731 mm uncompensated RMS is biased high by about 0.06–0.11 (TC: 0.767 vs a converged 0.66–0.68).
- Its 0.9–1.1 mm rim sag is about 0.09–0.15 too deep.
- Its FE-DA and surrogate training labels carry the same bias.

Its qualitative conclusion, that the rim sag dominates, survives on the converged mesh: the sag is about 1.02–1.03 on
both cones. The statement that "no command at z ≤ 0 can correct" the sag holds only for a one-sided process. A support
tool removes it.

Re-run the whole benchmark with fix 1 (about 90–115 CPU-h, or 50–65 h at penalty 3). If compute is short, first re-run
the 8 test parts to restate the headline numbers (about 4–5 CPU-h).

## 6. Refuted claims (do not reuse)

| Claim | Status | Evidence |
|---|---|---|
| "Kinematic hardening / toolpath pitch are worth ±0.3 mm of rim sag" (validation lens) | **Refuted**, about 30× too large: the measured change is +0.009 (kinematic) and 0.032 (pitch). The sag is formed in, not sprung back. | challenge C5, `challenge/results.csv` |
| "A 1 mm mesh cuts the shape error to 0.0045 mm" (CalculiX lens) | **Refuted**: the reference had the same in-plane size; on real parts 1 mm standard Hex8 is 0.035 (0.083) off | challenge C2 |
| "The Bauschinger effect grows as the model gets more accurate" (material lens) | **Refuted**: 0.034 (2 mm IM5) → 0.022 (0.833 mm IM5) | challenge C7 |
| 6-DOF best-fit deviation 0.659 mm (process lens) | **Invalid** (failed registration) | challenge C8 |
| "1.4 mm vs NUMISHEET = engine accuracy"; "forces 25–60 % high" applied to AA5754 | **Weakened**: these describe an unconverged replica | challenge C6 |

## 7. Provenance

Each lens directory has its own README with methods, sources (URLs) and the split between verified facts and inference.
The rules below were followed throughout:
- No tracked code was modified.
- All runs used one thread each, with at most 2 at a time. The CalculiX lens notes one 1.5-minute overlap with a
  third run.

What is not committed (see `.gitignore` files):
- Raw solver output; it is reproducible from the kept decks, except the 9 explicit decks (`convergence/cases/*_x05`, `validation/bm3/cases/x_std_*`), which need a build of commit 7d571e6.
- Literature PDFs and the non-open-access NUMISHEET full text; these are cited by DOI.
- Raw CalculiX logs.
