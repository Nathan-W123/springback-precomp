# Physics audit - lens: validation against published experiments

> **Cleanup note.** The BM3 replica's `x_std_*` cases (every completed run) used the explicit forming step (`form_explicit`), which was
> removed in the cleanup; at commit 7d571e6. Their committed outputs here are the record; re-running them needs a
> build of that commit, and references to `docs/forming.md` section 7 mean that commit's version.

Status: complete except where marked.

**Resume note (2026-10-01 01:15, after a machine restart):** `bm3/cases/x_std_dz4_v20_kin` had been cut off at 72 % of
its form step (no results); it was relaunched unchanged. A new case `bm3/cases/x_std_dz2_v20` (2 mm pitch over the full
depth, otherwise identical to `x_std_dz4_v20`) was added to separate the toolpath pitch from the floor/corner discrepancy of
finding 2. Both finished; results in section "Resumed runs" at the end, which **revises findings 2, 4 and 5** (force over-prediction is larger than first estimated; springback depends on pitch and hardening rule by ~30 % each). Findings and recommendations: section "Findings" below.

## What was chosen and why

* **Case: NUMISHEET 2014 Benchmark 3, SPIF truncated cone, AA7075-O** as documented by
  D.M. Neto, J.M.P. Martins, M.C. Oliveira, L.F. Menezes, J.L. Alves, "Evaluation of strain and stress
  states in the single point incremental forming process", Int J Adv Manuf Technol 85 (2016) 521-534,
  https://doi.org/10.1007/s00170-015-7954-9 (PDF: `literature/numisheet2014_bm3_uminho_s00170-015-7954-9.pdf`).
  It is the only public data set found that gives *everything* needed to replicate: material law
  (Swift K = 343.3 MPa, n = 0.184, eps0 = 0.0015; Yld91 c_i = 1.098/1.128/0.986/1/1/0.978, m = 8; E = 72 GPa,
  nu = 0.33), t = 1.63 mm, 222 mm blank clamped 32 mm from the edge (158 x 158 mm window), backing plate
  hole 140 mm with 4 mm edge radius, tool D = 12.66 mm, z-level circular contours, dz = dx = 0.5 mm,
  first tool-centre diameter 124 mm, 45 deg wall, 44 mm depth, friction 0.01 (benchmark spec), and
  **measured** results: the axial force Fz over the whole process and the final profiles in RD and TD
  ("after unloading"), plus the authors' own FE (DD3IMP, implicit, 71 522 Hex8, 2 layers).
* The figures are **vector graphics**: Fig. 8 (forces) and Fig. 10 (profiles) were extracted
  exactly from the PDF drawing operators, calibrated on the axis tick marks (`digitize/`), not
  image-digitised. Digitising error is therefore << 0.1 mm / 5 N (the line widths are 0.7-1 pt =
  0.6-0.8 mm in profile units; the markers' centroids are used).
* Second source kept for reference, not replicated: Jin et al., MATEC Web Conf. 401, 01005 (2024),
  https://doi.org/10.1051/matecconf/202440101005 - AA5754 (our benchmark's alloy!), 1.5 mm, 150 mm
  cone, 12.7 mm tool; open access, but it publishes **no numeric hardening parameters, no profile
  and no force** (only strain/thickness along a diagonal, as a figure without the Voce constants),
  so it cannot be replicated quantitatively. It does give measured r-values of AA5754-H111
  (r0/r45/r90 = 0.75/0.57/0.60, E = 69 GPa), which differ from our library's 0.75/0.70/0.80.
* Empirical force cross-check: Aerens, Eyckens, Van Bael, Duflou, Int J Adv Manuf Technol 46 (2010)
  969-982, https://doi.org/10.1007/s00170-009-2160-2: Fz = 0.0716 Rm t^1.57 d^0.41 h_s^0.09 alpha cos(alpha)
  (Rm MPa, t and d mm, h_s the scallop height mm, alpha deg). For BM3 (Rm = 198 MPa, as Neto et al.)
  it gives **1815 N** (Neto et al. quote 1816 N) - reproduced in `bm3/analyze_bm3.py`.

## What the published state of the art achieves on this case (no SparLab involved)

`bm3/paper_sim_vs_exp.json` (from the digitised Fig. 10, markers within |x| <= 75 mm):

| Profile vs measured | vertical RMS / max [mm] | normal RMS / max [mm] | wall radius at z = -22 mm (mean of both sides) | floor (|x| < 10) |
|---|---|---|---|---|
| Neto et al. DD3IMP (implicit, 71 522 solid elements, Yld91), RD | 1.56 / 2.67 | 1.18 / 2.00 | 42.44 mm (measured 40.64) | -43.56 (measured -42.74) |
| same, TD | 1.76 / 3.41 | 1.38 / 2.96 | 42.45 (measured 40.77) | -43.61 (measured -42.90) |
| CAD (target), RD | 2.02 / 3.11 | 1.56 / 2.38 | 43.02 | -44.49 |
| CAD, TD | 2.00 / 3.24 | 1.54 / 2.42 | 42.99 | -44.49 |

* The measured cone is **1.8 mm narrower** at mid-depth than the published FE and 2.3 mm narrower
  than the CAD; the authors note this ("the diameter of the cone is overestimated both by the CAD
  geometry and by the numerical simulation") without explaining it. The measured profile is also
  off-centre by 0.4 mm (RD) and 0.9 mm (TD) (left/right wall radii 41.03/40.25 and 41.65/39.90 mm).
  Neither surface of the sheet explains the offset (the tool-side and the far-side surface of a
  45-degree wall, referenced to the flange, lie at the same radius at a given height), so it is an
  unexplained experimental offset (toolpath definition, machine compensation or measurement
  registration) of about 2 mm - the size of the error budget itself.
* The published force model is steady at **1819 N** (700-1330 s); the measurement oscillates
  between 1316 N (5th percentile) and 1798 N (95th), mean **1525 N**, max 1899 N (`digitize/bm3_force_stats.json`):
  a well-established FE code over-predicts the mean steady force by 19 % (the authors attribute it to
  clamp slip and missing kinematic hardening, which Flores et al. showed lowers SPIF forces).
  Aerens' empirical formula gives 1815 N.
* **So "almost exactly" is not the state of the art for SPIF:** a careful solid-element implicit
  model with a calibrated anisotropic law lands within 1.2-1.4 mm normal RMS (2-3 mm max) of the
  measured final profile of a 44 mm deep cone and 19 % high on the mean force.

## The SparLab replica (`bm3/`) and what had to be reduced

`bm3/build_bm3.py` writes the deck, the O-grid mesh (`mesh.inp`) and the toolpath; `bm3/run_case.sh`
runs it (one thread); `bm3/analyze_bm3.py` extracts profiles, thickness and forces and compares them
with the digitised measurement; `bm3/plot_bm3.py` draws them.

| Item | Experiment / published FE | Replica | Why |
|---|---|---|---|
| Material | AA7075-O, Swift 343.3 (0.0015 + e)^0.184 MPa, Yld91 (c ~ 1, m = 8) | Swift fitted by linear + Voce (`bm3/swift_fit.json`: sigma0 = 103.8 MPa, H = 115.6 MPa, Q = 129.7 MPa, b = 29.9; 4.0 % RMS, 10 % max misfit, the worst at e < 0.01), **von Mises** | SparLab has J2/Hill48 and linear+Voce only. Yld91 with m = 8 has a ~5 % lower plane-strain yield stress than von Mises, so the replica's forces should read a few % high |
| Blank | 222 mm square, clamped outside a 158 mm square window | disc R = 79 mm clamped on its rim | O-grid mesh; the square's corners beyond R = 79 sit on the backing plate and barely move |
| Backing plate | 140 mm hole, 4 mm edge radius | frictionless rigid plane under r >= 70 mm, no edge radius | no toroidal tool in SparLab |
| Mesh | 71 522 Hex8 (0.3-1.3 mm), 2 layers, selective reduced integration | 6 912 standard Hex8, 2 layers (0.815 mm), 96 sectors (4.1 mm at r = 62, 1.3 mm at r = 20), 2.5 mm radial in r = 16-66 mm, 0.44 mm core | cost (below) |
| Toolpath | 88 z-level contours, dz = dx = 0.5 mm, tool-centre diameter 124 mm first | **spiral, 4 mm pitch** (11 revolutions + one flat one at 44 mm), same 45-degree tool-centre cone (r_c = 62.5 - d) | cost; a 4 mm z-level step-down plunges the ball 4 mm into the sheet at every contour (tried: element inversion at the first plunge), a spiral does not |
| Friction | mu = 0.01 (benchmark spec) | 0.01, penalty scale 3 | as specified; penalty 3 per the convergence lens |
| Kinematics | updated Lagrangian | `finite_logarithmic` | |

**Cost is the binding constraint.** Measured on this machine (4 cores shared by 4 jobs, one thread
each): the implicit solver needed about 4 s per Newton iteration on a 31 000-DOF O-grid, i.e.
~30-40 s per increment; the full experiment (22 m of toolpath at 0.5 mm step-down) would need
~11 000 increments at 2 mm travel, **~100 h**; the explicit solver at 37 time steps/s on 6 912
elements (logarithmic kernel, 27 ms per step) would need ~1.1 M steps even at an equivalent speed of
~130 m/s, **~8 h**. With a 4 mm pitch (2.97 m of path) the explicit run is ~2.3 h and the implicit
run ~1.5-3 h. The effect of the coarse pitch on the force is estimated with Aerens' scallop exponent
(h_s^0.09: the 4 mm pitch has h_s = 0.667 mm against 0.0099 mm, **x 1.46** on Fz), which is an
empirical correction, not a simulation.

## Results: SparLab replica vs measurement (and vs the published FE)

Evidence: `bm3/results.jsonl` (all metrics per case), `bm3/regions.csv` (normal-distance errors by
region), `bm3/cases/<case>/profile_*.csv`, `thickness_*.csv`, `force_by_depth.csv`,
`bm3/speed_check_x_std_dz4_v10_vs_x_std_dz4_v20.json`, figure `bm3/fig_bm3.png` (v10) and
`bm3/fig_bm3_v20.png`. Profiles are the tool-side surface, referenced like the paper's (flange at
z = 0); the far-side surface gives 0.04 mm less normal RMS (1.39 / 1.40 mm). "unload" = tool removed, still
clamped (the paper's Fig. 10 state as far as can be told); "release" = unclamped onto 3-2-1 supports.
Normal distance = shortest distance from each measured point to the model's section line, signed
+ when the measurement lies above (inside) the model.

**Profile, whole section (|x| <= 75 mm), 10 m/s run:**

| | normal RMS / max [mm] RD | TD | wall radius at z = -22 | floor centre |
|---|---|---|---|---|
| SparLab unload vs measured | **1.43 / 2.20** | 1.44 / 2.37 | 43.15 mm (measured 40.64) | -41.55 mm (measured -42.74) |
| SparLab release vs measured | 1.53 / 2.30 | 1.55 / 2.57 | 43.37 | -41.81 |
| published FE vs measured | 1.18 / 2.00 | 1.38 / 2.96 | 42.44 | -43.56 |
| SparLab unload vs published FE (code to code) | 0.95 / 2.17 | 1.14 / 2.51 | | |

**By region (normal distance, RD, unload; TD in `regions.csv`)** - mean (+ = measured above model) / RMS:

| region | SparLab vs measured | published FE vs measured | SparLab vs published FE |
|---|---|---|---|
| rim / flange bend, 58-75 mm (the unsupported bend at the backing plate) | **-0.06 / 0.54** (TD -0.22 / 0.67) | +0.53 / 0.64 (TD +1.00 / 1.51) | -0.59 / 0.68 |
| wall, 25-58 mm | +1.71 / 1.74 | +1.39 / 1.44 | +0.35 / 0.46 |
| corner, 15-25 mm | +1.49 / 1.57 | +0.75 / 1.06 | +0.69 / 0.91 |
| floor, 0-15 mm | -0.80 / 0.95 | +0.85 / 0.86 | -1.65 / 1.73 |

**Thickness** (vs Fig. 15): RMS 0.044 mm (RD) / 0.047 mm (TD) over |x| <= 75 mm; minimum 1.048 /
1.031 mm against 1.105 / 1.111 measured (5-7 % more thinning at the thinnest point; the model's
wall thickness oscillates by +-0.04 mm with the 4 mm pitch, the measurement does not).

**Force** (axial, mean per revolution; `force_by_depth.csv`): steady (depth >= 24 mm) **3 081 N** at the
4 mm pitch (v20: 2 947 N). Aerens at the same pitch: 2 652 N (model +16 %). Scaled to the experiment's
0.5 mm pitch with Aerens' scallop exponent (x 0.6845): **2 109 N** against **1 525 N measured mean**
(+38 %), 1 819 N published FE (+16 %) and 1 815 N Aerens (+16 %). The horizontal force (1 740 N) is ~4 x
the published FE's radial + tangential (413 + 187 N): at a 4 mm pitch the ball works on its flank, so
horizontal forces from this replica mean nothing for the experiment. See the 1 mm-pitch run below for
a direct check of the pitch correction.

**Direct pitch check** (`bm3/cases/x_std_dz1_d8_v20`: 1 mm pitch to 8 mm depth, 20 m/s, 4 332 s;
`force_by_depth.csv`): the 4 mm pitch raises Fz by x 1.28 over depths 0-4 mm and x 1.42 over 4-8 mm
(revolutions at 5-7 mm: 1 600 / 1 822 / 1 937 N at 1 mm pitch against 2 529 N for the 4 mm revolution
ending at 8 mm). Aerens' exponent predicts x 1.29 for 4 -> 1 mm: right at the start, 10 % low once the
wall has formed. With the measured x 1.42 (4 -> 1 mm) and Aerens' x 1.133 (1 -> 0.5 mm) the steady
force of the 10 m/s run becomes **~1 915 N**: +26 % against the measured mean (1 525 N), +5 % against
the published FE (1 819 N) and Aerens (1 815 N). At equal depth the 1 mm-pitch run, scaled to 0.5 mm,
gives 1 412 / 1 608 / 1 710 N at 5 / 6 / 7 mm against 1 128 / 1 228 / 1 303 N measured (+25 to +31 %)
and 1 358 / 1 466 / 1 567 N published FE (+4 to +10 %). (The 20 m/s runs read ~4.5 % lower than 10 m/s.)
So **SparLab agrees with an established implicit code on the force to within ~5-10 %**, the expected
gap between von Mises and the Yld91 (m = 8) yield surface in plane strain; both over-predict the
measured force, SparLab by ~25-30 %.

**Speed check** (v10 vs v20, docs/forming.md 7.2): final shapes differ by 0.11-0.12 mm RMS / 0.29 mm
max; the unclamping springback is 0.218 vs 0.198 mm RMS (RD; 10 %); the steady force 3 081 vs 2 947 N
(4.5 %). If the inertia error goes as v^2, the 10 m/s run is within ~0.04 mm RMS and ~3 % of springback of
the quasi-static answer - small against its 1.4 mm distance from the measurement.

**Unclamping springback** (release minus unload) of this 44 mm deep cone: 0.22 mm RMS, 0.34 mm max; the
floor drops 0.23 mm. It is a sixth of the model-to-measurement distance: on this part the
**formed** shape, not the release, carries the error.

## Findings, ranked by their effect on the predicted shape (mm)

1. **Against a fully specified published experiment, SparLab's final profile is 1.4-1.5 mm (normal
   RMS) from the measurement, 2.2-2.6 mm at worst, on a 44 mm deep, 124 mm cone (3-3.5 % of the
   depth)** - unload 1.43 / 1.44 mm (RD / TD), released 1.53 / 1.55 mm (`bm3/results.jsonl`,
   `x_std_dz4_v10`). The published state-of-the-art FE on the same data is at 1.18 / 1.38 mm. **No
   model reproduces this experiment "almost exactly"**, and about 1.4-1.7 mm of the error is a
   radial offset of the measured wall that both models share (measured cone 1.8 mm narrower than the
   published FE, 2.3 mm narrower than the CAD; 0.4-0.9 mm off-centre), which looks like an
   experimental toolpath/registration offset - it cannot be resolved from the paper.
2. **[revised below: the floor pillow is mostly the 4 mm pitch; 2 mm pitch brings the floor RMS from 0.95 to 0.44 mm] Code to code, SparLab and the published implicit FE agree to 0.46 mm RMS on the wall and 0.68 mm
   on the flange bend, but not on the floor (1.7 mm)**: SparLab leaves a 1.2 mm higher pillow on the
   untouched floor (-41.55 mm at the centre against -42.74 measured and -43.56 published FE) and a
   deeper tool imprint at the corner (`regions.csv`). The floor is where the replica's reductions
   bite hardest (4 mm pitch: the floor ring is swept by 1 revolution instead of 8, and 1.3-2.5 mm
   elements under a 6.33 mm ball); it was not separated from the model physics (a 0.5-1 mm-pitch,
   finer-mesh run of the full depth was beyond the budget, below).
3. **The flange/rim bend - the analogue of our benchmark's dominant 0.9-1.1 mm rim sag - is predicted
   without bias**: in the band 58-75 mm (the sheet bending over the backing-plate edge, unsupported)
   SparLab is -0.06 / 0.54 mm (mean / RMS, RD) and -0.22 / 0.67 mm (TD) from the measurement, better
   than the published FE (+0.53 / 0.64, +1.00 / 1.51). This is the one region where the experiment
   directly tests the mechanism behind the benchmark's rim sag, and the engine passes it at the
   0.5-0.7 mm level (on a part 10x deeper than the benchmark's; scale is not similitude here because
   the clamping differs: backing plate + 5 mm free span vs our 5 mm clamp margin).
4. **[revised below: springback changes by -32 % from 4 to 2 mm pitch and -31 % with kinematic hardening] Unclamping springback is small here (0.22 mm RMS, 0.34 mm max, the floor drops 0.23 mm) - a sixth
   of the model-to-measurement distance.** On deep SPIF parts the forming-stage shape (sheet bending
   during forming, pillow, tool-side imprint), not the release, carries the error; the speed check
   moves the springback by 10 % (0.22 -> 0.20 mm), i.e. ~0.02 mm.
5. **[revised below: isotropic +43-59 % vs measured, +20-33 % vs published FE; kinematic +25-39 % / +5-17 %] Forces: SparLab over-predicts the measured axial force by ~25-30 % and the published FE by
   ~5-10 %.** Steady Fz ~1 915 N (pitch-corrected) against 1 525 N measured, 1 819 N published FE,
   1 815 N Aerens. The 5-10 % over the published FE matches von Mises vs Yld91 (m = 8) in plane strain;
   the 19 % by which the published FE itself exceeds the measurement is attributed by its authors to
   clamp slip and missing kinematic hardening. Mixed hardening sensitivity: see the kinematic row
   below. For robot load/compliance prediction, take the engine's forces as ~+25 % (conservative).
6. **Thickness is right**: 0.044-0.047 mm RMS against the measured distribution; the minimum is 5-7 %
   thinner than measured (1.03-1.05 vs 1.11 mm), partly the 4 mm pitch's scallop oscillation.
7. **Cost is the practical limit for validation at industrial scale.** The implicit solver needed ~1.4 s
   per Newton iteration and ~11 s per increment on 31 k DOF, cut repeatedly for inverted elements under
   a 6.33 mm ball on 2.5 x 4.1 mm elements, and was abandoned at 6 % of a coarsened (4 mm pitch) path
   (projected > 7.5 h; the real 0.5 mm-pitch path ~100 h). The explicit solver did the 4 mm-pitch cone
   in 2.2 h (v = 10 m/s, equivalent ~127 m/s) and 1.1 h (20 m/s), with shapes 0.12 mm RMS apart; at
   40 m/s (equivalent ~500 m/s) it diverged. The experiment's real 0.5 mm pitch would take ~18 h at
   10 m/s on one thread. Both solvers' validity checks behaved as documented; the "energy error 0.22"
   warning on these runs is the onset artefact (absolute error 1e-5 of the internal work).

## What could not be replicated (honestly)

* The experiment's 0.5 mm pitch (88 contours): replaced by a 4 mm spiral for the full depth and a 1 mm
  spiral to 8 mm depth. Force corrected through the 1 mm run and Aerens' exponent; the floor/pillow and
  corner discrepancy may partly be the coarse pitch.
* Yld91 (m = 8): von Mises used. Swift law: fitted by linear + Voce, 4 % RMS / 10 % max misfit (worst
  below 1 % strain).
* The 158 mm square clamp window and the 4 mm backing-plate radius: disc clamp at R = 79 mm and a sharp
  plane edge at r = 70 mm.
* Mesh convergence of the replica (only 6 912 elements; the published FE had 71 522). The convergence
  lens (`../convergence/findings.md`) found 2 mm in-plane elements with a 4 mm tool ~0.1 mm RMS off on a
  3 mm deep part; here the circumferential element is 4.1 mm at the rim under a 6.33 mm ball.
* An implicit reference of the full cone (infeasible, finding 7).
* The state of the measured part ("after unloading") is not stated: compared both clamped and unclamped;
  the clamped state fits 0.1 mm better.
* The AA5754 SPIF study (Jin et al. 2024) gives no hardening constants, profiles or forces, so our
  benchmark's alloy could not be validated directly.

## Recommendations (with cost)

1. **Get a validation data set that matches the product: double-sided, robot-held, measured clamped
   and unclamped.** The open **DB4ISF** database (Ruhr-Universitaet Bochum; 76 robot-DSIF experiments
   with two industrial robots, toolpaths, robot programs, scans and the normal deviation at every
   toolpath point; DC04 steel 0.8 mm, 4 mm tool radius - our tool size; CC-BY 4.0;
   https://zenodo.org/records/10000815, 3.7 GB) is the closest public match to RoboForming; it needs
   DC04 hardening data from the literature or a tensile test and a robot-compliance model. Cost: ~1 day
   to set up a case, 2-20 h of compute per part (finding 7). Better still, an in-house coupon (a
   45-degree cone and a pyramid, ~100 mm, AA5754-O from the production coil, scanned clamped and
   unclamped, robot forces logged): 2-3 days of shop time; it would turn every number above into one
   about our own process.
2. **Measure the material** (tensile at 0/45/90 with DIC for r-values and the hardening to large strain
   by bulge or shear test; one cyclic or tension-compression test for the Bauschinger effect). The only
   measured AA5754 r-values found (Jin et al. 2024, H111: r0/r45/r90 = 0.75/0.57/0.60, E = 69 GPa)
   differ from the library's handbook 0.75/0.70/0.80. Cost: ~1 week external lab, ~EUR 3-5 k.
3. **Treat the predicted forces as ~25 % high** for robot load and compliance compensation until
   calibrated; the force error is mostly material law (yield surface, isotropic hardening), not the
   solver (code-to-code within 5-10 %).
4. **Model robot compliance.** Bharti et al. (Sci. Rep. 14 (2024) 20291,
   https://doi.org/10.1038/s41598-024-70746-3) attribute 40 % (cone) to 60 % (variable-wall-angle cone)
   of the geometric error of a roboformed part to robot compliance (ABB IRB7600; mean in-plane
   deviations 0.5-0.75 mm, tool-axis up to 0.94 mm) against 0.53-1.13 mm of springback: of the same
   order as everything else in this audit. The engine has no robot compliance (see the process lens).
5. **For full-size parts, the solver cost must come down ~10x before the engine can be validated or used
   at production scale**: shell or solid-shell elements (no through-thickness stable-step limit: ~2-4x
   larger explicit step and half the elements), multithreading of the explicit kernel (measured 1.9x
   on 2 threads in docs/forming.md 7.4), and keeping the equivalent speed at <= ~130 m/s (shape within
   ~0.1 mm). Cost: solid-shell element ~2-4 weeks of development + verification.
6. **Use a toolpath pitch <= 1 mm in any force prediction**: a 4 mm pitch inflates the axial force by
   1.3-1.4x and the horizontal force ~4x; the shape is less sensitive (not quantified here beyond the
   floor/corner discrepancy).

## Run log

* `bm3/cases/i_std_dz4` (implicit, same mesh and spiral, travel 4 mm, form split every 8 mm):
  **abandoned** after 1 722 s at 0.183 m of the 2.967 m path (6 % : 151 increments, 1 076 Newton
  iterations, 8 cuts, all from elements inverting under the tool; ~1.4 s per iteration, 7 iterations
  per increment, increments limited to 2.5 mm by the chord knots). Projected **> 7.5 h** for the
  4 mm-pitch cone - the implicit solver cannot replicate a full-size SPIF part within a working day on
  this hardware. (An earlier attempt with incompatible-mode elements failed its local mode iterations
  at the first plunge; with z-level contours at 4 mm the plunge inverted elements: `README` above.)
* `bm3/cases/x_std_dz4_v10` (explicit, 10 m/s, selective dynamic mass scaling to 1 us: element scale
  ~160 (max 232), equivalent speed ~127 m/s): **completed**, 297 k steps, 7 988 s (27 ms a step);
  `max_kinetic_ratio` 0.011, penetration 1.4 % of the element (warned), added mass 132 x.
* `bm3/cases/x_std_dz4_v20` (20 m/s, equivalent ~254 m/s): **completed**, 4 066 s; kinetic ratio 0.020,
  penetration 2.7 % (warned). Both warn of an energy-balance error of 0.22 "of the largest energy":
  `step_1_form_energy.csv` shows it is the onset artefact of docs/forming.md 7.1 (the absolute error
  never exceeds 0.045 J against 4 658 J of internal work, 1e-5).
* `bm3/cases/x_std_dz4_v40` (40 m/s, equivalent ~500 m/s): **diverged** at 51 % of the path after
  1 032 s (element inverted, plastic strain 12, kinetic ratio 0.05, penetration 6.6 %): the upper
  bound of usable speed on this mesh lies between 254 and 500 m/s equivalent.

## Resumed runs (2026-10-01)

### Kinematic hardening: `bm3/cases/x_std_dz4_v20_kin` (completed, 5 111 s)

Same as `x_std_dz4_v20` but the Voce part of the fitted Swift curve is an Armstrong-Frederick backstress
(C = Q b = 3 873 MPa, gamma = b = 29.9): identical monotonic curve, maximal Bauschinger effect (an upper bound -
AA7075-O / AA5754-O are mixed-hardening, not purely kinematic). Evidence: `bm3/results.jsonl` (row
`x_std_dz4_v20_kin`), `bm3/regions.csv`, `bm3/springback.csv` (`bm3/springback_table.py`).

| quantity | isotropic (v20) | kinematic AF (v20) | change | measured / published FE |
|---|---|---|---|---|
| steady Fz, 4 mm pitch [N] | 2 947 | 2 574 | **-12.6 %** | Aerens at 4 mm: 2 652 |
| peak Fz, 4 mm pitch [N] | 3 465 | 2 927 | -15.5 % | |
| steady Fz scaled to 0.5 mm (Aerens exponent) [N] | 2 017 | 1 762 | | 1 525 meas. / 1 819 FE |
| steady Fz scaled with the measured 4->1 mm ratio 1.42 and Aerens 1->0.5 x 1.133 (x 0.622) [N] | 1 832 (v10: 1 915) | **1 600** | | 1 525 meas. (+5 %) |
| unload normal RMS vs measured, RD / TD [mm] | 1.44 / 1.45 | **1.34 / 1.33** | -0.11 | FE 1.18 / 1.38 |
| release normal RMS vs measured, RD / TD [mm] | 1.54 / 1.55 | 1.41 / 1.40 | -0.14 | |
| floor centre, unload [mm] | -41.79 | -42.20 | 0.41 deeper | -42.74 meas. / -43.56 FE |
| wall radius at z = -22, unload [mm] | 43.21 | 43.04 | -0.17 | 40.64 meas. / 42.44 FE |
| vs published FE, unload, RD / TD [mm] | 0.88 / 1.08 | 0.74 / 0.98 | -0.12 | |
| **unclamping springback (release - unload) RMS / max, RD [mm]** | **0.198 / 0.308** | **0.137 / 0.214** | **-31 %** | not measured |
| same, TD [mm] | 0.208 / 0.255 | 0.128 / 0.155 | -38 % | |
| thickness RMS vs measured [mm] | 0.043 / 0.049 | 0.044 / 0.049 | none | |

Regions (unload, RD, mean / RMS, + = measured above model): floor -0.19 / 0.51 (iso v10: -0.80 / 0.95), corner
+1.45 / 1.51, wall +1.64 / 1.67, rim -0.13 / 0.55. The kinematic model fixes most of the floor pillow (the region
that is reverse-loaded when the tool passes again), and leaves the wall/corner offset (the shared radial offset of
finding 1) unchanged.

**Reading:** the hardening rule changes the *formed* shape by ~0.1 mm RMS (0.4 mm at the floor), the force by
13 % (bringing the pitch-corrected force to within ~5 % of the measured mean, against +20-26 % isotropic), and the
**unclamping springback by a third**. Both the force and floor of the experiment favour some kinematic
hardening; the experiment does not measure the clamped/unclamped difference, so the springback change is
model-to-model only. For our benchmark (no kinematic hardening; springback-dominated 0.9-1.1 mm rim sag on a
2-4 mm deep part) this is the single most consequential *material-model* choice found by this lens: a pure-AF
vs isotropic swing of ~30-40 % on the release springback would be ~0.3 mm on the benchmark's rim sag if it scales
proportionally (inference). The material lens ran the benchmark-scale test independently and found the same
direction and size: Bauschinger cuts springback by 23-31 % on the IM5 element (`../material/README.md`, its
recommendation 4) - two independent geometries (44 mm deep cone, 2-4 mm deep benchmark parts) agree.

### Toolpath pitch: `bm3/cases/x_std_dz2_v20` (completed, 7 605 s)

2 mm spiral pitch over the full 44 mm (path 5.76 m), otherwise identical to `x_std_dz4_v20` (isotropic). Evidence:
`bm3/results.jsonl`, `bm3/regions.csv`, `bm3/springback.csv`, `bm3/cases/x_std_dz2_v20/force_by_depth.csv`.

| quantity | 4 mm pitch (v20) | 2 mm pitch (v20) | change |
|---|---|---|---|
| unload normal RMS vs measured RD / TD [mm] | 1.44 / 1.45 | 1.35 / 1.36 | -0.09 |
| release normal RMS vs measured RD / TD [mm] | 1.54 / 1.55 | 1.42 / 1.43 | -0.12 |
| vs published FE, unload RD / TD [mm] | 0.88 / 1.08 | 0.76 / 0.99 | -0.10 |
| floor centre, unload [mm] (measured -42.74) | -41.79 | -42.22 | 0.43 deeper |
| wall radius at z = -22 [mm] (measured 40.64) | 43.21 | 43.08 | -0.13 |
| floor region mean / RMS vs measured, RD [mm] | (v10: -0.80 / 0.95) | -0.30 / 0.44 | pillow mostly gone |
| wall region mean / RMS vs measured, RD [mm] | | +1.67 / 1.70 | unchanged: the shared radial offset |
| thickness RMS RD / TD [mm]; minimum [mm] (measured 1.105) | 0.043 / 0.049; 1.04-1.06 | **0.034 / 0.043; 1.10** | minimum thickness now matches |
| **unclamping springback RMS / max RD [mm]** | **0.198 / 0.308** | **0.134 / 0.212** | **-32 %** |
| steady Fz [N] (depth >= 24 mm) | 2 947 | 2 718 | x 0.922 (Aerens predicts x 0.879) |
| horizontal force [N] | ~1 870 | ~970 | x 0.52 |

**Consequences (these revise findings 2, 4 and 5 above):**

* The floor/corner discrepancy of finding 2 is mostly **discretisation of the path, not physics**: halving the pitch
  moves the floor 0.43 mm toward the measurement (floor RMS 0.95 -> 0.44 mm). The kinematic run moved it by the
  same amount at 4 mm pitch, so the two effects cannot be separated further without a fine-pitch kinematic run
  (~4-5 h at 1 mm pitch on this machine; not affordable in this resume).
* **The unclamping springback is not converged in the toolpath pitch**: 0.198 -> 0.134 mm RMS from 4 to 2 mm pitch
  (-32 %), as much as the switch to kinematic hardening. Both values are small on this deep cone, but on our
  benchmark the springback *is* the error budget. The benchmark uses 1 mm pitch with a 2 mm-radius tool
  (pitch / tool radius 0.5) against 0.32 here at 2 mm, so it sits in the regime where this replica still changes;
  the convergence lens should confirm a 0.5 mm-pitch benchmark case (inference).
  The direction is not settled in the literature either: the 2024-T3 study "Effect of Toolpath on the Springback of 2024-T3 Aluminum During SPIF"
  (0.5 mm sheet; https://www.researchgate.net/publication/280316289; abstract only seen) report *more* springback at smaller steps, while the
  ESAFORM 2024 review (below) cites several studies where *larger* steps raise residual stress and springback.
* **Force (revises finding 5):** at equal depth the 2 mm and the 1 mm pitch differ by x 0.87 (6 mm depth: 2 095 vs
  1 822 N), and 4 -> 2 mm at steady state by only x 0.922, i.e. the earlier 4 -> 1 mm factor 1.42 (taken at 4-8 mm
  depth, before the wall forms) overstated the steady-state pitch effect. Best estimate of the isotropic steady Fz at
  the experiment's 0.5 mm pitch: 2 718 x 0.87 x 0.883 (Aerens 1 -> 0.5) = **~2 090 N** (bracket 2 090-2 320 N if the
  model's own weaker pitch sensitivity is extrapolated instead of Aerens'), +4.5 % for 10 m/s instead of 20 m/s:
  **~2 180-2 420 N isotropic = +43-59 % over the measured mean (1 525 N), +20-33 % over the published FE (1 819 N)**.
  With kinematic hardening (x 0.874 at 4 mm pitch): **~1 900-2 120 N = +25-39 % over measured, +5-17 % over the
  published FE**. The earlier "+26 % / +5 %" for the isotropic model was optimistic.

### Literature added in the resume: ESAFORM Benchmark 2024 (AA5754, 1.5 mm, SPIF/TPIF/DSIF)

M. Vanhulst et al., "ESAFORM benchmark 2024: study on the geometric accuracy of a complex shape with single point
incremental forming", Int J Mater Form 18 (2025) 72, open access, https://doi.org/10.1007/s12289-025-01928-1
(PDF: `literature/esaform2024.pdf`, text `literature/esaform2024.txt`). Verified from the text:

* **Same alloy as our benchmark**, one batch of AA5754 1.5 mm distributed to 15 institutes; 400 x 400 x 55.7 mm
  free-form part; 13 experimental contributions incl. DSIF and TPIF; CAD and scans on the ISF database platform.
* **Measured AA5754 data (U. Michigan, tensile at 0/45/90 with DIC):** Swift K0 = 408.52 MPa, eps0 = 0.001,
  n = 0.25; r0/r45/r90 = **0.654 / 0.880 / 0.684**; Hill48 F/G/H/N = 0.578/0.605/0.395/1.632; friction 0.025 used
  in the FE. Thermal camera: the sheet stays near room temperature (temperature effects negligible).
* **Comparison with our library (deck of the springback benchmark):** our linear+Voce curve is within -10 % / +9 % of
  the measured Swift curve for 0.002 <= eps_p <= 0.5 (-10 % at 1-5 % strain, +13 % at eps_p = 1), but our r-values
  0.75 / 0.70 / 0.80 differ from the measured 0.654 / 0.880 / 0.684 (r45 -20 %, r90 +17 %), and from a second source
  (Jin et al. 2024, H111: 0.75/0.57/0.60). r-values of "AA5754-O" vary by +-0.15 between batches - only the
  production coil can settle them.
* **Achieved accuracy:** "maximal deviations of around 3 mm for the best experiments" (a 400 mm part); unclamping
  visibly changes the shallow-wall zone ("high deviations at the shallow wall area (zone E) ... caused by residual
  stress releasing during unclamping"); the Abaqus/Explicit simulations (Hill48 + isotropic Swift, shells and solids)
  needed 180 cores and agree with the clamped scan only to within the plotted cross-section accuracy (no RMS
  given). The data are not replicable here (400 mm part, multi-hour forming, figures only in the paper; scans on a
  registration-only platform) but they are the best public AA5754 / DSIF target for the recommended in-house
  validation, together with DB4ISF.

### Run log (resume)

* `bm3/cases/x_std_dz4_v20_kin`: relaunched from scratch (the pre-restart run had reached 72 % of the form step and
  left no output); **completed** in 5 111 s; penetration 2.5 % of the element (warned), energy-balance warning 0.22
  (onset artefact, as in the other runs), over-constrained clamp reactions in `unload` as expected (1 857 N).
* `bm3/cases/x_std_dz2_v20`: built with `python3 build_bm3.py cases/x_std_dz2_v20 --dz 2 --nc 96 --hr 2.5 --explicit 20
  --form std --tp 0` (deck identical to `x_std_dz4_v20` except the pitch and path length - checked by diff);
  **completed** in 7 605 s (288 k steps); penetration 2.4 %, energy warning 0.20.
* Analysis: `python3 analyze_bm3.py cases/<c>`, `python3 regions_bm3.py cases/<c>`, `python3 springback_table.py cases/...`.
