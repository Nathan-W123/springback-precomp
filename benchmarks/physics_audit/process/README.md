# Physics audit, process-realism lens: is the modelled process the real one?

> **Cleanup note.** Where this lens names `form_explicit` as the route for metre-scale parts: that explicit
> forming step was removed in the cleanup; at commit 7d571e6. Full-size parts would need an explicit solver again,
> or a cheaper element; references to `docs/forming.md` section 7 mean that commit's version.

**Question.** The engine is meant to predict the formed-and-released shape of
parts made by robotic incremental sheet forming - the "RoboForming" process of
a start-up (Machina Labs) that forms sheet from CAD with two industrial robots,
one on each side of a clamped sheet, then scans, heat-treats and trims the
part. How close is the benchmark's model of that process
(`benchmarks/springback`: 40 x 40 x 1 mm AA5754-O blank, one rigid 4 mm ball
from above exactly on the path, 5 mm clamp, 1 mm step-down, friction 0.1,
quasi-static, no gravity, room temperature, 3-2-1 release without trimming),
and what does each difference do to the predicted shape, in mm?

**Method.** (a) What the real process is, from the company's public pages and
patent and from the robotic-ISF literature (sources at the end; "verified"
means read in the source's own text during this audit). (b) Every model
assumption is compared with it; where the engine can represent the real
condition, the benchmark's own deck is re-run with that one change and the
released shape is compared with the benchmark run of the same part (same
commanded shape, path, mesh, solver); where it cannot, the effect is bounded
by a side computation (robot and stylus compliance, gravity, CalculiX shells)
or by literature, and marked as such. Parts: the benchmark test parts
`truncated_cone-s2026-0000` (TC: footprint 15.4 mm, depth 3.09 mm, wall
38.6 deg; benchmark release RMS 0.767 mm), `pyramid-s2026-0001` (PY, 0.699 mm)
and `elliptic_cone-s2026-0001` (EC, 0.841 mm) - the three the material lens
used - plus all 8 test parts for the no-simulation analyses. All runs: this
worktree's Release build, one thread, two at a time. Lengths in mm.

"Effect" below = RMS (and largest) change of the released surface over the
part against the benchmark run of the same part: the error the model makes
when it leaves that aspect of the real process out.

## Bottom line

1. **The benchmark simulates a one-sided process; the company's cell is
   two-sided, and that is the largest gap by far.** Letting the second robot
   act as a moving support below the sheet (position-controlled at the
   target's underside, never gouging it) changes the released shape by
   **0.70 / 0.65 / 0.81 mm RMS (1.1-1.5 mm max)** on TC / PY / EC and removes
   most of the 1.1-1.2 mm rim sag that dominates the benchmark error
   (deviation from target 0.77 -> 0.28, 0.70 -> 0.30, 0.84 -> 0.31 mm RMS).
   The answer depends on how the support is driven: with 0.2 / 0.5 mm of
   clearance the change is 0.48 / 0.17 mm. Machina's actual two-robot
   strategy is not public; the model must be driven by the two real tool
   paths (and the support's force control) to predict their parts at all.
   Evidence: `runs/*/dsif_ng*`, `results.csv`.
2. **Robot compliance is the second largest and is not in the model.** With
   the simulated forces (0.55 kN mean, 0.92 kN peak, 8 test parts) a robot of
   1 um/N (1 N/um, typical articulated robot) is pushed off the path by
   0.52 mm on average (0.92 max), mostly away from the sheet; solved to the
   robot-sheet equilibrium (under-relaxed fixed point: input and output
   forces agree to 2-3 % in the mean, 9 % pointwise; the last two iterates
   give the same shape) the released shape changes by **0.41 mm RMS (0.59 max)
   and the part comes out 0.54 mm shallower** (TC; PY / EC first order 0.32 / 0.39 mm).
   An ABB IRB 7600-class robot (published joint stiffnesses) gives 0.08 mm
   RMS when it pushes along its reach (Machina-style vertical sheet) and
   0.22 mm when it presses down on a horizontal sheet; the stylus adds
   0.07-0.16 mm laterally. Robot deflection is **comparable to or larger than
   the whole springback** of these parts (0.14 mm RMS) and larger than every material
   effect (<= 0.05 mm, material lens). Evidence: `robot_deflection.csv`,
   `runs/truncated_cone-s2026-0000/rob_*`.
3. **The fixture geometry sets the dominant error of the benchmark.** The
   0.9-1.2 mm rim sag is formed in during forming (it is already -0.95 mm at
   the loaded state; springback adds 0.14 mm RMS) and scales with the free
   sheet between frame and part: a 24 / 30 / 50 mm window gives a rim sag of
   0.84 / 1.15 / 1.55 mm and a released RMS of 0.56 / 0.77 / 1.04 mm (effect
   0.22 / - / 0.28 mm RMS). Machina's frame holds a whole sheet (up to
   3.7 m) with "skirting" around the part, so the free span is theirs to
   choose and must be modelled as built. And
   the error is not size-independent: the same part and frame at twice the
   size (same 1 mm sheet, 4 mm tool, 1 mm step) comes out at 1.45 mm RMS
   instead of 0.77 (rim sag 2.34 instead of 1.15 mm, springback 0.23 instead
   of 0.12 mm) - the benchmark's absolute numbers do not transfer to
   Machina-size parts.
4. **The measured state differs from the simulated one.** Machina scans in
   the cell before unclamping and best-fits to CAD; the benchmark measures
   after a 3-2-1 release in the fixture frame. On the 8 test parts: loaded
   0.645, in-frame after unloading 0.636, released 0.731, released with a
   3-DOF best fit 0.526, with a 6-DOF ICP 0.659 mm RMS - the choice of state
   and alignment moves the reported error by up to 0.2 mm, more than any
   material or mesh effect. Evidence: `measure_states.csv`.
5. **Trimming and heat treatment are not modelled and cannot be with this
   engine** (no element removal, no thermal or creep relaxation); both
   release residual stress the model carries. On these parts the release
   springback is 0.14 mm RMS (0.02 mm in the clamp), which bounds the order of
   magnitude of what trimming the flange and a stress relief would change
   here (inference); on product-size parts it is millimetres.
6. **Small here:** step-down 0.5 instead of 1 mm 0.03 mm RMS; tool radius
   6 instead of 4 mm 0.14 mm; friction and rate/temperature <= 0.011 mm
   (material lens); gravity 1.4 um on this blank. An H32-like temper (yield 193
   instead of 100 MPa, flat hardening) doubles the springback (0.12 -> 0.27 mm
   RMS) but changes the released shape by only 0.07 mm RMS: its formed shape
   moves the other way. A constant 0.3 mm path error (robot absolute accuracy
   after calibration is 0.1-0.3 mm) moves the released shape by 0.22 mm RMS,
   0.29 mm in depth - the transfer of path errors is 0.5 (rim band) to 0.85
   (deep part).
7. **"Almost exactly" is not achievable with the benchmark's process model;
   it is achievable to about 0.1-0.3 mm RMS at this part size only if the model
   runs the real process:** both robots' executed paths (logged, not
   nominal), the robots' compliance coupled to the forming force, the real
   frame and skirt, and the same measurement state and alignment as the
   shop - then the remaining error is the material (0.02-0.05 mm, material
   lens), the discretisation (element lens), scanner noise (0.5 mm point
   accuracy quoted in Machina's patent - averaging over thousands of points
   is needed to see 0.1 mm), and the unmodelled trim / heat-treat steps.

## The real process (what the model is supposed to simulate)

"Verified" = read in the source's own text (web page, patent text, paper
HTML or table) during this audit; "summary" = only a search-engine summary
or an abstract was available; "inference" = our reasoning.

| Aspect | What is publicly known | Source | Status |
|---|---|---|---|
| Cell | two industrial robot arms on separate linear rails, a rigid frame between them holding the sheet; the robots face each other on opposite sides of the sheet; "7-axis robotics" | [S1], [S2], [S3] | verified |
| Role of the second robot | the patent claims a second arm "on an opposite side of the part" controlled "in conjunction" with the first; its figure compares one-robot and two-robot parts ("part 1515 includes more details and more closely resembles" the target). The exact strategy (support/squeeze, forming from both sides, force- or position-controlled) is not public | [S3] | verified (claim text); strategy unknown |
| Clamping | sheet edges clamped to the frame by "a series of clamps", hydraulic or electric (servo); "skirting" geometry around the part to secure material during forming | [S3], [S2] | verified |
| Sizes | parts up to 12 ft (3.7 m) long, 5 ft (1.5 m) deep; sheet up to 1/4" (6.35 mm); examples: 0.125" AA5052-H32 toroidal tank, 0.080" aluminium sheet | [S1], [S2] | verified |
| Materials | aluminium (2000, 5000, 6000, 7000 series named), steels, stainless, Ti, Ni superalloys, refractory, low-expansion alloys | [S1], [S2] | verified |
| Tools | "a stiff stylus"; roller tools that rotate to reduce friction; tool changer (forming, scanning, trimming end effectors) | [S3] | verified |
| Forces | load cells on the forming end effectors; "peak forces" of 2 000 N to 20 000 N "depending on material strength and thickness"; example: a 1 mm stainless part takes 4 h at 4 000 N peak, 8 h at 3 000 N | [S3] | verified |
| Control | force, torque and displacement sensors on each arm, "real-time, closed-loop feedback control"; in-process scans compared with CAD and corrections "fed back into the next pass"; online "short horizon" re-optimisation during forming | [S1], [S2], [S3] | verified |
| After forming | laser profiler scan in the cell, "aligned with the nominal CAD model"; optional heat treatment (stress relief, temper - "targeted post-forming heat treatments" against residual stress and springback); robotic trimming (spindle, laser or plasma) | [S1], [S2], [S3] | verified |
| Scanner | "surface scanners may have a point accuracy of 0.5 mm" (patent example) | [S3] | verified |
| Accuracy claimed | "+-0.3 % of the maximum dimension or +-1 mm (0.04")", "sub-millimeter precision possible"; Ra <= 3.2 um | [S1] | verified |
| Robot stiffness (literature) | ABB IRB 7600 (500 kg) joint stiffnesses 3.73e6, 4.82e6, 4.04e6, 6.38e5, 7.59e5, 4.72e5 N m/rad (VJM); robotic ISF of 1.22 mm AA1050 (10 mm tool, 0.5 mm step, 750 mm/min, 150 mm blank, 100 mm window, oil): robot compliance 0.91 / 1.05 mm mean resultant deviation, 40 / 60 % of the total error (cone / variable-angle cone), springback 1.13 / 0.53 mm, tool deflection 0.14-0.19 mm, forces up to 475 N | [L1] | verified (paper HTML, Tables 3 and 5) |
| Robot stiffness (general) | articulated robots < 1 N/um at the tool, machine tools tens of N/um | [L2] | summary |
| Robot accuracy | absolute accuracy about two orders of magnitude worse than repeatability; after calibration median 0.22 mm; KUKA KR 210 average 0.106 mm (repeatability +-0.06 mm), KR 500-3 max 0.32 mm after compensation | [L3] | verified (paper text) |
| Robotic ISF with a support tool | Roboforming (Bochum, two robots): complex parts "mostly formed with a local support tool which substitutes a full die"; without a die "the free compliant sheet area surrounding the formed part" makes "the geometry shift away from the forming tool" - reinforcing that area raises accuracy | [L4] | abstract |
| DSIF accuracy | three causes: machine compliance, in-process springback, post-process springback; compliance compensation of the forming tool and contact-force control of the support tool | [L5] | verified (lab page) |
| Unclamping / annealing | springback after unclamping reduced by annealing in dedicated clamping devices; "global springback... often requires annealing" | [L6], [L7] | abstract |

## Gap table

Effect = change of the released shape over the part vs the benchmark run of
the same part (RMS, largest in brackets), unless stated. "computed" = this
engine; "side calc" = outside it; "lit." = literature; "inference" = ours.

| # | Model / benchmark assumption | Real process | Effect on the released shape [mm] | Basis | How to close it (cost) |
|--:|---|---|---|---|---|
| 1 | one rigid tool from above (SPIF) | two robots on opposite sides of the sheet, controlled together [S1-S3]; strategy (support, squeeze, both-sided forming, force control) not public | **0.65-0.81 RMS (1.1-1.5 max)** with a support held at the target's underside; 0.48 / 0.17 with 0.2 / 0.5 mm clearance; rim sag -1.15 -> -0.20 | computed, TC / PY / EC (`dsif_ng*`) | run both robots' real paths as two tools - works today by deck patch (`proclens.py`); replace precomp's experimental `dsif_support_path` (starts 1.6-2 mm inside the sheet) with a non-gouging generator (2-3 days); a force-controlled support needs a spring- or force-driven tool in sparlab_form (1-2 weeks); data: Machina's two-robot strategy and logs |
| 2 | tool exactly on the commanded path (rigid robot) | serial robots 0.1-2 um/N at the tool; IRB 7600 VJM 0.07-0.93 um/N normal to the sheet, 0.5-2 in plane; 0.9-1.05 mm measured deviation from compliance in robotic ISF [L1] | **0.41 RMS (0.59 max), 0.54 shallower** at 1 um/N (converged); 0.32 PY (first order), 0.39 EC (first order); IRB vertical sheet 0.08, horizontal sheet 0.22; mean tool deflection 0.52 (1 um/N) - 1.04 (2 um/N) over the 8 test parts | computed (`rob_*`), side calc (`robot_deflection.csv`) | couple C f in the solver (a compliant tool: centre = command + C f, in Newton's loop; ~1 week) or iterate outside with precomp.robot, under-relaxed (3-5 runs, as here); measure each robot's compliance map in the frame (laser tracker + loads, 1-2 days per robot); or drive the model with the logged executed TCP path (no development) |
| 3 | rigid stylus | stylus and holder bend | 0.07-0.16 lateral (16 x 100 mm steel stylus at 136-321 N); 0.14-0.19 measured [L1] | side calc (`stylus.json`) | add to C (with #2) |
| 4 | path exact in space | absolute accuracy about 100x repeatability; 0.1-0.3 mm after calibration [L3] | 0.3 mm constant error -> 0.22 RMS, 0.29 in depth (transfer 0.5-0.85) | computed (`offset_z03`) | calibrate the robots in the frame; use measured/logged paths |
| 5 | 40 mm blank, 5 mm clamp (30 mm window) | whole sheets in a clamp frame up to 3.7 m, "skirting" around the part [S1-S3] | 0.22 RMS for a 24 mm window, 0.28 for 50 mm; rim sag 0.84 / 1.15 / 1.55 for 24 / 30 / 50 mm | computed (`clamp8`, `blank60`) | model the real frame, blank and skirt (larger models: `form_explicit` for m-scale) |
| 6 | parts 14-22 mm, tool 4 mm, 1 mm sheet | parts 0.1-3.7 m, sheet 1-6.35 mm | part x2 (and frame x2): released RMS vs target 1.45 instead of 0.77, rim sag 2.34 instead of 1.15, springback 0.23 instead of 0.12 - errors grow about in proportion to size; beyond x2 not computed | computed (`scale2`) | benchmark at representative scale (>= 100 mm), explicit forming (hours per part) |
| 7 | step-down 1 mm | sub-mm (0.5 mm in [L1]; Ra <= 3.2 um [S1]) | 0.03 RMS (0.09 max) | computed (`stepdown05`) | use the real value |
| 8 | tool radius 4 mm | 5 mm in [L1]; Machina not public; roller tools [S3] | 0.14 RMS (0.23 max) for 6 mm | computed (`tool6`) | use the real tool |
| 9 | AA5754-O card, 1 mm | 5052-H32 0.080-0.125" [S2], 2xxx/6xxx/7xxx, steels, Ti | H32-like temper: 0.07 RMS; springback x2.2 | computed (`mat_5052H32`); material lens 0.02-0.05 | coupons per alloy, temper, batch (material lens) |
| 10 | friction 0.1, no tool spin | lubricated stylus or roller tools | <= 0.003 | computed (material lens) | none needed for shape |
| 11 | quasi-static, isothermal | feeds ~1-6 m/min, 12-35 K heating | <= 0.011 | side calc (material lens) | none for aluminium at these feeds |
| 12 | no gravity | vertical sheet in the frame; scan / release orientation | 1.4 um here; flat 0.6-1.5 m sheets: 0.3-1.9 clamped horizontal, 70-280 on three points | side calc (`gravity/`) | body loads in sparlab_form (1-2 days) and the real scan fixture |
| 13 | 3-2-1 release, fixture frame, no alignment | in-cell laser scan (clamped), best fit to CAD [S1, S2] | same part: 0.645 loaded / 0.636 in frame / 0.731 released / 0.526-0.659 best fit (8 parts) | computed (`measure_states.csv`) | report the shop's state and alignment (no cost) |
| 14 | no trimming | robotic trimming (spindle, laser, plasma) [S3] | not computable; ~0.1-0.2 here (order of the release springback, 0.14 RMS), mm on large parts (inference) | - | element removal in a release step plus full stress output (1-2 weeks) |
| 15 | no heat treatment | optional stress relief / ageing / temper [S1, S2] | not computable; stress relief ~ springback order, quench more (inference) | - | measured correction per alloy, or a thermal relaxation model (weeks) |
| 16 | open-loop nominal path | closed-loop force / displacement control, in-process scans and corrections [S1-S3] | the controller changes the path itself; an open-loop prediction of the nominal path is not the part made | - | simulate the executed (logged) path; co-simulate the controller for planning |
| 17 | exact measurement | scanner point accuracy 0.5 mm (patent example) [S3] | noise of the size of the effects above | lit. | area averages, CMM or tracker reference on validation parts |

## Details

### 1. Two robots (DSIF) instead of one tool from above

The engine takes several rigid tools; `proclens.py` adds a second sphere
(R = 4 mm, friction 0.1) below the sheet, contacting the bottom faces. Its
path: in plan, opposite the primary contact along the target's normal
(offset R1 + t + R2, the geometry of precomp's experimental
`dsif_support_path`); in height, lifted until it touches the **target's
underside** (target - t0, i.e. the sine-law thickness t0 cos(alpha) along
the normal) without penetrating it anywhere - a drop-cutter from below. So
it can never penetrate the initial flat sheet and cannot reach into the rim
fillet, which is concave seen from below. Note: precomp's
`dsif_support_path` as it is would start the run with the support 1.6-2.0 mm
inside the flat sheet (it takes the target normal at a point the sheet has
not yet been formed to, and in the air it puts the support 1 mm above the
sheet's underside); it is unusable for a simulation as written.

| part | variant | release RMS vs target | rim-band bias | depth | effect vs benchmark | support force mean / peak [N] | primary force mean / peak [N] |
|---|---|--:|--:|--:|--:|--:|--:|
| TC | benchmark | 0.767 | -1.151 | 3.173 | - | - | 566 / 968 |
| TC | support at target underside | 0.277 | -0.200 | 2.877 | **0.696 (1.119)** | 566 / 1576 | 712 / 1675 |
| TC | support 0.2 mm clear | 0.358 | -0.497 | 2.989 | 0.475 (0.792) | 446 / 1038 | 627 / 1249 |
| TC | support 0.5 mm clear | 0.606 | -0.910 | 3.110 | 0.172 (0.363) | 332 / 561 | 568 / 1116 |
| PY | benchmark | 0.699 | -1.082 | 2.671 | - | - | 528 / 828 |
| PY | support at target underside | 0.304 | -0.165 | 2.425 | **0.654 (1.434)** | 688 / 2283 | 748 / 2130 |
| EC | benchmark | 0.841 | -1.241 | 3.296 | - | - | 629 / 1149 |
| EC | support at target underside | 0.305 | -0.133 | 3.117 | **0.814 (1.508)** | 1325 / 3328 | 1325 / 3411 |

What the support does on these small parts: the 4 mm support cannot reach
the wall under the first contours (the rim is concave from below), so it
rides under the flange 1-3 mm outside the part edge (centre radius 8-10 mm on
TC) and holds it up - a moving local backing plate; deeper, it reaches the
lower wall. That is exactly where the one-sided model's error is: the rim
sag goes from -1.15 to -0.20 mm. The part comes out 0.2-0.3 mm shallower in
the deep region (bias +0.26 to +0.30), because the primary no longer drags
the rim down with it. Forces rise: the primary +26-42 % (EC: x2), the
support carries 0.45-1.3 kN mean - with a position-controlled support at the
nominal geometry the two tools squeeze the sheet; a force-controlled support
(the Northwestern DSIF practice [L5]) would carry less and correct less.
Caveats: the support touches 1-2 nodes of a 2 mm mesh (under-resolved
contact), and position control is an idealisation; the 0.2 mm-clearance run
shows the sensitivity (0.48 instead of 0.70 mm).

### 2. Robot and stylus compliance

`robot_stiffness.py` (removed in the cleanup; at commit 7d571e6, with precomp.robot and `robot_deflection.py`; the
results below are the committed record): C = J K^-1 J^T at the tool for an ABB
IRB 7600-500/2.55-class arm with the joint stiffnesses identified by Bharti
et al. [L1] (approximate public DH dimensions; rigid links - real robots are
softer); tool 0.2-0.45 m, reach 1.4-2.2 m (`robot_compliance.json`):

* Machina-style vertical sheet (inference from "a rigid frame in between
  them holding up the feedstock sheet" [S2]), the robot pushing along its reach:
  0.07-0.27 um/N along the sheet normal but 0.5-2.1 um/N in the sheet plane.
* Horizontal sheet, the robot pressing down: 0.26-0.93 um/N normal to the
  sheet, 0.5-2.0 um/N in plane. So the stiff direction is the push direction
  only in the vertical cell, and in both cells the in-plane compliance is
  that of a "1 N/um" robot.
* Measured on an IRB 7600 in robotic ISF on a horizontal sheet [L1]:
  0.9-1.05 mm mean resultant deviation from compliance (0.5 mm mean along the
  tool axis early in forming, up to 0.75 / 0.52 mm in plane) at forces up to
  ~475 N - an effective compliance of 1-2 um/N or more (inference), above
  the rigid-link VJM's 0.3-0.9 um/N normal to such a sheet.

Tool-centre deflection along the 8 test parts' paths with their simulated
forces (`robot_deflection.csv`, mean over parts; in-contact points):

| compliance | mean abs | p95 | max | mean normal | max normal | mean in-plane | max in-plane |
|---|--:|--:|--:|--:|--:|--:|--:|
| 1 um/N isotropic | 0.52 | 0.85 | 0.92 | 0.50 | 0.91 | 0.12 | 0.32 |
| 2 um/N isotropic ([L1] effective) | 1.04 | 1.69 | 1.85 | 1.00 | 1.81 | 0.25 | 0.64 |
| IRB 7600, vertical sheet | 0.17 | 0.34 | 0.44 | 0.07 | 0.12 | 0.15 | 0.43 |
| IRB 7600, horizontal sheet | 0.34 | 0.62 | 0.76 | 0.27 | 0.54 | 0.21 | 0.54 |

The engine then forms the part along the path the robot actually takes,
p = p_cmd + C f(p). f depends on p (a deflected tool indents less), so this
is a fixed point: iteration 1 uses the rigid-path forces (upper bound),
iteration 2 the forces of iteration 1 - which oscillates (where iteration 1
lost contact at the bottom, iteration 2 is not deflected at all: 0.07 mm) -
so iterations 3-5 are under-relaxed (mean of input and output forces,
smoothed over 3 mm of path against the 1-2-node contact noise):

| TC, 1 um/N | mean normal deflection | release RMS vs target | depth | effect vs benchmark | mean force [N] |
|---|--:|--:|--:|--:|--:|
| benchmark (rigid) | 0 | 0.767 | 3.173 | - | 566 |
| it 1 (rigid-path forces) | 0.54 | 0.651 | 2.892 | 0.345 (0.529) | 462 |
| it 2 (forces of it 1) | 0.38 | 0.729 | 3.216 | 0.071 (0.189) | 512 |
| it 3 (relaxed) | 0.46 | 0.650 | 2.900 | 0.298 (0.462) | 498 |
| it 4 (relaxed) | 0.46 | 0.658 | 2.642 | 0.413 (0.589) | 467 |
| it 5 (relaxed, converged) | 0.46 | 0.662 | 2.628 | **0.413 (0.587)** | 482 |
| IRB vertical sheet (first order) | 0.07 | 0.732 | 3.079 | 0.076 (0.155) | 534 |
| IRB horizontal sheet (first order) | 0.28 | 0.681 | 3.001 | 0.220 (0.372) | 493 |
| PY, 1 um/N (first order) | - | 0.601 | 2.478 | 0.321 (0.501) | 442 |
| EC, 1 um/N (first order) | - | 0.728 | 2.956 | 0.389 (0.672) | 481 |

The shape follows the deflection almost one to one in the deep part
(0.47 mm RMS there for 0.46 mm mean deflection) and by about 0.6 in the rim
band (0.28 mm). The "improvement" in RMS vs target is incidental: the benchmark parts
are too deep, a compliant robot makes them shallower.

Stylus (`stylus.py`, steel cantilever): 16 mm x 100 mm - 0.07 mm at the mean
lateral force (136 N), 0.16 mm at the peak (321 N); 10 mm x 100 mm - 0.44 /
1.04 mm. Bharti et al. measured 0.14-0.19 mm for a 10 mm tool [L1].

Pre-compensating the path (precomp.robot.precompensate_path, removed in the cleanup; at commit 7d571e6) removes the
deflection to the accuracy of the force and compliance models: the forces of
this 2 mm mesh are 7 % (mean) to 23 % (peak) above the mesh-converged ones
(convergence lens, `h0.625_L1_im_tp7`: 528 / 747 N against 566 / 968 N), which
at 1 um/N leaves 0.04-0.22 mm; an uncalibrated compliance map (the rigid-link VJM is
softer than measured by a factor of about 2 or more, inference from [L1])
leaves most of the deflection.

### 3. Fixture: clamp width, free span, scale

| TC | window (free span) | release RMS vs target | rim-band bias | loaded-state rim bias | effect vs benchmark |
|---|--:|--:|--:|--:|--:|
| clamp 8 mm (tight backing plate) | 24 mm | 0.561 | -0.841 | -0.753 | 0.215 (0.390) |
| benchmark, clamp 5 mm | 30 mm | 0.767 | -1.151 | -1.029 | - |
| blank 60 mm, clamp 5 mm (wide frame) | 50 mm | 1.040 | -1.546 | -1.372 | 0.283 (0.469) |
| part and frame x2 (31 mm part, 60 mm window; tool, sheet, step, mesh size unchanged) | 60 mm | 1.449 | -2.338 | -2.110 | (different grid) |
| PY, blank 60 mm (wide frame) | 50 mm | 0.959 (bench 0.699) | -1.476 (bench -1.082) | -1.211 | 0.276 (0.497) |

The x2 run (`scale2`, 4 024 s, 1 631 increments) formed the part 6.30 mm deep
for a 6.18 mm target; relative to size its error is about the same
(1.45 / 2 = 0.72 mm against 0.77 mm): the absolute error roughly doubles with
the part. Nothing here supports an extrapolation to 0.3-3 m parts, which
need their own models (explicit forming) and data.

The rim sag is formed in: the loaded state already has most of it, and it
grows with the free sheet around the part. In SPIF practice the backing
plate follows the part outline a few mm out; Machina's frame holds the whole
sheet and uses "skirting" geometry [S2], and the second robot can support
the free sheet (section 1). Either way the model must carry the real frame,
skirt and support.

### 4. Path parameters, tool, temper

| TC | change | release RMS vs target | depth | effect vs benchmark |
|---|---|--:|--:|--:|
| step-down 0.5 mm | benchmark 1 mm (0.5 mm in [L1]; Machina's Ra <= 3.2 um implies sub-mm steps, inference) | 0.793 | 3.150 | 0.032 (0.088) |
| tool radius 6 mm | benchmark 4 mm (5 mm in [L1]) | 0.691 | 3.055 | 0.141 (0.229) |
| AA5052-H32-like temper (yield 193 MPa, Voce +50 MPa, H 50 MPa; inference-level card) | benchmark AA5754-O (Machina's example alloy is 5052-H32 [S2]) | 0.770 | 3.315 | 0.065 (0.160); springback 0.12 -> 0.27 mm RMS |
| constant path error: tool 0.3 mm short (+z) at every contact point | robot absolute accuracy 0.1-0.3 mm calibrated [L3] | 0.681 | 2.886 | 0.217 (0.304); depth -0.29 |

Friction (0.05-0.2: <= 0.003 mm), strain rate and heating (<= 0.011 mm,
feeds 1-6 m/min) are in the material lens (`../material/README.md`,
`rate_temperature.json`); a rotating or roller tool [S3] lowers friction,
which the engine cannot represent (no tool spin) but which does not matter
for shape here.

### 5. Measurement state and alignment (8 test parts, no new runs)

`measure_states.py` -> `measure_states.csv` (mean over the 8 parts):

| state / alignment | RMS vs target | rim-band bias |
|---|--:|--:|
| loaded (end of forming, tool at the bottom) | 0.645 | -0.947 |
| unloaded, still clamped (in-cell scan) | 0.636 | -0.931 |
| released onto 3-2-1 (benchmark metric) | 0.731 | -1.086 |
| released, 3-DOF best fit (z + tilts over the part) | 0.526 | - |
| released, 6-DOF robust ICP over the scanned surface | 0.659 | - |

Springback proper is small on these parts: 0.14 mm RMS free, 0.02 mm in the
clamp. Of the benchmark's 0.731 mm the loaded shape already has 0.645 mm
(78 % of the squared error): it is formed in, not sprung back. Scanner: the
patent's example 0.5 mm point accuracy [S3] is noise of the same size as the
effects above; area averages are needed to validate at 0.1 mm.

### 6. Gravity (`gravity.py`, CalculiX S4 shells, flat sheet = upper bound for flanges)

| case | 3-2-1 on corners | clamped in a horizontal frame (FE / Timoshenko) |
|---|--:|--:|
| benchmark 40 x 40 x 1 mm | 0.0014 | 0.000004 / 0.000004 |
| 0.6 m x 1 mm | 70 | 0.31 / 0.32 |
| 1.2 m x 2 mm | 281 | 1.80 / 1.88 |
| 1.2 m x 3.2 mm | 110 | 0.71 / 0.74 |
| 1.5 m x 3.2 mm | 268 | 1.86 / 1.93 |

Irrelevant for the benchmark; for product-size parts gravity decides the
measured shape unless the part is scanned in its frame or on a fitted
fixture - the flat-plate numbers (linear, a formed shell is far stiffer) show
that a 3-2-1 release of a large thin part is not a measurement set-up. In
Machina's vertical frame gravity acts in the sheet plane while forming. The
engine has no body loads.

### 7. Not computable here: trimming, heat treatment, closed-loop control

* **Trimming.** Machina trims robotically (spindle, laser, plasma) [S3]; the
  flange and skirt carry residual stress that trimming releases. SparLab has
  no element removal and writes no stress tensor (only von Mises), so neither
  it nor a CalculiX restart can trim the formed state. Bound (inference): of
  the order of the release springback, 0.1-0.2 mm RMS on these parts; the
  literature reports substantial deviation "as soon as parts are unclamped
  from the blank holder and trimmed to final shape" [L8].
* **Heat treatment.** Stress relief or ageing (2000/6000/7000 alloys) after
  forming [S1, S2] relaxes residual stress (the part moves, of the order of
  the springback) and a solution treatment and quench distorts thin parts by
  more; the engine is isothermal and rate-independent. Annealing in a
  fixture reduces unclamping springback [L6].
* **Closed-loop control.** The real cell corrects on force, displacement and
  in-process scans [S1-S3]; an open-loop simulation of the nominal path
  predicts a part the cell never makes. Simulate the executed (logged) path.

## Recommendations (ranked by effect on the predicted shape)

1. **Model the two-robot process, not SPIF** (effect 0.5-0.8 mm RMS here).
   Get Machina-style two-robot paths (or their strategy) and run them as two
   tools; the engine does this today (`proclens.py` patches the deck: second
   sphere on the bottom faces, its own trajectory on the same pseudo-time).
   Cost: 2-3 days for a non-gouging support-path generator in precomp to
   replace the experimental `dsif_support_path`; 1-2 weeks for a
   force-controlled or spring-mounted tool in sparlab_form (DSIF supports are
   force-controlled in practice [L5]). Until the strategy is known, the
   benchmark's rim-sag conclusion ("no command at z <= 0 can fix it") does not
   hold for a two-sided cell: a support removes most of it.
2. **Put the robot compliance into the forming loop** (0.1-0.4 mm RMS, up to
   0.5 mm in depth). Cheapest now: under-relaxed outer iteration with
   precomp.robot (removed in the cleanup; at commit 7d571e6; 3-5 runs per part, this lens's `rob_iso1_it3..5`: input and
   output forces within 2-3 % in the mean). Better: a compliant tool in sparlab_form (the
   tool centre = command + C f, solved with the contact; ~1 week). Data:
   the Cartesian compliance of each robot over the frame (laser tracker and
   known loads, 1-2 days per robot; the rigid-link VJM from published joint
   stiffnesses gives less deflection than [L1] measured), stylus stiffness, and
   logged TCP paths and forces for validation. The compensation then needs
   forces from a mesh-converged model (the benchmark mesh overestimates the
   peak by 23 %, 0.2 mm at 1 um/N).
3. **Model the real fixture and part scale** (0.2-0.3 mm RMS per fixture
   change here; the rim sag grows with the free span). The frame, the
   unsupported span and any skirt/addendum have to be in the deck as built;
   parts of 0.1-1 m need `form_explicit` (hours per run). A benchmark on
   14-22 mm parts with a 4 mm tool cannot be transferred to Machina-size
   parts by scaling (section 3).
4. **Measure what the shop measures** (0.1-0.2 mm from the definition
   alone). Compare in-cell scans with the `unload` state, released parts
   with a model of the same fixture and gravity, and use the same alignment
   (best fit or datums). No development.
5. **Add trimming and body loads** (trim not computable today; gravity mm
   at product scale): element removal in a release step with full stress
   output (1-2 weeks), gravity as a body load (1-2 days). Heat treatment:
   empirical correction per alloy/temper from measured parts until a
   relaxation model exists.
6. **Validation data - without it "almost exactly" cannot be claimed.**
   Three parts of different families on the real cell, SPIF and two-robot,
   with logged robot paths, forces, a tracker-measured TCP path on one of
   them, and scans at four states (loaded, unloaded in frame, released,
   trimmed). Order of cost: a week of cell time plus scanning.

**Achievable accuracy.** With items 1-4 in place, the remaining error at
this part size would be the material card (0.02-0.05 mm, material lens),
discretisation (element lens), the compliance and force models (20-30 %
uncertainty of a 0.2-0.5 mm deflection: 0.05-0.15 mm), the support
strategy's force control, and the unmodelled trim / heat treatment (~0.1-0.2
mm): about **0.1-0.3 mm RMS**. "Almost exactly" (a few hundredths of a mm) is
not reachable on the machine side: robot calibration (0.1-0.3 mm) and the
scanner (0.5 mm per point) are larger than that.

## Files

| file | content |
|---|---|
| `proclens.py` | deck building (DSIF support, offsets, scaling, material; the robot-deflection option is removed in the cleanup; at commit 7d571e6), running, measurement |
| `variants.py`, `run_queue.py`, `run_queue2.py` | the variant queue and its workers (at most two runs at a time, one thread) |
| `results.jsonl` -> `summarize.py` -> `results.csv` | every run: deviation from target (form / unload / release; part, rim band, deep part), change vs the benchmark run, forces, cost |
| `runs/<part>/<variant>/` | decks, tool paths (`toolpath.csv`, `support.csv`), `variant.json` (spec, deflection statistics), outputs, `surfaces.npz`; `runs/<part>/base/output` links the benchmark cache |
| `robot_compliance.json` | IRB 7600 VJM compliance at 42 poses (made by `robot_stiffness.py`, removed in the cleanup; at commit 7d571e6) |
| `robot_deflection.csv` | path deflection of the 8 test parts for four compliance models (made by `robot_deflection.py`, removed in the cleanup; at commit 7d571e6) |
| `measure_states.py` -> `measure_states.csv` | loaded / in-frame / released / best-fit deviations, springback, forces of the 8 test parts |
| `stylus.py` -> `stylus.json` | stylus bending |
| `gravity.py` -> `gravity/gravity.json` | CalculiX self-weight sag |
| `plot_profiles.py` -> `fig_profiles.png`, `profiles.csv` | radial profiles of TC for the variants (released and loaded) |
| `run_queue.py` | the first workers' version (claims `*.claim`); `run_queue2.py` claims `*.claim2` and reloads the helpers - use it |

`.gitignore` leaves out the raw solver output (VTK, node and element tables,
CalculiX result files; regenerated bit for bit from the decks) and the
links to the benchmark cache; decks, paths, summaries, tool forces and the
evaluated surfaces are kept.

Reproduce: `python3 run_queue2.py A` (and a second worker) runs the queue (delete `runs/*/*.claim2` first; finished variants are skipped);
`python3 summarize.py measure_states.py plot_profiles.py`
(one at a time) rebuild the tables.

## Sources

* [S1] Machina Labs, Capabilities, https://machinalabs.ai/capabilities (verified)
* [S2] Machina Labs, "Advanced Manufacturing: Incremental Sheet Metal Forming with Robotics and AI", https://machinalabs.ai/resources/advanced-manufacturing-incremental-sheet-metal-forming-with-robotics-and-ai (verified)
* [S3] E. Mehr (Machina Labs), US 11,865,716 B2, "Part forming using intelligent robotic system", https://patents.google.com/patent/US11865716B2/en (verified)
* [L1] S. Bharti et al., "Systematic analysis of geometric inaccuracy and its contributing factors in roboforming", Sci. Rep. 14 (2024), https://www.nature.com/articles/s41598-024-70746-3 (verified: text, Tables 3 and 5)
* [L2] Z. Pan, H. Zhang, robotic machining with force control (articulated robots "usually less than 1 N/um" at the tool, vs tens of N/um for machine tools), https://www.researchgate.net/publication/235249813 (search-engine summary only, not read)
* [L3] M. Franaszek, G. S. Cheok, "Using locally adjustable hand-eye calibrations to reduce robot localization error", SN Applied Sciences 2 (2020) 839, https://doi.org/10.1007/s42452-020-2626-2, PDF https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=928169 (verified text: accuracy about two orders of magnitude worse than repeatability; median 0.221 mm after local calibration; cited KUKA KR 210 0.106 mm average, KR 500-3 0.32 mm max after compensation)
* [L4] H. Meier et al., "Robot-Based Incremental Sheet Metal Forming - Increasing the Geometrical Accuracy of Complex Parts", Key Eng. Mater. 473 (2011) 853, https://doi.org/10.4028/www.scientific.net/KEM.473.853 (abstract)
* [L5] J. Cao lab, Northwestern, "Double-Sided Incremental Forming", https://www.cao.mech.northwestern.edu/flexible-metal-processes-dsif/ (verified)
* [L6] Z. Zhang et al., "Springback Reduction by Annealing for Incremental Sheet Forming", Procedia Manuf. 5 (2016), https://doi.org/10.1016/j.promfg.2016.08.057 (abstract)
* [L8] "Experimental and Numerical Investigation of the Influence of Process Parameters in Incremental Sheet Metal Forming on Residual Stresses", J. Manuf. Mater. Process. 3 (2019) 31, https://doi.org/10.3390/jmmp3020031 (search-engine summary only; the publisher blocked access)
* [L7] T. Grimm et al., "Single Point Incremental Forming Springback Reduction Using Edge Stiffener", IMECE2020-23205, https://doi.org/10.1115/IMECE2020-23205 (abstract)
