# Springback benchmark on the fine mesh, with and without support from below

Every formed part here is a **SparLab simulation** (`sparlab_form`: implicit,
quasi-static, `finite_logarithmic` kinematics, Hill48 plasticity with
**nominal handbook data** for AA5754-O). Nothing was formed or measured on a
machine. The numbers compare strategies and compensation methods *on this
model*. They are not a claim about real parts (see [Limitations](#limitations)).

This is stage 1 of the plan the physics audit recommended
(`benchmarks/physics_audit/README.md`, fixes 1 and 2). It re-runs the 8
held-out test parts of `benchmarks/springback` on the audit's converged mesh
instead of the 2 mm one. Each part is formed three ways: single-point
forming as before, on a backing plate, and with DSIF. Each runs uncompensated
and after one and two steps of FE displacement adjustment (DA).

## Headline

Protocol revision 2: 8 parts x 3 strategies x 3 methods = 72 simulations,
all complete. Deviation of the released part from the target over the part,
mean over the 8 test parts [mm]. **Rim sag** is the mean vertical deviation
of the part less than 1 mm deep (negative means too deep). **Interior** is
the part deeper than that.

| strategy | method | FE runs / part | vertical RMS | median | max | rim sag | interior RMS (bias) | parts better than own uncompensated | 2 mm benchmark, same method |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| single-point | uncompensated | 1 | **0.665** | 0.652 | 1.218 | -0.994 | 0.284 (-0.18) | - | 0.731 |
| single-point | FE-DA-1 | 2 | **0.555** | 0.555 | 1.093 | -0.819 | 0.232 (+0.02) | 8 / 8 | 0.623 |
| single-point | FE-DA-2 | 3 | **0.542** | 0.533 | 1.065 | -0.776 | 0.254 (+0.10) | 8 / 8 | 0.602 |
| backing plate, 1 mm clearance | uncompensated | 1 | **0.300** | 0.298 | 0.583 | -0.436 | 0.149 (-0.05) | - | |
| backing plate | FE-DA-1 | 2 | **0.263** | 0.252 | 0.535 | -0.334 | 0.162 (+0.08) | 7 / 8 | |
| backing plate | FE-DA-2 | 3 | **0.272** | 0.259 | 0.595 | -0.318 | 0.188 (+0.12) | 6 / 8 | |
| DSIF, sine-law gap | uncompensated | 1 | **0.273** | 0.250 | 0.576 | -0.215 | 0.240 (+0.17) | - | |
| DSIF | FE-DA-1 | 2 | **0.265** | 0.263 | 0.624 | -0.292 | 0.169 (+0.08) | 5 / 8 | |
| DSIF | FE-DA-2 | 3 | **0.277** | 0.278 | 0.679 | -0.326 | 0.164 (+0.09) | 4 / 8 | |

The full tables are in `tables.md`: normal RMS, upper-band RMS, flange
bias, runtime, Newton iterations and the per-part numbers. Per-case rows
with the physics checks are in `cases.csv`, and the tool forces and contact
are in `tools.csv`.

**In one sentence:** on the converged mesh, single-point forming is about
0.06-0.07 mm better than the 2 mm benchmark said, as the audit predicted,
and DA still only trims it (0.665 to 0.542 mm). Support from below more than
halves the deviation before any compensation (plate 0.300 mm, DSIF 0.273 mm).
With support, FE displacement adjustment gains little or nothing: one step
gives the plate 0.263 mm, DSIF 0.265 mm, and a second step makes both
slightly worse.

## What the numbers say

* **The mesh.** Formed with a single tool (no support), the fine mesh gives
  0.066 mm less RMS than the 2 mm benchmark uncompensated, 0.068 mm less
  after DA-1 and 0.060 mm less after DA-2. The rim sag is 0.09 mm
  shallower. The audit predicted a 0.06-0.11 mm bias in the 2 mm numbers,
  and these are at its low end. The conclusion of `benchmarks/springback`
  stands on the converged mesh: in single-point forming the rim sag
  (-0.99 mm uncompensated, -0.78 mm after DA-2) dominates, and no command at
  z <= 0 removes it.
* **Support.** On every part a backing plate or DSIF more than halves the
  uncompensated deviation (plate/none 0.45 on average). The flange comes out
  almost flat: bias -0.05 mm on the plate and +0.03 mm with DSIF, against
  -0.22 mm with a single tool. The rim sag falls from -0.99 mm to -0.44 mm on
  the plate and to -0.22 mm with DSIF. DSIF is better than the plate on 5 of
  8 parts uncompensated, and 0.027 mm better on average. As on the cone of
  `benchmarks/support_cone` (removed in the cleanup; at commit 7d571e6), its error moves rather than goes: the deeper
  part comes out 0.17 mm too shallow. The sine law gives a 0.78 mm gap
  (median over the walls), while the model's wall is 0.96 mm thick (median),
  so the support squeezes the wall wherever it reaches its opposite
  position.
* **Compensation with support.** One DA step improves the plate parts on 7
  of 8 (mean -0.037 mm) and the DSIF parts on 5 of 8 (mean -0.008 mm). A
  second step helps 3 of 8 plate and 2 of 8 DSIF parts relative to the first,
  and both means get worse. DA corrects the rim (plate sag -0.44 to -0.33 mm)
  but makes the deeper part too shallow (plate interior bias -0.05 to +0.08
  to +0.12 mm). Without truncated_cone-s2026-0001 (below) the means are:
  plate 0.294 / 0.245 / 0.250, DSIF 0.260 / 0.243 / 0.254. So one step is
  worth about 0.05 mm on the plate and 0.02 mm with DSIF, and the second step
  is worth nothing. The best results are plate FE-DA-1 (0.263 mm) and DSIF
  FE-DA-1 (0.265 mm). The difference between them is well inside the
  spread between parts (plate FE-DA-1 ranges from 0.21 to 0.39 mm).
* **Every run is physically sane** by the checks recorded in `cases.csv`.
  Every step completed with 0-5 cut-backs, all recovered. The 3-2-1 release
  leaves at most 7e-13 N. Each run has one warning: the clamp reaction in
  "unload" (127-216 N). It is flagged because the clamped unload is
  statically indeterminate by design, and it is the springback the fixture
  holds.

Per part, vertical RMS [mm], uncompensated / FE-DA-1 / FE-DA-2:

| part | single-point | backing plate | DSIF |
|---|--:|--:|--:|
| dome-s2026-0000 | 0.641 / 0.477 / 0.468 | 0.307 / 0.213 / 0.203 | 0.347 / 0.281 / 0.298 |
| dome-s2026-0001 | 0.638 / 0.543 / 0.489 | 0.290 / 0.223 / 0.216 | 0.225 / 0.310 / 0.258 |
| elliptic_cone-s2026-0000 | 0.602 / 0.490 / 0.483 | 0.277 / 0.252 / 0.294 | 0.254 / 0.289 / 0.327 |
| elliptic_cone-s2026-0001 | 0.751 / 0.664 / 0.659 | 0.305 / 0.266 / 0.260 | 0.307 / 0.200 / 0.228 |
| pyramid-s2026-0000 | 0.684 / 0.566 / 0.524 | 0.290 / 0.213 / 0.217 | 0.195 / 0.190 / 0.148 |
| pyramid-s2026-0001 | 0.648 / 0.538 / 0.542 | 0.308 / 0.299 / 0.304 | 0.246 / 0.245 / 0.309 |
| truncated_cone-s2026-0000 | 0.698 / 0.572 / 0.576 | 0.280 / 0.251 / 0.257 | 0.246 / 0.187 / 0.212 |
| truncated_cone-s2026-0001 | 0.657 / 0.592 / 0.597 | 0.344 / 0.386 / 0.426 | 0.364 / 0.415 / 0.438 |

## Revision 1, its failure and its regression

Revision 1 (kept in `rev1/`) produced 70 of 72 simulations. Two of its
results were wrong for reasons in the decks, not in the physics:

* **The failed DSIF run.** pyramid-s2026-0001 DSIF FE-DA-1 stalled. Its time
  step collapsed at t = 0.774 of 0.953 s (6e-6 s increments, 2 400
  increments and 19 000 Newton iterations in 9 700 s) and the run was killed
  (exit -15). In that deck the support path pinched the sheet: the minimum
  planned gap between support and tool was -0.19 mm
  (`rev1/deck_defects.csv`). The support knots were placed only at the
  tool's knots, so where the tool swings round a corner the straight segment
  between two support knots cut through the formed sheet. Commit 4c1bb91
  adds support knots there. In revision 2 the run completed (0.245 mm, 3 649
  s) and so did FE-DA-2 (0.309 mm).
* **The large FE-DA-1 regression on truncated_cone-s2026-0001.** In revision
  1 one DA step took this part from 0.344 to 0.550 mm on the plate and from
  0.364 to 0.679 mm with DSIF, with the deeper part 0.61 / 0.78 mm too
  shallow. Every DA command on this part asked for more depth than the tool
  path delivered. The path's deepest level stayed at 2.0 mm, though the tool
  could reach 2.87-3.00 mm into the command (shortfall 0.87 mm on the plate,
  1.00 mm with DSIF, `rev1/deck_defects.csv`), because a dimpled DA floor or
  pole dropped the deepest level. Commit 4e13c62 keeps that level. With it the
  regression shrinks to 0.344 to 0.386 mm (plate) and 0.364 to 0.415 mm
  (DSIF), but it does not go away (next section).

Two more spiral defects showed up while revision 2 ran. Each made a DA command's
deck refuse to build, and none changed a deck that did build:

* The last revolution, blended towards a small off-centre last loop, crossed
  the surface on the far side of the pocket (98bd680; elliptic_cone-s2026-0000
  plate FE-DA-2).
* The last revolution's line only grazed the last loop at its nearest point,
  with no sign change next to it (31baad2; pyramid-s2026-0000,
  pyramid-s2026-0001 and truncated_cone-s2026-0001 single-point FE-DA-2).
  The point fell at the grid's edge, 24 mm out with the tool, and the window
  check refused the deck.

The tool paths of all 123 decks in the run cache are identical before and
after 31baad2.

Revision 2 against revision 1 (mean vertical RMS [mm]; revision 1's DSIF DA
rows are over 7 parts):

| | none unc. | none DA-1 | none DA-2 | plate unc. | plate DA-1 | plate DA-2 | DSIF unc. | DSIF DA-1 | DSIF DA-2 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| revision 1 | 0.667 | 0.569 | 0.556 | 0.308 | 0.305 | 0.272 | 0.295 | 0.354 | 0.274 |
| revision 2 | 0.665 | 0.555 | 0.542 | 0.300 | 0.263 | 0.272 | 0.273 | 0.265 | 0.277 |

The single-point and plate uncompensated decks of the cones and pyramids did
not change. Every DSIF deck changed (support knots), and so did the domes'
decks, because their deepest level now reaches the pole: dome-s2026-0001
DSIF went from 0.336 to 0.225 mm. Revision 1's DSIF and plate DA rows should not
be quoted.

## The DA overshoot on truncated_cone-s2026-0001, and alpha = 0.5

This part has the steepest wall of the test set (54 deg, 8.1 mm top radius,
2.63 mm deep, a wide flat floor). It is the only part where DA makes the plate
result worse, the worst case with DSIF, and the only part where the second
step makes it worse even without support. Azimuthal means of the plate chain [mm], from the run cache
(target floor at -2.63 mm out to r = 5.5 mm, rim at r = 8.1 mm):

| r [mm] | target | command 0 | formed 0 | command 1 | formed 1 | command 2 | formed 2 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 0-1 | -2.63 | -2.63 | -2.40 | -2.85 | -2.96 | -2.52 | -2.75 |
| 2-3 | -2.63 | -2.63 | -2.53 | -2.72 | -2.74 | -2.61 | -2.65 |
| 4-5 | -2.63 | -2.63 | -2.66 | -2.60 | -2.50 | -2.73 | -2.39 |
| 5-6 | -2.61 | -2.61 | -2.41 | -2.82 | -2.18 | -3.25 | -2.05 |
| 6-7 | -2.17 | -2.17 | -1.90 | -2.44 | -1.64 | -2.34 | -1.51 |
| 7.5-8 | -0.65 | -0.65 | -1.11 | -0.21 | -0.96 | -0.07 | -0.90 |
| 8.5-9 | -0.09 | -0.09 | -0.63 | 0.00 | -0.57 | 0.00 | -0.54 |

Uncompensated, the lower wall and floor corner come out about 0.2-0.27 mm
too shallow and the upper wall 0.5 mm too deep. DA raises the upper wall
command to the plane and deepens the lower wall and the floor corner. The
upper wall barely responds (-1.11 to -0.96 mm). The floor corner, though,
comes out *shallower* as the command there deepens: at r = 5-6 mm the
command goes -2.61 / -2.82 / -3.25 and the result -2.41 / -2.18 / -2.05.
Two things cause this:

1. **The command stops being formable by the tool.** DA cuts a narrow
   trench into the floor corner and lifts the upper wall towards the plane.
   The 4 mm tool cannot follow that. The deepest tool tip reaches 2.85 /
   2.70 mm under commands 2.94 / 3.59 mm deep on the plate, 3.00 / 3.07 mm
   under 3.14 / 3.84 mm with DSIF, and 2.75 / 2.66 mm under 2.87 / 3.48 mm
   in single-point forming. Up to 0.89 mm of commanded depth is never
   formed, and DA's per-node update assumes that it is.
2. **The response is non-local.** Raising the upper wall and steepening the
   wall moves the lower wall the wrong way, so the vertical error at a node
   says little about how to move the command at that node.

Damped DA, alpha = 0.5, on this part (`alpha_0p5/`, 6 new runs; the
uncompensated runs are shared):

| strategy | uncompensated | alpha 1: DA-1 / DA-2 | alpha 0.5: DA-1 / DA-2 | deepest commanded / tool tip at DA-2, alpha 0.5 [mm] |
|---|--:|--:|--:|--:|
| single-point | 0.657 | 0.592 / 0.597 | 0.601 / 0.593 | 3.00 / 2.65 |
| backing plate | 0.344 | 0.386 / 0.426 | 0.366 / 0.368 | 3.06 / 2.70 |
| DSIF | 0.364 | 0.415 / 0.438 | 0.401 / 0.396 | 3.21 / 2.86 |

**The overshoot remains under damping.** Alpha = 0.5 halves the damage (plate
DA-2 0.368 instead of 0.426 mm) and stops the divergence between steps.
Even so, with either support it is still worse than not compensating at all
(0.366-0.368 against 0.344 mm on the plate, 0.396-0.401 against 0.364 mm
with DSIF). Without support it is a wash (0.593 against 0.592-0.597 mm). The
direction of the update is wrong on this part, not only its size, so a
smaller step cannot fix it. Two remedies follow from the diagnosis, and
neither has been run:

* Project each DA command onto what the tool can form (the drop-cutter
  envelope: the command's opening by the tool sphere) before it is used.
* Keep the best iterate as the delivered command. `displacement_adjustment`
  already does this, but the benchmark reports every iterate, and that is
  what a shop would see run by run.

## Penalty 3 with support (for the ML data)

The audit measured contact penalty 3 against 10 only in single-point forming
(+0.005-0.009 mm RMS, half the cost). With support, the plate and the DSIF
support get the tool's penalty, so the same setting changes those contacts
as well. We re-ran the uncompensated plate and DSIF runs of one part per
family with the `springback_fine_bulk` preset (`penalty3/`, 8 runs, 4 at a
time):

| strategy | parts | RMS change, penalty 3 minus 10 [mm] | rim sag change [mm] | Newton iterations, ratio | runtime, ratio | runtime per run, penalty 3 |
|---|--:|--:|--:|--:|--:|--:|
| backing plate | 4 | +0.003 (0.000 to +0.009) | -0.005 (-0.014 to -0.001) | 0.49 (0.20-0.65) | 0.50 | 1 164 s (675-1 569) |
| DSIF | 4 | -0.000 (-0.003 to +0.002) | -0.006 (-0.011 to -0.003) | 0.77 (0.63-0.83) | 0.89 | 1 992 s (1 154-2 333) |

Penalty 3 is accurate enough for bulk data with either support: at most
0.009 mm RMS, always towards slightly more sag, as in the audit. It halves
the cost only on the plate. With DSIF the two balls' contact dominates the
Newton work and the saving is about 10 %.

## Recommendation for the ML dataset stage (stage 2)

* **Strategy: the backing plate, 1 mm clearance.** It and DSIF give the same
  best result (0.263 / 0.265 mm after one DA step), but the plate is the
  better target for a surrogate:
  * Its DA response is regular: one step helps 7 of 8 parts, where DSIF
    helps 5 of 8 with a mean gain of 0.008 mm. A surrogate that emulates
    FE-DA has something to learn on the plate and almost nothing with DSIF.
  * Penalty 3 halves the plate's cost but saves only about 10 % with DSIF.
  * DSIF's lead before compensation rests on the sine-law gap squeezing the
    model's thicker wall (`benchmarks/support_cone`), and DSIF results moved
    the most when the support path was fixed (dome-s2026-0001: 0.336 to
    0.225 mm). The plate is less sensitive to deck details, and it is the
    simpler fixture.
* **Setup: `springback_fine_bulk`** (penalty 3) for training and calibration
  data. The test parts are evaluated at penalty 10, against the numbers
  above (`springback_fine`). Those plate test numbers already exist, so the
  ML shots are the only new test runs.
* **Data per part, as in `benchmarks/springback`:** the target and one FE-DA
  step from its simulated part (2 runs, with the plate's
  `compensation_masks` and `command_upper_bound`). Same Sobol design (seed
  2026), same nested split: indices 0-1 test, every 4th calibration, the
  rest train.
* **Size: stage n26, 72 train + 24 calibration parts** (26 design points per
  family, 192 runs). At 1 164 s per plate run at penalty 3, 4 at a time, plus
  about 20 % for the DA-1 variants' spread and part size, the data take about
  19 h on 4 cores. The ML first shots of 2 models on the 8 test parts at
  penalty 10 take about 4 h more (16 runs of about 3 400 s), so about 23 h in
  all. The design is nested, so run it as a learning curve: n18 (48 + 16
  parts, about 13 h) first, then n26 on top. If n26 lands well inside the
  budget, n34 (96 + 32 parts, about 6 h more, about 29 h in all) still fits
  in 36 h. Container restarts cost only the runs in flight, because the run
  cache is content-addressed.
* **Before generating data, fix two deck issues found here** (neither changes
  the conclusions above):
  1. *A gouge on the spiral's last revolution.* On 13 of the 48 DA decks,
     1-3 points of the last revolution keep the previous loop's position
     where no crossing is found. These points sit 0.04-0.68 mm below the
     drop-cutter surface (`toolpath_metadata.max_gouge_m`), so the tool
     presses into a DA-raised spot of the floor 1.5-5 mm from the centre.
     No uncompensated deck is affected.
  2. *DA commands the tool cannot form* (truncated_cone-s2026-0001, above).
     Clip each DA variant to the tool's reach, or the surrogate is trained
     on commands whose deepest 0.1-0.9 mm never reached the sheet.
* **Expectation, honestly:** on the plate the most a one-shot surrogate can
  win is roughly FE-DA-1's gain, 0.300 to 0.263 mm. That is small against
  the spread between parts, and it is measured on 8 test parts. The
  stage is worth running for the saved FE run per part and for the model's
  dz accuracy on the converged mesh. It is not likely to change the
  deviation much.

## Setup

| | |
|-|-|
| Parts | the 8 test parts of `benchmarks/springback` (design indices 0-1 of each family: truncated cone, pyramid, dome, elliptic cone; Sobol, seed 2026); 0.25 mm grid |
| Blank, material | 40 x 40 x 1 mm AA5754-O, nominal library data (Hill48 r0 / r45 / r90 = 0.75 / 0.70 / 0.80, linear + Voce hardening) |
| Mesh, solver | `springback_fine` preset: 48 x 48 x 1 Hex8 with incompatible modes (0.833 mm in plane), 5 points through the thickness, contact penalty 10; `finite_logarithmic`; implicit Newton with cut-backs; 8 h timeout |
| Fixture, tool, path | clamp 5 mm wide at the blank edge; 4 mm radius sphere, friction 0.1; spiral, 1 mm step-down, 1 mm spacing, 1 mm travel per increment; form - unload (clamped) - release onto 3-2-1 supports |
| Backing plate | `support: "backing_plate"`, clearance 1 mm to the target's outline, removed with the tool in "unload" |
| DSIF | `support: "dsif"`: 4 mm support ball, sine-law gap t cos(theta), support knots added where the support swings round the tool (4c1bb91) |
| DA | `displacement_adjustment`, vertical error, alpha = 1, no smoothing, 65 deg wall limit, support-aware masks and command bound (`compensation_masks`, `command_upper_bound`), the fixture made for the target; FE-DA-k is the part formed after k steps |
| Metric | vertical deviation of the released part from the target over the part (target below the plane): RMS and max; regions: upper band (< 1 mm deep), the deeper part (interior), the flange |

## Reproduce

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build -j3
benchmarks/springback_fine/run.sh                    # 4 workers, background, resumable
# the studies (same driver, own --out, shared run cache):
OPENBLAS_NUM_THREADS=1 PYTHONPATH=python python3 python/scripts/springback_fine_benchmark.py \
    --out benchmarks/springback_fine/alpha_0p5 --work benchmarks/springback_fine/work \
    --workers 3 --alpha 0.5 --parts truncated_cone-s2026-0001
OPENBLAS_NUM_THREADS=1 PYTHONPATH=python python3 python/scripts/springback_fine_benchmark.py \
    --out benchmarks/springback_fine/penalty3 --work benchmarks/springback_fine/work \
    --workers 4 --strategies backing_plate dsif \
    --parts dome-s2026-0000 elliptic_cone-s2026-0000 pyramid-s2026-0000 truncated_cone-s2026-0000
```

The penalty-3 study's `config.json` was written by hand from the protocol:
preset `springback_fine_bulk`, `fe_runs` 1. The driver reads it when it
exists. Every run goes through precomp's content-addressed cache in
`work/runs` (git-ignored). An identical deck never runs twice, so an
interrupted run repeats only the runs in flight. The solver is a copy of
`build/bin/sparlab_form` kept in `work/bin`: `sparlab 1.0.0 (bd26230795aa)`
for every run here.

Revision 2 ran from 1 October 19:40 to 2 October 07:55 UTC. It was
restarted twice: once after a container restart (only the runs in flight
were lost), and once to pick up the spiral fix 31baad2. The 4-core machine ran
4 simulations at a time, so the runtimes in `cases.csv` are under a load
average of about 4 and vary by up to a factor of 2 for the same work. The
Newton iteration counts do not depend on the load. Mean per run: single-point
1 350-1 660 s, plate 3 340-3 570 s, DSIF 2 580-2 840 s, or 0.46 / 0.99 / 0.72
CPU-h per part uncompensated (7-15 times the 2 mm runs' 243 s).

## Limitations

* **Simulation only**, with nominal material data. The converged mesh
  removes the discretisation bias the audit measured, not the model's other
  assumptions: no kinematic hardening, a rigid fixture and tools, no machine
  compliance.
* **Eight test parts, one run each.** Differences of a few hundredths of a
  millimetre, such as plate against DSIF after DA-1 (0.263 against 0.265 mm),
  are within what another part set could change. One part
  (truncated_cone-s2026-0001) moves the supported DA means by about 0.01 mm.
* **The DSIF gap follows the sine law**, which is 20 % thinner than the
  model's wall. DSIF's numbers include that squeeze. A gap of t was not run
  here; on one cone it gave a result close to the plate's
  (`benchmarks/support_cone`).
* **The plate's clearance is one value (1 mm)** and the plate is flat at the
  sheet plane. Nothing holds the sheet down onto it.
* **DA as specified, alpha = 1 without smoothing.** Alpha = 0.5 was run on
  one part only. The command is not clipped to the tool's reach, and 13 DA
  decks carry the last-revolution gouge described above.
* **Penalty 3 was checked on 4 parts uncompensated**, not on the DA variants.
