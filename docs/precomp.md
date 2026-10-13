# precomp: the Python layer for springback pre-compensation

`precomp` is the Python package around SparLab's forming solver `sparlab_form`
for robotic single-point incremental forming (SPIF). It turns a target part
into a tool path, a forming deck and a simulated formed shape, measures the
deviation of a formed shape from the target, and pre-compensates the
commanded shape by displacement adjustment (DA), with the finite-element model
or a learned surrogate in the loop (`precomp.ml`, [precomp_ml.md](precomp_ml.md)).

It adds no physics of its own. Every number it reports is either read from a
`sparlab_form` result or computed by a geometric procedure whose accuracy is
checked against an exact answer in `python/tests` ([Verification](#verification)).
Every formed shape in the benchmarks is a **SparLab simulation with nominal
material data**; nothing was formed or scanned.

## Install and test

```bash
pip install -e .                  # the package `precomp` from python/precomp
pip install -e '.[dev]'           # + pytest
pip install -e '.[torch]'         # + torch, for precomp.ml's MLPEnsemble
python3 -m pytest python/tests -q # 151 tests (42 for precomp.ml), about 3.5 min on a
                                  # 4-core machine; the 6 integration tests skip
                                  # without build/bin/sparlab_form
```

Python 3.10 or newer with numpy, scipy, pandas, scikit-learn, joblib, contourpy
and matplotlib; the tests and scripts import the package from the source tree.
`sparlab_form` is `FormingSetup.executable`, else `$PRECOMP_SPARLAB_FORM`, else
`build/bin/sparlab_form` relative to the working directory or the repository
root; its absence is an error naming every place looked at.

## Conventions

* **Units.** Strict SI ([conventions.md](conventions.md)): m, N, Pa, s,
  kg/m^3; angles carry their unit in their name (`wall_angle_deg`; radians
  otherwise). Millimetres appear only in human-facing output, labelled `[mm]`.
* **The surface.** The blank lies in the x-y plane, tool side at z = 0,
  material in z in [-t, 0]. A formed part is a height field z = f(x, y) <= 0 of
  the tool-side surface; targets, commanded and formed shapes are all a
  `HeightMap`, centred on the blank (and the mesh).
* **Arrays.** `z[ny, nx]` on a regular `Grid` of spacing `h`: row j is
  y = y0 + j h, column i is x = x0 + i h, x fastest (the node order of SparLab's
  structured grids). `z` is finite everywhere; nodes without data are `False`
  in `mask` and carry the nearest valid height.
* **Signs.** Normals point +z, towards the tool. A signed deviation is positive
  when the formed surface lies on the tool side of the target (too shallow).
  The solver's tool force is the force the sheet exerts on the tool (fz > 0
  while it pushes down; [forming.md](forming.md), section 3).

## Modules

There are no import cycles; `precomp.ml` builds on everything except `robot`.

| Module | Imports | Contents |
|--------|---------|----------|
| `precomp.geometry` | - | `Grid`, `HeightMap`, `zeros`, the part families (`TruncatedCone`, `Pyramid`, `Dome`, `EllipticCone`), `part_from_dict`, `families`, `MAX_WALL_ANGLE_DEG`; `geometry.raycast.raycast_top` (the result loader's ray cast) |
| `precomp.materials` | - | `Material` (elasticity, linear + Voce isotropic and Prager kinematic hardening, optional Armstrong-Frederick backstress and Hill48 r-values), `from_swift` / `from_hollomon`, `library`, `get_material`, `to_sparlab()` |
| `precomp.toolpath` | geometry | `tool_center_surface` (drop cutter), `tool_reach_surface`, `spiral_toolpath`, `Toolpath`, `max_gouge`, `dsif_support_points` / `dsif_support_path`, `rim_pass_path`, `lift_heights`, `lift_at_points`, `swept_ball_top`, `signed_outline_distance` |
| `precomp.metrology` | geometry | `align` (robust point-to-plane ICP), `signed_deviation`, `vertical_deviation`, `metrics`, `part_mask`, `flange_mask`, `wall_mask` |
| `precomp.robot` | toolpath | `CartesianCompliance`, `forces_on_path` |
| `precomp.fea` | geometry, toolpath, materials | `FormingSetup`, `PRESETS`, `build_deck`, `deck_hash`, `make_toolpath`, `run_deck`, `simulate` (content-addressed cache), `simulate_many`, `load_result` / `FormingResult`, `FormingError`; `precomp.fea.support` ([Rim support](#rim-support)) |
| `precomp.compensation` | geometry, metrology (fea, toolpath lazily) | `displacement_adjustment`, `limit_wall_angle`, `smooth_update`, `FEAPredictor`, `SurrogatePredictor`, the `FieldModel` protocol |
| `precomp.ml` | all but robot | learned deviation models, conformal intervals, the training envelope, surrogate compensation ([precomp_ml.md](precomp_ml.md)) |

## Geometry

| `HeightMap` method | Result | Notes |
|--------|--------|-------|
| `interpolate(x, y, masked=True)`, `sample(values, x, y)` | z [m], any field | bilinear; NaN off the grid and (with `masked`) where the stencil touches an invalid node |
| `gradient(sigma=None)` | (dz/dx, dz/dy) | central differences, or Gaussian derivatives at length scale `sigma` [m] |
| `normals(sigma)`, `wall_angle(sigma)` | (ny, nx, 3), rad | unit normal (-z_x, -z_y, 1)/norm |
| `curvature(sigma)` | (H [1/m], K [1/m^2]) | of the graph surface; H > 0 where it bends towards +z |
| `resample(grid)` | HeightMap | nodes off the source are invalid |
| `to_points()`, `from_points(points, grid, max_gap=None)` | (n, 3), HeightMap | scattered -> grid by linear Delaunay interpolation; outside the hull, or farther than `max_gap` from data, invalid |
| `save(path)`, `load(path)` | `.npz` | `z`, `mask`, `grid` and `metadata` (JSON), `format = "precomp.HeightMap/1"` |

The Gaussian derivatives use moment-normalised kernels (the second-derivative
kernel sums to 0), so they are exact for any quadratic whatever its offset from
z = 0; `scipy.ndimage`'s order-2 filter leaks the height into the curvature.

**Part families.** Each is a frozen dataclass with physical parameters,
`height(x, y)`, `heightmap(grid)`, `bounds`, `sample(rng)`,
`max_wall_angle_deg()`, `max_depth()`, `footprint_radius()` and `to_dict()` /
`part_from_dict()` (a `family` key). Every constructor refuses a wall steeper
than `MAX_WALL_ANGLE_DEG = 65` and inconsistent dimensions, naming the
parameters; `sample` redraws until the part is valid.

| Family | Parameters | Construction |
|--------|------------|--------------|
| `truncated_cone` | `top_radius`, `wall_angle_deg`, `depth`, `top_fillet`, `bottom_fillet` | profile of the radial distance |
| `pyramid` | `half_width_x`, `half_width_y`, `wall_angle_deg`, `depth`, `corner_radius`, fillets | profile of the distance to a rounded rectangle |
| `dome` | `opening_radius`, `depth`, `rim_fillet`, `aspect` | spherical cap tangent to a rim fillet (closed form); `aspect` != 1 scales y |
| `elliptic_cone` | `semi_axis_x`, `semi_axis_y`, `wall_angle_deg`, `depth`, fillets | profile of the exact distance to an ellipse |

The cone-like families are a C1 profile g(w) (`FilletedProfile`: a polyline
with circular fillets) of an exact, unit-gradient distance w from the nominal
opening, so every straight wall has exactly the design angle. The nominal
opening is where the wall, extended, meets the sheet plane; the rim fillet
widens the real opening by its tangent length.

## Materials

`Material.to_sparlab()` writes SparLab's `material` block
([configuration.md](configuration.md), `material.plasticity`): elasticity,
density and `plasticity` (`yield_stress`, `hardening_modulus`,
`saturation_stress`, `saturation_rate`, `kinematic_hardening_modulus`). With
r-values (`r0`, `r45`, `r90`) it adds `"yield_criterion": "hill48"` and an
`anisotropy` block (rolling direction along x; `yield_stress` is then along
it); with `af_C` > 0 [Pa], `af_gamma` >= 0 one Chaboche backstress
`{"modulus": af_C, "recovery": af_gamma}`.

Power laws are converted, never passed through: `Material.from_swift` keeps the
initial yield stress K eps0^n exact and fits H, Q and delta to
sigma = K (eps0 + a)^n over plastic strains 0-0.6; `from_hollomon` is Swift
through the yield point. The fit is recorded in `hardening_source`. The library
holds nominal handbook-order values, marked so in `source` - **not certified
data**:

| Material | E [GPa] | yield [MPa] | Hollomon K [MPa], n | r0 / r45 / r90 | fit RMS / max error |
|----------|--------:|------------:|--------------------|----------------|--------------------:|
| AA5754-O | 70.0 | 100 | 420, 0.30 | 0.75 / 0.70 / 0.80 | 1.6 % / 6.2 % |
| AA2024-O | 73.1 | 75 | 330, 0.20 | 0.70 / 0.80 / 0.70 | 3.3 % / 18.6 % |
| AA6061-T6 | 68.9 | 276 | 410, 0.06 | 0.60 / 0.70 / 0.65 | 0.9 % / 4.3 % |
| AA7075-O | 71.7 | 103 | 400, 0.17 | 0.70 / 0.90 / 0.80 | 3.2 % / 20.2 % |
| DC04 | 210 | 170 | 530, 0.22 | 1.90 / 1.50 / 2.20 | 1.6 % / 6.5 % |
| Ti-6Al-4V-annealed | 113.8 | 880 | 1100, 0.04 | - | 0.4 % / 1.7 % |

The largest fit errors sit in the first percent of plastic strain, where one
Voce term cannot follow a Hollomon curve with a small eps0. Kinematic hardening
is zero throughout: handbooks do not give reverse-loading data.

## Tool paths

`tool_center_surface(hm, R)` is the drop cutter of a ball of radius R:
c(x) = max over |d| <= R of [z(x + d) + sqrt(R^2 - |d|^2)], a grey-scale
dilation with a spherical structuring element, exact over the grid nodes (the
ball touches a node and contains none). Between nodes an interpolated surface
can rise above the node samples by about h^2 / (8 R cos^3 theta) at a wall of
angle theta: the gouge a grid of spacing h can hide.

`spiral_toolpath(target, R, step_down, spacing)` is built on z-level loops of
the drop-cutter surface at the tool-centre heights R - k step_down, the last
one step_down / 1000 above the deepest point (raised only as far as needed where
its loop is too short to follow, `deepest_level_raised_m`). Through revolution k
the height falls linearly with arc length from level k-1 to level k, the x-y
position found by bisection where the drop-cutter surface has that height: the
height never increases and the tool touches the surface after the first
revolution. Every level must have one loop (one pocket, no islands), otherwise
`ValueError`; `max_gouge` checks a path. `tool_reach_surface(hm, R)` is the
closing by the ball, what a ball pressing from above can form (the tool path is
unchanged).

`Toolpath` holds the points, a `level` per point (`AIR = -1` in the air), the
arc length `s` and the pseudo-time `t = s / L` (strictly increasing; duplicate
points are dropped). `to_sparlab_csv` writes `t,x,y,z` with 17 significant
digits; `summary(feed)` gives lengths, levels, height range and time at a feed.

## The forming deck

`FormingSetup` holds everything about a run but the commanded shape; `element`
accepts only `"hex8"` and `toolpath_style` only `"spiral"` (the fields remain so
saved setups load). `build_deck(setup, commanded, out_dir, toolpath=None,
target=None)` writes `deck.json` (`mesh`, `material`, `model`, `forming`),
`toolpath.csv` (`t,x,y,z`, t from 0 to 1) and, with `support: "dsif"`,
`support_path.csv` (on the forming tool's t knots; t from 2 to 3 for the rim
pass), all in the content hash; and, for provenance only, `commanded.npz` and
`precomp_deck.json` (setup, grid, tool-path summary, version, support report).

The mesh is `structured_hex` of the blank [-L/2, L/2]^2 x [-t, 0] with
`round(blank_size / element_size)` elements per side and `layers` through the
thickness. The `forming` block follows [forming.md](forming.md) (section 2):
`kinematics`, the spherical `tool` (on the top faces of the free window,
`trajectory: {"file": "toolpath.csv"}`), the steps, each with its own
constraints, and `output`:

| Step | Type | Tools | Constraints | What it does |
|------|------|-------|-------------|--------------|
| `form` | form | tool (+ support) | clamp | runs the trajectory from t = 0 to its last point in contact (the final retract is not simulated) |
| `unload` | release | none | clamp | removes the tool: the springback in the fixture (sparlab_form warns that the clamp holds reactions) |
| `release` | release | none | 3-2-1 | with `release: "321"` only: x, y, z at (-a, -a), y, z at (a, -a), z at (-a, a) on three top nodes of the former clamp, six constraints that remove the rigid-body modes and restrain no springback |

The clamp holds every node within `clamp_margin` of the blank edges; every
constraint is `"mode": "hold"`. `element_formulation` (`"standard"` or
`"incompatible_modes"`) and `thickness_points` (1-7; 0 leaves the solver's 2)
go into the `model` block, and are not written at their defaults. `kinematics`
is `"finite_logarithmic"` (default), `"finite"` or `"small_strain"`. `contact`
sets the tool's `penalty` (s in kappa = s E / h, default 10) and
`tangential_penalty` (default 1); `solver` sets the keys of `forming.newton`
and `friction_tangent`, `solver`, `mean_dilatation` of `forming`. Any other key
is refused when the setup is made, and `build_deck` refuses a path whose tool
would reach the clamp.

**Named setups.** `FormingSetup.preset(name, **overrides)` (`PRESETS`) gives
the springback benchmark's process (40 x 40 x 1 mm AA5754-O blank, 5 mm clamp,
4 mm tool, 1 mm spiral, 3-2-1 release) on a fixed discretisation. A preset is
never changed once a record uses it.

| Preset | Mesh | Penalty | Use |
|--------|------|--:|-----|
| `springback_2mm` | 20 x 20 x 2 standard Hex8 (2 mm, mean dilatation, 2 x 2 x 2 points) | 10 | the original 2 mm study (at commit `7d571e6`); 0.08-0.12 mm RMS from converged (`benchmarks/physics_audit`) |
| `springback_fine` | 48 x 48 x 1 Hex8 with incompatible modes (0.833 mm), 2 x 2 x 5 points | 10 | the physics audit's recommended setup: validation and compensation runs; about 0.02-0.03 mm RMS from converged, 7-11 times the cost |
| `springback_fine_bulk` | as `springback_fine` | 3 | bulk surrogate data: +0.005-0.009 mm RMS, about half the cost |

**Runs and the cache.** `simulate(setup, commanded, work_dir, target=None)`
builds the deck in `work_dir/staging/<uuid>`, hashes it (SHA-256 of the
canonical deck JSON, the trajectory files and `sparlab_form --version`; not the
executable path, timeout or threads) and looks the hash `h` up in
`work_dir/runs/<h[:2]>/<h>/`: the deck files, `output/`, `sparlab_form.log`,
`run.json` (command, exit code, runtime [s]) and, written last, `COMPLETE` or
`FAILED`. A `COMPLETE` entry is returned without running; a run is published by
one atomic rename; a recorded failure is raised again as `FormingError` unless
`retry_failed=True`. Each run is `sparlab_form --config deck.json --output
output --strict-config` with `OMP_NUM_THREADS = setup.threads`; exit status 2
(configuration), 3 (a step stopped, reason from `summary.json`) or 4 (I/O) is a
`FormingError`. `simulate_many(jobs, work_dir, max_workers, executor="process")`
runs (setup, commanded[, target]) jobs on a process pool and returns one
`SimulationOutcome` per job, in order, failures with their reason.

**Result files** ([forming.md](forming.md), section 3): `summary.json`
(required), `<files_stem>_nodes.csv` with `node,X,Y,Z,ux,uy,uz` (required per
completed step), `<files_stem>_elements.csv` with
`element,eq_plastic_strain,von_mises_Pa`, `tool_forces.csv` with
`step,increment,t,tool,cx,cy,cz,fx,fy,fz,active_nodes,max_penetration_m`, and
`mesh.json`, `config.json` and VTK files. `files_stem` (`step_<k>_<name>`) is
taken from the summary, so snapshot files are never mistaken for steps. Every
table is validated and a violation is an error naming the file.
`FormingResult.formed_surface(step, grid)` moves the nodes with reference Z = 0
to their deformed positions, triangulates them as the reference grid and
interpolates linearly at the grid nodes (`raycast_top`); a surface that folds
over in plan is refused. `thickness_map` is the distance between the deformed
top and bottom node of each node column; `forming_forces()` is the force table
with |f| added, the force on the tool per converged increment.

## Metrology

Every function takes metres. `align(points, target, mode)` registers an (n, 3)
point cloud by point-to-plane ICP (residual: the distance along the target
normal to the tangent plane below the point), re-weighted robustly (Tukey's
biweight at 4.685 MAD-sigmas by default, Huber, or none; optional trimming).
`mode` is `rigid`, `translation_z` or `none`; `fixture_mask` restricts the fit,
e.g. to the flange. Directions the data cannot fix get no update and `rank`
says how many were fixed (5 for an axisymmetric part, 3 on a flat flange); the
initial guess `auto` keeps identity unless a principal-axes candidate halves
its median residual.

`signed_deviation(formed, target)` gives, at each target node, the distance
along the target normal to the formed surface (Newton's method on a height
field; for a raw cloud, the intersection with a plane fitted to the nearest
eight points). `vertical_deviation` is z_formed - z_target. `metrics(dev, mask,
tolerance)` returns `rms`, `mae`, `max_abs`, `p95_abs`, `bias`, `std`, `n` and
`pct_within_tol` (percent); `part_mask`, `flange_mask` and `wall_mask` (wall
angle above 5 deg) select regions.

## Compensation

`displacement_adjustment(target, predictor)` iterates
c_(k+1) = c_k - alpha (f(c_k) - t) (Gan and Wagoner, 2004) over any predictor
(commanded HeightMap -> formed HeightMap), with the error measured vertically
or along the target normal. Every update is conditioned the same way: optional
Gaussian smoothing normalised over the adjusted region (`smooth_update`); the
hold region (`hold_mask`, the flange by default) kept at the target; the
command kept at z <= 0, or below `upper_bound` where a support pushing from
below can realise more ([Rim support](#rim-support)); the formability
projection `limit_wall_angle` (the smallest surface above the command whose
central-difference wall angle nowhere exceeds 65 deg; it only raises); and,
with `tool_radius`, the tool's reach (`tool_reach_surface`), so a pit narrower
than the tool is never commanded. `DAResult` keeps the best predicted iterate
(`commanded`, `formed`, `best_iteration`), the unverified next update
(`proposed`) and a history. For a linear springback f(c) = c + G c with
spectral radius rho(G) < 1 and alpha = 1 the error contracts as
e_(k+1) = -G e_k.

Predictors: `FEAPredictor(setup, work_dir, target=None)` simulates through the
cache (`target`: the part a support's fixture is made for).
`SurrogatePredictor(model, setup)` returns commanded + the mean of a
`FieldModel`, any object with `predict_deviation(commanded, setup) -> (mean,
std)`: dz = z_formed - z_commanded [m] on the commanded grid, `std` [m] or
None. `precomp.ml.DeviationSurrogate` implements it.

## Rim support

A tool pressing from above cannot stop the sheet between the clamp and the
first contour from bending down, and a command at z <= 0 cannot ask the rim to
come up. `FormingSetup.support` adds support from below with the rigid tools
`sparlab_form` already has (`precomp.fea.support`):

| `support` | What the deck gets | Steps |
|-----------|--------------------|-------|
| `"none"` (default) | the forming tool only | form, unload, release |
| `"backing_plate"` | `plate`: a still `plane` tool (normal +z) at z = -t on the bottom faces wholly outside the opening (the outline grown by `clearance`) | form (tool + plate), unload, release |
| `"dsif"` | `support`: a ball under the sheet, trajectory `support_path.csv` on the forming tool's pseudo-time | form (tool + support), unload, release |
| `"dsif"`, `rim_pass: true` | as dsif, and the support's rim pass on pseudo-time 2-3 | form, unload, rim_pass (support alone, clamp on), rim_unload, release |

`support_settings` (defaults in `precomp.fea.setup.SUPPORT_SETTINGS`, filled in
by `FormingSetup.resolved_support()`): backing plate `clearance` (1 mm);
`friction`, `penalty` (the tool's); DSIF `radius` (`tool_radius`), `squeeze`
(0: the balls' gap along the normal is (1 - squeeze) t_n), `thickness_law`
(`"sine"`: t_n = t cos(wall angle), or `"initial"`: t), `clearance` (2 mm below
the part in the air), `rim_pass` (false), `rim_inside` / `rim_outside` (2 mm /
1 mm from the outline), `rim_spacing` (1 mm), `rim_max_raise` (2 mm).

* **The plate's opening.** `plate_nodes` lists, as `node_ids` in the
  structured mesh's numbering (`include/sparlab/mesh/StructuredMesh.hpp`), the
  corners of every bottom cell whose four corners lie farther than `clearance`
  outside the outline (`outline_distance`): the opening is up to one element
  wider than asked, never narrower (`precomp_deck.json` records it).
* **The fixture is made for the part.** The opening and the rim-pass band
  follow the outline of `target`, the same for every commanded iterate;
  `build_deck`, `simulate`, `simulate_many` (a third job item), `FEAPredictor`
  and `precomp.ml.generate.sim_job` pass it on (without it: the command's).
* **The DSIF support path** (`dsif_support_points`): one ball centre per point
  of the forming path, on the same t, opposite the tool along the drop-cutter
  normal n (`p - (R1 + t_n + R2) n`, moved towards it by squeeze t_n), never
  above the lift cutter (`lift_heights`) of the *in-process* sheet, below the
  part in the air. Knots are added where it would move more than 0.5 mm between
  two knots in contact, so a straight move round a corner cannot pinch the sheet.
* **The rim pass** (`rim_pass_path`): closed loops at signed distances
  `rim_outside, ..., -rim_inside` from the outline, each point at the
  lift-cutter height of the commanded underside, exact over the grid nodes
  (`lift_at_points`); it reaches the top of the volume its ball sweeps
  (`swept_ball_top`).
* **Commands above the sheet plane.** The forming tool's path is made for the
  part of the command below the plane (`precomp.fea.deck.forming_surface`);
  the plate and the synchronised support push nothing above it. Only the rim
  pass pushes the sheet up on its own, so `command_upper_bound(setup, target)`
  is 0 except, with the rim pass, on the band and flange strip it sweeps
  (`rim_band`), where the command may rise by at most `rim_max_raise` and only
  as far as the rim pass written for it reaches (`rim_pass_reach`), iterated
  to a fixed point. `compensation_masks` adjusts the swept flange strip instead
  of holding it. `build_deck` refuses (`check_command`) a command more than
  `COMMAND_TOLERANCE` (1 um) above the bound, and one above the plane with a
  rim pass but no target.

`support` and `support_settings` are physics fields: a model trained on one
strategy refuses another. On the fine-mesh benchmark
(`benchmarks/springback_fine/README.md`, SparLab simulations, 8 test parts,
vertical RMS over the part) support more than halves the uncompensated
deviation (single-point 0.665 mm, plate 0.300 mm, DSIF 0.273 mm). One FE-DA step
gives 0.555, 0.263 and 0.265 mm; a second step helps only single-point (0.542 mm).

## Robot compliance

`CartesianCompliance(C)` holds the 3 x 3 compliance of the tool-centre point
[m/N] (or (n, 3, 3), one per path point); `deflection(forces)` is C f.
`from_joint_stiffness(J, K)` gives C = J_v K^-1 J_v^T for one pose.
`forces_on_path(path, tool_forces)` interpolates a simulated force history at
the path's pseudo-time. The physics audit (`benchmarks/physics_audit/process`)
uses them to bound the error of a rigid robot.

## How the workflow uses this

The scripts in `python/scripts` run from the repository root
(`OPENBLAS_NUM_THREADS=1 python3 python/scripts/<name>.py ...`); every
simulation goes through the run cache, so each is resumable:

| Script | Arguments | What it does |
|--------|-----------|--------------|
| `springback_fine_benchmark.py` | `--out DIR --work DIR [--workers N] [--strategies ...] [--parts ...] [--alpha A] [--tool-reach] [--fe-runs N] [--executable PATH]` | baselines: the 8 test parts uncompensated and with FE-DA per support strategy |
| `springback_fine_ml.py` | `--out DIR --work DIR --stages N [N ...] [--workers N] [--threads N] [--data-only] [--executable PATH]` | data generation, training and the ML first shot ([precomp_ml.md](precomp_ml.md)) |
| `combined_rim_ml.py` | `{all,one,select,status,report} [--max-time S]` | transfer model, surrogate DA and optimiser on the rim-pass setup ([precomp_ml.md](precomp_ml.md)) |

`python/scripts/common.py` holds their shared helpers (`deviation` by region,
`physics` checks of a run, `split_of`, `versions`). The physics audit uses
`dsif_support_path`, `CartesianCompliance`, `forces_on_path`, `align`,
`HeightMap.from_points` and `precomp.materials`. FE-DA-1 (two runs) in Python:

```python
from precomp.compensation import FEAPredictor, displacement_adjustment
from precomp.fea import FormingSetup
from precomp.fea.support import command_upper_bound, compensation_masks
from precomp.geometry import Grid, TruncatedCone

setup = FormingSetup.preset("springback_fine", support="backing_plate")
target = TruncatedCone(0.007, 40.0, 0.003, 0.001, 0.002).heightmap(
    Grid.centered(setup.meshed_blank_size, 2.5e-4))
hold, adjust = compensation_masks(setup, target)
da = displacement_adjustment(target, FEAPredictor(setup, "work/runs", target=target),
                             iterations=2, hold_mask=hold, adjust_mask=adjust,
                             upper_bound=command_upper_bound(setup, target),
                             tool_radius=setup.tool_radius)
da.rms_history, da.commanded       # [m] per run; the best simulated command
```

## Verification

Each row is a test in `python/tests` against an exact answer; the measured
value is from the test's own configuration.

| Check | Exact answer | Measured | Test |
|-------|--------------|----------|------|
| Drop cutter on a plane | c = z + R | bitwise equal | `test_toolpath` |
| Drop cutter on a 50 deg filleted cone, 120 centres, h = 0.25 mm | distance to the analytic surface = R | min - R = -4.2 um (bound ~6 um); exactly R at 58 node contacts | `test_toolpath` |
| Spiral path | no point below the drop-cutter surface; height non-increasing | gouge 0, height increase 0, contact to 1e-9 m after the first revolution | `test_toolpath` |
| ICP, elliptic cone, 20 000 points, 1e-5 m noise, 2 % outliers, (2, -1.5, 3) deg and (3, -2, 1.5) mm | the inverse transform | 1.1 um RMS point error, 0.0012 deg | `test_metrology` |
| Signed deviation of a cone offset by 0.3 mm along its normal | 0.3 mm | straight wall 0.86 um (bound 1.5 um), fillets 13.6 um (bound 20 um) | `test_metrology` |
| DA on formed = c - 0.2 S c (S a Gaussian smoothing) | contraction <= rho(G) = 0.2 per iteration | 0.194 per iteration, 5e-5 of the initial error after 6 steps | `test_compensation` |
| Wall-angle projection | angle <= 65 deg, never lower | 65.00 deg, raised only | `test_compensation` |
| Curvature of a spherical bowl (central differences) | H = 1/R, K = 1/R^2 | 2e-5 / 3e-5 relative at the pole | `test_geometry` |
| Smoothed derivatives of a quadratic 15 mm below the sheet plane | its exact derivatives | K to 1e-12 relative | `test_geometry` |
| Distance to an ellipse | brute force over 400 001 points | 2.5e-10 m (the sampling) | `test_parts` |
| Formed surface from a synthetic result (stretched, deflected sheet) | the deformed nodes | 1e-15 m | `test_fea_results` |
| DSIF support on a 40 deg cone, 4 mm balls, t = 1 mm | one centre per tool point; gap t cos(theta) along the normal where the sheet is formed; never inside the in-process sheet; below the part in the air | median gap error 0.14 um, never squeezed below it; 0 points inside (to the 1 um the last level keeps above the floor) | `test_toolpath` |
| Backing plate node ids | bottom nodes (z = -t) outside the opening, in the numbering of the mesh the solver writes | all, on `mesh.json` of the run (the real solver's too: positions to 1e-12 m) | `test_fea_support`, `test_integration_sparlab` |
| Backing plate on a cone and on a rounded-rectangle pyramid (14.9 x 16.7 mm), clearance 0, 1, 2.5 mm | exactly the bottom faces whose corners all lie farther than the clearance outside the outline | all; realised clearance >= asked | `test_fea_support` |
| DA with a rim pass on a sag of 1 mm | above the plane only in the swept band and strip, no higher than the rim pass written into the deck reaches (from `support_path.csv` alone), a fixed point of the bound; DA without the bound refused | raised 0.61 / 0.38 mm with a 1.5 / 2 mm ball, 7 um with a 4 mm one; no raised node more than 1 um above the written pass's reach; 0 elsewhere | `test_fea_support` |
| A ball swept along a path; the lift height at a point | a ball's cap; the half cylinder over a straight move; a cut of the underside to the sweep leaves every ball where it was | 1e-15 m; within max_step^2 / (8 R); 1e-15 m | `test_toolpath` |
| A command 10 nm / 10 um above the plane | accepted with every support, with or without a target / refused without a rim pass whatever the target | as expected | `test_fea_support` |
| A compensated command (rim raised to the plane) on a backing plate | `FEAPredictor(..., target=)` forms it on the target's plate; without the target on its own outline's | 384 plate nodes for the target, 404 for the command | `test_fea_support` |

With the real `sparlab_form` (`test_integration_sparlab.py`, skipped without
`build/bin/sparlab_form`), against the contract of [forming.md](forming.md) and
physical sense rather than exact answers (times on one thread):

| Check | Expected | Measured |
|-------|----------|----------|
| Every deck variant precomp writes: Hill48, von Mises and Chaboche materials; the three kinematics; contact and every solver key; clamped only; frictionless; incompatible modes with 5 thickness points; the fine preset with DSIF; backing plate; DSIF with squeeze, with the rim pass, clamped only | accepted by `--strict-config`, the analysis built | 15 of 15; a `contact` key in the tool refused (exit 2, naming it) |
| The test double and `sparlab_form` on one deck | the same files, CSV columns, summary and `mesh.json` keys, mesh and step windows | identical |
| Tiny SPIF: 20 x 20 x 1 mm AA5754-O blank, 8 x 8 x 2 Hex8, 4 mm tool, one spiral revolution to 1 mm ending in contact, unload, 3-2-1 release | depth about the tool's 1 mm; the sheet under the removed tool rises; no reaction after the release; force on the tool upwards, of order 1 kN | depth 0.937 / 0.913 / 0.953 mm after form / unload / release; rise 25-45 um; release reaction 2e-13 N; fz >= 0 at all 51 increments, peak 1 070 N; 71 increments, 328 iterations, about 3 s |
| A small cone: 80 mm blank, 20 x 20 x 1 Hex8, spiral to 4 mm | completes, depth below twice the target's | completes, 270 increments, about 75 s |
| Tiny cone (24 mm blank, 12 x 12 x 2 Hex8, 3 mm tool, 1 mm deep) on a backing plate, 0.5 mm clearance | the plate pushed down in "form" only, never through; no reaction after the release | plate in contact 33 of 40 increments, fz -2.2 kN to 0, penetration 0.27 um; release reaction 4.5e-13 N; about 13 s; the listed ids are the intended nodes of the solver's mesh |
| The same cone with a DSIF support and the rim pass | tool pushed up, support down; the rim pass pushes up in its own step; no reaction after the release | tool fz 0 to 589 N, support -544 N to 0 (steps 1 and 3), penetration <= 0.42 um; release reaction 5e-13 N; about 25 s |

## Limitations

* **The C++ contract.** The deck and the result files follow
  [forming.md](forming.md) (sections 2 and 3), checked by `--strict-config`
  on every run and by `test_integration_sparlab.py` against the test double
  (`python/tests/fake_sparlab_form.py`); a change of the contract is a change of
  `precomp.fea.deck.forming_block`, `precomp.fea.results` and the double.
* **Forming model.** One spherical tool and at most one support; Hex8 only; the
  final retract is not simulated. The node-to-surface contact resolves a tool
  poorly on elements not much smaller than it ([forming.md](forming.md),
  section 6) and Hex8 sheets with few layers are stiff in bending, so the
  numbers of a coarse deck are indicative.
* **Materials.** Nominal, not certified; Hill48 with r < 1 underestimates the
  equibiaxial yield stress; power laws become linear + Voce hardening.
* **Geometry and tool paths.** Four families, height fields only, centred on the
  blank, no sharp-cornered pyramids. Spiral paths only, one loop per level (one
  pocket, no islands), the rim pass too; the gouge hidden between grid nodes is
  about h^2 / (8 R cos^3 theta); the flat floor is not swept.
* **Support.** The DSIF gap follows the sine law, not the sheet's actual
  thickness: on the fine benchmark it is 0.78 mm where the model's wall is
  0.96 mm thick (medians), so the support squeezes the wall wherever it reaches
  its opposite position. The support's deflection is not modelled; the plate is
  flat, at the sheet plane. Upward commands exist only where the rim pass
  reaches them.
* **Metrology.** Clouds must be height fields of the tool side, in metres (no
  file reader). A flat flange fixes only z and two tilts; an axisymmetric
  part's rotation about its axis is never observable. The cloud deviation fits
  local planes (biased by about kappa r^2 / 4, r the radius of eight
  neighbours); gridding a dense cloud first (`HeightMap.from_points`) is more
  accurate.
* **Compensation.** The formability projection only raises: a target steeper
  than the limit is formed shallower there. Nodes without predicted data get no
  update.
* **Robot and thickness.** A constant compliance (or one per point) is
  first-order, with forces interpolated in pseudo-time. `thickness_map` is the
  straight top-to-bottom distance of a node column.
