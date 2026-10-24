# Incremental forming: moving rigid tools, step sequences, springback

`sparlab_form` runs the forming analysis of a deck's `forming` block: a
sheet (or any continuum model) formed by rigid tools that travel along
tabulated trajectories - the hemispherical stylus of single-point incremental
forming (SPIF), two opposed tools (DSIF), a backing plate, a flat punch - and
then released from its tools and clamps onto statically determinate
supports, where it springs back. It is the finite-element engine of a
springback pre-compensation loop: every result file it writes is part of an
output contract other tools read (below).

The library API is `include/sparlab/fem/Forming.hpp` (the analysis) and
`include/sparlab/fem/RigidTool.hpp` (trajectories, tools, contact); the
header comments state the formulation in full, and this page summarises it.

## 1. Formulation

### 1.1 State and steps

The analysis carries one state from step to step: the displacement `u` from
the reference configuration, the internal variables of every elastoplastic
integration point, the friction history of every tool's contact nodes and
the pseudo-time `t`. The elements, the kinematics (`finite`, total
Lagrangian; or `small_strain`), the J2 plasticity with isotropic and
kinematic hardening and the mean dilatation of the Hex8 are those of the
non-linear system (`NonlinearSystem`, `docs/formulation.md`):
every Newton iteration returns the points from their committed state, and
only a converged increment commits - a cut increment leaves no trace. No
load case acts; the tools and the prescribed displacements drive the model.

A step is `form` (the listed tools follow their trajectories over the
step's pseudo-time window) or `release` (the start imbalance is ramped out,
tools removed or a subset kept). Each step has its own Dirichlet partition:
a list of displacement constraints, each `hold` (the DOF stays where the
step found it) or `absolute` (moved linearly over the step from there to the
given value). A step without constraints takes the model's
`boundary_conditions` in `absolute` mode (a clamp with value 0 holds the
reference position). Every step's constraints must suppress every rigid-body
motion of the model; this is checked before the run and a violation names
the step and the free motions (e.g. "rotation about x").

### 1.2 Equilibrium of a step

With `s` in [0, 1] the fraction of the step, the prescribed DOFs at their
values of `s` and the free DOFs `f`:

```
R_f(u, t) - (1 - s) R_0f = 0,        R = f_int(u) - f_c(u, t)
```

`f_c` are the penalty contact forces of the step's active tools at pseudo-time
`t(s)` and `R_0 = R(u_0, t_0)` is the imbalance of the start state on the
step's free DOFs. A step that keeps the previous step's partition and tools
finds `R_0f` within tolerance of zero; after a change it holds the reactions
of the released constraints and the forces of the removed tools, which the
step ramps out linearly - the springback of a release. The ramped imbalance
is a dead load: fixed in space while the part moves.

A tool active in two consecutive steps must start the second where it ended
the first. A window that starts later than the previous step ended is
refused (before the run, naming the steps, the tool and the distance) if the
tool's path moves in between: the tool would jump over that part of its
path, and the penetration where it lands would enter `R_0f` and be ramped
out as if it were a load, loosening the tolerance of every later step
through the reference force. A tool held still over the gap, or taken away
by a step in between, is accepted. A restart checks the same against the
tools its state had active (`AnalysisState::tools_active`). A tool that
becomes active already in contact has the force of its penetration ramped
in over the step at its positions rather than reached by travel; the step
warns about it.

### 1.3 Penalty contact in the current configuration

For slave node `j` (the nodes of the boundary faces a tool's `surface`
selects) at its current position `x = X_j + u_j` and the tool's reference
point `c(t)`:

| tool | gap `g` | normal `n` | `dn/dx` |
|------|---------|------------|---------|
| sphere, radius `R` | `|x - c| - R` | `(x - c)/|x - c|` | `(I - n n^T)/d` |
| cylinder, axis `a` | `|r_perp| - R`, `r_perp = (I - a a^T)(x - c)` | `r_perp/|r_perp|` | `(I - a a^T - n n^T)/d` |
| plane, normal `n` | `n . (x - c)` | `n` (out of the tool) | 0 |

Where `g < 0` the node takes the normal force `f_N = -kappa_j A_j g n`,
with `A_j` its tributary area on the selected reference faces (`sum int N_j dA`,
times the thickness in 2-D) and `kappa_j = s E_j / h_j` [Pa/m] - `s` the
penalty scale (default 10), `E_j` the stiffest adjacent Young's modulus,
`h_j = sqrt(A_j)` (3-D) or `A_j / t` (2-D). The node's spring `kappa_j A_j`
is then about `s E h`, `s` times the stiffness of an element of its size,
and the penetration a contact pressure `p` leaves is about `p h / (s E)`
(4 um for 300 MPa on 1 mm aluminium elements at `s = 10`). The force enters
the residual as `-f_N`, with the exact node-diagonal tangent
`kappa A [n n^T + g dn/dx]`.

A node closer than half the radius to a sphere's centre (or a cylinder's
axis) means the increment drove the tool through the surface: the increment
is cut. At the start of a step (a path that starts inside the part) it stops
the analysis: the step is recorded, not completed, with the state it started
from, and the reason.

**Friction** is regularised Coulomb friction by an elastic-slip return map
(backward Euler over the increment), with a history per slave node and tool:
the tangential force `F_T` and the relative position `q = x - c` at the last
converged increment. The committed force is transported onto the current
tangent plane (its normal part removed, its magnitude kept), the slip
increment is `delta = (I - n n^T)(x - c(t) - q)`, the trial force
`F_tr = F_hat - kappa_T A delta` (`kappa_T = rho_T kappa`, the tangential
ratio `rho_T` default 1), and the node sticks where `|F_tr| <= mu p_N`,
slipping otherwise with `F_T = mu p_N F_tr/|F_tr|`. The tangent is the exact
derivative of the node force at fixed history (transport, projection, the
normal force in the slip cone): it is not symmetric, so the tangent is
factorised by LU. `friction_tangent: "symmetric"` replaces its friction part
by a positive semi-definite approximation, the friction force's stiffness in
the tangent plane alone - `kappa_T A P` in stick, `(mu p_N / |F_tr|)
kappa_T A (P - e e^T)` in slip (`P = I - n n^T`, `e = F_tr / |F_tr|`) -
without the transport and curvature terms and the slip force's dependence
on the gap, `mu kappa A e n^T`. The tangent then takes a Cholesky
factorisation, but Newton converges only linearly where nodes slip: ironing
an elastic block with a flat punch took 48 iterations instead of 31, an
elastoplastic block with a sphere 3 063 instead of 392. (The symmetric part
of the exact tangent is no substitute: in the plane of `n` and `e` a
slipping node's block is `kappa A [[1, mu/2], [mu/2, 0]]`, indefinite, and
Newton's method stalled with it where contact began.) A node that leaves
contact loses its history. The history is keyed by mesh node, so it
survives a change of partition.

**Search.** Before each increment (from the converged `u_0` at `t_0` to `t_1`)
a node is a candidate for a tool if `x_j(u_0)` lies within `R + m` of the
segment `[c(t_0), c(t_1)]` (increments never cross a trajectory knot, so the
tool moves on that segment), with the margin `m = 2 x (tool travel) + 2 x
(largest nodal displacement change of the last increment) + h_min`. Every
evaluation checks that no slave node has moved more than `m` from
`x_j(u_0)`; if one has, that evaluation takes every slave node. A node left
out therefore provably has `|x_j - c(t)| > R` for every `t` of the
increment: the filter never changes the result.

### 1.4 Increments and Newton's method

A `form` step advances the pseudo-time over its window in increments such
that no active tool travels more than `max_tool_travel` (default half the
smallest slave-node size `h_j`) in one; trajectory knots are reached exactly,
so no corner of a path is cut. A `release` step runs `s` from 0 to 1 in
`increments` equal parts (default 10). A failing increment is halved, up to
`max_cuts` times in a row; one that converges in at most 4 iterations lets
the next grow by 1.5, up to the step's cap.

Newton's method starts each increment from the converged state with the
prescribed DOFs at their new values and the tools at their new positions -
and, when the prescribed DOFs move as a rigid motion of the body (always so
for statically determinate supports, which cannot deform it), with the free
DOFs moved by that motion too, linearised about the current configuration:
moving the three nodes of 3-2-1 supports alone distorts the elements around
them, and on a plastic part the first iterates yield spuriously (a released
strip whose supports were translated by 3.7 mm took 40 increments, 765
iterations and 2 cuts without it, 10 and 41 with it, as held in place) -
with the consistent tangent and the energy line search of the non-linear
static analysis (full step when `|g(1)| <= 0.8 |g(0)|`, else regula falsi;
an inverted element or a node through a tool halves the step). The
evaluation of the full step, tangent included, is kept as the next
iteration's (and the accepted state's), so an iteration costs one element
evaluation when the line search accepts the full step. A node that the
correction would drive into a tool from outside has no penalty stiffness in
the tangent, and the step would overshoot it deep into the tool - after
which the line search creeps towards the contact, iteration after iteration.
Such nodes join the active set with the penalty law extended linearly to
their current gap, and the correction is solved again (a semismooth Newton
step on the predicted active set); on the smoke case below this removed
every increment cut and a third of the iterations.

An increment converges when both `|R_f - (1 - s) R_0f| <= eps_R F_ref` and the
correction is below `eps_u` times the increment of `u` - or the residual is
at its round-off floor - and the residual of the accepted state is checked
again before its history is committed. The **reference force** `F_ref` is the
largest of the ramped imbalance, the reactions, the contact forces and every
such value met at a converged state of the analysis so far: at the end of a
release onto determinate supports the external forces and the reactions
vanish, and a scale made of them alone would leave the tolerance at zero
(the defect the non-linear static driver has, which cannot converge to a
load-free residually stressed state). The round-off floor adds the
contact's own, `8 eps kappa A (|x| + |c|)`: a gap is a difference of
positions, each exact to its rounding. For the same reason a displacement is
resolved only to the rounding of the position `X + u` it defines: the
correction limit is no finer than `64 eps (|u| + |X|)`, and a state whose
correction is at that limit is accepted with a residual up to
`64 eps || |K| (|X| + |u|) ||` - even far above the forces of the state (a
stress-free body moved rigidly by its supports, a tool grazing the surface
from the reference state, whose reference force is itself round-off). That
floor is never accepted alone, since a runaway state inflates it.

### 1.5 Linear algebra

The free-free tangent's sparsity pattern is analysed once per Dirichlet
partition; each iteration scatters the assembled values into it through a
cached index map and refactorises numerically. With SuiteSparse
(`-DSPARLAB_WITH_CHOLMOD=AUTO`, the default, finds it; `ON` requires it,
`OFF` never uses it): CHOLMOD supernodal Cholesky for a symmetric tangent,
Eigen's `SimplicialLDLT` if that fails (the tangent is not positive
definite), and UMFPACK LU for a non-symmetric one (friction) or when
`LDL^T` meets a vanishing pivot. Without it, Eigen's `SimplicialLDLT` and
`SparseLU`. `"solver": "eigen"` in the deck selects Eigen's factorisations
in a SuiteSparse build (for comparisons). The summary records how often
each factorisation ran, how often one failed and fell back to the next, and
the time each phase took.

## 2. Configuration

The `forming` block of a deck (`sparlab_form` needs one). With it,
`boundary_conditions` may be omitted when every step lists its own
constraints (there are no load cases: the tools and the prescribed
displacements drive the model). Relative trajectory files resolve
against the deck's directory; every key is read strictly (`--strict-config`
refuses unknown ones).

```jsonc
"forming": {
  "kinematics": "finite",                  // or "finite_logarithmic", "small_strain"
  "mean_dilatation": "auto",               // "auto" | "all" | "none" | true | false
  "friction_tangent": "exact",             // or "symmetric"
  "solver": "auto",                        // or "eigen"
  "tools": [
    { "name": "tool", "shape": "sphere", "radius": 0.005,
      "surface": { "box": { "zmin": 0.0 } },
      "friction": 0.05, "penalty": 10.0, "tangential_penalty": 1.0,
      "trajectory": { "file": "toolpath.csv" } }   // or { "times": [...], "points": [[x,y,z], ...] }
  ],
  "steps": [
    { "name": "form", "type": "form", "tools": ["tool"], "time": [0.0, 3.1],
      "max_tool_travel": 5.0e-4,
      "boundary_conditions": [ { "name": "clamp", "fix": ["x","y","z"], "region": {...} } ] },
    { "name": "retract", "type": "form", "tools": ["tool"], "time": [3.1, 3.2],
      "boundary_conditions": [ { "name": "clamp", "fix": ["x","y","z"], "region": {...} } ] },
    { "name": "unclamp", "type": "release", "increments": 10,
      "boundary_conditions": [ { "name": "A", "fix": ["x","y","z"], "mode": "hold", "region": {...} },
                               { "name": "B", "fix": ["y","z"], "region": {...} },
                               { "name": "C", "fix": ["z"], "region": {...} } ] }
  ],
  "newton": { "max_iterations": 30, "residual_tolerance": 1e-6, "displacement_tolerance": 1e-6,
              "line_search": true, "max_cuts": 10, "max_increments": 100000 },
  "output": { "vtk": true, "snapshots": "steps" }   // "steps" | "none" | an increment stride
}
```

| Key | Default | Meaning |
|-----|---------|---------|
| `kinematics` | `"finite"` | `"finite"` (total Lagrangian in the Green-Lagrange strain: large rotation, small strain), `"finite_logarithmic"` (total Lagrangian in the logarithmic strain: large-strain plasticity, as in `nonlinear`, `docs/configuration.md`) or `"small_strain"`. `sparlab_form` models elastoplastic sheets: with `"finite"` or `"finite_logarithmic"` the material of every element (the top-level material where no material region replaces it, and each region's) needs a `plasticity` block, and a deck with an element of elastic material is refused when the analysis is built; only `"small_strain"` takes an elastic material (linear elasticity). The elastic large-strain laws and their key `material_model` were removed (`--strict-config` refuses the key) |
| `mean_dilatation` | `"auto"` | B-bar of the elastoplastic standard Hex8 |
| `friction_tangent` | `"exact"` | `"exact"` (consistent, LU) or `"symmetric"` (a positive semi-definite friction stiffness: Cholesky, linear convergence where nodes slip) |
| `solver` | `"auto"` | `"auto"`: SuiteSparse when built in; `"eigen"`: Eigen's factorisations |
| `tools[].name` | `"tool<i>"` | unique; letters, digits, `_`, `-` and `.` only (a field of `tool_forces.csv`) |
| `tools[].shape` | `"sphere"` | `"sphere"`, `"plane"` or `"cylinder"` |
| `tools[].radius` | required (sphere, cylinder) | `> 0` [m] |
| `tools[].normal` | required (plane) | the plane's normal, out of the tool towards the body |
| `tools[].axis` | `[0, 0, 1]` (cylinder) | the cylinder's axis (3-D; in 2-D it is z) |
| `tools[].surface` | required | a region: the boundary faces whose nodes all lie in it are the slave surface |
| `tools[].friction` | 0 | Coulomb coefficient, `>= 0` |
| `tools[].penalty` | 10 | penalty scale `s`, `kappa = s E / h` |
| `tools[].tangential_penalty` | 1 | `kappa_T / kappa` |
| `tools[].trajectory` | required | `{ "file": "path.csv" }` (header `t,x,y,z`, s and m; blank lines and `#` comments allowed) or `{ "times": [...], "points": [[...], ...] }`; at least two knots, times strictly increasing; the sphere's centre, a point of the plane, a point of the cylinder's axis; constant outside its span |
| `steps[].name` | `"step<k>"` | unique; part of the result file names |
| `steps[].type` | `"form"` | `"form"` or `"release"` |
| `steps[].tools` | `[]` | active tools; a release may keep only tools the step before it lists |
| `steps[].time` | see below | `[t_begin, t_end]` [s], not before the previous step's end; later only if the tools active in both steps do not move in between |
| `steps[].max_tool_travel` | half the smallest slave-node size | largest travel of an active tool in one increment [m] |
| `steps[].increments` | 1 (form), 10 (release) | the window in this many equal increments at the start (and at most) |
| `steps[].boundary_conditions` | the model's, `absolute` | as the deck's, plus `"mode"`: `"hold"` (default: the DOFs stay where the step finds them; a non-zero `value` is refused) or `"absolute"` (`value` the absolute end value) |
| `newton.*` | see above | Newton settings |
| `output.vtk` | true | per-step VTK files (`--no-vtk` overrides) |
| `output.snapshots` | `"steps"` | `"steps"`: per-step files; `"none"`: only summary, mesh and tool forces; an integer N: per-step files plus a snapshot every N increments |

A step's default window: for a `form` step with tools, the union of their
trajectories' spans, starting no earlier than the previous step ended; for a
`release` step (or a form step without tools) the unit interval after the
previous step. So a path that ends with the tool's retraction (here from
t = 3.1 to 3.2 s) is split by giving the form step the window up to it - left
to its default, the form step would take the whole path, and the retract
window would start before it ended. A step without `boundary_conditions`
takes the deck's top-level ones (in `absolute` mode), not the previous
step's: a step that keeps a clamp lists it again (as above), and a deck that
gives every step its own may omit the top-level ones.

The deck is refused - naming the key - for an unknown tool in a step, a
release keeping a tool the previous step does not list, a tool name with
other characters than letters, digits, `_`, `-` and `.`, a non-zero `value`
on a held step constraint, a non-positive radius, negative friction, a time
window that runs backwards, a trajectory with fewer than two knots or times
that do not increase, and (when the analysis is built) a surface that
selects no face, constraints that leave a rigid-body motion free, or a tool
that would jump between two steps.

## 3. Output contract

`sparlab_form --config deck.json --output dir [--threads n] [--verbosity lvl]
[--strict-config] [--no-vtk] [--version]` runs no linear static solve: it
builds the model (validating the mesh and the material), runs the steps and
writes, into `dir`:

| File | Content |
|------|---------|
| `summary.json` | `case`, `sparlab_version` (`"<version> (<git revision>)"`), `completed`, `termination`, `runtime_s`, `timing` (seconds per phase: `element_tangent_s`, `element_residual_s`, `contact_s`, `factorisation_s` with `factor_<kind>_s` per factorisation, `solve_s`, `output_s` (the step files written during the run), `total_s`; `increments`, `iterations`, `cuts`, `linear_solver` with the count of each factorisation used, `failed_factorisations` with the count of each attempt that failed - a Cholesky of a tangent that is not positive definite, then `LDL^T`; an `LDL^T` with a vanishing pivot, then LU; a singular LU, which fails the iteration - empty if none, `suitesparse`), `analysis` (kinematics, tolerances), `steps` (per step: `name`, `type`, `completed`, `increments`, `iterations`, `cuts`, `max_plastic_strain`, `reaction_norm_N`, `warnings`, and `termination`, `files_stem`, `t_begin_s`, `t_end_s`, `tools`, `constrained_dofs`, `start_imbalance_N`, `reference_force_N`, `max_displacement_change_m`, `max_displacement_m`), `tools` (per tool: shape, radius, friction, penalty, trajectory span and length, peak force and its time, largest contact node count and penetration), `mesh` (element type, dim, nodes, elements, DOFs, bounding box), `warnings`, `files`, `provenance` |
| `config.json` | the deck, verbatim |
| `mesh.json` | nodes, connectivity (`ResultWriter::write_mesh`) |
| `step_<k>_<s>_nodes.csv` | `node,X,Y,Z,ux,uy,uz`: reference coordinates and the displacement at the end of step `k` (1-based) named `s` [m]; Z and uz are 0 on a 2-D model |
| `step_<k>_<s>_elements.csv` | `element,eq_plastic_strain,von_mises_Pa`: the largest equivalent plastic strain of the element's integration points, and the von Mises stress of its point-averaged Cauchy stress |
| `step_<k>_<s>.vtk` | the same fields for ParaView (unless `--no-vtk`) |
| `step_<k>_<s>_inc_<i>_nodes.csv`, `.vtk` | snapshots every `output.snapshots` increments (step ends excluded) |
| `tool_forces.csv` | `step,increment,t,tool,cx,cy,cz,fx,fy,fz,active_nodes,max_penetration_m`: per converged increment and active tool, the tool's reference point [m] and the force the body exerts **on the tool** [N] |

`s` is the step name with every character other than letters, digits, `-`
and `_` replaced by `_`. Only completed steps have files. They are written
as the run goes: `config.json` and `mesh.json` before the first step (an
output directory that cannot be created fails the run before it starts),
each step's files and `tool_forces.csv` (every increment so far) as the step
ends, `summary.json` last - so a run that is killed keeps the steps it
completed, and a directory without `summary.json` holds an unfinished run.
The exit status is
0 when every step completed and 3 when one stopped (the reason is in
`summary.json` and on stderr; the steps before it are written); 2 for a
configuration error, 4 for an I/O error. `--version` prints one line,
`sparlab <version> (<git revision>)`; the revision is recorded at every
build, with `-dirty` for uncommitted changes.

## 4. Verification

`tests/test_forming.cpp` (`[forming]`):

| Check | Measured |
|-------|----------|
| Trajectory: interpolation, clamping, velocity, monotone and random lookups, CSV parsing and its errors (line numbers) | exact |
| Contact tangent vs central differences (step 1e-9 m) of the contact residual at fixed history (sphere, plane, cylinder; frictionless, stick, slip, new contact; 13 to 215 nodes in contact; a 2-D circle in slip) | largest error relative to the largest tangent entry `7.2e-10` (sphere `7.0e-10`, plane `5.4e-10`, cylinder `7.2e-10`, 2-D `1.7e-10`; tolerance `1e-6`) |
| Flat punch on an elastic column (one Hex8 in section): force vs `E A delta / H / (1 + E/(kappa H))`, the penalty in series | `4.5e-12`, `1.9e-11`, `3.6e-10` at `s` = 10, 100, 1000 (tolerance `1e-9`); against rigid contact `1.96e-2`, `2.0e-3`, `2.0e-4` - exactly `1/s` |
| Ironing, `mu = 0.2`: a flat punch pressed into an elastic block and dragged 10 mm at 30 degrees to x, a sphere (R = 20 mm) dragged along x | in steady sliding every contact node slips (231 and 9 nodes); flat punch: the friction resultant lies along the drag to `7e-16` and `\|F_T\| = mu F_z` to `2e-16`, every node's friction force along the drag to `3e-15` (tested to `1e-6`: x and y in the drag's proportion, which a friction force not following the slip would miss); sphere: `-F_x / F_z = 0.19986`, `mu` to `6.9e-4` (tested to `3e-3`), its contact normals tilted. (The friction load being `mu` times the normal load is an identity of the return map once every node slips, not a check.) |
| The symmetric friction tangent (`friction_tangent: "symmetric"`), at the tangent test's states and on the flat-punch ironing | the same forces; symmetric node blocks, positive semi-definite for the plane in stick and slip; the ironing completes without a cut (48 iterations, the exact tangent 31) at the same tool force to `1e-8` |
| Double-sided pinch: two spheres on the two faces of a clamped sheet, paths mirrored about its mid-plane, pressed and moved together with friction | equal and opposite normal forces and equal friction forces to `1e-6`; the reactions' resultant equals the total force on the tools to `1e-6` |
| Step windows with a gap: a sphere active in both steps, its path moving in between (and a restart from the first step's state) | refused, naming the steps and the 2.4 mm jump; held over the gap, or taken away by a release in between: accepted; a tool that becomes active in contact is warned about |
| Release of a stress-free block onto 3-2-1 supports | displacement exactly 0; supports moved by a translation: the body follows it to 1e-12, one iteration per increment |
| Release of a plastically bent strip (finite kinematics, restarted from the formed state) with the 3-2-1 values moved by a translation, and by a 0.03 rad rotation with it | the result is the held one moved rigidly: largest deviation `6.3e-18` and `1.1e-17 m` (tested to `1e-14 m`); plastic history unchanged; 41 iterations and no cut, as held (without the rigid predictor 765 iterations and 2 cuts, and 175) |
| Springback of an elastic-perfectly plastic beam (L = 8 h) bent by end displacements to `3 k_y` (one standard Hex8 layer through its thickness, small strain), released onto 3-2-1: curvature change of the central half vs the exact elastic unloading `M(3 k_y)/(EI) = 1.4444 k_y` | `1.4729`, `1.4551`, `1.4483 k_y` with 4, 8, 16 Hex8 through the depth: errors `2.0e-2`, `7.4e-3`, `2.7e-3` (order 1.4 then 1.5; tested below `5e-3` on the finest); support reactions below `1e-6` of the reference force |
| A sphere grazing the surface exactly at the end of the first increment, from the reference state (a round-off reference force) | converges without a cut, the first increment in at most 3 iterations |
| SPIF smoke case (section 5) on a coarse 12 x 12 x 2 mesh with 12 chords a contour (86 increments, 785 iterations; about 17 s on one thread) | completes; the centre sinks 1.70 mm (tool tip at 2 mm), the release moves the part by up to 0.16 mm; tool force positive in z at every increment in contact. The 20 x 20 x 2 deck sinks 1.65 mm and springs back 0.18 mm. In both the `retract` step exercises nothing: the tool has no node in contact by the end of the second contour (on the 20 x 20 deck, at the last two increments of `form`), and the retract moves the part by about `1e-18 m` - a 5 mm tool on 2 to 3.3 mm elements touches the deformed surface at one or two nodes and can pass between them |
| The `forming` block: every key, trajectory file, refusals; the result files and summary keys of the contract | every key read in strict mode; every refusal names its key; every file with its exact header and row count |

Every check also passes in a build without SuiteSparse. `ctest` runs each
case with one OpenMP thread (`tests/CMakeLists.txt`): under `ctest -j4`
several test processes share the cores, and OpenMP threads that spin while
they wait then take the CPU from the ones with work - on the shared machine
(load average 10 to 15) the fourteen `[forming]` cases took 78 s with a
thread per core each and 30 s with one: the ironing case 78 s and 30 s, the
smoke case 68 s and 26 s, the springback case 38 s and 9 s, the rigid
release 22 s and 4 s.

## 5. Cost

Measured with `sparlab_form --threads 4` on the development machine: four
cores shared with other jobs (load average 2 to 11 during the runs, so the
element times in particular vary by a factor of two between runs), GCC 13
`-O3`, SuiteSparse 7 over the **reference (netlib) BLAS**, which is slow for
CHOLMOD's supernodes and UMFPACK's fronts - an optimised BLAS would shorten
the factorisations markedly. `"solver": "eigen"` runs the same build with
Eigen's factorisations; the iteration counts are identical, and the results
agree to `1e-17 m`.

| Case | DOFs | Path | Increments | Newton iterations | SuiteSparse | Eigen |
|------|------|------|------------|-------------------|-------------|-------|
| Smoke (`configs/forming/spif_smoke.json`): 40 x 40 x 1 mm, 20 x 20 x 2 Hex8, 5 mm tool, 2 contours to 2 mm, retract, unclamp | 3 969 | 100 mm | 137 | 1 274 | **75 s** | 122 s |
| Cone, first quarter contour: 60 x 60 x 1 mm, 60 x 60 x 2 Hex8, plunge and a quarter of the 1 mm contour, unclamp | 33 489 | 23.6 mm | 61 | 693 | **939 s** | 1 289 s |
| Cone, full (`spif_cone_60.json`, in the history at commit `7d571e6`): ten contours with a 1 mm step-down to 10 mm, retract, unclamp | 33 489 | 622 mm | 1 470 (from the path) | ~18 700 (12.7 per increment) | **~7 h** (extrapolated) | ~9.6 h |

Where the time goes (seconds):

| Phase | Smoke, SuiteSparse | Smoke, Eigen | Cone quarter, SuiteSparse | Cone quarter, Eigen |
|-------|--------------------|--------------|---------------------------|---------------------|
| factorisation (total) | 41.1 (55 %) | 57.7 | 686.3 (73 %) | 1 041.9 (81 %) |
| - LU, friction (UMFPACK / SparseLU) | 35.4 (1 248 x 28 ms) | 48.3 | 582.5 (723 x 0.81 s) | 805.1 (723 x 1.11 s) |
| - Cholesky (CHOLMOD / SimplicialLDLT) | 4.7 (335 x 14 ms) | 8.0 | 93.2 (178 x 0.52 s) | 231.6 (184 x 1.26 s) |
| element tangent (evaluation and assembly) | 25.6 | 54.3 | 181.7 | 192.9 |
| element residual (line search, acceptance) | 2.5 | 4.6 | 19.4 | 21.4 |
| solve | 4.2 | 2.2 | 38.8 | 19.6 |
| contact (evaluation, search, anticipation) | 0.1 | 0.1 | 0.5 | 0.5 |
| **total** | **75.4** | **122.0** | **939.1** | **1 288.7** |

A build without SuiteSparse (`-DSPARLAB_WITH_CHOLMOD=OFF`) runs the Eigen
path: the smoke case took 170 s, 70 s of it factorisation, under a load
average of 11. Re-run after the fixes of the review (the smoke deck's
supports hold, so the rigid predictor does not act on it), the smoke case
took the same 137 increments and 1 274 iterations, in 125 s at a load
average of 3 to 7 (factorisation 57 s, element tangent 50 s).

The full cone was not run to the end: at about 17 s per increment it takes
about 7 hours. The estimate multiplies the 1 470 increments its path needs
(0.5 mm travel, 48 chords per circle, every knot reached; the same count
predicts the quarter run's 51 exactly) by the quarter run's 12.7 iterations
per increment and 1.36 s per iteration (1.86 s with Eigen); the deeper
contours, with more of the sheet plastic, may need more.

What the numbers say:

* **The factorisation dominates** - 55 % on 4 000 DOFs, 73 % on 33 000 -
  and friction makes the tangent non-symmetric, so every iteration with a
  slipping node takes an LU (UMFPACK, 1.6 to 2 times the cost of CHOLMOD
  here) rather than a Cholesky. SuiteSparse is worth 1.4 to 1.6 times overall on this machine.
* **Newton needs 9 to 13 iterations per increment** on these meshes: the
  first iteration meets the tool's new penetration (a residual of order
  `kappa A` times the travel), and the plastic zone under the tool is then
  redistributed over several iterations; on the coarse smoke mesh the tool
  carries its load on one or two nodes. Smaller tool travel means more,
  cheaper increments, not less work.
* The contact itself (evaluation, search, anticipation) is negligible.
* **The next gains** are, in order: an optimised BLAS under SuiteSparse; a
  modified Newton method that keeps a factorisation over several iterations
  (an iteration then costs an element evaluation, 0.26 s with the tangent at
  33 000 DOFs, and a back-substitution of about 50 ms, instead of adding a
  factorisation of 0.5 to 0.8 s);
  a symmetric factorisation with the friction's non-symmetric part applied
  iteratively; and, for full-size parts, explicit dynamics with mass
  scaling, which is how incremental forming is usually simulated at scale
  (an explicit step existed and was removed in the cleanup, section 7).

## 6. Limitations

* **Penalty contact.** The contact is enforced approximately: a node
  penetrates the tool by about `p h / (s E)`. Raising the penalty scale
  reduces it (the force error of the punch test falls as 1/s) but stiffens the
  system; there is no augmented-Lagrangian correction. The contact is
  node-to-rigid-surface: the slave surface is sampled at its nodes, so a tool
  much smaller than the elements (or a coarse mesh under a small tool) is
  resolved poorly - on the smoke case a 5 mm tool on 2 mm elements carries
  its load on one or two nodes, which is what makes its increments converge
  slowly.
* **Plasticity at large strain.** With `finite` kinematics the elastoplastic
  law is additive in the Green-Lagrange strain, meant for small elastic
  strains at large rotation; incremental forming reaches equivalent plastic
  strains of order one, beyond which its stresses are approximate (the run
  warns beyond a strain of 0.05). `finite_logarithmic` kinematics returns in
  the logarithmic strain, which holds at large strain (no warning; exact for
  coaxial stretches, `docs/formulation.md` 7d); the decks of `precomp` use it.
* **Implicit and quasi-static.** No inertia, no strain rate. Every increment
  is solved by Newton's method with a direct sparse factorisation per
  iteration, which is the dominant cost for large sheets (section 5); a
  forming path of hundreds of millimetres over a fine sheet takes hours.
  Full-size parts need explicit dynamics or a cheaper element; the explicit
  step that existed was removed in the cleanup (section 7).
* **Hex8 through the thickness.** A sheet meshed with a few fully integrated
  Hex8 layers is stiff in bending (shear locking grows with the element
  aspect ratio); mean dilatation cures volumetric, not shear, locking.
  `model.element_formulation: "incompatible_modes"` gives the Hex8 that
  bends with one element through the sheet, and
  `model.integration.thickness_points` up to seven Gauss points through each
  layer (`docs/formulation.md`; the springback study, in the history at commit `7d571e6`:
  one layer with 5 / 7 points within 3.5 % / 0.9 % of beam
  theory, the standard layer 18 %). The committed mode
  parameters are part of the restartable state. On the smoke deck
  (`spif_smoke.json`, 2 threads, load average 3.4 to 3.9 from other jobs):
  the standard 2-layer mesh (mean dilatation) runs in 111 s and springs
  back by up to 0.181 mm; with incompatible modes on 2 layers 171 s (180
  increments and 1 619 Newton iterations against 137 and 1 274, 3 cuts),
  0.210 mm; on 1 layer with a 2 x 2 x 5 rule 111 s, 0.204 mm (the standard
  element on that mesh: 89 s, 0.164 mm). With the logarithmic strain the
  incompatible-mode 1-layer run takes 338 s (298 increments, 16 cuts: 11
  from the local iteration of the modes, 5 element inversions; springback
  0.193 mm), where the Green-Lagrange one takes 169 s (124 increments, no
  cut; 0.203 mm; both 2 threads, load average 5.1 to 5.8): the remaining
  local failures are at global trial states under the tool where all 20
  points yield and `K_aa` is nearly singular (smallest eigenvalue about
  `1e-6` of the largest, changing sign between iterations) - the hourglass
  instability of the modes in compression (`docs/formulation.md`), which the
  step cut resolves. One element evaluation (J2, a single pinned thread,
  load average about 6, three local iterations) costs 2.5 times the
  averaged standard one on 8 points (65 against 26 us), 2.3 times on 20
  (145 against 63 us), and 1.9 and 1.8 times under logarithmic strains
  (114 against 59, 265 against 149 us). There is
  no solid-shell element (assumed natural strain), which curved thin walls
  with tapered cells would need: the modes lose their effect on trapezoidal
  cells, and in large compression the finite element can hourglass.
* **No remeshing, no trimming, no element deletion**; no thermal effects (the
  analysis applies no temperature); no tool spin (tools translate, so the
  friction direction is that of the translation only); no tool wear or
  deformable tools; one cell type per mesh (Hex8).
* **Release.** The imbalance a release ramps out is a dead load, fixed in
  space: supports that rotated the part far while it springs back would
  change the path of its unloading (and any reverse yielding along it). A
  release onto supports that are not statically determinate leaves
  reactions that hold the part deformed; the run warns when they exceed
  1e-3 of the reference force.
* **The ramp of a changed partition is linear.** A form step that starts
  from an imbalanced state (constraints released, a tool added in contact)
  ramps that imbalance out over the whole step, which is exact only at its
  end.

## 7. Explicit forming (removed)

An explicit central-difference forming step (`form_explicit`) existed until
the cleanup. The workflow never used it; the physics audit's BM3 replica and
its explicit-vs-implicit convergence cases ran on it. Its code, tests and
this section are in the history at commit
[`7d571e6`](https://github.com/Nathan-W123/springback-precomp/tree/7d571e63af6c899717d49f95f4d7842c9fd7c682).
A deck with a `form_explicit` step is now refused as an unknown step type.
