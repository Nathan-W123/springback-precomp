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
| `steps[].type` | `"form"` | `"form"`, `"release"` or `"form_explicit"` (section 7) |
| `steps[].tools` | `[]` | active tools; a release may keep only tools the step before it lists |
| `steps[].time` | see below | `[t_begin, t_end]` [s], not before the previous step's end; later only if the tools active in both steps do not move in between |
| `steps[].max_tool_travel` | half the smallest slave-node size | largest travel of an active tool in one increment [m] |
| `steps[].increments` | 1 (form), 10 (release) | the window in this many equal increments at the start (and at most) |
| `steps[].explicit` | required for `form_explicit` | the explicit integration's settings (section 7.2); refused on other step types, and `max_tool_travel` / `increments` are refused on a `form_explicit` step |
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
| `summary.json` | `case`, `sparlab_version` (`"<version> (<git revision>)"`), `completed`, `termination`, `runtime_s`, `timing` (seconds per phase: `element_tangent_s`, `element_residual_s`, `contact_s`, `factorisation_s` with `factor_<kind>_s` per factorisation, `solve_s`, `output_s` (the step files written during the run), `total_s`; `increments`, `iterations`, `cuts`, `linear_solver` with the count of each factorisation used, `failed_factorisations` with the count of each attempt that failed - a Cholesky of a tangent that is not positive definite, then `LDL^T`; an `LDL^T` with a vanishing pivot, then LU; a singular LU, which fails the iteration - empty if none, `suitesparse`), `analysis` (kinematics, tolerances), `steps` (per step: `name`, `type`, `completed`, `increments`, `iterations`, `cuts`, `max_plastic_strain`, `reaction_norm_N`, `warnings`, and `termination`, `files_stem`, `t_begin_s`, `t_end_s`, `tools`, `constrained_dofs`, `start_imbalance_N`, `reference_force_N`, `max_displacement_change_m`, `max_displacement_m`, and for a `form_explicit` step `explicit`: `steps`, `physical_time_s`, `tool_speed_m_s`, `duration_s`, `time_step_s`, `min_time_step_s`, `final_time_step_s`, `stable_time_step_s` (unscaled), `scaled_stable_time_step_s`, `stable_step_method`, `safety`, `step_updates`, `mass_scaling`, `target_time_step_s`, `dynamic_mass_scaling`, `mass_updates`, `mass_scale_max`, `scaled_elements`, `physical_mass_kg`, `scaled_mass_kg`, `added_mass_fraction`, `max_added_mass_fraction`, `damping_per_s`, `contact_stiffness`, `contact`, `max_kinetic_ratio`, `peak_kinetic_ratio`, `kinetic_ratio_basis`, `kinetic_ratio_warning`, `max_penetration_ratio`, `penetration_warning`, `max_energy_error`, `energy_tolerance`, `energy_limit`, `history_every`, `kernel`, `timing`, `wall_s`, `energy_file`), `tools` (per tool: shape, radius, friction, penalty, trajectory span and length, peak force and its time, largest contact node count and penetration), `mesh` (element type, dim, nodes, elements, DOFs, bounding box), `warnings`, `files`, `provenance` |
| `config.json` | the deck, verbatim |
| `mesh.json` | nodes, connectivity (`ResultWriter::write_mesh`) |
| `step_<k>_<s>_nodes.csv` | `node,X,Y,Z,ux,uy,uz`: reference coordinates and the displacement at the end of step `k` (1-based) named `s` [m]; Z and uz are 0 on a 2-D model |
| `step_<k>_<s>_elements.csv` | `element,eq_plastic_strain,von_mises_Pa`: the largest equivalent plastic strain of the element's integration points, and the von Mises stress of its point-averaged Cauchy stress |
| `step_<k>_<s>.vtk` | the same fields for ParaView (unless `--no-vtk`) |
| `step_<k>_<s>_inc_<i>_nodes.csv`, `.vtk` | snapshots every `output.snapshots` increments (step ends excluded) |
| `tool_forces.csv` | `step,increment,t,tool,cx,cy,cz,fx,fy,fz,active_nodes,max_penetration_m`: per converged increment and active tool, the tool's reference point [m] and the force the body exerts **on the tool** [N]; for a `form_explicit` step a row every `history_every` time steps (`increment` is the time step's number), the force averaged over those steps |
| `step_<k>_<s>_energy.csv` | `form_explicit` steps only: `step,t_s,pseudo_t_s,time_step_s,kinetic_J,internal_work_J,stored_J,elastic_J,plastic_dissipation_J,contact_normal_work_J,contact_friction_work_J,damping_J,external_work_J,mass_scaling_work_J,energy_error_J,kinetic_ratio` every `history_every` time steps (section 7.1): `elastic_J` the elastic strain energy of the state (`nan` through the generic element dispatch), `kinetic_ratio` the kinetic energy over the step's reference energy (`kinetic_ratio_basis`), unfiltered |

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
| Explicit forming step (`tests/test_explicit.cpp`, `[explicit][forming]`): a plastic sheet dented and lifted by a sphere in `form_explicit` steps (finite kinematics, friction, selective mass scaling), damped to rest by a tool-free explicit step, released implicitly with the same clamps; a restart from the explicit steps' end state | the restart releases to the same state bit for bit; with nothing changed at the handoff the release ramps out `6e-14` of the forming force and moves the part by `7e-19 m`, one iteration per increment; the `explicit` block read key by key strictly, every refusal naming its key; the energy CSV, tool force rows and summary fields written |

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
  scaling, which is how incremental forming is usually simulated at scale -
  now the `form_explicit` step (section 7).

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
  The explicit `form_explicit` step (section 7) is the answer for full-size
  parts, with its own limitations (section 7.5).
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

## 7. Explicit forming

A `form_explicit` step moves its tools along their paths in **physical
time** and integrates the model by explicit central differences
(`include/sparlab/fem/ExplicitDynamics.hpp`, whose header states the
formulation in full): no Newton iterations, no factorisation - a time step
costs one evaluation of the internal forces - but a time step bounded by the
stability limit. It is the industrial way of simulating incremental forming
at scale (a large lumped mass and a fast tool, checked for quasi-static
validity), followed here by the same implicit `release` as the implicit
analysis, which gives the springback.

### 7.1 Formulation

* **Equations of motion** on the step's free DOFs,
  `M a + alpha M v + f_int(u) = f_c(u, v, t)`: `M` the lumped mass (the row
  sums of the consistent element masses), `alpha` the
  optional mass-proportional damping [1/s], `f_int` the internal forces of the
  non-linear static analysis (the same elements, kinematics, plasticity,
  mean dilatation and history), `f_c` the contact forces of the step's tools.
  The step's constraints (`hold` or `absolute`, as for `form`) move linearly
  in physical time.
* **Central differences with half-step velocities** (Belytschko, Liu and
  Moran, *Nonlinear Finite Elements for Continua and Structures*, 2000, box
  6.1): `v(n+1/2) = v(n) + dt/2 a(n)`, `u(n+1) = u(n) + dt v(n+1/2)`,
  `a(n+1) = M^-1 (f_c - f_int(u(n+1)) - alpha M v(n+1/2))`,
  `v(n+1) = v(n+1/2) + dt/2 a(n+1)`. Every time step is final: the plastic
  history is updated in place, never iterated.
* **Time scaling.** The tool trajectories are tabulated in pseudo-time; the
  step maps the pseudo-time window to physical time either by `tool_speed`
  [m/s] - each segment between the paths' knots lasts the largest travel of a
  tool on it over the speed, so the fastest tool moves at exactly that speed
  and segments on which no tool moves take no time - or by a `duration` [s]
  (linearly).
* **Stable time step.** Central differences are stable for
  `omega_max dt <= 2`. The default estimate bounds `omega_max` by the largest
  element eigenvalue `omega_e^2 = lambda_max(M_e^-1/2 K_e M_e^-1/2)` - the
  element's linear elastic stiffness at its current configuration and its
  (scaled) lumped mass - an upper bound of the assembled system's by Irons'
  theorem; `power_iteration` estimates `omega_max` of the assembled,
  unconstrained `M^-1 K` (Rayleigh quotient times 1.05, tighter, whole model
  only); `element_length` takes `L_e / c_e` (volume over largest face,
  dilatational wave speed; the conventional estimate, not a bound). With
  finite kinematics the estimate is repeated every `update_every` steps on
  the current configuration (a thinning sheet lowers it), the step never
  growing by more than 1 % at an update. Damping and the contact penalty
  lower the limit to `omega^2 dt^2 + 2 alpha dt <= 4 - s_c`; the step is
  `safety` times that.
* **Mass scaling** multiplies an element's density by `s_e >= 1`: `uniform`,
  `s_e = (dt_target / dt_crit)^2` for every element; `selective`
  (conventional, per element), `s_e = max(1, (dt_target / dt_e)^2)`, which adds
  mass only where an element limits the step. With `dynamic` selective
  scaling the scales are raised at every stable-step update wherever an
  element's step has fallen below the target (a wall thinning and shearing
  under the tool), so the step stays at the target; the velocities are kept
  and the kinetic energy of the added mass, `W_m = 1/2 sum dm v^2`, is booked
  in the energy balance. The added mass is reported (`added_mass_fraction`,
  at the end of the run) and warned about above `max_added_mass_fraction`.
  (Selective mass scaling proper - a non-diagonal mass that keeps the
  rigid-body inertia, Olovsson, Simonsson and Unosson, IJNME 63 (2005)
  1436-1445 - needs an iterative mass solve and is not provided.)
* **Contact** with the same rigid tools (sphere, plane, cylinder; the slave
  surfaces, trajectories and tributary areas of the implicit analysis), node
  to surface in the current configuration, but with the **mass-based
  penalty** of explicit codes: `k_j = s_c m_j / dt^2` per slave node (`m_j`
  its lumped mass, `s_c` = `contact_stiffness`, default 0.5). It adds at most
  `s_c / dt^2` to `omega^2`, so it cannot destabilise the step - the implicit
  law `kappa = s E / h` is stiffer still and would. Its penetration, about
  `f dt^2 / (s_c m_j)`, does **not** fall with the tool speed (with mass
  scaling to a target step, `m_j / dt^2` is fixed): it biases the formed
  shape at any speed, so every run measures the largest penetration against
  the element thickness at the node (volume over the largest face of its
  thinnest element; `max_penetration_ratio`) and warns above
  `penetration_warning` (1 %). Friction is
  regularised Coulomb friction with a committed tangential force per node,
  carried onto the current tangent plane, incremented by the tangential
  motion relative to the tool times `k_T = tangential_penalty k_j`, and
  returned onto the cone `|F_T| <= mu |f_N|`; for a frictional tool the step
  reserves `s_c max(1, tangential_penalty)`, the stiffer of the two springs
  (a share of 4 or more is refused).
* **Internal forces.** A dedicated Hex8 kernel (finite, logarithmic or small
  strain; elastoplastic through the return of `Plasticity.hpp`, elastic
  elements with small strain only; mean dilatation) caches the reference
  gradients and weights, updates the history in place, gathers the element
  forces per node in ascending element order (the order of the serial
  assembly), and allocates nothing. It is checked against the generic element
  dispatch (`NonlinearSystem::evaluate`) at the start of every run and falls
  back to it above a relative difference of `1e-10`; every other element,
  formulation, kinematics or material takes the generic dispatch - the single
  entry point of the element technologies, so a new one works in explicit
  steps unchanged. The results are the same, bit for bit, on any number of
  threads.
* **Energy balance** (kept at every step, written every `history_every`):
  the kinetic energy `T`, the work of the internal forces `W_int` (the
  stored elastic and hardening energy plus the plastic dissipation), of the
  contact forces on the body (normal and friction parts), of the damping and
  of the reactions, as trapezoidal sums, and `W_m` of dynamic mass scaling;
  the balance error `T - T_0 + W_int + D - W_ext - W_c - W_m` relative to the
  largest energy so far - not judged over the ten time steps after the first
  contact, where the new contact forces' half-step kick is in `T` a step
  before their work is in the balance (a relative error of 1 at the first
  touch, when that kick is all the energy there is; `1e-2` ten steps on).
* **Validity checks.** Quasi-static: the largest ratio of `T` to a
  reference energy after the first contact (`max_kinetic_ratio`, warned
  above `kinetic_ratio_warning`, 0.1). From a virgin sheet the reference is
  the internal work `W_int`; for a step that continues a stressed part (a
  retract) it is the **elastic strain energy** of the state, which the
  dedicated kernel sums at every record (`kinetic_ratio_basis`): the stored
  energy of a formed part is mostly hardening energy, which no retract stirs
  (92 % of it after the test dent). The generic element dispatch does not
  separate the elastic energy and keeps the internal energy. The warned
  ratio skips the records whose reference energy is below 1 % of its final
  value - the first touch of a flat sheet, where any motion is large against
  the little work done; the unfiltered peak is reported beside it
  (`peak_kinetic_ratio`: 0.4 to 0.8 on the smoke runs, 0.2 to 0.4 on the
  test dent, whatever the speed). Also checked: the balance error (warned
  above `energy_tolerance`, 0.05), the penetration and the added mass. A
  run stops - the step not completed, its termination naming the time step
  and the time, the analysis keeping the last recorded state - when the
  state becomes non-finite, an element inverts, a return fails, a tool is
  found through the body, or the balance error exceeds `energy_limit` (0.5).
* **Handoff to the implicit release.** The explicit step hands on the
  analysis state (displacement, plastic history, friction history) and its
  velocity (which a following explicit step starts from; an implicit step
  starts at rest). A following `release` finds as its start imbalance, besides
  the reactions of released constraints and the forces of removed tools, the
  **inertia and damping forces** of the explicit step's end state, and ramps
  them out with the rest: after a quasi-static run they are small (`T/W_int`
  small), and a sheet brought to rest before the release leaves only
  round-off (`tests/test_explicit.cpp`, section 4). An implicit `form` step with a tool still in
  contact would evaluate the explicit penetration with the stiffer implicit
  penalty; follow an explicit step by a `release` (or retract the tool
  explicitly first, as the decks do).

### 7.2 Configuration and settings guide

```jsonc
{ "name": "form", "type": "form_explicit", "tools": ["tool"], "time": [0.0, 3.1],
  "explicit": {
    "tool_speed": 1.0,                       // [m/s]; or "duration": [s] - exactly one
    "mass_scaling": { "mode": "selective",   // "none" | "uniform" | "selective"
                      "target_time_step": 1.0e-6,          // [s], required with a mode
                      "max_added_mass_fraction": 0.05,      // warn above
                      "dynamic": true },                    // keep the step at the target
    "stable_step": { "method": "element_eigenvalue",        // | "element_length" | "power_iteration"
                     "safety": 0.9, "update_every": 1000, "power_iterations": 60 },
    "damping": 0.0,                          // alpha of C = alpha M [1/s]
    "contact_stiffness": 0.5,                // s_c of k = s_c m / dt^2, in (0, 1]
    "history_every": 100,                    // records (energy CSV, tool_forces.csv rows)
    "snapshot_every": 0,                     // node CSV snapshots every N time steps (0: none)
    "energy_tolerance": 0.05, "energy_limit": 0.5, "kinetic_ratio_warning": 0.1,
    "penetration_warning": 0.01 },
  "boundary_conditions": [ ... ] }
```

| Key | Default | Meaning |
|-----|---------|---------|
| `tool_speed` | - | the fastest active tool's speed along its path [m/s]; the window's physical time follows |
| `duration` | - | the window's physical time [s] (linear in pseudo-time); needed when no tool moves |
| `mass_scaling.mode` | `"none"` | `"uniform"`: one factor `(dt_target / dt_crit)^2`; `"selective"`: per element `max(1, (dt_target / dt_e)^2)` |
| `mass_scaling.target_time_step` | - | [s]; the time step is at most this |
| `mass_scaling.max_added_mass_fraction` | 0.05 | added / physical mass above which the run warns |
| `mass_scaling.dynamic` | false | selective, finite kinematics: at every stable-step update, raise the scale of the elements whose step has fallen below the target (a thinning wall), so the step stays at the target; the velocities are kept and the added mass's kinetic energy is booked in the balance |
| `stable_step.method` | `"element_eigenvalue"` | see 7.1; `"power_iteration"` cannot drive selective scaling (refused) |
| `stable_step.safety` | 0.9 | the step over the stability limit, in (0, 1] |
| `stable_step.update_every` | 1000 | finite kinematics: re-estimate every N steps (0: never) |
| `stable_step.power_iterations` | 60 | iterations of the power method |
| `damping` | 0 | mass-proportional damping `alpha` [1/s] |
| `contact_stiffness` | 0.5 | `s_c` of the mass-based penalty; `s_c max(1, tangential_penalty)` of a frictional tool must stay below 4 |
| `history_every` | 100 | a record every N steps (and the last): the energy CSV rows and the tool force rows, the force averaged over the N steps |
| `snapshot_every` | 0 | node snapshots every N time steps (`step_<k>_<s>_inc_<N>_nodes.csv`) |
| `energy_tolerance`, `energy_limit` | 0.05, 0.5 | balance error warned about / stopping the run |
| `kinetic_ratio_warning` | 0.1 | `T` over the reference energy (internal or elastic, 7.1) after the first contact warned about |
| `penetration_warning` | 0.01 | the largest contact penetration over the element thickness at the node warned about |

**How to choose the tool speed and the mass scaling.** Both make the run
cheaper and both raise the inertia forces: the tool speed `v` by `v^2`, the
mass scale `s` by `s` - so what governs the dynamics is the **equivalent
speed `v sqrt(s)`**, and the number of time steps is `path / (v dt)` with
`dt` proportional to `sqrt(s)`: the cost is inversely proportional to the
equivalent speed, however it is split. A sheet meshed with Hex8 is limited by
its through-thickness size: on the 1 mm blanks of the decks (0.5 mm layers)
the unscaled stable step is about `8e-8 s`, so a `1e-6 s` target scales every
element by 150 to 270 (`added_mass_fraction` about 200, which the decks
warn about: their threshold is the default) and `v = 1 m/s` is an
equivalent speed of about 14 m/s. The equivalent speed governs the inertia;
the contact penalty is a second, independent parameter (item 5). Guidance,
from the smoke study (section 7.3) and the `explicit-dent` validation
(in the history at commit `7d571e6`):

1. Pick the mass scaling target so that the unscaled stable step is not the
   bottleneck (selective scaling scales only the elements that limit it);
   then set the tool speed from the time budget.
2. Check the validity measures: `max_kinetic_ratio` below 0.1 (0.04 at the
   recommended 7 m/s on the smoke case; `peak_kinetic_ratio`, which includes
   the tool's plunge into the flat sheet, is 0.4 to 0.8 at any speed and is
   not a criterion), `max_energy_error` below 0.05, `max_penetration_ratio`
   below 0.01, the tool force's moving average smooth.
3. Check convergence: halve the equivalent speed once and compare the
   springback of the two runs on the same mesh (the comparison script,
   `compare_forming_runs.py`, is in the history at commit `7d571e6`); the difference should be small against the springback. Halving the
   speed does not reveal the penalty's bias (item 5).
4. Leave `damping` at 0 for forming: mass-proportional damping resists the
   rigid motion of the material that follows the tool and adds a force
   proportional to the speed. It is useful to bring a sheet to rest (a
   tool-free explicit step with a `duration`) before comparing states.
5. Keep `contact_stiffness` at 0.5 or above. The penetration,
   `f dt^2 / (s_c m)`, is the same at every tool speed. On the test dent
   (a 2 mm ball 0.3 mm into a 1 mm sheet, 0.5 mm elements) the former default
   0.1 penetrated 17 um - 3.4 % of the element - and left the shape 7 % of
   the depth from the implicit analysis at 4 and at 1 m/s alike; 0.5
   penetrates 3 um (0.6 %) and agrees to 0.9 %, the implicit reference's own
   increment difference. On the smoke case the penetration is 6 um at 0.1 and
   0.5 um at 1, and the shape does not move with it (section 7.3). The
   penalty's share lowers the stable step as `sqrt(4 - s_c)`: with selective
   scaling to a target step it costs added mass - 0.5 against 0.1 raises the
   equivalent speed by 5.6 %, 1 by 14 %.

### 7.3 Verification against the implicit analysis (smoke case)

`configs/forming/spif_smoke_explicit.json` is `spif_smoke.json` with its
`form` and `retract` steps explicit; it was run at four equivalent speeds and
three to four contact penalties and compared node by node (by
`python/scripts/compare_forming_runs.py`, in the history at commit `7d571e6`) with
the implicit analysis (`sparlab_form --threads 2`, load average 3 to 7 from
other jobs, so the run times are indicative only). The implicit reference is
itself discretised by its tool travel per increment: the deck's 1 mm and a
refined 0.5 mm differ by `0.033 mm` (largest) and `0.006 mm` (RMS) in the
formed tool-side surface and by 2.8 % in the mean tool force, so the
comparison is made with the refined run (springback `0.182 mm` largest,
`0.126 mm` RMS; depth 2.23 mm; mean tool force 418 N over the contours;
largest penetration `0.32 um`). Distances are over the 441 nodes of the
tool-side surface; the springback is the displacement over the `unclamp`
release; the tool force is `fz`, its moving average over 0.1 s of
pseudo-time (the explicit rows scatter by 12 % RMS about it).

**The equivalent speed**, at the former default penalty `contact_stiffness`
0.1:

| Equivalent speed `v sqrt(s)` | Settings | Time steps | Run [s] | `max_kinetic_ratio` (over the contours) | Springback difference, largest / RMS [mm] | Final shape difference [mm] | Formed shape difference [mm] | Mean tool force | Moving-average force difference, RMS |
|---|---|---|---|---|---|---|---|---|---|
| 28 m/s | 2 m/s, 1 us (`s` = 201) | 53 722 | 25 | 0.247 (0.103) | 0.042 / 0.023 (23 % of the springback) | 0.046 / 0.018 | 0.049 / 0.012 | -3.4 % | 9.0 % |
| 14 m/s | 1 m/s, 1 us | 107 694 | 47 | 0.142 (0.011) | 0.017 / 0.008 (9.1 %) | 0.020 / 0.008 | 0.029 / 0.006 | -3.3 % | 4.4 % |
| 7.1 m/s | 0.5 m/s, 1 us - or 1 m/s, 0.5 us (`s` = 50): the same, bit for bit | 215 532 | 93 | 0.038 (0.007) | 0.0076 / 0.0040 (4.2 %) | 0.014 / 0.005 | 0.021 / 0.004 | -1.4 % | 3.3 % |
| 3.5 m/s | 1 m/s, 0.25 us (`s` = 13) | 431 297 | 187 | 0.015 (0.004) | 0.0024 / 0.0011 (1.3 %) | 0.017 / 0.004 | 0.020 / 0.004 | -0.8 % | 3.4 % |
| implicit, 1 mm travel (the deck) | 137 increments, 1 274 iterations | - | 110 | - | 0.0032 / 0.0011 | 0.034 / 0.006 | 0.033 / 0.006 | -2.8 %  (against 0.5 mm) | 4.7 % |
| implicit, 0.5 mm travel (reference) | 249 increments, 1 862 iterations | - | 169 | - | - | - | - | - | - |

**The contact penalty** at three equivalent speeds (the unfiltered peak
of the kinetic ratio, over the plunge into the flat sheet, in brackets):

| Equivalent speed | `contact_stiffness` | Largest penetration | Springback difference, largest / RMS [mm] | Final shape [mm] | Formed shape [mm] | Formed depth [mm] | Mean tool force | `max_kinetic_ratio` (peak) | Energy balance |
|---|---|---|---|---|---|---|---|---|---|
| 14 m/s | 0.1 | 6.5 um | 0.017 / 0.008 (9.1 %) | 0.020 / 0.008 | 0.029 / 0.006 | 2.247 | -3.3 % | 0.142 | `5.6e-5` |
| 14 m/s | 0.4 | 1.5 um | 0.022 / 0.010 (12.2 %) | 0.023 / 0.008 | 0.028 / 0.007 | 2.256 | -3.6 % | 0.146 | `3.5e-4` |
| 14 m/s | 0.5 | 1.2 um | 0.026 / 0.011 (14.5 %) | 0.023 / 0.008 | 0.024 / 0.006 | 2.254 | -3.7 % | 0.148 (0.45) | `4.6e-4` |
| 14 m/s | 1 | 0.6 um | 0.013 / 0.008 (7.1 %) | 0.022 / 0.009 | 0.021 / 0.004 | 2.241 | -3.5 % | 0.156 (0.44) | `1.6e-3` |
| 7.1 m/s | 0.1 | 6.3 um | 0.0076 / 0.0040 (4.2 %) | 0.014 / 0.005 | 0.021 / 0.004 | 2.246 | -1.4 % | 0.038 | `6.0e-5` |
| 7.1 m/s | 0.4 | 1.5 um | 0.0054 / 0.0032 (3.0 %) | 0.018 / 0.005 | 0.023 / 0.005 | 2.249 | -1.7 % | 0.038 | `3.3e-4` |
| **7.1 m/s** | **0.5 (the deck)** | 1.2 um | **0.0063 / 0.0032 (3.5 %)** | 0.017 / 0.005 | 0.023 / 0.004 | 2.249 | -1.7 % | **0.039** (0.62) | `4.3e-4` |
| 7.1 m/s | 1 | 0.5 um | 0.0042 / 0.0026 (2.3 %) | 0.016 / 0.005 | 0.024 / 0.005 | 2.253 | -1.8 % | 0.049 (0.63) | `1.3e-3` |
| 3.5 m/s | 0.1 | 6.3 um | 0.0024 / 0.0011 (1.3 %) | 0.017 / 0.004 | 0.020 / 0.004 | 2.245 | -0.8 % | 0.015 | `5.7e-5` |
| 3.5 m/s | 0.4 | 1.5 um | 0.0023 / 0.0012 (1.3 %) | 0.020 / 0.005 | 0.022 / 0.005 | 2.248 | -1.2 % | 0.015 | `3.1e-4` |
| 3.5 m/s | 0.5 | 1.1 um | 0.0023 / 0.0011 (1.3 %) | 0.020 / 0.005 | 0.022 / 0.005 | 2.249 | -1.2 % | 0.015 (0.76) | `3.9e-4` |
| 3.5 m/s | 1 | 0.5 um | 0.0035 / 0.0016 (1.9 %) | 0.020 / 0.005 | 0.022 / 0.005 | 2.246 | -1.3 % | 0.014 (0.80) | `9.0e-4` |

(The 0.5 runs at the current revision - 52 s, 111 s and 394 s at load 4 to
4.6; the others with earlier builds, whose states are the same: their
Green-Lagrange kernel is unchanged, and the energy-balance onset of 7.1
only lets the `s_c = 1` runs past their first touch, where they stopped
before.)

What it shows:

* **The springback converges to the implicit one** as the equivalent speed
  falls - 23 %, 9.1 %, 4.2 % and 1.3 % of the springback for 28, 14, 7.1 and
  3.5 m/s at the former penalty, 14.5 %, 3.5 % and 1.3 % at the default
  one - and so does the mean tool force (-3.4 % to -0.8 %).
* **The penalty is a second parameter**, not a part of the equivalent speed.
  At 14 m/s the springback difference moves between 7 % and 15 % with it, at
  no monotonic trend - the inertia's error is then of that order anyway; at
  7.1 m/s between 2.3 % and 4.2 %, at 3.5 m/s between 1.3 % and 1.9 %. On this
  sheet (2 x 2 x 0.5 mm elements, 400 N on a 5 mm tool) the penetration is
  6 um at 0.1 and 1 um at 0.5: small against the shape differences, which do
  not follow it. On the test dent, a stiffer contact, it is 17 um at 0.1 and
  biases the shape by 7 % of the depth at any speed (`explicit-dent`,
  in the history at commit `7d571e6`); hence the default 0.5.
* **The formed and final shapes** level off at about `0.02 mm` largest and
  `0.004` to `0.005 mm` RMS on the 2.2 mm deep part, at every speed and every
  penalty - their explicit depth is 0.015 to 0.03 mm deeper than the
  reference's - of the size of the implicit reference's own increment
  difference (`0.033 mm` between 1 and 0.5 mm of travel, so its own error at
  0.5 mm is likely of that order). The study cannot resolve the shapes better
  than that.
* **Mass scaling and tool speed are interchangeable**: 1 m/s with a 0.5 us
  target gives the same run as 0.5 m/s with 1 us, bit for bit (the equations
  differ only in the unit of time, by a power of two).
* **The kinetic energy ratio** after the plunge is below the 0.1 warning at
  7.1 m/s and below, which is the recommended setting for springback work
  (3 to 4 % of the springback); 14 m/s gives the shapes and forces at half
  the cost, its springback 7 to 15 % off. The unfiltered peak, 0.4 to 0.8
  while the tool plunges into the flat sheet, does not fall with the speed:
  it measures the first touch, not the forming.
* The energy balance closes to `1.1e-4` (28 m/s) and `6e-5` (the others) of
  the internal work at 0.1, `3e-4` to `5e-4` at 0.4 and 0.5 and `1.6e-3` at 1:
  the stiffer penalty's contact work is integrated less exactly.
* The explicit run is not cheaper than the implicit one on this small sheet
  (4 000 DOFs, where a factorisation is cheap): the explicit cost grows with
  the number of elements times the path length, the implicit one faster
  (section 7.4).

### 7.4 Cost

Measured with `sparlab_form --threads 2` on the development machine (four
cores shared with other jobs; the load average is given with each number,
and above 4 the other jobs take cores from the run - a spinning OpenMP
barrier then waits for a descheduled thread, so the loaded runs are 2 to 3
times slower than the unloaded rate), GCC 13 `-O3`.

**Per time step** (`spif_cone_60_explicit.json`, in the history at commit `7d571e6`, the first
1 518 time steps: 7 200 Hex8, 33 489 DOFs, plastic, frictional contact;
load average 2.7 to 2.9 before each run):

| Threads | Time steps / s | ns per element and step (total) | of which internal forces | integration (vector updates, energies) | contact | stable-step updates |
|---|---|---|---|---|---|---|
| 1 | 158 | 879 | 715 | 0.93 s (12 %) | 0.12 s | 0.50 s |
| 2 | 296 | 470 | 358 | 0.69 s (13 %) | 0.10 s | 0.26 s |
| 4 | 195 (27 with the default active wait) | 711 | 548 | 1.17 s | 0.12 s | 0.29 s |

The four-thread row was measured with other jobs holding about three of the
four cores (load average 3.1 to 3.6; the one- and two-thread rows repeated
then gave 134 and 230 time steps a second): it shows the oversubscription,
not the code. With `OMP_WAIT_POLICY=PASSIVE` four threads still gained 10 %
over two; with OpenMP's default active wait a thread spinning at a barrier
took the core its partner needed, and the run was seven times slower. On a
shared machine, use no more threads than there are free cores (or passive
waiting).

The internal forces - the dedicated kernel's `3 000` instructions an element
and step, vectorised over the eight points - take 80 % and scale with the
threads; the vector updates are memory-bound (33 000 DOFs, a dozen arrays).
The step loop allocates nothing (tested); the stable-step updates (every
1 000 steps, an eigenvalue problem per element) and the records allocate.

**The full cone** (622 mm of toolpath, ten contours to 10 mm, 7 200 Hex8;
selective mass scaling to 1 us, scale about 224; `sparlab_form --threads 2`):

| Run | Tool speed | Time steps | Wall [s] (load) | Result |
|---|---|---|---|---|
| Green-Lagrange (`"finite"`), fixed scaling | 2 m/s | 367 800 | 1 871 (3.5 to 8.7) | stopped at pseudo-time 9.70 s of 11.9 (88 % of the path): element 1777 inverted; the step fell from 1 us to 0.43 us as the wall thinned; `max_kinetic_ratio` 0.012, energy balance `2.1e-5` |
| Green-Lagrange, dynamic scaling | 1 m/s | 539 600 | 4 598 (5 to 11) | stopped at the same place, the same element; the step held at 1 us (538 mass updates; added mass 224 to 261 times the physical, the largest element scale 1 262); 0.005, `3.0e-5` |
| **logarithmic, dynamic scaling** (the deck, revision `694eeaa`, `contact_stiffness` 0.5) | 4 m/s | 152 677 + 2 750 (retract) | **2 766** (2.9 to 3.9) | **completed**: depth 10.23 mm, largest plastic strain 1.05; `max_kinetic_ratio` 0.042 (form; unfiltered peak 0.56, at the plunge), 0.14 against the elastic energy in the retract (warned); largest penetration 1.8 % of the element (warned: about 9 um); energy balance `7.2e-4`; 152 mass updates (added mass 279 times the physical, largest element scale 645, warned); release 10 increments, 40 iterations; internal forces 94 % of the time (2.25 us an element and step on two threads). An earlier build (`2fea9a3`, `contact_stiffness` 0.1, iterative eigensolver) took 4 081 s at load 3 to 8 |

Both Green-Lagrange runs stop at the same point of the path - the tool's
step-down from the ninth contour to the tenth, its tip pressing the part's
floor at 7 mm radius - at the same element under it, whatever the speed and
the mass scaling: the limit is the law's, not the integration's (the
additive Green-Lagrange law softens in compression, section 7.5; the
plastic strain there is 1.04). With the logarithmic kinematics the same
path completes. On an unloaded machine (296 time steps a second on two
threads with the Green-Lagrange kernel) the Green-Lagrange path would take
305 000 steps, 17 minutes at 2 m/s; the logarithmic kernel costs 3.6 us
against 0.64 us an element and step on one thread, so the deck's
logarithmic run takes about 45 minutes on two threads, and 1.5 to 3 hours at
2 or 1 m/s (projections, not runs) - against the implicit analysis's
estimated 7 hours (with the Green-Lagrange law, which would meet the same
collapse). **The deck is a feasibility and cost run, not a springback
result**: its equivalent speed (4 m/s times the square root of a scale of
250 to 645, 63 to 100 m/s) lies far above the 28 m/s at which the smoke study
leaves the springback 23 % off, and the retract warns. The implicit analysis
could not be run over the full cone for comparison; the smoke study (section
7.3) and `explicit-dent` are the comparisons.

**Choosing the speed for a part.** The cone's filtered kinetic energy ratio
stays below the 0.1 warning even at 2 m/s with the Green-Lagrange law
(0.012, against 0.25 on the smoke case at the same equivalent speed): a
large part's internal work grows with its plastic zone while the kinetic
energy stays with the material near the tool. That makes the ratio a weak
check on a large part: the unfiltered peak (0.56 on the cone at 4 m/s) and
the retract's ratio against the elastic energy (0.14) say more. The ratio is
a necessary check only; the springback error at a given
equivalent speed is what the smoke study measured (section 7.3: 23 % of the
springback at 28 m/s, 9 % at 14 m/s, 4 % at 7 m/s), and a part's own
convergence check (halving the speed once) is the sufficient one.

### 7.5 Limitations

* **The Green-Lagrange elastoplastic law softens in compression.** Its
  additive split bounds the second Piola-Kirchhoff stress by the yield
  stress, so the nominal stress `F S` falls with the compressive stretch:
  one plastic Hex8 compressed uniaxially (free sides, 100 MPa yield,
  300 MPa hardening) carries at most 123 N at a stretch of 0.7-0.8 and
  71 N at 0.3, where the logarithmic kinematics carry 295 N and 1 531 N.
  Under a tool, an element past that limit can collapse - the full cone's
  runs stop there (section 7.4), whatever the speed or scaling. Parts
  formed to plastic strains of order one need
  `"kinematics": "finite_logarithmic"`, which the dedicated kernel runs at
  about 6 times the Green-Lagrange cost.

* **Inertia is real in the model.** Mass and time scaling make the run
  dynamic by design; the results approach the quasi-static ones as the
  equivalent speed `v sqrt(s)` falls (section 7.3), at a cost inversely
  proportional to it. `max_kinetic_ratio` is a necessary check, not a
  sufficient one: the springback is the sensitive quantity - check it by
  halving the equivalent speed once.
* **Conventional mass scaling** is diagonal: it adds rigid-body inertia to
  the scaled elements (on the decks, to every element - the sheet's mass is
  about 200 times the physical one). Selective mass scaling that leaves the
  translational inertia intact (Olovsson et al. 2005) is not provided, nor is
  subcycling.
* **Mass-based penalty contact** is softer than the implicit penalty (a
  penetration of about `f dt^2 / (s_c m)`: at the default 0.5, 3 um on the
  test dent and 1 um on the smoke case, against 0.4 and 0.3 um implicitly).
  The penetration does not fall with the tool speed, so it is a bias of its
  own beside the inertia: 7 % of the test dent's depth at 0.1, 0.9 % at 0.5
  (`explicit-dent`, in the history at commit `7d571e6`). Watch `max_penetration_ratio`;
  `contact_stiffness` above 1 is refused (the step would shrink as
  `sqrt(4 - s_c)`). An implicit `form` step directly after an explicit step
  with a tool still in contact would see that penetration with the stiffer
  implicit law. Follow an explicit forming step by a `release`, or retract
  the tool explicitly first.
* **The element eigenvalue bound is conservative**: on undeformed 1 mm
  sheets of 8 x 8 to 24 x 24 x 2 Hex8 clamped on one edge it lies at 0.93 of
  the exact limit of the lumped system with 1 x 1 x 0.5 mm elements (the
  cone's) and 0.98 with 2 x 2 x 0.5 mm ones (the smoke case's), so selective
  scaling to a target step adds 15 % and 4 % more mass than the exact limit
  would need (7 % and 2 % more equivalent speed); distorted, formed elements
  were not measured. Power iteration lies at 0.97 to 0.98 but gives no step
  per element and cannot drive selective scaling.
* **The kinetic energy ratio of a retract** is measured against the
  elastic energy only in the dedicated kernel; through the generic element
  dispatch it falls back to the internal energy, whose hardening share hides
  it (`kinetic_ratio_basis` says which). The onset window of the energy
  balance (10 time steps after the first contact) and the closed-form
  eigensolver's acceptance (64 units of round-off) are fixed, not settings.
* **The dedicated kernel** covers Hex8 with the 2 x 2 x 2 rule in 3-D, finite
  (Green-Lagrange), logarithmic or small-strain kinematics, the elastoplastic
  laws (and linear elasticity with small strain), no temperature, follower
  pressure or centrifugal load; everything else (other elements, an element
  technology added to the element dispatch) runs through the generic
  dispatch - correct, but about 35 times slower (a plastic 20 x 20 x 2 sheet
  on one thread: 21 us an element and step against 0.59 us; it allocates its
  buffers at every step and forms the plastic moduli it does not need). The
  logarithmic kinematics cost 3.6 us an element and step in the kernel on one
  thread (the cone's first 1 518 time steps, load 3.3; 4.0 us before the
  closed-form eigensolver, which is accepted only where its residual shows it
  accurate - the gain, 10 %, is smaller than the solver's own 2-fold speed-up
  in isolation, whose fallback rate on a real run was not measured; 30 us
  through the dispatch): about 6 times the Green-Lagrange kernel
  (section 7.4). No reduced integration with hourglass control is provided: a
  fully integrated Hex8 costs about 3 000 instructions a step in the kernel.
* **Bulk viscosity** is not provided (it matters for shocks, not for
  quasi-static forming); tools translate (no spin); the time step is uniform
  over the model.
* **The quasi-static release** after an explicit step starts from rest: the
  kinetic energy left at the end of the explicit step is dropped, and its
  inertia forces are ramped out as part of the start imbalance.
