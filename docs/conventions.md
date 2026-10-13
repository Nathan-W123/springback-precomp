# Conventions

Everything in SparLab is strict SI. There are no unit-conversion helpers and no
implicit scaling anywhere in the code, so a value read from a result file is
already in the unit its column header states. The one conversion factor is
explicit: a mesh file drawn in millimetres is read with `mesh.scale = 0.001`,
applied once as the file is read and recorded in the summary; a mesh whose
extent looks like millimetres read as metres, or one implausibly small, draws
a warning naming that key.

## Units

| Quantity | Unit | Symbol in code / files |
|----------|------|------------------------|
| Length, coordinate, displacement | metre | `m` |
| Force, reaction | newton | `N` |
| Stress, Young's modulus, yield stress, hardening moduli | pascal | `Pa` |
| Mass density | kilogram per cubic metre | `kg/m^3` |
| Mass | kilogram | `kg` |
| Strain (engineering shears), equivalent plastic strain | dimensionless | `[-]` |
| Energy, work | joule | `J` (= N m) |
| Time (the forming pseudo-time and the explicit physical time), time step | second | `s` |
| Velocity, acceleration | metre per second (squared) | `m/s`, `m/s^2` |

## Coordinate system

Right-handed Cartesian. The sheet lies in the x-y plane with its thickness
along z; the forming tool works from `+z` (the tool side is the top surface).

### Structured meshes

A `structured_hex` mesh is a box with its lower corner at `(x0, y0, z0)`.
Node numbering is `node(i, j, k) = k (nx+1)(ny+1) + j (nx+1) + i` and element
numbering `elem(i, j, k) = k nx ny + j nx + i`, x fastest, then y, then z. The
eight nodes of a Hex8 follow the VTK hexahedron convention: the bottom face
(`zeta = -1`) counter-clockwise seen from above, then the top face in the same
order:

```
        7 --------- 6
       /|          /|          zeta
      4 --------- 5 |           ^   eta
      | |         | |           |  /
      | 3 --------|-2           | /
      |/          |/            +-----> xi
      0 --------- 1
```

The six faces are numbered `0: bottom (0 3 2 1)`, `1: top (4 5 6 7)`,
`2: front y = -1 (0 1 5 4)`, `3: right x = +1 (1 2 6 5)`,
`4: back y = +1 (2 3 7 6)`, `5: left x = -1 (3 0 4 7)`, each wound so its
right-hand normal points out of the element. A cell with a non-positive
Jacobian determinant is rejected with a message naming the element.

### Meshes read from a file

An Abaqus / CalculiX `.inp` file whose solid cells are 8-node hexahedra
(`C3D8` and its formulation variants such as `C3D8R`, all read as the deck's
Hex8). Nodes keep the order of the file, renumbered consecutively from 0 after
the nodes no cell uses are dropped; cells keep the order of the file. A
hexahedron in the mirrored orientation is re-ordered as it is read and the
number of such cells is reported. Tetrahedra, prisms, second-order solids and
2-D cells are refused with a message saying to mesh with hexahedra; line and
surface elements are skipped or used as sets.

## Degrees of freedom

Three translations per node, numbered node-major:

```
  dof(node n, component c) = 3 n + c,     c = 0 -> u_x,  1 -> u_y,  2 -> u_z
```

An element's DOF vector follows the same ordering, `{u_x1, u_y1, u_z1, ...}`.
There are no rotational degrees of freedom: the Hex8 is a continuum element.
(The library still carries the 2-D branches of its kernels, unreachable now
that every mesh is Hex8.)

## Signs

* Displacements and forces are positive along `+x`, `+y`, `+z`; a formed part
  has negative `z` displacements on the tool side.
* Tensile stress is positive: `sigma_xx > 0` is tension along x.
* Reactions are reported as the force the *supports apply to the structure*.
* Tool forces (`tool_forces.csv`) are the force the sheet exerts **on the
  tool**, so a tool pressing down from above reports `fz > 0`
  ([`forming.md`](forming.md), section 3).
* Plasticity: the equivalent (accumulated) plastic strain is
  `a = int sqrt(2/3) ||d eps_p||`, which in uniaxial tension is the plastic
  strain itself; the yield stress `sigma_y(a)` and the hardening moduli are
  uniaxial values. The plastic strain is stored with engineering shears (as
  every strain), the back stress with tensorial components (as every stress).

## Voigt notation

Stress and strain are stored as vectors with the *engineering* shear strain:

```
  sigma = { sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_zx }
  eps   = { eps_xx,   eps_yy,   eps_zz,   gamma_xy, gamma_yz, gamma_zx },   gamma_ij = 2 eps_ij
```

This pairing makes `sigma^T eps` the strain-energy density, which is why the
constitutive matrix carries `G` rather than `2G` in its shear entries. The CSV
columns follow the same order (`sxx, syy, szz, sxy, syz, szx`).

## Tolerances

The implicit forming steps' Newton tolerances (`forming.newton`:
`residual_tolerance`, `displacement_tolerance`, `max_iterations`, `max_cuts`,
`max_increments`) and the explicit steps' settings (`forming.steps[].explicit`)
are deck keys, documented with their defaults in [`forming.md`](forming.md),
sections 2 and 7.2; every run records the values it used in `summary.json`.
The material-point and element tolerances are fixed constants:

| Tolerance | Value | What it bounds |
|-----------|-------|----------------|
| J2 / Hill48 return, Voce hardening | `1e-13` | the consistency residual of the plastic multiplier over the yield radius; 60 Newton iterations, else an error |
| elastic return at the yield surface | `1e-12` | a trial overshoot of the yield radius below this is elastic (round-off must not start plastic flow); a point that yielded in its last step and sits within `1e-10` of its surface keeps the continuum tangent |
| incompatible-mode parameters (local iteration) | `1e-8` | `\|r_alpha\|_inf` of the nine mode parameters over the gross scale `max_i sum_q w_q \|B~_q\|^T \|sigma_q\|`, plus a round-off floor of `1e-13` of the forces of a unit relative strain error; 25 Newton iterations (each with at most 8 line-search trials, a shifted direction lifting the smallest eigenvalue of an indefinite `K_aa` to `1e-3` of the largest, and at most 8 Levenberg-Marquardt trials from `1e-6` of the largest diagonal entry of `K^T K`), else `SolverError` and the step is cut; constants in `src/fem/IncompatibleModes.hpp`, not config keys |
| explicit energy balance onset | 10 time steps | the balance is not judged over this many time steps after the first contact (the new contact forces' half-step lag) |
| explicit kernel closed-form eigensolver | `64 eps` | residual and orthogonality of the closed-form 3 x 3 eigendecomposition (relative to the largest entry of the matrix, and absolute) above which the logarithmic strain takes the iterative solver |
| explicit Hex8 kernel self-check | `1e-10` | relative difference between the dedicated kernel and the element dispatch at the start of a run above which the dispatch is used instead |

Exceeding a solver tolerance raises an exception with a diagnosis, or cuts
the increment where the forming analysis allows it; nothing is silently
accepted.

## Determinism

The code has no randomness outside the tests: the perturbed-mesh generator of
the tests (`make_perturbed_hex_mesh`) takes an explicit seed. Repeated runs of
the same deck on the same build with one thread give bit-identical output
files; `precomp` runs every deck with one thread (`FormingSetup.threads = 1`),
and the regression checks of the cleanup compared outputs that way. Runs with
more OpenMP threads (`sparlab_form --threads n`) were not checked for bitwise
reproducibility here.
