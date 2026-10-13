# Mathematical formulation

This document states exactly what SparLab's forming analysis solves for its
continuum: the element, the assembly and the boundary conditions, the linear
algebra, the large-deformation statics and the plasticity. What is specific
to forming - the tools, their contact and friction, the steps, the
increments and the explicit integrator - is in `docs/forming.md`. Notation
follows `docs/conventions.md`; every equation below is implemented in the
file named beside it.

## 1. Continuum problem

Small-strain linear elasticity on a solid domain `Omega` with displacement
boundary `Gamma_u`; the rest of the boundary is free except where a rigid
tool's penalty contact acts (`docs/forming.md`, section 1.3):

```
  div sigma = 0                in Omega          (equilibrium)
  eps = 1/2 (grad u + grad u^T)                  (kinematics)
  sigma = D : eps                                (constitutive)
  u = u_bar                    on Gamma_u
```

The weak form, for all admissible `v`, is

```
  integral_Omega  eps(v)^T D eps(u) dOmega  =  sum_k v(x_k)^T F_k
```

with nodal forces `F_k` - in the forming analysis the contact forces of the
tools, which apply no other load. Each element may carry its own material
(`material_regions`), so `D` and `rho` are per element. Sections 6 and 7
extend the problem to large deformation and plasticity.

With Voigt ordering `{sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_zx}`
and the three engineering shear strains, the isotropic law in terms of the
Lame constants `lambda = E nu / ((1+nu)(1-2nu))` and `G = E / (2(1+nu))` is
(`material/IsotropicMaterial.cpp`)

```
        | lambda+2G   lambda     lambda    0  0  0 |
        | lambda    lambda+2G    lambda    0  0  0 |
  D  =  | lambda      lambda   lambda+2G   0  0  0 |          (6 x 6)
        |    0          0         0        G  0  0 |
        |    0          0         0        0  G  0 |
        |    0          0         0        0  0  G |
```

which the test suite checks entry by entry against this Lame form. The shear
entries are `G`, not `2G`, because Voigt stress is paired with *engineering*
shear strain. There is no thickness: a deck that sets `model.thickness` to
anything but 1 is refused.

## 2. Element formulation

### Shape functions and isoparametric mapping

The element is the eight-node trilinear hexahedron (`elements/Hex8.cpp`) on
the reference cube `(xi, eta, zeta) in [-1,1]^3`, nodes in the VTK order of
`docs/conventions.md`:

```
  N_a = 1/8 (1 + xi_a xi)(1 + eta_a eta)(1 + zeta_a zeta),   (xi_a, eta_a, zeta_a) = +-1
```

These satisfy `sum_a N_a = 1` everywhere, `N_a = delta_ab` at node `b`, and
`sum_a grad N_a = 0` - all three are unit-tested. Geometry and displacement
use the same basis,

```
  x(xi) = sum_a N_a(xi) x_a        u(xi) = sum_a N_a(xi) u_a
```

The `3 x 3` Jacobian `J_ij = dx_i / dxi_j` is inverted explicitly through its
adjugate (the element loops run it at every Gauss point of every element),
and `dN_a/dx = J^-T dN_a/dxi`. `detJ <= 0` at any quadrature point means the
element is inverted, folded or its nodes do not follow the VTK order, and
raises `MeshError` naming the parametric point.

### Strain-displacement operator

```
          | dN_a/dx     0        0     |
          |   0      dN_a/dy     0     |
  B_a  =  |   0         0     dN_a/dz  |        columns of node a, rows in the
          | dN_a/dy  dN_a/dx     0     |        order xx, yy, zz, xy, yz, zx
          |   0      dN_a/dz  dN_a/dy  |
          | dN_a/dz     0     dN_a/dx  |
```

so `eps = B u_e` with the `6 x 24` operator and `u_e` in node-major order.
Because the trilinear space contains all linear fields exactly, `B u_e`
reproduces a constant strain field exactly at *every* parametric point, on
distorted cells as well as boxes; the unit tests check it with six
independent constant strains on a distorted cell, and `sum_g w_g detJ`
against the volume of a sheared box.

### Quadrature

Gauss-Legendre tensor-product rules on the reference cube
(`elements/Quadrature.cpp`). An `n`-point rule per direction integrates
polynomials of degree `2n - 1` exactly.

| Kernel | Default rule | Why |
|--------|--------------|-----|
| Stiffness `K_e`, internal forces | 2 x 2 x 2 | exact for the Hex8 on a parallelepiped; full integration, no reduced-integration hourglass modes |
| Mass `M_e` | 3 x 3 x 3 | `N^T N` is quadratic in each direction, and on a general cell `detJ` is not constant |
| Contact areas of boundary faces | 3 x 3 | the tributary areas of the tools' slave nodes (`docs/forming.md`, section 1.3) |

The stiffness and mass rules are set per run by `model.integration` (1 to 4
points per direction), and a sheet's rule through its thickness below. On a
box the 2x2x2, 3x3x3 and 4x4x4 stiffness matrices agree to round-off, which
is unit-tested.

### Element matrices

```
  K_e = integral_Omega_e  B^T D B  dOmega
      = sum_g  w_g  detJ(xi_g)  B(xi_g)^T D B(xi_g)                 [N/m]

  M_e = integral_Omega_e  rho N^T N dOmega
      = sum_g  w_g  rho detJ(xi_g)  N(xi_g)^T N(xi_g)               [kg]
```

where `N` is the `3 x 24` shape-function matrix. Both are symmetrised after
integration, since any asymmetry is pure round-off.

`K_e` is symmetric with exactly **six** zero eigenvalues - three
translations and three rotations - and eighteen strictly positive ones. Both
parts of that statement are unit-tested on a box and on a distorted element.

Row-sum lumping of `M_e` gives the diagonal mass matrix of an explicit
forming step. The row sums add up to the exact element mass, and the sum of
every entry of `M_e` is `3 rho V`, one factor per translation direction;
both are unit-tested on a box and on a distorted cell.

The fully integrated Hex8 is stiff in bending: it needs several elements
through a bending depth, or the incompatible modes below.

### The Hex8 with incompatible modes

With `model.element_formulation: "incompatible_modes"` a Hex8 mesh takes the
element of Wilson, Taylor, Doherty and Ghaboussi (1973, *Incompatible
displacement models*, in *Numerical and Computer Methods in Structural
Mechanics*) with the correction of Taylor, Beresford and Wilson (1976, *A
non-conforming element for stress analysis*, IJNME 10)
(`elements/Hex8Incompatible.cpp`, `fem/IncompatibleModes.cpp`). The
trilinear element cannot bend: a pure-bending field puts a parasitic shear
`gamma_xz ~ x` into every cell, because its sides stay straight, so an
element long against its depth is too stiff by a factor that grows as
`(L/h)^2`. That is what spoils a sheet meshed with one or two elements
through its thickness, and with it the springback, which is the elastic
unloading of a bending moment.

**The modes.** Three bubble fields `phi_k = 1 - xi_k^2` per displacement
component (nine parameters `alpha` per element), which vanish at the nodes
and are not continuous between elements. Their gradients are taken with the
Jacobian `J_0` of the element centre, scaled by the ratio of determinants:

```
  g~_k(xi) = (j_0 / j(xi)) (-2 xi_k) J_0^-T e_k            (Taylor's correction)
  H = U G^T + A G~^T                                        (A: 3 x 3, column m = pseudo-node m)
```

so `sum_q w_q j_q g~_k = 0` for every rule symmetric in each direction: a
constant strain puts no load on the modes, `alpha = 0`, and the element
passes the patch test on any geometry (Wilson's original fails it on a
distorted cell). The modes enter every kinematic operator as three extra
"pseudo-nodes": the element is an eleven-node element of 33 degrees of
freedom `(u, alpha)`, and the linear `B`, the Green-Lagrange
`B_NL(F, [G | G~])` and the geometric stiffness `([G|G~]^T S [G|G~]) x I` are
those of the library applied to eleven gradient columns. In small strain the
strain span of the modes is the enhanced field EAS-9 (Andelfinger and Ramm
1993), and the condensed stiffness equals Simo and Rifai's (1990). In finite
deformation this is the enhanced deformation gradient `F = I + H` of Simo
and Armero (1992, *Geometrically non-linear enhanced strain mixed methods
and the method of incompatible modes*, IJNME 33) and Simo, Armero and Taylor
(1993, CMAME 110); it is objective (a rigid turn `R` is carried by `R A`).

**The linear stiffness** - which the stable-step estimate of an explicit
step takes - condenses in closed form, with fixed-size 24 x 24, 24 x 9 and
9 x 9 blocks:

```
  K* = K_uu - K_ua K_aa^-1 K_au,     K_aa = int B~^T D B~ dV,   K_au = int B~^T D B dV
  B^ = B - B~ K_aa^-1 K_au          (int B^^T D B^ dV = K*)
```

`strain_operator` stays the compatible `B` (its normal rows are read as
shape-function gradients elsewhere). The mass is that of the Hex8: the modes
carry no inertia.

**Non-linear analysis** (the forming analysis; small strain, `finite`,
`finite_logarithmic`; J2, Hill48, Chaboche, and elastic with small
strain). The modes are internal, `r_alpha(u, alpha) = 0`, and on every
evaluation of an element the parameters are found by Newton's method from
the parameters committed at the last converged step, with every point's
return from its committed state:

```
  alpha_0 = alpha_n,      alpha_{i+1} = alpha_i + s_i d_i,      d_i = -K_aa^-1 r_alpha
  f*  = f_u - K_ua K_aa^-1 r_alpha        (the last correction: the error of f* is quadratic)
  K*  = K_uu - K_ua K_aa^-1 K_au          (K_au, not K_ua^T, for a non-symmetric tangent)
```

so an element is a pure function of `u` and the committed history, which the
line search, the increment cuts and the bitwise-deterministic parallel
evaluation rely on (the Simo-Rifai update of `alpha` with the global iterate
would not be). It stops at
`|r_alpha|_inf <= 1e-8` of the gross scale `max_i sum_q w_q |B~_q|^T |sigma_q|`
plus a round-off floor, and raises `SolverError`, which cuts the increment,
after 25 iterations or at a singular `K_aa`. The full step (`s_i = 1`) is
taken whenever it reduces `|r_alpha|`; far from the solution - a return far
from its committed state, a load reversal - the points switch between
elastic and plastic along the step, `r_alpha` is only piecewise smooth and
Newton's iterates can cycle, so `s_i` then comes from a regula-falsi search
for the root of `d_i^T r_alpha(alpha_i + s d_i)`, the derivative of the
incremental potential along `d_i` wherever there is one (Crisfield 1991,
*Non-linear finite element analysis of solids and structures*, vol. 1,
sec. 9.3); a trial that inverts a point counts as beyond the root, and where
that derivative brackets no root on the step (a non-associative law, a
non-convex potential) the step is halved on `|r_alpha|`, the best trial
taken only if it lowers `|r_alpha|`. Where the symmetric part of `K_aa` is
indefinite (a softening plastic tangent against the geometric stress term
of finite kinematics - frequent under logarithmic strains far from the
solution) and the full Newton step does not lower `|r_alpha|`, the search
runs along the shifted direction `(K_aa + mu I) d = -r_alpha`, `mu` lifting
the smallest eigenvalue of that symmetric part to `1e-3` of the largest (a
modified Newton method, a descent direction of the potential; Nocedal and
Wright 2006, sec. 3.4), then along Newton's direction on `|r_alpha|`. If
nothing lowers `|r_alpha|`, Levenberg-Marquardt steps
`(K^T K + mu I) d = -K^T r_alpha`, `mu` growing tenfold from `1e-6` of the
largest diagonal entry (More 1978); no fallback moves to a worse point, and
if none helps the increment is cut. Near the solution the full Newton step
is always taken, also to a solution where `K_aa` is indefinite. Elastic
small strain is linear in `alpha`: one iteration. Plastic steps converge
quadratically; in the unit tests a converged step takes at most 3 local
iterations, a reversal to six times the strain in one increment 3 to 12,
and a reversal of 2 % logarithmic strain to 12 to 30 % 9 to 18 (four laws,
5 and 7 thickness points; of 48 such cases 1 still fails, Chaboche on the
2 x 2 x 2 rule at ten times, against 18 before the shifted direction).
Everything runs on fixed-size Eigen types (6 x 33 operators, the 9 x 9,
24 x 9, 24 x 24 blocks) in a per-thread workspace: the local iteration
allocates nothing.

`NonlinearSystem` keeps the committed `alpha` of every element, elastic ones
too, and commits it with the plastic states; the stress output evaluates
the points at the committed parameters (the converged ones), so element
stresses and strains are those of the enhanced `F = I + H`. The forming
analysis carries the parameters in its restartable `AnalysisState`
(`internal`, per element); a state of the other formulation is refused. The
committed parameters are only the start of the local iteration, not
history: a restart without them finds the same state to the local
tolerance, from further away.

**Mean dilatation** (E-bar) is not combined with the modes: `all` is
refused and `auto` does not apply it to the incompatible-mode Hex8.
Averaging the dilatation of the augmented element leaves a zero-energy
mode (the field `u = k (x z, y z, (z^2 - x^2 - y^2)/2)`, whose strain `k z I`
averages to nothing), and averaging only the compatible dilatation leaves
a sheet one element thick with a fraction of its bending stiffness. The
modes relax the isochoric constraint of plastic flow themselves: they
absorb the linear part of every point's dilatation, which is the part that
locks.

**Hourglassing in compression.** The enhanced elements of finite
deformation can lose stability under large compression although the
material and the compatible element are stable (Wriggers and Reese 1996,
*A note on enhanced strain methods for large deformations*, CMAME 135). A
plastic material hourglasses early, since its tangent keeps little
deviatoric stiffness against the geometric stress term: on a J2 unit cube
(`E = 70 GPa`, `sigma_y = 100 MPa`, logarithmic strains) in homogeneous
uniaxial compression, the condensed tangent of the compression test loses
definiteness in a spurious hourglass mode between the stretches 0.80 and
0.79 without hardening (about 21 % compression, plastic strain 0.23) and
between 0.62 and 0.61 with `H = 300 MPa`, while the standard Hex8's stays
positive definite to 0.55; before that its smallest eigenvalue is 3
(hardening) to 20 (none) times below the standard element's (unit test).
The onset thus depends on the hardening left, which saturating (Voce)
hardening runs out of in forming. Springback (small elastic strains) is far
from it; a formed part in in-plane compression of some 20 % (a flange, a
wrinkle, a tool's contact zone) is not, and should be checked against the
standard element there.

**Limits.** The Taylor-corrected modes are exact in pure bending on
affine cells (boxes, parallelepipeds) and lose much of their effect on
trapezoidal ones: the MacNeal-Harder beam on trapezoidal cells gives 5 % of
the reference deflection in plane and 3 % out of plane (MacNeal 1987, *A
theorem regarding the locking of tapered four-noded membrane elements*,
IJNME 24); curved walls of cones and pyramids are of that kind.

### A sheet's rule through its thickness

`model.integration.thickness_points` (1 to 7) with
`thickness_direction` (`x`, `y` or `z`, default `z`) gives every Hex8,
standard or with incompatible modes, the `n x n x m` Gauss rule
(`gauss_legendre_box`; `n = stiffness_points`, `m = thickness_points` along
the natural axis that the structured generator aligns with that global
axis; a mesh whose elements have another axis across the sheet is warned
about). The elastic-plastic stress through a bent sheet is kinked where
the plastic zones begin, which two Gauss points a layer integrate badly:
springback is sensitive to it (Wagoner and Li 2007, *Simulation of
springback: through-thickness integration*, Int. J. Plasticity 23). The
rules of 5 to 7 points come from Newton's iteration on the Legendre
recurrence; equal orders give the cube rule point for point and bit for
bit. The history of the non-linear analysis (one plastic state per point),
the stress-evaluation points and every consumer follow the rule, in the
point order of the other rules (`zeta` the slowest index, `xi` the fastest).
The integration of the moment of a rigid-plastic shell outside an elastic
core of a quarter of the half-thickness (bending to `4 k_y`) is short by
3.5 % with 5 points and 0.9 % with 7; the springback study measured 3.5 %
and 0.9 % against beam theory (`docs/verification.md`, section 27, in the
history at commit 7d571e6).

## 3. Assembly

`fem/Assembler.cpp` builds the sparse global matrices,

```
  K = sum_e  s_e  A_e^T K_e^0 A_e            M = sum_e  m_e  A_e^T M_e^0 A_e
```

`A_e` is the (implicit) DOF gather operator, and `s_e`, `m_e` are optional
per-element scale factors, 1 when omitted: the explicit integrator scales
the mass of selected elements (mass scaling, `docs/forming.md`, section 7.1).

On a **uniform structured mesh** all cells are geometrically identical, so
`K_e^0` and `M_e^0` are integrated once and reused. The generic per-element
path is always available, and the test suite asserts the two produce the
same matrices.

The **sparsity pattern** depends only on the mesh: one `3 x 3` block for
every pair of nodes that share an element. It is built once, together with
the position of each element's node blocks in it, and every later assembly
scatters the scaled element matrices straight into the value array, in
element order, which is the order a triplet assembly sums duplicates in -
so the two paths give bitwise identical matrices, without the sort and the
triplet storage. The tangent of a deformed state is assembled into the same
pattern, and its free-free block is extracted through an index map cached
for the current constraint partition. A zero scale factor on some element
falls back to the triplet path, whose pattern would differ.

## 4. Boundary conditions and the reduced system

Dirichlet conditions are applied by **partitioning**, not by penalty terms
(`fem/DofManager.cpp`, `fem/Forming.cpp`). Each step of the forming analysis
splits the DOFs into free `f` and prescribed `p` (its constraints,
`docs/forming.md`, section 1.1), sets the prescribed DOFs to their values,
and solves for the free ones only: Newton's method on

```
  R_f(u) - (1 - s) R_0f = 0 ,       R = f_int(u) - f_c(u, t) ,
  K_T,ff du_f = -(R_f - (1 - s) R_0f)
```

(section 6). The entries of the residual `R` at the prescribed DOFs are the
support reactions, so they come with no second assembly, and `K_T,ff` takes
no artificial large diagonal entries that would wreck its conditioning.

## 5. Linear solvers

The forming analysis factorises the free-free tangent directly at every
Newton iteration (`fem/Forming.cpp`): the sparsity pattern is analysed once
per Dirichlet partition, and each iteration scatters the values into it
through a cached index map and refactorises numerically.

| Factorisation | Implementation | Use |
|---------------|----------------|-----|
| supernodal Cholesky | CHOLMOD (SuiteSparse) | a symmetric tangent, when SuiteSparse is built in |
| `LDL^T` | Eigen `SimplicialLDLT` | a symmetric tangent without SuiteSparse, or after a failed Cholesky (a tangent that is not positive definite) |
| LU | UMFPACK (SuiteSparse), else Eigen `SparseLU` | a non-symmetric tangent (friction, a recovering backstress), or after `LDL^T` meets a vanishing pivot |

`"solver": "eigen"` in the `forming` block selects Eigen's factorisations
in a SuiteSparse build. `docs/forming.md`, section 1.5, gives the build
option and what the summary records about the factorisations. An explicit
step factorises nothing (section 8).

## 6. Geometrically non-linear statics

`fem/TotalLagrangian.cpp`, `fem/Elastoplastic.cpp`,
`fem/NonlinearSystem.hpp`. The implicit steps of the forming analysis are
quasi-static, with large displacement and rotation written on the reference
configuration (total Lagrangian). With `H = grad_X u` the displacement
gradient, `F = I + H` the deformation gradient and
`E = (H + H^T + H^T H) / 2` the Green-Lagrange strain,

```
  R(u, t)  = f_int(u) - f_c(u, t) = 0 ,
  f_int,e  = integral B_NL^T S dV0 ,       S: second Piola-Kirchhoff, section 7
  K_T,e    = integral ( B_NL^T D_T B_NL + G^T S G ) dV0 ,
```

`f_c` the contact forces of the tools (`docs/forming.md`, section 1.2), where
`B_NL` maps nodal displacement variations to `delta E` (its rows are
`F_ki N_a,i` on the normal components, `F_ki N_a,j + F_kj N_a,i` on the
engineering shears), `D_T = dS/dE` is the consistent material tangent, and
the second term is the initial-stress stiffness with the current `S`, `G`
the matrix of shape-function gradients. The element's stiffness quadrature
integrates everything.

**Materials.** Every element takes the return of section 7: `sparlab_form`
models elastoplastic sheets, so under `finite` and `finite_logarithmic`
kinematics the material of every element - the deck's where no material
region replaces it, and each region's - needs a `plasticity` block, and an
element whose material has none is refused when the analysis is built.
(The elastic large-strain laws, Saint Venant-Kirchhoff and neo-Hookean,
and their deck key `forming.material_model` were removed.)
The elastic part of the return is linear elasticity in the Green-Lagrange
strain, `S = D E_e` - large rotation, small elastic strain - and Hencky's
law in the logarithmic strain with `finite_logarithmic`.

**Kinematics.** `finite` is the formulation above; `finite_logarithmic`
takes the logarithmic strain for the plastic law (section 7);
`small_strain` keeps the linear strain on the undeformed geometry, with no
geometric stiffness - the one kinematics that takes an elastic material,
where it reproduces the linear element (unit-tested); with plastic ones it
is the classical small-strain elastoplastic analysis.

**Newton's method.** Each increment of a forming step solves its residual
by Newton's method with the consistent tangent - the elements, the contact
and, with friction, its non-symmetric part - and an energy line search:
`g(a) = du . R_f(u + a du)` is the derivative of the potential along the
correction `du`; the full step stays when `|g(1)| <= 0.8 |g(0)|`
(Crisfield's slack), else regula falsi on `g` finds a shorter one, no
shorter than a tenth. The norm of the residual would be the wrong measure:
in a slender structure a good Newton step stirs up large axial forces, and
backtracking on `||R||` stalls Newton into a linear crawl. A direction of
ascent, as a non-symmetric or indefinite tangent can give, takes the full
step. The increments, the predictor, the convergence test with its
round-off floor and the cuts of a failing increment are in
`docs/forming.md`, section 1.4.

## 7. Plasticity

`material.plasticity` makes a material elastoplastic: J2 (von Mises)
plasticity with isotropic and kinematic hardening, integrated at every
integration point by the backward-Euler radial return of Simo and Hughes
(*Computational Inelasticity*, 1998, boxes 3.1 and 3.2), with its consistent
tangent (`src/material/Plasticity.cpp`, `src/fem/Elastoplastic.cpp`).

**The law.** The strain splits additively, `eps = eps_e + eps_p`, the stress
is `sigma = K tr(eps_e) I + 2 G dev(eps_e)`, and with the relative stress
`xi = dev(sigma) - beta` the yield function is

```
  f = ||xi|| - sqrt(2/3) sigma_y(a) <= 0 ,
  sigma_y(a) = sigma_y0 + H a + Q (1 - exp(-delta a)) ,
```

`||.||` the tensor norm. The flow is associative, `d eps_p = d gamma n` with
`n = xi / ||xi||`; the accumulated plastic strain grows by
`da = sqrt(2/3) d gamma` (in uniaxial tension it is the plastic strain) and
the back stress by `d beta = (2/3) H_kin d gamma n` (Prager). Every point
carries the full 3-D state; Voigt strains have engineering shears, stresses
tensorial components.

**The return.** From the internal variables of the last converged step, the
trial deviatoric stress `s_tr = 2 G dev(eps - eps_p,n)` and
`xi_tr = s_tr - beta_n` either stay inside the surface - an elastic step -
or return onto it along `n = xi_tr / ||xi_tr||`, which the return does not
change (radial return). The multiplier solves

```
  g(dg) = ||xi_tr|| - (2 G + 2/3 H_kin) dg - sqrt(2/3) sigma_y(a_n + sqrt(2/3) dg) = 0 ,
```

in closed form for linear hardening and by Newton for Voce saturation (`g`
is convex and decreasing there, so Newton from `dg = 0` rises monotonically
onto the root), to `1e-13` of the radius `sqrt(2/3) sigma_y` - or, far
outside the surface, where one ulp of `||xi_tr||` exceeds that (beyond
about 450 radii: a deviatoric trial increment of order one, which the
Newton iterates of a large-strain run reach), to the round-off of
`||xi_tr||` itself (`8 eps ||xi_tr||`). Then `sigma = sigma_tr - 2 G dg n`,
`eps_p = eps_p,n + dg n`, `beta = beta_n + (2/3) H_kin dg n`. On a path along
which `n` keeps its direction - uniaxial stress, for one - the return is
exact for linear hardening and any step: the unit tests compare it with the
closed-form uniaxial curves to `1e-12`. The consistent tangent, the
derivative of the returned stress,

```
  C = K 1 x 1 + 2 G theta (I - 1 x 1 / 3) - 2 G theta_bar n x n ,
  theta = 1 - 2 G dg / ||xi_tr|| ,
  theta_bar = 1 / (1 + (sigma_y'(a) + H_kin) / (3 G)) - (1 - theta) ,
```

keeps Newton's quadratic convergence; it is symmetric (associative flow) and
checked against central differences of the return (`1e-6`). A point that
yielded in its last converged step and still sits on its surface returns
elastically at zero increment - but with the continuum elastoplastic
tangent (`dg = 0` in `C`), so that the next increment's first iteration
already sees the softer response of continued loading.

**Hill48 anisotropy and Chaboche kinematic hardening.** With
`yield_criterion: hill48` or `backstresses` the return is a closest-point
projection (Simo and Hughes, box 3.4) of the general law
(`src/material/Plasticity.cpp`, `return_general_3d`; the header
`Plasticity.hpp` carries the full derivation):

```
  f = sqrt(xi^T P xi) - sigma_y(a) <= 0 ,   xi = sigma - sum_i alpha_i ,
  d eps_p = d lambda m ,   m = P xi / sigma_bar ,   da = d lambda ,
  d alpha_i = (2/3) C_i d eps_p - gamma_i alpha_i d lambda ,
```

P the Hill48 matrix in global axes, `T^T P_mat T` with `P_mat` built from
`F, G, H, L, M, N` in the frame RD, TD, ND and `T` the rotation of tensorial
Voigt stresses, found once when the material is set (von Mises is
`F = G = H = 1/2, L = M = N = 3/2`). `m` is the engineering form of the flow
direction (the derivative by Voigt components doubles the shears); its
normal part sums to zero, so the flow is isochoric and mean dilatation stays
valid. `lambda` is the work-conjugate accumulated strain,
`xi : d eps_p = sigma_bar d lambda`, which for von Mises is the radial
return's `sqrt(2/3) dg`. Over a step with the direction of its end each
backstress is `alpha_i = theta_i alpha_i,n + (2/3) C_i chi_i m_t` - the
exponential `theta = exp(-gamma dl)`, `chi = (1 - theta) / gamma`, exact for a
fixed direction, or backward Euler `theta = 1 / (1 + gamma dl)`,
`chi = theta dl` - and, the elasticity being isotropic and `m` deviatoric,
`xi = xi_tilde(dl) - b(dl) m_t` with `xi_tilde = s_tr - sum theta_i alpha_i,n`
and `b = 2 G dl + (2/3) sum C_i chi_i`: the backstresses drop out and Newton
solves seven equations for `(xi, dl)`,

```
  R1 = xi - xi_tilde + b S^-1 P xi / sigma_bar = 0 ,   R2 = sigma_bar - sigma_y(a_n + dl) = 0 ,
```

from the trial point with a backtracking line search, to `1e-13 sigma_y` -
tight enough that the stress output's return from the committed state is
elastic. Differentiating the converged equations by the strain, which enters
only through `s_tr = 2 G dev(eps)` (the linear map `D_dev`), gives
`[A; a^T] = J^-1 [D_dev; 0]` with J the Newton Jacobian, and the consistent
tangent

```
  C = C_e - 2 G [ m_t a^T + (dl / sigma_bar) S^-1 (P - m m^T) A ] ,
```

symmetric for linear kinematic hardening and non-symmetric as soon as a
backstress recovers; the element then keeps it unsymmetrised and the solver
factorises it by LU. At `dl = 0` it is the continuum tangent the loading flag
uses. Von Mises without backstresses keeps the radial return above, bit for
bit; the general return with `r = 1` reproduces it (stress, tangent, state) to
`1e-12`. Hill48 with isotropic hardening follows the exact directional
hardening curve and r-value in uniaxial stress along any direction of the
sheet, and von Mises with backstresses the exact Armstrong-Frederick branch
solutions - monotonic, reversed and cyclic - at any step size, since both
paths keep their flow direction; backward Euler converges to them at first
order (unit tests).

**The element.** With small strain the point strain is `eps = B u_e`, the
internal force `int B^T sigma dV` and the tangent `int B^T C B dV`. Plastic
flow is isochoric, and on an element with several integration points each
point's dilatation becomes a constraint: the fully integrated Hex8 then
locks. The mean-dilatation operator of Hughes (1980) replaces every point's
dilatation by the element's volume average,
`B_bar = B + (1/3) m (b_bar - b)^T` with the dilatation row `b = B^T m` and
`m = {1, 1, 1, 0, 0, 0}`. It is the default for the standard Hex8
(`mean_dilatation: auto`).

**Finite kinematics.** With `kinematics: finite` the return takes the
Green-Lagrange strain `E` for the strain and gives the second
Piola-Kirchhoff stress `S`: `E = E_e + E_p` with `S = D E_e` (linear
elasticity in the Green-Lagrange strain, the Saint Venant-Kirchhoff form),
exact under any rigid rotation and meant for strains that stay small (at
large displacement and rotation with small strain the materially non-linear
relations may be written between `S` and `E`; Bathe, *Finite Element
Procedures*, 1996, ch. 6). The element takes `B_NL` of section 6,
`f = int B_NL^T S dV_0`, `K = int B_NL^T C B_NL dV_0 + int (G^T S G) x I dV_0`.
Mean dilatation acts on the Green strain,
`E_bar = E + (1/3) m (mean(tr E) - tr E)`; its second variation adds
`(1/3) tr(S) (mean(G^T G) - G^T G) x I` to the geometric stiffness, and the
tangent stays symmetric and consistent (checked by central differences with
a rigid turn of 0.5 rad). Objectivity - a rigidly rotated state returns the
same `S`, plastic state and energy, and its forces turn with it - is
unit-tested. The model is not finite-strain plasticity: beyond a Green
strain of 0.05 the run warns - large strains take the logarithmic
kinematics below. A large rotation superposed on plastic flow also sharpens
a real effect: under a large hydrostatic pressure (a 3 % volume change,
5 GPa) the geometric stiffness outweighs a small plastic tangent and a
homogeneous state loses stability - at strains well beyond the model's
range.

**Logarithmic-strain finite plasticity.** With `kinematics:
finite_logarithmic` the strains may be large (Miehe, Apel and Lambrecht,
*Anisotropic additive plasticity in the logarithmic strain space*, CMAME
191, 2002; `src/material/LogarithmicStrain.cpp`). Total Lagrangian as
above, but the return - the same J2, Hill48 and Chaboche return,
unchanged - takes the logarithmic (Hencky) strain

```
  E_log = ln(C) / 2 = sum_a (ln lambda_a / 2) N_a N_a ,   C = F^T F = I + 2 E ,
  tr E_log = ln J ,
```

from the spectral decomposition of E (`lambda_a = 1 + 2 mu_a`, and
`ln(lambda_a) / 2 = log1p(2 mu_a) / 2`, which keeps the relative accuracy of
E at a strain of `1e-6`), and gives the stress T conjugate to it with its
algorithmic tangent `C_alg = dT/dE_log`. The additive split
`E_log = E_log,e + E_log,p` with `T = D E_log,e` is exact for a coaxial
homogeneous stretch - the log strains of successive coaxial stretches add -
so uniaxial J2 at any strain is the 1-D small-strain law in the log strain,
with T the Kirchhoff stress. The flow stays isochoric (`det F_p = 1`), the
Hill48 axes are those of the reference configuration and turn with the
material (no plastic spin of the texture), and the model is objective: C,
and so everything but the push-forward, ignores a superposed rotation.

With `dE_log = P dE` (`P = 2 dE_log/dC`) and `dE = B_NL du`:

```
  S = P^T T ,   f = int B_NL^T S dV_0 ,
  K = int B_NL^T (P^T C_alg P + T : L) B_NL dV_0 + int (G^T S G) x I dV_0 ,
```

`T : L` the Hessian of `T : E_log(E)` at fixed T (`L = 4 d2E_log/dC dC`,
the sixth-order term). By the Daleckii-Krein formulas, in the principal axes
of C,

```
  [dE_log]_ab = l[lambda_a, lambda_b] [dE]_ab ,
  T : d2E_log[dE1, dE2] = 2 sum_abc T_ab l[lambda_a, lambda_c, lambda_b]
                            ([dE1]_ac [dE2]_cb + [dE2]_ac [dE1]_cb) ,
```

with the divided differences of ln, `l[x, y] = (ln x - ln y)/(x - y)` and
`l[x, y, z] = (l[x, y] - l[y, z])/(x - z)`, `l[x, x] = 1/x`,
`l[x, x, x] = -1/(2 x^2)`. They carry the whole treatment of equal and nearly
equal eigenvalues: `l[x, y]` is evaluated as `log1p((x - y)/y)/(x - y)`,
exact to round-off at any gap and at none; `l[x, y, z]` as the difference
quotient when the spread `x - z` exceeds `5e-3 y`, and otherwise as the
Taylor series about the middle eigenvalue to fifth order,
`y^-2 sum_n (-1)^(n+1) h_n(u, w)/(n + 2)` with `u = (x - y)/y`,
`w = (z - y)/y` and `h_n` the complete homogeneous symmetric polynomials -
the threshold balancing the cancellation of the quotient (about
`4 eps / 5e-3 = 2e-13`) against the truncation of the series
(`(5e-3)^6 = 2e-14`), so that distinct, two equal, three equal and nearly
equal eigenvalues are all evaluated to about `1e-13` and continuously.
Checked against an independent matrix logarithm (Schur-Pade, `4.8e-15`) and
against fourth-order central differences of `E_log(E)` at distinct
eigenvalues, two equal, `C = I`, `C = s^2 I`, and gaps of `1e-9`, `1e-7`,
`1e-4` (two and three eigenvalues) and on both sides of the series threshold
(`P` to `4.1e-12`, `T : L` to `4.3e-12`: the differences' own accuracy).

The elasticity is Hencky's, quadratic in `E_log`.

`E_log` sees only `C = F^T F`, which is positive definite at an inverted
point (`det F < 0`) too, where `E_log`, the energy and `tr E_log =
ln |det F|` are those of a valid state with one axis reflected - an
equilibrium the solver would otherwise accept. Each point therefore checks
the sign of `det F`, and an inverted one fails the evaluation: the line
search shortens a step into it and an increment
that ends there is cut, so a block crushed to zero volume stops short of it
instead of converging on the far side.

*Mean dilatation* acts on the logarithmic volumetric strain `ln J`: each
point's `ln J` is replaced by the element's volume average,
`E_log_bar = E_log + (1/3) m (mean(ln J) - ln J)`, exactly the log strain of
the modified deformation `F_bar = (J_bar/J)^(1/3) F`, so it stays objective.
With `b = m^T P B_NL` the variation of `ln J` and `B_bar = P B_NL +
(1/3) m (mean(b) - b)`, the force is `int B_bar^T T` and the consistent
tangent

```
  K = int [ B_bar^T C_alg B_bar + B_NL^T (T' : L) B_NL + (G^T S' G) x I ] dV_0 ,
  T' = dev T + p_bar m ,   S' = P^T T' ,
```

`p_bar` the element's mean Kirchhoff pressure: the second variation of
`int T : E_log_bar` is that of `int (T - p m) : E_log` plus `int p` times the
second variation of `mean(ln J)`, and both are initial-stress terms because
`ln J = m : E_log`.

The element output of the forming analysis (`von_mises_Pa`) is the von
Mises stress of the points' average Cauchy stress, `tau / J_bar` with the
Kirchhoff stress `tau = F S F^T` (with the modified `F_bar`,
`F_bar S_bar F_bar^T`, the same) and `J_bar = exp(tr E_log_bar)`.

Uniaxial tension to a stretch of 2 with linear and Voce hardening, a
tension-compression cycle to `+-0.3` log strain with Armstrong-Frederick
backstresses, and Hill48 pulled along RD and TD to a stretch of 1.6 (whose
plastic lateral log strains keep the ratio `r` exactly) were exact to
`1.2e-13` on a distorted Hex8 patch (`docs/verification.md`, section 24, in
the history at commit 7d571e6). The element tangent is the central
difference of the internal force for the Hex8, elastic and plastic, J2,
Hill48 and Chaboche, with and without mean dilatation, at a stretch of 1.35
and a turn of 0.6 rad, and at strains of `1e-6` the element agrees with the
Green-Lagrange one to the order of the strain (unit tests).

**History.** Each point keeps `(eps_p, beta, a)` of the last converged step.
Every Newton iterate, line-search probe and rejected increment returns from
those, never from the last iterate, and only a converged increment commits
the new ones - so halving an increment leaves no trace, and the path is the
sequence of converged increments.

## 8. Explicit dynamics

A `form_explicit` step integrates `M a + alpha M v + f_int(u) = f_c(u, v, t)`
by central differences, with the lumped mass of section 2 (scaled per
element by mass scaling), the internal forces `f_int` of sections 6 and 7,
and a stable time step bounded through the element eigenvalues of the
linear element stiffness of section 2. `docs/forming.md`, section 7.1, and
`include/sparlab/fem/ExplicitDynamics.hpp` state the formulation in full.
