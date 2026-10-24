# Assumptions and limitations

This is what `sparlab_form` does **not** do, and what its results do **not**
claim. Nothing here is a bug list; every item is a deliberate scope boundary.
The forming analysis's own limits (tools, contact, increments) are in
[`forming.md`](forming.md), section 6; the springback
model's error budget against a real cell is in
[`benchmarks/physics_audit`](../benchmarks/physics_audit/README.md).

## Physics and idealisation

**A solid continuum of Hex8 cells, nothing else.** The sheet is meshed with
eight-node hexahedra (a structured box, or `C3D8` cells from an `.inp` file).
There are no shell, plate or beam elements, so bending is resolved by the
Hex8 through the thickness - one incompatible-mode layer with 5 Gauss points
in the workflow's fine mesh - and the tools are rigid surfaces (sphere,
cylinder, plane), not meshed bodies.

**Plasticity.** J2 or Hill's quadratic 1948 criterion, with isotropic Voce
hardening and kinematic hardening (Prager, or Chaboche's sum of at most four
Armstrong-Frederick backstresses), in small strain, in the Green-Lagrange
strain at large rotation (`finite`), or additive in the logarithmic strain
(`finite_logarithmic`, the workflow's setting; [`formulation.md`](formulation.md)).
What it does not do:

* plasticity is rate-independent: no creep, viscoplasticity, damage,
  fracture, non-associative or pressure-dependent yield, or
  temperature-dependent properties;
* with `finite` kinematics the return in the Green-Lagrange strain is sound
  for large rotation with small strains (the run warns beyond a strain of
  0.05). `finite_logarithmic` is plasticity additive in the logarithmic
  strain (Miehe, Apel and Lambrecht 2002) with Hencky elasticity, exact for
  coaxial stretches of any size and objective. It is not the multiplicative
  `F = F_e F_p` theory: the two agree when the elastic strains are small, as
  in metals, but differ for large elastic strains under rotating principal
  axes; the Hill48 axes stay those of the reference configuration (no plastic
  spin of the texture); the kinematic hardening is additive in the log
  strain;
* the anisotropy is Hill's quadratic criterion only (no Yld2000-2d or other
  non-quadratic criterion, which fit aluminium's yield locus better), with
  isotropic elasticity and a fixed material frame; its out-of-plane shear
  coefficients L and M are not measured by the r-values and default to the
  isotropic 1.5. With r < 1, as for AA5754-O, it under-predicts the
  equibiaxial yield stress (the physics audit's M1 item);
* no ratcheting refinements (Ohno-Wang), cyclic isotropic softening,
  plastic-strain-range memory or Yoshida-Uemori; a recovering backstress
  makes the tangent non-symmetric, which then takes LU;
* the backward-Euler return is first-order accurate on a non-proportional
  path, whatever the law;
* no purely elastic model at large rotation or strain: `finite` and
  `finite_logarithmic` kinematics take elastoplastic materials only (the
  material of every element, after `material_regions`, needs a `plasticity`
  block; an element of elastic material is refused), and an elastic
  material runs with `small_strain` kinematics only, as linear elasticity.
  The elastic large-strain laws (Saint Venant-Kirchhoff, neo-Hookean) and
  their deck key `forming.material_model` were removed.

**No temperature.** The forming analysis is isothermal: no thermal strain,
no heating from deformation or friction, no heat treatment after forming.
The material keys `thermal_expansion`, `reference_temperature` and
`conductivity`, which were read but had no effect, were removed in the
cleanup: a deck that sets them is refused under `--strict-config` and warned
about otherwise.

**Locking and the Hex8 with incompatible modes.** A fully integrated
standard Hex8 locks under the isochoric flow of a fully plastic state
without the mean dilatation that is its default; a locked state comes out
too stiff. `model.element_formulation: "incompatible_modes"` removes the
Hex8's parasitic bending stiffness (one layer through the thickness bends
like Timoshenko on affine cells). Its limits:

* the modes are exact in bending on affine cells only; on trapezoidal cells
  they lose most of their effect, and so on the tapered cells of a curved
  wall; there is no assumed-natural-strain or solid-shell element (Hauptmann
  and Schweizerhof 1998, Schwarze and Reese 2009) and no EAS-21;
* the finite-deformation element loses stability in a spurious hourglass
  mode under compression, where the standard Hex8 is stable (Wriggers and
  Reese 1996): a J2 cube at about 21 % compression without hardening, 39 %
  with `H = 300 MPa` (`E = 70 GPa`, `sigma_y = 100 MPa`) - beyond
  springback, but not beyond a compressed flange or a tool's contact zone;
* the local iteration for the nine mode parameters can fail under
  logarithmic strains far from the solution (a hard increment's trial
  states), and the step is then cut ([`forming.md`](forming.md));
* it is not combined with mean dilatation (`auto` leaves the modes without
  it, `all` is refused).

`model.integration.thickness_points` gives a Hex8 sheet 1 to 7 Gauss points
through each layer. The springback of a strip bent to four times its
first-yield curvature is 3.5 % short of beam theory with 5 points and 0.9 %
with 7 on one incompatible-mode layer (the Gauss quadrature of the kinked
stress; the study is in the history at commit `7d571e6`). The rule refines
the elements' natural axis, so a mesh from a file must have it across the
sheet (warned about otherwise).

**Meshes.** `structured_hex` boxes, or an Abaqus / CalculiX `.inp` file with
`C3D8` cells; one cell type per mesh; Abaqus parts must be flat (no instances
with translations, one part); supports, loads and materials in the file are
ignored in favour of the deck; SparLab does not mesh, and there is no
adaptive refinement.

## Numerics

**Direct sparse factorisations.** Every implicit increment refactorises the
tangent (CHOLMOD, or Eigen's `SimplicialLDLT`; UMFPACK or `SparseLU` for a
non-symmetric tangent; [`forming.md`](forming.md), section 1.5). The cost
therefore grows with the mesh faster than linearly; a run on the
workflow's fine mesh takes about 45-90 minutes on one core.

**Mechanism modes are not detected analytically.** An unsupported or
internally mobile model surfaces as a singular factorisation, which the
solver reports with a hint, not as a silent answer.

## Verification and validation

**Verification is by tests and regression, validation is narrow.** The C++
tests (`ctest`) check the element, material-point and forming code against
exact answers and derivative checks (patch tests, tangents against central
differences of the internal force, homogeneous deformations, restarts that
hand state between steps bit for bit), and the Python integration tests run
every deck variant `precomp` writes through the real executable
([`precomp.md`](precomp.md), "Verification"). The forming analysis is
cross-checked against CalculiX where both solve the same problem (released
shapes agree to 2 um) and against a published experiment whose replica is
not converged enough to judge
([`benchmarks/physics_audit`](../benchmarks/physics_audit/README.md)). There
is **no comparison against a part formed and scanned in a real cell**.

## Reporting

**Runtimes are machine-specific.** Every reported time comes from one machine
and one build, both recorded in the result summaries; absolute values will
differ elsewhere, and several were measured with other jobs sharing the
machine.
