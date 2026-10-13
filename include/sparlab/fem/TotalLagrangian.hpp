/// \file TotalLagrangian.hpp
/// \brief Internal force and consistent tangent of the total Lagrangian
///        formulation, written once for every continuum element.
///
/// Everything is referred to the undeformed configuration. At a quadrature
/// point with reference shape-function gradients \f$G = \partial N/\partial X\f$
/// (dim x n, read off the element's strain operator) and nodal displacements
/// \f$U\f$ (dim x n):
/// \f[
///   F = I + U G^T, \qquad E = \tfrac12(F^T F - I),
/// \f]
/// and the variation of the Green-Lagrange strain is
/// \f$\delta E = B_{NL}\,\delta u\f$ with, for the Voigt component \f$(ij)\f$
/// and the DOF \f$k\f$ of node \f$a\f$,
/// \f[
///   (B_{NL})_{(ii),ak} = F_{ki} G_{ia}, \qquad
///   (B_{NL})_{(ij),ak} = F_{ki} G_{ja} + F_{kj} G_{ia} \quad (i \ne j).
/// \f]
/// Then
/// \f[
///   f_{int} = \int_{\Omega_0} B_{NL}^T S\,t\,dV_0 ,\qquad
///   K_T = \int_{\Omega_0} B_{NL}^T \mathbb{C} B_{NL}\,t\,dV_0
///       + \int_{\Omega_0} (G^T \mathbf{S}\,G) \otimes I\,t\,dV_0 ,
/// \f]
/// the material and the geometric (initial-stress) parts, with \f$S\f$ and
/// \f$\mathbb{C}\f$ from Hyperelastic.hpp. The integration rule is the
/// element's stiffness rule. For a small displacement \f$F \to I\f$,
/// \f$B_{NL} \to B\f$ and \f$K_T\f$ tends to the linear stiffness; a rigid
/// rotation gives \f$E = 0\f$ and no internal force (both tested). In plane
/// stress the thickness is taken as constant: the Saint Venant-Kirchhoff law
/// with the plane-stress \f$D\f$ applied to the in-plane Green strain, exact
/// for large rotations with small strains.
///
/// Where this fits: Simulations stage - one elastic element under finite kinematics (no
///   plastic return), and the gradient helpers fem/Elastoplastic reuses.
/// Called from: fem/NonlinearSystem.hpp:NonlinearSystem::evaluate (total_lagrangian_element),
///   fem/Forming.cpp:FormingAnalysis::run, fem/Elastoplastic.cpp:kinematics_of
/// Calls into: material/Hyperelastic, elements/Element (strain_operator), fem/IncompatibleModes
/// Start reading at: total_lagrangian_element (fem/TotalLagrangian.cpp)
#pragma once

#include "sparlab/core/Types.hpp"
#include "sparlab/fem/FemModel.hpp"
#include "sparlab/material/Hyperelastic.hpp"

namespace sparlab {

/// Shape-function gradients with respect to the reference coordinates,
/// \f$G = \partial N/\partial X\f$ (dim x n), read off the normal-strain rows
/// of an element's strain operator.
Matrix reference_gradients(const StrainOperator& op, int dim, int nodes);

/// The operator \f$B_{NL}\f$ of the variation of the Green-Lagrange strain,
/// \f$\delta E = B_{NL}\,\delta u\f$ (Voigt, engineering shear; nv x dim n),
/// at the deformation gradient `f` (dim x dim) with the reference gradients
/// `g` (dim x n) - the formula of the file comment.
Matrix green_lagrange_operator(const Matrix& f, const Matrix& g);

/// Element contribution of the total Lagrangian formulation.
struct TotalLagrangianElement {
  Vector internal_force;  ///< num_dofs [N]
  Matrix tangent;         ///< num_dofs x num_dofs [N/m]; empty unless requested
  /// Derivative of the internal force with respect to a scale on the
  /// temperature change, \f$\partial f_{int}/\partial s\f$ at \f$\Delta T = s\,(T - T_{ref})\f$
  /// [N]; empty without a temperature.
  Vector thermal_force_rate;
  Scalar energy = 0.0;    ///< \f$\int W\,t\,dV_0\f$ [J]
  /// An element with internal modes: their parameters at this displacement
  /// [m] and the local iterations that found them (Elastoplastic.hpp).
  Vector internal;
  int internal_iterations = 0;
};

/// Evaluate element `e` of `model` at element displacements `ue` (node-major,
/// num_dofs).
/// \param law the material law.
/// \param temperature nodal temperatures [K] or nullptr; the temperature
///        change entering the law is `temperature_scale * (T - T_ref)`.
/// \param want_tangent assemble the tangent as well.
/// \param internal an element with internal modes (the incompatible-mode
///        Hex8): their committed parameters, the start of the local Newton
///        iteration (IncompatibleModes.hpp; null: zero).
/// \throws SolverError when an element inverts under a neo-Hookean law, or
///         the local iteration of internal modes fails.
TotalLagrangianElement total_lagrangian_element(const FemModel& model, Index e,
                                                const Vector& ue, HyperelasticModel law,
                                                const Vector* temperature,
                                                Scalar temperature_scale, bool want_tangent,
                                                const Vector* internal = nullptr);

/// Stresses of element `e` averaged over its stiffness quadrature points:
/// the second Piola-Kirchhoff stress and the Cauchy stress (Voigt), and the
/// out-of-plane Cauchy stress of a plane-strain element.
struct TotalLagrangianStress {
  Vector cauchy;
  Scalar cauchy_zz = 0.0;
  Scalar max_green_strain = 0.0;  ///< largest |E_ij| over the element's points
};

/// With internal modes, `internal` holds their committed parameters.
TotalLagrangianStress total_lagrangian_stress(const FemModel& model, Index e, const Vector& ue,
                                              HyperelasticModel law, const Vector* temperature,
                                              Scalar temperature_scale,
                                              const Vector* internal = nullptr);

}  // namespace sparlab
