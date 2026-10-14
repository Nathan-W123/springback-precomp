/// \file TotalLagrangian.hpp
/// \brief The kinematic operators of the total Lagrangian formulation, written
///        once for every continuum element: the reference gradients, the
///        Green-Lagrange strain and its variation.
///
/// Everything is referred to the undeformed configuration. At a quadrature
/// point with reference shape-function gradients \f$G = \partial N/\partial X\f$
/// (dim x n, read off the element's strain operator) and nodal displacements
/// \f$U\f$ (dim x n):
/// \f[
///   H = U G^T, \qquad F = I + H, \qquad
///   E = \tfrac12(F^T F - I) = \tfrac12(H + H^T + H^T H),
/// \f]
/// and the variation of the Green-Lagrange strain is
/// \f$\delta E = B_{NL}\,\delta u\f$ with, for the Voigt component \f$(ij)\f$
/// and the DOF \f$k\f$ of node \f$a\f$,
/// \f[
///   (B_{NL})_{(ii),ak} = F_{ki} G_{ia}, \qquad
///   (B_{NL})_{(ij),ak} = F_{ki} G_{ja} + F_{kj} G_{ia} \quad (i \ne j).
/// \f]
/// The elastoplastic element (Elastoplastic.hpp) builds its finite and
/// logarithmic kinematics on them: \f$f_{int} = \int B_{NL}^T S\,dV_0\f$ with
/// the geometric stiffness \f$\int (G^T\mathbf{S}\,G)\otimes I\,dV_0\f$. For
/// a small displacement \f$F \to I\f$ and \f$B_{NL} \to B\f$; a rigid
/// rotation gives \f$E = 0\f$.
///
/// Voigt ordering follows Material (\f$\{11, 22, 12\}\f$ in 2-D,
/// \f$\{11, 22, 33, 12, 23, 31\}\f$ in 3-D) with engineering shear strain.
///
/// Where this fits: Simulations stage - the Green-Lagrange kinematics the elastoplastic
///   element and the explicit kernel build on.
/// Called from: fem/Elastoplastic.cpp:kinematics_of,
///   fem/ExplicitKernel.cpp:ExplicitInternalForce::build_dedicated (reference_gradients)
/// Calls into: elements/Element (StrainOperator)
/// Start reading at: green_lagrange_operator (fem/TotalLagrangian.cpp)
#pragma once

#include "sparlab/core/Types.hpp"
#include "sparlab/elements/Element.hpp"

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

/// Green-Lagrange strain of the displacement gradient `grad_u`
/// (\f$H = \partial u/\partial X\f$, dim x dim), Voigt form with engineering
/// shear. It takes \f$H\f$ rather than \f$F\f$ because
/// \f$E = \tfrac12(H + H^T + H^T H)\f$ keeps full relative precision at small
/// strain, where \f$\tfrac12(F^T F - I)\f$ - and forming \f$F = I + H\f$ at
/// all - rounds a strain of \f$10^{-6}\f$ to ten significant digits and
/// floors the residual a Newton iteration can reach.
Vector green_lagrange_voigt(const Matrix& grad_u);

}  // namespace sparlab
