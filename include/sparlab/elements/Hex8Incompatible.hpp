/// \file Hex8Incompatible.hpp
/// \brief Trilinear hexahedron with Wilson-Taylor incompatible modes: the
///        Hex8 that does not lock in bending.
///
/// **Why.** The fully integrated trilinear Hex8 cannot bend: a pure bending
/// field puts a parasitic transverse shear strain \f$\gamma_{xz}\propto x\f$
/// into every element (its sides stay straight), and a stiff lateral strain
/// with the Poisson effect, so an element that is long against its
/// thickness is too stiff by a factor that grows as \f$(L/h)^2\f$ - the
/// shear locking that spoils a sheet with one or two elements through its
/// thickness, and with it the springback, which is the elastic unloading of
/// a bending moment.
///
/// **The modes.** Wilson, Taylor, Doherty and Ghaboussi (1973,
/// *Incompatible displacement models*) add to each displacement component
/// the three bubble fields \f$\phi_k = 1 - \xi_k^2\f$ (\f$\xi_k\f$ = \f$\xi,
/// \eta, \zeta\f$), which vanish at the nodes and are not continuous
/// between elements: nine parameters \f$\alpha\f$ per element, condensed
/// inside it. They supply the quadratic displacements that bending needs.
/// Taylor, Beresford and Wilson (1976, *A non-conforming element for stress
/// analysis*, IJNME 10) evaluate their gradients with the Jacobian of the
/// element centre, scaled by the ratio of determinants,
/// \f[
///   \tilde g_k(\xi) = \frac{j_0}{j(\xi)}\,(-2\xi_k)\,J_0^{-T} e_k ,
/// \f]
/// (\f$J_0\f$, \f$j_0\f$ the Jacobian and its determinant at
/// \f$\xi = 0\f$), so that \f$\int \tilde g_k\,dV = -2 j_0 J_0^{-T} e_k
/// \sum_q w_q \xi_{q,k} = 0\f$ for every rule symmetric in each direction:
/// a constant strain puts no load on the modes, \f$\alpha = 0\f$, and the
/// element passes the patch test on any geometry (the plain Wilson element
/// fails it on a distorted cell).
///
/// **As pseudo-nodes.** The three modes enter as three extra "nodes" with
/// the reference gradients \f$\tilde g_k\f$ (`internal_mode_gradients`):
/// the displacement gradient is \f$H = UG^T + A\tilde G^T\f$ with
/// \f$A\f$ the 3 x 3 matrix of the parameters, and every kinematic
/// operator of the library - the linear B, the Green-Lagrange
/// \f$B_{NL}(F, [G\,|\,\tilde G])\f$, the geometric stiffness
/// \f$([G|\tilde G]^T S [G|\tilde G])\otimes I\f$ - is that of an
/// eleven-node element. Linear: the strain span of the modes is the
/// enhanced-strain field EAS-9 (Andelfinger and Ramm 1993), and the
/// condensed stiffness equals Simo and Rifai's (1990). Finite: this is the
/// enhanced deformation gradient \f$F = I + H\f$ of Simo and Armero
/// (1992, *Geometrically non-linear enhanced strain mixed methods and the
/// method of incompatible modes*, IJNME 33) and Simo, Armero and Taylor
/// (1993, CMAME 110). It is objective (a rigid rotation R of the element
/// is carried by \f$RA\f$). Under large *compression* the enhanced
/// elements can show spurious hourglass instabilities (Wriggers and Reese
/// 1996); the tests and docs/formulation.md record where.
///
/// **Linear stiffness** (this class): with \f$\tilde B\f$ the strain
/// operator of the modes,
/// \f[
///   K^* = K_{uu} - K_{u\alpha}K_{\alpha\alpha}^{-1}K_{\alpha u},\qquad
///   K_{\alpha\alpha} = \int\tilde B^TD\tilde B\,dV,\quad
///   K_{\alpha u} = \int\tilde B^TDB\,dV,
/// \f]
/// evaluated with fixed-size 24 x 24 / 9 x 24 / 9 x 9 blocks. The mass and
/// the loads are those of the Hex8: the modes carry no inertia and no
/// surface load. The non-linear kernels (IncompatibleModes.hpp) solve for
/// the parameters by a local Newton iteration and condense the tangent.
///
/// `strain_operator` stays the compatible B of the nodes, as everywhere
/// in the library (its normal rows are read as the shape-function
/// gradients); the condensed strain is `condensed_strain_operator`.
///
/// Where this fits: Simulations stage - the incompatible-mode Hex8 (element_formulation
///   "incompatible_modes"), the sheet element that bends without shear locking.
/// Called from: elements/Element.cpp:make_element,
///   fem/IncompatibleModes.cpp:set_up (hex8_incompatible_gradients)
/// Calls into: elements/Hex8Kernels, elements/Hex8, elements/Quadrature
/// Start reading at: Hex8IncompatibleElement::stiffness, hex8_incompatible_gradients
#pragma once

#include "sparlab/elements/Element.hpp"
#include "sparlab/elements/Hex8.hpp"

namespace sparlab {

class Hex8IncompatibleElement final : public Element {
 public:
  ElementType type() const override { return ElementType::Hex8; }
  int dim() const override { return 3; }
  int num_nodes() const override { return 8; }
  int num_internal_nodes() const override { return 3; }

  /// The condensed stiffness \f$K^*\f$ of the file comment [N/m].
  Matrix stiffness(const Matrix& coords, const Matrix& d, Scalar thickness,
                   const IntegrationOptions& opts) const override;

  Matrix consistent_mass(const Matrix& coords, Scalar density, Scalar thickness,
                         const IntegrationOptions& opts) const override;

  StrainOperator strain_operator(const Matrix& coords,
                                 const NaturalPoint& point) const override;

  Vector shape_functions(const NaturalPoint& point) const override;

  std::vector<IntegrationPoint> integration_rule(
      const IntegrationOptions& opts) const override;

  /// The Taylor-corrected mode gradients \f$\tilde g_k\f$, 3 x 3 (column k
  /// for \f$\phi_k = 1 - \xi_k^2\f$) [1/m].
  Matrix internal_mode_gradients(const Matrix& coords, const NaturalPoint& point) const override;


 private:
  Hex8Element standard_;  ///< the compatible element: mass, shape functions, rule
};

/// The Taylor-corrected incompatible-mode gradients of a Hex8 at the natural
/// point (xi, eta, zeta) whose Jacobian determinant is `det`, from the
/// inverse Jacobian `centre_inverse` and determinant `centre_det` of the
/// element centre: column k is \f$(j_0/j)(-2\xi_k)\,J_0^{-T}e_k\f$ [1/m].
Matrix3 hex8_incompatible_gradients(const Matrix3& centre_inverse, Scalar centre_det,
                                    Scalar det, Scalar xi, Scalar eta, Scalar zeta);

}  // namespace sparlab
