/// \file Hyperelastic.hpp
/// \brief Material laws of the large-deformation (total Lagrangian) analysis:
///        the second Piola-Kirchhoff stress and its tangent as functions of
///        the deformation gradient.
///
/// Both laws derive from a strain energy density \f$W\f$ per unit reference
/// volume, so \f$S = \partial W/\partial E\f$ and the tangent
/// \f$\mathbb{C} = \partial S/\partial E\f$ is symmetric, with the Green-Lagrange
/// strain \f$E = \tfrac12(F^T F - I)\f$ and \f$C = F^T F\f$.
///
/// * **Saint Venant-Kirchhoff**: \f$S = D\,(E - E_\theta)/\vartheta\f$ with
///   the same constitutive matrix \f$D\f$ as the linear analysis (plane
///   stress, plane strain or 3-D). A temperature change enters through the
///   multiplicative split \f$F = F_e\,\vartheta I\f$ with the thermal stretch
///   \f$\vartheta = 1 + \alpha\Delta T\f$ (Lu and Pister, 1975): the thermal
///   strain \f$E_\theta = \alpha\Delta T\,(1 + \alpha\Delta T/2)\f$ in each
///   direction is the Green-Lagrange strain of the free stretch, so a freely
///   heated body takes exactly \f$\vartheta\f$ with no stress, and the elastic
///   energy \f$\tfrac12 E_e : D : E_e\f$ is counted per unit volume of the
///   expanded, stress-free state, so that the material keeps its modulus
///   there: \f$W = \vartheta^3\,\tfrac12 E_e : D : E_e\f$ per reference
///   volume with \f$E_e = (E - E_\theta)/\vartheta^2\f$. (Subtracting
///   \f$E_\theta\f$ from \f$E\f$ alone would stiffen a heated material by
///   \f$\vartheta\f$.) To first order in \f$\alpha\Delta T\f$ this is the
///   linear analysis's thermal strain; \f$E_\theta\f$ is IsotropicMaterial's
///   linear thermal strain at the effective change
///   `thermal_green_lagrange_change`.
///   It is linear elasticity written in the Lagrangian strain: exact for
///   large rotations, and the right model for large deflection with small
///   strain - a slender beam, a plate, a shell-like part. Under strong
///   compression (a stretch below about 0.58 in uniaxial compression) its
///   stress falls again and the model is unsuitable; it is also what
///   CalculiX's linear elastic material becomes under NLGEOM.
/// * **Compressible neo-Hookean**:
///   \f$W = \tfrac{\mu}{2}(\mathrm{tr}\,C - 3) - \mu\ln J + \tfrac{\lambda}{2}(\ln J)^2\f$,
///   \f$S = \mu(I - C^{-1}) + \lambda\ln J\,C^{-1}\f$,
///   \f$\mathbb{C}_{ijkl} = \lambda C^{-1}_{ij}C^{-1}_{kl} +
///   (\mu - \lambda\ln J)(C^{-1}_{ik}C^{-1}_{jl} + C^{-1}_{il}C^{-1}_{jk})\f$,
///   with the Lame constants of the small-strain limit, so both laws agree
///   with linear elasticity for small strains. It stays stable in large
///   compression (\f$W\to\infty\f$ as \f$J\to0\f$) and suits rubber-like
///   large strain. It is available for plane strain (\f$F_{33} = 1\f$) and
///   3-D; plane stress would need the thickness stretch as an extra unknown
///   and is refused, as is a thermal strain.
///
/// Voigt ordering follows IsotropicMaterial (\f$\{11, 22, 12\}\f$ in 2-D,
/// \f$\{11, 22, 33, 12, 23, 31\}\f$ in 3-D) with engineering shear strain,
/// so the tangent's entries are \f$\mathbb{C}_{ijkl}\f$ for the index pairs of
/// the row and the column.
#pragma once

#include "sparlab/core/Types.hpp"
#include "sparlab/material/IsotropicMaterial.hpp"

#include <string>

namespace sparlab {

enum class HyperelasticModel {
  SaintVenantKirchhoff,  ///< S = D (E - E_0)
  NeoHookean             ///< compressible neo-Hookean, ln J form
};

/// "saint_venant_kirchhoff" or "neo_hookean".
/// \throws ConfigError for any other name.
HyperelasticModel parse_hyperelastic_model(const std::string& text);

/// Response of a law at one deformation.
struct HyperelasticResponse {
  Vector stress;       ///< second Piola-Kirchhoff stress S, Voigt [Pa]
  Matrix tangent;      ///< dS/dE, Voigt with engineering shear [Pa]
  Scalar energy = 0.0; ///< strain energy density W per reference volume [J/m^3]
  /// Plane strain: the out-of-plane second Piola-Kirchhoff stress S_33 [Pa].
  Scalar s33 = 0.0;
  /// Plane stress: the thickness stretch l_3 = sqrt(1 + 2 E_33) that S_33 = 0
  /// implies, with E_33 = [(1 + nu) e_th - nu (E_11 + E_22)] / (1 - nu) for
  /// the Saint Venant-Kirchhoff law; 1 otherwise.
  Scalar thickness_stretch = 1.0;
};

/// Evaluate `model` for material `m` in idealisation `state` at the
/// displacement gradient \f$H = \partial u/\partial X\f$ (dim x dim), so
/// \f$F = I + H\f$ (in plane strain the out-of-plane stretch is 1).
/// `delta_t` is the temperature change (Saint Venant-Kirchhoff only; its
/// thermal strain is `m.thermal_strain`).
///
/// The laws take \f$H\f$ rather than \f$F\f$ because
/// \f$E = \tfrac12(H + H^T + H^T H)\f$ keeps full relative precision at small
/// strain, where \f$\tfrac12(F^T F - I)\f$ - and forming \f$F = I + H\f$ at
/// all - rounds a strain of \f$10^{-6}\f$ to ten significant digits and
/// floors the residual a Newton iteration can reach. For the same reason
/// \f$I - C^{-1}\f$ is formed as \f$2C^{-1}E\f$ and \f$J - 1\f$ from the
/// invariants of \f$H\f$.
/// \throws ConfigError for a neo-Hookean law in plane stress or with a
///         temperature change; SolverError for det F <= 0 under the
///         neo-Hookean law (an inverted element).
HyperelasticResponse evaluate_hyperelastic(HyperelasticModel model, const IsotropicMaterial& m,
                                           StressState state, const Matrix& grad_u,
                                           Scalar delta_t = 0.0);

/// Green-Lagrange strain of the displacement gradient `grad_u`, Voigt form
/// with engineering shear.
Vector green_lagrange_voigt(const Matrix& grad_u);

/// The temperature change at which IsotropicMaterial's linear thermal strain
/// \f$\alpha\,\Delta T_e\f$ equals the Green-Lagrange strain of the free
/// thermal stretch: \f$\Delta T_e = \Delta T\,(1 + \alpha\Delta T/2)\f$.
Scalar thermal_green_lagrange_change(const IsotropicMaterial& m, Scalar delta_t);

/// The free thermal stretch \f$\vartheta = 1 + \alpha\Delta T\f$.
Scalar thermal_stretch(const IsotropicMaterial& m, Scalar delta_t);

/// Cauchy stress \f$\sigma = J^{-1} F S F^T\f$ (Voigt) from the response of a
/// law at the displacement gradient `grad_u`. In plane strain its `s33`
/// completes the tensor and sigma_zz is returned in `sigma_zz`; in plane
/// stress J includes its thickness stretch.
Vector cauchy_from_piola_kirchhoff(const Matrix& grad_u, const HyperelasticResponse& response,
                                   StressState state, Scalar* sigma_zz = nullptr);

}  // namespace sparlab
