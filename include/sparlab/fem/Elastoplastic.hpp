/// \file Elastoplastic.hpp
/// \brief One element of an elastoplastic analysis: internal force,
///        consistent tangent and updated internal variables from the return
///        of Plasticity.hpp (J2, or Hill48 and Chaboche) at each integration
///        point, with small-strain, finite (total Lagrangian, Green-Lagrange
///        strain) or logarithmic-strain finite kinematics.
///
/// **Small strain.** With the linear strain operator \f$B\f$ (the element's
/// own, in the 3-D Voigt order - a plane model's out-of-plane row carries the
/// plane-strain or mean-dilatation \f$\varepsilon_{33}\f$), the strain at a
/// point is \f$\varepsilon = Bu_e\f$ and
/// \f[
///   f_e = \int B^T\sigma\,dV ,\qquad K_e = \int B^T C\,B\,dV ,
/// \f]
/// with \f$\sigma\f$ and the consistent tangent \f$C\f$ of the return from
/// the last converged state.
///
/// **Finite kinematics** (large displacement and rotation, small strain).
/// The return takes the Green-Lagrange strain \f$E\f$ for the strain and
/// gives the second Piola-Kirchhoff stress \f$S\f$: the additive split
/// \f$E = E^e + E^p + E^\theta\f$ with \f$S = D E^e\f$ (linear elasticity
/// written in the Green-Lagrange strain, the Saint Venant-Kirchhoff form),
/// exact under any rigid rotation and meant for strains that stay small
/// (with small strains the materially-non-linear-only relations may be
/// written between \f$S\f$ and \f$E\f$ at large displacement and rotation:
/// Bathe, *Finite Element Procedures*, 1996, ch. 6). Then
/// \f[
///   f_e = \int B_{NL}^T S\,dV_0 ,\qquad
///   K_e = \int B_{NL}^T C\,B_{NL}\,dV_0 + \int (G^T\mathbf{S}\,G)\otimes I\,dV_0
/// \f]
/// with the operators of TotalLagrangian.hpp. The thermal strain
/// \f$E^\theta\f$ is the Green strain of the free thermal stretch
/// \f$\vartheta = 1 + \alpha\Delta T\f$, \f$\alpha\Delta T\,(1 + \alpha\Delta T/2)\f$
/// in each direction, so a freely heated body is stress-free; the elastic
/// stiffness is not rescaled by the thermal stretch (a relative difference
/// of \f$\alpha\Delta T\f$ in a thermal stress, below the model's
/// small-strain premise).
///
/// **Mean dilatation.** Plastic flow is isochoric. On an element with
/// several integration points (the Hex8) each point's dilatation is
/// then a constraint, and in plane strain and 3-D the constraints outnumber
/// the element's degrees of freedom: the element locks and overestimates the
/// collapse load (Nagtegaal, Parks and Rice 1974). The mean-dilatation
/// operator (Hughes 1980) replaces every point's dilatation by the element's
/// volume average, \f$\bar B = B + \tfrac{1}{3}\,m\,(\bar b - b)^T\f$ with
/// \f$b = B^T m\f$ the dilatation row, \f$\bar b\f$ its average and
/// \f$m = \{1,1,1,0,0,0\}\f$, which leaves one constraint per element. In
/// plane strain the average dilatation gives \f$\varepsilon_{33} =
/// (\bar\theta - \theta)/3\f$ at a point, zero on average. With finite
/// kinematics the same is done to the Green strain,
/// \f$\bar E = E + \tfrac{1}{3}\,m\,(\overline{\mathrm{tr}E} - \mathrm{tr}E)\f$,
/// which is objective and whose second variation adds
/// \f$\tfrac{1}{3}\,\mathrm{tr}S\,(\overline{G^TG} - G^TG)\otimes I\f$ to the
/// geometric stiffness; being derived from the strain, it keeps the tangent
/// symmetric and consistent. Plane stress has no incompressibility
/// constraint (the thickness strain is free) and keeps \f$B\f$.
///
/// **Logarithmic-strain finite kinematics** (large strain; Miehe, Apel and
/// Lambrecht 2002, *Anisotropic additive plasticity in the logarithmic
/// strain space*). Total Lagrangian: the return of Plasticity.hpp - the same
/// J2, Hill48, Chaboche, plane-stress and thermal return, unchanged - takes
/// the logarithmic strain \f$E_{\log} = \tfrac12\ln C\f$ of
/// LogarithmicStrain.hpp as its strain and gives the stress T conjugate to
/// it and \f$C_{alg} = dT/dE_{\log}\f$. The additive split
/// \f$E_{\log} = E_{\log}^e + E_{\log}^p\f$ with \f$T = D E_{\log}^e\f$ is
/// exact for a coaxial homogeneous stretch (the log strains of successive
/// coaxial stretches add), so uniaxial J2 at any strain is the 1-D law in
/// the log strain, with T the Kirchhoff stress; the plastic flow is
/// isochoric in \f$\det F\f$; the Hill48 axes are those of the reference
/// configuration and convect with the material; and the model is objective
/// (C, hence everything but the push-forward, ignores a superposed
/// rotation). With \f$dE_{\log} = P\,dE\f$ (P = \f$2\,\partial
/// E_{\log}/\partial C\f$) and \f$dE = B_{NL}\,du\f$,
/// \f[
///   S = P^T T ,\qquad f_e = \int B_{NL}^T S\,dV_0 ,\qquad
///   K_e = \int B_{NL}^T\big(P^T C_{alg} P + T : \mathbb{L}\big) B_{NL}\,dV_0
///       + \int (G^T\mathbf{S}\,G)\otimes I\,dV_0 ,
/// \f]
/// \f$T : \mathbb{L}\f$ the Hessian of \f$T : E_{\log}(E)\f$ at fixed T.
/// The elasticity is Hencky's (quadratic in \f$E_{\log}\f$). The thermal
/// strain is the log strain of the free thermal stretch
/// \f$1 + \alpha\Delta T\f$ (the stretch of the finite path),
/// \f$\ln(1 + \alpha\Delta T)\,I\f$, which the return takes as the
/// change \f$\ln(1 + \alpha\Delta T)/\alpha\f$: a freely heated body is
/// stress-free. In plane stress the return finds \f$E_{\log,33}\f$ with
/// \f$T_{33} = 0\f$ (which is \f$S_{33} = 0\f$: the normal is principal),
/// and the thickness stretches by \f$\exp E_{\log,33}\f$. Since
/// \f$C = F^TF\f$ is positive definite at an inverted point too, where
/// \f$E_{\log}\f$ is that of a valid state with an axis reflected, every
/// point checks \f$\det F > 0\f$ and an inverted one throws SolverError
/// (a failed step).
///
/// *Mean dilatation* on the logarithmic volumetric strain: since
/// \f$\mathrm{tr}\,E_{\log} = \ln J\f$, the average replaces each point's
/// \f$\ln J\f$ by the element's volume average,
/// \f$\bar E_{\log} = E_{\log} + \tfrac13\,m\,(\overline{\ln J} - \ln J)\f$ -
/// exactly the log strain of the modified deformation
/// \f$\bar F = (\bar J/J)^{1/3}F\f$, so it stays objective. Its variation
/// is \f$\bar B = P B_{NL} + \tfrac13 m(\bar b - b)\f$ with
/// \f$b = m^TPB_{NL}\f$ the variation of \f$\ln J\f$, and the force is
/// \f$\int\bar B^T T\f$. The second variation of
/// \f$\int T : \bar E_{\log}\f$ at fixed T is
/// \f$\int (T - p\,m) : d^2E_{\log} + \big(\int p\big)\,d^2\overline{\ln J}\f$
/// with the Kirchhoff mean stress \f$p = \mathrm{tr}\,T/3\f$, and since
/// \f$\ln J = m : E_{\log}\f$ both are initial-stress terms of the
/// stress \f$T' = \mathrm{dev}\,T + \bar p\,m\f$ - the point's deviator
/// with the element's mean pressure \f$\bar p\f$:
/// \f[
///   K_e = \int \big[\bar B^T C_{alg}\bar B + B_{NL}^T(T' : \mathbb{L})B_{NL}
///         + (G^T\mathbf{S}'G)\otimes I\big] dV_0 ,\qquad S' = P^T T' ,
/// \f]
/// the classical mean-dilatation result, consistent (tested against the
/// derivative of the force) and assembled after all the element's returns.
///
/// *Stress recovery*: the Kirchhoff stress \f$\tau = F S F^T\f$ (with the
/// modified \f$\bar F\f$, \f$\bar F\bar S\bar F^T\f$, the same), the Cauchy
/// stress \f$\sigma = \tau/\bar J\f$ with \f$\bar J = \exp\mathrm{tr}\,\bar
/// E_{\log}\f$ (J itself without averaging), and the log strain itself.
///
/// Where this fits: Simulations stage - one element's force, tangent and plastic return.
/// Called from: fem/NonlinearSystem.hpp:NonlinearSystem::evaluate (elastoplastic_element),
///   fem/Forming.cpp:FormingAnalysis::run (elastoplastic_stress),
///   fem/Forming.cpp:FormingAnalysis::FormingAnalysis and
///   fem/NonlinearSystem.hpp:NonlinearSystem::NonlinearSystem (require_elastoplastic_materials)
/// Calls into: material/Plasticity (plastic_return), fem/TotalLagrangian,
///   material/LogarithmicStrain, fem/IncompatibleModes
/// Start reading at: elastoplastic_element (fem/Elastoplastic.cpp)
#pragma once

#include "sparlab/core/Types.hpp"
#include "sparlab/fem/FemModel.hpp"
#include "sparlab/material/Plasticity.hpp"

#include <string>
#include <vector>

namespace sparlab {

/// Kinematics of a non-linear analysis.
enum class Kinematics {
  Finite,       ///< total Lagrangian, Green-Lagrange strain: large displacement and rotation
  SmallStrain,  ///< linear strain: small displacement and rotation
  /// total Lagrangian, logarithmic strain: large displacement, rotation and
  /// strain (Hencky elasticity, plasticity in the log strain)
  FiniteLogarithmic
};

std::string to_string(Kinematics kinematics);
/// "finite", "small_strain" or "finite_logarithmic".
Kinematics parse_kinematics(const std::string& text);

/// sparlab_form models elastoplastic sheets: under "finite" or
/// "finite_logarithmic" kinematics the material of every element must have
/// a plasticity block (the elastic large-strain laws, Saint Venant-Kirchhoff
/// and neo-Hookean, were removed). "small_strain" takes any material, an
/// elastic one as linear elasticity. The forming analysis and every
/// non-linear system check it before they evaluate an element.
/// \throws ConfigError naming the first material without plasticity under
///         finite kinematics.
void require_elastoplastic_materials(const FemModel& model, Kinematics kinematics);

/// Which elements of an elastoplastic material average their dilatation (see
/// above); elastic materials always keep their plain operator, and plane
/// stress and one-point elements have nothing to average.
enum class MeanDilatation {
  /// The Hex8, whose full integration locks under isochoric plastic flow.
  Auto,
  All,   ///< every element with several integration points
  None
};

/// "auto", "all" or "none".
MeanDilatation parse_mean_dilatation(const std::string& text);

/// How the elements of a non-linear system are evaluated (NonlinearSystem.hpp,
/// the forming analysis and the explicit integrator).
struct NonlinearOptions {
  Kinematics kinematics = Kinematics::Finite;
  MeanDilatation mean_dilatation = MeanDilatation::Auto;
};

struct ElastoplasticElement {
  Vector internal_force;               ///< \f$f_e\f$ [N]
  Matrix tangent;                      ///< \f$K_e\f$ [N/m], if requested
  /// \f$\partial f_e/\partial\lambda\f$ at fixed displacement for a
  /// temperature change \f$\lambda\Delta T_0\f$ [N]; empty without one.
  Vector thermal_force_rate;
  Scalar energy = 0.0;                 ///< stored energy, elastic + hardening [J]
  std::vector<PlasticState> states;    ///< per integration point, after the return
  int yielding_points = 0;             ///< points whose return was plastic
  /// The tangent is symmetric (symmetrised to round-off); false for a
  /// material whose backstresses recover, whose tangent is not.
  bool symmetric = true;
  /// An element with internal modes (the incompatible-mode Hex8): their
  /// parameters at this displacement, found by the local Newton iteration of
  /// IncompatibleModes.hpp from the committed ones [m], and its iterations;
  /// empty and 0 otherwise.
  Vector internal;
  int internal_iterations = 0;
};

/// The element at displacement `ue` from the converged states `committed`
/// (one per point of the element's stiffness rule), with the nodal
/// temperature changes `temperature` (per unit load factor; may be null)
/// scaled by `temperature_scale`. `mean_dilatation` selects B-bar (ignored
/// in plane stress). An element with internal modes (the incompatible-mode
/// Hex8) solves for them from their committed parameters `internal` (null:
/// zero) and returns the condensed force and tangent (IncompatibleModes.hpp).
/// \throws SolverError when a return, or the local iteration of internal
///         modes, fails to converge; ConfigError for mean dilatation with
///         internal modes.
ElastoplasticElement elastoplastic_element(const FemModel& model, Index e, const Vector& ue,
                                           const std::vector<PlasticState>& committed,
                                           bool mean_dilatation, const Vector* temperature,
                                           Scalar temperature_scale, bool want_tangent,
                                           Kinematics kinematics = Kinematics::SmallStrain,
                                           const Vector* internal = nullptr);

/// Stress state of an element at a converged state, from the point returns:
/// the point averages of the Cauchy stress (small strain: \f$\sigma\f$) and
/// of the second Piola-Kirchhoff stress (small strain: \f$\sigma\f$ again),
/// 3-D Voigt, and measures of the deformation.
struct ElastoplasticStress {
  Vector6 cauchy = Vector6::Zero();           ///< [Pa]
  Vector6 piola_kirchhoff = Vector6::Zero();  ///< [Pa]
  Scalar von_mises = 0.0;                     ///< of the average Cauchy stress [Pa]
  Scalar max_equivalent_plastic_strain = 0.0; ///< \f$\max\bar\alpha\f$ over its points [-]
  /// The largest tensor strain component over its points: \f$\varepsilon\f$
  /// (small strain), \f$E\f$ (finite) or \f$E_{\log}\f$ (logarithmic), with
  /// the thickness strain [-].
  Scalar max_strain = 0.0;
  Scalar min_jacobian = 1.0;                  ///< finite: the smallest det F [-]
  /// Logarithmic kinematics: the point averages of the Kirchhoff stress
  /// \f$\tau\f$ [Pa] and of the (Lagrangian) log strain \f$E_{\log}\f$ the
  /// return took, tensor components [-]; 3-D Voigt. Zero otherwise.
  Vector6 kirchhoff = Vector6::Zero();
  Vector6 logarithmic_strain = Vector6::Zero();
};
/// With internal modes, `internal` holds their committed (converged)
/// parameters, and the measures are those of the enhanced deformation.
ElastoplasticStress elastoplastic_stress(const FemModel& model, Index e, const Vector& ue,
                                         const std::vector<PlasticState>& committed,
                                         bool mean_dilatation, const Vector* temperature,
                                         Scalar temperature_scale,
                                         Kinematics kinematics = Kinematics::SmallStrain,
                                         const Vector* internal = nullptr);

/// Number of integration points of the model's stiffness rule.
int elastoplastic_points(const FemModel& model);

}  // namespace sparlab
