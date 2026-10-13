/// \file Plasticity.hpp
/// \brief Rate-independent plasticity at a material point: the backward-Euler
///        radial return of J2 (von Mises) plasticity and its consistent
///        tangent (Simo and Hughes 1998, boxes 3.1 and 3.2), and the general
///        closest-point return of Hill's 1948 anisotropic criterion with
///        multi-backstress Chaboche (Armstrong-Frederick) kinematic and Voce
///        isotropic hardening and its consistent tangent, in 3-D, plane
///        strain and plane stress.
///
/// **The law (J2, radial return).** The strain splits additively,
/// \f$\varepsilon = \varepsilon^e + \varepsilon^p + \varepsilon^\theta\f$ (the thermal strain
/// \f$\alpha\Delta T\,I\f$), the stress is \f$\sigma = D\varepsilon^e\f$, and
/// with the relative stress \f$\xi = \mathrm{dev}\,\sigma - \beta\f$ the
/// yield function is
/// \f[
///   f = \|\xi\| - \sqrt{2/3}\,\sigma_y(\bar\alpha) \le 0 ,
/// \f]
/// with associative flow \f$\dot\varepsilon^p = \dot\gamma\,n\f$,
/// \f$n = \xi/\|\xi\|\f$, the accumulated plastic strain
/// \f$\dot{\bar\alpha} = \sqrt{2/3}\,\dot\gamma\f$ and the back stress
/// \f$\dot\beta = \tfrac{2}{3}H_{kin}\,\dot\gamma\,n\f$
/// (IsotropicMaterial.hpp has \f$\sigma_y\f$).
///
/// **Integration.** From the state of the last converged step, the elastic
/// trial stress \f$s^{tr} = 2G\,\mathrm{dev}(\varepsilon - \varepsilon^p_n)\f$
/// either stays inside the yield surface or returns radially onto it:
/// \f$\Delta\gamma\f$ solves
/// \f[
///   \|\xi^{tr}\| - (2G + \tfrac{2}{3}H_{kin})\,\Delta\gamma
///   - \sqrt{2/3}\,\sigma_y(\bar\alpha_n + \sqrt{2/3}\,\Delta\gamma) = 0
/// \f]
/// (in closed form for linear hardening, by Newton with Voce saturation -
/// to \f$10^{-13}\f$ of the radius, or to \f$8\epsilon\|\xi^{tr}\|\f$
/// where that round-off floor is the larger, far outside the surface),
/// and \f$\sigma = \sigma^{tr} - 2G\Delta\gamma\,n\f$. The consistent tangent
/// is
/// \f[
///   C = K\,1\otimes 1 + 2G\theta\,(I - \tfrac{1}{3}1\otimes 1)
///       - 2G\bar\theta\,n\otimes n ,\quad
///   \theta = 1 - \frac{2G\Delta\gamma}{\|\xi^{tr}\|} ,\quad
///   \bar\theta = \frac{1}{1 + (\sigma_y' + H_{kin})/(3G)} - (1 - \theta) .
/// \f]
/// The return is exact for linear hardening under a strain increment that
/// keeps its direction, and first-order accurate otherwise.
///
/// **Idealisations.** Every point carries the full 3-D state (strain with
/// engineering shears, stress tensorial, in the Voigt order
/// \f$\{11, 22, 33, 12, 23, 31\}\f$). Plane strain returns with the given
/// \f$\varepsilon_{33}\f$ (0, or the B-bar value of Elastoplastic.hpp).
/// Plane stress finds \f$\varepsilon_{33}\f$ by Newton so that
/// \f$\sigma_{33} = 0\f$ and condenses it out of the tangent,
/// \f$C_{ab} - C_{a3}C_{3b}/C_{33}\f$ (valid for a non-symmetric C too).
///
/// **Hill48 and Chaboche: the general return.** With the Hill48 criterion
/// or Armstrong-Frederick backstresses (PlasticityParameters::general())
/// the return is a closest-point projection (Simo and Hughes 1998, box 3.4;
/// Simo and Taylor 1986) of the elastic trial state onto
/// \f[
///   f = \bar\sigma(\xi) - \sigma_y(\bar\alpha) \le 0 ,\qquad
///   \bar\sigma(\xi) = \sqrt{\xi^T P\,\xi} ,\qquad
///   \xi = \sigma - \textstyle\sum_i \alpha_i ,
/// \f]
/// P the yield matrix of IsotropicMaterial.hpp (tensorial Voigt components,
/// global axes; its normal rows sum to zero, so f ignores the pressure). The
/// flow is associative, \f$\dot\varepsilon^p = \dot\lambda\,m\f$ with
/// \f$m = \partial\bar\sigma/\partial\sigma = P\xi/\bar\sigma\f$ -
/// which, differentiated by Voigt components, is the *engineering* form of
/// the flow tensor, \f$m_t = S^{-1}m\f$ its tensorial form with
/// \f$S = \mathrm{diag}(1,1,1,2,2,2)\f$ - and \f$\bar\alpha\f$ grows by
/// \f$\dot\lambda\f$, the work-conjugate equivalent plastic strain
/// (\f$\xi : \dot\varepsilon^p = \bar\sigma\dot\lambda\f$, P being
/// quadratic). For von Mises, \f$m_t = \sqrt{3/2}\,n\f$ and
/// \f$\Delta\lambda = \sqrt{2/3}\,\Delta\gamma\f$, the radial return's
/// accumulated strain. The backstresses (Chaboche 1986; Armstrong and
/// Frederick 1966) evolve as
/// \f$\dot\alpha_i = \tfrac{2}{3}C_i\,\dot\varepsilon^p_t - \gamma_i\,\alpha_i\,\dot\lambda\f$,
/// Prager's \f$H_{kin}\f$ joining them as one with \f$\gamma = 0\f$.
///
/// *The discrete equations.* Over a step with the flow direction of its
/// end, each backstress is
/// \f$\alpha_i = \theta_i\,\alpha_{i,n} + \tfrac{2}{3}C_i\,\chi_i\,m_t\f$ with
/// \f$\theta = e^{-\gamma\Delta\lambda}\f$,
/// \f$\chi = (1 - e^{-\gamma\Delta\lambda})/\gamma\f$ (exponential: the
/// exact solution of the linear ODE for a fixed \f$m_t\f$;
/// \f$\theta' = -\gamma\theta\f$, \f$\chi' = \theta\f$), or
/// \f$\theta = 1/(1+\gamma\Delta\lambda)\f$, \f$\chi = \theta\Delta\lambda\f$
/// (backward Euler; \f$\theta' = -\gamma\theta^2\f$, \f$\chi' = \theta^2\f$).
/// The elasticity is isotropic and \f$m\f$ deviatoric, so
/// \f$\sigma = \sigma^{tr} - 2G\Delta\lambda\,m_t\f$ and, the backstresses
/// eliminated in closed form,
/// \f[
///   \xi = \tilde\xi(\Delta\lambda) - b(\Delta\lambda)\,m_t ,\quad
///   \tilde\xi = s^{tr} - \textstyle\sum_i\theta_i\,\alpha_{i,n} ,\quad
///   b = 2G\Delta\lambda + \tfrac{2}{3}\textstyle\sum_i C_i\chi_i .
/// \f]
/// Newton solves the seven equations in \f$x = (\xi, \Delta\lambda)\f$
/// \f[
///   R_1 = \xi - \tilde\xi + b\,S^{-1}P\xi/\bar\sigma = 0 ,\qquad
///   R_2 = \bar\sigma(\xi) - \sigma_y(\bar\alpha_n + \Delta\lambda) = 0 ,
/// \f]
/// from the trial point \f$(\xi^{tr}, 0)\f$ - its first step is the
/// cutting-plane estimate - with the Jacobian (using
/// \f$\partial(P\xi/\bar\sigma)/\partial\xi = (P - mm^T)/\bar\sigma\f$)
/// \f[
///   J = \begin{bmatrix} I + \frac{b}{\bar\sigma}S^{-1}(P - mm^T) &
///       \sum_i\theta_i'\alpha_{i,n} + b'\,m_t
///       \\ m^T & -\sigma_y' \end{bmatrix} ,\qquad
///   b' = 2G + \tfrac{2}{3}\textstyle\sum_i C_i\chi_i' ,
/// \f]
/// a backtracking line search on \f$\|R\|\f$ and \f$\Delta\lambda \ge 0\f$,
/// to \f$\|R\|_\infty \le 10^{-13}\sigma_y\f$ (or 32 round-offs of the trial
/// stress, for a step so large that this is more), and then one more full
/// Newton step, which takes the residual to round-off: the state is rebuilt
/// from \f$(\Delta\lambda, m_t)\f$, and its yield function differs from
/// \f$R_2\f$ by about \f$m\cdot R_1\f$. The yield function of the returned
/// state is then round-off (at most \f$10^{-13}\sigma_y\f$, tested), so the
/// elastic re-check of stress recovery, from the committed state at the
/// converged strain, stays elastic; in plane stress for the same reason the
/// general return drives \f$\sigma_{33}\f$ to \f$10^{-13}\f$ of the stress
/// (the radial return keeps its \f$10^{-12}\f$). A line search that stalls
/// counts as converged only within four times the tolerance.
/// A plane model (plane strain or plane stress) has no out-of-plane shear
/// strain, so a Hill48 frame must have z along one of its axes
/// (PlasticityParameters::plane_compatible()); any other frame couples that
/// shear to the in-plane flow and is refused.
///
/// *The consistent tangent.* The strain enters only through
/// \f$s^{tr} = 2G\,\mathrm{dev}\,\varepsilon\f$, i.e.
/// \f$\partial R_1/\partial\varepsilon = -D_{dev}\f$ with \f$D_{dev}\f$ the
/// linear map of an engineering strain onto \f$2G\,\mathrm{dev}\f$ of its
/// tensor, and \f$\partial R_2/\partial\varepsilon = 0\f$. Differentiating
/// \f$R(x(\varepsilon), \varepsilon) = 0\f$ at the solution,
/// \f[
///   \begin{bmatrix} A \\ a^T \end{bmatrix} =
///   \frac{\partial(\xi, \Delta\lambda)}{\partial\varepsilon} =
///   J^{-1}\begin{bmatrix} D_{dev} \\ 0 \end{bmatrix} ,
/// \f]
/// and from \f$\sigma = K\,\mathrm{tr}\,\varepsilon^e_{tr}\,1 + s^{tr} -
/// 2G\Delta\lambda\,m_t\f$ with
/// \f$dm_t = S^{-1}(P - mm^T)\,d\xi/\bar\sigma\f$,
/// \f[
///   C_{alg} = C^e - 2G\Big[m_t\,a^T + \frac{\Delta\lambda}{\bar\sigma}
///             S^{-1}(P - mm^T)\,A\Big] .
/// \f]
/// It is symmetric for linear kinematic hardening and non-symmetric as soon
/// as a backstress recovers (\f$\gamma_i > 0\f$: the recovery term of J is not
/// along \f$m_t\f$); PlasticResponse::symmetric says which. At
/// \f$\Delta\lambda = 0\f$ it is the continuum tangent, which the loading
/// flag uses: for von Mises without backstresses
/// \f$a = 2G\,m_t/(3G + \sigma_y')\f$ and
/// \f$C = C^e - 2G\,n\otimes n/(1 + \sigma_y'/3G)\f$, the radial return's.
/// The stored energy adds \f$\sum_i \tfrac{3}{4C_i}\|\alpha_i\|^2\f$
/// (Prager's \f$\tfrac{H_{kin}}{3}\|\varepsilon^p\|^2\f$ from a virgin state).
///
/// *Accuracy.* The return is backward Euler in the flow direction: exact on
/// a proportional path (uniaxial stress along any direction for Hill48 with
/// isotropic hardening; von Mises with backstresses whose recovery the
/// exponential integrator follows exactly), first-order accurate otherwise.
/// Hill48 with backstresses is in general not proportional even in uniaxial
/// stress (\f$\dot\alpha \parallel S^{-1}P\xi\f$ is not parallel to \f$\xi\f$).
/// The von Mises law without backstresses keeps the radial return above,
/// bit for bit.
#pragma once

#include "sparlab/core/Types.hpp"
#include "sparlab/material/IsotropicMaterial.hpp"

#include <array>

namespace sparlab {

/// kMaxBackstresses zero tensors.
inline std::array<Vector6, kMaxBackstresses> zero_back_stresses() {
  std::array<Vector6, kMaxBackstresses> out;
  for (Vector6& v : out) v.setZero();
  return out;
}

/// Internal variables of one integration point.
struct PlasticState {
  Vector6 plastic_strain = Vector6::Zero();  ///< \f$\varepsilon^p\f$, engineering shears [-]
  /// \f$\beta = \sum_i\alpha_i\f$, the whole back stress, tensorial [Pa].
  Vector6 back_stress = Vector6::Zero();
  /// The general return's backstresses \f$\alpha_i\f$, in the order of
  /// PlasticityParameters::kinematic_term (Prager's last), tensorial and
  /// deviatoric [Pa]; the radial return keeps Prager's in back_stress alone.
  std::array<Vector6, kMaxBackstresses> back_stresses = zero_back_stresses();
  Scalar equivalent_plastic_strain = 0.0;    ///< \f$\bar\alpha\f$ [-]
  Scalar thickness_strain = 0.0;             ///< plane stress: total \f$\varepsilon_{33}\f$ [-]
  /// The return that produced this state was plastic. A point still on its
  /// yield surface is then assumed to go on loading: its tangent at zero
  /// strain increment is the continuum elastoplastic one (\f$\Delta\gamma = 0\f$
  /// in the consistent tangent), not the elastic one - which only changes the
  /// predictor of the next step, never a converged state.
  bool loading = false;
};

/// The result of one return.
struct PlasticResponse {
  Vector6 stress = Vector6::Zero();    ///< \f$\sigma\f$ [Pa]
  Matrix6 tangent = Matrix6::Zero();   ///< \f$d\sigma/d\varepsilon\f$ [Pa]; condensed in plane stress
  PlasticState state;                  ///< the updated internal variables
  bool yielding = false;               ///< the return was plastic
  Scalar energy = 0.0;                 ///< stored energy density, elastic + hardening [J/m^3]
  Scalar strain_33 = 0.0;              ///< the total \f$\varepsilon_{33}\f$ it used [-]
  /// The tangent is symmetric (false for a plastic return with a recovering
  /// backstress).
  bool symmetric = true;
};

/// Stress, consistent tangent and updated state at the total strain
/// `strain` (all six components; in plane stress component 2 is ignored and
/// found) from the converged state `committed`, with the thermal strain
/// \f$\alpha\,\Delta T\f$ of `delta_t`: the radial return for von Mises
/// with Prager's kinematic hardening at most, the general closest-point
/// return otherwise. Without `want_tangent` the general 3-D return skips the
/// tangent (left elastic); plane stress always needs it.
/// \throws SolverError when the plane-stress iteration or the return fails
///         to converge; ConfigError for a plane state with a Hill48 frame
///         whose axes miss z (PlasticityParameters::plane_compatible()).
PlasticResponse plastic_return(const IsotropicMaterial& material, StressState state,
                               const Vector6& strain, const PlasticState& committed,
                               Scalar delta_t = 0.0, bool want_tangent = true);

/// The von Mises stress of a 3-D Voigt stress [Pa].
Scalar von_mises_stress(const Vector6& stress);

}  // namespace sparlab
