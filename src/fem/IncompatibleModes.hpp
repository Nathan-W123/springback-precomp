/// \file IncompatibleModes.hpp
/// \brief The non-linear kernels of the incompatible-mode Hex8
///        (Hex8Incompatible.hpp): the local Newton iteration for the nine
///        mode parameters and the static condensation of the force and the
///        tangent. Internal to the library; the
///        public entry points are `elastoplastic_element` and
///        `elastoplastic_stress`, which dispatch here.
///
/// **The augmented element.** With the parameters \f$\alpha\f$ of the three
/// modes as the displacements of three pseudo-nodes whose reference
/// gradients are the Taylor-corrected \f$\tilde g_k\f$, the displacement
/// gradient at a point is \f$H = U G^T + A\tilde G^T\f$ and the element is
/// an eleven-node element of 33 degrees of freedom \f$(u, \alpha)\f$: its
/// force \f$r = (f_u, r_\alpha)\f$ and tangent
/// \f$\begin{bmatrix}K_{uu} & K_{u\alpha}\\ K_{\alpha u} & K_{\alpha\alpha}\end{bmatrix}\f$
/// are those of Elastoplastic.hpp (on the operators of TotalLagrangian.hpp)
/// on the augmented gradients - small strain, finite (Green-Lagrange) and
/// logarithmic kinematics alike, with the same returns (J2, Hill48,
/// Chaboche).
///
/// **Local solve.** The modes are internal: they carry no load, so
/// \f$r_\alpha(u, \alpha) = 0\f$. On every evaluation the parameters are
/// found by Newton's method *from the committed parameters* \f$\alpha_n\f$
/// (those of the last converged step), never from the last trial:
/// \f[
///   \alpha^{(0)} = \alpha_n ,\qquad
///   \alpha^{(i+1)} = \alpha^{(i)} - K_{\alpha\alpha}^{-1} r_\alpha ,
/// \f]
/// with every point's return from its committed state, so the element is a
/// pure function of \f$(u, \lambda)\f$ and the committed history - which is
/// what the line search, the step cuts, the arc length and the parallel
/// evaluation rely on (the Simo-Rifai update of \f$\alpha\f$ with the global
/// iterate is not, and is not used). It stops when
/// \f$\|r_\alpha\|_\infty \le 10^{-8}\f$ times the gross scale
/// \f$\max_i\sum_q w_q\,|\tilde B_q|^T|\sigma_q|\f$ of the sums that form it
/// (the condensed force below is then exact to the square of that), plus
/// a round-off floor of \f$10^{-13}\max_i\sum_q w_q|\tilde B_q|^T 1\,
/// |C_q|\,|H_q|(1 + |H_q|)\f$ - the forces of a strain error of that
/// relative size, which is all a rigid rotation leaves; it
/// throws SolverError, which cuts the step, after 25 iterations or at a
/// singular \f$K_{\alpha\alpha}\f$. Near the solution the full Newton
/// step is taken. Far from it - a return far from its committed state, a
/// load reversal, a start without the committed parameters - the points
/// switch between elastic and plastic along the step, \f$r_\alpha\f$ is
/// only piecewise smooth and Newton's iterates can cycle, and
/// \f$\|r_\alpha\|\f$ is no merit function; a step that does not reduce
/// it is shortened by a line search on the directional residual
/// \f$\phi(s) = d^T r_\alpha(\alpha + s d)\f$ (regula falsi, Crisfield
/// 1991, *Non-linear finite element analysis of solids and structures*
/// vol. 1, sec. 9.3), whose root minimises the incremental potential along
/// the direction wherever there is one (an associative return), each trial
/// kept a tenth of the bracket from its ends; where \f$\phi\f$ brackets no
/// root on the step (\f$K_{\alpha\alpha}\f$ not positive definite, a
/// non-convex potential) the step is halved on \f$\|r_\alpha\|\f$ and the
/// best trial kept. Elastic small strain is linear in \f$\alpha\f$ and
/// converges in one iteration.
///
/// **Condensation** at the converged parameters:
/// \f[
///   K^* = K_{uu} - K_{u\alpha}K_{\alpha\alpha}^{-1}K_{\alpha u} ,\qquad
///   f^* = f_u - K_{u\alpha}K_{\alpha\alpha}^{-1}r_\alpha ,
/// \f]
/// the derivative of \f$f^*(u) = f_u(u, \alpha(u))\f$ with
/// \f$d\alpha/du = -K_{\alpha\alpha}^{-1}K_{\alpha u}\f$; the term in
/// \f$r_\alpha\f$ is the last Newton correction, which makes the error of
/// \f$f^*\f$ quadratic in the local residual.
/// \f$K_{\alpha u}\f$ is not \f$K_{u\alpha}^T\f$ when the material tangent
/// is not symmetric (Chaboche), and \f$K^*\f$ is then left non-symmetric.
///
/// **Storage.** Everything runs on fixed-size Eigen matrices (6 x 33 strain
/// operators, 9 x 9, 24 x 9 and 24 x 24 blocks) in a per-thread workspace
/// that is reused: the local iteration allocates nothing.
///
/// **Mean dilatation** is not combined with the modes (see
/// docs/formulation.md): averaging the dilatation of the augmented element
/// leaves a zero-energy mode (the field \f$u = \kappa(xz, yz,
/// \tfrac12(z^2 - x^2 - y^2))\f$, whose strain \f$\kappa z\,I\f$ averages to
/// nothing), and averaging only the compatible dilatation leaves a sheet
/// of one element through its thickness a fraction of its bending
/// stiffness. The modes themselves relax the isochoric constraint of plastic
/// flow: they absorb the linear part of every point's dilatation.
///
/// Where this fits: Simulations stage - the incompatible-mode Hex8: solves its mode
///   parameters and condenses them out of the element's force and tangent.
/// Called from: fem/Elastoplastic.cpp:elastoplastic_element, elastoplastic_stress
/// Calls into: elements/Hex8Incompatible, material/Plasticity, material/LogarithmicStrain
/// Start reading at: incompatible_elastoplastic_element, solve_and_condense (IncompatibleModes.cpp)
#pragma once

#include "sparlab/fem/Elastoplastic.hpp"

#include <vector>

namespace sparlab {
namespace detail {

/// The largest number of local Newton iterations.
inline constexpr int kMaxLocalIterations = 25;
/// A full local Newton step is taken when it reduces
/// \f$\|r_\alpha\|\f$ at all (by this factor, or converges); otherwise
/// its length comes from the line search.
inline constexpr Scalar kFullStepReduction = 1.0 - 1.0e-4;
/// The most evaluations of one line search (regula falsi on the
/// directional residual).
inline constexpr int kMaxLineSearch = 8;
/// The line search stops at \f$|d^T r_\alpha(\alpha + s d)| \le\f$ this
/// times \f$|d^T r_\alpha(\alpha)|\f$.
inline constexpr Scalar kLineSearchRatio = 0.5;
/// Where the symmetric part of K_aa is not positive definite, the local
/// direction solves (K_aa + mu I) d = -r_alpha with mu shifting its smallest
/// eigenvalue to this times its largest magnitude (IncompatibleModes.cpp).
inline constexpr Scalar kLocalShift = 1.0e-3;
/// The first damping of the Levenberg-Marquardt fallback of the local
/// iteration, relative to the largest diagonal entry of K_aa^T K_aa.
inline constexpr Scalar kLocalMarquardt = 1.0e-6;
/// The relative tolerance of the local residual \f$r_\alpha\f$. The
/// condensed force carries the last Newton correction, so its error is
/// quadratic in this (1e-16); the points' states are consistent with it to
/// this relative level. (Measured on a strip bent past yield: the residual
/// runs 0.64, 0.10, 2e-4, 5e-10, 5e-16 at the onset of yield, three
/// iterations.)
inline constexpr Scalar kLocalTolerance = 1.0e-8;
/// The round-off floor of the local residual, relative to the force a unit
/// relative error of the strain of H would cause (IncompatibleModes.cpp):
/// a rigid rotation leaves only round-off stresses, which no relative
/// tolerance on their own sum can resolve.
inline constexpr Scalar kLocalRoundOff = 1.0e-13;

/// \throws ConfigError when `mean_dilatation` is set for an element with
///         internal modes.
void refuse_mean_dilatation(bool mean_dilatation);

ElastoplasticElement incompatible_elastoplastic_element(
    const FemModel& model, Index e, const Vector& ue, const std::vector<PlasticState>& committed,
    const Vector* internal, bool want_tangent, Kinematics kinematics);

ElastoplasticStress incompatible_elastoplastic_stress(const FemModel& model, Index e,
                                                      const Vector& ue,
                                                      const std::vector<PlasticState>& committed,
                                                      const Vector* internal,
                                                      Kinematics kinematics);

/// Shared with Elastoplastic.cpp: SolverError at an inverted point
/// (logarithmic kinematics).
void refuse_inverted_point(const Matrix& f, int dim);

}  // namespace detail
}  // namespace sparlab
