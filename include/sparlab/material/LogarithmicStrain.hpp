/// \file LogarithmicStrain.hpp
/// \brief The logarithmic (Hencky) strain of a deformation and its first and
///        second derivatives with respect to the Green-Lagrange strain, by
///        spectral decomposition - the kinematic map of the
///        logarithmic-strain finite plasticity of Elastoplastic.hpp (Miehe,
///        Apel and Lambrecht 2002).
///
/// **The strain.** With the right Cauchy-Green tensor
/// \f$C = F^TF = I + 2E\f$ (E the Green-Lagrange strain) and its spectral
/// decomposition \f$C = \sum_a \lambda_a\,N_a\otimes N_a\f$,
/// \f[
///   E_{\log} = \tfrac12\ln C = \sum_a \tfrac12\ln\lambda_a\,N_a\otimes N_a ,
///   \qquad \mathrm{tr}\,E_{\log} = \ln J .
/// \f]
/// E is decomposed rather than C, \f$\lambda_a = 1 + 2\mu_a\f$ with
/// \f$\mu_a\f$ the eigenvalues of E, and \f$\tfrac12\ln\lambda_a =
/// \tfrac12\,\mathrm{log1p}(2\mu_a)\f$: at a strain of \f$10^{-6}\f$ the log
/// strain keeps the relative accuracy of E instead of losing six digits to
/// the round-off of \f$1 + 2\mu\f$.
///
/// **First derivative** (the fourth-order \f$\mathbb{P} = 2\,\partial
/// E_{\log}/\partial C = \partial E_{\log}/\partial E\f$). For an isotropic
/// tensor function the Daleckii-Krein formula gives, in the eigenbasis,
/// \f[
///   [dE_{\log}]_{ab} = \ell[\lambda_a, \lambda_b]\,[dE]_{ab} ,\qquad
///   \ell[x, y] = \frac{\ln x - \ln y}{x - y} ,\quad \ell[x, x] = \frac1x ,
/// \f]
/// the first divided difference of ln, evaluated as
/// \f$\mathrm{log1p}((x - y)/y)/(x - y)\f$ with \f$x - y = 2(\mu_a - \mu_b)\f$:
/// accurate to round-off at any gap, equal eigenvalues included, with no
/// branch. The work conjugate of \f$E_{\log}\f$, T, gives the second
/// Piola-Kirchhoff stress \f$S = T : \mathbb{P}\f$.
///
/// **Second derivative** (the sixth-order
/// \f$\mathbb{L} = 4\,\partial^2 E_{\log}/\partial C\,\partial C\f$
/// contracted with T). The second-order Daleckii-Krein formula,
/// \f[
///   T : \frac{\partial^2 E_{\log}}{\partial E^2}[dE_1, dE_2] =
///   2\sum_{abc} T_{ab}\,\ell[\lambda_a, \lambda_c, \lambda_b]
///     \big([dE_1]_{ac}[dE_2]_{cb} + [dE_2]_{ac}[dE_1]_{cb}\big)
/// \f]
/// (eigenbasis components; \f$\ell[x, y, z]\f$ the second divided
/// difference of ln), is the Hessian of \f$T : E_{\log}(E)\f$ at fixed T,
/// \f$\partial S/\partial E\f$ at fixed T - a symmetric matrix. It is
/// evaluated as \f$(\ell[x, y] - \ell[y, z])/(x - z)\f$ (x > y > z) when the
/// spread \f$x - z\f$ exceeds \f$5\cdot10^{-3}\f$ of y, and otherwise by
/// the Taylor series about the middle eigenvalue,
/// \f$y^{-2}\sum_{k=0}^{5} (-1)^{k+1} h_k(u, w)/(k+2)\f$ with
/// \f$u = (x - y)/y\f$, \f$w = (z - y)/y\f$ and \f$h_k\f$ the complete
/// homogeneous symmetric polynomials. The threshold balances the two
/// errors: the difference quotient loses \f$\approx 4\epsilon/\tau\f$
/// (\f$2\cdot10^{-13}\f$ at the threshold \f$\tau\f$) to cancellation, the
/// truncated series \f$\tau^6 \approx 2\cdot10^{-14}\f$, so equal, nearly
/// equal and distinct eigenvalues are all evaluated to about
/// \f$10^{-13}\f$ relative, continuously across the threshold (tested
/// against central differences of \f$E_{\log}(E)\f$ at distinct, two equal,
/// three equal and nearly equal eigenvalues).
///
/// **Voigt forms.** Strains (E, \f$E_{\log}\f$) are engineering-shear Voigt
/// vectors in the order \f$\{11, 22, 33, 12, 23, 31\}\f$, stresses (T, S)
/// tensorial, as everywhere in SparLab; `projection` maps an engineering dE
/// to an engineering \f$dE_{\log}\f$, so \f$S = P^T T\f$, and the Hessian
/// is with respect to the engineering components of E.
///
/// References: Miehe, Apel and Lambrecht, *Anisotropic additive plasticity
/// in the logarithmic strain space*, CMAME 191 (2002); Miehe and Lambrecht,
/// *Algorithms for computation of stresses and elasticity moduli in terms
/// of Seth-Hill's family of generalized strain tensors*, Commun. Numer.
/// Meth. Engng 17 (2001); Bhatia, *Matrix Analysis* (1997), ch. V (the
/// Daleckii-Krein formulas).
///
/// Where this fits: Simulations stage - the logarithmic strain of the "finite_logarithmic"
///   kinematics, the default of precomp's FormingSetup (fea/setup.py).
/// Called from: fem/Elastoplastic.cpp:kinematics_of, fem/IncompatibleModes.cpp:kinematics_at
/// Calls into: Eigen (SelfAdjointEigenSolver), core/Exceptions
/// Start reading at: logarithmic_strain, logarithmic_stress
#pragma once

#include "sparlab/core/Types.hpp"

namespace sparlab {

/// The logarithmic strain of one deformation and what its derivatives need.
struct LogarithmicStrain {
  Matrix3 axes = Matrix3::Identity();  ///< the principal directions \f$N_a\f$ (columns)
  Vector3 green = Vector3::Zero();     ///< the principal Green-Lagrange strains \f$\mu_a\f$ [-]
  Vector6 strain = Vector6::Zero();    ///< \f$E_{\log}\f$, engineering shears [-]
  /// \f$dE_{\log} = P\,dE\f$ (engineering to engineering); \f$S = P^T T\f$.
  Matrix6 projection = Matrix6::Identity();
  /// \f$\ell[\lambda_a, \lambda_b]\f$, the first divided differences of ln.
  Matrix3 first = Matrix3::Identity();
};

/// The logarithmic strain of the Green-Lagrange strain `green` (engineering
/// Voigt), with its first derivative.
/// \throws SolverError when an eigenvalue of \f$C = I + 2E\f$ is not
///         positive (a degenerate point, or an E that no deformation has)
///         or the strain is not finite. An inverted point is *not* seen
///         here: \f$C = F^TF\f$ is positive definite at \f$\det F < 0\f$
///         too, so the element checks the sign of \f$\det F\f$
///         (Elastoplastic.cpp).
LogarithmicStrain logarithmic_strain(const Vector6& green);

/// The second Piola-Kirchhoff stress \f$S = P^T T\f$ of the stress T
/// conjugate to \f$E_{\log}\f$ (tensorial Voigt) [Pa].
Vector6 logarithmic_stress(const LogarithmicStrain& log, const Vector6& t);

/// \f$\partial^2 (T : E_{\log})/\partial E^2\f$ at fixed T (tensorial
/// Voigt): the symmetric 6 x 6 matrix of \f$T : \mathbb{L}\f$ over the
/// engineering components of E [Pa].
Matrix6 logarithmic_curvature(const LogarithmicStrain& log, const Vector6& t);

}  // namespace sparlab
