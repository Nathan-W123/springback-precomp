#include "sparlab/material/Plasticity.hpp"

#include "sparlab/core/Exceptions.hpp"

#include <Eigen/LU>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <sstream>

namespace sparlab {
namespace {

const Scalar kSqrt23 = std::sqrt(2.0 / 3.0);

/// The norm of a symmetric tensor given by its tensorial Voigt components.
Scalar tensor_norm(const Vector6& v) {
  return std::sqrt(v(0) * v(0) + v(1) * v(1) + v(2) * v(2) +
                   2.0 * (v(3) * v(3) + v(4) * v(4) + v(5) * v(5)));
}

/// The deviatoric stress 2G dev(eps) of an engineering-shear strain, as
/// tensorial Voigt components.
Vector6 deviatoric_stress(Scalar g, const Vector6& strain) {
  const Scalar mean = (strain(0) + strain(1) + strain(2)) / 3.0;
  Vector6 s;
  for (int i = 0; i < 3; ++i) s(i) = 2.0 * g * (strain(i) - mean);
  for (int i = 3; i < 6; ++i) s(i) = g * strain(i);  // 2G (gamma / 2)
  return s;
}

/// The squared norm of the deviator of an engineering-shear strain.
Scalar deviatoric_norm2(const Vector6& strain) {
  const Scalar mean = (strain(0) + strain(1) + strain(2)) / 3.0;
  Scalar out = 0.0;
  for (int i = 0; i < 3; ++i) out += (strain(i) - mean) * (strain(i) - mean);
  for (int i = 3; i < 6; ++i) out += 0.5 * strain(i) * strain(i);  // 2 (gamma / 2)^2
  return out;
}

/// The squared tensor norm of an engineering-shear strain.
Scalar strain_norm2(const Vector6& strain) {
  Scalar out = 0.0;
  for (int i = 0; i < 3; ++i) out += strain(i) * strain(i);
  for (int i = 3; i < 6; ++i) out += 0.5 * strain(i) * strain(i);
  return out;
}

/// The isotropic elasticity tensor in Voigt form: K m m^T + 2G (I - m m^T/3)
/// with G on the engineering-shear diagonal.
Matrix6 elastic_tangent(Scalar k, Scalar g) {
  Matrix6 c = Matrix6::Zero();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) c(i, j) = k - 2.0 * g / 3.0;
    c(i, i) = k + 4.0 * g / 3.0;
    c(3 + i, 3 + i) = g;
  }
  return c;
}

/// The 3-D return at a fully given strain.
PlasticResponse return_3d(const IsotropicMaterial& m, const Vector6& strain,
                          const PlasticState& committed, Scalar delta_t) {
  const PlasticityParameters& p = m.plasticity();
  const Scalar g = m.shear_modulus();
  const Scalar k = m.lame_lambda() + 2.0 * g / 3.0;
  const Scalar hk = p.kinematic_hardening_modulus;

  Vector6 elastic = strain - committed.plastic_strain;
  const Scalar thermal = m.thermal_expansion() * delta_t;
  if (thermal != 0.0) elastic.head(3).array() -= thermal;
  const Scalar volumetric = elastic(0) + elastic(1) + elastic(2);
  const Vector6 s_trial = deviatoric_stress(g, elastic);
  const Vector6 xi_trial = s_trial - committed.back_stress;
  const Scalar xi_norm = tensor_norm(xi_trial);
  const Scalar alpha_n = committed.equivalent_plastic_strain;
  const Scalar radius_n = kSqrt23 * p.yield(alpha_n);
  const Scalar f_trial = xi_norm - radius_n;

  PlasticResponse out;
  out.state = committed;
  out.state.loading = false;
  out.strain_33 = strain(2);
  // Round-off in the trial stress must not start plastic flow on the yield
  // surface itself.
  if (!p.enabled() || f_trial <= 1.0e-12 * radius_n) {
    out.stress = s_trial;
    out.stress.head(3).array() += k * volumetric;
    out.tangent = elastic_tangent(k, g);
    out.energy = 0.5 * k * volumetric * volumetric + g * deviatoric_norm2(elastic) +
                 p.isotropic_energy(alpha_n) + hk / 3.0 * strain_norm2(committed.plastic_strain);
    // Still on the surface after a plastic step: assume further loading
    // (the continuum tangent, Delta gamma = 0) - see PlasticState::loading.
    if (p.enabled() && committed.loading && f_trial > -1.0e-10 * radius_n && xi_norm > 0.0) {
      const Vector6 n = xi_trial / xi_norm;
      out.tangent -= 2.0 * g / (1.0 + (p.yield_slope(alpha_n) + hk) / (3.0 * g)) *
                     (n * n.transpose());
      out.state.loading = true;
    }
    return out;
  }

  // The consistency condition for the plastic multiplier.
  const Scalar stiffness = 2.0 * g + 2.0 / 3.0 * hk;
  Scalar dgamma = f_trial / (stiffness + 2.0 / 3.0 * p.yield_slope(alpha_n));
  if (p.saturation_stress != 0.0) {
    // Voce: g(dgamma) is convex and decreasing, so Newton from the left of the
    // root rises monotonically onto it.
    dgamma = 0.0;
    bool converged = false;
    for (int it = 0; it < 60; ++it) {
      const Scalar alpha = alpha_n + kSqrt23 * dgamma;
      const Scalar residual = xi_norm - stiffness * dgamma - kSqrt23 * p.yield(alpha);
      if (std::abs(residual) <= 1.0e-13 * radius_n) {
        converged = true;
        break;
      }
      dgamma += residual / (stiffness + 2.0 / 3.0 * p.yield_slope(alpha));
    }
    if (!converged) {
      // Far outside the surface - |xi_trial| beyond about 450 radii, a
      // deviatoric trial increment of order one, which large-strain Newton
      // iterates reach - 1e-13 of the radius is below one ulp of
      // |xi_trial|: Newton sits on the root to round-off but the residual
      // cannot meet that tolerance. Accept it at that floor, a few ulps of
      // |xi_trial|. (Every return that met the tolerance is unchanged.)
      const Scalar alpha = alpha_n + kSqrt23 * dgamma;
      const Scalar residual = xi_norm - stiffness * dgamma - kSqrt23 * p.yield(alpha);
      converged =
          std::abs(residual) <= 8.0 * std::numeric_limits<Scalar>::epsilon() * xi_norm;
    }
    if (!converged) {
      std::ostringstream os;
      os << "the J2 return of material '" << m.name()
         << "' did not converge for the plastic multiplier";
      throw SolverError(os.str());
    }
  }
  const Scalar alpha = alpha_n + kSqrt23 * dgamma;
  const Vector6 n = xi_trial / xi_norm;

  out.yielding = true;
  out.stress = s_trial - 2.0 * g * dgamma * n;
  out.stress.head(3).array() += k * volumetric;
  PlasticState& st = out.state;
  st.loading = true;
  st.equivalent_plastic_strain = alpha;
  st.back_stress += 2.0 / 3.0 * hk * dgamma * n;
  for (int i = 0; i < 3; ++i) st.plastic_strain(i) += dgamma * n(i);
  for (int i = 3; i < 6; ++i) st.plastic_strain(i) += 2.0 * dgamma * n(i);  // engineering shear

  const Scalar theta = 1.0 - 2.0 * g * dgamma / xi_norm;
  const Scalar theta_bar = 1.0 / (1.0 + (p.yield_slope(alpha) + hk) / (3.0 * g)) - (1.0 - theta);
  Matrix6 c = Matrix6::Zero();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) c(i, j) = k - 2.0 * g * theta / 3.0;
    c(i, i) = k + 4.0 * g * theta / 3.0;
    c(3 + i, 3 + i) = g * theta;
  }
  c -= 2.0 * g * theta_bar * (n * n.transpose());
  out.tangent = c;

  Vector6 elastic_new = elastic;
  for (int i = 0; i < 3; ++i) elastic_new(i) -= dgamma * n(i);
  for (int i = 3; i < 6; ++i) elastic_new(i) -= 2.0 * dgamma * n(i);
  out.energy = 0.5 * k * volumetric * volumetric + g * deviatoric_norm2(elastic_new) +
               p.isotropic_energy(alpha) + hk / 3.0 * strain_norm2(st.plastic_strain);
  return out;
}

// ---------------------------------------------------------------------------
// The general return: Hill48 and/or Chaboche backstresses (see the file
// comment of Plasticity.hpp for the equations and the tangent's derivation).
// ---------------------------------------------------------------------------

using Vector7 = Eigen::Matrix<Scalar, 7, 1>;
using Matrix7 = Eigen::Matrix<Scalar, 7, 7>;
using Matrix76 = Eigen::Matrix<Scalar, 7, 6>;

/// The factors of one Armstrong-Frederick backstress over a step with
/// multiplier dl, alpha = theta alpha_n + (2/3) C chi m_t, and their
/// derivatives with respect to dl.
struct Decay {
  Scalar theta = 1.0;
  Scalar chi = 0.0;
  Scalar dtheta = 0.0;
  Scalar dchi = 1.0;
};

Decay decay(Scalar gamma, Scalar dl, KinematicIntegration integration) {
  Decay d;
  if (gamma == 0.0) {
    d.chi = dl;  // Prager: theta = 1, chi = dl
    return d;
  }
  if (integration == KinematicIntegration::Exponential) {
    // alpha' = (2/3) C m_t - gamma alpha, integrated exactly for a fixed m_t.
    d.theta = std::exp(-gamma * dl);
    d.chi = -std::expm1(-gamma * dl) / gamma;
    d.dtheta = -gamma * d.theta;
    d.dchi = d.theta;
  } else {
    // alpha (1 + gamma dl) = alpha_n + (2/3) C dl m_t.
    d.theta = 1.0 / (1.0 + gamma * dl);
    d.chi = d.theta * dl;
    d.dtheta = -gamma * d.theta * d.theta;
    d.dchi = d.theta * d.theta;
  }
  return d;
}

/// S^-1 v: the tensorial components of an engineering-shear Voigt vector.
Vector6 tensorial(const Vector6& v) {
  Vector6 t = v;
  t.tail<3>() *= 0.5;
  return t;
}

/// The linear map D_dev of an engineering-shear strain onto 2G dev of its
/// tensor (tensorial components): the strain derivative of s_tr.
Matrix6 deviatoric_tangent(Scalar g) {
  Matrix6 d = Matrix6::Zero();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) d(i, j) = -2.0 * g / 3.0;
    d(i, i) = 4.0 * g / 3.0;
    d(3 + i, 3 + i) = g;
  }
  return d;
}

/// The residual of the closest-point equations at (xi, dl), with everything
/// the Jacobian and the update need.
struct GeneralPoint {
  Scalar equivalent = 0.0;  ///< sigma_bar(xi)
  Vector6 m;                ///< P xi / sigma_bar, engineering
  Vector6 mt;               ///< S^-1 m, tensorial
  Scalar b = 0.0;           ///< 2 G dl + (2/3) sum C_i chi_i
  Scalar db = 0.0;          ///< d b / d dl
  Vector6 back_rate;        ///< sum theta_i' alpha_i,n = -d xi_tilde / d dl
  std::array<Decay, kMaxBackstresses> decays;
  Vector7 residual;
};

class GeneralReturn {
 public:
  GeneralReturn(const IsotropicMaterial& m, const PlasticState& committed, const Vector6& s_trial)
      : p_(m.plasticity()), g_(m.shear_modulus()), committed_(committed), s_trial_(s_trial),
        count_(p_.kinematic_terms()) {
    for (int i = 0; i < count_; ++i) terms_[static_cast<std::size_t>(i)] = p_.kinematic_term(i);
  }

  /// R(xi, dl); false where sigma_bar(xi) vanishes (the flow direction is
  /// undefined there).
  bool evaluate(const Vector6& xi, Scalar dl, GeneralPoint& e) const {
    const Vector6 pxi = p_.yield_matrix * xi;
    const Scalar q = xi.dot(pxi);
    if (!(q > 0.0) || !std::isfinite(q)) return false;
    e.equivalent = std::sqrt(q);
    e.m = pxi / e.equivalent;
    e.mt = tensorial(e.m);
    Vector6 xi_tilde = s_trial_;
    e.b = 2.0 * g_ * dl;
    e.db = 2.0 * g_;
    e.back_rate.setZero();
    for (int i = 0; i < count_; ++i) {
      const std::size_t u = static_cast<std::size_t>(i);
      const Decay d = decay(terms_[u].recovery, dl, p_.kinematic_integration);
      const Vector6& alpha_n = committed_.back_stresses[u];
      xi_tilde -= d.theta * alpha_n;
      e.b += 2.0 / 3.0 * terms_[u].modulus * d.chi;
      e.db += 2.0 / 3.0 * terms_[u].modulus * d.dchi;
      e.back_rate += d.dtheta * alpha_n;
      e.decays[u] = d;
    }
    e.residual.head<6>() = xi - xi_tilde + e.b * e.mt;
    e.residual(6) = e.equivalent - p_.yield(committed_.equivalent_plastic_strain + dl);
    return true;
  }

  /// dR/d(xi, dl) at an evaluated point.
  Matrix7 jacobian(const GeneralPoint& e, Scalar dl) const {
    Matrix7 j;
    j.topLeftCorner<6, 6>() = Matrix6::Identity() + (e.b / e.equivalent) * flow_curvature(e);
    j.topRightCorner<6, 1>() = e.back_rate + e.db * e.mt;
    j.bottomLeftCorner<1, 6>() = e.m.transpose();
    j(6, 6) = -p_.yield_slope(committed_.equivalent_plastic_strain + dl);
    return j;
  }

  /// S^-1 (P - m m^T) = sigma_bar d m_t / d xi.
  Matrix6 flow_curvature(const GeneralPoint& e) const {
    Matrix6 c = p_.yield_matrix - e.m * e.m.transpose();
    c.bottomRows<3>() *= 0.5;
    return c;
  }

  /// The consistent tangent at a converged point:
  /// C^e - 2G [m_t a^T + (dl / sigma_bar) S^-1 (P - m m^T) A] with
  /// [A; a^T] = J^-1 [D_dev; 0].
  Matrix6 tangent(const GeneralPoint& e, Scalar dl, const Matrix6& elastic) const {
    Matrix76 rhs = Matrix76::Zero();
    rhs.topRows<6>() = deviatoric_tangent(g_);
    const Matrix76 da = jacobian(e, dl).partialPivLu().solve(rhs);
    Matrix6 c = elastic - 2.0 * g_ * (e.mt * da.row(6));
    if (dl > 0.0) {
      c.noalias() -= (2.0 * g_ * dl / e.equivalent) * (flow_curvature(e) * da.topRows<6>());
    }
    return c;
  }

  int count() const { return count_; }
  const Backstress& term(int i) const { return terms_[static_cast<std::size_t>(i)]; }

 private:
  const PlasticityParameters& p_;
  Scalar g_;
  const PlasticState& committed_;
  const Vector6& s_trial_;
  int count_;
  std::array<Backstress, kMaxBackstresses> terms_{};
};

/// The stored energy of the backstresses, sum_i 3 |alpha_i|^2 / (4 C_i).
Scalar kinematic_energy(const GeneralReturn& r, const PlasticState& state) {
  Scalar out = 0.0;
  for (int i = 0; i < r.count(); ++i) {
    const Scalar norm = tensor_norm(state.back_stresses[static_cast<std::size_t>(i)]);
    out += 0.75 * norm * norm / r.term(i).modulus;
  }
  return out;
}

/// The 3-D closest-point return of Hill48 and/or Chaboche at a fully given
/// strain.
PlasticResponse return_general_3d(const IsotropicMaterial& m, const Vector6& strain,
                                  const PlasticState& committed, Scalar delta_t,
                                  bool want_tangent) {
  const PlasticityParameters& p = m.plasticity();
  const Scalar g = m.shear_modulus();
  const Scalar k = m.lame_lambda() + 2.0 * g / 3.0;

  Vector6 elastic = strain - committed.plastic_strain;
  const Scalar thermal = m.thermal_expansion() * delta_t;
  if (thermal != 0.0) elastic.head(3).array() -= thermal;
  const Scalar volumetric = elastic(0) + elastic(1) + elastic(2);
  const Vector6 s_trial = deviatoric_stress(g, elastic);
  const GeneralReturn gr(m, committed, s_trial);
  Vector6 xi_trial = s_trial;
  for (int i = 0; i < gr.count(); ++i) {
    xi_trial -= committed.back_stresses[static_cast<std::size_t>(i)];
  }
  const Scalar alpha_n = committed.equivalent_plastic_strain;
  const Scalar sy_n = p.yield(alpha_n);
  const Scalar eq_trial = std::sqrt(std::max(xi_trial.dot(p.yield_matrix * xi_trial), 0.0));
  const Scalar f_trial = eq_trial - sy_n;
  const Matrix6 ce = elastic_tangent(k, g);
  if (!std::isfinite(f_trial)) {
    throw SolverError("the closest-point return of material '" + m.name() +
                      "' met a non-finite trial stress");
  }

  PlasticResponse out;
  out.state = committed;
  out.state.loading = false;
  out.strain_33 = strain(2);
  if (!p.enabled() || f_trial <= 1.0e-12 * sy_n) {
    out.stress = s_trial;
    out.stress.head(3).array() += k * volumetric;
    out.tangent = ce;
    out.energy = 0.5 * k * volumetric * volumetric + g * deviatoric_norm2(elastic) +
                 p.isotropic_energy(alpha_n) + kinematic_energy(gr, committed);
    // Still on the surface after a plastic step: the continuum tangent
    // (the consistent one at dl = 0) - see PlasticState::loading.
    if (p.enabled() && committed.loading && f_trial > -1.0e-10 * sy_n && eq_trial > 0.0) {
      out.state.loading = true;
      GeneralPoint e;
      if (want_tangent && gr.evaluate(xi_trial, 0.0, e)) out.tangent = gr.tangent(e, 0.0, ce);
    }
    return out;
  }

  // Newton on (xi, dl) from the trial point, with a backtracking line search
  // on |R| and dl >= 0.
  Vector6 xi = xi_trial;
  Scalar dl = 0.0;
  GeneralPoint e;
  gr.evaluate(xi, dl, e);  // f_trial > 0 and finite: xi_trial is finite and non-zero
  const Scalar scale =
      std::max({sy_n, s_trial.cwiseAbs().maxCoeff(), xi_trial.cwiseAbs().maxCoeff()});
  const Scalar eps_scale = std::numeric_limits<Scalar>::epsilon() * scale;
  const Scalar tol = std::max(1.0e-13 * sy_n, 32.0 * eps_scale);
  bool converged = false;
  int it = 0;
  for (; it < 50; ++it) {
    const Scalar r_inf = e.residual.cwiseAbs().maxCoeff();
    if (r_inf <= tol) {
      converged = true;
      break;
    }
    const Vector7 step = gr.jacobian(e, dl).partialPivLu().solve(-e.residual);
    const Scalar r0 = e.residual.norm();
    Scalar t = 1.0;
    bool accepted = false;
    GeneralPoint trial;
    for (int ls = 0; ls < 40; ++ls) {
      const Vector6 xi_t = xi + t * step.head<6>();
      const Scalar dl_t = std::max(dl + t * step(6), 0.0);
      if (gr.evaluate(xi_t, dl_t, trial) && trial.residual.norm() <= (1.0 - 1.0e-4 * t) * r0) {
        xi = xi_t;
        dl = dl_t;
        e = trial;
        accepted = true;
        break;
      }
      t *= 0.5;
    }
    if (!accepted) {
      // No descent left: round-off stagnation at the tolerance's own scale
      // is convergence, anything else a failure.
      converged = r_inf <= 4.0 * tol;
      break;
    }
  }
  if (!converged) {
    std::ostringstream os;
    os << "the " << (p.criterion == YieldCriterion::Hill48 ? "Hill48" : "von Mises")
       << " closest-point return of material '" << m.name() << "' did not converge (residual "
       << e.residual.cwiseAbs().maxCoeff() / sy_n << " of the yield stress after " << it
       << " iterations)";
    throw SolverError(os.str());
  }
  if (e.residual.cwiseAbs().maxCoeff() > 4.0 * eps_scale) {
    // One more full Newton step: from within the tolerance, quadratic
    // convergence takes the residual to round-off. The state below is
    // rebuilt from (dl, m_t), and its yield function differs from R_2 by
    // about m . R_1 - up to a few times the tolerance without this step,
    // enough for the elastic re-check of stress recovery (1e-12 sigma_y) to
    // take a spurious plastic step after a large increment.
    const Vector7 step = gr.jacobian(e, dl).partialPivLu().solve(-e.residual);
    const Vector6 xi_p = xi + step.head<6>();
    const Scalar dl_p = std::max(dl + step(6), 0.0);
    GeneralPoint polished;
    if (gr.evaluate(xi_p, dl_p, polished) &&
        polished.residual.cwiseAbs().maxCoeff() < e.residual.cwiseAbs().maxCoeff()) {
      xi = xi_p;
      dl = dl_p;
      e = polished;
    }
  }

  out.yielding = true;
  out.symmetric = p.symmetric_tangent();
  out.stress = s_trial - 2.0 * g * dl * e.mt;
  out.stress.head(3).array() += k * volumetric;
  PlasticState& st = out.state;
  st.loading = true;
  st.equivalent_plastic_strain = alpha_n + dl;
  st.plastic_strain += dl * e.m;  // engineering shear
  st.back_stress.setZero();
  for (int i = 0; i < gr.count(); ++i) {
    const std::size_t u = static_cast<std::size_t>(i);
    const Decay& d = e.decays[u];
    st.back_stresses[u] =
        d.theta * committed.back_stresses[u] + (2.0 / 3.0 * gr.term(i).modulus * d.chi) * e.mt;
    st.back_stress += st.back_stresses[u];
  }
  const Vector6 elastic_new = elastic - dl * e.m;
  out.energy = 0.5 * k * volumetric * volumetric + g * deviatoric_norm2(elastic_new) +
               p.isotropic_energy(st.equivalent_plastic_strain) + kinematic_energy(gr, st);
  out.tangent = want_tangent ? gr.tangent(e, dl, ce) : ce;
  return out;
}

/// Plane stress around a 3-D return: eps_33 such that sigma_33 = 0 to
/// `tolerance` times the stress. The elastic predictor of eps_33 is exact
/// for an elastic step (the elasticity is isotropic); a plastic one
/// converges quadratically with the consistent C_33,33 > 0.
template <class Return3d>
PlasticResponse plane_stress(const IsotropicMaterial& material, const Return3d& return_3d_at,
                             const Vector6& strain, const PlasticState& committed,
                             Scalar delta_t, Scalar tolerance) {
  const Scalar g = material.shear_modulus();
  const Scalar lambda = material.lame_lambda();
  const Scalar thermal = material.thermal_expansion() * delta_t;
  const Vector6& ep = committed.plastic_strain;
  Vector6 trial = strain;
  trial(2) = ep(2) + thermal -
             lambda / (lambda + 2.0 * g) *
                 ((strain(0) - ep(0) - thermal) + (strain(1) - ep(1) - thermal));
  PlasticResponse out = return_3d_at(trial);
  const Scalar scale =
      std::max(out.stress.head(3).cwiseAbs().maxCoeff(), material.plasticity().yield_stress);
  const Scalar limit = tolerance * std::max(scale, std::numeric_limits<Scalar>::min());
  int it = 0;
  while (std::abs(out.stress(2)) > limit) {
    if (++it > 30) {
      std::ostringstream os;
      os << "the plane-stress return of material '" << material.name()
         << "' did not reach sigma_33 = 0 (left " << out.stress(2) << " Pa)";
      throw SolverError(os.str());
    }
    trial(2) -= out.stress(2) / out.tangent(2, 2);
    out = return_3d_at(trial);
  }
  // Condense eps_33 out: d sigma_ab = (C_ab - C_a3 C_3b / C_33) d eps_b.
  const Matrix6 c = out.tangent;
  out.tangent = c - c.col(2) * c.row(2) / c(2, 2);
  out.tangent.row(2).setZero();
  out.tangent.col(2).setZero();
  out.state.thickness_strain = trial(2);
  out.strain_33 = trial(2);
  return out;
}

}  // namespace

PlasticResponse plastic_return(const IsotropicMaterial& material, StressState state,
                               const Vector6& strain, const PlasticState& committed,
                               Scalar delta_t, bool want_tangent) {
  if (!material.plasticity().general()) {
    if (state != StressState::PlaneStress) return return_3d(material, strain, committed, delta_t);
    return plane_stress(
        material,
        [&](const Vector6& trial) { return return_3d(material, trial, committed, delta_t); },
        strain, committed, delta_t, 1.0e-12);
  }
  if (state != StressState::ThreeDimensional && !material.plasticity().plane_compatible()) {
    throw ConfigError("material '" + material.name() +
                      "': a plane model needs the out-of-plane axis z along the rolling, "
                      "transverse or normal direction of its Hill48 frame (it has no "
                      "out-of-plane shear strain, which any other frame couples to the "
                      "in-plane flow)");
  }
  if (state != StressState::PlaneStress) {
    return return_general_3d(material, strain, committed, delta_t, want_tangent);
  }
  // sigma_33 to 1e-13 of the stress: a residual sigma_33 moves the yield
  // function of the elastic re-check (whose predictor zeroes it) by about as
  // much, and that must stay below the re-check's 1e-12 sigma_y.
  return plane_stress(
      material,
      [&](const Vector6& trial) {
        return return_general_3d(material, trial, committed, delta_t, true);
      },
      strain, committed, delta_t, 1.0e-13);
}

Scalar von_mises_stress(const Vector6& s) {
  const Scalar a = s(0) - s(1);
  const Scalar b = s(1) - s(2);
  const Scalar c = s(2) - s(0);
  return std::sqrt(0.5 * (a * a + b * b + c * c) +
                   3.0 * (s(3) * s(3) + s(4) * s(4) + s(5) * s(5)));
}

}  // namespace sparlab
