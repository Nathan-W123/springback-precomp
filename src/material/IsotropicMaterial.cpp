// Where this fits: see include/sparlab/material/IsotropicMaterial.hpp

#include "sparlab/material/IsotropicMaterial.hpp"

#include "sparlab/core/Exceptions.hpp"

#include <Eigen/Geometry>

#include <cmath>
#include <sstream>
#include <string>
#include <utility>

namespace sparlab {

IsotropicMaterial::IsotropicMaterial(Scalar youngs_modulus, Scalar poisson_ratio,
                                     Scalar density, std::string name)
    : e_(youngs_modulus), nu_(poisson_ratio), rho_(density), name_(std::move(name)) {
  if (!(e_ > 0.0)) {
    std::ostringstream os;
    os << "material '" << name_ << "' needs a positive Young's modulus (got " << e_
       << " Pa)";
    throw ConfigError(os.str());
  }
  if (!(nu_ > -1.0 && nu_ < 0.5)) {
    std::ostringstream os;
    os << "material '" << name_ << "' needs a Poisson ratio in (-1, 0.5) for a "
       << "positive-definite constitutive matrix (got " << nu_ << ")";
    throw ConfigError(os.str());
  }
  if (!(rho_ >= 0.0)) {
    std::ostringstream os;
    os << "material '" << name_ << "' needs a non-negative density (got " << rho_
       << " kg/m^3)";
    throw ConfigError(os.str());
  }
}

Matrix3 IsotropicMaterial::plane_stress_matrix() const {
  Matrix3 d = Matrix3::Zero();
  const Scalar f = e_ / (1.0 - nu_ * nu_);
  d(0, 0) = f;
  d(0, 1) = f * nu_;
  d(1, 0) = f * nu_;
  d(1, 1) = f;
  d(2, 2) = f * (1.0 - nu_) / 2.0;
  return d;
}

Matrix3 IsotropicMaterial::plane_strain_matrix() const {
  Matrix3 d = Matrix3::Zero();
  const Scalar f = e_ / ((1.0 + nu_) * (1.0 - 2.0 * nu_));
  d(0, 0) = f * (1.0 - nu_);
  d(0, 1) = f * nu_;
  d(1, 0) = f * nu_;
  d(1, 1) = f * (1.0 - nu_);
  d(2, 2) = f * (1.0 - 2.0 * nu_) / 2.0;
  return d;
}

Matrix6 IsotropicMaterial::three_dimensional_matrix() const {
  Matrix6 d = Matrix6::Zero();
  const Scalar lambda = lame_lambda();
  const Scalar g = shear_modulus();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) d(i, j) = lambda;
    d(i, i) = lambda + 2.0 * g;
    d(3 + i, 3 + i) = g;
  }
  return d;
}

void IsotropicMaterial::set_thermal(Scalar expansion, Scalar reference_temperature,
                                    Scalar conductivity) {
  if (!std::isfinite(expansion) || !std::isfinite(reference_temperature) ||
      !std::isfinite(conductivity)) {
    throw ConfigError("material '" + name_ + "' has a non-finite thermal property");
  }
  if (conductivity < 0.0) {
    std::ostringstream os;
    os << "material '" << name_ << "' needs a non-negative conductivity (got "
       << conductivity << " W/(m K))";
    throw ConfigError(os.str());
  }
  alpha_ = expansion;
  t_ref_ = reference_temperature;
  k_ = conductivity;
}

Scalar PlasticityParameters::yield(Scalar alpha) const {
  return yield_stress + hardening_modulus * alpha +
         saturation_stress * -std::expm1(-saturation_rate * alpha);
}

Scalar PlasticityParameters::yield_slope(Scalar alpha) const {
  return hardening_modulus +
         saturation_stress * saturation_rate * std::exp(-saturation_rate * alpha);
}

Scalar PlasticityParameters::isotropic_energy(Scalar alpha) const {
  // int_0^a H s + Q (1 - e^(-delta s)) ds = H a^2/2 + Q (a + expm1(-delta a) / delta).
  Scalar out = 0.5 * hardening_modulus * alpha * alpha;
  if (saturation_stress != 0.0) {
    out += saturation_stress * (alpha + std::expm1(-saturation_rate * alpha) / saturation_rate);
  }
  return out;
}

Matrix6 von_mises_yield_matrix() {
  // (3/2) |dev xi|^2 = 1/2 [(x11 - x22)^2 + (x22 - x33)^2 + (x33 - x11)^2]
  //                    + 3 (x12^2 + x23^2 + x31^2).
  Matrix6 p = Matrix6::Zero();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) p(i, j) = -0.5;
    p(i, i) = 1.0;
    p(3 + i, 3 + i) = 3.0;
  }
  return p;
}

bool PlasticityParameters::symmetric_tangent() const {
  for (int i = 0; i < num_backstresses; ++i) {
    if (backstresses[static_cast<std::size_t>(i)].recovery > 0.0) return false;
  }
  return true;
}

namespace {

/// The map of tensorial Voigt stress components from global axes to the axes
/// that are the rows of `q`: the Voigt form of sigma' = q sigma q^T, column
/// by column as the image of each basis tensor (e_i e_j + e_j e_i for a
/// shear, whose Voigt component multiplies both).
Matrix6 stress_rotation(const Matrix3& q) {
  static const int ii[6] = {0, 1, 2, 0, 1, 2};
  static const int jj[6] = {0, 1, 2, 1, 2, 0};
  Matrix6 t;
  for (int k = 0; k < 6; ++k) {
    Matrix3 e = Matrix3::Zero();
    e(ii[k], jj[k]) = 1.0;
    e(jj[k], ii[k]) = 1.0;
    const Matrix3 r = q * e * q.transpose();
    for (int row = 0; row < 6; ++row) t(row, k) = r(ii[row], jj[row]);
  }
  return t;
}

bool finite_positive(Scalar v) { return std::isfinite(v) && v > 0.0; }

/// The Hill48 coefficients of a calibration, the material axes and the
/// yield matrix in global axes.
void derive_hill(Hill48Parameters& h, Matrix6& yield_matrix, const std::string& name) {
  const auto fail = [&](const std::string& what) {
    throw ConfigError("material '" + name + "': " + what);
  };
  switch (h.calibration) {
    case HillCalibration::RValues: {
      if (!finite_positive(h.r0) || !finite_positive(h.r45) || !finite_positive(h.r90)) {
        fail("the Hill48 r-values must be finite and positive");
      }
      h.G = 1.0 / (1.0 + h.r0);
      h.H = h.r0 / (1.0 + h.r0);
      h.F = h.r0 / (h.r90 * (1.0 + h.r0));
      h.N = (h.r0 + h.r90) * (1.0 + 2.0 * h.r45) / (2.0 * h.r90 * (1.0 + h.r0));
      h.L = h.shear_l;
      h.M = h.shear_m;
      break;
    }
    case HillCalibration::StressRatios: {
      if (!finite_positive(h.sigma45) || !finite_positive(h.sigma90) ||
          !finite_positive(h.sigma_biaxial)) {
        fail("the Hill48 yield stress ratios must be finite and positive");
      }
      // G + H = 1, F + H = a, F + G = b.
      const Scalar a = 1.0 / (h.sigma90 * h.sigma90);
      const Scalar b = 1.0 / (h.sigma_biaxial * h.sigma_biaxial);
      h.F = 0.5 * (a + b - 1.0);
      h.G = 0.5 * (1.0 - a + b);
      h.H = 0.5 * (1.0 + a - b);
      h.N = 0.5 * (4.0 / (h.sigma45 * h.sigma45) - h.F - h.G);
      h.L = h.shear_l;
      h.M = h.shear_m;
      break;
    }
    case HillCalibration::Coefficients:
      break;
  }
  const Scalar c[6] = {h.F, h.G, h.H, h.L, h.M, h.N};
  for (Scalar v : c) {
    if (!std::isfinite(v)) fail("the Hill48 coefficients must be finite");
  }
  // Positive definite on deviators: the normal part is
  // (F + H) x^2 + 2 H x y + (G + H) y^2 in x = s22 - s33, y = s33 - s11.
  if (!(h.L > 0.0 && h.M > 0.0 && h.N > 0.0) || !(h.F + h.H > 0.0) ||
      !(h.F * h.G + h.G * h.H + h.H * h.F > 0.0)) {
    std::ostringstream os;
    os << "the Hill48 coefficients F = " << h.F << ", G = " << h.G << ", H = " << h.H
       << ", L = " << h.L << ", M = " << h.M << ", N = " << h.N
       << " do not make a convex yield surface (it needs L, M, N > 0, F + H > 0 and "
          "FG + GH + HF > 0)";
    fail(os.str());
  }
  // The frame: ND, RD projected normal to it, TD = ND x RD.
  if (!h.rolling_direction.allFinite() || !h.sheet_normal.allFinite() ||
      !(h.sheet_normal.norm() > 0.0) || !(h.rolling_direction.norm() > 0.0)) {
    fail("the rolling direction and the sheet normal must be finite non-zero vectors");
  }
  const Vector3 nd = h.sheet_normal.normalized();
  Vector3 rd = h.rolling_direction.normalized();
  rd -= rd.dot(nd) * nd;
  if (!(rd.norm() > 1.0e-6)) fail("the rolling direction is parallel to the sheet normal");
  rd.normalize();
  const Vector3 td = nd.cross(rd);
  h.axes.row(0) = rd.transpose();
  h.axes.row(1) = td.transpose();
  h.axes.row(2) = nd.transpose();
  h.z_on_axis = false;
  for (int r = 0; r < 3; ++r) {
    h.z_on_axis = h.z_on_axis || std::abs(std::abs(h.axes(r, 2)) - 1.0) <= 1.0e-12;
  }

  Matrix6 pm = Matrix6::Zero();
  pm(0, 0) = h.G + h.H;
  pm(1, 1) = h.F + h.H;
  pm(2, 2) = h.F + h.G;
  pm(0, 1) = pm(1, 0) = -h.H;
  pm(1, 2) = pm(2, 1) = -h.F;
  pm(0, 2) = pm(2, 0) = -h.G;
  pm(3, 3) = 2.0 * h.N;  // 12
  pm(4, 4) = 2.0 * h.L;  // 23
  pm(5, 5) = 2.0 * h.M;  // 31
  const Matrix6 t = stress_rotation(h.axes);
  yield_matrix = t.transpose() * pm * t;
}

}  // namespace

void IsotropicMaterial::set_plasticity(const PlasticityParameters& parameters) {
  PlasticityParameters p = parameters;
  const Scalar values[] = {p.yield_stress, p.hardening_modulus, p.kinematic_hardening_modulus,
                           p.saturation_stress, p.saturation_rate};
  for (Scalar v : values) {
    if (!std::isfinite(v) || v < 0.0) {
      throw ConfigError("material '" + name_ +
                        "' needs finite, non-negative plasticity parameters");
    }
  }
  if (p.saturation_stress > 0.0 && !(p.saturation_rate > 0.0)) {
    throw ConfigError("material '" + name_ +
                      "' has a Voce saturation stress but no positive saturation rate");
  }
  if (p.num_backstresses < 0 || p.num_backstresses > kMaxBackstresses) {
    std::ostringstream os;
    os << "material '" << name_ << "' has " << p.num_backstresses
       << " backstresses; at most " << kMaxBackstresses << " are supported";
    throw ConfigError(os.str());
  }
  for (int i = 0; i < p.num_backstresses; ++i) {
    const Backstress& b = p.backstresses[static_cast<std::size_t>(i)];
    if (!finite_positive(b.modulus) || !std::isfinite(b.recovery) || b.recovery < 0.0) {
      std::ostringstream os;
      os << "material '" << name_ << "': backstress " << i + 1
         << " needs a finite positive modulus and a finite non-negative recovery";
      throw ConfigError(os.str());
    }
  }
  if (!p.enabled() && (p.hardening_modulus > 0.0 || p.kinematic_hardening_modulus > 0.0 ||
                       p.saturation_stress > 0.0 || p.num_backstresses > 0)) {
    throw ConfigError("material '" + name_ + "' has hardening parameters but no yield stress");
  }
  if (!p.enabled() && p.criterion != YieldCriterion::VonMises) {
    throw ConfigError("material '" + name_ + "' has a yield criterion but no yield stress");
  }
  if (p.general() && p.kinematic_terms() > kMaxBackstresses) {
    std::ostringstream os;
    os << "material '" << name_ << "' has " << p.num_backstresses
       << " backstresses and Prager's kinematic hardening modulus, which the general return "
          "integrates as one more: at most "
       << kMaxBackstresses << " in all";
    throw ConfigError(os.str());
  }
  if (p.criterion == YieldCriterion::Hill48) {
    derive_hill(p.hill, p.yield_matrix, name_);
  } else {
    p.yield_matrix = von_mises_yield_matrix();
  }
  plasticity_ = p;
}

Vector IsotropicMaterial::thermal_strain(StressState state, Scalar delta_t) const {
  const Scalar e0 = alpha_ * delta_t;
  switch (state) {
    case StressState::PlaneStress: {
      Vector v(3);
      v << e0, e0, 0.0;
      return v;
    }
    case StressState::PlaneStrain: {
      // eps_zz = 0 restrains the out-of-plane expansion, which reappears as
      // nu alpha dT in each in-plane direction.
      Vector v(3);
      v << (1.0 + nu_) * e0, (1.0 + nu_) * e0, 0.0;
      return v;
    }
    case StressState::ThreeDimensional: {
      Vector v = Vector::Zero(6);
      v.head(3).setConstant(e0);
      return v;
    }
  }
  throw ConfigError("unhandled stress state");
}

Matrix IsotropicMaterial::constitutive(StressState state) const {
  switch (state) {
    case StressState::PlaneStress: return plane_stress_matrix();
    case StressState::PlaneStrain: return plane_strain_matrix();
    case StressState::ThreeDimensional: return three_dimensional_matrix();
  }
  throw ConfigError("unhandled stress state");
}

}  // namespace sparlab
