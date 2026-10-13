/// \file test_anisotropy.cpp
/// \brief Hill48 anisotropic plasticity and Chaboche (multi-backstress
///        Armstrong-Frederick) kinematic hardening: the coefficients and the
///        material frame, the general closest-point return at a material
///        point, and the elastoplastic element with its non-symmetric
///        tangent.
///
/// Every check is exact or a derivative check. Uniaxial stress along any
/// direction of the sheet is a proportional path for Hill48 with isotropic
/// hardening, so the directional yield stress, the r-value and the linear
/// hardening curve are compared with their closed forms to round-off. For
/// von Mises with backstresses the flow direction of a uniaxial branch is
/// fixed, over which the exponential integration of each backstress is
/// exact: monotonic, reversed (Bauschinger) and cyclic branches follow the
/// closed-form Armstrong-Frederick solutions to round-off in one step or
/// many, and backward Euler converges to them at first order. The von Mises
/// limit of the general return is the radial return to round-off.
/// Consistent tangents are compared with central differences of the stress
/// and of the element's internal force.
#include "TestSupport.hpp"

#include "sparlab/core/Exceptions.hpp"
#include "sparlab/fem/Elastoplastic.hpp"
#include "sparlab/io/Config.hpp"
#include "sparlab/material/Plasticity.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace sparlab;
using namespace sparlab::testing;
using Catch::Approx;

namespace {

const Scalar kPi = 3.14159265358979323846;

/// The constitutive law of a test: Hill48 from r-values in a frame, isotropic
/// hardening, Prager and Chaboche backstresses.
struct Law {
  bool hill = false;
  Scalar r0 = 1.0, r45 = 1.0, r90 = 1.0;
  Vector3 rd = Vector3::UnitX();
  Vector3 nd = Vector3::UnitZ();
  Scalar yield = 250.0e6;
  Scalar h = 0.0;
  Scalar hk = 0.0;
  Scalar q = 0.0;
  Scalar delta = 0.0;
  std::vector<Backstress> back;
  KinematicIntegration integration = KinematicIntegration::Exponential;
};

IsotropicMaterial make(const Law& law) {
  IsotropicMaterial m(200.0e9, 0.3, 7800.0, "sheet");
  PlasticityParameters p;
  p.yield_stress = law.yield;
  p.hardening_modulus = law.h;
  p.kinematic_hardening_modulus = law.hk;
  p.saturation_stress = law.q;
  p.saturation_rate = law.delta;
  if (law.hill) {
    p.criterion = YieldCriterion::Hill48;
    p.hill.r0 = law.r0;
    p.hill.r45 = law.r45;
    p.hill.r90 = law.r90;
    p.hill.rolling_direction = law.rd;
    p.hill.sheet_normal = law.nd;
  }
  p.num_backstresses = static_cast<int>(law.back.size());
  for (std::size_t i = 0; i < law.back.size(); ++i) p.backstresses[i] = law.back[i];
  p.kinematic_integration = law.integration;
  m.set_plasticity(p);
  return m;
}

/// The rolling direction at `degrees` from x in the x-y plane.
Vector3 in_plane(Scalar degrees) {
  const Scalar a = degrees * kPi / 180.0;
  return Vector3(std::cos(a), std::sin(a), 0.0);
}

/// A material rolled at theta from x, i.e. loaded along x at -theta from its
/// rolling direction; with r-values (1.9, 1.5, 2.3) unless given.
Law hill_law(Scalar rd_degrees, Scalar r0 = 1.9, Scalar r45 = 1.5, Scalar r90 = 2.3) {
  Law law;
  law.hill = true;
  law.r0 = r0;
  law.r45 = r45;
  law.r90 = r90;
  law.rd = in_plane(rd_degrees);
  return law;
}

/// The return at a strain whose components `free` Newton finds so that
/// a sigma = 0 (one row of `a` per free component); the other components
/// are as given.
PlasticResponse drive(const IsotropicMaterial& m, StressState state, Vector6 strain,
                      const std::vector<int>& free, const Matrix& a,
                      const PlasticState& committed, Scalar delta_t = 0.0) {
  const int n = static_cast<int>(free.size());
  PlasticResponse r;
  for (int it = 0; it < 60; ++it) {
    r = plastic_return(m, state, strain, committed, delta_t);
    const Vector residual = a * r.stress;
    const Scalar scale = std::max(r.stress.cwiseAbs().maxCoeff(), m.plasticity().yield_stress);
    if (residual.cwiseAbs().maxCoeff() <= 1.0e-13 * scale) return r;
    Matrix j(n, n);
    for (int c = 0; c < n; ++c) j.col(c) = a * r.tangent.col(free[static_cast<std::size_t>(c)]);
    const Vector step = j.lu().solve(residual);
    for (int c = 0; c < n; ++c) strain(free[static_cast<std::size_t>(c)]) -= step(c);
  }
  FAIL("the stress-state iteration did not converge");
  return r;
}

/// Uniaxial stress sigma_11 at the axial strain e11: in 3-D the other five
/// strain components found so that their stresses vanish; in plane stress
/// eps_22 and gamma_12 (sigma_33 = 0 by the return).
PlasticResponse uniaxial(const IsotropicMaterial& m, StressState state, Scalar e11,
                         const PlasticState& committed, Scalar delta_t = 0.0) {
  const std::vector<int> free = state == StressState::PlaneStress ? std::vector<int>{1, 3}
                                                                  : std::vector<int>{1, 2, 3, 4, 5};
  Matrix a = Matrix::Zero(static_cast<Eigen::Index>(free.size()), 6);
  for (std::size_t i = 0; i < free.size(); ++i) a(static_cast<Eigen::Index>(i), free[i]) = 1.0;
  Vector6 strain = committed.plastic_strain;
  strain(0) = e11;
  const Scalar th = m.thermal_expansion() * delta_t;
  for (int i = 1; i < 3; ++i) {
    strain(i) = committed.plastic_strain(i) + th -
                m.poisson_ratio() * (e11 - committed.plastic_strain(0) - th);
  }
  return drive(m, state, strain, free, a, committed, delta_t);
}

/// Equibiaxial plane stress sigma_11 = sigma_22 at eps_11 = e11.
PlasticResponse equibiaxial(const IsotropicMaterial& m, Scalar e11) {
  Matrix a = Matrix::Zero(2, 6);
  a(0, 1) = 1.0;
  a(0, 0) = -1.0;
  a(1, 3) = 1.0;
  Vector6 strain = Vector6::Zero();
  strain(0) = strain(1) = e11;
  return drive(m, StressState::PlaneStress, strain, {1, 3}, a, PlasticState());
}

/// Hill's directional factor: sigma_theta / sigma_0 = phi^(-1/2) for the
/// uniaxial stress at theta from RD.
Scalar hill_phi(const Hill48Parameters& h, Scalar theta) {
  const Scalar s = std::sin(theta);
  const Scalar c = std::cos(theta);
  const Scalar c2 = std::cos(2.0 * theta);
  return h.F * s * s * s * s + h.G * c * c * c * c + h.H * c2 * c2 + 2.0 * h.N * s * s * c * c;
}

/// Hill's r-value at theta from RD.
Scalar hill_r(const Hill48Parameters& h, Scalar theta) {
  const Scalar s = std::sin(theta);
  const Scalar c = std::cos(theta);
  return (h.H + (2.0 * h.N - h.F - h.G - 4.0 * h.H) * s * s * c * c) / (h.F * s * s + h.G * c * c);
}

Vector6 random_strain(Scalar amplitude, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<Scalar> dist(-1.0, 1.0);
  Vector6 v;
  for (int i = 0; i < 6; ++i) v(i) = amplitude * dist(gen);
  return v;
}

/// The consistent tangent against central differences of the stress, over
/// the strain components the idealisation takes.
Scalar tangent_error(const IsotropicMaterial& m, StressState state, const Vector6& strain,
                     const PlasticState& committed, Scalar delta_t = 0.0) {
  const PlasticResponse r = plastic_return(m, state, strain, committed, delta_t);
  const Scalar h = 1.0e-8;
  Scalar worst = 0.0;
  for (int j = 0; j < 6; ++j) {
    if (state != StressState::ThreeDimensional && (j == 4 || j == 5)) continue;
    if (state == StressState::PlaneStress && j == 2) continue;
    Vector6 up = strain;
    Vector6 um = strain;
    up(j) += h;
    um(j) -= h;
    const Vector6 fd = (plastic_return(m, state, up, committed, delta_t).stress -
                        plastic_return(m, state, um, committed, delta_t).stress) /
                       (2.0 * h);
    worst = std::max(worst, (fd - r.tangent.col(j)).cwiseAbs().maxCoeff());
  }
  return worst / r.tangent.cwiseAbs().maxCoeff();
}

Scalar relative(const Matrix& a, const Matrix& b) {
  return (a - b).cwiseAbs().maxCoeff() / std::max(b.cwiseAbs().maxCoeff(), 1.0e-300);
}

/// |f| of a returned state, sigma_bar(dev sigma - sum alpha_i) - sigma_y
/// (the deviator taken first, so that the pressure's round-off stays out).
Scalar yield_function(const IsotropicMaterial& m, const PlasticResponse& r) {
  Vector6 xi = r.stress - r.state.back_stress;
  xi.head(3).array() -= (r.stress(0) + r.stress(1) + r.stress(2)) / 3.0;
  return std::abs(equivalent_stress(m.plasticity(), xi) -
                  m.plasticity().yield(r.state.equivalent_plastic_strain));
}

/// Tensor of tensorial (stress) and of engineering-shear (strain) Voigt
/// components, and back.
Matrix3 stress_tensor(const Vector6& v) {
  Matrix3 t;
  t << v(0), v(3), v(5), v(3), v(1), v(4), v(5), v(4), v(2);
  return t;
}
Vector6 stress_voigt(const Matrix3& t) {
  Vector6 v;
  v << t(0, 0), t(1, 1), t(2, 2), t(0, 1), t(1, 2), t(2, 0);
  return v;
}
Vector6 rotate_stress(const Matrix3& q, const Vector6& v) {
  return stress_voigt(q * stress_tensor(v) * q.transpose());
}
Vector6 rotate_strain(const Matrix3& q, const Vector6& v) {
  Vector6 t = v;
  t.tail(3) *= 0.5;
  Vector6 out = rotate_stress(q, t);
  out.tail(3) *= 2.0;
  return out;
}

/// The Armstrong-Frederick sum's uniaxial stress on a monotonic branch from
/// a virgin state: sigma = sigma_y(p) + sum C_i / gamma_i (1 - e^(-gamma_i p)).
Scalar chaboche_stress(const PlasticityParameters& p, Scalar plastic) {
  Scalar out = p.yield(plastic);
  for (int i = 0; i < p.kinematic_terms(); ++i) {
    const Backstress b = p.kinematic_term(i);
    out += b.recovery > 0.0 ? b.modulus / b.recovery * -std::expm1(-b.recovery * plastic)
                            : b.modulus * plastic;
  }
  return out;
}

}  // namespace

TEST_CASE("Hill48 coefficients, frame and yield matrix follow the calibration",
          "[plasticity][anisotropy][material]") {
  // From r-values.
  const IsotropicMaterial m = make(hill_law(0.0));
  const Hill48Parameters& h = m.plasticity().hill;
  REQUIRE(h.F == Approx(1.9 / (2.3 * 2.9)).epsilon(1e-15));
  REQUIRE(h.G == Approx(1.0 / 2.9).epsilon(1e-15));
  REQUIRE(h.H == Approx(1.9 / 2.9).epsilon(1e-15));
  REQUIRE(h.N == Approx((1.9 + 2.3) * 4.0 / (2.0 * 2.3 * 2.9)).epsilon(1e-15));
  REQUIRE(h.L == 1.5);
  REQUIRE(h.M == 1.5);
  // r0 = H / G and r90 = H / F.
  REQUIRE(h.H / h.G == Approx(1.9).epsilon(1e-15));
  REQUIRE(h.H / h.F == Approx(2.3).epsilon(1e-15));
  REQUIRE(hill_r(h, kPi / 4.0) == Approx(1.5).epsilon(1e-14));

  // r = 1 is von Mises.
  const IsotropicMaterial iso = make(hill_law(0.0, 1.0, 1.0, 1.0));
  const Hill48Parameters& hv = iso.plasticity().hill;
  for (const Scalar c : {hv.F, hv.G, hv.H}) REQUIRE(c == Approx(0.5).epsilon(1e-15));
  for (const Scalar c : {hv.L, hv.M, hv.N}) REQUIRE(c == Approx(1.5).epsilon(1e-15));
  REQUIRE(relative(iso.plasticity().yield_matrix, von_mises_yield_matrix()) <= 1.0e-15);
  REQUIRE(iso.plasticity().general());
  REQUIRE(iso.plasticity().symmetric_tangent());
  // sigma_bar of von Mises is the von Mises stress.
  const Vector6 s = random_strain(3.0e8, 11u);
  REQUIRE(equivalent_stress(iso.plasticity(), s) == Approx(von_mises_stress(s)).epsilon(1e-14));

  // Pressure-insensitive in any frame: every row of P sums to zero over the
  // normal columns (P annihilates the hydrostatic tensor).
  Law tilted = hill_law(0.0);
  tilted.rd = Vector3(1.0, 0.3, -0.2);
  tilted.nd = Vector3(0.2, -0.4, 1.0);
  const IsotropicMaterial mt = make(tilted);
  const Matrix6& pt = mt.plasticity().yield_matrix;
  Vector6 hydro = Vector6::Zero();
  hydro.head(3).setOnes();
  REQUIRE((pt * hydro).cwiseAbs().maxCoeff() <= 1.0e-14 * pt.cwiseAbs().maxCoeff());
  REQUIRE(relative(pt, pt.transpose()) <= 1.0e-15);
  // The frame is orthonormal and right-handed, RD projected normal to ND.
  const Matrix3& axes = mt.plasticity().hill.axes;
  REQUIRE(relative(axes * axes.transpose(), Matrix3::Identity()) <= 1.0e-15);
  REQUIRE(axes.determinant() == Approx(1.0).epsilon(1e-15));
  REQUIRE((axes.row(2).transpose() - tilted.nd.normalized()).norm() <= 1.0e-15);

  // With r0 = r90 a quarter turn of the rolling direction in the sheet plane
  // leaves the criterion unchanged; with r0 != r90 it does not.
  REQUIRE(relative(make(hill_law(90.0, 1.7, 1.2, 1.7)).plasticity().yield_matrix,
                   make(hill_law(0.0, 1.7, 1.2, 1.7)).plasticity().yield_matrix) <= 1.0e-15);
  REQUIRE(relative(make(hill_law(90.0)).plasticity().yield_matrix,
                   make(hill_law(0.0)).plasticity().yield_matrix) > 0.01);

  // The stress-ratio calibration of the same sheet, and the coefficients
  // given directly, reproduce the criterion.
  IsotropicMaterial other(200.0e9, 0.3, 7800.0, "sheet");
  PlasticityParameters p = m.plasticity();
  p.hill.calibration = HillCalibration::StressRatios;
  p.hill.sigma45 = 1.0 / std::sqrt(hill_phi(h, kPi / 4.0));
  p.hill.sigma90 = 1.0 / std::sqrt(hill_phi(h, kPi / 2.0));
  p.hill.sigma_biaxial = 1.0 / std::sqrt(h.F + h.G);
  p.hill.F = p.hill.G = p.hill.H = p.hill.N = 0.0;
  other.set_plasticity(p);
  REQUIRE(other.plasticity().hill.F == Approx(h.F).epsilon(1e-14));
  REQUIRE(other.plasticity().hill.G == Approx(h.G).epsilon(1e-14));
  REQUIRE(other.plasticity().hill.H == Approx(h.H).epsilon(1e-14));
  REQUIRE(other.plasticity().hill.N == Approx(h.N).epsilon(1e-14));
  REQUIRE(relative(other.plasticity().yield_matrix, m.plasticity().yield_matrix) <= 1.0e-14);
  p = m.plasticity();
  p.hill.calibration = HillCalibration::Coefficients;
  p.hill.r0 = p.hill.r45 = p.hill.r90 = -1.0;  // ignored
  other.set_plasticity(p);
  REQUIRE(relative(other.plasticity().yield_matrix, m.plasticity().yield_matrix) == 0.0);
  // The copy with a new modulus keeps everything.
  REQUIRE(m.with_youngs_modulus(100.0e9).plasticity().yield_matrix ==
          m.plasticity().yield_matrix);

  // Refusals.
  const auto refuses = [](PlasticityParameters bad) {
    IsotropicMaterial x(200.0e9, 0.3, 7800.0, "bad");
    REQUIRE_THROWS_AS(x.set_plasticity(bad), ConfigError);
  };
  p = m.plasticity();
  p.hill.r45 = 0.0;
  refuses(p);
  p = m.plasticity();
  p.hill.r0 = std::numeric_limits<Scalar>::quiet_NaN();
  refuses(p);
  p = m.plasticity();
  p.hill.rolling_direction = Vector3(0.0, 0.0, 2.0);  // along the normal
  refuses(p);
  p = m.plasticity();
  p.hill.sheet_normal = Vector3::Zero();
  refuses(p);
  p = m.plasticity();
  p.hill.shear_l = 0.0;
  refuses(p);
  p = m.plasticity();
  p.hill.calibration = HillCalibration::Coefficients;
  p.hill.F = p.hill.G = -1.0;  // F + H > 0 but FG + GH + HF < 0
  p.hill.H = 1.2;
  refuses(p);
  p = m.plasticity();
  p.hill.calibration = HillCalibration::StressRatios;
  p.hill.sigma45 = 1.0;
  p.hill.sigma90 = 0.3;  // F + H = 11, F + G = 1: G + H = 1 gives G = -4.5
  p.hill.sigma_biaxial = 1.0;
  refuses(p);
  p = m.plasticity();
  p.yield_stress = 0.0;  // a criterion without a yield stress
  refuses(p);
  PlasticityParameters c;
  c.yield_stress = 250.0e6;
  c.num_backstresses = kMaxBackstresses + 1;
  refuses(c);
  c.num_backstresses = 1;
  c.backstresses[0] = {0.0, 10.0};  // no modulus
  refuses(c);
  c.backstresses[0] = {1.0e9, -1.0};
  refuses(c);
  c.backstresses[0] = {1.0e9, 10.0};
  c.yield_stress = 0.0;  // backstresses without a yield stress
  refuses(c);
  c.yield_stress = 250.0e6;
  c.num_backstresses = kMaxBackstresses;
  for (Backstress& b : c.backstresses) b = {1.0e9, 10.0};
  c.kinematic_hardening_modulus = 1.0e9;  // Prager makes a fifth
  refuses(c);
  c.kinematic_hardening_modulus = 0.0;
  IsotropicMaterial ok(200.0e9, 0.3, 7800.0, "chaboche");
  REQUIRE_NOTHROW(ok.set_plasticity(c));
  REQUIRE(ok.plasticity().general());
  REQUIRE_FALSE(ok.plasticity().symmetric_tangent());
  c.backstresses[1].recovery = 0.0;
  for (Backstress& b : c.backstresses) b.recovery = 0.0;
  ok.set_plasticity(c);
  REQUIRE(ok.plasticity().symmetric_tangent());
}

TEST_CASE("Hill48: uniaxial stress along any direction yields at sigma_theta with the r-value "
          "r_theta, and equibiaxial stress at sigma_b",
          "[plasticity][anisotropy][material]") {
  // Perfectly plastic: strained past yield along x in one step, the
  // uniaxial stress is sigma_theta = sigma_y phi(theta)^(-1/2) exactly, and
  // the plastic strain rates are those of the fixed flow direction, whose
  // width over thickness ratio is r_theta.
  const Hill48Parameters h = make(hill_law(0.0)).plasticity().hill;
  for (const StressState state : {StressState::ThreeDimensional, StressState::PlaneStress}) {
    for (const Scalar theta : {0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0}) {
      // Rolled at -theta from x: loaded along x at theta from RD.
      const IsotropicMaterial m = make(hill_law(-theta));
      const Scalar t = theta * kPi / 180.0;
      const Scalar sigma_theta = 250.0e6 / std::sqrt(hill_phi(h, t));
      const PlasticResponse r = uniaxial(m, state, 0.01, PlasticState());
      INFO(to_string(state) << ", theta = " << theta);
      REQUIRE(r.yielding);
      REQUIRE(r.stress(0) == Approx(sigma_theta).epsilon(1e-12));
      REQUIRE(equivalent_stress(m.plasticity(), r.stress) == Approx(250.0e6).epsilon(1e-12));
      const Vector6& ep = r.state.plastic_strain;
      const Scalar thickness = -(ep(0) + ep(1));  // isochoric
      if (state == StressState::ThreeDimensional) {
        REQUIRE(std::abs(ep(0) + ep(1) + ep(2)) <= 1.0e-15 * ep(0));
        REQUIRE(ep(2) == Approx(thickness).epsilon(1e-12));
        // No out-of-plane shear flow with the sheet normal along z.
        REQUIRE(std::abs(ep(4)) + std::abs(ep(5)) <= 1.0e-12 * ep(0));
      }
      REQUIRE(ep(1) / thickness == Approx(hill_r(h, t)).epsilon(1e-12));
      // The accumulated plastic strain is work-conjugate: sigma eps_p,11 =
      // sigma_bar alpha.
      REQUIRE(r.state.equivalent_plastic_strain ==
              Approx(r.stress(0) * ep(0) / 250.0e6).epsilon(1e-12));
    }
  }
  REQUIRE(hill_r(h, 0.0) == Approx(1.9).epsilon(1e-15));
  REQUIRE(hill_r(h, kPi / 2.0) == Approx(2.3).epsilon(1e-15));
  // Equibiaxial: sigma_b = sigma_y (F + G)^(-1/2) = sigma_y sqrt(r90 (1 +
  // r0) / (r0 + r90)), and the plastic strain ratio eps_22 / eps_11 = F / G
  // = r0 / r90.
  const IsotropicMaterial m = make(hill_law(0.0));
  const PlasticResponse b = equibiaxial(m, 0.01);
  REQUIRE(b.yielding);
  REQUIRE(b.stress(0) == Approx(250.0e6 * std::sqrt(2.3 * 2.9 / 4.2)).epsilon(1e-12));
  REQUIRE(b.stress(1) == Approx(b.stress(0)).epsilon(1e-13));
  REQUIRE(b.state.plastic_strain(1) / b.state.plastic_strain(0) ==
          Approx(1.9 / 2.3).epsilon(1e-12));
}

TEST_CASE("Hill48 with linear isotropic hardening follows the exact uniaxial curve along any "
          "direction in one step or many",
          "[plasticity][anisotropy][material]") {
  // sigma / k = sigma_y0 + H alpha with alpha = k eps_p (k = sigma_theta /
  // sigma_0), so sigma = E (k sigma_y0 + k^2 H eps) / (E + k^2 H).
  const Scalar hmod = 2.0e9;
  const Scalar e = 200.0e9;
  const Hill48Parameters h = make(hill_law(0.0)).plasticity().hill;
  for (const StressState state : {StressState::ThreeDimensional, StressState::PlaneStress}) {
    for (const Scalar theta : {0.0, 22.5, 45.0, 67.5, 90.0}) {
      Law law = hill_law(-theta);
      law.h = hmod;
      const IsotropicMaterial m = make(law);
      const Scalar k = 1.0 / std::sqrt(hill_phi(h, theta * kPi / 180.0));
      const Scalar target = 0.012;
      const Scalar exact = e * (k * 250.0e6 + k * k * hmod * target) / (e + k * k * hmod);
      INFO(to_string(state) << ", theta = " << theta);
      REQUIRE(uniaxial(m, state, target, PlasticState()).stress(0) ==
              Approx(exact).epsilon(1e-12));
      PlasticState point;
      PlasticResponse r;
      for (int step = 1; step <= 40; ++step) {
        r = uniaxial(m, state, target * step / 40.0, point);
        point = r.state;
      }
      REQUIRE(r.stress(0) == Approx(exact).epsilon(1e-12));
      const Scalar eps_p = target - exact / e;
      REQUIRE(point.plastic_strain(0) == Approx(eps_p).epsilon(1e-10));
      REQUIRE(point.equivalent_plastic_strain == Approx(k * eps_p).epsilon(1e-10));
      // Stored energy: sigma^2 / 2E + H alpha^2 / 2 (the elastic energy of
      // uniaxial stress).
      REQUIRE(r.energy == Approx(exact * exact / (2.0 * e) +
                                 0.5 * hmod * point.equivalent_plastic_strain *
                                     point.equivalent_plastic_strain)
                              .epsilon(1e-10));
    }
  }
}

TEST_CASE("the Hill48 return is frame-indifferent: a rotated sheet under the rotated strain "
          "returns the rotated state",
          "[plasticity][anisotropy][material]") {
  // RD at phi loaded at phi + psi is RD at 0 loaded at psi, and so for any
  // rotation of the sheet and of the strain history together - with Voce
  // hardening and recovering backstresses, along a non-proportional path.
  Law base = hill_law(0.0);
  base.q = 80.0e6;
  base.delta = 15.0;
  base.back = {{20.0e9, 200.0}, {3.0e9, 0.0}};
  const IsotropicMaterial m0 = make(base);
  const Scalar phi = 0.7;
  Matrix3 in_plane_turn;
  in_plane_turn << std::cos(phi), -std::sin(phi), 0.0, std::sin(phi), std::cos(phi), 0.0, 0.0,
      0.0, 1.0;
  const Matrix3 general_turn =
      Eigen::AngleAxisd(1.1, Vector3(0.3, -0.5, 0.8).normalized()).toRotationMatrix();
  for (const Matrix3& q : {in_plane_turn, general_turn}) {
    Law turned = base;
    turned.rd = q * base.rd;
    turned.nd = q * base.nd;
    const IsotropicMaterial m1 = make(turned);
    PlasticState s0;
    PlasticState s1;
    int yielded = 0;
    for (const unsigned seed : {3u, 5u, 8u}) {
      const Vector6 strain = random_strain(5.0e-3, seed);
      const PlasticResponse r0 = plastic_return(m0, StressState::ThreeDimensional, strain, s0);
      const PlasticResponse r1 =
          plastic_return(m1, StressState::ThreeDimensional, rotate_strain(q, strain), s1);
      REQUIRE(r1.yielding == r0.yielding);
      if (r0.yielding) ++yielded;
      REQUIRE(relative(r1.stress, rotate_stress(q, r0.stress)) <= 1.0e-12);
      REQUIRE(relative(r1.state.plastic_strain, rotate_strain(q, r0.state.plastic_strain)) <=
              1.0e-11);
      REQUIRE(relative(r1.state.back_stress, rotate_stress(q, r0.state.back_stress)) <= 1.0e-11);
      REQUIRE(r1.state.equivalent_plastic_strain ==
              Approx(r0.state.equivalent_plastic_strain).epsilon(1e-12));
      REQUIRE(r1.energy == Approx(r0.energy).epsilon(1e-12));
      s0 = r0.state;
      s1 = r1.state;
    }
    REQUIRE(yielded >= 2);
  }
}

TEST_CASE("the general return in the von Mises limit is the radial return",
          "[plasticity][anisotropy][material]") {
  // Hill48 with r = 1 (Prager's modulus mapped to a backstress), and von
  // Mises with Prager's rule given as one backstress without recovery,
  // against the radial return, along the random non-proportional path of
  // the J2 tangent test with every hardening mechanism and a temperature.
  Law radial;
  radial.h = 1.5e9;
  radial.hk = 4.0e9;
  radial.q = 100.0e6;
  radial.delta = 20.0;
  Law hill = radial;
  hill.hill = true;
  Law chaboche = radial;
  chaboche.hk = 0.0;
  chaboche.back = {{4.0e9, 0.0}};
  IsotropicMaterial reference = make(radial);
  reference.set_thermal(1.2e-5, 20.0, 50.0);
  for (const Law& law : {hill, chaboche}) {
    IsotropicMaterial general = make(law);
    general.set_thermal(1.2e-5, 20.0, 50.0);
    REQUIRE(general.plasticity().general());
    REQUIRE_FALSE(reference.plasticity().general());
    for (const StressState state :
         {StressState::ThreeDimensional, StressState::PlaneStrain, StressState::PlaneStress}) {
      PlasticState a;
      PlasticState b;
      const Vector6 first = random_strain(4.0e-3, 3u);
      const Vector6 second = first + random_strain(3.0e-3, 5u);
      const Vector6 back = second - 0.1 * (second - first);
      int step = 0;
      for (Vector6 strain : {first, second, back, back}) {
        if (state != StressState::ThreeDimensional) {
          strain(4) = strain(5) = 0.0;
          if (state == StressState::PlaneStrain) strain(2) = 0.0;
        }
        const Scalar dt = 10.0 * step++;
        const PlasticResponse ra = plastic_return(reference, state, strain, a, dt);
        const PlasticResponse rb = plastic_return(general, state, strain, b, dt);
        INFO(to_string(state) << ", step " << step << (law.hill ? ", Hill48" : ", Chaboche"));
        REQUIRE(ra.yielding == rb.yielding);
        REQUIRE(ra.state.loading == rb.state.loading);
        REQUIRE(relative(rb.stress, ra.stress) <= 1.0e-12);
        REQUIRE(relative(rb.tangent, ra.tangent) <= 1.0e-12);
        REQUIRE(relative(rb.state.plastic_strain, ra.state.plastic_strain) <= 1.0e-11);
        REQUIRE(relative(rb.state.back_stress, ra.state.back_stress) <= 1.0e-11);
        REQUIRE(rb.state.equivalent_plastic_strain ==
                Approx(ra.state.equivalent_plastic_strain).epsilon(1e-11));
        REQUIRE(rb.energy == Approx(ra.energy).epsilon(1e-11));
        REQUIRE(rb.strain_33 == Approx(ra.strain_33).epsilon(1e-10));
        a = ra.state;
        b = rb.state;
      }
    }
  }
}

TEST_CASE("Chaboche backstresses with Voce hardening follow the exact monotonic uniaxial "
          "curve: exponential integration exactly, backward Euler at first order",
          "[plasticity][anisotropy][material]") {
  // sigma = sigma_y(p) + sum C_i / gamma_i (1 - e^(-gamma_i p)), eps = p +
  // sigma / E, solved for p by bisection.
  Law law;
  law.yield = 170.0e6;
  law.q = 90.0e6;
  law.delta = 12.0;
  law.back = {{60.0e9, 600.0}, {8.0e9, 60.0}, {1.0e9, 0.0}};
  const IsotropicMaterial m = make(law);
  const PlasticityParameters& p = m.plasticity();
  const Scalar e = m.youngs_modulus();
  const auto exact = [&](Scalar strain) {
    Scalar lo = 0.0;
    Scalar hi = strain;
    for (int it = 0; it < 200; ++it) {
      const Scalar mid = 0.5 * (lo + hi);
      (mid + chaboche_stress(p, mid) / e - strain > 0.0 ? hi : lo) = mid;
    }
    return chaboche_stress(p, 0.5 * (lo + hi));
  };
  for (const Scalar target : {0.002, 0.02, 0.2}) {
    const Scalar sigma = exact(target);
    INFO("strain " << target);
    const PlasticResponse one = uniaxial(m, StressState::ThreeDimensional, target, PlasticState());
    REQUIRE(one.stress(0) == Approx(sigma).epsilon(1e-11));
    PlasticState point;
    PlasticResponse r;
    for (int step = 1; step <= 25; ++step) {
      r = uniaxial(m, StressState::ThreeDimensional, target * step / 25.0, point);
      point = r.state;
    }
    REQUIRE(r.stress(0) == Approx(sigma).epsilon(1e-11));
    // Each backstress X_i = (3/2) alpha_i,11 on its own curve.
    const Scalar plastic = point.equivalent_plastic_strain;
    REQUIRE(point.plastic_strain(0) == Approx(plastic).epsilon(1e-10));
    for (int i = 0; i < p.kinematic_terms(); ++i) {
      const Backstress b = p.kinematic_term(i);
      const Scalar x = b.recovery > 0.0
                           ? b.modulus / b.recovery * -std::expm1(-b.recovery * plastic)
                           : b.modulus * plastic;
      REQUIRE(1.5 * point.back_stresses[static_cast<std::size_t>(i)](0) ==
              Approx(x).epsilon(1e-10));
    }
    REQUIRE(point.back_stress(0) ==
            Approx((point.back_stresses[0] + point.back_stresses[1] + point.back_stresses[2])(0))
                .epsilon(1e-15));
  }
  // Backward Euler: the error at a fixed strain halves as the steps double.
  law.integration = KinematicIntegration::BackwardEuler;
  const IsotropicMaterial be = make(law);
  const Scalar target = 0.02;
  const Scalar sigma = exact(target);
  std::vector<Scalar> errors;
  for (const int steps : {10, 20, 40, 80}) {
    PlasticState point;
    PlasticResponse r;
    for (int step = 1; step <= steps; ++step) {
      r = uniaxial(be, StressState::ThreeDimensional, target * step / steps, point);
      point = r.state;
    }
    errors.push_back(std::abs(r.stress(0) - sigma) / sigma);
  }
  REQUIRE(errors.front() > 1.0e-4);
  for (std::size_t i = 1; i < errors.size(); ++i) {
    INFO("backward Euler errors " << errors[i - 1] << " -> " << errors[i]);
    REQUIRE(errors[i - 1] / errors[i] == Approx(2.0).epsilon(0.1));
  }
}

TEST_CASE("Chaboche: a reversed uniaxial branch follows the Armstrong-Frederick solution "
          "and reverse yield shows the Bauschinger effect",
          "[plasticity][anisotropy][material]") {
  // On a branch of direction nu = +-1 from (eps_p0, X_i0, p0):
  // X_i = nu C_i / gamma_i + (X_i0 - nu C_i / gamma_i) e^(-gamma_i D),
  // sigma = sum X_i + nu sigma_y(p0 + D), eps = eps_p0 + nu D + sigma / E.
  Law law;
  law.h = 0.5e9;
  law.q = 60.0e6;
  law.delta = 25.0;
  law.back = {{40.0e9, 400.0}, {5.0e9, 40.0}};
  const IsotropicMaterial m = make(law);
  const PlasticityParameters& p = m.plasticity();
  const Scalar e = m.youngs_modulus();
  struct Branch {
    Scalar eps_p = 0.0;
    Scalar plastic = 0.0;
    std::vector<Scalar> x;
  };
  const auto back_of = [&](const Branch& b, Scalar nu, Scalar d, int i) {
    const Backstress t = p.kinematic_term(i);
    const Scalar limit = nu * t.modulus / t.recovery;
    return limit + (b.x[static_cast<std::size_t>(i)] - limit) * std::exp(-t.recovery * d);
  };
  const auto stress_of = [&](const Branch& b, Scalar nu, Scalar d) {
    Scalar s = nu * p.yield(b.plastic + d);
    for (int i = 0; i < p.kinematic_terms(); ++i) s += back_of(b, nu, d, i);
    return s;
  };
  // The branch's stress at total strain eps (eps monotonic along nu).
  const auto branch = [&](const Branch& b, Scalar nu, Scalar eps) {
    Scalar lo = 0.0;
    Scalar hi = 1.0;
    for (int it = 0; it < 200; ++it) {
      const Scalar mid = 0.5 * (lo + hi);
      const Scalar strain = b.eps_p + nu * mid + stress_of(b, nu, mid) / e;
      (nu * (strain - eps) > 0.0 ? hi : lo) = mid;
    }
    return 0.5 * (lo + hi);
  };
  const Scalar e1 = 0.01;
  Branch virgin;
  virgin.x = {0.0, 0.0};
  const Scalar d1 = branch(virgin, 1.0, e1);
  const Scalar s1 = stress_of(virgin, 1.0, d1);
  const PlasticResponse loaded = uniaxial(m, StressState::ThreeDimensional, e1, PlasticState());
  REQUIRE(loaded.stress(0) == Approx(s1).epsilon(1e-11));
  Branch top;
  top.eps_p = d1;
  top.plastic = d1;
  top.x = {back_of(virgin, 1.0, d1, 0), back_of(virgin, 1.0, d1, 1)};
  for (int i = 0; i < 2; ++i) {
    REQUIRE(1.5 * loaded.state.back_stresses[static_cast<std::size_t>(i)](0) ==
            Approx(top.x[static_cast<std::size_t>(i)]).epsilon(1e-10));
  }
  // Reverse yield at sum X_i - sigma_y(p1): elastic up to it...
  const Scalar reverse = top.x[0] + top.x[1] - p.yield(d1);
  REQUIRE(reverse > -p.yield(d1));  // the Bauschinger effect: earlier than -sigma_y
  const Scalar e_reverse = e1 - (s1 - reverse) / e;
  const PlasticResponse at = uniaxial(m, StressState::ThreeDimensional, e_reverse * (1.0 + 1e-9),
                                      loaded.state);
  REQUIRE_FALSE(at.yielding);
  REQUIRE(at.stress(0) == Approx(reverse + e * e_reverse * 1e-9).epsilon(1e-9));
  REQUIRE(uniaxial(m, StressState::ThreeDimensional, e_reverse - 1.0e-6, loaded.state).yielding);
  // ... then on the reversed branch to -e1, in one step and in 20.
  const Scalar d2 = branch(top, -1.0, -e1);
  const Scalar s2 = stress_of(top, -1.0, d2);
  const PlasticResponse one = uniaxial(m, StressState::ThreeDimensional, -e1, loaded.state);
  REQUIRE(one.stress(0) == Approx(s2).epsilon(1e-11));
  PlasticState point = loaded.state;
  PlasticResponse r;
  for (int step = 1; step <= 20; ++step) {
    r = uniaxial(m, StressState::ThreeDimensional, e1 - 2.0 * e1 * step / 20.0, point);
    point = r.state;
  }
  REQUIRE(r.stress(0) == Approx(s2).epsilon(1e-11));
  REQUIRE(point.equivalent_plastic_strain == Approx(d1 + d2).epsilon(1e-10));
  for (int i = 0; i < 2; ++i) {
    REQUIRE(1.5 * point.back_stresses[static_cast<std::size_t>(i)](0) ==
            Approx(back_of(top, -1.0, d2, i)).epsilon(1e-10));
  }
}

TEST_CASE("Chaboche: symmetric strain cycles follow the half-cycle recursion and the "
          "stabilised loop peaks at sigma_y + (C / gamma) tanh(gamma D / 2)",
          "[plasticity][anisotropy][material]") {
  // Kinematic hardening only, one backstress. A half-cycle of plastic range
  // D from X_k: X_k+1 = nu C / gamma + (X_k - nu C / gamma) e^(-gamma D);
  // in the stabilised loop every half-cycle has the same D and
  // X_max = (C / gamma) tanh(gamma D / 2).
  const Scalar c = 30.0e9;
  const Scalar gamma = 150.0;
  Law law;
  law.back = {{c, gamma}};
  const IsotropicMaterial m = make(law);
  const Scalar sy = 250.0e6;
  const Scalar e = m.youngs_modulus();
  // Strain-controlled cycles of +-0.8 %, each half-cycle in one step.
  PlasticState point;
  Scalar x = 0.0;
  Scalar eps_p = 0.0;
  for (int half = 0; half < 12; ++half) {
    const Scalar nu = half % 2 == 0 ? 1.0 : -1.0;
    const PlasticResponse r = uniaxial(m, StressState::ThreeDimensional, nu * 0.008, point);
    REQUIRE(r.yielding);
    const Scalar range = std::abs(r.state.plastic_strain(0) - eps_p);
    const Scalar next = nu * c / gamma + (x - nu * c / gamma) * std::exp(-gamma * range);
    INFO("half-cycle " << half);
    REQUIRE(1.5 * r.state.back_stresses[0](0) == Approx(next).epsilon(1e-11));
    REQUIRE(r.stress(0) == Approx(next + nu * sy).epsilon(1e-11));
    REQUIRE(r.state.equivalent_plastic_strain ==
            Approx(point.equivalent_plastic_strain + range).epsilon(1e-12));
    x = next;
    eps_p = r.state.plastic_strain(0);
    point = r.state;
  }
  // The stabilised loop, started at its compressive tip, in one step and in
  // 25: the tensile tip at plastic strain D / 2.
  const Scalar range = 0.01;
  const Scalar x_max = c / gamma * std::tanh(gamma * range / 2.0);
  PlasticState tip;
  tip.plastic_strain << -range / 2.0, range / 4.0, range / 4.0, 0.0, 0.0, 0.0;
  tip.back_stresses[0] << -x_max / 1.5, x_max / 3.0, x_max / 3.0, 0.0, 0.0, 0.0;
  tip.back_stress = tip.back_stresses[0];
  const Scalar sigma_max = sy + x_max;
  const Scalar target = range / 2.0 + sigma_max / e;
  const PlasticResponse one = uniaxial(m, StressState::ThreeDimensional, target, tip);
  REQUIRE(one.stress(0) == Approx(sigma_max).epsilon(1e-11));
  REQUIRE(1.5 * one.state.back_stresses[0](0) == Approx(x_max).epsilon(1e-11));
  const Scalar start = -range / 2.0 - sigma_max / e;
  PlasticState state = tip;
  PlasticResponse r;
  for (int step = 1; step <= 25; ++step) {
    r = uniaxial(m, StressState::ThreeDimensional, start + (target - start) * step / 25.0, state);
    state = r.state;
  }
  REQUIRE(r.stress(0) == Approx(sigma_max).epsilon(1e-11));
  REQUIRE(state.plastic_strain(0) == Approx(range / 2.0).epsilon(1e-10));
}

TEST_CASE("the general consistent tangent is the derivative of the return, non-symmetric "
          "only with recovery",
          "[plasticity][anisotropy][material]") {
  struct Case {
    const char* name;
    Law law;
    bool symmetric;
  };
  Law hill = hill_law(25.0);
  hill.h = 1.0e9;
  hill.q = 80.0e6;
  hill.delta = 20.0;
  Law chaboche;
  chaboche.back = {{50.0e9, 500.0}, {6.0e9, 50.0}, {1.0e9, 0.0}};
  chaboche.q = 60.0e6;
  chaboche.delta = 15.0;
  Law both = hill;
  both.back = chaboche.back;
  both.hk = 2.0e9;
  both.rd = Vector3(1.0, 0.4, 0.3);  // tilted in 3-D
  both.nd = Vector3(-0.3, 0.1, 1.0);
  Law both_be = both;
  both_be.integration = KinematicIntegration::BackwardEuler;
  Law both_plane = both;  // rolled in the x-y plane: the plane states too
  both_plane.rd = in_plane(25.0);
  both_plane.nd = Vector3::UnitZ();
  Law prager = hill;
  prager.back = {{6.0e9, 0.0}};
  prager.hk = 1.0e9;
  const std::vector<Case> cases = {{"Hill48 + Voce", hill, true},
                                   {"Chaboche + Voce", chaboche, false},
                                   {"Hill48 + Chaboche + Prager + Voce", both, false},
                                   {"the same, backward Euler", both_be, false},
                                   {"the same, exponential, in the sheet plane", both_plane, false},
                                   {"Hill48 + two linear backstresses", prager, true}};
  for (const Case& c : cases) {
    IsotropicMaterial m = make(c.law);
    m.set_thermal(1.2e-5, 20.0, 50.0);
    REQUIRE(m.plasticity().symmetric_tangent() == c.symmetric);
    for (const StressState state :
         {StressState::ThreeDimensional, StressState::PlaneStrain, StressState::PlaneStress}) {
      // A 2-D model needs z along a material axis: the tilted frame is 3-D
      // only, and refused by the plane states.
      if (state != StressState::ThreeDimensional && !m.plasticity().plane_compatible()) {
        REQUIRE_THROWS_AS(plastic_return(m, state, random_strain(4.0e-3, 3u), PlasticState()),
                          ConfigError);
        continue;
      }
      Vector6 first = random_strain(4.0e-3, 3u);
      Vector6 second = first + random_strain(3.0e-3, 5u);
      if (state != StressState::ThreeDimensional) {
        first(4) = first(5) = second(4) = second(5) = 0.0;
        if (state == StressState::PlaneStrain) first(2) = second(2) = 0.0;
      }
      const PlasticResponse r1 = plastic_return(m, state, first, PlasticState(), 20.0);
      REQUIRE(r1.yielding);
      const PlasticResponse r2 = plastic_return(m, state, second, r1.state, 35.0);
      INFO(c.name << ", " << to_string(state));
      REQUIRE(r2.yielding);
      REQUIRE(r2.symmetric == c.symmetric);
      REQUIRE(tangent_error(m, state, second, r1.state, 35.0) <= 1.0e-8);
      const Scalar asymmetry = relative(r2.tangent, r2.tangent.transpose());
      if (c.symmetric) {
        REQUIRE(asymmetry <= 1.0e-12);
      } else {
        REQUIRE(asymmetry >= 1.0e-4);
      }
      // The elastic branch too: a small step back inside the surface.
      const Vector6 back = second - 0.1 * (second - first);
      REQUIRE_FALSE(plastic_return(m, state, back, r2.state).yielding);
      REQUIRE(tangent_error(m, state, back, r2.state) <= 1.0e-8);
      // The converged state is on its surface to 1e-13 sigma_y, so the
      // return from it at the same strain (stress recovery) is elastic and
      // reproduces it.
      const PlasticResponse again = plastic_return(m, state, second, r2.state, 35.0);
      REQUIRE_FALSE(again.yielding);
      REQUIRE(relative(again.stress, r2.stress) <= 1.0e-12);
      REQUIRE(yield_function(m, r2) <= 1.0e-13 * m.plasticity().yield_stress);
      // Without the tangent, the same state.
      const PlasticResponse lean = plastic_return(m, state, second, r1.state, 35.0, false);
      REQUIRE(relative(lean.stress, r2.stress) == 0.0);
    }
    // A return that cannot proceed - here from a non-finite strain - throws
    // SolverError, which the solvers take as a failed step to cut.
    Vector6 bad = random_strain(4.0e-3, 3u);
    bad(0) = std::numeric_limits<Scalar>::quiet_NaN();
    REQUIRE_THROWS_AS(plastic_return(m, StressState::ThreeDimensional, bad, PlasticState()),
                      SolverError);
  }
}

TEST_CASE("plane stress with Hill48 and Chaboche returns sigma_33 = 0 and the 3-D state at "
          "its eps_33",
          "[plasticity][anisotropy][material]") {
  Law law = hill_law(30.0);
  law.back = {{30.0e9, 300.0}};
  law.h = 1.0e9;
  const IsotropicMaterial m = make(law);
  Vector6 strain = Vector6::Zero();
  strain(0) = 6.0e-3;
  strain(1) = -1.0e-3;
  strain(3) = 4.0e-3;
  const PlasticResponse r = plastic_return(m, StressState::PlaneStress, strain, PlasticState());
  REQUIRE(r.yielding);
  REQUIRE(std::abs(r.stress(2)) <= 1.0e-11 * std::abs(r.stress(0)));
  REQUIRE(r.state.thickness_strain == r.strain_33);
  Vector6 full = strain;
  full(2) = r.strain_33;
  const PlasticResponse check =
      plastic_return(m, StressState::ThreeDimensional, full, PlasticState());
  REQUIRE((check.stress - r.stress).cwiseAbs().maxCoeff() <= 1.0e-12 * check.stress.norm());
  REQUIRE(relative(check.state.plastic_strain, r.state.plastic_strain) <= 1.0e-12);
  REQUIRE(r.tangent.row(2).cwiseAbs().maxCoeff() == 0.0);
  REQUIRE(r.tangent.col(2).cwiseAbs().maxCoeff() == 0.0);
  REQUIRE_FALSE(r.symmetric);
}

TEST_CASE("the general return's loading flag gives the continuum tangent at zero increment",
          "[plasticity][anisotropy][material]") {
  Law law = hill_law(20.0);
  law.h = 2.0e9;
  law.back = {{30.0e9, 300.0}, {2.0e9, 0.0}};
  const IsotropicMaterial m = make(law);
  Vector6 strain = Vector6::Zero();
  strain(0) = 5.0e-3;
  strain(1) = 1.0e-3;
  strain(3) = 2.0e-3;
  const PlasticResponse r1 =
      plastic_return(m, StressState::ThreeDimensional, strain, PlasticState());
  REQUIRE(r1.state.loading);
  const PlasticResponse again = plastic_return(m, StressState::ThreeDimensional, strain, r1.state);
  REQUIRE_FALSE(again.yielding);
  REQUIRE(again.state.loading);
  REQUIRE(relative(again.stress, r1.stress) <= 1.0e-12);
  // The limit of the consistent tangent: a small increment along the flow
  // direction follows it to first order.
  const Vector6 xi = r1.stress - r1.state.back_stress;
  Vector6 d = m.plasticity().yield_matrix * xi;  // engineering flow direction
  d *= 1.0e-9 / d.norm();
  const PlasticResponse further =
      plastic_return(m, StressState::ThreeDimensional, strain + d, r1.state);
  REQUIRE(further.yielding);
  REQUIRE((further.stress - r1.stress - again.tangent * d).norm() <=
          1.0e-6 * (again.tangent * d).norm());
  REQUIRE(relative(again.tangent, m.three_dimensional_matrix()) > 0.01);
  // Unloading: the elastic tangent, and the flag cleared.
  const PlasticResponse unload =
      plastic_return(m, StressState::ThreeDimensional, 0.9 * strain, r1.state);
  REQUIRE_FALSE(unload.state.loading);
  REQUIRE(relative(unload.tangent, m.three_dimensional_matrix()) <= 1.0e-14);
}

TEST_CASE("the general return leaves its state on the yield surface to round-off, so the "
          "re-return from it is elastic, however large the increment",
          "[plasticity][anisotropy][material]") {
  // Random Hill48 sheets with Voce, a recovering and a linear backstress,
  // along random paths of six increments of up to `amplitude` per
  // component. Stress recovery re-runs the return from the committed state
  // at the converged strain and relies on it being an elastic re-check.
  std::mt19937 gen(4u);
  std::uniform_real_distribution<Scalar> dist(-1.0, 1.0);
  Law base;
  base.hill = true;
  base.q = 80.0e6;
  base.delta = 20.0;
  base.back = {{40.0e9, 400.0}, {3.0e9, 0.0}};
  for (const StressState state : {StressState::ThreeDimensional, StressState::PlaneStress}) {
    for (const Scalar amplitude : {1.0e-3, 1.0e-2, 1.0e-1}) {
      Scalar worst = 0.0;
      int plastic = 0;
      int again = 0;
      Scalar drift = 0.0;
      for (int trial = 0; trial < 100; ++trial) {
        Law law = base;
        law.rd = in_plane(90.0 * dist(gen));
        law.r0 = 1.5 + dist(gen);
        law.r45 = 1.5 + dist(gen);
        law.r90 = 1.5 + dist(gen);
        law.h = 0.5e9 * (1.0 + dist(gen));
        const IsotropicMaterial m = make(law);
        Vector6 strain = Vector6::Zero();
        PlasticState point;
        for (int k = 0; k < 6; ++k) {
          for (int i = 0; i < 6; ++i) strain(i) += amplitude * dist(gen);
          if (state == StressState::PlaneStress) strain(2) = strain(4) = strain(5) = 0.0;
          const PlasticResponse r = plastic_return(m, state, strain, point);
          if (r.yielding) {
            ++plastic;
            worst = std::max(worst, yield_function(m, r) / law.yield);
            const PlasticResponse re = plastic_return(m, state, strain, r.state);
            if (re.yielding) ++again;
            drift = std::max(drift, relative(re.stress, r.stress));
          }
          point = r.state;
        }
      }
      INFO(to_string(state) << ", increments up to " << amplitude << ": " << plastic
                            << " plastic returns, |f| up to " << worst << " sigma_y");
      REQUIRE(plastic > 100);
      // Increments of 0.1 per component put the trial stress at some
      // hundred sigma_y, whose own round-off (32 of it is the return's
      // tolerance floor) is then 1e-13 sigma_y.
      REQUIRE(worst <= (amplitude < 0.05 ? 1.0e-13 : 3.0e-13));
      REQUIRE(again == 0);
      REQUIRE(drift <= 1.0e-12);
    }
  }
}

TEST_CASE("a plane model refuses a Hill48 frame whose axes miss z at the return",
          "[plasticity][anisotropy][material]") {
  // A plane model has no out-of-plane shear strain; with z off the axes the
  // return would couple it to the in-plane flow (sigma_23, gamma^p_23 != 0).
  Law law = hill_law(0.0);
  law.rd = Vector3(1.0, 0.0, 0.5);
  law.nd = Vector3(0.0, 0.3, 1.0);
  const IsotropicMaterial tilted = make(law);
  REQUIRE_FALSE(tilted.plasticity().plane_compatible());
  Vector6 strain = Vector6::Zero();
  strain(0) = 6.0e-3;
  strain(1) = -1.0e-3;
  strain(3) = 2.0e-3;
  REQUIRE_THROWS_AS(plastic_return(tilted, StressState::PlaneStress, strain, PlasticState()),
                    ConfigError);
  REQUIRE_THROWS_AS(plastic_return(tilted, StressState::PlaneStrain, strain, PlasticState()),
                    ConfigError);
  REQUIRE(plastic_return(tilted, StressState::ThreeDimensional, strain, PlasticState()).yielding);
  // Any of RD, TD, ND along z will do: a section through the thickness.
  Law section = hill_law(0.0);
  section.rd = Vector3(0.0, 0.0, 1.0);
  section.nd = Vector3(0.0, 1.0, 0.0);
  const IsotropicMaterial aligned = make(section);
  REQUIRE(aligned.plasticity().plane_compatible());
  REQUIRE(plastic_return(aligned, StressState::PlaneStrain, strain, PlasticState()).yielding);
}

namespace {

/// An independent reference for Hill48 with Chaboche, Prager and Voce
/// hardening on any strain path: the continuum rate equations integrated by
/// the classical fourth-order Runge-Kutta method in many substeps, the yield
/// point within a substep found by bisection. It shares nothing with the
/// return but the parameters: its own Hill matrix, polarised from Hill's
/// quadratic form in its own frame, and its own elasticity.
struct RateReference {
  Matrix6 hill = Matrix6::Zero();  ///< tensorial Voigt, global axes
  Matrix6 elastic = Matrix6::Zero();  ///< engineering strain -> tensorial stress
  Scalar sy0 = 0.0, h = 0.0, q = 0.0, delta = 0.0;
  std::vector<Backstress> terms;  ///< Prager's as one with no recovery

  struct State {
    Vector6 stress = Vector6::Zero();
    std::vector<Vector6> back;
    Scalar plastic = 0.0;
  };

  Scalar yield(Scalar a) const { return sy0 + h * a + q * (1.0 - std::exp(-delta * a)); }
  Scalar slope(Scalar a) const { return h + q * delta * std::exp(-delta * a); }
  Vector6 relative_stress(const State& s) const {
    Vector6 xi = s.stress;
    for (const Vector6& a : s.back) xi -= a;
    return xi;
  }
  Scalar f(const State& s) const {
    const Vector6 xi = relative_stress(s);
    return std::sqrt(xi.dot(hill * xi)) - yield(s.plastic);
  }
  /// The rates on the plastic branch at the engineering strain rate `e`:
  /// eps_p' = l m, alpha_i' = l ((2/3) C_i m_t - gamma_i alpha_i), p' = l,
  /// with l from the consistency condition m . (sigma' - sum alpha_i')
  /// = sigma_y' l.
  State rate(const State& s, const Vector6& e) const {
    const Vector6 xi = relative_stress(s);
    const Vector6 m = hill * xi / std::sqrt(xi.dot(hill * xi));
    Vector6 mt = m;
    mt.tail(3) *= 0.5;
    Scalar stiffness = m.dot(elastic * m) + slope(s.plastic);
    for (std::size_t i = 0; i < terms.size(); ++i) {
      stiffness += m.dot(2.0 / 3.0 * terms[i].modulus * mt - terms[i].recovery * s.back[i]);
    }
    const Scalar l = std::max(m.dot(elastic * e) / stiffness, 0.0);
    State d;
    d.stress = elastic * (e - l * m);
    d.back.resize(terms.size());
    for (std::size_t i = 0; i < terms.size(); ++i) {
      d.back[i] = l * (2.0 / 3.0 * terms[i].modulus * mt - terms[i].recovery * s.back[i]);
    }
    d.plastic = l;
    return d;
  }
  static State step(const State& s, Scalar dt, const State& d) {
    State out = s;
    out.stress += dt * d.stress;
    for (std::size_t i = 0; i < s.back.size(); ++i) out.back[i] += dt * d.back[i];
    out.plastic += dt * d.plastic;
    return out;
  }
  void runge_kutta(State& s, const Vector6& e, Scalar dt) const {
    const State k1 = rate(s, e);
    const State k2 = rate(step(s, 0.5 * dt, k1), e);
    const State k3 = rate(step(s, 0.5 * dt, k2), e);
    const State k4 = rate(step(s, dt, k3), e);
    s.stress += dt / 6.0 * (k1.stress + 2.0 * k2.stress + 2.0 * k3.stress + k4.stress);
    for (std::size_t i = 0; i < s.back.size(); ++i) {
      s.back[i] += dt / 6.0 * (k1.back[i] + 2.0 * k2.back[i] + 2.0 * k3.back[i] + k4.back[i]);
    }
    s.plastic += dt / 6.0 * (k1.plastic + 2.0 * k2.plastic + 2.0 * k3.plastic + k4.plastic);
  }
  /// A straight strain segment `de` in `n` substeps.
  void segment(State& s, const Vector6& de, int n) const {
    const Scalar dt = 1.0 / n;
    for (int k = 0; k < n; ++k) {
      const Vector6 xi = relative_stress(s);
      if (f(s) > -1.0e-12 * sy0 && (hill * xi).dot(elastic * de) > 0.0) {
        runge_kutta(s, de, dt);  // on the surface and loading
        continue;
      }
      State trial = s;
      trial.stress += dt * (elastic * de);
      if (f(trial) <= 0.0) {
        s = trial;
        continue;
      }
      Scalar lo = 0.0;  // elastic up to the yield point, plastic after it
      Scalar hi = 1.0;
      for (int it = 0; it < 100; ++it) {
        const Scalar mid = 0.5 * (lo + hi);
        State at = s;
        at.stress += mid * dt * (elastic * de);
        (f(at) > 0.0 ? hi : lo) = mid;
      }
      s.stress += lo * dt * (elastic * de);
      runge_kutta(s, de, (1.0 - lo) * dt);
    }
  }
};

}  // namespace

TEST_CASE("Hill48 with recovering backstresses converges at first order to an independent "
          "Runge-Kutta integration of the rate equations on a non-proportional path",
          "[plasticity][anisotropy][material]") {
  // No closed form exists for Hill48 with Armstrong-Frederick backstresses
  // (the backstress rate is along P xi, not xi, so even uniaxial stress is
  // not a proportional path). Along a non-proportional 3-D path with a
  // reversal, in a frame tilted out of every coordinate plane, the return
  // must converge to the continuum solution at first order in the step.
  const Scalar r0 = 1.9, r45 = 1.5, r90 = 2.3, l = 1.3, mm = 1.7;
  const Vector3 rd(1.0, 0.4, 0.3);
  const Vector3 nd(-0.3, 0.1, 1.0);
  RateReference ref;
  ref.sy0 = 200.0e6;
  ref.h = 0.6e9;
  ref.q = 80.0e6;
  ref.delta = 15.0;
  for (const Backstress& b : {Backstress{40.0e9, 400.0}, Backstress{6.0e9, 40.0},
                              Backstress{1.0e9, 0.0}, Backstress{0.5e9, 0.0}}) {
    ref.terms.push_back(b);
  }
  {
    const Vector3 n = nd.normalized();
    const Vector3 r = (rd - rd.dot(n) * n).normalized();
    Matrix3 q;
    q.row(0) = r.transpose();
    q.row(1) = n.cross(r).transpose();
    q.row(2) = n.transpose();
    const Scalar g = 1.0 / (1.0 + r0);
    const Scalar hh = r0 / (1.0 + r0);
    const Scalar f = hh / r90;
    const Scalar nn = (f + g) * (1.0 + 2.0 * r45) / 2.0;
    const auto form = [&](const Vector6& s) {
      const Matrix3 t = q * stress_tensor(s) * q.transpose();
      const Scalar a = t(1, 1) - t(2, 2);
      const Scalar b = t(2, 2) - t(0, 0);
      const Scalar c = t(0, 0) - t(1, 1);
      return f * a * a + g * b * b + hh * c * c + 2.0 * l * t(1, 2) * t(1, 2) +
             2.0 * mm * t(2, 0) * t(2, 0) + 2.0 * nn * t(0, 1) * t(0, 1);
    };
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        const Vector6 ei = Vector6::Unit(i);
        const Vector6 ej = Vector6::Unit(j);
        ref.hill(i, j) = 0.5 * (form(ei + ej) - form(ei) - form(ej));
      }
    }
    const Scalar e = 200.0e9;
    const Scalar nu = 0.3;
    const Scalar lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    const Scalar mu = e / (2.0 * (1.0 + nu));
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) ref.elastic(i, j) = lambda;
      ref.elastic(i, i) = lambda + 2.0 * mu;
      ref.elastic(3 + i, 3 + i) = mu;
    }
  }
  IsotropicMaterial m(200.0e9, 0.3, 7800.0, "sheet");
  PlasticityParameters p;
  p.yield_stress = ref.sy0;
  p.hardening_modulus = ref.h;
  p.saturation_stress = ref.q;
  p.saturation_rate = ref.delta;
  p.kinematic_hardening_modulus = 0.5e9;
  p.criterion = YieldCriterion::Hill48;
  p.hill.r0 = r0;
  p.hill.r45 = r45;
  p.hill.r90 = r90;
  p.hill.shear_l = l;
  p.hill.shear_m = mm;
  p.hill.rolling_direction = rd;
  p.hill.sheet_normal = nd;
  p.num_backstresses = 3;
  for (std::size_t i = 0; i < 3; ++i) p.backstresses[i] = ref.terms[i];

  std::vector<Vector6> corners(4, Vector6::Zero());
  corners[1] << 6.0e-3, -2.0e-3, -2.5e-3, 3.0e-3, 0.0, 1.0e-3;
  corners[2] << 7.0e-3, 1.0e-3, -4.0e-3, -2.0e-3, 3.0e-3, 0.0;
  corners[3] << -2.0e-3, -1.0e-3, 3.0e-3, 1.0e-3, 1.0e-3, -2.0e-3;
  RateReference::State s;
  s.back.assign(ref.terms.size(), Vector6::Zero());
  std::vector<RateReference::State> exact;
  for (std::size_t c = 1; c < corners.size(); ++c) {
    ref.segment(s, corners[c] - corners[c - 1], 40000);
    exact.push_back(s);
  }
  REQUIRE(std::abs(ref.f(s)) <= 1.0e-9 * ref.sy0);  // the reference stays on its surface
  for (const KinematicIntegration integration :
       {KinematicIntegration::Exponential, KinematicIntegration::BackwardEuler}) {
    p.kinematic_integration = integration;
    m.set_plasticity(p);
    std::vector<Scalar> errors;
    for (const int n : {100, 200, 400, 800}) {
      PlasticState point;
      Scalar worst = 0.0;
      for (std::size_t c = 1; c < corners.size(); ++c) {
        PlasticResponse r;
        for (int k = 1; k <= n; ++k) {
          const Vector6 strain =
              corners[c - 1] + (corners[c] - corners[c - 1]) * (static_cast<Scalar>(k) / n);
          r = plastic_return(m, StressState::ThreeDimensional, strain, point);
          point = r.state;
        }
        const RateReference::State& x = exact[c - 1];
        worst = std::max(worst, (r.stress - x.stress).cwiseAbs().maxCoeff() / ref.sy0);
        for (std::size_t i = 0; i < ref.terms.size(); ++i) {
          worst = std::max(worst,
                           (point.back_stresses[i] - x.back[i]).cwiseAbs().maxCoeff() / ref.sy0);
        }
        worst = std::max(worst, std::abs(point.equivalent_plastic_strain - x.plastic) / x.plastic);
      }
      errors.push_back(worst);
    }
    INFO((integration == KinematicIntegration::Exponential ? "exponential" : "backward Euler")
         << ": errors " << errors[0] << ", " << errors[1] << ", " << errors[2] << ", "
         << errors[3]);
    for (std::size_t i = 1; i < errors.size(); ++i) {
      REQUIRE(errors[i - 1] / errors[i] == Approx(2.0).margin(0.05));
    }
    REQUIRE(errors[2] <= 1.0e-3);
  }
}

namespace {

/// A distorted Hex8 element.
FemModel one_element(ElementType type, StressState state, const IsotropicMaterial& m) {
  StructuredMeshSpec spec;
  spec.nx = spec.ny = spec.nz = 2;
  spec.lx = 1.0;
  spec.ly = 0.8;
  spec.lz = 0.6;
  Mesh mesh = make_perturbed_hex_mesh(spec, 0.2);
  REQUIRE(mesh.element_type() == type);
  return FemModel(std::move(mesh), m, 1.0, state, IntegrationOptions());
}

Vector element_displacement(const FemModel& model, Index e, Scalar amplitude, unsigned seed) {
  const int dim = model.dim();
  const Matrix x0 = model.mesh().element_coordinates(e);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<Scalar> dist(-1.0, 1.0);
  Vector ue(dim * x0.cols());
  for (Eigen::Index a = 0; a < x0.cols(); ++a) {
    for (int k = 0; k < dim; ++k) {
      const Scalar homogeneous = k == 0 ? 0.8 * x0(0, a) + 0.3 * x0(1, a) : -0.2 * x0(k, a);
      ue(dim * a + k) = amplitude * (homogeneous + 0.5 * dist(gen));
    }
  }
  return ue;
}

/// A rigid turn of 0.5 rad about z of the element's nodes.
Vector rigid_turn(const FemModel& model, Scalar angle) {
  const int dim = model.dim();
  const Matrix x0 = model.mesh().element_coordinates(0);
  Matrix rot = Matrix::Identity(dim, dim);
  rot(0, 0) = rot(1, 1) = std::cos(angle);
  rot(0, 1) = -std::sin(angle);
  rot(1, 0) = std::sin(angle);
  const Matrix rigid = (rot - Matrix::Identity(dim, dim)) * x0;
  Vector turn(dim * x0.cols());
  for (Eigen::Index a = 0; a < x0.cols(); ++a) {
    for (int k = 0; k < dim; ++k) turn(dim * a + k) = rigid(k, a);
  }
  return turn;
}

}  // namespace

TEST_CASE("the element tangent of Hill48 with recovering backstresses is the non-symmetric "
          "derivative of its internal force",
          "[plasticity][anisotropy][element]") {
  Law law = hill_law(30.0);
  law.back = {{40.0e9, 400.0}, {3.0e9, 0.0}};
  law.q = 50.0e6;
  law.delta = 40.0;
  const IsotropicMaterial m = make(law);
  struct Case {
    ElementType type;
    StressState state;
  };
  const std::vector<Case> cases = {{ElementType::Hex8, StressState::ThreeDimensional}};
  for (const Case& c : cases) {
    FemModel model = one_element(c.type, c.state, m);
    for (const Kinematics kinematics : {Kinematics::SmallStrain, Kinematics::Finite}) {
      for (const bool bbar : {false, true}) {
        const std::vector<PlasticState> virgin(
            static_cast<std::size_t>(elastoplastic_points(model)));
        const Vector turn =
            kinematics == Kinematics::Finite ? rigid_turn(model, 0.5) : Vector::Zero(0);
        Vector u1 = element_displacement(model, 0, 6.0e-3, 7u);
        if (turn.size() > 0) u1 += turn;
        const ElastoplasticElement first =
            elastoplastic_element(model, 0, u1, virgin, bbar, nullptr, 0.0, false, kinematics);
        REQUIRE(first.yielding_points > 0);
        const Vector u2 = u1 + element_displacement(model, 0, 3.0e-3, 9u);
        const ElastoplasticElement base = elastoplastic_element(model, 0, u2, first.states, bbar,
                                                                nullptr, 0.0, true, kinematics);
        const Scalar h = 1.0e-8;
        Scalar worst = 0.0;
        for (Eigen::Index j = 0; j < u2.size(); ++j) {
          Vector up = u2;
          Vector um = u2;
          up(j) += h;
          um(j) -= h;
          const Vector fd = (elastoplastic_element(model, 0, up, first.states, bbar, nullptr,
                                                   0.0, false, kinematics)
                                 .internal_force -
                             elastoplastic_element(model, 0, um, first.states, bbar, nullptr,
                                                   0.0, false, kinematics)
                                 .internal_force) /
                            (2.0 * h);
          worst = std::max(worst, (fd - base.tangent.col(j)).cwiseAbs().maxCoeff());
        }
        INFO(to_string(c.type) << " " << to_string(c.state) << " " << to_string(kinematics)
                               << (bbar ? " mean dilatation" : ""));
        REQUIRE(base.yielding_points > 0);
        REQUIRE_FALSE(base.symmetric);
        REQUIRE(worst <= 1.0e-8 * base.tangent.cwiseAbs().maxCoeff());
        REQUIRE(relative(base.tangent, base.tangent.transpose()) >= 1.0e-4);
      }
    }
  }
  // Without recovery the element tangent stays symmetric.
  Law linear = hill_law(30.0);
  linear.back = {{3.0e9, 0.0}};
  FemModel model = one_element(ElementType::Hex8, StressState::ThreeDimensional, make(linear));
  const std::vector<PlasticState> virgin(static_cast<std::size_t>(elastoplastic_points(model)));
  const ElastoplasticElement sym = elastoplastic_element(
      model, 0, element_displacement(model, 0, 6.0e-3, 7u), virgin, true, nullptr, 0.0, true);
  REQUIRE(sym.symmetric);
  REQUIRE(sym.yielding_points > 0);
  REQUIRE(relative(sym.tangent, sym.tangent.transpose()) <= 1.0e-14);
}

TEST_CASE("finite-kinematics Hill48 plasticity is objective: the material frame convects",
          "[plasticity][anisotropy][element]") {
  Law law = hill_law(0.0);
  law.rd = Vector3(1.0, 0.5, 0.2);
  law.nd = Vector3(0.0, -0.3, 1.0);
  law.back = {{30.0e9, 300.0}};
  law.h = 1.0e9;
  const IsotropicMaterial m = make(law);
  for (const ElementType type : {ElementType::Hex8}) {
    FemModel model = one_element(type, StressState::ThreeDimensional, m);
    const Matrix x0 = model.mesh().element_coordinates(0);
    const std::vector<PlasticState> virgin(static_cast<std::size_t>(elastoplastic_points(model)));
    const Vector u = element_displacement(model, 0, 5.0e-3, 3u);
    const Matrix3 rot =
        Eigen::AngleAxisd(0.9, Vector3(0.2, 1.0, -0.4).normalized()).toRotationMatrix();
    Vector rotated(u.size());
    for (Eigen::Index a = 0; a < x0.cols(); ++a) {
      const Vector3 x = x0.col(a) + u.segment(3 * a, 3);
      rotated.segment(3 * a, 3) = rot * x - x0.col(a);
    }
    for (const bool averaged : {false, true}) {
      const ElastoplasticElement r1 = elastoplastic_element(model, 0, u, virgin, averaged,
                                                            nullptr, 0.0, false,
                                                            Kinematics::Finite);
      const ElastoplasticElement r2 = elastoplastic_element(model, 0, rotated, virgin, averaged,
                                                            nullptr, 0.0, false,
                                                            Kinematics::Finite);
      INFO(to_string(type) << (averaged ? " averaged" : ""));
      REQUIRE(r1.yielding_points > 0);
      REQUIRE(r2.yielding_points == r1.yielding_points);
      REQUIRE(r2.energy == Approx(r1.energy).epsilon(1e-10));
      for (std::size_t q = 0; q < r1.states.size(); ++q) {
        REQUIRE(r2.states[q].equivalent_plastic_strain ==
                Approx(r1.states[q].equivalent_plastic_strain).margin(1e-14));
        REQUIRE((r2.states[q].back_stress - r1.states[q].back_stress).norm() <=
                1.0e-8 * std::max(1.0, r1.states[q].back_stress.norm()));
      }
      for (Eigen::Index a = 0; a < x0.cols(); ++a) {
        const Vector3 expected = rot * r1.internal_force.segment(3 * a, 3);
        REQUIRE((r2.internal_force.segment(3 * a, 3) - expected).norm() <=
                1.0e-9 * r1.internal_force.cwiseAbs().maxCoeff());
      }
    }
  }
}

TEST_CASE("the Hill48 and Chaboche keys of a deck parse and validate",
          "[plasticity][anisotropy][config]") {
  const auto deck = [](const std::string& plasticity) {
    return json::parse(R"({
      "mesh": { "type": "structured_hex", "nx": 4, "ny": 1, "nz": 1, "lx": 1.0, "ly": 0.1,
                "lz": 0.1 },
      "material": { "youngs_modulus": 200e9, "poisson_ratio": 0.3,
                    "plasticity": )" + plasticity + R"( },
      "boundary_conditions": [ { "fix": ["x", "y", "z"], "region": { "box": { "xmax": 0.0 } } } ]
    })");
  };
  const std::string sheet = R"({ "yield_stress": 170e6, "saturation_stress": 90e6,
      "saturation_rate": 12, "kinematic_hardening_modulus": 1e9,
      "yield_criterion": "hill48",
      "anisotropy": { "r0": 1.9, "r45": 1.5, "r90": 2.3, "out_of_plane_shear": [1.4, 1.6],
                      "rolling_direction": [1, 0, 0], "sheet_normal": [0, 1, 0] },
      "backstresses": [ { "modulus": 60e9, "recovery": 600 }, { "modulus": 8e9, "recovery": 60 },
                        { "modulus": 1e9 } ],
      "kinematic_integration": "backward_euler" })";
  const Configuration config = parse_configuration(deck(sheet), "inline", true);
  const PlasticityParameters& p = config.material().plasticity();
  REQUIRE(p.criterion == YieldCriterion::Hill48);
  REQUIRE(p.hill.calibration == HillCalibration::RValues);
  REQUIRE(p.hill.G == Approx(1.0 / 2.9).epsilon(1e-15));
  REQUIRE(p.hill.L == 1.4);
  REQUIRE(p.hill.M == 1.6);
  REQUIRE((p.hill.axes.row(2).transpose() - Vector3::UnitY()).norm() == 0.0);
  REQUIRE((p.hill.axes.row(1).transpose() - Vector3(0.0, 0.0, -1.0)).norm() == 0.0);
  REQUIRE(p.num_backstresses == 3);
  REQUIRE(p.backstresses[0].modulus == 60.0e9);
  REQUIRE(p.backstresses[1].recovery == 60.0);
  REQUIRE(p.backstresses[2].recovery == 0.0);
  REQUIRE(p.kinematic_terms() == 4);  // Prager's modulus as the fourth
  REQUIRE(p.kinematic_integration == KinematicIntegration::BackwardEuler);
  REQUIRE_FALSE(p.symmetric_tangent());

  REQUIRE(build_model(config).material().plasticity().criterion == YieldCriterion::Hill48);

  // The other calibrations, and the rolling angle.
  const Configuration ratios = parse_configuration(
      deck(R"({ "yield_stress": 250e6, "yield_criterion": "hill48",
                "anisotropy": { "stress_ratios": { "sigma_45": 1.05, "sigma_90": 1.02,
                                                   "sigma_biaxial": 1.1 },
                                "rolling_angle": 30 } })"),
      "inline", true);
  const Hill48Parameters& hr = ratios.material().plasticity().hill;
  REQUIRE(hr.calibration == HillCalibration::StressRatios);
  REQUIRE(hr.G + hr.H == Approx(1.0).epsilon(1e-15));
  REQUIRE(hr.F + hr.G == Approx(1.0 / 1.21).epsilon(1e-15));
  REQUIRE(hr.axes(0, 0) == Approx(std::sqrt(3.0) / 2.0).epsilon(1e-15));
  REQUIRE(hr.axes(0, 1) == Approx(0.5).epsilon(1e-15));
  REQUIRE(hr.axes(2, 2) == 1.0);
  const Configuration coefficients = parse_configuration(
      deck(R"({ "yield_stress": 250e6, "yield_criterion": "hill48",
                "anisotropy": { "coefficients": { "F": 0.3, "G": 0.4, "H": 0.6, "L": 1.5,
                                                  "M": 1.5, "N": 1.4 },
                                "rolling_direction": [1, 1, 0], "sheet_normal": [0, 0, 2] },
                "backstresses": [ { "modulus": 5e9, "recovery": 50 } ] })"),
      "inline", true);
  const PlasticityParameters& pc = coefficients.material().plasticity();
  REQUIRE(pc.hill.calibration == HillCalibration::Coefficients);
  REQUIRE(pc.hill.N == 1.4);
  REQUIRE(pc.hill.axes(0, 1) == Approx(std::sqrt(0.5)).epsilon(1e-15));
  REQUIRE(pc.kinematic_integration == KinematicIntegration::Exponential);  // the default

  const auto refuses = [&](const std::string& plasticity) {
    INFO(plasticity);
    REQUIRE_THROWS_AS(parse_configuration(deck(plasticity), "inline", true), ConfigError);
  };
  const std::string y = R"("yield_stress": 250e6, )";
  const std::string hill48 = y + R"("yield_criterion": "hill48", )";
  refuses("{" + y + R"("yield_criterion": "hill48" })");  // no anisotropy block
  refuses("{" + y + R"("anisotropy": { "r0": 2, "r45": 1, "r90": 2 } })");  // von Mises
  refuses("{" + y + R"("yield_criterion": "tresca" })");
  refuses("{" + hill48 + R"("anisotropy": { "rolling_angle": 10 } })");  // no calibration
  refuses("{" + hill48 + R"("anisotropy": { "r0": 2, "r45": 1, "r90": 2,
      "coefficients": { "F": 0.3, "G": 0.4, "H": 0.6, "L": 1.5, "M": 1.5, "N": 1.4 } } })");
  refuses("{" + hill48 + R"("anisotropy": { "r0": 2, "r45": 1, "r90": 2,
      "stress_ratios": { "sigma_45": 1, "sigma_90": 1, "sigma_biaxial": 1 } } })");
  refuses("{" + hill48 + R"("anisotropy": { "r0": 2, "r90": 2 } })");  // r45 missing
  refuses("{" + hill48 + R"("anisotropy": { "r0": -2, "r45": 1, "r90": 2 } })");
  refuses("{" + hill48 + R"("anisotropy": { "r0": 2, "r45": 1, "r90": 2,
      "out_of_plane_shear": [1.5] } })");
  refuses("{" + hill48 + R"("anisotropy": {
      "coefficients": { "F": 0.3, "G": 0.4, "H": 0.6, "L": 1.5, "M": 1.5, "N": 1.4 },
      "out_of_plane_shear": [1.5, 1.5] } })");
  refuses("{" + hill48 + R"("anisotropy": { "r0": 2, "r45": 1, "r90": 2, "rolling_angle": 10,
      "rolling_direction": [1, 0, 0] } })");
  refuses("{" + hill48 + R"("anisotropy": { "r0": 2, "r45": 1, "r90": 2,
      "rolling_direction": [1, 0] } })");
  refuses("{" + hill48 + R"("anisotropy": { "r0": 2, "r45": 1, "r90": 2,
      "sheet_normal": [1, 0, 0] } })");  // parallel to the default rolling direction
  refuses("{" + hill48 + R"("anisotropy": { "r0": 2, "r45": 1, "r90": 2, "r_45": 1 } })");
  const std::string four =
      R"({ "modulus": 1e9, "recovery": 10 }, { "modulus": 1e9, "recovery": 10 },
         { "modulus": 1e9, "recovery": 10 }, { "modulus": 1e9, "recovery": 10 })";
  refuses("{" + y + R"("backstresses": [ )" + four + R"(, { "modulus": 1e9 } ] })");
  refuses("{" + y + R"("kinematic_hardening_modulus": 1e9, "backstresses": [ )" + four +
          " ] }");
  REQUIRE_NOTHROW(
      parse_configuration(deck("{" + y + R"("backstresses": [ )" + four + " ] }"),
                          "inline", true));
  refuses("{" + y + R"("backstresses": [ { "recovery": 10 } ] })");  // no modulus
  refuses("{" + y + R"("backstresses": [ { "modulus": 1e9, "recovery": -1 } ] })");
  refuses("{" + y + R"("backstresses": [ { "modulus": 0 } ] })");
  refuses("{" + y + R"("backstresses": [ { "modulus": 1e9, "rate": 1 } ] })");  // unknown key
  refuses("{" + y + R"("backstresses": [ { "modulus": 1e9 } ], "kinematic_integration": "rk4" })");
  refuses(R"({ "backstresses": [ { "modulus": 1e9 } ] })");  // no yield stress
}
