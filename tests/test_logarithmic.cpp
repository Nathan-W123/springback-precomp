/// \file test_logarithmic.cpp
/// \brief Logarithmic-strain finite plasticity (Kinematics::FiniteLogarithmic):
///        the Hencky strain and its derivatives, the element's consistent
///        tangent, objectivity, inverted points and the small-strain limit.
///
/// The strain and its derivatives are compared with an independent matrix
/// logarithm (Eigen's Schur-Pade) and with fourth-order central differences
/// at distinct, equal and nearly equal eigenvalues.
#include "TestSupport.hpp"

#include "sparlab/core/Exceptions.hpp"
#include "sparlab/fem/Elastoplastic.hpp"
#include "sparlab/io/Config.hpp"
#include "sparlab/material/LogarithmicStrain.hpp"
#include "sparlab/material/Plasticity.hpp"

#include <unsupported/Eigen/MatrixFunctions>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace sparlab;
using namespace sparlab::testing;
using Catch::Approx;

namespace {

Vector6 engineering(const Matrix3& t) {
  Vector6 v;
  v << t(0, 0), t(1, 1), t(2, 2), 2.0 * t(0, 1), 2.0 * t(1, 2), 2.0 * t(2, 0);
  return v;
}

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

Scalar relative(const Matrix& a, const Matrix& b) {
  return (a - b).cwiseAbs().maxCoeff() / std::max(b.cwiseAbs().maxCoeff(), 1.0e-300);
}

/// Fourth-order central differences of a Voigt-valued f with respect to
/// the six (engineering) components of its argument.
Matrix6 derivative(const std::function<Vector6(const Vector6&)>& f, const Vector6& x, Scalar h) {
  Matrix6 d;
  for (int j = 0; j < 6; ++j) {
    const auto at = [&](Scalar s) {
      Vector6 y = x;
      y(j) += s;
      return f(y);
    };
    d.col(j) = (-at(2.0 * h) + 8.0 * at(h) - 8.0 * at(-h) + at(-2.0 * h)) / (12.0 * h);
  }
  return d;
}

/// A material of the tests: steel with the given plasticity (none: elastic).
IsotropicMaterial steel(const PlasticityParameters* p = nullptr) {
  IsotropicMaterial m(200.0e9, 0.3, 7800.0, "steel");
  if (p != nullptr) m.set_plasticity(*p);
  return m;
}

PlasticityParameters j2(Scalar h, Scalar q = 0.0, Scalar delta = 0.0, Scalar hk = 0.0) {
  PlasticityParameters p;
  p.yield_stress = 250.0e6;
  p.hardening_modulus = h;
  p.saturation_stress = q;
  p.saturation_rate = delta;
  p.kinematic_hardening_modulus = hk;
  return p;
}

PlasticityParameters hill48(const Vector3& rd, const Vector3& nd, Scalar h) {
  PlasticityParameters p = j2(h, 60.0e6, 20.0);
  p.criterion = YieldCriterion::Hill48;
  p.hill.r0 = 1.9;
  p.hill.r45 = 1.5;
  p.hill.r90 = 2.3;
  p.hill.rolling_direction = rd;
  p.hill.sheet_normal = nd;
  return p;
}

PlasticityParameters chaboche() {
  PlasticityParameters p = j2(0.5e9, 60.0e6, 25.0);
  p.num_backstresses = 2;
  p.backstresses[0] = {40.0e9, 400.0};
  p.backstresses[1] = {5.0e9, 40.0};
  return p;
}

}  // namespace

TEST_CASE("the logarithmic strain and its first and second derivatives are exact at distinct, "
          "equal and nearly equal eigenvalues",
          "[logarithmic][material]") {
  // C = Q diag(lambda) Q^T in a general frame; T an arbitrary stress, not
  // coaxial with C (as anisotropic plasticity gives).
  const Matrix3 q =
      Eigen::AngleAxisd(0.7, Vector3(0.3, -1.0, 0.5).normalized()).toRotationMatrix();
  struct Case {
    const char* name;
    Vector3 lambda;
  };
  const Scalar a = 1.4;
  const std::vector<Case> cases = {
      {"distinct", Vector3(2.3, 0.6, 1.1)},
      {"two equal", Vector3(1.7, 1.7, 0.5)},
      {"three equal, C = I", Vector3(1.0, 1.0, 1.0)},
      {"three equal, C = s^2 I", Vector3(1.69, 1.69, 1.69)},
      {"two within 1e-9", Vector3(a, a * (1.0 + 1.0e-9), 0.8)},
      {"two within 1e-7", Vector3(a, a * (1.0 + 1.0e-7), 0.8)},
      {"two within 1e-4", Vector3(a, a * (1.0 + 1.0e-4), 0.8)},
      {"three within 1e-9", Vector3(a, a * (1.0 + 1.0e-9), a * (1.0 + 2.0e-9))},
      {"three within 1e-7", Vector3(a, a * (1.0 + 0.4e-7), a * (1.0 + 1.0e-7))},
      {"three within 1e-4", Vector3(a, a * (1.0 + 1.0e-4), a * (1.0 + 2.5e-4))},
      {"three within 4e-3 (series)", Vector3(a, a * (1.0 + 2.0e-3), a * (1.0 + 4.0e-3))},
      {"three within 6e-3 (quotient)", Vector3(a, a * (1.0 + 3.0e-3), a * (1.0 + 6.0e-3))},
      {"stretch 2, two nearly equal", Vector3(4.0, 0.25, 0.25 * (1.0 + 1.0e-8))},
  };
  Vector6 t;
  t << 3.0e8, -1.0e8, 0.5e8, 0.7e8, -0.4e8, 0.2e8;
  Scalar worst_log = 0.0;
  Scalar worst_p = 0.0;
  Scalar worst_l = 0.0;
  for (const Case& c : cases) {
    INFO(c.name);
    const Matrix3 cg = q * c.lambda.asDiagonal() * q.transpose();
    const Vector6 green = engineering(0.5 * (cg - Matrix3::Identity()));
    const LogarithmicStrain log = logarithmic_strain(green);
    // The strain against an independent matrix logarithm (Schur-Pade).
    const Matrix3 reference = 0.5 * cg.log();
    worst_log = std::max(worst_log, relative(log.strain, engineering(reference)));
    REQUIRE(relative(log.strain, engineering(reference)) <= 1.0e-14);
    // ln J = tr E_log.
    REQUIRE(log.strain(0) + log.strain(1) + log.strain(2) ==
            Approx(0.5 * std::log(cg.determinant())).margin(4.0e-15));
    // P and T : L against fourth-order central differences of E_log(E)
    // and of S(E) = P(E)^T T, at a step well above any of the gaps (their
    // round-off, eps / h, bounds the agreement at about 4e-12).
    const Scalar h = 5.0e-4 * c.lambda.minCoeff();
    const Matrix6 p_fd = derivative(
        [](const Vector6& e) { return logarithmic_strain(e).strain; }, green, h);
    const Matrix6 l_fd = derivative(
        [&](const Vector6& e) { return logarithmic_stress(logarithmic_strain(e), t); }, green, h);
    const Matrix6 curvature = logarithmic_curvature(log, t);
    worst_p = std::max(worst_p, relative(log.projection, p_fd));
    worst_l = std::max(worst_l, relative(curvature, l_fd));
    REQUIRE(relative(log.projection, p_fd) <= 1.0e-11);
    REQUIRE(relative(curvature, l_fd) <= 1.0e-11);
    // T : L is a Hessian: symmetric. S = P^T T.
    REQUIRE(relative(curvature, curvature.transpose()) <= 1.0e-15);
    REQUIRE(relative(logarithmic_stress(log, t), log.projection.transpose() * t) <= 1.0e-15);
  }
  // Continuity through coalescence: two eigenvalues 1e-9 apart give P and
  // T : L of the equal pair to O(1e-9).
  const LogarithmicStrain equal =
      logarithmic_strain(engineering(0.5 * (q * Vector3(a, a, 0.8).asDiagonal() * q.transpose() -
                                            Matrix3::Identity())));
  const LogarithmicStrain near = logarithmic_strain(engineering(
      0.5 * (q * Vector3(a, a * (1.0 + 1.0e-9), 0.8).asDiagonal() * q.transpose() -
             Matrix3::Identity())));
  REQUIRE(relative(near.projection, equal.projection) <= 2.0e-9);
  REQUIRE(relative(logarithmic_curvature(near, t), logarithmic_curvature(equal, t)) <= 2.0e-9);
  // At C = I, P is the identity and T : L = -2 sym(T . dE . dE) (the second
  // order of ln(I + 2E) / 2 = E - E^2 + ...).
  const LogarithmicStrain identity = logarithmic_strain(Vector6::Zero());
  REQUIRE(identity.projection.isIdentity(1.0e-15));
  // A strain of 1e-6 keeps the relative accuracy of E: E_log = E - E^2 +
  // (4/3) E^3 - ... to round-off.
  std::mt19937 gen(5u);
  std::uniform_real_distribution<Scalar> dist(-1.0, 1.0);
  Matrix3 e = Matrix3::Zero();
  for (int i = 0; i < 3; ++i) {
    for (int j = i; j < 3; ++j) e(i, j) = e(j, i) = 1.0e-6 * dist(gen);
  }
  const Matrix3 series = e - e * e + 4.0 / 3.0 * e * e * e;
  REQUIRE(relative(logarithmic_strain(engineering(e)).strain, engineering(series)) <= 1.0e-15);
  INFO("worst: log " << worst_log << ", P " << worst_p << ", T : L " << worst_l);
  REQUIRE(worst_log <= 1.0e-14);
  // A Green-Lagrange strain that no deformation has - C = I + 2E not
  // positive definite - or a non-finite one is refused as a failed step.
  // (C = F^T F of an inverted point is positive definite, so ln C cannot
  // see an inversion: the element checks det F - see the next test.)
  Vector6 degenerate = Vector6::Zero();
  degenerate(0) = -0.6;  // 1 + 2 E_11 < 0
  REQUIRE_THROWS_AS(logarithmic_strain(degenerate), SolverError);
  degenerate(0) = std::numeric_limits<Scalar>::quiet_NaN();
  REQUIRE_THROWS_AS(logarithmic_strain(degenerate), SolverError);
}

namespace {

/// One distorted element of a 2 x 2 x 2 Hex8 mesh.
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

/// The displacement of element e's nodes under u = (F - I) X plus a random
/// perturbation of `noise` (relative to the element size).
Vector element_motion(const FemModel& model, Index e, const Matrix3& f, Scalar noise,
                      unsigned seed) {
  const int dim = model.dim();
  const Matrix x0 = model.mesh().element_coordinates(e);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<Scalar> dist(-1.0, 1.0);
  Vector ue(dim * x0.cols());
  for (Eigen::Index a = 0; a < x0.cols(); ++a) {
    Vector3 x = Vector3::Zero();
    x.head(dim) = x0.col(a);
    const Vector3 u = (f - Matrix3::Identity()) * x;
    for (int k = 0; k < dim; ++k) ue(dim * a + k) = u(k) + noise * dist(gen);
  }
  return ue;
}

/// A large deformation: stretches of 1.35, 0.85, 0.9 with a shear, turned
/// by 0.6 rad about z (and so in-plane for a 2-D model).
Matrix3 large_deformation(Scalar scale = 1.0) {
  Matrix3 u;
  u << 1.0 + 0.35 * scale, 0.2 * scale, 0.05 * scale, -0.1 * scale, 1.0 - 0.15 * scale,
      0.08 * scale, 0.04 * scale, -0.06 * scale, 1.0 - 0.1 * scale;
  const Matrix3 turn = Eigen::AngleAxisd(0.6, Vector3::UnitZ()).toRotationMatrix();
  return turn * u;
}

Matrix3 plane(Matrix3 f) {
  f.row(2).setZero();
  f.col(2).setZero();
  f(2, 2) = 1.0;
  return f;
}

}  // namespace

TEST_CASE("the logarithmic element tangent is the derivative of its internal force: Hex8, "
          "elastic, J2, Hill48 and Chaboche",
          "[logarithmic][element]") {
  struct Law {
    const char* name;
    bool plastic;
    PlasticityParameters p;
  };
  const std::vector<Law> laws = {
      {"elastic (Hencky)", false, PlasticityParameters()},
      {"J2 with Voce, linear and Prager hardening", true, j2(1.0e9, 60.0e6, 30.0, 2.0e9)},
      {"Hill48 in the sheet plane with Voce", true,
       hill48(Vector3(std::cos(0.5), std::sin(0.5), 0.0), Vector3::UnitZ(), 1.0e9)},
      {"Chaboche with Voce (non-symmetric)", true, chaboche()}};
  struct Case {
    ElementType type;
    StressState state;
  };
  const std::vector<Case> cases = {{ElementType::Hex8, StressState::ThreeDimensional}};
  Scalar worst = 0.0;
  for (const Law& law : laws) {
    const IsotropicMaterial m = steel(law.plastic ? &law.p : nullptr);
    for (const Case& c : cases) {
      FemModel model = one_element(c.type, c.state, m);
      const int dim = model.dim();
      const Matrix3 f1 = dim == 3 ? large_deformation(1.0) : plane(large_deformation(1.0));
      const Matrix3 f2 = dim == 3 ? large_deformation(1.1) : plane(large_deformation(1.1));
      for (const bool averaged : {false, true}) {
        if (averaged && c.state == StressState::PlaneStress) continue;
        const std::vector<PlasticState> virgin(
            static_cast<std::size_t>(elastoplastic_points(model)));
        const Vector u1 = element_motion(model, 0, f1, 0.01, 7u);
        const ElastoplasticElement first =
            elastoplastic_element(model, 0, u1, virgin, averaged, nullptr, 0.0, false,
                                  Kinematics::FiniteLogarithmic);
        const Vector u2 = element_motion(model, 0, f2, 0.01, 9u);
        const ElastoplasticElement base =
            elastoplastic_element(model, 0, u2, first.states, averaged, nullptr, 0.0, true,
                                  Kinematics::FiniteLogarithmic);
        const Scalar h = 1.0e-7;
        Scalar error = 0.0;
        for (Eigen::Index j = 0; j < u2.size(); ++j) {
          Vector up = u2;
          Vector um = u2;
          up(j) += h;
          um(j) -= h;
          const Vector fd = (elastoplastic_element(model, 0, up, first.states, averaged, nullptr,
                                                   0.0, false, Kinematics::FiniteLogarithmic)
                                 .internal_force -
                             elastoplastic_element(model, 0, um, first.states, averaged, nullptr,
                                                   0.0, false, Kinematics::FiniteLogarithmic)
                                 .internal_force) /
                            (2.0 * h);
          error = std::max(error, (fd - base.tangent.col(j)).cwiseAbs().maxCoeff());
        }
        error /= base.tangent.cwiseAbs().maxCoeff();
        worst = std::max(worst, error);
        INFO(law.name << ", " << to_string(c.type) << " " << to_string(c.state)
                      << (averaged ? ", mean dilatation" : "") << ": " << error);
        if (law.plastic) {
          REQUIRE(first.yielding_points > 0);
          REQUIRE(base.yielding_points > 0);
        }
        REQUIRE(error <= 1.0e-8);
        REQUIRE(base.symmetric == m.plasticity().symmetric_tangent());
      }
    }
  }
  INFO("worst " << worst);
  REQUIRE(worst <= 1.0e-8);
}

TEST_CASE("the logarithmic thermal load rate is the derivative of the internal force with "
          "respect to the temperature scale",
          "[logarithmic][element]") {
  PlasticityParameters p = j2(1.0e9, 60.0e6, 30.0);
  IsotropicMaterial m = steel(&p);
  m.set_thermal(1.2e-5, 20.0, 50.0);
  for (const StressState state : {StressState::ThreeDimensional}) {
    FemModel model = one_element(ElementType::Hex8, state, m);
    const int dim = model.dim();
    Vector temperature(model.mesh().num_nodes());
    for (Index n = 0; n < model.mesh().num_nodes(); ++n) {
      temperature(n) = 20.0 + 300.0 * model.mesh().node(n).x() + 100.0 * model.mesh().node(n).y();
    }
    const Matrix3 f = dim == 3 ? large_deformation(0.5) : plane(large_deformation(0.5));
    const Vector u = element_motion(model, 0, f, 0.005, 3u);
    const std::vector<PlasticState> virgin(static_cast<std::size_t>(elastoplastic_points(model)));
    const Scalar lambda = 0.8;
    const ElastoplasticElement base = elastoplastic_element(
        model, 0, u, virgin, false, &temperature, lambda, true, Kinematics::FiniteLogarithmic);
    REQUIRE(base.yielding_points > 0);
    // (A step of 1e-4 keeps the plane-stress iteration's residual sigma_33,
    // 1e-12 of the stress, out of the difference.)
    const Scalar h = 1.0e-4;
    const Vector fd = (elastoplastic_element(model, 0, u, virgin, false, &temperature,
                                             lambda + h, false, Kinematics::FiniteLogarithmic)
                           .internal_force -
                       elastoplastic_element(model, 0, u, virgin, false, &temperature,
                                             lambda - h, false, Kinematics::FiniteLogarithmic)
                           .internal_force) /
                      (2.0 * h);
    INFO(to_string(state) << ": " << relative(base.thermal_force_rate, fd));
    REQUIRE(relative(base.thermal_force_rate, fd) <= 1.0e-8);
  }
}

TEST_CASE("logarithmic kinematics is objective: a superposed rotation leaves T, S, the energy "
          "and the history unchanged and turns the forces and the Kirchhoff stress",
          "[logarithmic][element]") {
  PlasticityParameters p = hill48(Vector3(1.0, 0.5, 0.2), Vector3(0.0, -0.3, 1.0), 1.0e9);
  p.num_backstresses = 1;
  p.backstresses[0] = {30.0e9, 300.0};
  const IsotropicMaterial m = steel(&p);
  const Matrix3 rot =
      Eigen::AngleAxisd(0.9, Vector3(0.2, 1.0, -0.4).normalized()).toRotationMatrix();
  for (const ElementType type : {ElementType::Hex8}) {
    FemModel model = one_element(type, StressState::ThreeDimensional, m);
    const Matrix x0 = model.mesh().element_coordinates(0);
    const std::vector<PlasticState> virgin(static_cast<std::size_t>(elastoplastic_points(model)));
    const Vector u = element_motion(model, 0, large_deformation(0.8), 0.01, 3u);
    Vector turned(u.size());
    for (Eigen::Index a = 0; a < x0.cols(); ++a) {
      const Vector3 x = x0.col(a) + u.segment(3 * a, 3);
      turned.segment(3 * a, 3) = rot * x - x0.col(a);
    }
    for (const bool averaged : {false, true}) {
      INFO(to_string(type) << (averaged ? ", mean dilatation" : ""));
      const ElastoplasticElement r1 = elastoplastic_element(
          model, 0, u, virgin, averaged, nullptr, 0.0, false, Kinematics::FiniteLogarithmic);
      const ElastoplasticElement r2 = elastoplastic_element(
          model, 0, turned, virgin, averaged, nullptr, 0.0, false, Kinematics::FiniteLogarithmic);
      REQUIRE(r1.yielding_points > 0);
      REQUIRE(r2.yielding_points == r1.yielding_points);
      REQUIRE(r2.energy == Approx(r1.energy).epsilon(1e-12));
      for (std::size_t k = 0; k < r1.states.size(); ++k) {
        const PlasticState& s1 = r1.states[k];
        const PlasticState& s2 = r2.states[k];
        REQUIRE(s2.equivalent_plastic_strain ==
                Approx(s1.equivalent_plastic_strain).epsilon(1e-11));
        REQUIRE(relative(s2.plastic_strain, s1.plastic_strain) <= 1.0e-11);
        REQUIRE(relative(s2.back_stress, s1.back_stress) <= 1.0e-11);
      }
      for (Eigen::Index a = 0; a < x0.cols(); ++a) {
        REQUIRE((r2.internal_force.segment(3 * a, 3) - rot * r1.internal_force.segment(3 * a, 3))
                    .norm() <= 1.0e-11 * r1.internal_force.cwiseAbs().maxCoeff());
      }
      const ElastoplasticStress t1 = elastoplastic_stress(model, 0, u, r1.states, averaged,
                                                          nullptr, 0.0,
                                                          Kinematics::FiniteLogarithmic);
      const ElastoplasticStress t2 = elastoplastic_stress(model, 0, turned, r2.states, averaged,
                                                          nullptr, 0.0,
                                                          Kinematics::FiniteLogarithmic);
      REQUIRE(relative(t2.piola_kirchhoff, t1.piola_kirchhoff) <= 1.0e-11);
      REQUIRE(relative(t2.logarithmic_strain, t1.logarithmic_strain) <= 1.0e-12);
      REQUIRE(relative(t2.kirchhoff,
                       stress_voigt(rot * stress_tensor(t1.kirchhoff) * rot.transpose())) <=
              1.0e-11);
      REQUIRE(relative(t2.cauchy, stress_voigt(rot * stress_tensor(t1.cauchy) * rot.transpose())) <=
              1.0e-11);
      REQUIRE(t2.min_jacobian == Approx(t1.min_jacobian).epsilon(1e-13));
    }
  }
}

TEST_CASE("an inverted point is refused with logarithmic kinematics: det F <= 0 throws although "
          "C = F^T F stays positive definite",
          "[logarithmic][element]") {
  // A reflection F = diag(-1/2, 1, 1) has C = diag(1/4, 1, 1), the C of the
  // compression diag(1/2, 1, 1): the log strain, the energy and |det F| are
  // those of a valid state, and only the sign of det F tells them apart.
  // Every inverted F is refused - by the element and by the stress recovery
  // - while a turn by pi (det F = +1, F with negative entries) and a
  // compression to det F = 0.05 evaluate, the turn with the energy of the
  // unturned state.
  const PlasticityParameters p = j2(1.0e9);
  const IsotropicMaterial m = steel(&p);
  struct Case {
    ElementType type;
    StressState state;
  };
  const std::vector<Case> cases = {{ElementType::Hex8, StressState::ThreeDimensional}};
  const Matrix3 flip = Vector3(1.0, -1.0, 1.0).asDiagonal();
  const Matrix3 turn = Vector3(-1.0, -1.0, 1.0).asDiagonal();  // by pi about z
  for (const Case& c : cases) {
    FemModel model = one_element(c.type, c.state, m);
    const int dim = model.dim();
    const std::vector<PlasticState> virgin(static_cast<std::size_t>(elastoplastic_points(model)));
    const auto in_plane = [&](const Matrix3& f) { return dim == 3 ? f : plane(f); };
    for (const bool averaged : {false, true}) {
      if (averaged && c.state == StressState::PlaneStress) continue;
      INFO(to_string(c.type) << " " << to_string(c.state)
                             << (averaged ? ", mean dilatation" : ""));
      const auto element = [&](const Matrix3& f) {
        return elastoplastic_element(model, 0, element_motion(model, 0, in_plane(f), 0.0, 1u),
                                     virgin, averaged, nullptr, 0.0, true,
                                     Kinematics::FiniteLogarithmic);
      };
      const auto stress = [&](const Matrix3& f) {
        return elastoplastic_stress(model, 0, element_motion(model, 0, in_plane(f), 0.0, 1u),
                                    virgin, averaged, nullptr, 0.0,
                                    Kinematics::FiniteLogarithmic);
      };
      const Matrix3 reflection = Vector3(-0.5, 1.0, 1.0).asDiagonal();
      const Matrix3 inverted = large_deformation(1.0) * flip;  // a general inverted F
      for (const Matrix3& f : {reflection, inverted}) {
        REQUIRE(in_plane(f).determinant() < 0.0);
        REQUIRE_THROWS_AS(element(f), SolverError);
        REQUIRE_THROWS_AS(stress(f), SolverError);
      }
      const Matrix3 compressed = Vector3(0.05, 1.0, 1.0).asDiagonal();
      const ElastoplasticElement squeezed = element(compressed);
      REQUIRE(squeezed.yielding_points > 0);
      REQUIRE(stress(compressed).min_jacobian > 0.0);
      const ElastoplasticElement straight = element(large_deformation(1.0));
      const ElastoplasticElement turned = element(turn * large_deformation(1.0));
      REQUIRE(turned.energy == Approx(straight.energy).epsilon(1e-12));
      REQUIRE(stress(turn * large_deformation(1.0)).min_jacobian ==
              Approx(stress(large_deformation(1.0)).min_jacobian).epsilon(1e-13));
    }
  }
}

TEST_CASE("at strains of 1e-6 the logarithmic and the Green-Lagrange element agree to the "
          "order of the strain",
          "[logarithmic][element]") {
  // ln(I + 2E) / 2 = E - E^2 + ..., so forces and tangents differ by O(E)
  // relative: at most 20 E at strains of 1e-6, and ten times less at 1e-7.
  // Elastoplastically too, with a yield stress scaled with the strain so
  // that the points yield alike.
  for (const bool plastic : {false, true}) {
    for (const ElementType type : {ElementType::Hex8}) {
      const StressState state = StressState::ThreeDimensional;
      std::vector<Scalar> differences;
      for (const Scalar amplitude : {1.0e-6, 1.0e-7}) {
        PlasticityParameters low = j2(1.0e9);
        low.yield_stress = 1.5e5 * amplitude / 1.0e-6;
        const IsotropicMaterial m = steel(plastic ? &low : nullptr);
        FemModel model = one_element(type, state, m);
        const int dim = model.dim();
        Matrix3 f = Matrix3::Identity();
        f(0, 0) += amplitude;
        f(1, 1) -= 0.4 * amplitude;
        f(0, 1) += 0.7 * amplitude;
        if (dim == 3) f(2, 0) -= 0.5 * amplitude;
        const Vector u = element_motion(model, 0, f, 0.1 * amplitude, 11u);
        const std::vector<PlasticState> virgin(
            static_cast<std::size_t>(elastoplastic_points(model)));
        const ElastoplasticElement lg = elastoplastic_element(
            model, 0, u, virgin, true, nullptr, 0.0, true, Kinematics::FiniteLogarithmic);
        const ElastoplasticElement gl = elastoplastic_element(model, 0, u, virgin, true, nullptr,
                                                              0.0, true, Kinematics::Finite);
        INFO(to_string(type) << (plastic ? ", plastic" : ", elastic") << ", strain "
                             << amplitude);
        REQUIRE(lg.yielding_points == gl.yielding_points);
        if (plastic) REQUIRE(lg.yielding_points > 0);
        const Scalar force = relative(lg.internal_force, gl.internal_force);
        const Scalar tangent = relative(lg.tangent, gl.tangent);
        REQUIRE(force <= 20.0 * amplitude);
        REQUIRE(tangent <= 20.0 * amplitude);
        differences.push_back(force);
      }
      INFO(to_string(type) << (plastic ? ", plastic" : ", elastic") << ": " << differences[0]
                           << " at 1e-6, " << differences[1] << " at 1e-7");
      REQUIRE(differences[0] / differences[1] == Approx(10.0).epsilon(0.05));
    }
  }
}

TEST_CASE("the logarithmic kinematics key parses and prints", "[logarithmic][config]") {
  REQUIRE(parse_kinematics("finite_logarithmic") == Kinematics::FiniteLogarithmic);
  REQUIRE(to_string(Kinematics::FiniteLogarithmic) == "finite_logarithmic");
  REQUIRE_THROWS_AS(parse_kinematics("logarithmic"), ConfigError);
}
