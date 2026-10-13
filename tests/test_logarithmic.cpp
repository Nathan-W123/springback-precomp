/// \file test_logarithmic.cpp
/// \brief Logarithmic-strain finite plasticity (Kinematics::FiniteLogarithmic):
///        the Hencky strain and its derivatives, the element's consistent
///        tangent, objectivity, the exact large-strain uniaxial response, the
///        small-strain limit, thermal expansion, and the solver.
///
/// The strain and its derivatives are compared with an independent matrix
/// logarithm (Eigen's Schur-Pade) and with fourth-order central differences
/// at distinct, equal and nearly equal eigenvalues. The large-strain
/// benchmarks are exact: for a coaxial homogeneous stretch the log strains
/// of successive increments add, so uniaxial stress to a stretch of 2 is the
/// one-dimensional small-strain law in the log strain, the Kirchhoff stress
/// its stress - to round-off, for J2 with linear and Voce hardening, for
/// Armstrong-Frederick backstresses along a tension-compression cycle, and
/// for Hill48 along its axes, whose plastic lateral log strains keep the
/// ratio r exactly.
#include "TestSupport.hpp"

#include "sparlab/core/Exceptions.hpp"
#include "sparlab/fem/Dynamics.hpp"
#include "sparlab/fem/Elastoplastic.hpp"
#include "sparlab/fem/NonlinearStatic.hpp"
#include "sparlab/io/CalculixWriter.hpp"
#include "sparlab/io/Config.hpp"
#include "sparlab/io/ResultWriter.hpp"
#include "sparlab/material/LogarithmicStrain.hpp"
#include "sparlab/material/Plasticity.hpp"

#include <unsupported/Eigen/MatrixFunctions>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace sparlab;
using namespace sparlab::testing;
using Catch::Approx;

namespace {

constexpr Scalar kInf = std::numeric_limits<Scalar>::infinity();

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
    REQUIRE(log.log_jacobian == Approx(0.5 * std::log(cg.determinant())).margin(2.0e-15));
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

/// One distorted element of a 2 x 2 (x 2) mesh.
FemModel one_element(ElementType type, StressState state, const IsotropicMaterial& m) {
  StructuredMeshSpec spec;
  spec.nx = spec.ny = spec.nz = 2;
  spec.lx = 1.0;
  spec.ly = 0.8;
  spec.lz = 0.6;
  Mesh mesh = type == ElementType::Quad4  ? make_perturbed_quad_mesh(spec, 0.2)
              : type == ElementType::Hex8 ? make_perturbed_hex_mesh(spec, 0.2)
                                          : make_perturbed_tet10_mesh(spec, 0.2);
  const Scalar thickness = mesh.dim() == 2 ? 0.05 : 1.0;
  return FemModel(std::move(mesh), m, thickness, state, IntegrationOptions());
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

TEST_CASE("the logarithmic element tangent is the derivative of its internal force: Hex8, Q4 "
          "plane strain and plane stress, Tet10, elastic, J2, Hill48 and Chaboche",
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
  const std::vector<Case> cases = {{ElementType::Hex8, StressState::ThreeDimensional},
                                   {ElementType::Quad4, StressState::PlaneStrain},
                                   {ElementType::Quad4, StressState::PlaneStress},
                                   {ElementType::Tet10, StressState::ThreeDimensional}};
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
  for (const StressState state : {StressState::ThreeDimensional, StressState::PlaneStress}) {
    const ElementType type =
        state == StressState::ThreeDimensional ? ElementType::Hex8 : ElementType::Quad4;
    FemModel model = one_element(type, state, m);
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
  for (const ElementType type : {ElementType::Hex8, ElementType::Tet10}) {
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
  const std::vector<Case> cases = {{ElementType::Hex8, StressState::ThreeDimensional},
                                   {ElementType::Quad4, StressState::PlaneStrain},
                                   {ElementType::Quad4, StressState::PlaneStress},
                                   {ElementType::Tet10, StressState::ThreeDimensional}};
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
    for (const ElementType type : {ElementType::Hex8, ElementType::Tet10, ElementType::Quad4}) {
      const StressState state =
          type == ElementType::Quad4 ? StressState::PlaneStrain : StressState::ThreeDimensional;
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

namespace {

/// A block [0, L] x [0, W] x [0, H] of Hex8 on symmetry planes x = 0, y = 0,
/// z = 0, its face x = L pulled to u_x = `pull`: uniaxial stress, a
/// homogeneous stretch on any mesh. Monitors: the end force and the mean
/// lateral displacements of the faces y = W and z = H.
struct Specimen {
  Scalar length = 1.0;
  Scalar width = 0.5;
  Scalar height = 0.4;
};

FemModel uniaxial_model(const Specimen& s, const IsotropicMaterial& m, bool patch, Scalar pull) {
  StructuredMeshSpec spec;
  spec.nx = spec.ny = spec.nz = patch ? 2 : 1;
  spec.lx = s.length;
  spec.ly = s.width;
  spec.lz = s.height;
  Mesh mesh = patch ? make_perturbed_hex_mesh(spec, 0.3) : make_structured_hex_mesh(spec);
  FemModel model(std::move(mesh), m, 1.0, StressState::ThreeDimensional, IntegrationOptions());
  const auto plane_at = [&](int axis, Scalar value) {
    SelectorGroup g;
    Selector sel;
    sel.kind = SelectorKind::Box;
    sel.xmin = axis == 0 ? value : -kInf;
    sel.xmax = axis == 0 ? value : kInf;
    sel.ymin = axis == 1 ? value : -kInf;
    sel.ymax = axis == 1 ? value : kInf;
    sel.zmin = axis == 2 ? value : -kInf;
    sel.zmax = axis == 2 ? value : kInf;
    g.members.push_back(sel);
    g.name = "plane";
    return g;
  };
  for (int axis = 0; axis < 3; ++axis) {
    DisplacementConstraint sym;
    sym.region = plane_at(axis, 0.0);
    sym.set(axis, true, 0.0);
    model.constraints().push_back(sym);
  }
  DisplacementConstraint end;
  end.region = plane_at(0, s.length);
  end.set(0, true, pull);
  model.constraints().push_back(end);
  LoadCaseSpec lc;
  lc.name = "pull";
  lc.prescribed_displacement_only = true;
  model.load_case_specs().push_back(lc);
  model.finalize();
  return model;
}

NonlinearOptions uniaxial_options(const Specimen& s, std::vector<Scalar> path, int steps) {
  NonlinearOptions o;
  o.kinematics = Kinematics::FiniteLogarithmic;
  o.steps = steps;
  o.load_path = std::move(path);
  o.residual_tolerance = 1.0e-12;
  o.displacement_tolerance = 1.0e-12;
  const auto face = [&](int axis, Scalar value) {
    NonlinearMonitor mon;
    Selector sel;
    sel.kind = SelectorKind::Box;
    sel.xmin = axis == 0 ? value : -kInf;
    sel.xmax = axis == 0 ? value : kInf;
    sel.ymin = axis == 1 ? value : -kInf;
    sel.ymax = axis == 1 ? value : kInf;
    sel.zmin = axis == 2 ? value : -kInf;
    sel.zmax = axis == 2 ? value : kInf;
    mon.region.members.push_back(sel);
    mon.component = axis;
    return mon;
  };
  NonlinearMonitor force = face(0, s.length);
  force.name = "force";
  force.quantity = NonlinearMonitor::Quantity::Reaction;
  NonlinearMonitor uy = face(1, s.width);
  uy.name = "uy";
  NonlinearMonitor uz = face(2, s.height);
  uz.name = "uz";
  o.monitors = {force, uy, uz};
  return o;
}

/// The 1-D law of the return in uniaxial stress, exact along a path of
/// monotonic legs: J2 with linear and Voce isotropic hardening and
/// Armstrong-Frederick backstresses, integrated per leg by the branch
/// solution X_i = nu C_i/gamma_i + (X_i0 - nu C_i/gamma_i) exp(-gamma_i D).
struct Uniaxial {
  PlasticityParameters p;
  Scalar e = 0.0;
  Scalar plastic_strain = 0.0;
  Scalar p_accumulated = 0.0;
  std::vector<Scalar> x;

  Scalar back(int i, Scalar nu, Scalar d) const {
    const Backstress t = p.kinematic_term(i);
    const Scalar x0 = x[static_cast<std::size_t>(i)];
    if (t.recovery == 0.0) return x0 + nu * t.modulus * d;
    const Scalar limit = nu * t.modulus / t.recovery;
    return limit + (x0 - limit) * std::exp(-t.recovery * d);
  }
  Scalar back_sum(Scalar nu, Scalar d) const {
    Scalar s = 0.0;
    for (int i = 0; i < p.kinematic_terms(); ++i) s += back(i, nu, d);
    return s;
  }
  /// The stress at the total strain `eps`, advancing the state.
  Scalar at(Scalar eps) {
    if (x.empty()) x.assign(static_cast<std::size_t>(p.kinematic_terms()), 0.0);
    const Scalar trial = e * (eps - plastic_strain);
    const Scalar relative_stress = trial - back_sum(1.0, 0.0);
    if (std::abs(relative_stress) <= p.yield(p_accumulated)) return trial;
    const Scalar nu = relative_stress > 0.0 ? 1.0 : -1.0;
    // f(D) = nu (E (eps - eps_p - nu D) - sum X_i(D)) - sigma_y(p + D)
    // falls monotonically from f(0) > 0.
    const auto f = [&](Scalar d) {
      return nu * (e * (eps - plastic_strain - nu * d) - back_sum(nu, d)) -
             p.yield(p_accumulated + d);
    };
    Scalar lo = 0.0;
    Scalar hi = 1.0e-3;
    while (f(hi) > 0.0) hi *= 2.0;
    for (int it = 0; it < 200; ++it) {
      const Scalar mid = 0.5 * (lo + hi);
      (f(mid) > 0.0 ? lo : hi) = mid;
    }
    const Scalar d = 0.5 * (lo + hi);
    for (int i = 0; i < p.kinematic_terms(); ++i) x[static_cast<std::size_t>(i)] = back(i, nu, d);
    plastic_strain += nu * d;
    p_accumulated += d;
    return e * (eps - plastic_strain);
  }
};

}  // namespace

TEST_CASE("uniaxial tension to a stretch of 2 is exact: the Kirchhoff stress against the log "
          "strain is the 1-D law, for J2 with linear and Voce hardening",
          "[logarithmic][solver]") {
  // A coaxial homogeneous stretch diag(l1, l2, l2): E_log = diag(ln l1,
  // ln l2, ln l2), the return is the small-strain J2 return in it (exact on
  // this proportional path), T = diag(tau, 0, 0) is the Kirchhoff stress
  // and the end force is tau A0 / l1.
  const Specimen s;
  const Scalar area = s.width * s.height;
  for (const bool voce : {false, true}) {
    const PlasticityParameters p = voce ? j2(0.5e9, 150.0e6, 12.0) : j2(1.5e9);
    const IsotropicMaterial m = steel(&p);
    for (const bool patch : {false, true}) {
      FemModel model = uniaxial_model(s, m, patch, s.length);  // to a stretch of 2
      Assembler assembler(model);
      const NonlinearResult r =
          NonlinearStaticAnalysis(model, assembler, uniaxial_options(s, {}, 20)).solve(0);
      INFO((voce ? "Voce" : "linear") << (patch ? ", distorted 2 x 2 x 2 patch" : ", one Hex8"));
      REQUIRE(r.completed);
      REQUIRE(r.kinematics == "finite_logarithmic");
      Uniaxial law;
      law.p = p;
      law.e = m.youngs_modulus();
      Scalar worst = 0.0;
      Scalar lateral = 0.0;
      Scalar tau_end = 0.0;
      for (const NonlinearStep& step : r.steps) {
        const Scalar l1 = 1.0 + step.load_factor;
        const Scalar eps = std::log(l1);
        const Scalar tau = law.at(eps);
        tau_end = tau;
        worst = std::max(worst, std::abs(step.monitors[0] * l1 / area - tau) / std::abs(tau));
        // Lateral log strains: equal, and with the axial one the elastic
        // volume change (1 - 2 nu) tau / E (the flow is isochoric).
        const Scalar e2 = std::log1p(step.monitors[1] / s.width);
        const Scalar e3 = std::log1p(step.monitors[2] / s.height);
        lateral = std::max(lateral, std::abs(e2 - e3));
        lateral = std::max(lateral, std::abs(eps + e2 + e3 - (1.0 - 2.0 * 0.3) * tau / law.e));
      }
      INFO("Kirchhoff stress error " << worst << ", lateral " << lateral);
      REQUIRE(worst <= 1.0e-12);
      REQUIRE(lateral <= 1.0e-12);
      REQUIRE(r.max_plastic_strain == Approx(law.p_accumulated).epsilon(1e-12));
      // Stress recovery: the Kirchhoff stress, the log strain, the Cauchy
      // stress tau / J of every element.
      const Scalar tau = tau_end;
      const Scalar e2 = std::log1p(r.steps.back().monitors[1] / s.width);
      const Scalar e3 = std::log1p(r.steps.back().monitors[2] / s.height);
      for (Index el = 0; el < model.mesh().num_elements(); ++el) {
        REQUIRE(r.element_kirchhoff(0, el) == Approx(tau).epsilon(1e-12));
        REQUIRE(std::abs(r.element_kirchhoff(1, el)) <= 1.0e-12 * tau);
        REQUIRE(r.element_log_strain(0, el) == Approx(std::log(2.0)).epsilon(1e-12));
        REQUIRE(r.element_log_strain(1, el) == Approx(e2).epsilon(1e-12));
        REQUIRE(r.element_cauchy(0, el) ==
                Approx(tau / std::exp(std::log(2.0) + e2 + e3)).epsilon(1e-12));
      }
      REQUIRE(r.max_green_strain == Approx(std::log(2.0)).epsilon(1e-12));
      REQUIRE(r.warnings.empty());  // no small-strain warning at a log strain of 0.69
    }
  }
}

TEST_CASE("a tension-compression cycle to +-0.3 log strain with Chaboche backstresses follows "
          "the exact Armstrong-Frederick solution in the log strain",
          "[logarithmic][solver]") {
  const Specimen s;
  const Scalar area = s.width * s.height;
  const PlasticityParameters p = chaboche();
  const IsotropicMaterial m = steel(&p);
  // Stretch e^0.3 at lambda = 1; lambda = (e^-0.3 - 1) / (e^0.3 - 1) gives
  // e^-0.3.
  const Scalar pull = s.length * std::expm1(0.3);
  const Scalar back = std::expm1(-0.3) / std::expm1(0.3);
  for (const bool patch : {false, true}) {
    FemModel model = uniaxial_model(s, m, patch, pull);
    Assembler assembler(model);
    const NonlinearResult r =
        NonlinearStaticAnalysis(model, assembler, uniaxial_options(s, {1.0, back, 1.0}, 12))
            .solve(0);
    INFO((patch ? "distorted 2 x 2 x 2 patch" : "one Hex8"));
    REQUIRE(r.completed);
    Uniaxial law;
    law.p = p;
    law.e = m.youngs_modulus();
    Scalar worst = 0.0;
    int reversed = 0;
    Scalar previous = 0.0;
    for (const NonlinearStep& step : r.steps) {
      const Scalar l1 = 1.0 + step.load_factor * pull / s.length;
      const Scalar tau = law.at(std::log(l1));
      worst = std::max(worst, std::abs(step.monitors[0] * l1 / area - tau) / law.p.yield_stress);
      if (step.load_factor < previous && step.yielding_points > 0) ++reversed;
      previous = step.load_factor;
    }
    INFO("Kirchhoff stress error " << worst << " sigma_y");
    REQUIRE(reversed > 0);  // reverse plastic flow on the way down
    REQUIRE(worst <= 1.0e-12);
    REQUIRE(r.load_factor == 1.0);
    REQUIRE(r.max_plastic_strain == Approx(law.p_accumulated).epsilon(1e-12));
  }
}

TEST_CASE("Hill48 pulled along RD and TD to a stretch of 1.6 keeps the plastic lateral "
          "log-strain ratio r0 and r90 exactly",
          "[logarithmic][solver]") {
  // The frame's axes along x, y, z: uniaxial stress along RD (or TD) is a
  // proportional path in the log strain, T = diag(tau, 0, 0); the plastic
  // width over thickness log strain is the r-value at any strain, and tau
  // follows E (k sigma_y + k^2 H eps) / (E + k^2 H), k = sigma_theta / sigma_0.
  const Specimen s;
  const Scalar area = s.width * s.height;
  for (const bool transverse : {false, true}) {
    PlasticityParameters p =
        hill48(transverse ? Vector3::UnitY() : Vector3::UnitX(), Vector3::UnitZ(), 1.0e9);
    p.saturation_stress = 0.0;
    p.saturation_rate = 0.0;
    const IsotropicMaterial m = steel(&p);
    const Hill48Parameters& h = m.plasticity().hill;
    // Loading along x: RD, or TD (rolled along y, with ND = z the thickness
    // direction and the width along RD).
    const Scalar r_expected = transverse ? 2.3 : 1.9;
    const Scalar k = transverse ? 1.0 / std::sqrt(h.F + h.H) : 1.0;
    FemModel model = uniaxial_model(s, m, true, 0.6 * s.length);
    Assembler assembler(model);
    const NonlinearResult r =
        NonlinearStaticAnalysis(model, assembler, uniaxial_options(s, {}, 12)).solve(0);
    INFO((transverse ? "TD" : "RD"));
    REQUIRE(r.completed);
    const Scalar e = m.youngs_modulus();
    const Scalar nu = m.poisson_ratio();
    Scalar ratio_error = 0.0;
    Scalar stress_error = 0.0;
    int plastic = 0;
    for (const NonlinearStep& step : r.steps) {
      const Scalar l1 = 1.0 + 0.6 * step.load_factor;
      const Scalar eps = std::log(l1);
      const Scalar tau = step.monitors[0] * l1 / area;
      if (step.yielding_points == 0) continue;
      ++plastic;
      const Scalar expected = e * (k * p.yield_stress + k * k * p.hardening_modulus * eps) /
                              (e + k * k * p.hardening_modulus);
      stress_error = std::max(stress_error, std::abs(tau - expected) / expected);
      // Plastic lateral log strains: the total less the elastic -nu tau / E.
      const Scalar width = std::log1p(step.monitors[1] / s.width) + nu * tau / e;
      const Scalar thickness = std::log1p(step.monitors[2] / s.height) + nu * tau / e;
      ratio_error = std::max(ratio_error, std::abs(width / thickness - r_expected) / r_expected);
    }
    INFO("stress error " << stress_error << ", r error " << ratio_error);
    REQUIRE(plastic >= 10);
    REQUIRE(stress_error <= 1.0e-12);
    REQUIRE(ratio_error <= 1.0e-12);
  }
}

TEST_CASE("a block crushed past zero volume stops with logarithmic kinematics instead of "
          "converging to an inverted state",
          "[logarithmic][solver]") {
  // One Hex8 on symmetry planes with its end pushed to u_x = -1.5 L in 10
  // steps: at lambda = 2/3 it is flat, beyond it F_11 < 0 - inverted. The
  // Hencky energy sees only |det F|, so an inverted state is an equilibrium
  // of it; the run must stop short of lambda = 2/3 with every converged
  // state valid, as the neo-Hookean one does.
  const Specimen s;
  const PlasticityParameters p = j2(1.5e9);
  for (const bool plastic : {false, true}) {
    const IsotropicMaterial m = steel(plastic ? &p : nullptr);
    const Scalar push = -1.5 * s.length;
    FemModel model = uniaxial_model(s, m, false, push);
    Assembler assembler(model);
    const NonlinearResult r =
        NonlinearStaticAnalysis(model, assembler, uniaxial_options(s, {}, 10)).solve(0);
    INFO((plastic ? "J2" : "elastic") << ": stopped at lambda = " << r.load_factor
                                      << ", min det F " << r.min_jacobian << ": "
                                      << r.termination);
    REQUIRE_FALSE(r.completed);
    REQUIRE(r.load_factor > 0.5);
    REQUIRE(r.load_factor < 2.0 / 3.0);
    REQUIRE(r.min_jacobian > 0.0);
    for (const NonlinearStep& step : r.steps) {
      REQUIRE(1.0 + step.load_factor * push / s.length > 0.0);
      REQUIRE(step.monitors[1] > 0.0);  // the lateral faces move out, not through
    }
  }
}

TEST_CASE("plane stress with logarithmic kinematics: the thickness stretches by exp(E_log,33) "
          "and uniaxial tension to a stretch of 1.6 is exact",
          "[logarithmic][solver]") {
  // A strip of distorted plane-stress Q4 on symmetry lines x = 0, y = 0,
  // pulled along x. J2 is isotropic, so the thickness log strain the return
  // finds equals the in-plane lateral one, and tau against ln l1 is the 1-D
  // law.
  const PlasticityParameters p = j2(1.5e9);
  const IsotropicMaterial m = steel(&p);
  StructuredMeshSpec spec;
  spec.nx = 4;
  spec.ny = 2;
  spec.lx = 1.0;
  spec.ly = 0.5;
  const Scalar thickness = 0.02;
  FemModel model(make_perturbed_quad_mesh(spec, 0.25), m, thickness, StressState::PlaneStress,
                 IntegrationOptions());
  const auto line = [](int axis, Scalar value) {
    SelectorGroup g;
    Selector sel;
    sel.kind = SelectorKind::Box;
    sel.xmin = axis == 0 ? value : -kInf;
    sel.xmax = axis == 0 ? value : kInf;
    sel.ymin = axis == 1 ? value : -kInf;
    sel.ymax = axis == 1 ? value : kInf;
    g.members.push_back(sel);
    return g;
  };
  for (int axis = 0; axis < 2; ++axis) {
    DisplacementConstraint sym;
    sym.region = line(axis, 0.0);
    sym.set(axis, true, 0.0);
    model.constraints().push_back(sym);
  }
  DisplacementConstraint end;
  end.region = line(0, 1.0);
  end.set(0, true, 0.6);
  model.constraints().push_back(end);
  LoadCaseSpec lc;
  lc.name = "pull";
  lc.prescribed_displacement_only = true;
  model.load_case_specs().push_back(lc);
  model.finalize();
  Assembler assembler(model);
  NonlinearOptions o;
  o.kinematics = Kinematics::FiniteLogarithmic;
  o.steps = 12;
  o.residual_tolerance = 1.0e-12;
  o.displacement_tolerance = 1.0e-12;
  NonlinearMonitor force;
  force.name = "force";
  force.region = line(0, 1.0);
  force.quantity = NonlinearMonitor::Quantity::Reaction;
  NonlinearMonitor uy;
  uy.name = "uy";
  uy.region = line(1, 0.5);
  uy.component = 1;
  o.monitors = {force, uy};
  const NonlinearResult r = NonlinearStaticAnalysis(model, assembler, o).solve(0);
  REQUIRE(r.completed);
  Uniaxial law;
  law.p = p;
  law.e = m.youngs_modulus();
  Scalar worst = 0.0;
  Scalar tau = 0.0;
  for (const NonlinearStep& step : r.steps) {
    const Scalar l1 = 1.0 + 0.6 * step.load_factor;
    tau = law.at(std::log(l1));
    worst = std::max(worst,
                     std::abs(step.monitors[0] * l1 / (0.5 * thickness) - tau) / std::abs(tau));
  }
  INFO("Kirchhoff stress error " << worst);
  REQUIRE(worst <= 1.0e-12);
  const Scalar lateral = std::log1p(r.steps.back().monitors[1] / 0.5);
  for (Index el = 0; el < model.mesh().num_elements(); ++el) {
    REQUIRE(r.element_log_strain(0, el) == Approx(std::log(1.6)).epsilon(1e-12));
    REQUIRE(r.element_log_strain(1, el) == Approx(lateral).epsilon(1e-12));
    REQUIRE(r.element_log_strain(2, el) == Approx(lateral).epsilon(1e-12));  // the thickness
    REQUIRE(r.element_kirchhoff(0, el) == Approx(tau).epsilon(1e-12));
    REQUIRE(std::abs(r.element_kirchhoff(2, el)) <= 1.0e-12 * tau);
    // J = l1 l2 l3 with l3 = exp(E_log,33).
    REQUIRE(r.element_cauchy(0, el) ==
            Approx(tau / (1.6 * std::exp(2.0 * lateral))).epsilon(1e-12));
  }
}

TEST_CASE("heated with logarithmic kinematics a free body expands stress-free by the stretch "
          "1 + alpha dT",
          "[logarithmic][solver][thermal]") {
  // The thermal strain is ln(1 + alpha dT) I, the log strain of the free
  // thermal stretch: on statically determinate supports u = alpha dT x
  // exactly, without stress or yielding (as with finite kinematics).
  const Scalar alpha = 1.2e-5;
  const Scalar t_ref = 20.0;
  const Scalar dt = 500.0;
  PlasticityParameters p = j2(1.0e9, 60.0e6, 30.0, 2.0e9);
  IsotropicMaterial m = steel(&p);
  m.set_thermal(alpha, t_ref, 50.0);
  const auto hold = [](const Vector& x, int dim, const std::array<bool, 3>& on) {
    DisplacementConstraint bc;
    Selector s;
    s.kind = SelectorKind::NearestNode;
    s.point = Vector3::Zero();
    s.point.head(dim) = x;
    bc.region.members.push_back(s);
    for (int k = 0; k < dim; ++k) {
      if (on[static_cast<std::size_t>(k)]) bc.set(k, true, 0.0);
    }
    return bc;
  };
  NonlinearOptions options;
  options.kinematics = Kinematics::FiniteLogarithmic;
  options.steps = 4;
  options.residual_tolerance = 1.0e-11;
  options.displacement_tolerance = 1.0e-11;
  for (const ElementType type : {ElementType::Quad4, ElementType::Hex8, ElementType::Tet10}) {
    INFO(to_string(type));
    StructuredMeshSpec spec;
    spec.nx = spec.ny = spec.nz = 3;
    Mesh mesh = type == ElementType::Quad4  ? make_perturbed_quad_mesh(spec, 0.25)
                : type == ElementType::Hex8 ? make_perturbed_hex_mesh(spec, 0.25)
                                            : make_perturbed_tet10_mesh(spec, 0.25);
    FemModel model(std::move(mesh), m, type == ElementType::Quad4 ? 0.01 : 1.0,
                   type == ElementType::Quad4 ? StressState::PlaneStress
                                              : StressState::ThreeDimensional,
                   IntegrationOptions());
    const int dim = model.dim();
    model.constraints().push_back(hold(Vector::Zero(dim), dim, {true, true, true}));
    Vector corner = Vector::Zero(dim);
    corner(0) = 1.0;
    model.constraints().push_back(hold(corner, dim, {false, true, true}));
    if (dim == 3) {
      corner.setZero();
      corner(1) = 1.0;
      model.constraints().push_back(hold(corner, dim, {false, false, true}));
    }
    LoadCaseSpec lc;
    lc.name = "heat";
    lc.temperature.source = TemperatureSpec::Source::Uniform;
    lc.temperature.uniform = t_ref + dt;
    model.load_case_specs().push_back(lc);
    model.finalize();
    Assembler assembler(model);
    const NonlinearResult r = NonlinearStaticAnalysis(model, assembler, options).solve(0);
    REQUIRE(r.completed);
    REQUIRE(r.plastic_points == 0);
    Scalar error = 0.0;
    for (Index node = 0; node < model.mesh().num_nodes(); ++node) {
      const Vector3 x = model.mesh().node(node);
      for (int k = 0; k < dim; ++k) {
        error = std::max(error, std::abs(r.displacement(dim * node + k) - alpha * dt * x(k)));
      }
    }
    REQUIRE(error <= 1.0e-13);
    // Stress-free (to round-off of E alpha dT = 1.2 GPa), the log strain
    // that of the stretch 1 + alpha dT - the thickness's too in plane stress.
    REQUIRE(r.element_kirchhoff.cwiseAbs().maxCoeff() <= 1.0e-12 * 200.0e9 * alpha * dt);
    REQUIRE(r.element_cauchy.cwiseAbs().maxCoeff() <= 1.0e-12 * 200.0e9 * alpha * dt);
    for (Index el = 0; el < model.mesh().num_elements(); ++el) {
      for (int i = 0; i < 3; ++i) {
        REQUIRE(r.element_log_strain(i, el) == Approx(std::log1p(alpha * dt)).epsilon(1e-12));
      }
    }
  }
}

TEST_CASE("the logarithmic kinematics key of a deck parses, runs and is reported",
          "[logarithmic][config]") {
  const auto deck = [](const std::string& nonlinear) {
    return json::parse(R"({
      "mesh": { "type": "structured_hex", "nx": 4, "ny": 1, "nz": 1, "lx": 1.0, "ly": 0.1,
                "lz": 0.1 },
      "material": { "youngs_modulus": 200e9, "poisson_ratio": 0.3,
                    "plasticity": { "yield_stress": 250e6, "hardening_modulus": 1e9 } },
      "boundary_conditions": [ { "fix": ["x", "y", "z"], "region": { "box": { "xmax": 0.0 } } },
                               { "fix": ["x"], "value": [0.05, 0, 0],
                                 "region": { "box": { "xmin": 1.0 } } } ],
      "load_cases": [ { "name": "pull", "prescribed_displacement_only": true } ],
      "nonlinear": )" + nonlinear + " }");
  };
  const Configuration config = parse_configuration(
      deck(R"({ "enabled": true, "kinematics": "finite_logarithmic", "steps": 4 })"), "inline",
      true);
  REQUIRE(config.nonlinear.options.kinematics == Kinematics::FiniteLogarithmic);
  REQUIRE(parse_kinematics("finite_logarithmic") == Kinematics::FiniteLogarithmic);
  REQUIRE(to_string(Kinematics::FiniteLogarithmic) == "finite_logarithmic");
  FemModel model = build_model(config);
  Assembler assembler(model);
  const NonlinearResult r =
      NonlinearStaticAnalysis(model, assembler, config.nonlinear.options).solve(0);
  REQUIRE(r.completed);
  REQUIRE(r.law == "hencky");
  REQUIRE(r.element_kirchhoff.rows() == 6);
  REQUIRE(r.element_log_strain.cols() == model.mesh().num_elements());
  const json::Value summary = nonlinear_json({r}, config.nonlinear.options, model, {});
  REQUIRE(summary.find("kinematics")->string_value() == "finite_logarithmic");
  REQUIRE(summary.find("material_model")->string_value() == "hencky");
  const json::Value& run = summary.find("load_cases")->array_items()[0];
  REQUIRE(run.find("max_logarithmic_strain")->number_value() >= std::log(1.05));
  // The CSV and VTK carry the Kirchhoff stress and the log strain.
  const std::string directory = "results/_test_tmp/logarithmic";
  ResultWriter(directory, config).write_nonlinear(model, r);
  std::ifstream csv(directory + "/nonlinear_stress_pull.csv");
  std::string header;
  std::getline(csv, header);
  REQUIRE(header.find("kirchhoff_xx[Pa]") != std::string::npos);
  REQUIRE(header.find("log_strain_zz[-]") != std::string::npos);
  std::ifstream vtk(directory + "/nonlinear_pull.vtk");
  const std::string text((std::istreambuf_iterator<char>(vtk)), std::istreambuf_iterator<char>());
  REQUIRE(text.find("kirchhoff_xy") != std::string::npos);
  REQUIRE(text.find("log_strain_xx") != std::string::npos);
  // No CalculiX counterpart; the neo-Hookean law is refused with it.
  REQUIRE(calculix_logarithmic_obstacle().find("logarithmic") != std::string::npos);
  REQUIRE_THROWS_AS(
      parse_configuration(deck(R"({ "enabled": true, "kinematics": "finite_logarithmic",
                                    "material_model": "neo_hookean" })"),
                          "inline", true),
      ConfigError);
  REQUIRE_THROWS_AS(
      parse_configuration(deck(R"({ "enabled": true, "kinematics": "logarithmic" })"), "inline",
                          true),
      ConfigError);
}
