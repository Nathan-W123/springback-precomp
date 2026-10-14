/// \file test_incompatible_modes.cpp
/// \brief The incompatible-mode Hex8 (Hex8Incompatible.hpp,
///        IncompatibleModes.hpp) and the through-thickness rule of a sheet:
///        element invariants, patch tests, exact pure bending, the
///        consistent tangents of every kinematics and law, the committed
///        history of the modes and the configuration path.
///
/// Every check is exact (to round-off), a central-difference derivative
/// check, or an invariant.
#include "TestSupport.hpp"

#include "sparlab/core/Exceptions.hpp"
#include "sparlab/elements/Hex8.hpp"
#include "sparlab/elements/Hex8Incompatible.hpp"
#include "sparlab/elements/Quadrature.hpp"
#include "sparlab/fem/Elastoplastic.hpp"
#include "sparlab/fem/ExplicitDynamics.hpp"
#include "sparlab/fem/Forming.hpp"
#include "sparlab/io/Config.hpp"
#include "sparlab/io/Json.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#if defined(SPARLAB_HAVE_OPENMP)
#include <omp.h>
#endif

using namespace sparlab;
using namespace sparlab::testing;
using Catch::Approx;

namespace {

IntegrationOptions incompatible_options(int thickness_points = 0) {
  IntegrationOptions opts;
  opts.formulation = ElementFormulation::IncompatibleModes;
  opts.thickness_points = thickness_points;
  return opts;
}

Matrix6 solid_d(Scalar young = 200.0e9, Scalar nu = 0.3) {
  return Material(young, nu, 7850.0, "steel").three_dimensional_matrix();
}

/// A rotation of 0.7 rad about the axis (1, 2, 3) / |(1, 2, 3)|.
Matrix3 rotation() {
  return Eigen::AngleAxis<Scalar>(0.7, Vector3(1.0, 2.0, 3.0).normalized()).toRotationMatrix();
}

/// The eigenvalues of a symmetric element matrix, ascending.
Vector eigenvalues(const Matrix& k) {
  Eigen::SelfAdjointEigenSolver<Matrix> es(k, Eigen::EigenvaluesOnly);
  return es.eigenvalues();
}

/// f^T K^+ f with the pseudo-inverse over the non-zero eigenvalues: the
/// compliance of a free element under a self-equilibrated load.
Scalar free_compliance(const Matrix& k, const Vector& f) {
  Eigen::SelfAdjointEigenSolver<Matrix> es(k);
  const Scalar top = es.eigenvalues().cwiseAbs().maxCoeff();
  Scalar c = 0.0;
  for (Eigen::Index i = 0; i < k.rows(); ++i) {
    const Scalar lambda = es.eigenvalues()(i);
    if (std::abs(lambda) <= 1.0e-10 * top) continue;
    const Scalar proj = es.eigenvectors().col(i).dot(f);
    c += proj * proj / lambda;
  }
  return c;
}

/// The exact 3-D pure-bending field of a prismatic bar along x with its
/// neutral axis at z = z0: u = kappa (x z', -nu y z', -(x^2 + nu (z'^2 -
/// y^2)) / 2) with z' = z - z0, sigma_xx = E kappa z' the only stress.
Vector3 pure_bending(const Vector3& x, Scalar kappa, Scalar nu, Scalar z0) {
  const Scalar z = x(2) - z0;
  return Vector3(kappa * x(0) * z, -nu * kappa * x(1) * z,
                 -0.5 * kappa * (x(0) * x(0) + nu * (z * z - x(1) * x(1))));
}

/// Central-difference check of an element tangent against its force: the
/// largest entry error over the largest entry of the tangent.
template <class Force>
Scalar tangent_error(const Matrix& tangent, const Vector& ue, Scalar h, Force&& force) {
  Scalar err = 0.0;
  for (Eigen::Index j = 0; j < ue.size(); ++j) {
    Vector up = ue;
    Vector um = ue;
    up(j) += h;
    um(j) -= h;
    const Vector column = (force(up) - force(um)) / (2.0 * h);
    err = std::max(err, (column - tangent.col(j)).cwiseAbs().maxCoeff());
  }
  return err / tangent.cwiseAbs().maxCoeff();
}

}  // namespace

// ---------------------------------------------------------------------------
// Quadrature through the thickness
// ---------------------------------------------------------------------------

TEST_CASE("the extended Gauss-Legendre line rules integrate polynomials of degree 2n - 1 "
          "exactly",
          "[quadrature][element]") {
  for (int n = 1; n <= kMaxThicknessPoints; ++n) {
    const std::vector<QuadraturePoint1D>& rule = gauss_legendre_line_extended(n);
    REQUIRE(static_cast<int>(rule.size()) == n);
    for (int k = 0; k <= 2 * n - 1; ++k) {
      Scalar sum = 0.0;
      for (const QuadraturePoint1D& p : rule) sum += p.weight * std::pow(p.xi, k);
      const Scalar exact = k % 2 == 1 ? 0.0 : 2.0 / (k + 1.0);
      INFO("n = " << n << ", degree " << k);
      REQUIRE(sum == Approx(exact).margin(2.0e-15));
    }
    for (std::size_t i = 1; i < rule.size(); ++i) REQUIRE(rule[i].xi > rule[i - 1].xi);
  }
  // The first four are the closed-form rules of gauss_legendre_cube (whose
  // first n points vary xi alone).
  for (int n = 1; n <= 4; ++n) {
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      REQUIRE(gauss_legendre_line_extended(n)[i].xi == gauss_legendre_cube(n)[i].xi);
    }
  }
  REQUIRE_THROWS_AS(gauss_legendre_line_extended(0), ConfigError);
  REQUIRE_THROWS_AS(gauss_legendre_line_extended(8), ConfigError);
}

TEST_CASE("the box rule equals the cube rule for equal orders and orients its thickness points",
          "[quadrature][element]") {
  for (int n = 1; n <= 4; ++n) {
    for (int axis = 0; axis < 3; ++axis) {
      const std::vector<QuadraturePoint3D>& box = gauss_legendre_box(n, n, axis);
      const std::vector<QuadraturePoint3D>& cube = gauss_legendre_cube(n);
      REQUIRE(box.size() == cube.size());
      for (std::size_t i = 0; i < box.size(); ++i) {
        REQUIRE(box[i].xi == cube[i].xi);
        REQUIRE(box[i].eta == cube[i].eta);
        REQUIRE(box[i].zeta == cube[i].zeta);
        REQUIRE(box[i].weight == cube[i].weight);
      }
    }
  }
  for (int axis = 0; axis < 3; ++axis) {
    const std::vector<QuadraturePoint3D>& box = gauss_legendre_box(2, 5, axis);
    REQUIRE(box.size() == 20u);
    Scalar total = 0.0;
    Scalar moment = 0.0;  // int t^8 dV along the thickness axis: 2/9 * 4
    for (const QuadraturePoint3D& p : box) {
      total += p.weight;
      const Scalar t = axis == 0 ? p.xi : axis == 1 ? p.eta : p.zeta;
      moment += p.weight * std::pow(t, 8);
    }
    INFO("axis " << axis);
    REQUIRE(total == Approx(8.0).epsilon(1.0e-14));
    REQUIRE(moment == Approx(4.0 * 2.0 / 9.0).epsilon(1.0e-13));
    // xi still varies fastest.
    REQUIRE(box[1].xi > box[0].xi);
  }
  REQUIRE_THROWS_AS(gauss_legendre_box(5, 3, 2), ConfigError);
  REQUIRE_THROWS_AS(gauss_legendre_box(2, 8, 2), ConfigError);
  REQUIRE_THROWS_AS(gauss_legendre_box(2, 3, 3), ConfigError);
}

TEST_CASE("a Hex8 with thickness points integrates with the box rule and keeps its default "
          "bit for bit",
          "[solid][element][quadrature]") {
  const Hex8Element hex;
  const Matrix coords = distorted_hex_coords();
  const Matrix d = solid_d();
  IntegrationOptions plain;
  IntegrationOptions same;
  same.thickness_points = 2;
  const Matrix k0 = hex.stiffness(coords, d, 1.0, plain);
  REQUIRE((hex.stiffness(coords, d, 1.0, same) - k0).cwiseAbs().maxCoeff() == 0.0);
  IntegrationOptions sheet;
  sheet.thickness_points = 5;
  REQUIRE(hex.integration_rule(sheet).size() == 20u);
  // On a box the 2 x 2 x 2 rule is already exact, so five points through
  // the thickness give the same matrix.
  const Matrix box = unit_box_coords(2.0, 1.0, 0.1);
  REQUIRE((hex.stiffness(box, d, 1.0, sheet) - hex.stiffness(box, d, 1.0, plain))
              .cwiseAbs()
              .maxCoeff() <= 1.0e-12 * hex.stiffness(box, d, 1.0, plain).cwiseAbs().maxCoeff());
}

// ---------------------------------------------------------------------------
// The incompatible-mode Hex8: linear element
// ---------------------------------------------------------------------------

TEST_CASE("the element factory builds the incompatible-mode Hex8",
          "[incompatible][element]") {
  const std::unique_ptr<Element> e = make_element(ElementType::Hex8, incompatible_options());
  REQUIRE(e->type() == ElementType::Hex8);
  REQUIRE(e->num_internal_nodes() == 3);
  REQUIRE(e->num_internal_dofs() == 9);
  REQUIRE(make_element(ElementType::Hex8, IntegrationOptions())->num_internal_dofs() == 0);
  REQUIRE(parse_element_formulation("incompatible_modes") == ElementFormulation::IncompatibleModes);
  REQUIRE(to_string(ElementFormulation::Standard) == "standard");
  REQUIRE_THROWS_AS(parse_element_formulation("eas"), ConfigError);
}

TEST_CASE("the condensed incompatible-mode stiffness is symmetric with exactly six zero "
          "eigenvalues",
          "[incompatible][element]") {
  const Hex8IncompatibleElement hex;
  const Matrix d = solid_d();
  struct Case {
    const char* name;
    Matrix coords;
    int thickness_points;
  };
  const std::vector<Case> cases = {{"box", unit_box_coords(1.0, 0.7, 0.4), 0},
                                   {"thin box", unit_box_coords(1.0, 1.0, 0.01), 0},
                                   {"distorted", distorted_hex_coords(), 0},
                                   {"distorted, 2 x 2 x 5", distorted_hex_coords(), 5}};
  for (const Case& c : cases) {
    INFO(c.name);
    IntegrationOptions opts = incompatible_options(c.thickness_points);
    const Matrix k = hex.stiffness(c.coords, d, 1.0, opts);
    REQUIRE((k - k.transpose()).cwiseAbs().maxCoeff() <= 1.0e-15 * k.cwiseAbs().maxCoeff());
    const Vector lambda = eigenvalues(k);
    const Scalar top = lambda.cwiseAbs().maxCoeff();
    // Six round-off eigenvalues, then a gap of many orders to the softest
    // deformation mode (the plate bending of the thin box, 4e-9 of the
    // largest - the physical t^2/L^2 ratio squared).
    const Scalar rigid = lambda.head(6).cwiseAbs().maxCoeff();
    REQUIRE(rigid <= 1.0e-12 * top);
    REQUIRE(lambda(6) > 1.0e3 * rigid);
    // Softer than the compatible element (Loewner order): K - K* is positive
    // semi-definite.
    const Matrix standard = Hex8Element().stiffness(c.coords, d, 1.0, opts);
    REQUIRE(eigenvalues(standard - k).minCoeff() >= -1.0e-10 * top);
    // The generic condensation of Element reproduces it:
    // sum_q w B-hat^T D B-hat = K*.
    const InternalCondensation cond = hex.condense_internal(c.coords, d, 1.0, opts);
    Matrix again = Matrix::Zero(24, 24);
    for (const IntegrationPoint& ip : hex.integration_rule(opts)) {
      const Matrix bhat = hex.condensed_strain_operator(c.coords, ip.point, cond);
      again += ip.weight * hex.strain_operator(c.coords, ip.point).detJ *
               (bhat.transpose() * d * bhat);
    }
    REQUIRE((again - k).cwiseAbs().maxCoeff() <= 1.0e-12 * top);
  }
}

TEST_CASE("the incompatible modes take no part in a linear displacement field on a distorted "
          "cell",
          "[incompatible][element][patch]") {
  const Hex8IncompatibleElement hex;
  const Matrix coords = distorted_hex_coords();
  const Matrix d = solid_d();
  Matrix3 grad;
  grad << 3.0e-4, 1.0e-4, -0.5e-4, 2.0e-4, -2.0e-4, 0.7e-4, -0.5e-4, 0.4e-4, 1.5e-4;
  Vector ue(24);
  for (int a = 0; a < 8; ++a) ue.segment<3>(3 * a) = grad * coords.col(a) + Vector3(1e-3, 0, 0);
  for (int n : {2, 3}) {
    for (int t : {0, 3, 7}) {
      INFO("points " << n << ", thickness points " << t);
      IntegrationOptions opts = incompatible_options(t);
      opts.stiffness_points = n;
      const InternalCondensation cond = hex.condense_internal(coords, d, 1.0, opts);
      REQUIRE((cond.coupling * ue).cwiseAbs().maxCoeff() <= 1.0e-13 * ue.cwiseAbs().maxCoeff());
      // The condensed force is the compatible one.
      const Vector f = hex.stiffness(coords, d, 1.0, opts) * ue;
      const Vector f0 = Hex8Element().stiffness(coords, d, 1.0, opts) * ue;
      REQUIRE((f - f0).cwiseAbs().maxCoeff() <= 1.0e-11 * f0.cwiseAbs().maxCoeff());
    }
  }
}

TEST_CASE("the incompatible-mode Hex8 is exact in pure bending where the standard Hex8 locks",
          "[incompatible][element][verification]") {
  const Hex8IncompatibleElement hex;
  const Scalar young = 70.0e9;
  const Scalar nu = 0.3;
  const Matrix d = solid_d(young, nu);
  const Scalar kappa = 1.0e-3;
  for (Scalar slenderness : {1.0, 10.0, 100.0}) {
    const Scalar h = 0.01;
    const Scalar length = slenderness * h;
    const Scalar width = 0.02;
    const Matrix coords = unit_box_coords(length, width, h);
    Vector ue(24);
    for (int a = 0; a < 8; ++a) ue.segment<3>(3 * a) = pure_bending(coords.col(a), kappa, nu, 0.5 * h);
    // Consistent nodal forces of the end tractions -+ E kappa z': a node on
    // the top edge (z' = h/2) takes w/2 E kappa h^2/12, the bottom one the
    // opposite, on the face x = L; mirrored on x = 0.
    Vector f = Vector::Zero(24);
    for (int a = 0; a < 8; ++a) {
      const Scalar sign_x = coords(0, a) > 0.0 ? 1.0 : -1.0;
      const Scalar sign_z = coords(2, a) > 0.0 ? 1.0 : -1.0;
      f(3 * a) = sign_x * sign_z * 0.5 * width * young * kappa * h * h / 12.0;
    }
    const Matrix k = hex.stiffness(coords, d, 1.0, incompatible_options());
    INFO("L/h = " << slenderness);
    // Round-off of K u: the membrane stiffness times the displacement, which
    // outgrows the bending forces as (L/h)^2.
    const Scalar gross = (k.cwiseAbs() * ue.cwiseAbs()).maxCoeff();
    REQUIRE((k * ue - f).cwiseAbs().maxCoeff() <= 1.0e-13 * gross);
    // The standard element under the same load: its compliance against the
    // exact one (the locking ratio, documented in docs/formulation.md).
    const Matrix k0 = Hex8Element().stiffness(coords, d, 1.0, IntegrationOptions());
    const Scalar exact = f.dot(ue);
    const Scalar ratio = free_compliance(k0, f) / exact;
    REQUIRE(free_compliance(k, f) / exact == Approx(1.0).epsilon(1.0e-9));
    INFO("standard Hex8 compliance ratio " << ratio);
    REQUIRE(ratio < (slenderness > 5.0 ? 0.05 : 0.8));
  }
}

TEST_CASE("the incompatible-mode stiffness is frame invariant and independent of the node "
          "numbering",
          "[incompatible][element]") {
  const Hex8IncompatibleElement hex;
  const Matrix coords = distorted_hex_coords();
  const Matrix d = solid_d();
  const IntegrationOptions opts = incompatible_options();
  const Matrix k = hex.stiffness(coords, d, 1.0, opts);
  const Matrix3 r = rotation();
  // The rotated material of an isotropic solid is the same material.
  Matrix q = Matrix::Zero(24, 24);
  for (int a = 0; a < 8; ++a) q.block<3, 3>(3 * a, 3 * a) = r;
  const Matrix kr = hex.stiffness(r * coords, d, 1.0, opts);
  REQUIRE((kr - q * k * q.transpose()).cwiseAbs().maxCoeff() <= 1.0e-12 * k.cwiseAbs().maxCoeff());
  // Renumbering the nodes by a quarter turn about zeta (an orientation-
  // preserving relabelling) permutes K*.
  const int perm[8] = {1, 2, 3, 0, 5, 6, 7, 4};
  Matrix renumbered(3, 8);
  for (int a = 0; a < 8; ++a) renumbered.col(a) = coords.col(perm[a]);
  const Matrix kp = hex.stiffness(renumbered, d, 1.0, opts);
  for (int a = 0; a < 8; ++a) {
    for (int b = 0; b < 8; ++b) {
      REQUIRE((kp.block<3, 3>(3 * a, 3 * b) - k.block<3, 3>(3 * perm[a], 3 * perm[b]))
                  .cwiseAbs()
                  .maxCoeff() <= 1.0e-12 * k.cwiseAbs().maxCoeff());
    }
  }
}

TEST_CASE("rules of two, three and four points give the same condensed stiffness on a box",
          "[incompatible][element]") {
  const Hex8IncompatibleElement hex;
  const Matrix coords = unit_box_coords(1.3, 0.8, 0.2);
  const Matrix d = solid_d();
  IntegrationOptions opts = incompatible_options();
  const Matrix k2 = hex.stiffness(coords, d, 1.0, opts);
  for (int n : {3, 4}) {
    opts.stiffness_points = n;
    REQUIRE((hex.stiffness(coords, d, 1.0, opts) - k2).cwiseAbs().maxCoeff() <=
            1.0e-12 * k2.cwiseAbs().maxCoeff());
  }
  opts.stiffness_points = 2;
  opts.thickness_points = 6;
  REQUIRE((hex.stiffness(coords, d, 1.0, opts) - k2).cwiseAbs().maxCoeff() <=
          1.0e-12 * k2.cwiseAbs().maxCoeff());
}

// ---------------------------------------------------------------------------
// The non-linear kernels: consistent tangents, patch tests, invariance
// ---------------------------------------------------------------------------

namespace {

/// A single distorted incompatible-mode Hex8 of `m`.
FemModel single_hex(const Material& m, int thickness_points = 0) {
  Mesh mesh(distorted_hex_coords(), {0, 1, 2, 3, 4, 5, 6, 7}, ElementType::Hex8);
  return FemModel(std::move(mesh), m, 1.0, StressState::ThreeDimensional,
                  incompatible_options(thickness_points));
}

/// A displacement pattern with stretch, shear, bending and twist, scaled.
Vector deformation(const Matrix& x, Scalar scale) {
  Vector ue(24);
  for (int a = 0; a < 8; ++a) {
    const Vector3 p = x.col(a);
    ue.segment<3>(3 * a) =
        scale * Vector3(0.8 * p(0) + 0.6 * p(0) * p(2) + 0.2 * p(1),
                        -0.3 * p(1) + 0.5 * p(1) * p(2) - 0.4 * p(0) * p(1),
                        -0.4 * p(2) - 0.5 * p(0) * p(0) + 0.3 * p(0) * p(1) * p(2));
  }
  return ue;
}

Material j2_steel() {
  Material m(200.0e9, 0.3, 7850.0, "j2");
  PlasticityParameters p;
  p.yield_stress = 250.0e6;
  p.hardening_modulus = 2.0e9;
  p.kinematic_hardening_modulus = 1.0e9;
  m.set_plasticity(p);
  return m;
}

Material hill_sheet() {
  Material m(200.0e9, 0.3, 7850.0, "hill");
  PlasticityParameters p;
  p.yield_stress = 250.0e6;
  p.hardening_modulus = 1.5e9;
  p.criterion = YieldCriterion::Hill48;
  p.hill.r0 = 1.9;
  p.hill.r45 = 1.5;
  p.hill.r90 = 2.3;
  p.hill.rolling_direction = Vector3(1.0, 0.5, 0.0);
  m.set_plasticity(p);
  return m;
}

Material chaboche_steel() {
  Material m(200.0e9, 0.3, 7850.0, "chaboche");
  PlasticityParameters p;
  p.yield_stress = 250.0e6;
  p.saturation_stress = 100.0e6;
  p.saturation_rate = 20.0;
  p.num_backstresses = 2;
  p.backstresses[0] = Backstress{40.0e9, 400.0};
  p.backstresses[1] = Backstress{5.0e9, 20.0};
  m.set_plasticity(p);
  return m;
}

}  // namespace

TEST_CASE("the condensed elastoplastic tangent of the incompatible-mode Hex8 is the derivative "
          "of its force",
          "[incompatible][nonlinear][plasticity]") {
  struct Case {
    const char* name;
    Material material;
    Kinematics kinematics;
    Scalar scale;        // of the deformation pattern
    int thickness_points;
  };
  const Material elastic(200.0e9, 0.3, 7850.0, "elastic");
  const std::vector<Case> cases = {
      {"small strain, elastic", elastic, Kinematics::SmallStrain, 1.0e-3, 0},
      {"small strain, J2", j2_steel(), Kinematics::SmallStrain, 4.0e-3, 0},
      {"small strain, Hill48", hill_sheet(), Kinematics::SmallStrain, 4.0e-3, 3},
      {"small strain, Chaboche", chaboche_steel(), Kinematics::SmallStrain, 4.0e-3, 0},
      {"finite, elastic", elastic, Kinematics::Finite, 0.05, 0},
      {"finite, J2", j2_steel(), Kinematics::Finite, 0.02, 0},
      {"finite, Hill48", hill_sheet(), Kinematics::Finite, 0.02, 5},
      {"finite, Chaboche", chaboche_steel(), Kinematics::Finite, 0.02, 0},
      {"logarithmic, elastic", elastic, Kinematics::FiniteLogarithmic, 0.05, 0},
      {"logarithmic, J2", j2_steel(), Kinematics::FiniteLogarithmic, 0.1, 0},
      {"logarithmic, Hill48", hill_sheet(), Kinematics::FiniteLogarithmic, 0.1, 3},
      {"logarithmic, Chaboche", chaboche_steel(), Kinematics::FiniteLogarithmic, 0.1, 0}};
  for (const Case& c : cases) {
    INFO(c.name);
    const FemModel model = single_hex(c.material, c.thickness_points);
    const std::size_t points = static_cast<std::size_t>(elastoplastic_points(model));
    const std::vector<PlasticState> virgin(points);
    const Matrix x = model.mesh().element_coordinates(0);
    const bool plastic = c.material.plasticity().enabled();
    // A first step, committed; the check is at a second one from it.
    const ElastoplasticElement first = elastoplastic_element(
        model, 0, deformation(x, c.scale), virgin, false, nullptr, 0.0, true, c.kinematics);
    REQUIRE(first.internal.size() == 9);
    const std::vector<PlasticState> committed = plastic ? first.states : virgin;
    const Vector alpha = first.internal;
    Vector ue = deformation(x, 1.3 * c.scale);
    for (int a = 0; a < 8; ++a) ue(3 * a + 2) += 0.2 * c.scale * x(0, a) * x(1, a);
    const ElastoplasticElement el = elastoplastic_element(model, 0, ue, committed, false, nullptr,
                                                          0.0, true, c.kinematics, &alpha);
    if (plastic) REQUIRE(el.yielding_points == static_cast<int>(points));
    REQUIRE(el.internal_iterations >= 1);
    REQUIRE(el.internal_iterations <= (c.kinematics == Kinematics::SmallStrain && !plastic ? 1 : 5));
    REQUIRE(el.symmetric == c.material.plasticity().symmetric_tangent());
    if (el.symmetric) {
      REQUIRE((el.tangent - el.tangent.transpose()).cwiseAbs().maxCoeff() == 0.0);
    } else {
      REQUIRE((el.tangent - el.tangent.transpose()).cwiseAbs().maxCoeff() >
              1.0e-6 * el.tangent.cwiseAbs().maxCoeff());
    }
    const Scalar err = tangent_error(el.tangent, ue, 1.0e-8, [&](const Vector& v) {
      return elastoplastic_element(model, 0, v, committed, false, nullptr, 0.0, false,
                                   c.kinematics, &alpha)
          .internal_force;
    });
    INFO("relative tangent error " << err << ", local iterations " << el.internal_iterations);
    REQUIRE(err <= 1.0e-6);
    // Elastic: the force is the derivative of the stored energy (the modes
    // at their stationary point).
    if (!plastic) {
      Scalar energy_err = 0.0;
      for (int j = 0; j < 24; ++j) {
        Vector up = ue;
        Vector um = ue;
        up(j) += 1.0e-7;
        um(j) -= 1.0e-7;
        const Scalar wp = elastoplastic_element(model, 0, up, committed, false, nullptr, 0.0,
                                                false, c.kinematics, &alpha).energy;
        const Scalar wm = elastoplastic_element(model, 0, um, committed, false, nullptr, 0.0,
                                                false, c.kinematics, &alpha).energy;
        energy_err = std::max(energy_err, std::abs((wp - wm) / 2.0e-7 - el.internal_force(j)));
      }
      REQUIRE(energy_err <= 1.0e-6 * el.internal_force.cwiseAbs().maxCoeff());
    }
  }
}

TEST_CASE("the local iteration of the incompatible modes converges through a load reversal "
          "far from the committed state",
          "[incompatible][nonlinear][plasticity]") {
  // A plastic step committed, then a reversal to six times the strain the
  // other way in one increment: the points unload and yield again, on which
  // a plain Newton iteration on the piecewise-smooth r_alpha can cycle, and
  // the full step of the logarithmic kinematics inverts points; the line
  // search on the directional residual takes it through.
  for (const Material& material : {j2_steel(), chaboche_steel()}) {
    for (Kinematics k : {Kinematics::SmallStrain, Kinematics::FiniteLogarithmic}) {
      INFO(material.name() << ", " << to_string(k));
      const FemModel model = single_hex(material, 5);
      const std::size_t points = static_cast<std::size_t>(elastoplastic_points(model));
      const std::vector<PlasticState> virgin(points);
      const Matrix x = model.mesh().element_coordinates(0);
      const Scalar scale = k == Kinematics::SmallStrain ? 4.0e-3 : 0.02;
      const ElastoplasticElement first = elastoplastic_element(
          model, 0, deformation(x, scale), virgin, false, nullptr, 0.0, false, k);
      REQUIRE(first.yielding_points > 0);
      Vector ue = deformation(x, -6.0 * scale);
      for (int a = 0; a < 8; ++a) ue(3 * a) += 2.0 * scale * x(2, a) * x(2, a);
      const ElastoplasticElement el = elastoplastic_element(
          model, 0, ue, first.states, false, nullptr, 0.0, true, k, &first.internal);
      INFO("local iterations " << el.internal_iterations);
      REQUIRE(el.yielding_points > 0);
      // Measured: 3 (small strain, J2 and Chaboche), 9 (logarithmic, J2),
      // 12 (logarithmic, Chaboche, whose tangent is not symmetric).
      REQUIRE(el.internal_iterations <= 15);
      // And the result is the solution: a restart of the local iteration
      // from it stays there.
      const ElastoplasticElement again = elastoplastic_element(
          model, 0, ue, first.states, false, nullptr, 0.0, false, k, &el.internal);
      REQUIRE(again.internal_iterations <= 1);
      REQUIRE((again.internal_force - el.internal_force).cwiseAbs().maxCoeff() <=
              1.0e-8 * el.internal_force.cwiseAbs().maxCoeff());
    }
  }
}

TEST_CASE("the local iteration of the incompatible modes converges from reversals of up to "
          "fifteen times the strain under logarithmic strains, where K_aa is indefinite",
          "[incompatible][nonlinear][plasticity]") {
  // A harder case than the one above: logarithmic strains of 2 %, reversed
  // to 12 to 30 % in one increment, with perfect plasticity among the laws.
  // The plastic tangent keeps little deviatoric stiffness against the
  // geometric term of the stress, so the symmetric part of K_aa is
  // indefinite over most of the iteration (measured: smallest eigenvalue
  // down to -0.03 of the largest), and Newton's direction climbs the
  // potential; the iteration then searches along the shifted direction of
  // (K_aa + mu I) d = -r_alpha, and never moves to a worse |r_alpha| on its
  // fallbacks. Before, 18 of 48 such cases (the four laws, rules of 2, 5 and
  // 7 thickness points, reversals of 3, 6, 10 and 15 times) ran out of
  // iterations or inverted every trial; now 1 does (Chaboche with the
  // 2 x 2 x 2 rule at 10 times, a non-associative law without a potential).
  Material perfect(200.0e9, 0.3, 7850.0, "perfectly plastic");
  PlasticityParameters p;
  p.yield_stress = 250.0e6;
  perfect.set_plasticity(p);
  for (const Material& material : {perfect, j2_steel(), chaboche_steel(), hill_sheet()}) {
    for (int thickness_points : {5, 7}) {
      for (Scalar factor : {-6.0, -10.0, -15.0}) {
        INFO(material.name() << ", " << thickness_points << " thickness points, reversal "
                             << factor);
        const FemModel model = single_hex(material, thickness_points);
        const std::size_t points = static_cast<std::size_t>(elastoplastic_points(model));
        const std::vector<PlasticState> virgin(points);
        const Matrix x = model.mesh().element_coordinates(0);
        const Kinematics k = Kinematics::FiniteLogarithmic;
        const ElastoplasticElement first = elastoplastic_element(
            model, 0, deformation(x, 0.02), virgin, false, nullptr, 0.0, false, k);
        const Vector ue = deformation(x, factor * 0.02);
        const ElastoplasticElement el = elastoplastic_element(
            model, 0, ue, first.states, false, nullptr, 0.0, false, k, &first.internal);
        INFO("local iterations " << el.internal_iterations);
        REQUIRE(el.yielding_points > 0);
        // Measured: 9 to 18.
        REQUIRE(el.internal_iterations <= 20);
        const ElastoplasticElement again = elastoplastic_element(
            model, 0, ue, first.states, false, nullptr, 0.0, false, k, &el.internal);
        REQUIRE(again.internal_iterations <= 1);
        REQUIRE((again.internal_force - el.internal_force).cwiseAbs().maxCoeff() <=
                1.0e-8 * el.internal_force.cwiseAbs().maxCoeff());
      }
    }
  }
}

TEST_CASE("the thermal load rate of the incompatible-mode Hex8 is the derivative of its force",
          "[incompatible][nonlinear][thermal]") {
  Material m = j2_steel();
  m.set_thermal(1.2e-5, 20.0, 50.0);
  const FemModel model = single_hex(m);
  const Matrix x = model.mesh().element_coordinates(0);
  Vector temperature(8);
  for (int a = 0; a < 8; ++a) temperature(a) = 20.0 + 300.0 * x(0, a) - 150.0 * x(2, a);
  const std::vector<PlasticState> virgin(8);
  for (Kinematics k : {Kinematics::SmallStrain, Kinematics::Finite, Kinematics::FiniteLogarithmic}) {
    INFO(to_string(k));
    const Vector ue = deformation(x, 2.0e-3);
    const Scalar lambda = 0.7;
    const ElastoplasticElement el =
        elastoplastic_element(model, 0, ue, virgin, false, &temperature, lambda, false, k);
    REQUIRE(el.thermal_force_rate.size() == 24);
    const Scalar h = 1.0e-6;
    const Vector fp = elastoplastic_element(model, 0, ue, virgin, false, &temperature,
                                            lambda + h, false, k).internal_force;
    const Vector fm = elastoplastic_element(model, 0, ue, virgin, false, &temperature,
                                            lambda - h, false, k).internal_force;
    REQUIRE(((fp - fm) / (2.0 * h) - el.thermal_force_rate).cwiseAbs().maxCoeff() <=
            1.0e-6 * el.thermal_force_rate.cwiseAbs().maxCoeff());
  }
}

TEST_CASE("a homogeneous deformation leaves the incompatible modes at zero under every "
          "kinematics",
          "[incompatible][nonlinear][patch]") {
  Matrix3 f;
  f << 1.04, 0.03, -0.02, 0.01, 0.97, 0.05, -0.03, 0.02, 1.02;
  const std::vector<std::pair<const char*, Material>> materials = {
      {"elastic", Material(200.0e9, 0.3, 7850.0, "elastic")},
      {"J2", j2_steel()},
      {"Hill48", hill_sheet()}};
  for (const auto& [name, material] : materials) {
    const FemModel model = single_hex(material, 3);
    const FemModel standard(Mesh(distorted_hex_coords(), {0, 1, 2, 3, 4, 5, 6, 7},
                                 ElementType::Hex8),
                            material, 1.0, StressState::ThreeDimensional, [] {
                              IntegrationOptions o;
                              o.thickness_points = 3;
                              return o;
                            }());
    const Matrix x = model.mesh().element_coordinates(0);
    for (Kinematics k : {Kinematics::SmallStrain, Kinematics::Finite,
                         Kinematics::FiniteLogarithmic}) {
      INFO(name << ", " << to_string(k));
      const Scalar scale = k == Kinematics::SmallStrain ? 0.1 : 1.0;
      Vector ue(24);
      for (int a = 0; a < 8; ++a) ue.segment<3>(3 * a) = scale * (f - Matrix3::Identity()) * x.col(a);
      const std::vector<PlasticState> virgin(static_cast<std::size_t>(elastoplastic_points(model)));
      const ElastoplasticElement el =
          elastoplastic_element(model, 0, ue, virgin, false, nullptr, 0.0, false, k);
      const ElastoplasticElement ref =
          elastoplastic_element(standard, 0, ue, virgin, false, nullptr, 0.0, false, k);
      REQUIRE(el.internal.cwiseAbs().maxCoeff() <= 1.0e-12);
      REQUIRE((el.internal_force - ref.internal_force).cwiseAbs().maxCoeff() <=
              1.0e-10 * ref.internal_force.cwiseAbs().maxCoeff());
    }
  }
}

TEST_CASE("the non-linear incompatible-mode Hex8 is objective and its tangent at rest is the "
          "linear one",
          "[incompatible][nonlinear]") {
  const Material material = j2_steel();
  const FemModel model = single_hex(material);
  const Matrix x = model.mesh().element_coordinates(0);
  const std::vector<PlasticState> virgin(8);
  const Matrix3 r = rotation();
  for (Kinematics k : {Kinematics::Finite, Kinematics::FiniteLogarithmic}) {
    INFO(to_string(k));
    // A rigid rotation: no force, no modes.
    Vector rigid(24);
    for (int a = 0; a < 8; ++a) rigid.segment<3>(3 * a) = (r - Matrix3::Identity()) * x.col(a);
    const ElastoplasticElement at_rest =
        elastoplastic_element(model, 0, rigid, virgin, false, nullptr, 0.0, false, k);
    REQUIRE(at_rest.internal_force.cwiseAbs().maxCoeff() <= 1.0e-3);  // [N], of 1e11 N/m
    REQUIRE(at_rest.internal.cwiseAbs().maxCoeff() <= 1.0e-14);
    // A rotation superposed on a plastic deformation rotates the force and
    // the modes.
    const Vector ue = deformation(x, 0.02);
    Vector rotated(24);
    for (int a = 0; a < 8; ++a) {
      rotated.segment<3>(3 * a) = r * (x.col(a) + ue.segment<3>(3 * a)) - x.col(a);
    }
    const ElastoplasticElement el =
        elastoplastic_element(model, 0, ue, virgin, false, nullptr, 0.0, false, k);
    const ElastoplasticElement er =
        elastoplastic_element(model, 0, rotated, virgin, false, nullptr, 0.0, false, k);
    REQUIRE(el.yielding_points == 8);
    for (int a = 0; a < 8; ++a) {
      REQUIRE((er.internal_force.segment<3>(3 * a) - r * el.internal_force.segment<3>(3 * a))
                  .cwiseAbs()
                  .maxCoeff() <= 1.0e-9 * el.internal_force.cwiseAbs().maxCoeff());
    }
    for (int m = 0; m < 3; ++m) {
      REQUIRE((er.internal.segment<3>(3 * m) - r * el.internal.segment<3>(3 * m))
                  .cwiseAbs()
                  .maxCoeff() <= 1.0e-9 * el.internal.cwiseAbs().maxCoeff());
    }
  }
  // At rest every kinematics has the linear condensed stiffness.
  const Material elastic(200.0e9, 0.3, 7850.0, "elastic");
  const FemModel em = single_hex(elastic);
  const Matrix k = em.element().stiffness(x, elastic.three_dimensional_matrix(), 1.0,
                                          em.integration());
  for (Kinematics kin : {Kinematics::SmallStrain, Kinematics::Finite,
                         Kinematics::FiniteLogarithmic}) {
    const ElastoplasticElement el = elastoplastic_element(em, 0, Vector::Zero(24), virgin, false,
                                                          nullptr, 0.0, true, kin);
    REQUIRE((el.tangent - k).cwiseAbs().maxCoeff() <= 1.0e-12 * k.cwiseAbs().maxCoeff());
    REQUIRE(el.internal_iterations == 0);
  }
}

// ---------------------------------------------------------------------------
// Analyses: determinism, the committed modes, restart, configuration
// ---------------------------------------------------------------------------

namespace {

/// A clamped strip along x, one element through its thickness (z), under a
/// tip load `load` [N] along -z.
FemModel strip(const Material& m, Index nx, int thickness_points,
               ElementFormulation formulation = ElementFormulation::IncompatibleModes,
               Scalar perturbation = 0.0) {
  StructuredMeshSpec spec;
  spec.nx = nx;
  spec.ny = 1;
  spec.nz = 1;
  spec.lx = 0.1;
  spec.ly = 0.01;
  spec.lz = 0.002;
  IntegrationOptions o;
  o.formulation = formulation;
  o.thickness_points = thickness_points;
  FemModel model(perturbation > 0.0 ? make_perturbed_hex_mesh(spec, perturbation)
                                    : make_structured_hex_mesh(spec),
                 m, 1.0, StressState::ThreeDimensional, o);
  DisplacementConstraint root;
  Selector s;
  s.kind = SelectorKind::Box;
  s.xmax = 0.0;
  root.region.members.push_back(s);
  root.fix_x = root.fix_y = root.fix_z = true;
  model.constraints().push_back(root);
  LoadCaseSpec lc;
  lc.name = "tip";
  model.load_case_specs().push_back(lc);
  model.finalize();
  return model;
}

Material aluminium() {
  Material m(70.0e9, 0.33, 2700.0, "aluminium");
  PlasticityParameters p;
  p.yield_stress = 150.0e6;
  p.hardening_modulus = 500.0e6;
  m.set_plasticity(p);
  return m;
}

}  // namespace

TEST_CASE("the incompatible-mode system evaluates bitwise alike on one and several threads",
          "[incompatible][nonlinear]") {
  const FemModel model = strip(aluminium(), 12, 3, ElementFormulation::IncompatibleModes, 0.2);
  Assembler assembler(model);
  NonlinearOptions nl;
  nl.kinematics = Kinematics::Finite;
  // The generic element dispatch of NonlinearSystem, which the forming
  // analysis evaluates (the explicit integrator's comparison hook).
  const ExplicitDynamics system(model, assembler, nl, {}, ExplicitOptions());
  Vector u = Vector::Zero(model.dofs().num_dofs());
  for (Index n = 0; n < model.mesh().num_nodes(); ++n) {
    const Vector3 x = model.mesh().node(n);
    u(3 * n + 2) = -2.0 * x(0) * x(0);  // a bending that yields
    u(3 * n) = -4.0 * x(0) * (x(2) - 0.001);
  }
#if defined(SPARLAB_HAVE_OPENMP)
  const int threads = omp_get_max_threads();
  omp_set_num_threads(1);
#endif
  const Vector serial = system.internal_force(u, {}, /*generic=*/true);
#if defined(SPARLAB_HAVE_OPENMP)
  omp_set_num_threads(3);
#endif
  const Vector parallel = system.internal_force(u, {}, /*generic=*/true);
#if defined(SPARLAB_HAVE_OPENMP)
  omp_set_num_threads(threads);
#endif
  REQUIRE(serial == parallel);
}

TEST_CASE("a forming restart carries the committed incompatible modes", "[incompatible][forming]") {
  const Scalar length = 0.04;
  StructuredMeshSpec spec;
  spec.nx = 8;
  spec.ny = 1;
  spec.nz = 1;
  spec.lx = length;
  spec.ly = 0.006;
  spec.lz = 0.002;
  IntegrationOptions o = incompatible_options(5);
  FemModel model(make_structured_hex_mesh(spec), aluminium(), 1.0,
                 StressState::ThreeDimensional, o);
  LoadCaseSpec lc;
  lc.name = "none";
  model.load_case_specs().push_back(lc);
  model.finalize();
  Assembler assembler(model);
  const auto region = [](Scalar xmin, Scalar xmax, const char* name) {
    SelectorGroup g;
    g.name = name;
    Selector s;
    s.kind = SelectorKind::Box;
    s.xmin = xmin;
    s.xmax = xmax;
    g.members.push_back(s);
    return g;
  };
  const auto held = [](const SelectorGroup& g, std::vector<int> components, Scalar value) {
    StepConstraint c;
    c.name = g.name;
    c.constraint.region = g;
    for (int k : components) c.constraint.set(k, true, k == 2 ? value : 0.0);
    c.mode = StepConstraint::Mode::Absolute;
    return c;
  };
  FormingOptions fo;
  fo.kinematics = Kinematics::Finite;
  fo.residual_tolerance = 1.0e-10;
  fo.displacement_tolerance = 1.0e-10;
  FormingStep bend;
  bend.name = "bend";
  bend.increments = 4;
  bend.constraints.push_back(held(region(-1.0, 0.0, "root"), {0, 1, 2}, 0.0));
  bend.constraints.push_back(held(region(length, 1.0, "tip"), {2}, 0.004));
  FormingStep unload = bend;
  unload.name = "unload";
  unload.constraints[1] = held(region(length, 1.0, "tip"), {2}, 0.001);
  // In one run, and as a restart from the state after the bend.
  FormingOptions both = fo;
  both.steps = {bend, unload};
  const FormingResult whole = FormingAnalysis(model, assembler, both).run();
  REQUIRE(whole.completed);
  FormingOptions first = fo;
  first.steps = {bend};
  const FormingResult part = FormingAnalysis(model, assembler, first).run();
  REQUIRE(part.completed);
  REQUIRE(part.steps[0].max_plastic_strain > 1.0e-3);
  REQUIRE(part.final_state.internal.size() == static_cast<std::size_t>(model.mesh().num_elements()));
  Scalar largest = 0.0;
  for (const Vector& a : part.final_state.internal) largest = std::max(largest, a.cwiseAbs().maxCoeff());
  REQUIRE(largest > 1.0e-7);  // [m]: the modes carry the bending
  FormingOptions second = fo;
  second.steps = {unload};
  const FormingResult rest = FormingAnalysis(model, assembler, second).run(part.final_state);
  REQUIRE(rest.completed);
  const Vector& u_whole = whole.final_state.displacement;
  REQUIRE((rest.final_state.displacement - u_whole).cwiseAbs().maxCoeff() <=
          1.0e-12 * u_whole.cwiseAbs().maxCoeff());
  // The committed modes are the start of the local iteration, not history:
  // without them the restart finds the same state, to the local tolerance.
  AnalysisState cold = part.final_state;
  cold.internal.clear();
  const FormingResult again = FormingAnalysis(model, assembler, second).run(cold);
  REQUIRE(again.completed);
  REQUIRE((again.final_state.displacement - u_whole).cwiseAbs().maxCoeff() <=
          1.0e-8 * u_whole.cwiseAbs().maxCoeff());
  // A state of another formulation is refused.
  AnalysisState wrong = part.final_state;
  wrong.internal.pop_back();
  REQUIRE_THROWS_AS(FormingAnalysis(model, assembler, second).run(wrong), ConfigError);
}

TEST_CASE("mean dilatation is refused with incompatible modes, and auto does not apply it",
          "[incompatible][nonlinear][config]") {
  const FemModel model = strip(aluminium(), 4, 0);
  Assembler assembler(model);
  NonlinearOptions nl;
  nl.mean_dilatation = MeanDilatation::All;
  REQUIRE_THROWS_AS(ExplicitDynamics(model, assembler, nl, {}, ExplicitOptions()), ConfigError);
  const std::vector<PlasticState> virgin(8);
  REQUIRE_THROWS_AS(elastoplastic_element(model, 0, Vector::Zero(24), virgin, true, nullptr, 0.0,
                                          false, Kinematics::SmallStrain),
                    ConfigError);
}

TEST_CASE("the element formulation and the thickness rule are read from a deck and refused "
          "where wrong",
          "[incompatible][config][io]") {
  const auto deck = [](const std::string& mesh, const std::string& model) {
    return R"({"mesh": )" + mesh + R"(, "material": {"youngs_modulus": 7e10, "poisson_ratio": 0.3},
      "model": )" + model + R"(,
      "boundary_conditions": [{"fix": ["x", "y"], "region": {"box": {"xmax": 0}}}]})";
  };
  const std::string hex = R"({"type": "structured_hex", "nx": 4, "ny": 1, "nz": 1, "lx": 1,
      "ly": 0.1, "lz": 0.02})";
  const auto parse = [](const std::string& text) {
    return parse_configuration(json::parse(text, "deck"), "deck", /*strict=*/true);
  };
  const auto fix3 = [](std::string text) {
    const std::string two = R"("fix": ["x", "y"])";
    text.replace(text.find(two), two.size(), R"("fix": ["x", "y", "z"])");
    return text;
  };
  const Configuration config = parse(fix3(deck(hex, R"({"element_formulation": "incompatible_modes",
      "integration": {"thickness_points": 5, "thickness_direction": "z"}})")));
  REQUIRE(config.integration.formulation == ElementFormulation::IncompatibleModes);
  REQUIRE(config.integration.thickness_points == 5);
  REQUIRE(config.integration.thickness_axis == 2);
  const FemModel model = build_model(config);
  REQUIRE(model.element().num_internal_dofs() == 9);
  REQUIRE(elastoplastic_points(model) == 20);
  // The default is the standard element with its cube rule.
  const FemModel plain = build_model(parse(fix3(deck(hex, "{}"))));
  REQUIRE(plain.element().num_internal_dofs() == 0);
  REQUIRE(elastoplastic_points(plain) == 8);
  // Refusals.
  REQUIRE_THROWS_AS(parse(fix3(deck(hex, R"({"element_formulation": "eas"})"))), ConfigError);
  REQUIRE_THROWS_AS(parse(fix3(deck(hex, R"({"integration": {"thickness_points": 8}})"))),
                    ConfigError);
  REQUIRE_THROWS_AS(parse(fix3(deck(hex, R"({"integration": {"thickness_points": -1}})"))),
                    ConfigError);
  REQUIRE_THROWS_AS(
      parse(fix3(deck(hex, R"({"integration": {"thickness_points": 3, "thickness_direction": "t"}})"))),
      ConfigError);
}

TEST_CASE("a rule of one point along a natural axis is refused for the incompatible-mode Hex8",
          "[incompatible][config]") {
  // The gradients of the modes vanish at the element centre: one point
  // leaves the modes without stiffness, which would only surface later as a
  // singular K_aa and a cut step.
  const auto build = [](ElementFormulation f, int stiffness_points, int thickness_points) {
    IntegrationOptions o;
    o.formulation = f;
    o.stiffness_points = stiffness_points;
    o.thickness_points = thickness_points;
    return FemModel(Mesh(unit_box_coords(), {0, 1, 2, 3, 4, 5, 6, 7}, ElementType::Hex8),
                    Material(200.0e9, 0.3, 7850.0, "steel"), 1.0,
                    StressState::ThreeDimensional, o);
  };
  REQUIRE_THROWS_AS(build(ElementFormulation::IncompatibleModes, 1, 0), ConfigError);
  REQUIRE_THROWS_AS(build(ElementFormulation::IncompatibleModes, 2, 1), ConfigError);
  REQUIRE_THROWS_AS(build(ElementFormulation::IncompatibleModes, 1, 3), ConfigError);
  REQUIRE(build(ElementFormulation::IncompatibleModes, 2, 0).element().num_internal_dofs() == 9);
  REQUIRE(build(ElementFormulation::IncompatibleModes, 2, 2).element().num_internal_dofs() == 9);
  // The standard element keeps its one-point rules.
  REQUIRE(build(ElementFormulation::Standard, 1, 0).element().num_internal_dofs() == 0);
  REQUIRE(build(ElementFormulation::Standard, 2, 1).element().num_internal_dofs() == 0);
}

namespace {

/// The smallest eigenvalue, relative to the largest, of an element tangent
/// reduced by the held degrees of freedom [node][component].
Scalar reduced_smallest_eigenvalue(const Matrix& tangent, const bool held[8][3]) {
  std::vector<Eigen::Index> free;
  for (int a = 0; a < 8; ++a) {
    for (int k = 0; k < 3; ++k) {
      if (!held[a][k]) free.push_back(3 * a + k);
    }
  }
  const Eigen::Index n = static_cast<Eigen::Index>(free.size());
  Matrix reduced(n, n);
  for (Eigen::Index i = 0; i < n; ++i) {
    for (Eigen::Index j = 0; j < n; ++j) {
      reduced(i, j) = tangent(free[static_cast<std::size_t>(i)], free[static_cast<std::size_t>(j)]);
    }
  }
  const Vector lambda = eigenvalues(0.5 * (reduced + reduced.transpose()));
  return lambda(0) / lambda.cwiseAbs().maxCoeff();
}

}  // namespace

TEST_CASE("under plastic compression the incompatible-mode Hex8 develops its hourglass mode the "
          "sooner the less the material hardens",
          "[incompatible][nonlinear][stability][plasticity]") {
  // Wriggers and Reese (1996, CMAME 135): the enhanced elements of finite
  // deformation lose stability under large compression although the
  // material and the compatible element are stable - here a spurious
  // hourglass mode of an elastoplastic cube (J2, E = 70 GPa, yield stress
  // 100 MPa, logarithmic strains) in homogeneous uniaxial compression along
  // x, the lateral stretch found per increment of 0.01 so that the faces
  // y = 1 and z = 1 carry no force, the plastic state committed each
  // increment, its tangent reduced to the test: u_x held on the faces x = 0
  // and 1, the remaining rigid motions removed at nodes 0 and 3. A plastic tangent keeps little
  // deviatoric stiffness against the geometric term of the stress, so the
  // onset depends on the hardening.
  const auto run = [](Scalar hardening, ElementFormulation f, Scalar stop) {
    Material m(70.0e9, 0.3, 2700.0, "aluminium");
    PlasticityParameters p;
    p.yield_stress = 100.0e6;
    p.hardening_modulus = hardening;
    m.set_plasticity(p);
    IntegrationOptions o;
    o.formulation = f;
    const FemModel model(Mesh(unit_box_coords(), {0, 1, 2, 3, 4, 5, 6, 7}, ElementType::Hex8), m,
                         1.0, StressState::ThreeDimensional, o);
    const Matrix x = unit_box_coords();
    bool held[8][3] = {};
    for (int a = 0; a < 8; ++a) held[a][0] = true;
    held[0][1] = held[0][2] = held[3][2] = true;
    std::vector<PlasticState> committed(static_cast<std::size_t>(elastoplastic_points(model)));
    Vector alpha;
    Scalar lateral = 1.0;
    std::vector<std::pair<Scalar, Scalar>> path;  // (axial stretch, smallest eigenvalue)
    for (int i = 1; 1.0 - 0.01 * i >= stop - 1.0e-12; ++i) {
      const Scalar axial = 1.0 - 0.01 * i;
      const auto element = [&](Scalar l, bool tangent) {
        Vector ue(24);
        for (int a = 0; a < 8; ++a) {
          ue.segment<3>(3 * a) =
              Vector3(axial - 1.0, l - 1.0, l - 1.0).cwiseProduct(Vector3(x.col(a)));
        }
        return elastoplastic_element(model, 0, ue, committed, false, nullptr, 0.0, tangent,
                                     Kinematics::FiniteLogarithmic, &alpha);
      };
      // The lateral force: f_y on the face y = 1.
      const auto lateral_force = [&](const ElastoplasticElement& el) {
        Scalar sum = 0.0;
        for (int a = 0; a < 8; ++a) {
          if (x(1, a) > 0.5) sum += el.internal_force(3 * a + 1);
        }
        return sum;
      };
      // Secant iteration on the lateral stretch.
      Scalar l0 = lateral;
      Scalar f0 = lateral_force(element(l0, false));
      Scalar l1 = lateral + 0.005;
      for (int it = 0; it < 40; ++it) {
        const Scalar f1 = lateral_force(element(l1, false));
        if (std::abs(f1) <= 1.0e-4 || f1 == f0) break;
        const Scalar next = l1 - f1 * (l1 - l0) / (f1 - f0);
        l0 = l1;
        f0 = f1;
        l1 = next;
      }
      lateral = l1;
      const ElastoplasticElement el = element(lateral, true);
      REQUIRE(std::abs(lateral_force(el)) <= 1.0e-3);  // [N], of 1e8 N
      committed = el.states;
      alpha = el.internal;
      path.emplace_back(axial, reduced_smallest_eigenvalue(el.tangent, held));
    }
    return path;
  };
  // Measured: the smallest eigenvalue of the incompatible-mode element
  // passes zero between the stretches 0.80 and 0.79 without hardening
  // (7.8e-6 and -5.3e-6 of the largest; about 21 % compression, with the
  // plastic strain about 0.23), between 0.62 and 0.61 with H = 300 MPa
  // (1.0e-4 and -3.9e-6). Before that it is 20 (no hardening) or 3 times
  // smaller than that of the standard element, which stays at 3.1e-3 to
  // 7.0e-3 down to 0.55.
  struct Onset {
    Scalar hardening;  // [Pa]
    Scalar stable;     // the smallest stretch still stable
  };
  for (const Onset& c : {Onset{0.0, 0.80}, Onset{300.0e6, 0.62}}) {
    const auto modes = run(c.hardening, ElementFormulation::IncompatibleModes, 0.55);
    const auto standard = run(c.hardening, ElementFormulation::Standard, 0.55);
    REQUIRE(modes.size() == standard.size());
    for (std::size_t i = 0; i < modes.size(); ++i) {
      const Scalar axial = modes[i].first;
      INFO("H = " << c.hardening << " Pa, stretch " << axial << ": " << modes[i].second
                  << " (standard " << standard[i].second << ")");
      REQUIRE(standard[i].second > 3.0e-3);
      REQUIRE(modes[i].second < standard[i].second);
      if (axial >= c.stable - 1.0e-12) REQUIRE(modes[i].second > 0.0);
      if (axial <= c.stable - 0.01 + 1.0e-12) REQUIRE(modes[i].second < 0.0);
    }
  }
}
