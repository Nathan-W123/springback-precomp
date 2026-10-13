/// \file test_hyperelastic.cpp
/// \brief The large-deformation material laws and the total Lagrangian
///        element on the standard Hex8.
///
/// Every check is exact or a derivative check: the laws' stresses against
/// their energies and their tangents against their stresses, the element
/// tangent against central differences of the internal force, objectivity
/// under a rigid rotation and the small-strain limit.
#include "TestSupport.hpp"

#include "sparlab/core/Exceptions.hpp"
#include "sparlab/fem/TotalLagrangian.hpp"
#include "sparlab/material/Hyperelastic.hpp"

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

constexpr Scalar kInf = std::numeric_limits<Scalar>::infinity();

SelectorGroup box_region(Scalar xmin, Scalar xmax) {
  SelectorGroup g;
  Selector s;
  s.kind = SelectorKind::Box;
  s.xmin = xmin;
  s.xmax = xmax;
  g.members.push_back(s);
  g.name = "box";
  return g;
}

DisplacementConstraint fix(const SelectorGroup& region, bool x, bool y, bool z) {
  DisplacementConstraint bc;
  bc.region = region;
  bc.fix_x = x;
  bc.fix_y = y;
  bc.fix_z = z;
  return bc;
}

Matrix random_matrix(int rows, int cols, Scalar amplitude, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<Scalar> dist(-1.0, 1.0);
  Matrix m(rows, cols);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) m(i, j) = amplitude * dist(gen);
  }
  return m;
}

/// A distorted 2 x 2 x 2 Hex8 block of 1 x 0.8 x 0.6 m; its element 0 is the
/// cell the element tests evaluate.
Mesh perturbed_block() {
  StructuredMeshSpec spec;
  spec.nx = spec.ny = spec.nz = 2;
  spec.lx = 1.0;
  spec.ly = 0.8;
  spec.lz = 0.6;
  return make_perturbed_hex_mesh(spec, 0.2);
}

std::string law_name(HyperelasticModel law) {
  return law == HyperelasticModel::NeoHookean ? "neo_hookean" : "saint_venant_kirchhoff";
}

}  // namespace

TEST_CASE("the hyperelastic laws are consistent: energy, stress and tangent",
          "[hyperelastic][material]") {
  IsotropicMaterial m(200.0e9, 0.3, 7800.0, "steel");
  for (const HyperelasticModel law :
       {HyperelasticModel::SaintVenantKirchhoff, HyperelasticModel::NeoHookean}) {
    const StressState state = StressState::ThreeDimensional;
    // A large deformation: stretches, shears and a rotation.
    const Matrix grad = random_matrix(3, 3, 0.3, 7u);
    const HyperelasticResponse r = evaluate_hyperelastic(law, m, state, grad);
    const Matrix dg = random_matrix(3, 3, 1.0, 11u);
    const Scalar h = 1.0e-6;
    const HyperelasticResponse rp = evaluate_hyperelastic(law, m, state, grad + h * dg);
    const HyperelasticResponse rm = evaluate_hyperelastic(law, m, state, grad - h * dg);
    const Vector de = (green_lagrange_voigt(grad + h * dg) - green_lagrange_voigt(grad - h * dg)) /
                      (2.0 * h);
    INFO(law_name(law));
    // dW = S : dE and dS = C dE.
    const Scalar dw = (rp.energy - rm.energy) / (2.0 * h);
    REQUIRE(dw == Approx(r.stress.dot(de)).epsilon(1e-7));
    const Vector ds = (rp.stress - rm.stress) / (2.0 * h);
    const Vector predicted = r.tangent * de;
    REQUIRE((ds - predicted).norm() <= 1e-6 * predicted.norm());
    // The tangent of a hyperelastic law is symmetric.
    REQUIRE((r.tangent - r.tangent.transpose()).norm() <= 1e-12 * r.tangent.norm());
  }
}

TEST_CASE("both laws reduce to linear elasticity for small strain", "[hyperelastic][material]") {
  IsotropicMaterial m(70.0e9, 0.33, 2700.0);
  for (const HyperelasticModel law :
       {HyperelasticModel::SaintVenantKirchhoff, HyperelasticModel::NeoHookean}) {
    const Matrix grad = random_matrix(3, 3, 1.0e-7, 3u);
    const HyperelasticResponse r =
        evaluate_hyperelastic(law, m, StressState::ThreeDimensional, grad);
    const Matrix eps = 0.5 * (grad + grad.transpose());
    Vector ev(6);
    ev << eps(0, 0), eps(1, 1), eps(2, 2), 2.0 * eps(0, 1), 2.0 * eps(1, 2), 2.0 * eps(2, 0);
    const Vector linear = m.three_dimensional_matrix() * ev;
    INFO(law_name(law));
    // The difference is second order in the strain, 1e-7 here: no round-off
    // floor above it now that the laws work from the displacement gradient.
    REQUIRE((r.stress - linear).norm() <= 1e-6 * linear.norm());
    REQUIRE((r.tangent - m.three_dimensional_matrix()).norm() <=
            1e-5 * m.three_dimensional_matrix().norm());
  }
}

TEST_CASE("the neo-Hookean law refuses plane stress and a temperature",
          "[hyperelastic][material]") {
  IsotropicMaterial m(1.0e6, 0.45, 1000.0);
  REQUIRE_THROWS_AS(evaluate_hyperelastic(HyperelasticModel::NeoHookean, m,
                                          StressState::PlaneStress, Matrix::Zero(2, 2)),
                    ConfigError);
  m.set_thermal(1.0e-4, 20.0, 0.2);
  REQUIRE_THROWS_AS(evaluate_hyperelastic(HyperelasticModel::NeoHookean, m,
                                          StressState::ThreeDimensional, Matrix::Zero(3, 3),
                                          10.0),
                    ConfigError);
  // An inverted material point is reported, not evaluated.
  m.set_thermal(0.0, 0.0, 0.0);
  REQUIRE_THROWS_AS(evaluate_hyperelastic(HyperelasticModel::NeoHookean, m,
                                          StressState::ThreeDimensional,
                                          -2.0 * Matrix::Identity(3, 3)),
                    SolverError);
  REQUIRE(parse_hyperelastic_model("neo_hookean") == HyperelasticModel::NeoHookean);
  REQUIRE_THROWS_AS(parse_hyperelastic_model("mooney"), ConfigError);
}

TEST_CASE("the total Lagrangian tangent is the derivative of the internal force",
          "[hyperelastic][element]") {
  for (const HyperelasticModel law :
       {HyperelasticModel::SaintVenantKirchhoff, HyperelasticModel::NeoHookean}) {
    IsotropicMaterial mat(200.0e9, 0.3, 7800.0);
    FemModel model(perturbed_block(), mat, 1.0, StressState::ThreeDimensional,
                   IntegrationOptions());
    model.constraints().push_back(fix(box_region(-kInf, 0.0), true, true, true));
    LoadCaseSpec lc;
    lc.name = "load";
    model.load_case_specs().push_back(lc);
    model.finalize();
    const int dim = model.dim();
    REQUIRE(dim == 3);
    const Index e = 0;
    const int nd = dim * model.mesh().nodes_per_elem();
    // Large displacements: ~15 % of the cell plus a rigid turn of 0.4 rad.
    const Matrix x0 = model.mesh().element_coordinates(e);
    Matrix rot = Matrix::Identity(dim, dim);
    rot(0, 0) = std::cos(0.4);
    rot(0, 1) = -std::sin(0.4);
    rot(1, 0) = std::sin(0.4);
    rot(1, 1) = std::cos(0.4);
    // (Strains of up to ~10 %: enough to exercise every non-linear term,
    // short of inverting the cell.)
    const Matrix disp = (rot - Matrix::Identity(dim, dim)) * x0 +
                        random_matrix(dim, static_cast<int>(x0.cols()), 0.02, 5u);
    Vector ue(nd);
    for (int a = 0; a < model.mesh().nodes_per_elem(); ++a) {
      for (int k = 0; k < dim; ++k) ue(dim * a + k) = disp(k, a);
    }
    const TotalLagrangianElement base =
        total_lagrangian_element(model, e, ue, law, nullptr, 1.0, true);
    const Scalar h = 1.0e-7;
    Scalar worst = 0.0;
    for (int j = 0; j < nd; ++j) {
      Vector up = ue;
      Vector um = ue;
      up(j) += h;
      um(j) -= h;
      const Vector fd = (total_lagrangian_element(model, e, up, law, nullptr, 1.0, false)
                             .internal_force -
                         total_lagrangian_element(model, e, um, law, nullptr, 1.0, false)
                             .internal_force) /
                        (2.0 * h);
      worst = std::max(worst, (fd - base.tangent.col(j)).cwiseAbs().maxCoeff());
    }
    INFO(law_name(law));
    REQUIRE(worst <= 1.0e-6 * base.tangent.cwiseAbs().maxCoeff());
    // And the internal force is the derivative of the energy.
    const Vector dir = random_matrix(nd, 1, 1.0, 13u);
    const Scalar de =
        (total_lagrangian_element(model, e, ue + h * dir, law, nullptr, 1.0, false).energy -
         total_lagrangian_element(model, e, ue - h * dir, law, nullptr, 1.0, false).energy) /
        (2.0 * h);
    REQUIRE(de == Approx(base.internal_force.dot(dir)).epsilon(1e-6));
  }
}

TEST_CASE("a rigid rotation leaves no internal force and the reference tangent is linear",
          "[hyperelastic][element]") {
  IsotropicMaterial mat(200.0e9, 0.3, 7800.0);
  FemModel model(perturbed_block(), mat, 1.0, StressState::ThreeDimensional,
                 IntegrationOptions());
  const int dim = model.dim();
  REQUIRE(dim == 3);
  const Matrix x0 = model.mesh().element_coordinates(0);
  const int nd = dim * model.mesh().nodes_per_elem();
  // A 90-degree turn about z (and a translation).
  Matrix rot = Matrix::Zero(dim, dim);
  rot(0, 1) = -1.0;
  rot(1, 0) = 1.0;
  rot(2, 2) = 1.0;
  const Matrix disp = (rot - Matrix::Identity(dim, dim)) * x0 +
                      Vector::Constant(dim, 0.3).replicate(1, x0.cols());
  Vector ue(nd);
  for (int a = 0; a < model.mesh().nodes_per_elem(); ++a) {
    for (int k = 0; k < dim; ++k) ue(dim * a + k) = disp(k, a);
  }
  for (const HyperelasticModel law :
       {HyperelasticModel::SaintVenantKirchhoff, HyperelasticModel::NeoHookean}) {
    const TotalLagrangianElement rigid =
        total_lagrangian_element(model, 0, ue, law, nullptr, 1.0, false);
    INFO(law_name(law));
    // Scale: E times the cell's measure (below 1 here), times round-off -
    // a rotation by 90 degrees leaves E = 0 only to round-off in H^T H.
    REQUIRE(rigid.internal_force.cwiseAbs().maxCoeff() <= 1.0e-12 * 200.0e9);
    REQUIRE(std::abs(rigid.energy) <= 1.0e-13 * 200.0e9);
    // At zero displacement the tangent is the linear stiffness.
    const TotalLagrangianElement zero =
        total_lagrangian_element(model, 0, Vector::Zero(nd), law, nullptr, 1.0, true);
    const Matrix k =
        model.element().stiffness(x0, model.constitutive(), 1.0, model.integration());
    REQUIRE((zero.tangent - k).cwiseAbs().maxCoeff() <= 1.0e-10 * k.cwiseAbs().maxCoeff());
  }
}
