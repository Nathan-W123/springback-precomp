/// \file test_solid.cpp
/// \brief Three-dimensional verification: the Hex8 element, structured hex
///        meshes, the 3-D patch test, tractions, reactions, diagnostics,
///        modal analysis, beam-theory validation and the 3-D configuration
///        path.
///
/// The 2-D suite is the regression net for the shared code; this file checks
/// that everything the dimension-generic core does for a solid mesh is exact
/// where it has to be (patch test, rigid modes, mass, tractions, gradients)
/// and lands where beam theory says it should where it is a model comparison.
#include "TestSupport.hpp"

#include "sparlab/core/Exceptions.hpp"
#include "sparlab/elements/Hex8.hpp"
#include "sparlab/elements/Quadrature.hpp"
#include "sparlab/fem/ModalAnalysis.hpp"
#include "sparlab/fem/ModelDiagnostics.hpp"
#include "sparlab/fem/StressRecovery.hpp"
#include "sparlab/io/Config.hpp"
#include "sparlab/io/CsvWriter.hpp"
#include "sparlab/io/ResultWriter.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <Eigen/Dense>

#include <cstdio>
#include <fstream>
#include <sstream>

using namespace sparlab;
using namespace sparlab::testing;
using Catch::Approx;

namespace {

Matrix3 gradient_field() {
  Matrix3 g;
  g << 3.0e-4, 1.0e-4, -0.5e-4,
       1.0e-4, -2.0e-4, 0.7e-4,
       -0.5e-4, 0.7e-4, 1.5e-4;
  return g;
}

/// Engineering-strain Voigt vector of the symmetric gradient `g`.
Vector6 voigt_strain(const Matrix3& g) {
  Vector6 e;
  e << g(0, 0), g(1, 1), g(2, 2), g(0, 1) + g(1, 0), g(1, 2) + g(2, 1),
      g(2, 0) + g(0, 2);
  return e;
}

}  // namespace

// ---------------------------------------------------------------------------
// Hex8 element
// ---------------------------------------------------------------------------

TEST_CASE("Hex8 shape functions satisfy the partition of unity and nodal delta",
          "[solid][shape]") {
  const Scalar corners[8][3] = {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
                                {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1}};
  for (int a = 0; a < 8; ++a) {
    const auto n = hex8_shape_functions(corners[a][0], corners[a][1], corners[a][2]);
    for (int b = 0; b < 8; ++b) {
      REQUIRE(n(b) == Approx(a == b ? 1.0 : 0.0).margin(1.0e-15));
    }
  }
  const Scalar samples[4][3] = {{0, 0, 0}, {0.3, -0.7, 0.2}, {-0.9, 0.2, -0.5}, {0.5, 0.5, 0.9}};
  for (const auto& s : samples) {
    const auto n = hex8_shape_functions(s[0], s[1], s[2]);
    REQUIRE(n.sum() == Approx(1.0).margin(1.0e-15));
    const auto g = hex8_shape_gradients_natural(s[0], s[1], s[2]);
    for (int j = 0; j < 3; ++j) REQUIRE(g.col(j).sum() == Approx(0.0).margin(1.0e-15));

    // Gradients match central differences.
    const Scalar h = 1.0e-6;
    for (int a = 0; a < 8; ++a) {
      const Scalar dxi = (hex8_shape_functions(s[0] + h, s[1], s[2])(a) -
                          hex8_shape_functions(s[0] - h, s[1], s[2])(a)) / (2.0 * h);
      const Scalar deta = (hex8_shape_functions(s[0], s[1] + h, s[2])(a) -
                           hex8_shape_functions(s[0], s[1] - h, s[2])(a)) / (2.0 * h);
      const Scalar dzeta = (hex8_shape_functions(s[0], s[1], s[2] + h)(a) -
                            hex8_shape_functions(s[0], s[1], s[2] - h)(a)) / (2.0 * h);
      REQUIRE(g(a, 0) == Approx(dxi).margin(1.0e-9));
      REQUIRE(g(a, 1) == Approx(deta).margin(1.0e-9));
      REQUIRE(g(a, 2) == Approx(dzeta).margin(1.0e-9));
    }
  }
}

TEST_CASE("Gauss-Legendre cube rules are tensor products with total weight 8",
          "[solid][quadrature]") {
  for (int n = 1; n <= 4; ++n) {
    const auto& rule = gauss_legendre_cube(n);
    REQUIRE(static_cast<int>(rule.size()) == n * n * n);
    Scalar total = 0.0;
    for (const auto& p : rule) total += p.weight;
    REQUIRE(total == Approx(8.0).margin(1.0e-14));
  }
  // 2 x 2 x 2 integrates xi^2 eta^2 zeta^2 exactly: (2/3)^3.
  Scalar value = 0.0;
  for (const auto& p : gauss_legendre_cube(2)) {
    value += p.weight * p.xi * p.xi * p.eta * p.eta * p.zeta * p.zeta;
  }
  REQUIRE(value == Approx(8.0 / 27.0).margin(1.0e-14));
  REQUIRE_THROWS_AS(gauss_legendre_cube(6), ConfigError);
}

TEST_CASE("Hex8 stiffness is symmetric and annihilates exactly six rigid modes",
          "[solid][element][verification]") {
  Hex8Element element;
  const Matrix6 d = default_material().three_dimensional_matrix();
  const IntegrationOptions opts;

  for (const Matrix& coords : {unit_box_coords(2.0, 0.5, 0.7), distorted_hex_coords()}) {
    const Matrix ke = element.stiffness(coords, d, 1.0, opts);
    REQUIRE(ke.rows() == 24);
    REQUIRE(ke.cols() == 24);
    const Scalar scale = ke.cwiseAbs().maxCoeff();
    REQUIRE((ke - ke.transpose()).cwiseAbs().maxCoeff() / scale ==
            Approx(0.0).margin(1.0e-15));

    // Rigid modes about the element centroid.
    Vector3 centroid = Vector3::Zero();
    for (int a = 0; a < 8; ++a) centroid += coords.col(a);
    centroid /= 8.0;
    Matrix modes = Matrix::Zero(24, 6);
    for (int a = 0; a < 8; ++a) {
      const Vector3 r = Vector3(coords.col(a)) - centroid;
      for (int k = 0; k < 3; ++k) modes(3 * a + k, k) = 1.0;
      for (int j = 0; j < 3; ++j) {
        Vector3 omega = Vector3::Zero();
        omega(j) = 1.0;
        const Vector3 u = omega.cross(r);
        for (int k = 0; k < 3; ++k) modes(3 * a + k, 3 + j) = u(k);
      }
    }
    for (int m = 0; m < 6; ++m) {
      const Vector residual = ke * modes.col(m);
      REQUIRE(residual.cwiseAbs().maxCoeff() / (scale * modes.col(m).norm()) ==
              Approx(0.0).margin(1.0e-14));
    }

    // The null space is exactly six-dimensional.
    Eigen::SelfAdjointEigenSolver<Matrix> es(ke);
    const Vector eig = es.eigenvalues();
    int near_zero = 0;
    for (Eigen::Index i = 0; i < eig.size(); ++i) {
      if (std::abs(eig(i)) < 1.0e-9 * scale) ++near_zero;
    }
    REQUIRE(near_zero == 6);
    REQUIRE(eig(eig.size() - 1) > 0.0);
  }
}

TEST_CASE("Hex8 stiffness of a box is exact with the 2 x 2 x 2 rule",
          "[solid][element][verification]") {
  Hex8Element element;
  const Matrix6 d = default_material().three_dimensional_matrix();
  const Matrix coords = unit_box_coords(0.3, 0.2, 0.15);
  IntegrationOptions two;
  IntegrationOptions three;
  three.stiffness_points = 3;
  IntegrationOptions four;
  four.stiffness_points = 4;
  const Matrix k2 = element.stiffness(coords, d, 1.0, two);
  const Matrix k3 = element.stiffness(coords, d, 1.0, three);
  const Matrix k4 = element.stiffness(coords, d, 1.0, four);
  const Scalar scale = k2.cwiseAbs().maxCoeff();
  REQUIRE((k3 - k2).cwiseAbs().maxCoeff() / scale == Approx(0.0).margin(1.0e-13));
  REQUIRE((k4 - k2).cwiseAbs().maxCoeff() / scale == Approx(0.0).margin(1.0e-13));

  // Scales linearly with the modulus.
  const Matrix6 d3 = IsotropicMaterial(3.0 * 70.0e9, 0.3, 1.0).three_dimensional_matrix();
  const Matrix k_scaled = element.stiffness(coords, d3, 1.0, two);
  REQUIRE((k_scaled - 3.0 * k2).cwiseAbs().maxCoeff() / k_scaled.cwiseAbs().maxCoeff() ==
          Approx(0.0).margin(1.0e-14));
}

TEST_CASE("Hex8 strain operator reproduces a linear field on a distorted cell",
          "[solid][element][verification]") {
  Hex8Element element;
  const Matrix coords = distorted_hex_coords();
  const Vector3 a(1.0e-3, -2.0e-3, 0.5e-3);
  const Matrix3 g = gradient_field();

  Vector ue(24);
  for (int n = 0; n < 8; ++n) {
    const Vector3 u = a + g * Vector3(coords.col(n));
    for (int k = 0; k < 3; ++k) ue(3 * n + k) = u(k);
  }
  const Vector6 expected = voigt_strain(g);

  const Scalar pts[5][3] = {{0, 0, 0}, {-0.9, 0.7, 0.1}, {0.6, -0.3, -0.8}, {1, 1, 1}, {-1, -1, -1}};
  for (const auto& p : pts) {
    NaturalPoint np;
    np.xi = p[0];
    np.eta = p[1];
    np.zeta = p[2];
    const StrainOperator op = element.strain_operator(coords, np);
    REQUIRE(op.b.rows() == 6);
    REQUIRE(op.b.cols() == 24);
    REQUIRE(op.detJ > 0.0);
    const Vector strain = op.b * ue;
    REQUIRE((strain - expected).cwiseAbs().maxCoeff() == Approx(0.0).margin(1.0e-17));
  }
}

TEST_CASE("Hex8 Jacobian integrates to the cell volume", "[solid][element]") {
  // A box and a sheared box (parallelepiped) have closed-form volumes.
  REQUIRE(hex8_volume(unit_box_coords(2.0, 0.5, 0.25)) == Approx(0.25));
  Matrix sheared = unit_box_coords(1.0, 1.0, 1.0);
  for (int a = 0; a < 8; ++a) sheared(0, a) += 0.3 * sheared(2, a);  // shear x by z
  REQUIRE(hex8_volume(sheared) == Approx(1.0));

  // The volume equals the 3 x 3 x 3 integral of detJ on a distorted cell.
  Hex8Element element;
  const Matrix coords = distorted_hex_coords();
  Scalar volume = 0.0;
  for (const auto& gp : gauss_legendre_cube(3)) {
    NaturalPoint p;
    p.xi = gp.xi;
    p.eta = gp.eta;
    p.zeta = gp.zeta;
    volume += gp.weight * element.strain_operator(coords, p).detJ;
  }
  REQUIRE(volume == Approx(hex8_volume(coords)).epsilon(1.0e-12));
  REQUIRE(volume > 0.9);  // sanity: close to the unit cube it was built from

  // Face areas: the top face of a box is a x b.
  const Matrix box = unit_box_coords(2.0, 0.5, 0.7);
  Matrix top(3, 4);
  for (int a = 0; a < 4; ++a) top.col(a) = box.col(4 + a);
  REQUIRE(hex8_face_area(top) == Approx(1.0));
}

TEST_CASE("inverted, degenerate and mis-sized hexahedra are rejected",
          "[solid][element][diagnostics]") {
  Hex8Element element;
  const Matrix6 d = default_material().three_dimensional_matrix();
  const IntegrationOptions opts;

  // Swapping the top and bottom faces inverts the cell.
  Matrix inverted = unit_box_coords();
  for (int a = 0; a < 4; ++a) inverted.col(a).swap(inverted.col(a + 4));
  REQUIRE_THROWS_AS(element.stiffness(inverted, d, 1.0, opts), MeshError);

  Matrix collapsed = unit_box_coords();
  collapsed.row(2).setZero();
  REQUIRE_THROWS_AS(element.stiffness(collapsed, d, 1.0, opts), MeshError);

  const Matrix wrong_size = unit_square_coords();
  NaturalPoint p;
  REQUIRE_THROWS_AS(element.strain_operator(wrong_size, p), MeshError);

  // A solid has no thickness: anything but 1 is refused.
  REQUIRE_THROWS_AS(element.stiffness(unit_box_coords(), d, 0.01, opts), ConfigError);
  REQUIRE_THROWS_AS(element.consistent_mass(unit_box_coords(), 1.0, 2.0, opts),
                    ConfigError);

  // A plane constitutive matrix cannot drive a solid element.
  REQUIRE_THROWS_AS(
      element.stiffness(unit_box_coords(), default_material().plane_stress_matrix(), 1.0,
                        opts),
      ModelError);
}

TEST_CASE("Hex8 consistent mass is symmetric, positive definite and conserves mass",
          "[solid][element][modal][verification]") {
  Hex8Element element;
  const Scalar density = 2700.0;
  const IntegrationOptions opts;
  for (const Matrix& coords : {unit_box_coords(1.3, 0.7, 0.4), distorted_hex_coords()}) {
    const Matrix me = element.consistent_mass(coords, density, 1.0, opts);
    REQUIRE((me - me.transpose()).cwiseAbs().maxCoeff() / me.cwiseAbs().maxCoeff() ==
            Approx(0.0).margin(1.0e-15));
    Eigen::SelfAdjointEigenSolver<Matrix> es(me);
    REQUIRE(es.eigenvalues().minCoeff() > 0.0);

    const Scalar mass = density * hex8_volume(coords);
    // Sum of all entries = 3 * mass (one factor per translation).
    REQUIRE(me.sum() == Approx(3.0 * mass).epsilon(1.0e-12));
    for (int k = 0; k < 3; ++k) {
      Vector t = Vector::Zero(24);
      for (int a = 0; a < 8; ++a) t(3 * a + k) = 1.0;
      REQUIRE(t.dot(me * t) == Approx(mass).epsilon(1.0e-12));
    }
    // Row-sum lumping conserves the total.
    Scalar lumped = 0.0;
    for (int i = 0; i < 24; ++i) {
      REQUIRE(me.row(i).sum() > 0.0);
      lumped += me.row(i).sum();
    }
    REQUIRE(lumped == Approx(me.sum()).epsilon(1.0e-14));
  }
}

TEST_CASE("Hex8 face traction produces the exact consistent nodal load",
          "[solid][element][loads][verification]") {
  Hex8Element element;
  const Matrix coords = unit_box_coords(2.0, 0.5, 0.7);
  const Vector3 traction(0.0, -1.0e5, 3.0e4);  // Pa
  IntegrationOptions opts;

  // Local face 1 is the top face (z = 0.7), nodes 4..7, area 2 x 0.5 = 1 m^2.
  const Vector fe = element.boundary_traction(coords, 1, traction, 1.0, opts);
  REQUIRE(fe.size() == 24);
  Vector3 resultant = Vector3::Zero();
  for (int a = 0; a < 8; ++a) {
    for (int k = 0; k < 3; ++k) resultant(k) += fe(3 * a + k);
  }
  REQUIRE(resultant.isApprox(traction * 1.0, 1.0e-12));
  // Constant traction on a rectangle splits evenly over its four nodes, and
  // the bottom nodes carry nothing.
  for (int a = 4; a < 8; ++a) {
    REQUIRE(fe(3 * a + 1) == Approx(0.25 * traction.y() * 1.0));
    REQUIRE(fe(3 * a + 2) == Approx(0.25 * traction.z() * 1.0));
  }
  for (int a = 0; a < 4; ++a) {
    for (int k = 0; k < 3; ++k) REQUIRE(fe(3 * a + k) == Approx(0.0));
  }
  // Every face of the box carries its own area.
  const Scalar areas[6] = {1.0, 1.0, 1.4, 0.35, 1.4, 0.35};
  for (int f = 0; f < 6; ++f) {
    const Vector ff = element.boundary_traction(coords, f, Vector3(1.0, 0.0, 0.0), 1.0, opts);
    Scalar sum = 0.0;
    for (int a = 0; a < 8; ++a) sum += ff(3 * a);
    REQUIRE(sum == Approx(areas[f]));
  }
  REQUIRE_THROWS_AS(element.boundary_traction(coords, 6, traction, 1.0, opts), MeshError);
  REQUIRE_THROWS_AS(element.boundary_traction(coords, 1, traction, 0.5, opts), ConfigError);
}

TEST_CASE("element factory returns a Hex8 with the right shape", "[solid][element]") {
  const auto element = make_element(ElementType::Hex8);
  REQUIRE(element->type() == ElementType::Hex8);
  REQUIRE(element->dim() == 3);
  REQUIRE(element->num_nodes() == 8);
  REQUIRE(element->num_dofs() == 24);
  REQUIRE(element->num_voigt() == 6);
  REQUIRE(element->num_faces() == 6);
  REQUIRE(element->face_nodes(1) == std::vector<int>({4, 5, 6, 7}));
  REQUIRE(element->stress_evaluation_points(IntegrationOptions()).size() == 8);
  REQUIRE(nodes_per_element(ElementType::Hex8) == 8);
  REQUIRE(element_dimension(ElementType::Hex8) == 3);
  REQUIRE(to_string(ElementType::Hex8) == "Hex8");
}

// ---------------------------------------------------------------------------
// Material and stress invariants
// ---------------------------------------------------------------------------

TEST_CASE("three-dimensional constitutive matrix matches the Lame form",
          "[solid][material]") {
  const Scalar e = 70.0e9;
  const Scalar nu = 0.33;
  const IsotropicMaterial m(e, nu, 2700.0);
  const Matrix6 d = m.three_dimensional_matrix();
  const Scalar lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
  const Scalar g = e / (2.0 * (1.0 + nu));
  REQUIRE(d(0, 0) == Approx(lambda + 2.0 * g));
  REQUIRE(d(0, 1) == Approx(lambda));
  REQUIRE(d(2, 0) == Approx(lambda));
  REQUIRE(d(3, 3) == Approx(g));
  REQUIRE(d(5, 5) == Approx(g));
  REQUIRE(d(0, 3) == 0.0);
  REQUIRE(m.lame_lambda() == Approx(lambda));
  Eigen::SelfAdjointEigenSolver<Matrix6> es(d);
  REQUIRE(es.eigenvalues().minCoeff() > 0.0);
  REQUIRE((m.constitutive(StressState::ThreeDimensional) - d).cwiseAbs().maxCoeff() ==
          Approx(0.0));

  // Uniaxial stress recovers E: sigma_xx / eps_xx with lateral strains -nu eps.
  Vector6 strain = Vector6::Zero();
  strain << 1.0e-3, -nu * 1.0e-3, -nu * 1.0e-3, 0.0, 0.0, 0.0;
  const Vector6 stress = d * strain;
  REQUIRE(stress(0) == Approx(e * 1.0e-3).epsilon(1.0e-12));
  REQUIRE(stress(1) == Approx(0.0).margin(1.0e-3));
  REQUIRE(stress(2) == Approx(0.0).margin(1.0e-3));
}

TEST_CASE("3-D von Mises and principal stresses are consistent invariants",
          "[solid][stress]") {
  Vector s(6);
  s << 120.0e6, -40.0e6, 30.0e6, 25.0e6, -15.0e6, 10.0e6;
  const Vector3 principal = principal_stresses_3d(s);
  REQUIRE(principal(0) >= principal(1));
  REQUIRE(principal(1) >= principal(2));
  // Trace invariant.
  REQUIRE(principal.sum() == Approx(s(0) + s(1) + s(2)));
  // von Mises from principal stresses equals the Voigt formula.
  const Scalar vm_principal =
      std::sqrt(0.5 * (std::pow(principal(0) - principal(1), 2) +
                       std::pow(principal(1) - principal(2), 2) +
                       std::pow(principal(2) - principal(0), 2)));
  REQUIRE(von_mises(s, StressState::ThreeDimensional, 0.3) ==
          Approx(vm_principal).epsilon(1.0e-12));
  // Hydrostatic stress has zero von Mises.
  Vector hydro(6);
  hydro << 50.0e6, 50.0e6, 50.0e6, 0.0, 0.0, 0.0;
  REQUIRE(von_mises(hydro, StressState::ThreeDimensional, 0.3) == Approx(0.0).margin(1.0));
  // Pure shear gives sqrt(3) tau.
  Vector shear = Vector::Zero(6);
  shear(4) = 20.0e6;
  REQUIRE(von_mises(shear, StressState::ThreeDimensional, 0.3) ==
          Approx(std::sqrt(3.0) * 20.0e6));
  REQUIRE_THROWS_AS(principal_stresses_3d(Vector::Zero(3)), ModelError);
  REQUIRE_THROWS_AS(von_mises(Vector::Zero(4), StressState::ThreeDimensional, 0.3),
                    ModelError);
}

// ---------------------------------------------------------------------------
// Meshes
// ---------------------------------------------------------------------------

TEST_CASE("structured hex generator produces the documented numbering",
          "[solid][mesh]") {
  StructuredMeshSpec spec;
  spec.nx = 3;
  spec.ny = 2;
  spec.nz = 2;
  spec.lx = 3.0;
  spec.ly = 2.0;
  spec.lz = 1.0;
  const Mesh mesh = make_structured_hex_mesh(spec);

  REQUIRE(mesh.dim() == 3);
  REQUIRE(mesh.element_type() == ElementType::Hex8);
  REQUIRE(mesh.num_nodes() == 4 * 3 * 3);
  REQUIRE(mesh.num_elements() == 12);
  REQUIRE(mesh.nodes_per_elem() == 8);
  REQUIRE(mesh.node(0).isApprox(Vector3(0.0, 0.0, 0.0)));
  REQUIRE(mesh.node(3).isApprox(Vector3(3.0, 0.0, 0.0)));
  REQUIRE(mesh.node(4).isApprox(Vector3(0.0, 1.0, 0.0)));
  REQUIRE(mesh.node(12).isApprox(Vector3(0.0, 0.0, 0.5)));
  REQUIRE(mesh.node(35).isApprox(Vector3(3.0, 2.0, 1.0)));

  // element(0): bottom face 0,1,5,4 then top face 12,13,17,16
  const Index* e0 = mesh.element_nodes(0);
  const Index expected[8] = {0, 1, 5, 4, 12, 13, 17, 16};
  for (int a = 0; a < 8; ++a) REQUIRE(e0[a] == expected[a]);

  for (Index e = 0; e < mesh.num_elements(); ++e) {
    REQUIRE(mesh.element_measure(e) == Approx(0.5));
  }
  REQUIRE(mesh.element_centroid(0).isApprox(Vector3(0.5, 0.5, 0.25)));

  const BoundingBox bb = mesh.bounding_box();
  REQUIRE(bb.lower.isApprox(Vector3::Zero()));
  REQUIRE(bb.upper.isApprox(Vector3(3.0, 2.0, 1.0)));

  const StructuredGridInfo& info = *mesh.structured_info();
  REQUIRE(info.uniform);
  REQUIRE(info.nz == 2);
  REQUIRE(structured_node_index(info, 2, 1, 1) == 18);
  REQUIRE(structured_element_index(info, 2, 1, 1) == 11);
  REQUIRE_THROWS_AS(structured_node_index(info, 0, 0, 3), MeshError);

  // Every boundary face is a quad, there are 2(nx ny + ny nz + nz nx) of them
  // and their areas add up to the surface of the box.
  const std::vector<Mesh::BoundaryFace> faces = mesh.boundary_faces();
  REQUIRE(faces.size() == 2 * (3 * 2 + 2 * 2 + 2 * 3));
  Scalar area = 0.0;
  for (const Mesh::BoundaryFace& f : faces) {
    REQUIRE(f.nodes.size() == 4);
    REQUIRE(f.local_face >= 0);
    REQUIRE(f.local_face < 6);
    area += mesh.face_measure(f);
  }
  REQUIRE(area == Approx(2.0 * (3.0 * 2.0 + 2.0 * 1.0 + 1.0 * 3.0)));

  // Bad specifications.
  StructuredMeshSpec bad = spec;
  bad.nz = 0;
  REQUIRE_THROWS_AS(make_structured_hex_mesh(bad), ConfigError);
  bad = spec;
  bad.lz = -1.0;
  REQUIRE_THROWS_AS(make_structured_hex_mesh(bad), ConfigError);
}

TEST_CASE("hex mesh validation catches inverted cells and dimension mismatches",
          "[solid][mesh][diagnostics]") {
  const Matrix coords = unit_box_coords();
  // Top and bottom faces swapped: negative Jacobian.
  REQUIRE_THROWS_AS(Mesh(coords, {4, 5, 6, 7, 0, 1, 2, 3}, ElementType::Hex8).validate(),
                    MeshError);
  // 2-D coordinates cannot carry hexahedra and vice versa.
  REQUIRE_THROWS_AS(Mesh(unit_square_coords(), {0, 1, 2, 3, 0, 1, 2, 3}, ElementType::Hex8),
                    MeshError);
  REQUIRE_THROWS_AS(Mesh(coords, {0, 1, 2, 3}, ElementType::Quad4), MeshError);
  // A valid single cell passes.
  REQUIRE_NOTHROW(Mesh(coords, {0, 1, 2, 3, 4, 5, 6, 7}, ElementType::Hex8).validate());
}

TEST_CASE("perturbed hex mesh keeps its boundary and volume", "[solid][mesh]") {
  StructuredMeshSpec spec;
  spec.nx = 4;
  spec.ny = 3;
  spec.nz = 3;
  spec.lx = 2.0;
  spec.ly = 1.0;
  spec.lz = 0.75;
  const Mesh uniform = make_structured_hex_mesh(spec);
  const Mesh perturbed = make_perturbed_hex_mesh(spec, 0.25, 42u);
  REQUIRE_FALSE(perturbed.structured_info()->uniform);
  REQUIRE_NOTHROW(perturbed.validate());
  Scalar v_uniform = 0.0;
  Scalar v_perturbed = 0.0;
  for (Index e = 0; e < uniform.num_elements(); ++e) {
    v_uniform += uniform.element_measure(e);
    v_perturbed += perturbed.element_measure(e);
  }
  REQUIRE(v_uniform == Approx(2.0 * 1.0 * 0.75));
  REQUIRE(v_perturbed == Approx(v_uniform));
  REQUIRE(perturbed.bounding_box().upper.isApprox(uniform.bounding_box().upper));
  // Interior nodes did move.
  REQUIRE((perturbed.coordinates() - uniform.coordinates()).cwiseAbs().maxCoeff() > 0.01);
  REQUIRE(make_perturbed_hex_mesh(spec, 0.25, 42u).coordinates().isApprox(
      perturbed.coordinates()));
  REQUIRE_THROWS_AS(make_perturbed_hex_mesh(spec, 0.5), ConfigError);
}

TEST_CASE("3-D selectors", "[solid][mesh][selector]") {
  StructuredMeshSpec spec;
  spec.nx = 4;
  spec.ny = 4;
  spec.nz = 4;
  spec.lx = 1.0;
  spec.ly = 1.0;
  spec.lz = 1.0;
  const Mesh mesh = make_structured_hex_mesh(spec);

  SECTION("a degenerate box selects a plane of nodes") {
    SelectorGroup group;
    Selector box;
    box.kind = SelectorKind::Box;
    box.zmax = 0.0;
    group.members.push_back(box);
    REQUIRE(group.select_nodes(mesh).size() == 25);
  }
  SECTION("a sphere selects by 3-D distance") {
    SelectorGroup group;
    Selector sphere;
    sphere.kind = SelectorKind::Sphere;
    sphere.center = Vector3(0.5, 0.5, 0.5);
    sphere.radius = 0.3;
    group.members.push_back(sphere);
    const std::vector<Index> elements = group.select_elements(mesh);
    REQUIRE(elements.size() == 8);  // the eight cells around the centre
    for (Index e : elements) {
      REQUIRE((mesh.element_centroid(e) - sphere.center).norm() <= 0.3 + 1.0e-9);
    }
  }
  SECTION("a circle about the y axis is a cylinder through the block") {
    SelectorGroup group;
    Selector circle;
    circle.kind = SelectorKind::Circle;
    circle.center = Vector3(0.5, 0.0, 0.5);
    circle.radius = 0.3;
    circle.axis = 1;
    group.members.push_back(circle);
    const std::vector<Index> elements = group.select_elements(mesh);
    REQUIRE(elements.size() == 4 * 4);  // four cells per layer, all four layers
    for (Index e : elements) {
      const Vector3 c = mesh.element_centroid(e);
      REQUIRE(std::hypot(c.x() - 0.5, c.z() - 0.5) <= 0.3 + 1.0e-9);
    }
  }
}

// ---------------------------------------------------------------------------
// Model-level verification
// ---------------------------------------------------------------------------

TEST_CASE("constant-strain patch test passes on distorted hex meshes",
          "[solid][patch][verification]") {
  for (Scalar perturbation : {0.0, 0.15, 0.30}) {
    StructuredMeshSpec spec;
    spec.nx = 3;
    spec.ny = 3;
    spec.nz = 3;
    spec.lx = 1.5;
    spec.ly = 1.0;
    spec.lz = 1.2;
    const Vector3 offset(1.0e-4, -2.0e-4, 0.5e-4);
    const Matrix3 g = gradient_field();
    const Vector6 exact_strain = voigt_strain(g);

    const IsotropicMaterial material(200.0e9, 0.3, 7850.0, "patch3d");
    FemModel model(make_perturbed_hex_mesh(spec, perturbation, 7u), material, 1.0,
                   StressState::ThreeDimensional, IntegrationOptions());

    std::vector<char> on_boundary(static_cast<std::size_t>(model.mesh().num_nodes()), 0);
    for (const Mesh::BoundaryFace& f : model.mesh().boundary_faces()) {
      for (Index n : f.nodes) on_boundary[static_cast<std::size_t>(n)] = 1;
    }
    std::vector<Index> boundary_nodes;
    for (Index n = 0; n < model.mesh().num_nodes(); ++n) {
      if (on_boundary[static_cast<std::size_t>(n)]) boundary_nodes.push_back(n);
    }
    // 3^3 cells have exactly 2^3 = 8 interior nodes.
    REQUIRE(boundary_nodes.size() == static_cast<std::size_t>(model.mesh().num_nodes() - 8));

    DisplacementConstraint bc;
    Selector sel;
    sel.kind = SelectorKind::NodeIds;
    sel.ids = boundary_nodes;
    bc.region.members.push_back(sel);
    bc.fix_x = bc.fix_y = bc.fix_z = true;
    model.constraints().push_back(bc);
    LoadCaseSpec load;
    load.name = "patch";
    load.prescribed_displacement_only = true;
    model.load_case_specs().push_back(load);
    model.finalize();
    for (Index n : boundary_nodes) {
      const Vector3 u = offset + g * model.mesh().node(n);
      for (int k = 0; k < 3; ++k) model.dofs().prescribe(n, k, u(k));
    }

    Assembler assembler(model);
    StaticAnalysisOptions options;
    options.linear.residual_tolerance = 1.0e-9;
    StaticAnalysis analysis(model, assembler, options);
    const Vector u = analysis.solve_all().front().displacement;

    Scalar u_error = 0.0;
    Scalar u_scale = 0.0;
    for (Index n = 0; n < model.mesh().num_nodes(); ++n) {
      const Vector3 expected = offset + g * model.mesh().node(n);
      for (int k = 0; k < 3; ++k) {
        u_error = std::max(u_error, std::abs(u(n * 3 + k) - expected(k)));
      }
      u_scale = std::max(u_scale, expected.cwiseAbs().maxCoeff());
    }
    INFO("perturbation " << perturbation);
    REQUIRE(u_error / u_scale == Approx(0.0).margin(1.0e-11));

    const StressField field = recover_stresses(model, assembler, u);
    const Vector6 exact_stress = material.three_dimensional_matrix() * exact_strain;
    for (Index e = 0; e < model.mesh().num_elements(); ++e) {
      const Vector6 strain_e = field.element_strain.col(e);
      const Vector6 stress_e = field.element_stress.col(e);
      REQUIRE((strain_e - exact_strain).cwiseAbs().maxCoeff() /
                  exact_strain.cwiseAbs().maxCoeff() == Approx(0.0).margin(1.0e-11));
      REQUIRE((stress_e - exact_stress).cwiseAbs().maxCoeff() /
                  exact_stress.cwiseAbs().maxCoeff() == Approx(0.0).margin(1.0e-11));
    }
    // Strain energy of a uniform field: 1/2 eps^T sigma * volume.
    REQUIRE(field.element_strain_energy.sum() ==
            Approx(0.5 * exact_strain.dot(exact_stress) * model.domain_volume())
                .epsilon(1.0e-9));
    REQUIRE(field.element_principal_mid.size() == model.mesh().num_elements());
  }
}

TEST_CASE("3-D face tractions give a mesh-independent resultant and reactions balance",
          "[solid][loads][reactions][verification]") {
  const Vector3 traction(2.0e5, -1.0e5, 5.0e4);
  for (Index n : {2, 4}) {
    StructuredMeshSpec spec;
    spec.nx = 2 * n;
    spec.ny = n;
    spec.nz = n;
    spec.lx = 1.0;
    spec.ly = 0.5;
    spec.lz = 0.25;
    FemModel model(make_structured_hex_mesh(spec), default_material(), 1.0,
                   StressState::ThreeDimensional, IntegrationOptions());
    DisplacementConstraint bc;
    Selector box;
    box.kind = SelectorKind::Box;
    box.xmax = 0.0;
    bc.region.members.push_back(box);
    bc.fix_x = bc.fix_y = bc.fix_z = true;
    model.constraints().push_back(bc);
    LoadCaseSpec load;
    load.name = "top";
    TractionLoadSpec t;
    Selector top;
    top.kind = SelectorKind::Box;
    top.zmin = spec.lz;
    t.region.members.push_back(top);
    t.traction = traction;
    load.tractions.push_back(t);
    model.load_case_specs().push_back(load);
    model.finalize();

    const Vector& f = model.load_vectors().front();
    Vector3 resultant = Vector3::Zero();
    for (Index i = 0; i < model.mesh().num_nodes(); ++i) {
      for (int k = 0; k < 3; ++k) resultant(k) += f(i * 3 + k);
    }
    const Scalar area = spec.lx * spec.ly;
    REQUIRE(resultant.isApprox(traction * area, 1.0e-12));

    Assembler assembler(model);
    StaticAnalysis analysis(model, assembler, StaticAnalysisOptions());
    const StaticSolution sol = analysis.solve_all().front();
    REQUIRE(sol.equilibrium.applied_force.isApprox(traction * area, 1.0e-12));
    REQUIRE(sol.equilibrium.relative_force_error == Approx(0.0).margin(1.0e-10));
    REQUIRE(sol.equilibrium.relative_moment_error == Approx(0.0).margin(1.0e-9));
    // The applied moment is a genuine 3-vector here.
    REQUIRE(sol.equilibrium.applied_moment.norm() > 0.0);
    REQUIRE(sol.compliance == Approx(2.0 * sol.strain_energy).epsilon(1.0e-10));
    // Reactions vanish at free DOFs.
    for (Index d : model.dofs().free_dofs()) REQUIRE(sol.reactions(d) == 0.0);
  }
}

TEST_CASE("a 2-D model rejects out-of-plane input", "[solid][diagnostics]") {
  FemModel model = make_small_plate();
  DisplacementConstraint bc;
  Selector box;
  box.kind = SelectorKind::Box;
  box.xmax = 0.0;
  bc.region.members.push_back(box);
  bc.fix_z = true;
  REQUIRE_THROWS_AS(apply_constraints(model.mesh(), {bc}, model.dofs()), ConfigError);

  LoadCaseSpec load;
  PointLoadSpec p;
  Selector all;
  p.region.members.push_back(all);
  p.force = Vector3(0.0, 0.0, 1.0);
  load.point_loads.push_back(p);
  REQUIRE_THROWS_AS(assemble_load_vector(model.mesh(), model.element(), load,
                                         model.thickness(), model.integration()),
                    ConfigError);

  // A plane idealisation on a solid mesh and a thickness on a solid are refused.
  StructuredMeshSpec spec;
  spec.nx = spec.ny = spec.nz = 2;
  REQUIRE_THROWS_AS(FemModel(make_structured_hex_mesh(spec), default_material(), 1.0,
                             StressState::PlaneStress, IntegrationOptions()),
                    ConfigError);
  REQUIRE_THROWS_AS(FemModel(make_structured_hex_mesh(spec), default_material(), 0.01,
                             StressState::ThreeDimensional, IntegrationOptions()),
                    ConfigError);
  REQUIRE_THROWS_AS(FemModel(make_structured_quad_mesh(spec), default_material(), 1.0,
                             StressState::ThreeDimensional, IntegrationOptions()),
                    ConfigError);
}

TEST_CASE("3-D rigid-body diagnostics count six modes", "[solid][diagnostics][verification]") {
  StructuredMeshSpec spec;
  spec.nx = 4;
  spec.ny = 2;
  spec.nz = 2;
  const auto build_with = [&](const std::vector<DisplacementConstraint>& bcs) {
    FemModel model(make_structured_hex_mesh(spec), default_material(), 1.0,
                   StressState::ThreeDimensional, IntegrationOptions());
    model.constraints() = bcs;
    LoadCaseSpec load;
    PointLoadSpec p;
    Selector nearest;
    nearest.kind = SelectorKind::NearestNode;
    nearest.point = Vector3(1.0, 0.0, 0.0);
    p.region.members.push_back(nearest);
    p.force = Vector3(0.0, -10.0, 0.0);
    load.point_loads.push_back(p);
    model.load_case_specs().push_back(load);
    model.finalize();
    return model;
  };

  SECTION("no constraints: six free modes and a singular assembled matrix") {
    FemModel model = build_with({});
    const ModelDiagnostics diag = diagnose_model(model);
    REQUIRE_FALSE(diag.well_posed());
    REQUIRE(diag.components.front().rigid_null_dimension == 6);
    Assembler assembler(model);
    const SparseMatrix k = assembler.assemble_stiffness();
    const Matrix modes = rigid_body_modes_3d(model.mesh());
    const Scalar scale = Matrix(k).cwiseAbs().maxCoeff();
    for (int m = 0; m < 6; ++m) {
      const Vector residual = k * modes.col(m);
      REQUIRE(residual.cwiseAbs().maxCoeff() / (scale * modes.col(m).norm()) ==
              Approx(0.0).margin(1.0e-13));
    }
  }
  SECTION("one pinned node leaves the three rotations") {
    DisplacementConstraint bc;
    Selector nearest;
    nearest.kind = SelectorKind::NearestNode;
    nearest.point = Vector3::Zero();
    bc.region.members.push_back(nearest);
    bc.fix_x = bc.fix_y = bc.fix_z = true;
    REQUIRE(diagnose_model(build_with({bc})).components.front().rigid_null_dimension == 3);
  }
  SECTION("a face fixed in z only leaves two translations and one rotation") {
    DisplacementConstraint bc;
    Selector box;
    box.kind = SelectorKind::Box;
    box.zmax = 0.0;
    bc.region.members.push_back(box);
    bc.fix_z = true;
    REQUIRE(diagnose_model(build_with({bc})).components.front().rigid_null_dimension == 3);
  }
  SECTION("a fully clamped face is well posed") {
    DisplacementConstraint bc;
    Selector box;
    box.kind = SelectorKind::Box;
    box.xmax = 0.0;
    bc.region.members.push_back(box);
    bc.fix_x = bc.fix_y = bc.fix_z = true;
    const ModelDiagnostics diag = diagnose_model(build_with({bc}));
    REQUIRE(diag.well_posed());
    REQUIRE(diag.components.front().rigid_null_dimension == 0);
  }
  SECTION("three non-collinear point supports suffice") {
    DisplacementConstraint pin;
    Selector n1;
    n1.kind = SelectorKind::NearestNode;
    n1.point = Vector3(0.0, 0.0, 0.0);
    pin.region.members.push_back(n1);
    pin.fix_x = pin.fix_y = pin.fix_z = true;
    DisplacementConstraint slider;
    Selector n2;
    n2.kind = SelectorKind::NearestNode;
    n2.point = Vector3(1.0, 0.0, 0.0);
    slider.region.members.push_back(n2);
    slider.fix_y = slider.fix_z = true;
    DisplacementConstraint roller;
    Selector n3;
    n3.kind = SelectorKind::NearestNode;
    n3.point = Vector3(0.0, 1.0, 0.0);
    roller.region.members.push_back(n3);
    roller.fix_z = true;
    REQUIRE(diagnose_model(build_with({pin, slider, roller})).well_posed());
  }
}

TEST_CASE("3-D mass matrix conserves mass and the cantilever frequencies follow beam theory",
          "[solid][modal][validation]") {
  SolidCantileverCase c;
  c.poisson = 0.0;
  FemModel model = make_cantilever_3d(c, 60, 4, 4);
  Assembler assembler(model);
  const Scalar expected_mass = c.density * c.length * c.height * c.width;
  REQUIRE(assembler.total_mass() == Approx(expected_mass).epsilon(1.0e-12));
  const SparseMatrix m = assembler.assemble_mass(MassType::Consistent);
  REQUIRE(m.sum() == Approx(3.0 * expected_mass).epsilon(1.0e-11));

  ModalAnalysisOptions options;
  options.num_modes = 6;
  const ModalResult modal = solve_modal(model, assembler, options);
  REQUIRE(modal.converged);
  REQUIRE(modal.total_mass == Approx(expected_mass).epsilon(1.0e-11));
  REQUIRE(modal.warnings.empty());

  // The first mode bends about the weak (z) axis: the section is h x b with
  // b < h, so the lowest frequency uses the width as the "height".
  const Scalar f1_weak = cantilever_bending_frequency(c.youngs, c.density, c.length,
                                                      c.width, c.height, 1);
  const Scalar f1_strong = cantilever_bending_frequency(c.youngs, c.density, c.length,
                                                        c.height, c.width, 1);
  INFO("f1 " << modal.frequencies_hz(0) << " Hz vs weak-axis theory " << f1_weak
             << " Hz; f " << modal.frequencies_hz(1) << " Hz vs strong-axis theory "
             << f1_strong << " Hz");
  REQUIRE(std::abs(modal.frequencies_hz(0) - f1_weak) / f1_weak < 0.04);
  // The strong-axis bending mode is present among the first few frequencies.
  Scalar best = std::numeric_limits<Scalar>::max();
  for (Eigen::Index i = 0; i < modal.frequencies_hz.size(); ++i) {
    best = std::min(best, std::abs(modal.frequencies_hz(i) - f1_strong) / f1_strong);
  }
  REQUIRE(best < 0.04);
  for (Eigen::Index i = 1; i < modal.frequencies_hz.size(); ++i) {
    REQUIRE(modal.frequencies_hz(i) >= modal.frequencies_hz(i - 1));
  }
}

TEST_CASE("solid cantilever tip deflection converges towards Timoshenko theory",
          "[solid][beam][validation][convergence]") {
  SolidCantileverCase c;
  c.poisson = 0.0;
  const Scalar ti = c.timoshenko_tip();
  const Scalar eb = c.euler_bernoulli_tip();

  std::vector<Scalar> tips;
  std::vector<Scalar> errors;
  for (const auto& m : std::vector<std::array<Index, 3>>{{20, 2, 1}, {40, 4, 2}, {80, 8, 2}}) {
    FemModel model = make_cantilever_3d(c, m[0], m[1], m[2]);
    Assembler assembler(model);
    StaticAnalysis analysis(model, assembler, StaticAnalysisOptions());
    const StaticSolution sol = analysis.solve_all().front();
    REQUIRE(sol.equilibrium.relative_force_error == Approx(0.0).margin(1.0e-9));
    tips.push_back(tip_deflection_3d(model, sol.displacement));
    errors.push_back(std::abs(tips.back() - ti) / std::abs(ti));
    INFO("mesh " << m[0] << "x" << m[1] << "x" << m[2] << ": tip " << tips.back()
                 << " m, Timoshenko " << ti << " m, Euler-Bernoulli " << eb << " m");
  }
  // The error against Timoshenko shrinks under refinement and ends within a
  // few percent; the model is stiffer than the beam on every mesh (a
  // displacement-based element under-predicts deflection).
  REQUIRE(errors[1] < errors[0]);
  REQUIRE(errors[2] < errors[1]);
  REQUIRE(errors.back() < 0.03);
  REQUIRE(std::abs(tips.back()) < std::abs(ti));
  REQUIRE(std::abs(tips.back()) > 0.95 * std::abs(eb));
}

// ---------------------------------------------------------------------------
// Configuration and writers
// ---------------------------------------------------------------------------

namespace {

std::string hex_deck() {
  return R"({
    "name": "solid_case",
    "mesh": { "type": "structured_hex", "nx": 8, "ny": 2, "nz": 2,
              "lx": 0.4, "ly": 0.1, "lz": 0.1 },
    "material": { "youngs_modulus": 70e9, "poisson_ratio": 0.3, "density": 2700 },
    "boundary_conditions": [
      { "name": "root", "fix": ["x", "y", "z"], "region": { "box": { "xmax": 0.0 } } }
    ],
    "load_cases": [
      { "name": "tip", "point_loads": [
          { "force": [0.0, -1000.0, 200.0], "region": { "box": { "xmin": 0.4 } } } ] },
      { "name": "top", "tractions": [
          { "traction": [0.0, 0.0, -2.0e5], "region": { "box": { "zmin": 0.1 } } } ] }
    ],
    "modal": { "enabled": true, "num_modes": 3 }
  })";
}

}  // namespace

TEST_CASE("a structured_hex deck parses into a 3-D model", "[solid][io][config]") {
  const Configuration config =
      parse_configuration(json::parse(hex_deck(), "hex"), "hex", /*strict=*/true);
  REQUIRE(config.dim() == 3);
  REQUIRE(config.mesh_kind == MeshKind::StructuredHex);
  REQUIRE(config.stress_state == StressState::ThreeDimensional);
  REQUIRE(config.mesh_spec.nz == 2);
  REQUIRE(config.mesh_spec.lz == Approx(0.1));
  REQUIRE(config.constraints.front().fix_z);
  REQUIRE(config.load_cases[0].point_loads.front().force.z() == Approx(200.0));
  REQUIRE(config.load_cases[1].tractions.front().traction.z() == Approx(-2.0e5));

  FemModel model = build_model(config);
  REQUIRE(model.dim() == 3);
  REQUIRE(model.dofs().num_dofs() == 3 * 9 * 3 * 3);
  REQUIRE(model.load_vectors().size() == 2);

  Assembler assembler(model);
  StaticAnalysis analysis(model, assembler, config.analysis);
  const std::vector<StaticSolution> sols = analysis.solve_all();
  REQUIRE(sols.size() == 2);
  REQUIRE(sols[1].equilibrium.applied_force.z() == Approx(-2.0e5 * 0.4 * 0.1));
  REQUIRE(sols[0].equilibrium.relative_force_error == Approx(0.0).margin(1.0e-10));
}

TEST_CASE("dimension mismatches in a deck are rejected with a reason",
          "[solid][io][config][diagnostics]") {
  const auto parse = [](const std::string& body) {
    return parse_configuration(json::parse(body, "bad"), "bad");
  };
  const std::string head2d = R"({"mesh": {"nx": 2, "ny": 2, "lx": 1, "ly": 1},
    "material": {"youngs_modulus": 1e9, "poisson_ratio": 0.3},)";
  const std::string head3d = R"({"mesh": {"type": "structured_hex", "nx": 2, "ny": 2, "nz": 2,
    "lx": 1, "ly": 1, "lz": 1}, "material": {"youngs_modulus": 1e9, "poisson_ratio": 0.3},)";
  const std::string bc2 = R"("boundary_conditions": [{"fix": ["x","y"], "region": {"box": {"xmax": 0}}}],)";
  const std::string bc3 = R"("boundary_conditions": [{"fix": ["x","y","z"], "region": {"box": {"xmax": 0}}}],)";

  // Three-entry force on a 2-D mesh.
  REQUIRE_THROWS_AS(parse(head2d + bc2 + R"("load_cases": [{"point_loads": [
      {"force": [1,0,0], "region": {"all": true}}]}]})"), ConfigError);
  // Two-entry force on a 3-D mesh.
  REQUIRE_THROWS_AS(parse(head3d + bc3 + R"("load_cases": [{"point_loads": [
      {"force": [1,0], "region": {"all": true}}]}]})"), ConfigError);
  // z bounds on a 2-D mesh.
  REQUIRE_THROWS_AS(parse(head2d + R"("boundary_conditions": [{"fix": ["x","y"],
      "region": {"box": {"zmax": 0}}}], "load_cases": [{"point_loads": [
      {"force": [1,0], "region": {"all": true}}]}]})"), ConfigError);
  // A plane stress state on a hex mesh.
  REQUIRE_THROWS_AS(parse(head3d + R"("model": {"stress_state": "plane_stress"},)" + bc3 +
      R"("load_cases": [{"point_loads": [{"force": [1,0,0], "region": {"all": true}}]}]})"),
      ConfigError);
  // A thickness on a hex mesh.
  REQUIRE_THROWS_AS(parse(head3d + R"("model": {"thickness": 0.01},)" + bc3 +
      R"("load_cases": [{"point_loads": [{"force": [1,0,0], "region": {"all": true}}]}]})"),
      ConfigError);
  // nz on a quad mesh.
  REQUIRE_THROWS_AS(parse(R"({"mesh": {"nx": 2, "ny": 2, "nz": 2, "lx": 1, "ly": 1},
    "material": {"youngs_modulus": 1e9, "poisson_ratio": 0.3},)" + bc2 +
      R"("load_cases": [{"point_loads": [{"force": [1,0], "region": {"all": true}}]}]})"),
      ConfigError);
  // A circle about x on a 2-D mesh.
  REQUIRE_THROWS_AS(parse(head2d + R"("boundary_conditions": [{"fix": ["x","y"],
      "region": {"circle": {"center": [0,0], "radius": 1, "axis": "x"}}}],
      "load_cases": [{"point_loads": [{"force": [1,0], "region": {"all": true}}]}]})"),
      ConfigError);
  // The 3-D form of the same deck is accepted.
  REQUIRE_NOTHROW(parse(head3d + bc3 + R"("load_cases": [{"point_loads": [
      {"force": [1,0,0], "region": {"circle": {"center": [0,0,0], "radius": 1, "axis": "x"}}}]}]})"));
}

TEST_CASE("3-D results are written with z columns and six stress components",
          "[solid][io][writers]") {
  const Configuration config =
      parse_configuration(json::parse(hex_deck(), "hex"), "hex", /*strict=*/true);
  FemModel model = build_model(config);
  Assembler assembler(model);
  StaticAnalysis analysis(model, assembler, config.analysis);
  const std::vector<StaticSolution> sols = analysis.solve_all();
  const StressField field = recover_stresses(model, assembler, sols[0].displacement);

  const std::string dir = "results/_test_tmp/solid";
  ResultWriter writer(dir, config);
  writer.write_mesh(model);
  writer.write_displacement(model.mesh(), "tip", sols[0].displacement);
  writer.write_stress(model.mesh(), "tip", field);
  writer.write_reactions(model.mesh(), model.dofs(), "tip", sols[0].reactions);
  writer.write_static_vtk(model.mesh(), "tip", sols[0].displacement, field, nullptr, nullptr);

  const auto first_line = [](const std::string& path) {
    std::ifstream in(path);
    std::string line;
    std::getline(in, line);
    return line;
  };
  REQUIRE(first_line(path_join(dir, "displacement_tip.csv")) ==
          "node,x[m],y[m],z[m],ux[m],uy[m],uz[m],umag[m]");
  REQUIRE(first_line(path_join(dir, "reactions_tip.csv")) ==
          "node,x[m],y[m],z[m],rx[N],ry[N],rz[N],rmag[N]");
  const std::string stress_header = first_line(path_join(dir, "stress_tip.csv"));
  REQUIRE(stress_header.find("cz[m]") != std::string::npos);
  REQUIRE(stress_header.find("szz[Pa]") != std::string::npos);
  REQUIRE(stress_header.find("gzx[-]") != std::string::npos);
  REQUIRE(stress_header.find("principal_mid[Pa]") != std::string::npos);
  REQUIRE(stress_header.find("volume[m3]") != std::string::npos);

  std::ifstream vtk(path_join(dir, "fields_tip.vtk"));
  std::stringstream buffer;
  buffer << vtk.rdbuf();
  const std::string content = buffer.str();
  REQUIRE(content.find("CELL_TYPES") != std::string::npos);
  REQUIRE(content.find("\n12\n") != std::string::npos);
  REQUIRE(content.find("SCALARS sigma_zx double 1") != std::string::npos);

  const json::Value mesh_doc = json::parse_file(path_join(dir, "mesh.json"));
  REQUIRE(mesh_doc.find("dim")->number_value() == Approx(3.0));
  REQUIRE(mesh_doc.find("nodes_m")->array_items().front().array_items().size() == 3);
  const json::Value& forces =
      *mesh_doc.find("load_cases")->array_items().front().find("nodal_forces");
  REQUIRE(forces.array_items().front().find("fz_N") != nullptr);

  TimingLedger timings;
  const ModelDiagnostics diag = diagnose_model(model);
  const json::Value summary =
      make_static_summary(config, model, diag, sols, {field}, nullptr, timings);
  REQUIRE(summary.find("mesh")->find("dim")->number_value() == Approx(3.0));
  REQUIRE(summary.find("mesh")->find("bounding_box_xmin_ymin_zmin_xmax_ymax_zmax_m")
              ->array_items().size() == 6);
  const json::Value& eq = *summary.find("load_cases")->array_items().front().find("equilibrium");
  REQUIRE(eq.find("applied_moment_Nm")->is_array());
  REQUIRE(eq.find("applied_force_N")->array_items().size() == 3);
}
