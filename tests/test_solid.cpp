/// \file test_solid.cpp
/// \brief Three-dimensional verification: the Hex8 element and its
///        quadrature, the 3-D constitutive matrix, structured hex meshes,
///        3-D selectors and the 3-D configuration path.
///
/// The element checks are exact where they have to be: rigid modes, the
/// stiffness of a box, a linear field on a distorted cell, volume and mass.
#include "TestSupport.hpp"

#include "sparlab/core/Exceptions.hpp"
#include "sparlab/elements/Hex8.hpp"
#include "sparlab/elements/Quadrature.hpp"
#include "sparlab/io/Config.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <Eigen/Dense>


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

  const Matrix wrong_size = unit_box_coords().leftCols(4);
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
          "[solid][element][mass][verification]") {
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

TEST_CASE("element factory returns a Hex8 with the right shape", "[solid][element]") {
  const auto element = make_element(ElementType::Hex8);
  REQUIRE(element->type() == ElementType::Hex8);
  REQUIRE(element->dim() == 3);
  REQUIRE(element->num_nodes() == 8);
  REQUIRE(element->num_dofs() == 24);
  REQUIRE(element->num_voigt() == 6);
  REQUIRE(element->integration_rule(IntegrationOptions()).size() == 8);
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
  // node(i, j, k) = k (nx + 1)(ny + 1) + j (nx + 1) + i, elem(i, j, k) =
  // k nx ny + j nx + i.
  REQUIRE(mesh.node(18).isApprox(Vector3(2.0, 1.0, 0.5)));
  REQUIRE(mesh.element_centroid(11).isApprox(Vector3(2.5, 1.5, 0.75)));

  // Every boundary face is a quad, there are 2(nx ny + ny nz + nz nx) of them
  // and their areas add up to the surface of the box.
  const std::vector<Mesh::BoundaryFace> faces = mesh.boundary_faces();
  REQUIRE(faces.size() == 2 * (3 * 2 + 2 * 2 + 2 * 3));
  Scalar area = 0.0;
  for (const Mesh::BoundaryFace& f : faces) {
    REQUIRE(f.nodes.size() == 4);
    REQUIRE(f.local_face >= 0);
    REQUIRE(f.local_face < 6);
    // A planar quadrilateral: half the norm of its diagonals' cross product.
    const Vector3 d1 = mesh.node(f.nodes[2]) - mesh.node(f.nodes[0]);
    const Vector3 d2 = mesh.node(f.nodes[3]) - mesh.node(f.nodes[1]);
    area += 0.5 * d1.cross(d2).norm();
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
  // 2-D coordinates cannot carry hexahedra, nor can a connectivity of the
  // wrong stride.
  REQUIRE_THROWS_AS(Mesh(Matrix::Zero(2, 4), {0, 1, 2, 3, 0, 1, 2, 3}, ElementType::Hex8),
                    MeshError);
  REQUIRE_THROWS_AS(Mesh(coords, {0, 1, 2, 3}, ElementType::Hex8), MeshError);
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
    ]
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
  REQUIRE(config.describe_mesh() == "structured_hex 8 x 2 x 2");

  FemModel model = build_model(config);
  REQUIRE(model.dim() == 3);
  REQUIRE(model.dofs().num_dofs() == 3 * 9 * 3 * 3);
  REQUIRE(model.dofs().num_constrained() == 3 * 3 * 3);
  // The one empty load case of a model the forming analysis drives.
  REQUIRE(model.load_case_specs().size() == 1);
  REQUIRE(model.load_case_specs().front().name == "forming");
}

TEST_CASE("dimension mismatches in a deck are rejected with a reason",
          "[solid][io][config][diagnostics]") {
  const auto parse = [](const std::string& body) {
    return parse_configuration(json::parse(body, "bad"), "bad");
  };
  const std::string head3d = R"({"mesh": {"type": "structured_hex", "nx": 2, "ny": 2, "nz": 2,
    "lx": 1, "ly": 1, "lz": 1}, "material": {"youngs_modulus": 1e9, "poisson_ratio": 0.3},)";
  const std::string bc3 = R"("boundary_conditions": [{"fix": ["x","y","z"], "region": {"box": {"xmax": 0}}}])";

  // A two-entry value on a 3-D mesh.
  REQUIRE_THROWS_AS(parse(head3d + R"("boundary_conditions": [{"fix": ["x"], "value": [0, 0],
      "region": {"box": {"xmax": 0}}}]})"), ConfigError);
  // A plane stress state on a hex mesh.
  REQUIRE_THROWS_AS(parse(head3d + R"("model": {"stress_state": "plane_stress"},)" + bc3 + "}"),
                    ConfigError);
  // A thickness on a hex mesh.
  REQUIRE_THROWS_AS(parse(head3d + R"("model": {"thickness": 0.01},)" + bc3 + "}"),
                    ConfigError);
  // A mesh type SparLab does not have.
  REQUIRE_THROWS_AS(parse(R"({"mesh": {"type": "structured_quad", "nx": 2, "ny": 2, "lx": 1,
    "ly": 1}, "material": {"youngs_modulus": 1e9, "poisson_ratio": 0.3},)" + bc3 + "}"),
      ConfigError);
  // A circle about x is a 3-D region.
  REQUIRE_NOTHROW(parse(head3d + R"("boundary_conditions": [{"fix": ["x","y","z"],
      "region": {"circle": {"center": [0,0,0], "radius": 1, "axis": "x"}}}]})"));
}
