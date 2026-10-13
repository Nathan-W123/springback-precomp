/// \file test_tet10.cpp
/// \brief The ten-node quadratic tetrahedron: shape functions, simplex
///        quadrature, element kernels, the quadratic patch test, the Tet4 ->
///        Tet10 elevation, and Tet10 meshes through every reader and writer.
///
/// What must hold exactly for a straight-sided quadratic tetrahedron is
/// checked exactly: the Kronecker property, the published consistent mass
/// coefficients, the corner-free consistent load of a flat face, and the
/// reproduction of a quadratic displacement field (pure bending) on a
/// distorted mesh, which the constant-strain Tet4 cannot pass.
#include "TestSupport.hpp"

#include "sparlab/core/Exceptions.hpp"
#include "sparlab/elements/Quadrature.hpp"
#include "sparlab/elements/Tet10.hpp"
#include "sparlab/elements/Tet4.hpp"
#include "sparlab/fem/StressRecovery.hpp"
#include "sparlab/io/CalculixWriter.hpp"
#include "sparlab/io/Config.hpp"
#include "sparlab/io/CsvWriter.hpp"
#include "sparlab/io/MeshReader.hpp"
#include "sparlab/io/VtkWriter.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <Eigen/Eigenvalues>

#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <sstream>

using namespace sparlab;
using namespace sparlab::testing;
using Catch::Approx;
using Catch::Matchers::ContainsSubstring;

namespace {

const char* kTmp = "results/_test_tmp/tet10";

Scalar factorial(int n) {
  Scalar f = 1.0;
  for (int i = 2; i <= n; ++i) f *= i;
  return f;
}

/// Exact integral of xi^a eta^b zeta^c over the reference tetrahedron.
Scalar tet_monomial(int a, int b, int c) {
  return factorial(a) * factorial(b) * factorial(c) / factorial(a + b + c + 3);
}

/// Exact integral of xi^a eta^b over the reference triangle.
Scalar tri_monomial(int a, int b) {
  return factorial(a) * factorial(b) / factorial(a + b + 2);
}

int count_zero_modes(const Matrix& k, Scalar ratio = 1.0e-10) {
  Eigen::SelfAdjointEigenSolver<Matrix> eig(k);
  const Vector values = eig.eigenvalues();
  const Scalar largest = values.cwiseAbs().maxCoeff();
  int zeros = 0;
  for (Eigen::Index i = 0; i < values.size(); ++i) {
    if (std::abs(values(i)) < ratio * largest) ++zeros;
  }
  return zeros;
}

/// A straight-sided Tet10 on the given corners (edge nodes at midpoints).
Matrix straight_tet10(const Matrix& corners) {
  static const int edges[6][2] = {{0, 1}, {1, 2}, {2, 0}, {0, 3}, {1, 3}, {2, 3}};
  Matrix x(3, 10);
  x.leftCols(4) = corners;
  for (int k = 0; k < 6; ++k) {
    x.col(4 + k) = 0.5 * (corners.col(edges[k][0]) + corners.col(edges[k][1]));
  }
  return x;
}

Matrix skewed_corners() {
  Matrix c(3, 4);
  c << 0.1, 1.2, 0.3, 0.2,
       0.0, 0.2, 1.1, 0.3,
       0.05, 0.1, 0.2, 0.9;
  return c;
}

StructuredMeshSpec box_spec(Index nx, Index ny, Index nz, Scalar lx, Scalar ly, Scalar lz) {
  StructuredMeshSpec spec;
  spec.nx = nx;
  spec.ny = ny;
  spec.nz = nz;
  spec.lx = lx;
  spec.ly = ly;
  spec.lz = lz;
  return spec;
}

/// The pure-bending field about the y axis: u_x = -k x z,
/// u_y = nu k y z, u_z = k/2 (x^2 + nu (z^2 - y^2)). It is an exact
/// elasticity solution without body force, with sigma_xx = -E k z the only
/// non-zero stress.
Vector3 bending_field(const Vector3& x, Scalar kappa, Scalar nu) {
  return Vector3(-kappa * x.x() * x.z(), nu * kappa * x.y() * x.z(),
                 0.5 * kappa * (x.x() * x.x() + nu * (x.z() * x.z() - x.y() * x.y())));
}

std::vector<Index> boundary_node_list(const Mesh& mesh) {
  std::vector<char> on(static_cast<std::size_t>(mesh.num_nodes()), 0);
  for (const Mesh::BoundaryFace& f : mesh.boundary_faces()) {
    for (Index n : f.nodes) on[static_cast<std::size_t>(n)] = 1;
  }
  std::vector<Index> out;
  for (Index n = 0; n < mesh.num_nodes(); ++n) {
    if (on[static_cast<std::size_t>(n)]) out.push_back(n);
  }
  return out;
}

/// Prescribe the bending field on the boundary nodes, solve, and return the
/// displacement and the largest nodal error relative to the field's size.
Scalar bending_patch_error(FemModel& model, Scalar kappa, Vector* u_out) {
  const Scalar nu = model.material().poisson_ratio();
  const std::vector<Index> boundary = boundary_node_list(model.mesh());
  DisplacementConstraint bc;
  Selector sel;
  sel.kind = SelectorKind::NodeIds;
  sel.ids = boundary;
  bc.region.members.push_back(sel);
  bc.fix_x = bc.fix_y = bc.fix_z = true;
  model.constraints().push_back(bc);
  LoadCaseSpec load;
  load.name = "bending";
  load.prescribed_displacement_only = true;
  model.load_case_specs().push_back(load);
  model.finalize();
  for (Index n : boundary) {
    const Vector3 u = bending_field(model.mesh().node(n), kappa, nu);
    for (int k = 0; k < 3; ++k) model.dofs().prescribe(n, k, u(k));
  }
  Assembler assembler(model);
  StaticAnalysisOptions options;
  options.linear.residual_tolerance = 1.0e-9;
  StaticAnalysis analysis(model, assembler, options);
  const Vector u = analysis.solve_all().front().displacement;
  Scalar err = 0.0;
  Scalar scale = 0.0;
  for (Index n = 0; n < model.mesh().num_nodes(); ++n) {
    const Vector3 expected = bending_field(model.mesh().node(n), kappa, nu);
    err = std::max(err, (u.segment<3>(3 * n) - expected).cwiseAbs().maxCoeff());
    scale = std::max(scale, expected.cwiseAbs().maxCoeff());
  }
  if (u_out != nullptr) *u_out = u;
  return err / scale;
}

/// Gmsh MSH 2.2 text of a Tet10 mesh: 10-node tetrahedra (type 11) in Gmsh's
/// node order and the boundary faces of each group as 6-node triangles
/// (type 9).
long long node_tag(Index n) { return 10 * static_cast<long long>(n) + 3; }

std::string write_tet10_msh22(const Mesh& mesh,
                              const std::vector<std::pair<std::string,
                                                          std::function<bool(const Vector3&)>>>& groups,
                              bool flip_first_cell = false) {
  std::ostringstream os;
  os << std::setprecision(17);
  os << "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n";
  os << "$PhysicalNames\n" << groups.size() + 1 << "\n";
  for (std::size_t g = 0; g < groups.size(); ++g) {
    os << "2 " << g + 1 << " \"" << groups[g].first << "\"\n";
  }
  os << "3 " << groups.size() + 1 << " \"solid\"\n$EndPhysicalNames\n";
  os << "$Nodes\n" << mesh.num_nodes() << "\n";
  for (Index n = 0; n < mesh.num_nodes(); ++n) {
    const Vector3 x = mesh.node(n);
    os << node_tag(n) << " " << x.x() << " " << x.y() << " " << x.z() << "\n";
  }
  os << "$EndNodes\n";
  std::vector<std::string> lines;
  long long tag = 1;
  for (const Mesh::BoundaryFace& f : mesh.boundary_faces()) {
    Vector3 c = Vector3::Zero();
    for (int a = 0; a < 3; ++a) c += mesh.node(f.nodes[static_cast<std::size_t>(a)]);
    c /= 3.0;
    for (std::size_t g = 0; g < groups.size(); ++g) {
      if (!groups[g].second(c)) continue;
      std::ostringstream line;
      line << tag++ << " 9 2 " << g + 1 << " " << 100 + g;
      for (Index n : f.nodes) line << " " << node_tag(n);
      lines.push_back(line.str());
      break;
    }
  }
  for (Index e = 0; e < mesh.num_elements(); ++e) {
    std::array<Index, 10> n{};
    const Index* en = mesh.element_nodes(e);
    for (int a = 0; a < 10; ++a) n[static_cast<std::size_t>(a)] = en[a];
    if (flip_first_cell && e == 0) {
      // The mirror image of the cell: corners 1 and 2 exchanged, and with
      // them the edges 0-1 / 2-0 and 1-3 / 2-3.
      std::swap(n[1], n[2]);
      std::swap(n[4], n[6]);
      std::swap(n[8], n[9]);
    }
    std::swap(n[8], n[9]);  // Gmsh order: ..., edge 2-3, edge 1-3
    std::ostringstream line;
    line << tag++ << " 11 2 " << groups.size() + 1 << " 1";
    for (Index node : n) line << " " << node_tag(node);
    lines.push_back(line.str());
  }
  os << "$Elements\n" << lines.size() << "\n";
  for (const std::string& l : lines) os << l << "\n";
  os << "$EndElements\n";
  return os.str();
}

void require_same_mesh(const Mesh& a, const Mesh& b, Scalar tol = 0.0) {
  REQUIRE(a.element_type() == b.element_type());
  REQUIRE(a.num_nodes() == b.num_nodes());
  REQUIRE(a.num_elements() == b.num_elements());
  REQUIRE((a.coordinates() - b.coordinates()).cwiseAbs().maxCoeff() <= tol);
  REQUIRE(a.connectivity() == b.connectivity());
}

std::string read_text(const std::string& path) {
  std::ifstream in(path);
  std::stringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

/// Clamped-root cantilever box with a shear traction on the tip face.
FemModel make_box_cantilever(Mesh mesh, Scalar length, Scalar height, Scalar traction) {
  FemModel model(std::move(mesh), IsotropicMaterial(70.0e9, 0.0, 2700.0, "beam"), 1.0,
                 StressState::ThreeDimensional, IntegrationOptions());
  DisplacementConstraint root;
  Selector box;
  box.kind = SelectorKind::Box;
  box.xmax = 0.0;
  root.region.members.push_back(box);
  root.fix_x = root.fix_y = root.fix_z = true;
  model.constraints().push_back(root);
  LoadCaseSpec load;
  load.name = "tip";
  TractionLoadSpec tip;
  Selector tip_box;
  tip_box.kind = SelectorKind::Box;
  tip_box.xmin = length;
  tip.region.members.push_back(tip_box);
  tip.traction = Vector3(0.0, 0.0, traction);
  load.tractions.push_back(tip);
  model.load_case_specs().push_back(load);
  model.finalize();
  (void)height;
  return model;
}

Scalar mean_tip_deflection(const FemModel& model, const Vector& u, Scalar length) {
  Scalar sum = 0.0;
  int count = 0;
  for (Index n = 0; n < model.mesh().num_nodes(); ++n) {
    if (std::abs(model.mesh().node(n).x() - length) < 1.0e-12) {
      sum += u(3 * n + 2);
      ++count;
    }
  }
  return sum / count;
}

}  // namespace

// ---------------------------------------------------------------------------
// Quadrature and shape functions
// ---------------------------------------------------------------------------

TEST_CASE("simplex quadrature rules integrate monomials to their stated degree",
          "[tet10][quadrature]") {
  const auto tet_error = [](const std::vector<QuadraturePoint3D>& rule, int degree) {
    Scalar worst = 0.0;
    for (int a = 0; a <= degree; ++a) {
      for (int b = 0; a + b <= degree; ++b) {
        for (int c = 0; a + b + c <= degree; ++c) {
          Scalar sum = 0.0;
          for (const auto& p : rule) {
            sum += p.weight * std::pow(p.xi, a) * std::pow(p.eta, b) * std::pow(p.zeta, c);
          }
          worst = std::max(worst, std::abs(sum - tet_monomial(a, b, c)) / tet_monomial(a, b, c));
        }
      }
    }
    return worst;
  };
  REQUIRE(tetrahedron_rule_4().size() == 4);
  REQUIRE(tet_error(tetrahedron_rule_4(), 2) < 1.0e-14);
  REQUIRE(tet_error(tetrahedron_rule_4(), 3) > 1.0e-3);  // degree 2, not 3
  for (int n = 2; n <= 4; ++n) {
    INFO("collapsed tetrahedron rule, n = " << n);
    REQUIRE(collapsed_gauss_tetrahedron(n).size() == static_cast<std::size_t>(n * n * n));
    REQUIRE(tet_error(collapsed_gauss_tetrahedron(n), 2 * n - 3) < 1.0e-13);
  }
  // One point per direction integrates nothing exactly: degree 2n - 3 < 0.
  REQUIRE(tet_error(collapsed_gauss_tetrahedron(1), 0) > 0.1);
  REQUIRE(tet_error(collapsed_gauss_tetrahedron(4), 6) > 1.0e-6);

  for (int n = 1; n <= 4; ++n) {
    INFO("collapsed triangle rule, n = " << n);
    Scalar worst = 0.0;
    for (int a = 0; a <= 2 * n - 2; ++a) {
      for (int b = 0; a + b <= 2 * n - 2; ++b) {
        Scalar sum = 0.0;
        for (const auto& p : collapsed_gauss_triangle(n)) {
          sum += p.weight * std::pow(p.xi, a) * std::pow(p.eta, b);
        }
        worst = std::max(worst, std::abs(sum - tri_monomial(a, b)) / tri_monomial(a, b));
      }
    }
    REQUIRE(worst < 1.0e-13);
  }
  REQUIRE_THROWS_AS(collapsed_gauss_tetrahedron(5), ConfigError);
  REQUIRE_THROWS_AS(collapsed_gauss_triangle(0), ConfigError);
}

TEST_CASE("Tet10 shape functions interpolate the nodes and reproduce quadratics",
          "[tet10][element]") {
  const Scalar nodes[10][3] = {{0, 0, 0},     {1, 0, 0},     {0, 1, 0},   {0, 0, 1},
                               {0.5, 0, 0},   {0.5, 0.5, 0}, {0, 0.5, 0}, {0, 0, 0.5},
                               {0.5, 0, 0.5}, {0, 0.5, 0.5}};
  for (int a = 0; a < 10; ++a) {
    const Eigen::Matrix<Scalar, 10, 1> n =
        tet10_shape_functions(nodes[a][0], nodes[a][1], nodes[a][2]);
    for (int b = 0; b < 10; ++b) REQUIRE(n(b) == Approx(a == b ? 1.0 : 0.0).margin(1.0e-15));
  }
  // At arbitrary points: partition of unity, zero-sum gradients, and exact
  // interpolation of a quadratic (the gradients checked by finite
  // differences).
  const auto quadratic = [](Scalar x, Scalar y, Scalar z) {
    return 0.3 + 1.1 * x - 0.4 * y + 0.7 * z + 2.0 * x * x - 1.3 * x * y + 0.9 * y * z -
           0.6 * z * z + 0.8 * x * z;
  };
  Eigen::Matrix<Scalar, 10, 1> values;
  for (int a = 0; a < 10; ++a) values(a) = quadratic(nodes[a][0], nodes[a][1], nodes[a][2]);
  for (const auto& p : collapsed_gauss_tetrahedron(3)) {
    const Eigen::Matrix<Scalar, 10, 1> n = tet10_shape_functions(p.xi, p.eta, p.zeta);
    const Eigen::Matrix<Scalar, 10, 3> g = tet10_shape_gradients_natural(p.xi, p.eta, p.zeta);
    REQUIRE(n.sum() == Approx(1.0).margin(1.0e-14));
    REQUIRE(g.colwise().sum().cwiseAbs().maxCoeff() < 1.0e-13);
    REQUIRE(n.dot(values) == Approx(quadratic(p.xi, p.eta, p.zeta)).epsilon(1.0e-13));
    const Scalar h = 1.0e-6;
    const Eigen::Matrix<Scalar, 10, 1> fd =
        (tet10_shape_functions(p.xi + h, p.eta, p.zeta) -
         tet10_shape_functions(p.xi - h, p.eta, p.zeta)) / (2.0 * h);
    REQUIRE((fd - g.col(0)).cwiseAbs().maxCoeff() < 1.0e-8);
  }
}

// ---------------------------------------------------------------------------
// Element kernels
// ---------------------------------------------------------------------------

TEST_CASE("Tet10 stiffness is symmetric, has six rigid modes and is exact for "
          "quadratic fields",
          "[tet10][element]") {
  const Tet10Element element;
  const Matrix x = straight_tet10(skewed_corners());
  const IsotropicMaterial material = default_material(0.3);
  const Matrix d = material.three_dimensional_matrix();
  const Matrix k = element.stiffness(x, d, 1.0, IntegrationOptions());
  REQUIRE(k.rows() == 30);
  REQUIRE((k - k.transpose()).cwiseAbs().maxCoeff() <= 1.0e-12 * k.cwiseAbs().maxCoeff());
  REQUIRE(count_zero_modes(k) == 6);
  Eigen::SelfAdjointEigenSolver<Matrix> eig(k);
  REQUIRE(eig.eigenvalues().minCoeff() > -1.0e-9 * eig.eigenvalues().maxCoeff());

  // A straight-sided Tet10 has the corner volume.
  Scalar min_det = 0.0;
  const Scalar volume = tet10_volume(x, &min_det);
  REQUIRE(volume == Approx(tet4_volume(skewed_corners())).epsilon(1.0e-13));
  REQUIRE(min_det == Approx(6.0 * volume).epsilon(1.0e-12));
  REQUIRE(tet10_jacobian_ratio(x) == Approx(1.0).epsilon(1.0e-12));

  // Energy of a quadratic field against an independent evaluation: the
  // strain is linear, eps(x) = eps0 + sum_i x_i eps_i, and its energy
  // density is integrated with the degree-5 collapsed rule over the affine
  // image of the reference tetrahedron.
  const auto field = [](const Vector3& p) {
    return Vector3(1.0e-4 * p.x() * p.y() - 2.0e-4 * p.z() * p.z() + 3.0e-4 * p.y(),
                   -1.5e-4 * p.x() * p.x() + 0.5e-4 * p.y() * p.z(),
                   2.0e-4 * p.x() * p.z() - 1.0e-4 * p.y() * p.y() + 0.4e-4 * p.x());
  };
  const auto strain = [](const Vector3& p) {
    Vector6 e;
    e << 1.0e-4 * p.y(), 0.5e-4 * p.z(), 2.0e-4 * p.x(),
        1.0e-4 * p.x() + 3.0e-4 - 3.0e-4 * p.x(),
        0.5e-4 * p.y() - 2.0e-4 * p.y(),
        -4.0e-4 * p.z() + 2.0e-4 * p.z() + 0.4e-4;
    return e;
  };
  Vector u(30);
  for (int a = 0; a < 10; ++a) u.segment<3>(3 * a) = field(x.col(a));
  const Matrix corners = skewed_corners();
  Matrix3 jac;
  for (int j = 0; j < 3; ++j) jac.col(j) = corners.col(j + 1) - corners.col(0);
  Scalar energy = 0.0;
  for (const auto& p : collapsed_gauss_tetrahedron(4)) {
    const Vector3 point = Vector3(corners.col(0)) + jac * Vector3(p.xi, p.eta, p.zeta);
    const Vector6 e = strain(point);
    energy += 0.5 * p.weight * jac.determinant() * e.dot(d * e);
  }
  REQUIRE(0.5 * u.dot(k * u) == Approx(energy).epsilon(1.0e-11));

  REQUIRE_THROWS_AS(element.stiffness(x, d, 0.5, IntegrationOptions()), ConfigError);
}

TEST_CASE("Tet10 consistent and lumped mass match the published coefficients",
          "[tet10][element][mass]") {
  const Tet10Element element;
  Matrix corners(3, 4);
  corners << 0.0, 1.0, 0.0, 0.0,
             0.0, 0.0, 2.0, 0.0,
             0.0, 0.0, 0.0, 3.0;
  const Matrix x = straight_tet10(corners);
  const Scalar volume = 1.0;
  const Scalar rho = 7850.0;
  const Matrix m = element.consistent_mass(x, rho, 1.0, IntegrationOptions());
  // rho V / 420 times: corner diagonal 6, corner pairs 1, a corner with an
  // edge node on its edges -4 and on the other edges -6, edge-node diagonal
  // 32, edge nodes sharing a corner 16, opposite edges 8.
  const Scalar c = rho * volume / 420.0;
  REQUIRE(m(0, 0) == Approx(6.0 * c));
  REQUIRE(m(0, 3) == Approx(1.0 * c));
  REQUIRE(m(0, 3 * 4) == Approx(-4.0 * c));   // corner 0, edge 0-1
  REQUIRE(m(0, 3 * 5) == Approx(-6.0 * c));   // corner 0, edge 1-2
  REQUIRE(m(3 * 4, 3 * 4) == Approx(32.0 * c));
  REQUIRE(m(3 * 4, 3 * 5) == Approx(16.0 * c));  // edges 0-1 and 1-2
  REQUIRE(m(3 * 4, 3 * 9) == Approx(8.0 * c));   // edges 0-1 and 2-3
  REQUIRE(m(0, 1) == 0.0);
  Scalar mz = 0.0;
  for (int a = 0; a < 10; ++a) {
    for (int b = 0; b < 10; ++b) mz += m(3 * a + 2, 3 * b + 2);
  }
  REQUIRE(mz == Approx(rho * volume));
  Eigen::SelfAdjointEigenSolver<Matrix> eig(m);
  REQUIRE(eig.eigenvalues().minCoeff() > 0.0);
  // The corner rows sum to -1/20 of the mass, so row-sum lumping would be
  // indefinite.
  REQUIRE(m.row(0).sum() == Approx(-rho * volume / 20.0));

  // HRZ lumping in the assembler: 1/36 of the mass on each corner and 4/27 on
  // each edge node, per direction.
  FemModel model(Mesh(x, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, ElementType::Tet10),
                 IsotropicMaterial(200.0e9, 0.3, rho, "steel"), 1.0,
                 StressState::ThreeDimensional, IntegrationOptions());
  DisplacementConstraint fix;
  Selector node;
  node.kind = SelectorKind::NearestNode;
  node.point = Vector3::Zero();
  fix.region.members.push_back(node);
  fix.fix_x = fix.fix_y = fix.fix_z = true;
  model.constraints().push_back(fix);
  LoadCaseSpec load;
  load.name = "none";
  load.prescribed_displacement_only = true;
  model.load_case_specs().push_back(load);
  model.finalize();
  Assembler assembler(model);
  const SparseMatrix lumped = assembler.assemble_mass(MassType::Lumped);
  REQUIRE(lumped.nonZeros() == 30);
  for (int k = 0; k < 3; ++k) {
    REQUIRE(lumped.coeff(k, k) == Approx(rho * volume / 36.0));
    REQUIRE(lumped.coeff(3 * 7 + k, 3 * 7 + k) == Approx(4.0 * rho * volume / 27.0));
  }
  REQUIRE(Vector(lumped * Vector::Ones(30)).sum() == Approx(3.0 * rho * volume));
  REQUIRE(assembler.total_mass() == Approx(rho * volume));
}

TEST_CASE("Tet10 face loads put the resultant on the edge nodes of a flat face",
          "[tet10][element][load]") {
  const Tet10Element element;
  Matrix corners(3, 4);
  corners << 0.0, 1.0, 0.0, 0.0,
             0.0, 0.0, 2.0, 0.0,
             0.0, 0.0, 0.0, 3.0;
  const Matrix x = straight_tet10(corners);
  // Face 0 is the z = 0 face (0, 2, 1) with edge nodes 6, 5, 4.
  const std::vector<int>& face = element.face_nodes(0);
  REQUIRE(face == std::vector<int>({0, 2, 1, 6, 5, 4}));
  const Vector3 q(0.0, 0.0, -5.0e5);
  const Vector f = element.boundary_traction(x, 0, q, 1.0, IntegrationOptions());
  const Scalar area = 0.5 * 1.0 * 2.0;
  for (int node : {0, 1, 2}) REQUIRE(f(3 * node + 2) == Approx(0.0).margin(1.0e-6));
  for (int node : {4, 5, 6}) REQUIRE(f(3 * node + 2) == Approx(q.z() * area / 3.0));
  REQUIRE(f.segment<3>(9).isZero());
  REQUIRE(f.sum() == Approx(q.z() * area));

  // Every face is wound outward, its edge nodes lie on its corner edges, and
  // the 6-node area of a flat face is the triangle's.
  const Vector3 centroid = corners.rowwise().mean();
  for (int lf = 0; lf < 4; ++lf) {
    const std::vector<int>& fn = element.face_nodes(lf);
    const Vector3 a = x.col(fn[0]);
    const Vector3 b = x.col(fn[1]);
    const Vector3 c = x.col(fn[2]);
    REQUIRE((b - a).cross(c - a).dot((a + b + c) / 3.0 - centroid) > 0.0);
    REQUIRE((Vector3(x.col(fn[3])) - 0.5 * (a + b)).norm() < 1.0e-15);
    REQUIRE((Vector3(x.col(fn[4])) - 0.5 * (b + c)).norm() < 1.0e-15);
    REQUIRE((Vector3(x.col(fn[5])) - 0.5 * (c + a)).norm() < 1.0e-15);
    Matrix xf(3, 6);
    for (int k = 0; k < 6; ++k) xf.col(k) = x.col(fn[static_cast<std::size_t>(k)]);
    REQUIRE(tri6_face_area(xf) == Approx(0.5 * (b - a).cross(c - a).norm()).epsilon(1.0e-13));
  }
}

TEST_CASE("curved Tet10 cells: exact volume, Jacobian ratio and folding checks",
          "[tet10][element][mesh]") {
  Matrix reference(3, 4);
  reference << 0.0, 1.0, 0.0, 0.0,
               0.0, 0.0, 1.0, 0.0,
               0.0, 0.0, 0.0, 1.0;
  Matrix x = straight_tet10(reference);
  // Pushing the edge node of 0-1 out of the face y = 0 by delta adds
  // delta / 6 to the volume: det(I + d g^T) = 1 + g . d with
  // int grad N_4 = (grad L_0 + grad L_1) / 6 = (0, -1, -1) / 6.
  const Scalar delta = 0.1;
  x(1, 4) = -delta;
  REQUIRE(tet10_volume(x) == Approx(1.0 / 6.0 + delta / 6.0).epsilon(1.0e-13));
  REQUIRE(tet10_jacobian_ratio(x) < 1.0);
  REQUIRE(tet10_jacobian_ratio(x) > 0.5);
  // The bulged face has more area than the flat triangle.
  const Tet10Element element;
  const std::vector<int>& fn = element.face_nodes(1);  // (0, 1, 3): the y = 0 face
  Matrix xf(3, 6);
  for (int k = 0; k < 6; ++k) xf.col(k) = x.col(fn[static_cast<std::size_t>(k)]);
  REQUIRE(tri6_face_area(xf) > 0.5);
  const Vector3 q(0.0, -1.0e6, 0.0);
  const Vector f = element.boundary_traction(x, 1, q, 1.0, IntegrationOptions());
  REQUIRE(f.sum() == Approx(q.y() * tri6_face_area(xf)).epsilon(1.0e-12));

  // Pulled inwards far enough, the quadratic map folds: det J = 1 - 3.6 L_1.
  Matrix folded = straight_tet10(reference);
  folded(1, 4) = 0.9;
  Scalar min_det = 0.0;
  REQUIRE(tet10_volume(folded, &min_det) > 0.0);
  REQUIRE(min_det < 0.0);
  REQUIRE_THROWS_WITH(element.stiffness(folded, default_material().three_dimensional_matrix(),
                                        1.0, IntegrationOptions()),
                      ContainsSubstring("folds"));
  const Mesh bad(folded, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, ElementType::Tet10);
  REQUIRE_THROWS_WITH(bad.validate(), ContainsSubstring("Mesh.HighOrderOptimize"));
  Matrix inverted = straight_tet10(reference);
  inverted.col(1).swap(inverted.col(2));
  const Mesh flipped(inverted, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, ElementType::Tet10);
  REQUIRE_THROWS_WITH(flipped.validate(), ContainsSubstring("corner-tetrahedron volume"));
}

// ---------------------------------------------------------------------------
// Patch tests and bending accuracy
// ---------------------------------------------------------------------------

TEST_CASE("Tet10 passes the pure-bending (quadratic) patch test on distorted meshes; "
          "Tet4 cannot",
          "[tet10][patch][verification]") {
  const Scalar kappa = 2.0e-3;
  for (Scalar perturbation : {0.0, 0.15, 0.3}) {
    INFO("perturbation " << perturbation);
    const StructuredMeshSpec spec = box_spec(3, 3, 3, 1.5, 1.0, 1.2);
    const IsotropicMaterial material(200.0e9, 0.3, 7850.0, "patch");
    FemModel model(make_perturbed_tet10_mesh(spec, perturbation, 7u), material, 1.0,
                   StressState::ThreeDimensional, IntegrationOptions());
    REQUIRE(model.mesh().element_type() == ElementType::Tet10);
    Vector u;
    REQUIRE(bending_patch_error(model, kappa, &u) == Approx(0.0).margin(1.0e-10));
    // The stress is sigma_xx = -E kappa z, linear, so each element's average
    // is its value at the centroid; the other components vanish.
    Assembler assembler(model);
    const StressField field = recover_stresses(model, assembler, u);
    const Scalar scale = material.youngs_modulus() * kappa * 1.2;
    for (Index e = 0; e < model.mesh().num_elements(); ++e) {
      const Vector3 c = model.mesh().element_coordinates(e).leftCols(4).rowwise().mean();
      Vector6 exact = Vector6::Zero();
      exact(0) = -material.youngs_modulus() * kappa * c.z();
      REQUIRE((Vector6(field.element_stress.col(e)) - exact).cwiseAbs().maxCoeff() <=
              1.0e-9 * scale);
    }
  }
  // The same field on the linear mesh is not reproduced.
  FemModel linear(make_perturbed_tet_mesh(box_spec(3, 3, 3, 1.5, 1.0, 1.2), 0.15, 7u),
                  IsotropicMaterial(200.0e9, 0.3, 7850.0, "patch"), 1.0,
                  StressState::ThreeDimensional, IntegrationOptions());
  REQUIRE(bending_patch_error(linear, kappa, nullptr) > 1.0e-3);
}

TEST_CASE("Tet10 cantilevers are accurate in bending where Tet4 locks",
          "[tet10][verification]") {
  // A 1 m x 0.1 m x 0.1 m clamped beam under a tip shear of 1 kN, meshed with
  // 10 x 1 x 1 cells of six tetrahedra. Euler-Bernoulli plus Timoshenko
  // shear: w = P L^3 / (3 E I) + P L / (kappa G A).
  const Scalar length = 1.0;
  const Scalar h = 0.1;
  const Scalar load = -1.0e3;
  const Scalar e = 70.0e9;
  const Scalar inertia = h * h * h * h / 12.0;
  const Scalar area = h * h;
  const Scalar shear = e / 2.0;  // nu = 0
  const Scalar theory =
      load * length * length * length / (3.0 * e * inertia) + load * length / (5.0 / 6.0 * shear * area);
  const StructuredMeshSpec spec = box_spec(10, 1, 1, length, h, h);
  Scalar tip[2] = {0.0, 0.0};
  for (int order : {1, 2}) {
    FemModel model = make_box_cantilever(
        order == 1 ? make_structured_tet_mesh(spec) : make_structured_tet10_mesh(spec), length,
        h, load / area);
    Assembler assembler(model);
    StaticAnalysis analysis(model, assembler, StaticAnalysisOptions());
    const StaticSolution sol = analysis.solve_all().front();
    REQUIRE(sol.equilibrium.relative_force_error < 1.0e-10);
    REQUIRE(sol.equilibrium.applied_force.z() == Approx(load).epsilon(1.0e-12));
    tip[order - 1] = mean_tip_deflection(model, sol.displacement, length);
  }
  INFO("theory " << theory << " m, Tet4 " << tip[0] << " m, Tet10 " << tip[1] << " m");
  // Measured: the Tet4 mesh gives 24 % of the beam-theory deflection, the
  // Tet10 mesh of the same cells 99.8 %.
  REQUIRE(std::abs(tip[1] / theory - 1.0) < 0.03);
  REQUIRE(tip[0] / theory < 0.5);
}

// ---------------------------------------------------------------------------
// Meshes: elevation, boundary faces, sub-meshes
// ---------------------------------------------------------------------------

TEST_CASE("elevating a Tet4 mesh adds shared edge nodes and carries named sets",
          "[tet10][mesh]") {
  const StructuredMeshSpec spec = box_spec(3, 2, 2, 0.6, 0.4, 0.4);
  Mesh linear = make_structured_tet_mesh(spec);
  // Groups: a face (x = 0), a line (x = 0, y = 0), a point, and a block of
  // cells that is both an element set and a node set.
  std::vector<Index> face, line, block_nodes, block_cells;
  for (Index n = 0; n < linear.num_nodes(); ++n) {
    const Vector3 p = linear.node(n);
    if (p.x() == 0.0) face.push_back(n);
    if (p.x() == 0.0 && p.y() == 0.0) line.push_back(n);
  }
  for (Index e = 0; e < linear.num_elements(); ++e) {
    if (linear.element_centroid(e).x() > 0.4) {
      block_cells.push_back(e);
      const Index* en = linear.element_nodes(e);
      block_nodes.insert(block_nodes.end(), en, en + 4);
    }
  }
  linear.set_node_set("face", face);
  linear.set_node_set("line", line);
  linear.set_node_set("point", {line.front()});
  linear.set_node_set("block", block_nodes);
  linear.set_element_set("block", block_cells);

  const Mesh quadratic = elevate_to_tet10(linear);
  REQUIRE(quadratic.element_type() == ElementType::Tet10);
  REQUIRE(quadratic.num_elements() == linear.num_elements());
  // Every edge of a Kuhn-split box is a grid edge, a face diagonal or a
  // cell's main diagonal, whose midpoints are the grid's edge, face and cell
  // centres: the elevated mesh has the nodes of the half-cell grid,
  // (2 nx + 1)(2 ny + 1)(2 nz + 1).
  REQUIRE(quadratic.num_nodes() == (2 * 3 + 1) * (2 * 2 + 1) * (2 * 2 + 1));
  REQUIRE(quadratic.coordinates().leftCols(linear.num_nodes()) == linear.coordinates());
  for (Index e = 0; e < linear.num_elements(); ++e) {
    REQUIRE(quadratic.element_measure(e) == Approx(linear.element_measure(e)).epsilon(1.0e-13));
    for (int a = 0; a < 4; ++a) REQUIRE(quadratic.element_nodes(e)[a] == linear.element_nodes(e)[a]);
  }
  REQUIRE(quadratic.mean_element_size() == Approx(linear.mean_element_size()));
  REQUIRE(elevate_to_tet10(linear).connectivity() == quadratic.connectivity());

  const auto expected = [&](const std::function<bool(const Vector3&)>& on) {
    std::vector<Index> out;
    for (Index n = 0; n < quadratic.num_nodes(); ++n) {
      if (on(quadratic.node(n))) out.push_back(n);
    }
    return out;
  };
  REQUIRE(quadratic.node_sets().at("face") ==
          expected([](const Vector3& p) { return std::abs(p.x()) < 1.0e-12; }));
  REQUIRE(quadratic.node_sets().at("line") ==
          expected([](const Vector3& p) {
            return std::abs(p.x()) < 1.0e-12 && std::abs(p.y()) < 1.0e-12;
          }));
  REQUIRE(quadratic.node_sets().at("point").size() == 1);
  std::vector<Index> cell_nodes;
  for (Index e : block_cells) {
    const Index* en = quadratic.element_nodes(e);
    cell_nodes.insert(cell_nodes.end(), en, en + 10);
  }
  std::sort(cell_nodes.begin(), cell_nodes.end());
  cell_nodes.erase(std::unique(cell_nodes.begin(), cell_nodes.end()), cell_nodes.end());
  REQUIRE(quadratic.node_sets().at("block") == cell_nodes);
  REQUIRE(quadratic.element_sets().at("block") == block_cells);

  // Boundary faces: the Tet4's, each with its three edge nodes.
  const std::vector<Mesh::BoundaryFace> faces = quadratic.boundary_faces();
  REQUIRE(faces.size() == linear.boundary_faces().size());
  Scalar area = 0.0;
  for (const Mesh::BoundaryFace& f : faces) {
    REQUIRE(f.nodes.size() == 6);
    area += quadratic.face_measure(f);
  }
  REQUIRE(area == Approx(2.0 * (0.6 * 0.4 + 0.6 * 0.4 + 0.4 * 0.4)).epsilon(1.0e-13));
  REQUIRE(quadratic.quality().min == Approx(linear.quality().min).epsilon(1.0e-12));

  REQUIRE_THROWS_WITH(elevate_to_tet10(make_structured_hex_mesh(spec)),
                      ContainsSubstring("only a Tet4 mesh"));
}

TEST_CASE("Tet10 meshes export VTK cells", "[tet10][mesh][io]") {
  ensure_directory(kTmp);
  const Mesh mesh = make_perturbed_tet10_mesh(box_spec(3, 2, 2, 0.6, 0.4, 0.3), 0.2, 9u);
  const std::string path = std::string(kTmp) + "/tet10.vtk";
  VtkWriter(mesh, "tet10 mesh").write(path);
  const std::string text = read_text(path);
  const Index ne = mesh.num_elements();
  REQUIRE(text.find("CELLS " + std::to_string(ne) + " " + std::to_string(11 * ne)) !=
          std::string::npos);
  std::istringstream lines(text.substr(text.find("CELL_TYPES")));
  std::string header;
  std::getline(lines, header);
  int value = 0;
  lines >> value;
  REQUIRE(value == 24);
}

// ---------------------------------------------------------------------------
// Readers and writers
// ---------------------------------------------------------------------------

TEST_CASE("Gmsh 10-node tetrahedra are read in SparLab's node order, repaired and grouped",
          "[tet10][io][gmsh]") {
  const Mesh original = make_perturbed_tet10_mesh(box_spec(3, 2, 2, 0.3, 0.2, 0.2), 0.15, 5u);
  const std::vector<std::pair<std::string, std::function<bool(const Vector3&)>>> groups = {
      {"fixed", [](const Vector3& c) { return c.x() < 1.0e-9; }},
      {"top", [](const Vector3& c) { return c.z() > 0.2 - 1.0e-9; }}};
  {
    std::istringstream in(write_tet10_msh22(original, groups));
    MeshReadReport report;
    const Mesh mesh = read_gmsh(in, "part.msh", MeshReadOptions(), &report);
    require_same_mesh(mesh, original, 1.0e-15);
    REQUIRE(report.element_type == ElementType::Tet10);
    REQUIRE(report.reoriented == 0);
    REQUIRE(report.boundary_elements == 2 * 2 * 2 + 2 * 3 * 2);
    // The 6-node boundary triangles bring their edge nodes into the groups:
    // the (2 ny + 1)(2 nz + 1) half-cell grid of the face.
    REQUIRE(mesh.node_sets().at("fixed").size() == 5 * 5);
    for (Index n : mesh.node_sets().at("top")) REQUIRE(mesh.node(n).z() == Approx(0.2));
  }
  {
    // A cell listed as its mirror image comes back in the original order.
    std::istringstream in(write_tet10_msh22(original, groups, true));
    MeshReadReport report;
    const Mesh mesh = read_gmsh(in, "part.msh", MeshReadOptions(), &report);
    REQUIRE(report.reoriented == 1);
    require_same_mesh(mesh, original, 1.0e-15);
  }
  // Other second-order cells are still refused, with the way out.
  std::istringstream bad("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n6\n1 0 0 0\n2 1 0 0\n"
                         "3 0 1 0\n4 0.5 0 0\n5 0.5 0.5 0\n6 0 0.5 0\n$EndNodes\n$Elements\n1\n"
                         "1 9 2 1 1 1 2 3 4 5 6\n$EndElements\n");
  REQUIRE_THROWS_WITH(read_gmsh(bad, "tri6.msh"), ContainsSubstring("10-node tetrahedron"));
}

TEST_CASE("C3D10 decks from SparLab's CalculiX writer read back node for node",
          "[tet10][io][abaqus]") {
  ensure_directory(kTmp);
  const Mesh original = make_perturbed_tet10_mesh(box_spec(3, 2, 2, 0.3, 0.2, 0.2), 0.2, 8u);
  FemModel model(Mesh(original), default_material(), 1.0, StressState::ThreeDimensional,
                 IntegrationOptions());
  DisplacementConstraint root;
  Selector box;
  box.kind = SelectorKind::Box;
  box.xmax = 0.0;
  root.region.members.push_back(box);
  root.fix_x = root.fix_y = root.fix_z = true;
  model.constraints().push_back(root);
  LoadCaseSpec load;
  load.name = "tip";
  PointLoadSpec tip;
  Selector tip_node;
  tip_node.kind = SelectorKind::NearestNode;
  tip_node.point = Vector3(0.3, 0.0, 0.0);
  tip.region.members.push_back(tip_node);
  tip.force = Vector3(0.0, -100.0, 0.0);
  load.point_loads.push_back(tip);
  model.load_case_specs().push_back(load);
  model.finalize();
  REQUIRE(calculix_element_type(model) == "C3D10");
  const std::vector<std::string> decks =
      write_calculix_decks(model, std::string(kTmp) + "/ccx_tet10", "rt");
  MeshReadReport report;
  const Mesh mesh = read_mesh_file(decks.front(), MeshReadOptions(), &report);
  require_same_mesh(mesh, original, 0.0);
  REQUIRE(report.element_type == ElementType::Tet10);
  std::remove(decks.front().c_str());
}

TEST_CASE("CalculiX decks keep every data field within CalculiX's 20 characters",
          "[tet10][io][abaqus][cross-validation]") {
  // A face traction on Tet10 cells leaves round-off corner loads such as
  // 1.4408030432795649e-18 (22 characters at 17 digits), which CalculiX
  // refuses to read. The writer trims the digits until a field fits.
  ensure_directory(kTmp);
  FemModel model(make_structured_tet10_mesh(box_spec(4, 2, 2, 1.0, 0.05, 0.05)),
                 default_material(), 1.0, StressState::ThreeDimensional, IntegrationOptions());
  DisplacementConstraint root;
  Selector base;
  base.kind = SelectorKind::Box;
  base.xmax = 0.0;
  root.region.members.push_back(base);
  root.fix_x = root.fix_y = root.fix_z = true;
  model.constraints().push_back(root);
  LoadCaseSpec load;
  load.name = "axial";
  TractionLoadSpec end;
  Selector face;
  face.kind = SelectorKind::Box;
  face.xmin = 1.0;
  end.region.members.push_back(face);
  end.traction = Vector3(-400.0, 0.0, 0.0);
  load.tractions.push_back(end);
  model.load_case_specs().push_back(load);
  model.finalize();
  const std::vector<std::string> decks =
      write_calculix_decks(model, std::string(kTmp) + "/ccx_fields", "fields");
  std::ifstream in(decks.front());
  std::string line;
  std::size_t fields = 0;
  std::size_t longest = 0;
  bool in_loads = false;
  Scalar total = 0.0;
  while (std::getline(in, line)) {
    if (!line.empty() && line[0] == '*') {
      in_loads = line.rfind("*CLOAD", 0) == 0;
      continue;
    }
    if (line.rfind("SparLab", 0) == 0) continue;  // the heading's text line
    std::stringstream row(line);
    std::string item;
    std::vector<std::string> items;
    while (std::getline(row, item, ',')) {
      const std::size_t first = item.find_first_not_of(' ');
      items.push_back(first == std::string::npos ? "" : item.substr(first));
    }
    for (const std::string& f : items) {
      longest = std::max(longest, f.size());
      ++fields;
    }
    if (in_loads) total += std::stod(items.at(2));
  }
  INFO("longest field " << longest << " characters");
  REQUIRE(fields > 0);
  REQUIRE(longest <= 20);
  // Trimmed, the loads still add up to the applied resultant.
  REQUIRE(total == Approx(-400.0 * 0.05 * 0.05).epsilon(1e-13));
  std::remove(decks.front().c_str());
}

TEST_CASE("mesh.order builds Tet10 decks and refuses what it cannot mean",
          "[tet10][io][config]") {
  const std::string dir = std::string(kTmp) + "/deck";
  ensure_directory(dir);
  const auto parse = [&](const std::string& mesh_block) {
    const json::Value doc = json::parse(
        R"({ "mesh": )" + mesh_block + R"(,
             "material": { "youngs_modulus": 70e9, "poisson_ratio": 0.3 },
             "boundary_conditions": [ { "fix": ["x", "y", "z"], "region": { "all": true } } ],
             "load_cases": [ { "name": "c", "prescribed_displacement_only": true } ] })");
    return parse_configuration(doc, "inline", true, dir);
  };
  const Configuration structured = parse(
      R"({ "type": "structured_tet", "nx": 4, "ny": 2, "nz": 2, "lx": 1, "ly": 0.5, "lz": 0.5,
           "order": 2 })");
  REQUIRE(structured.mesh_order == 2);
  REQUIRE_THAT(structured.describe_mesh(), ContainsSubstring("(Tet10)"));
  const Mesh built = build_mesh(structured);
  REQUIRE(built.element_type() == ElementType::Tet10);
  REQUIRE(built.num_elements() == 4 * 2 * 2 * 6);

  // A linear tetrahedral file is elevated on request; a quadratic one needs
  // no key and refuses order 1.
  const Mesh linear = make_structured_tet_mesh(box_spec(2, 2, 2, 0.2, 0.2, 0.2));
  {
    FemModel model(Mesh(linear), default_material(), 1.0, StressState::ThreeDimensional,
                   IntegrationOptions());
    DisplacementConstraint all;
    Selector every;
    every.kind = SelectorKind::All;
    all.region.members.push_back(every);
    all.fix_x = all.fix_y = all.fix_z = true;
    model.constraints().push_back(all);
    LoadCaseSpec load;
    load.name = "none";
    load.prescribed_displacement_only = true;
    model.load_case_specs().push_back(load);
    model.finalize();
    const std::vector<std::string> decks = write_calculix_decks(model, dir + "/lin", "x");
    std::rename(decks.front().c_str(), (dir + "/linear.inp").c_str());
    FemModel quad_model(elevate_to_tet10(linear), default_material(), 1.0,
                        StressState::ThreeDimensional, IntegrationOptions());
    quad_model.constraints() = model.constraints();
    quad_model.load_case_specs() = model.load_case_specs();
    quad_model.finalize();
    const std::vector<std::string> quad_decks =
        write_calculix_decks(quad_model, dir + "/quad", "x");
    std::rename(quad_decks.front().c_str(), (dir + "/quadratic.inp").c_str());
  }
  const Configuration elevated = parse(R"({ "type": "file", "path": "linear.inp", "order": 2 })");
  REQUIRE(elevated.mesh_elevated);
  REQUIRE(elevated.mesh_order == 2);
  REQUIRE_THAT(elevated.describe_mesh(), ContainsSubstring("Tet4 elevated to Tet10"));
  REQUIRE(build_mesh(elevated).num_nodes() == elevate_to_tet10(linear).num_nodes());
  const Configuration native = parse(R"({ "type": "file", "path": "quadratic.inp" })");
  REQUIRE(native.mesh_order == 2);
  REQUIRE_FALSE(native.mesh_elevated);
  REQUIRE(parse(R"({ "type": "file", "path": "linear.inp" })").mesh_order == 1);
  REQUIRE_THROWS_WITH(parse(R"({ "type": "file", "path": "quadratic.inp", "order": 1 })"),
                      ContainsSubstring("does not drop edge nodes"));
  REQUIRE_THROWS_WITH(
      parse(R"({ "type": "structured_hex", "nx": 2, "ny": 2, "nz": 2, "lx": 1, "ly": 1,
                 "lz": 1, "order": 2 })"),
      ContainsSubstring("structured_tet"));
  REQUIRE_THROWS_WITH(
      parse(R"({ "type": "structured_tet", "nx": 2, "ny": 2, "nz": 2, "lx": 1, "ly": 1,
                 "lz": 1, "order": 3 })"),
      ContainsSubstring("must be 1 or 2"));
}
