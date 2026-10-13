/// \file test_unstructured.cpp
/// \brief Linear triangles and tetrahedra, the structured simplex generators,
///        and unstructured meshes read from Gmsh (.msh 2.2 / 4.1) and Abaqus /
///        CalculiX (.inp) files, with their named groups.
///
/// The element tests check what must be exact for a constant-strain element:
/// rigid modes, the energy of a linear field, mass and consistent loads. The
/// patch tests run on distorted simplex meshes. The reader tests feed files
/// written independently of the mesh classes (by hand, or by the small writers
/// below, or by SparLab's own CalculiX exporter) and compare the result node
/// by node, and they check that each malformed input fails with a message
/// that says what to change.
#include "TestSupport.hpp"

#include "sparlab/core/Exceptions.hpp"
#include "sparlab/elements/Element.hpp"
#include "sparlab/elements/Tet4.hpp"
#include "sparlab/elements/Tri3.hpp"
#include "sparlab/fem/ModalAnalysis.hpp"
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
#include <set>
#include <sstream>

using namespace sparlab;
using namespace sparlab::testing;
using Catch::Approx;
using Catch::Matchers::ContainsSubstring;

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Number of eigenvalues below `ratio` times the largest one.
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

/// Engineering-strain Voigt vector of a 2-D displacement gradient.
Vector3 voigt_2d(const Matrix2& g) { return Vector3(g(0, 0), g(1, 1), g(0, 1) + g(1, 0)); }

Vector6 voigt_3d(const Matrix3& g) {
  Vector6 e;
  e << g(0, 0), g(1, 1), g(2, 2), g(0, 1) + g(1, 0), g(1, 2) + g(2, 1), g(2, 0) + g(0, 2);
  return e;
}

Matrix2 gradient_2d() {
  Matrix2 g;
  g << 2.0e-4, -1.5e-4,
       0.5e-4, -3.0e-4;
  return g;
}

Matrix3 gradient_3d() {
  Matrix3 g;
  g << 3.0e-4, 1.0e-4, -0.5e-4,
       1.0e-4, -2.0e-4, 0.7e-4,
       -0.5e-4, 0.7e-4, 1.5e-4;
  return g;
}

/// A predicate on boundary-face centroids that names a boundary group.
struct BoundaryGroup {
  std::string name;
  std::function<bool(const Vector3&)> contains;
};

Vector3 face_centroid(const Mesh& mesh, const Mesh::BoundaryFace& f) {
  Vector3 c = Vector3::Zero();
  for (Index n : f.nodes) c += mesh.node(n);
  return c / static_cast<Scalar>(f.nodes.size());
}

int gmsh_type(ElementType type) {
  switch (type) {
    case ElementType::Tri3: return 2;
    case ElementType::Quad4: return 3;
    case ElementType::Tet4: return 4;
    case ElementType::Hex8: return 5;
    case ElementType::Tet10: return 11;
  }
  return -1;
}

int gmsh_face_type(const Mesh& mesh) {
  if (mesh.dim() == 2) return 1;  // 2-node line
  return mesh.element_type() == ElementType::Tet4 ? 2 : 3;
}

/// Node tags are deliberately sparse and shifted so a reader that confuses
/// tags with positions fails.
long long node_tag(Index n) { return 10 * static_cast<long long>(n) + 7; }

/// MSH 2.2 text of `mesh`: the cells in physical group `cell_group` and the
/// boundary faces matching each group as lower-dimensional elements.
std::string write_msh22(const Mesh& mesh, const std::vector<BoundaryGroup>& groups,
                        const std::string& cell_group) {
  std::ostringstream os;
  os << std::setprecision(17);
  const int dim = mesh.dim();
  os << "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n";
  os << "$PhysicalNames\n" << groups.size() + 1 << "\n";
  for (std::size_t g = 0; g < groups.size(); ++g) {
    os << dim - 1 << " " << g + 1 << " \"" << groups[g].name << "\"\n";
  }
  os << dim << " " << groups.size() + 1 << " \"" << cell_group << "\"\n";
  os << "$EndPhysicalNames\n";
  os << "$Nodes\n" << mesh.num_nodes() << "\n";
  for (Index n = 0; n < mesh.num_nodes(); ++n) {
    const Vector3 x = mesh.node(n);
    os << node_tag(n) << " " << x.x() << " " << x.y() << " " << x.z() << "\n";
  }
  os << "$EndNodes\n";
  std::vector<std::string> lines;
  long long tag = 1;
  for (const Mesh::BoundaryFace& f : mesh.boundary_faces()) {
    const Vector3 c = face_centroid(mesh, f);
    for (std::size_t g = 0; g < groups.size(); ++g) {
      if (!groups[g].contains(c)) continue;
      std::ostringstream line;
      line << tag++ << " " << gmsh_face_type(mesh) << " 2 " << g + 1 << " " << 100 + g;
      for (Index n : f.nodes) line << " " << node_tag(n);
      lines.push_back(line.str());
      break;
    }
  }
  for (Index e = 0; e < mesh.num_elements(); ++e) {
    std::ostringstream line;
    line << tag++ << " " << gmsh_type(mesh.element_type()) << " 2 " << groups.size() + 1
         << " 1";
    const Index* en = mesh.element_nodes(e);
    for (int a = 0; a < mesh.nodes_per_elem(); ++a) line << " " << node_tag(en[a]);
    lines.push_back(line.str());
  }
  os << "$Elements\n" << lines.size() << "\n";
  for (const std::string& l : lines) os << l << "\n";
  os << "$EndElements\n";
  return os.str();
}

/// MSH 4.1 text of the same content: one entity per boundary group plus one
/// for the cells, physical groups attached through $Entities.
std::string write_msh41(const Mesh& mesh, const std::vector<BoundaryGroup>& groups,
                        const std::string& cell_group) {
  std::ostringstream os;
  os << std::setprecision(17);
  const int dim = mesh.dim();
  const BoundingBox bb = mesh.bounding_box();
  const std::size_t ng = groups.size();
  os << "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n";
  os << "$PhysicalNames\n" << ng + 1 << "\n";
  for (std::size_t g = 0; g < ng; ++g) {
    os << dim - 1 << " " << g + 1 << " \"" << groups[g].name << "\"\n";
  }
  os << dim << " " << ng + 1 << " \"" << cell_group << "\"\n$EndPhysicalNames\n";
  os << "$Entities\n";
  if (dim == 2) {
    os << "0 " << ng << " 1 0\n";
  } else {
    os << "0 0 " << ng << " 1\n";
  }
  const auto bbox = [&]() {
    std::ostringstream b;
    b << bb.lower.x() << " " << bb.lower.y() << " " << bb.lower.z() << " " << bb.upper.x()
      << " " << bb.upper.y() << " " << bb.upper.z();
    return b.str();
  };
  for (std::size_t g = 0; g < ng; ++g) os << g + 1 << " " << bbox() << " 1 " << g + 1 << " 0\n";
  os << "1 " << bbox() << " 1 " << ng + 1 << " 0\n";
  os << "$EndEntities\n";

  // Nodes: one block on the cell entity.
  os << "$Nodes\n1 " << mesh.num_nodes() << " " << node_tag(0) << " "
     << node_tag(mesh.num_nodes() - 1) << "\n";
  os << dim << " 1 0 " << mesh.num_nodes() << "\n";
  for (Index n = 0; n < mesh.num_nodes(); ++n) os << node_tag(n) << "\n";
  for (Index n = 0; n < mesh.num_nodes(); ++n) {
    const Vector3 x = mesh.node(n);
    os << x.x() << " " << x.y() << " " << x.z() << "\n";
  }
  os << "$EndNodes\n";

  std::vector<std::vector<std::string>> blocks(ng);
  long long tag = 1;
  for (const Mesh::BoundaryFace& f : mesh.boundary_faces()) {
    const Vector3 c = face_centroid(mesh, f);
    for (std::size_t g = 0; g < ng; ++g) {
      if (!groups[g].contains(c)) continue;
      std::ostringstream line;
      line << tag++;
      for (Index n : f.nodes) line << " " << node_tag(n);
      blocks[g].push_back(line.str());
      break;
    }
  }
  std::vector<std::string> cells;
  for (Index e = 0; e < mesh.num_elements(); ++e) {
    std::ostringstream line;
    line << tag++;
    const Index* en = mesh.element_nodes(e);
    for (int a = 0; a < mesh.nodes_per_elem(); ++a) line << " " << node_tag(en[a]);
    cells.push_back(line.str());
  }
  os << "$Elements\n" << ng + 1 << " " << tag - 1 << " 1 " << tag - 1 << "\n";
  for (std::size_t g = 0; g < ng; ++g) {
    os << dim - 1 << " " << g + 1 << " " << gmsh_face_type(mesh) << " " << blocks[g].size()
       << "\n";
    for (const std::string& l : blocks[g]) os << l << "\n";
  }
  os << dim << " 1 " << gmsh_type(mesh.element_type()) << " " << cells.size() << "\n";
  for (const std::string& l : cells) os << l << "\n";
  os << "$EndElements\n";
  return os.str();
}

void require_same_mesh(const Mesh& a, const Mesh& b, Scalar tol = 0.0) {
  REQUIRE(a.dim() == b.dim());
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

void write_text(const std::string& path, const std::string& text) {
  std::ofstream out(path);
  out << text;
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

/// Constrain the given nodes to the linear field u = offset + g x, solve, and
/// return the largest nodal error relative to the field's magnitude.
template <int D>
Scalar linear_patch_error(FemModel& model, const std::vector<Index>& boundary_nodes,
                          const Eigen::Matrix<Scalar, D, D>& g,
                          const Eigen::Matrix<Scalar, D, 1>& offset, Vector* u_out) {
  DisplacementConstraint bc;
  Selector sel;
  sel.kind = SelectorKind::NodeIds;
  sel.ids = boundary_nodes;
  bc.region.members.push_back(sel);
  bc.fix_x = bc.fix_y = true;
  bc.fix_z = D == 3;
  model.constraints().push_back(bc);
  LoadCaseSpec load;
  load.name = "patch";
  load.prescribed_displacement_only = true;
  model.load_case_specs().push_back(load);
  model.finalize();
  for (Index n : boundary_nodes) {
    const Eigen::Matrix<Scalar, D, 1> x = model.mesh().node(n).head(D);
    const Eigen::Matrix<Scalar, D, 1> u = offset + g * x;
    for (int k = 0; k < D; ++k) model.dofs().prescribe(n, k, u(k));
  }
  Assembler assembler(model);
  StaticAnalysisOptions options;
  options.linear.residual_tolerance = 1.0e-9;
  StaticAnalysis analysis(model, assembler, options);
  const Vector u = analysis.solve_all().front().displacement;
  Scalar err = 0.0;
  Scalar scale = 0.0;
  for (Index n = 0; n < model.mesh().num_nodes(); ++n) {
    const Eigen::Matrix<Scalar, D, 1> x = model.mesh().node(n).head(D);
    const Eigen::Matrix<Scalar, D, 1> expected = offset + g * x;
    for (int k = 0; k < D; ++k) err = std::max(err, std::abs(u(n * D + k) - expected(k)));
    scale = std::max(scale, expected.cwiseAbs().maxCoeff());
  }
  if (u_out != nullptr) *u_out = u;
  return err / scale;
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

const char* kTmp = "results/_test_tmp/unstructured";

}  // namespace

// ---------------------------------------------------------------------------
// Tri3
// ---------------------------------------------------------------------------

TEST_CASE("Tri3 stiffness is symmetric, exact for linear fields and has three rigid modes",
          "[unstructured][element][tri3]") {
  const Tri3Element element;
  Matrix coords(2, 3);
  coords << 0.1, 1.3, 0.4,
            0.2, 0.5, 1.4;
  const IsotropicMaterial material = default_material(0.3);
  const Matrix d = material.plane_stress_matrix();
  const Scalar t = 0.02;
  const Matrix k = element.stiffness(coords, d, t, IntegrationOptions());
  REQUIRE(k.rows() == 6);
  REQUIRE((k - k.transpose()).cwiseAbs().maxCoeff() <= 1.0e-12 * k.cwiseAbs().maxCoeff());
  REQUIRE(count_zero_modes(k) == 3);

  // Closed form K = t A B^T D B with the textbook B.
  Scalar area = 0.0;
  const Eigen::Matrix<Scalar, 3, 2> grad = tri3_shape_gradients(coords, &area);
  const Scalar x1 = coords(0, 0), y1 = coords(1, 0), x2 = coords(0, 1), y2 = coords(1, 1),
               x3 = coords(0, 2), y3 = coords(1, 2);
  REQUIRE(area == Approx(0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))));
  Matrix b = Matrix::Zero(3, 6);
  const Scalar a2 = 2.0 * area;
  const Scalar bi[3] = {y2 - y3, y3 - y1, y1 - y2};
  const Scalar ci[3] = {x3 - x2, x1 - x3, x2 - x1};
  for (int a = 0; a < 3; ++a) {
    REQUIRE(grad(a, 0) == Approx(bi[a] / a2));
    REQUIRE(grad(a, 1) == Approx(ci[a] / a2));
    b(0, 2 * a) = bi[a] / a2;
    b(1, 2 * a + 1) = ci[a] / a2;
    b(2, 2 * a) = ci[a] / a2;
    b(2, 2 * a + 1) = bi[a] / a2;
  }
  const Matrix expected = t * area * b.transpose() * d * b;
  REQUIRE((k - expected).cwiseAbs().maxCoeff() <= 1.0e-10 * k.cwiseAbs().maxCoeff());

  // The energy of a linear displacement field is exact: 1/2 eps^T D eps t A.
  const Matrix2 g = gradient_2d();
  Vector u(6);
  for (int a = 0; a < 3; ++a) u.segment<2>(2 * a) = g * Vector2(coords.col(a));
  const Vector3 eps = voigt_2d(g);
  REQUIRE(0.5 * u.dot(k * u) == Approx(0.5 * eps.dot(d * eps) * t * area).epsilon(1.0e-12));

  // Rigid translations and the infinitesimal rotation carry no energy.
  Matrix modes = Matrix::Zero(6, 3);
  for (int a = 0; a < 3; ++a) {
    modes(2 * a, 0) = 1.0;
    modes(2 * a + 1, 1) = 1.0;
    modes(2 * a, 2) = -coords(1, a);
    modes(2 * a + 1, 2) = coords(0, a);
  }
  REQUIRE((k * modes).cwiseAbs().maxCoeff() <= 1.0e-9 * k.cwiseAbs().maxCoeff());
}

TEST_CASE("Tri3 mass, edge load and orientation checks", "[unstructured][element][tri3]") {
  const Tri3Element element;
  Matrix coords(2, 3);
  coords << 0.0, 2.0, 0.5,
            0.0, 0.0, 1.5;
  const Scalar area = 0.5 * 2.0 * 1.5;
  const Scalar rho = 2700.0;
  const Scalar t = 0.01;
  const Matrix m = element.consistent_mass(coords, rho, t, IntegrationOptions());
  // Each direction carries the full mass once.
  Scalar mx = 0.0;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) mx += m(2 * a, 2 * b);
  }
  REQUIRE(mx == Approx(rho * t * area));
  REQUIRE(m(0, 0) == Approx(rho * t * area / 6.0));
  REQUIRE(m(0, 2) == Approx(rho * t * area / 12.0));
  REQUIRE(m(0, 1) == 0.0);
  Eigen::SelfAdjointEigenSolver<Matrix> eig(m);
  REQUIRE(eig.eigenvalues().minCoeff() > 0.0);

  // A uniform traction on edge 1 (nodes 1 -> 2) puts t L q / 2 on each node.
  const Vector3 q(3.0e4, -1.0e4, 0.0);
  const Vector f = element.boundary_traction(coords, 1, q, t, IntegrationOptions());
  const Scalar length = (Vector2(coords.col(2)) - Vector2(coords.col(1))).norm();
  REQUIRE(f.segment<2>(0).isZero());
  REQUIRE(f(2) == Approx(0.5 * t * length * q.x()));
  REQUIRE(f(3) == Approx(0.5 * t * length * q.y()));
  REQUIRE(f(4) == Approx(0.5 * t * length * q.x()));
  REQUIRE(f(5) == Approx(0.5 * t * length * q.y()));

  // Clockwise and collapsed triangles are rejected with the reason.
  Matrix clockwise = coords;
  clockwise.col(1).swap(clockwise.col(2));
  REQUIRE_THROWS_WITH(element.stiffness(clockwise, default_material().plane_stress_matrix(),
                                        t, IntegrationOptions()),
                      ContainsSubstring("counter-clockwise"));
  Matrix collapsed(2, 3);
  collapsed << 0.0, 1.0, 2.0,
               0.0, 1.0, 2.0;
  REQUIRE_THROWS_AS(element.stiffness(collapsed, default_material().plane_stress_matrix(), t,
                                      IntegrationOptions()),
                    MeshError);
  REQUIRE(make_element(ElementType::Tri3)->num_nodes() == 3);
  REQUIRE(make_element(ElementType::Tri3)->num_faces() == 3);
}

// ---------------------------------------------------------------------------
// Tet4
// ---------------------------------------------------------------------------

TEST_CASE("Tet4 stiffness is symmetric, exact for linear fields and has six rigid modes",
          "[unstructured][element][tet4]") {
  const Tet4Element element;
  Matrix coords(3, 4);
  coords << 0.1, 1.2, 0.3, 0.2,
            0.0, 0.2, 1.1, 0.3,
            0.05, 0.1, 0.2, 0.9;
  REQUIRE(tet4_volume(coords) > 0.0);
  const IsotropicMaterial material = default_material(0.3);
  const Matrix d = material.three_dimensional_matrix();
  const Matrix k = element.stiffness(coords, d, 1.0, IntegrationOptions());
  REQUIRE(k.rows() == 12);
  REQUIRE((k - k.transpose()).cwiseAbs().maxCoeff() <= 1.0e-12 * k.cwiseAbs().maxCoeff());
  REQUIRE(count_zero_modes(k) == 6);

  Scalar volume = 0.0;
  const Eigen::Matrix<Scalar, 4, 3> grad = tet4_shape_gradients(coords, &volume);
  REQUIRE(volume == Approx(tet4_volume(coords)));
  // Gradients of a partition of unity sum to zero, and grad N_a . (x_b - x_0)
  // reproduces the nodal delta.
  REQUIRE(grad.colwise().sum().cwiseAbs().maxCoeff() < 1.0e-12);
  for (int a = 0; a < 4; ++a) {
    for (int b = 1; b < 4; ++b) {
      const Vector3 edge = coords.col(b) - coords.col(0);
      const Scalar value = grad.row(a).dot(edge);
      const Scalar expected = (a == b ? 1.0 : 0.0) - (a == 0 ? 1.0 : 0.0);
      REQUIRE(value == Approx(expected).margin(1.0e-12));
    }
  }

  const Matrix3 g = gradient_3d();
  Vector u(12);
  for (int a = 0; a < 4; ++a) u.segment<3>(3 * a) = g * Vector3(coords.col(a));
  const Vector6 eps = voigt_3d(g);
  REQUIRE(0.5 * u.dot(k * u) == Approx(0.5 * eps.dot(d * eps) * volume).epsilon(1.0e-12));

  // Solid elements take no thickness.
  REQUIRE_THROWS_AS(element.stiffness(coords, d, 0.5, IntegrationOptions()), ConfigError);
}

TEST_CASE("Tet4 mass, face load and orientation checks", "[unstructured][element][tet4]") {
  const Tet4Element element;
  Matrix coords(3, 4);
  coords << 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 2.0, 0.0,
            0.0, 0.0, 0.0, 3.0;
  const Scalar volume = 1.0 * 2.0 * 3.0 / 6.0;
  const Scalar rho = 7850.0;
  const Matrix m = element.consistent_mass(coords, rho, 1.0, IntegrationOptions());
  Scalar mz = 0.0;
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) mz += m(3 * a + 2, 3 * b + 2);
  }
  REQUIRE(mz == Approx(rho * volume));
  REQUIRE(m(0, 0) == Approx(rho * volume / 10.0));
  REQUIRE(m(0, 3) == Approx(rho * volume / 20.0));
  Eigen::SelfAdjointEigenSolver<Matrix> eig(m);
  REQUIRE(eig.eigenvalues().minCoeff() > 0.0);

  // Face 0 is (0, 2, 1): the z = 0 face with its normal pointing to -z.
  const std::vector<int>& face = element.face_nodes(0);
  REQUIRE(face == std::vector<int>({0, 2, 1}));
  const Vector3 q(0.0, 0.0, -5.0e5);
  const Vector f = element.boundary_traction(coords, 0, q, 1.0, IntegrationOptions());
  const Scalar area = 0.5 * 1.0 * 2.0;
  for (int node : {0, 1, 2}) REQUIRE(f(3 * node + 2) == Approx(q.z() * area / 3.0));
  REQUIRE(f.segment<3>(9).isZero());
  REQUIRE(f.sum() == Approx(q.z() * area));

  // Every face is wound outward.
  const Vector3 centroid = coords.rowwise().mean();
  for (int lf = 0; lf < 4; ++lf) {
    const std::vector<int>& fn = element.face_nodes(lf);
    const Vector3 a = coords.col(fn[0]);
    const Vector3 b = coords.col(fn[1]);
    const Vector3 c = coords.col(fn[2]);
    const Vector3 normal = (b - a).cross(c - a);
    REQUIRE(normal.dot((a + b + c) / 3.0 - centroid) > 0.0);
  }

  Matrix inverted = coords;
  inverted.col(1).swap(inverted.col(2));
  REQUIRE_THROWS_WITH(
      element.stiffness(inverted, default_material().three_dimensional_matrix(), 1.0,
                        IntegrationOptions()),
      ContainsSubstring("inverted"));
}

// ---------------------------------------------------------------------------
// Structured simplex meshes, quality and patch tests
// ---------------------------------------------------------------------------

TEST_CASE("structured triangle and tetrahedron meshes split the grid conformingly",
          "[unstructured][mesh]") {
  SECTION("triangles") {
    const StructuredMeshSpec spec = box_spec(5, 3, 1, 1.0, 0.6, 1.0);
    const Mesh mesh = make_structured_tri_mesh(spec);
    REQUIRE(mesh.element_type() == ElementType::Tri3);
    REQUIRE(mesh.num_elements() == 2 * 5 * 3);
    REQUIRE(mesh.num_nodes() == 6 * 4);
    REQUIRE_FALSE(mesh.structured_info().has_value());
    Scalar area = 0.0;
    for (Index e = 0; e < mesh.num_elements(); ++e) {
      REQUIRE(mesh.element_measure(e) > 0.0);
      area += mesh.element_measure(e);
      // Elements 2c and 2c + 1 lie in cell c.
      const Index cell = e / 2;
      const Vector3 c = mesh.element_centroid(e);
      REQUIRE(std::floor(c.x() / 0.2) == static_cast<Scalar>(cell % 5));
      REQUIRE(std::floor(c.y() / 0.2) == static_cast<Scalar>(cell / 5));
    }
    REQUIRE(area == Approx(0.6));
    // Only the outline is boundary: every interior edge is shared.
    REQUIRE(mesh.boundary_faces().size() == 2 * (5 + 3));
    // Alternating diagonals: both orientations occur.
    int diag_02 = 0;
    for (Index c = 0; c < 15; ++c) {
      const Index* n0 = mesh.element_nodes(2 * c);
      const Index* n1 = mesh.element_nodes(2 * c + 1);
      const std::set<Index> shared = [&] {
        std::set<Index> s;
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            if (n0[a] == n1[b]) s.insert(n0[a]);
          }
        }
        return s;
      }();
      REQUIRE(shared.size() == 2);
      const Index i = c % 5;
      const Index j = c / 5;
      const Index lower_left = j * 6 + i;
      if (shared.count(lower_left)) ++diag_02;
    }
    REQUIRE(diag_02 == 8);  // cells with i + j even
    // Mean edge length of right isosceles triangles with legs h.
    REQUIRE(mesh.mean_element_size() == Approx(0.2 * (2.0 + std::sqrt(2.0)) / 3.0));
  }

  SECTION("tetrahedra") {
    const StructuredMeshSpec spec = box_spec(3, 2, 2, 0.6, 0.4, 0.4);
    const Mesh mesh = make_structured_tet_mesh(spec);
    REQUIRE(mesh.element_type() == ElementType::Tet4);
    REQUIRE(mesh.num_elements() == 6 * 3 * 2 * 2);
    REQUIRE(mesh.num_nodes() == 4 * 3 * 3);
    Scalar volume = 0.0;
    for (Index e = 0; e < mesh.num_elements(); ++e) {
      REQUIRE(mesh.element_measure(e) > 0.0);
      REQUIRE(mesh.element_measure(e) == Approx(0.2 * 0.2 * 0.2 / 6.0));
      volume += mesh.element_measure(e);
    }
    REQUIRE(volume == Approx(0.6 * 0.4 * 0.4));
    // Two triangles per boundary quad of the grid.
    REQUIRE(mesh.boundary_faces().size() == 2 * 2 * (3 * 2 + 2 * 2 + 2 * 3));
    for (const Mesh::BoundaryFace& f : mesh.boundary_faces()) REQUIRE(f.nodes.size() == 3);
    REQUIRE(mesh.mean_element_size() ==
            Approx(0.2 * (3.0 + 2.0 * std::sqrt(2.0) + std::sqrt(3.0)) / 6.0));
  }
}

TEST_CASE("mesh quality is 1 for ideal cells and flags slivers", "[unstructured][mesh]") {
  {
    Matrix x(2, 3);
    x << 0.0, 1.0, 0.5,
         0.0, 0.0, std::sqrt(3.0) / 2.0;
    const Mesh equilateral(x, {0, 1, 2}, ElementType::Tri3);
    REQUIRE(equilateral.quality().min == Approx(1.0));
    Matrix r(2, 3);
    r << 0.0, 1.0, 0.0,
         0.0, 0.0, 1.0;
    const Mesh right(r, {0, 1, 2}, ElementType::Tri3);
    REQUIRE(right.quality().min == Approx(std::sqrt(3.0) / 2.0));
  }
  {
    Matrix x(3, 4);
    x << 1.0, -1.0, -1.0, 1.0,
         1.0, -1.0, 1.0, -1.0,
         1.0, 1.0, -1.0, -1.0;
    // Positively oriented regular tetrahedron.
    if (tet4_volume(x) < 0.0) x.col(1).swap(x.col(2));
    const Mesh regular(x, {0, 1, 2, 3}, ElementType::Tet4);
    REQUIRE(regular.quality().min == Approx(1.0));
    // A sliver: four nearly coplanar points.
    Matrix s(3, 4);
    s << 0.0, 1.0, 1.0, 0.0,
         0.0, 0.0, 1.0, 1.0,
         0.0, 0.01, 0.0, 0.01;
    if (tet4_volume(s) < 0.0) s.col(1).swap(s.col(2));
    const Mesh sliver(s, {0, 1, 2, 3}, ElementType::Tet4);
    const MeshQuality q = sliver.quality();
    REQUIRE(q.min < 0.05);
    REQUIRE(q.poor_elements == 1);
    REQUIRE(q.worst_element == 0);
  }
  // Kuhn tetrahedra of a cube: 6 sqrt(2) V / l_rms^3 with l_rms^2 = 10/6.
  const Mesh kuhn = make_structured_tet_mesh(box_spec(1, 1, 1, 1.0, 1.0, 1.0));
  REQUIRE(kuhn.quality().min ==
          Approx(6.0 * std::sqrt(2.0) / 6.0 / std::pow(10.0 / 6.0, 1.5)));
  // Squares and cubes are ideal for the scaled-Jacobian metric.
  REQUIRE(make_structured_quad_mesh(box_spec(2, 2, 1, 1.0, 1.0, 1.0)).quality().min ==
          Approx(1.0));
  REQUIRE(make_structured_hex_mesh(box_spec(2, 2, 2, 1.0, 1.0, 1.0)).quality().min ==
          Approx(1.0));
}

TEST_CASE("constant-strain patch tests pass on distorted triangle and tetrahedron meshes",
          "[unstructured][patch][verification]") {
  for (Scalar perturbation : {0.0, 0.2, 0.4}) {
    INFO("perturbation " << perturbation);
    const StructuredMeshSpec spec = box_spec(4, 3, 1, 1.2, 0.9, 1.0);
    const IsotropicMaterial material(200.0e9, 0.3, 7850.0, "patch");
    FemModel model(make_perturbed_tri_mesh(spec, perturbation, 11u), material, 0.01,
                   StressState::PlaneStress, IntegrationOptions());
    const std::vector<Index> boundary = boundary_node_list(model.mesh());
    REQUIRE(boundary.size() == static_cast<std::size_t>(model.mesh().num_nodes() - 3 * 2));
    const Matrix2 g = gradient_2d();
    Vector u;
    const Scalar err =
        linear_patch_error<2>(model, boundary, g, Vector2(1.0e-4, -2.0e-4), &u);
    REQUIRE(err == Approx(0.0).margin(1.0e-11));
    Assembler assembler(model);
    const StressField field = recover_stresses(model, assembler, u);
    const Vector3 exact_stress = material.plane_stress_matrix() * voigt_2d(g);
    for (Index e = 0; e < model.mesh().num_elements(); ++e) {
      REQUIRE((Vector3(field.element_stress.col(e)) - exact_stress).cwiseAbs().maxCoeff() <=
              1.0e-9 * exact_stress.cwiseAbs().maxCoeff());
    }
  }
  for (Scalar perturbation : {0.0, 0.15, 0.3}) {
    INFO("perturbation " << perturbation);
    const StructuredMeshSpec spec = box_spec(3, 3, 3, 1.5, 1.0, 1.2);
    const IsotropicMaterial material(200.0e9, 0.3, 7850.0, "patch3d");
    FemModel model(make_perturbed_tet_mesh(spec, perturbation, 7u), material, 1.0,
                   StressState::ThreeDimensional, IntegrationOptions());
    const std::vector<Index> boundary = boundary_node_list(model.mesh());
    REQUIRE(boundary.size() == static_cast<std::size_t>(model.mesh().num_nodes() - 8));
    const Matrix3 g = gradient_3d();
    Vector u;
    const Scalar err =
        linear_patch_error<3>(model, boundary, g, Vector3(1.0e-4, -2.0e-4, 0.5e-4), &u);
    REQUIRE(err == Approx(0.0).margin(1.0e-11));
    Assembler assembler(model);
    const StressField field = recover_stresses(model, assembler, u);
    const Vector6 exact_stress = material.three_dimensional_matrix() * voigt_3d(g);
    for (Index e = 0; e < model.mesh().num_elements(); ++e) {
      REQUIRE((Vector6(field.element_stress.col(e)) - exact_stress).cwiseAbs().maxCoeff() <=
              1.0e-9 * exact_stress.cwiseAbs().maxCoeff());
    }
    REQUIRE(field.element_strain_energy.sum() ==
            Approx(0.5 * voigt_3d(g).dot(exact_stress) * model.domain_volume())
                .epsilon(1.0e-9));
  }
}

TEST_CASE("simplex meshes of the same box conserve mass and converge on the Hex8 answer",
          "[unstructured][verification]") {
  // Compliance of a clamped block under a tip load: the Tet4 mesh of the
  // same grid is stiffer than the Hex8 one (constant strain locks in
  // bending), and the gap closes under refinement.
  const auto compliance = [](const Mesh& mesh) {
    FemModel model(Mesh(mesh), default_material(0.3), 1.0, StressState::ThreeDimensional,
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
    TractionLoadSpec t;
    Selector tip;
    tip.kind = SelectorKind::Box;
    tip.xmin = 1.0;
    t.region.members.push_back(tip);
    t.traction = Vector3(0.0, -1.0e6, 0.0);
    load.tractions.push_back(t);
    model.load_case_specs().push_back(load);
    model.finalize();
    Assembler assembler(model);
    StaticAnalysis analysis(model, assembler, StaticAnalysisOptions());
    const StaticSolution sol = analysis.solve_all().front();
    REQUIRE(sol.equilibrium.relative_force_error < 1.0e-10);
    return sol.compliance;
  };
  std::vector<Scalar> gaps;
  for (Index n : {2, 4}) {
    const StructuredMeshSpec spec = box_spec(5 * n, n, n, 1.0, 0.2, 0.2);
    const Scalar c_hex = compliance(make_structured_hex_mesh(spec));
    const Scalar c_tet = compliance(make_structured_tet_mesh(spec));
    INFO("n = " << n << ": hex " << c_hex << ", tet " << c_tet);
    REQUIRE(c_tet < c_hex);
    gaps.push_back((c_hex - c_tet) / c_hex);
  }
  REQUIRE(gaps[1] < 0.5 * gaps[0]);

  // Consistent mass of the tet and triangle meshes equals rho * volume.
  const Mesh tets = make_structured_tet_mesh(box_spec(3, 2, 2, 0.3, 0.2, 0.2));
  FemModel solid(Mesh(tets), default_material(), 1.0, StressState::ThreeDimensional,
                 IntegrationOptions());
  DisplacementConstraint fix;
  Selector left;
  left.kind = SelectorKind::Box;
  left.xmax = 0.0;
  fix.region.members.push_back(left);
  fix.fix_x = fix.fix_y = fix.fix_z = true;
  solid.constraints().push_back(fix);
  solid.finalize(false);
  Assembler assembler(solid);
  ModalAnalysisOptions options;
  options.num_modes = 3;
  const ModalResult modes = solve_modal(solid, assembler, options);
  REQUIRE(modes.total_mass == Approx(2700.0 * 0.3 * 0.2 * 0.2).epsilon(1.0e-12));
  REQUIRE(modes.frequencies_hz(0) > 0.0);
}

TEST_CASE("simplex meshes export VTK cells and CalculiX elements",
          "[unstructured][geometry][io]") {
  ensure_directory(kTmp);
  const Mesh tets = make_perturbed_tet_mesh(box_spec(3, 2, 2, 0.6, 0.4, 0.3), 0.2, 9u);
  const std::string path = std::string(kTmp) + "/tets.vtk";
  VtkWriter(tets, "tet mesh").write(path);
  const std::string text = read_text(path);
  REQUIRE(text.find("CELLS 72 360") != std::string::npos);
  std::istringstream lines(text.substr(text.find("CELL_TYPES 72")));
  std::string header;
  std::getline(lines, header);
  int value = 0;
  lines >> value;
  REQUIRE(value == 10);
}

// ---------------------------------------------------------------------------
// Gmsh reader
// ---------------------------------------------------------------------------

TEST_CASE("Gmsh 2.2 and 4.1 files round-trip triangle and tetrahedron meshes with groups",
          "[unstructured][io][gmsh]") {
  SECTION("2-D, MSH 2.2") {
    const Mesh original =
        make_perturbed_tri_mesh(box_spec(6, 3, 1, 0.6, 0.3, 1.0), 0.2, 21u);
    const std::vector<BoundaryGroup> groups = {
        {"left", [](const Vector3& c) { return c.x() < 1.0e-9; }},
        {"right", [](const Vector3& c) { return c.x() > 0.6 - 1.0e-9; }}};
    std::istringstream in(write_msh22(original, groups, "plate"));
    MeshReadReport report;
    const Mesh mesh = read_gmsh(in, "plate.msh", MeshReadOptions(), &report);
    require_same_mesh(mesh, original, 1.0e-15);
    REQUIRE(report.format == "gmsh");
    REQUIRE(report.version == "2.2");
    REQUIRE(report.dimension == 2);
    REQUIRE(report.element_type == ElementType::Tri3);
    REQUIRE(report.cells == original.num_elements());
    REQUIRE(report.boundary_elements == 6);
    REQUIRE(report.reoriented == 0);
    REQUIRE(report.unreferenced_nodes == 0);
    REQUIRE(report.duplicate_nodes == 0);
    REQUIRE(report.warnings.empty());
    REQUIRE(mesh.node_sets().at("left").size() == 4);
    REQUIRE(mesh.node_sets().at("right").size() == 4);
    REQUIRE(mesh.node_sets().at("plate").size() ==
            static_cast<std::size_t>(mesh.num_nodes()));
    REQUIRE(mesh.element_sets().at("plate").size() ==
            static_cast<std::size_t>(mesh.num_elements()));
    REQUIRE(mesh.element_sets().count("left") == 0);
    for (Index n : mesh.node_sets().at("left")) REQUIRE(mesh.node(n).x() == 0.0);
    REQUIRE(report.quality.min > 0.1);
  }

  SECTION("3-D, MSH 4.1") {
    const Mesh original =
        make_perturbed_tet_mesh(box_spec(4, 2, 2, 0.4, 0.2, 0.2), 0.15, 5u);
    const std::vector<BoundaryGroup> groups = {
        {"fixed", [](const Vector3& c) { return c.x() < 1.0e-9; }},
        {"top", [](const Vector3& c) { return c.z() > 0.2 - 1.0e-9; }}};
    std::istringstream in(write_msh41(original, groups, "part"));
    MeshReadReport report;
    const Mesh mesh = read_gmsh(in, "part.msh", MeshReadOptions(), &report);
    require_same_mesh(mesh, original, 1.0e-15);
    REQUIRE(report.version == "4.1");
    REQUIRE(report.element_type == ElementType::Tet4);
    REQUIRE(report.boundary_elements == 2 * 2 * 2 + 2 * 4 * 2);
    REQUIRE(mesh.node_sets().at("fixed").size() == 9);
    REQUIRE(mesh.node_sets().at("top").size() == 15);
    REQUIRE(mesh.element_sets().at("part").size() ==
            static_cast<std::size_t>(mesh.num_elements()));
  }

  SECTION("MSH 2.2 hexahedra and quadrilaterals") {
    const Mesh hex = make_perturbed_hex_mesh(box_spec(3, 2, 2, 0.3, 0.2, 0.2), 0.2, 4u);
    std::istringstream in3(write_msh22(hex, {}, "block"));
    require_same_mesh(read_gmsh(in3, "block.msh"), hex, 1.0e-15);
    const Mesh quad = make_perturbed_quad_mesh(box_spec(4, 3, 1, 0.4, 0.3, 1.0), 0.2, 4u);
    std::istringstream in2(write_msh41(quad, {}, "sheet"));
    require_same_mesh(read_gmsh(in2, "sheet.msh"), quad, 1.0e-15);
  }
}

TEST_CASE("the Gmsh reader repairs orientation, drops unused nodes and handles units",
          "[unstructured][io][gmsh]") {
  // Unit square, two triangles; the second is listed clockwise, node 9 is a
  // geometry point no cell uses, and the edge x = 0 is a physical line.
  const std::string text = R"($MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
2
1 1 "support"
2 2 "sheet"
$EndPhysicalNames
$Nodes
5
1 0 0 0
2 1000 0 0
3 1000 1000 0
4 0 1000 0
9 500 500 0
$EndNodes
$Elements
4
1 15 2 0 1 9
2 1 2 1 4 4 1
3 2 2 2 1 1 2 3
4 2 2 2 1 1 4 3
$EndElements
)";
  {
    std::istringstream in(text);
    MeshReadReport report;
    const Mesh mesh = read_gmsh(in, "mm.msh", MeshReadOptions(), &report);
    REQUIRE(mesh.num_nodes() == 4);
    REQUIRE(mesh.num_elements() == 2);
    REQUIRE(report.reoriented == 1);
    REQUIRE(report.unreferenced_nodes == 1);
    REQUIRE(report.boundary_elements == 2);  // the point and the line
    for (Index e = 0; e < 2; ++e) REQUIRE(mesh.element_measure(e) == Approx(5.0e5));
    REQUIRE(mesh.node_sets().at("support") == std::vector<Index>({0, 3}));
    // 1.4 km across: the reader suggests millimetres.
    bool unit_warning = false;
    for (const std::string& w : report.warnings) {
      unit_warning = unit_warning || w.find("mesh.scale = 0.001") != std::string::npos;
    }
    REQUIRE(unit_warning);
  }
  {
    std::istringstream in(text);
    MeshReadOptions options;
    options.scale = 1.0e-3;
    MeshReadReport report;
    const Mesh mesh = read_gmsh(in, "mm.msh", options, &report);
    REQUIRE(mesh.bounding_box().upper.isApprox(Vector3(1.0, 1.0, 0.0)));
    REQUIRE(report.warnings.empty());
    REQUIRE(report.scale == 1.0e-3);
  }
}

TEST_CASE("coincident nodes are reported, and merged on request",
          "[unstructured][io][gmsh][diagnostics]") {
  // Two triangles meeting along the diagonal with their own copies of its
  // end nodes: a seam that leaves the triangles unconnected.
  const std::string text = R"($MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
6
1 0 0 0
2 1 0 0
3 1 1 0
4 0 0 0
5 1 1 0
6 0 1 0
$EndNodes
$Elements
2
1 2 2 0 1 1 2 3
2 2 2 0 1 4 5 6
$EndElements
)";
  {
    std::istringstream in(text);
    MeshReadReport report;
    const Mesh mesh = read_gmsh(in, "seam.msh", MeshReadOptions(), &report);
    REQUIRE(mesh.num_nodes() == 6);
    REQUIRE(report.duplicate_nodes == 2);
    REQUIRE_FALSE(report.duplicates_merged);
    REQUIRE(report.warnings.size() == 1);
    REQUIRE_THAT(report.warnings.front(), ContainsSubstring("merge_duplicate_nodes"));
  }
  {
    std::istringstream in(text);
    MeshReadOptions options;
    options.merge_duplicate_nodes = true;
    MeshReadReport report;
    const Mesh mesh = read_gmsh(in, "seam.msh", options, &report);
    REQUIRE(mesh.num_nodes() == 4);
    REQUIRE(report.duplicates_merged);
    REQUIRE(report.nodes_used == 4);
    REQUIRE(mesh.boundary_faces().size() == 4);
  }
  {
    // A tolerance larger than the cells collapses them: refused.
    std::istringstream in(text);
    MeshReadOptions options;
    options.merge_duplicate_nodes = true;
    options.duplicate_tolerance = 2.0;
    REQUIRE_THROWS_WITH(read_gmsh(in, "seam.msh", options),
                        ContainsSubstring("duplicate_tolerance"));
  }
}

TEST_CASE("malformed or unsupported Gmsh files fail with an actionable message",
          "[unstructured][io][gmsh][diagnostics]") {
  const std::string header = "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n";
  const std::string nodes =
      "$Nodes\n5\n1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n5 2 0 0\n$EndNodes\n";
  const auto read = [](const std::string& text) {
    std::istringstream in(text);
    return read_gmsh(in, "bad.msh");
  };
  // Mixed triangles and quadrilaterals.
  REQUIRE_THROWS_WITH(read(header + nodes +
                           "$Elements\n2\n1 3 2 0 1 1 2 3 4\n2 2 2 0 1 2 5 3\n$EndElements\n"),
                      ContainsSubstring("RecombineAll"));
  // Second-order triangles.
  REQUIRE_THROWS_WITH(
      read(header + nodes + "$Elements\n1\n1 9 2 0 1 1 2 3 4 5 1\n$EndElements\n"),
      ContainsSubstring("ElementOrder = 1"));
  // Binary files.
  REQUIRE_THROWS_WITH(read("$MeshFormat\n4.1 1 8\n$EndMeshFormat\n"),
                      ContainsSubstring("Mesh.Binary = 0"));
  // Unsupported version.
  REQUIRE_THROWS_WITH(read("$MeshFormat\n4.0 0 8\n$EndMeshFormat\n"),
                      ContainsSubstring("MshFileVersion"));
  // A plane mesh off the z = 0 plane.
  REQUIRE_THROWS_WITH(read(header +
                           "$Nodes\n3\n1 0 0 0\n2 1 0 0\n3 0 1 0.5\n$EndNodes\n"
                           "$Elements\n1\n1 2 2 0 1 1 2 3\n$EndElements\n"),
                      ContainsSubstring("z = 0"));
  // A cell referencing an undefined node.
  REQUIRE_THROWS_WITH(read(header + nodes + "$Elements\n1\n1 2 2 0 1 1 2 7\n$EndElements\n"),
                      ContainsSubstring("references node 7"));
  // Only boundary groups were saved.
  REQUIRE_THROWS_WITH(read(header + nodes + "$Elements\n1\n1 1 2 3 1 1 2\n$EndElements\n"),
                      ContainsSubstring("Mesh.SaveAll = 1"));
  // Truncated section.
  REQUIRE_THROWS_AS(read(header + "$Nodes\n3\n1 0 0 0\n"), IoError);
  // Not a Gmsh file at all.
  REQUIRE_THROWS_WITH(read("solid cube\nfacet normal 0 0 1\n"),
                      ContainsSubstring("Gmsh"));
  // A folded (self-intersecting) quadrilateral is not mistaken for a mirrored one.
  REQUIRE_THROWS_AS(read(header + nodes + "$Elements\n1\n1 3 2 0 1 1 3 2 4\n$EndElements\n"),
                    MeshError);
  REQUIRE_THROWS_WITH(read_mesh_file(std::string(kTmp) + "/does_not_exist.msh"),
                      ContainsSubstring("cannot open"));
  REQUIRE_THROWS_WITH(read_mesh_file("mesh.xyz"), ContainsSubstring("mesh.format"));
}

// ---------------------------------------------------------------------------
// Abaqus / CalculiX reader
// ---------------------------------------------------------------------------

TEST_CASE("SparLab's own CalculiX decks read back node for node",
          "[unstructured][io][abaqus][cross-validation]") {
  ensure_directory(kTmp);
  for (int dim : {2, 3}) {
    INFO("dim " << dim);
    const Mesh original =
        dim == 2 ? make_perturbed_tri_mesh(box_spec(5, 3, 1, 0.5, 0.3, 1.0), 0.2, 8u)
                 : make_perturbed_tet_mesh(box_spec(3, 2, 2, 0.3, 0.2, 0.2), 0.2, 8u);
    FemModel model(Mesh(original), default_material(), dim == 2 ? 0.01 : 1.0,
                   dim == 2 ? StressState::PlaneStress : StressState::ThreeDimensional,
                   IntegrationOptions());
    DisplacementConstraint root;
    Selector box;
    box.kind = SelectorKind::Box;
    box.xmax = 0.0;
    root.region.members.push_back(box);
    root.fix_x = root.fix_y = true;
    root.fix_z = dim == 3;
    model.constraints().push_back(root);
    LoadCaseSpec load;
    load.name = "tip";
    PointLoadSpec tip;
    Selector tip_node;
    tip_node.kind = SelectorKind::NearestNode;
    tip_node.point = Vector3(0.5, 0.0, 0.0);
    tip.region.members.push_back(tip_node);
    tip.force = Vector3(0.0, -100.0, 0.0);
    load.point_loads.push_back(tip);
    model.load_case_specs().push_back(load);
    model.finalize();
    REQUIRE(calculix_element_type(model) == (dim == 2 ? "CPS3" : "C3D4"));
    const std::vector<std::string> decks =
        write_calculix_decks(model, std::string(kTmp) + "/ccx" + std::to_string(dim), "rt");
    MeshReadReport report;
    const Mesh mesh = read_mesh_file(decks.front(), MeshReadOptions(), &report);
    require_same_mesh(mesh, original, 0.0);
    REQUIRE(report.format == "abaqus");
    REQUIRE(report.node_sets.at("NALL") == original.num_nodes());
    REQUIRE(report.element_sets.at("EALL") == original.num_elements());
    REQUIRE(report.ignored.count("*BOUNDARY cards") == 1);
    REQUIRE(report.ignored.count("*MATERIAL cards") == 1);
    bool warned = false;
    for (const std::string& w : report.warnings) {
      warned = warned || w.find("not imported") != std::string::npos;
    }
    REQUIRE(warned);
    std::remove(decks.front().c_str());
  }
}

TEST_CASE("the Abaqus reader handles includes, sets, GENERATE and continuation lines",
          "[unstructured][io][abaqus]") {
  const std::string dir = std::string(kTmp) + "/inp";
  ensure_directory(dir);
  // A 2 x 1 strip of CPS4 quads, the nodes in an included file, the second
  // element continued on a new line, and boundary T2D2 "elements" that name
  // the loaded edge.
  write_text(dir + "/nodes.inp",
             "** node file\n"
             "*NODE, NSET=ALLN\n"
             "101, 0.0, 0.0\n102, 1.0, 0.0\n103, 2.0, 0.0\n"
             "104, 0.0, 1.0\n105, 1.0, 1.0\n106, 2.0, 1.0\n");
  write_text(dir + "/main.inp",
             "*HEADING\nstrip\n"
             "*INCLUDE, INPUT=nodes.inp\n"
             "*ELEMENT, TYPE=CPS4R, ELSET=PLATE\n"
             "1, 101, 102, 105, 104\n"
             "2, 102, 103,\n"
             "   106, 105\n"
             "*ELEMENT, TYPE=T2D2, ELSET=TIPEDGE\n"
             "10, 103, 106\n"
             "*NSET, NSET=ROOT, GENERATE\n"
             "101, 104, 3\n"
             "*ELSET, ELSET=FIRST\n1\n"
             "*ELSET, ELSET=BOTH\nFIRST, 2\n"
             "*MATERIAL, NAME=STEEL\n*ELASTIC\n210000., 0.3\n"
             "*BOUNDARY\nROOT, 1, 2\n");
  MeshReadReport report;
  const Mesh mesh = read_mesh_file(dir + "/main.inp", MeshReadOptions(), &report);
  REQUIRE(mesh.element_type() == ElementType::Quad4);
  REQUIRE(mesh.num_nodes() == 6);
  REQUIRE(mesh.num_elements() == 2);
  REQUIRE(mesh.element_measure(1) == Approx(1.0));
  REQUIRE(mesh.node_sets().at("ROOT") == std::vector<Index>({0, 3}));
  REQUIRE(mesh.node_sets().at("TIPEDGE") == std::vector<Index>({2, 5}));
  REQUIRE(mesh.element_sets().at("PLATE") == std::vector<Index>({0, 1}));
  REQUIRE(mesh.element_sets().at("FIRST") == std::vector<Index>({0}));
  REQUIRE(mesh.element_sets().at("BOTH") == std::vector<Index>({0, 1}));
  REQUIRE(mesh.element_sets().count("TIPEDGE") == 0);
  REQUIRE(report.boundary_elements == 1);
  REQUIRE(report.ignored.at("*HEADING cards") == 1);
  REQUIRE(report.ignored.at("*BOUNDARY cards") == 1);

  // Keyword and data problems name the line.
  const auto read = [](const std::string& text) {
    std::istringstream in(text);
    return read_abaqus_inp(in, "bad.inp");
  };
  REQUIRE_THROWS_WITH(read("*NODE\n1, 0, 0\n*ELEMENT\n1, 1\n"), ContainsSubstring("TYPE="));
  REQUIRE_THROWS_WITH(read("1, 0, 0\n"), ContainsSubstring("bad.inp:1"));
  // A 20-node hexahedron is second order but not a tetrahedron; the message
  // names the one quadratic cell SparLab has.
  REQUIRE_THROWS_WITH(read("*NODE\n1, 0, 0\n2, 1, 0\n3, 0, 1\n*ELEMENT, TYPE=C3D20\n"
                           "1, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2\n"),
                      ContainsSubstring("10-node tetrahedron"));
  REQUIRE_THROWS_WITH(read("*NODE\n1, 0, 0\n2, 1, 0\n3, 0, 1\n*ELEMENT, TYPE=C3D6\n"
                           "1, 1, 2, 3, 1, 2, 3\n"),
                      ContainsSubstring("tetrahedra"));
  REQUIRE_THROWS_WITH(read("*NODE\n1, 0, 0\n*NSET, NSET=A\nB\n"),
                      ContainsSubstring("not defined before it"));
  REQUIRE_THROWS_WITH(read("*PART, NAME=A\n*NODE\n1, 0, 0\n*END PART\n*PART, NAME=B\n"
                           "*NODE\n1, 0, 0\n*END PART\n"),
                      ContainsSubstring("flat"));
  REQUIRE_THROWS_WITH(read("*NODE\n1, 0, 0\n*INSTANCE, NAME=I, PART=A\n1.0, 0.0, 0.0\n"),
                      ContainsSubstring("translation"));
  REQUIRE_THROWS_WITH(read("*NODE\n1, 0, 0\n*INCLUDE, INPUT=missing.inp\n"),
                      ContainsSubstring("cannot open included file"));
}

// ---------------------------------------------------------------------------
// Groups in selectors, sub-meshes and decks
// ---------------------------------------------------------------------------

TEST_CASE("mesh groups drive selectors",
          "[unstructured][selector][mesh]") {
  const Mesh original = make_structured_tri_mesh(box_spec(4, 2, 1, 0.4, 0.2, 1.0));
  const std::vector<BoundaryGroup> groups = {
      {"root", [](const Vector3& c) { return c.x() < 1.0e-9; }},
      {"tip", [](const Vector3& c) { return c.x() > 0.4 - 1.0e-9; }}};
  std::istringstream in(write_msh22(original, groups, "body"));
  Mesh mesh = read_gmsh(in, "body.msh");

  SelectorGroup root;
  root.name = "root_bc";
  Selector group;
  group.kind = SelectorKind::Group;
  group.group = "root";
  root.members.push_back(group);
  REQUIRE(root.select_nodes(mesh) == mesh.node_sets().at("root"));
  REQUIRE_THROWS_WITH(root.select_elements(mesh), ContainsSubstring("node set"));

  SelectorGroup body;
  Selector body_sel;
  body_sel.kind = SelectorKind::Group;
  body_sel.group = "body";
  body.members.push_back(body_sel);
  REQUIRE(body.select_elements(mesh).size() == static_cast<std::size_t>(mesh.num_elements()));

  SelectorGroup missing;
  missing.name = "load";
  Selector wrong;
  wrong.kind = SelectorKind::Group;
  wrong.group = "Tip";
  missing.members.push_back(wrong);
  REQUIRE_THROWS_WITH(missing.select_nodes(mesh), ContainsSubstring("'tip' (3)"));

  // Groups combine with geometric primitives.
  SelectorGroup mixed;
  Selector corner;
  corner.kind = SelectorKind::NearestNode;
  corner.point = Vector3(0.2, 0.2, 0.0);
  mixed.members = {group, corner};
  REQUIRE(mixed.select_nodes(mesh).size() == 4);
  REQUIRE_THROWS_AS(mesh.set_node_set("bad", {0, 1000}), MeshError);
}

TEST_CASE("a deck reads its mesh from a file relative to itself and addresses groups",
          "[unstructured][io][config]") {
  const std::string dir = std::string(kTmp) + "/deck";
  ensure_directory(dir + "/meshes");
  const Mesh original =
      make_perturbed_tri_mesh(box_spec(12, 3, 1, 0.6, 0.15, 1.0), 0.15, 17u);
  const std::vector<BoundaryGroup> groups = {
      {"clamp", [](const Vector3& c) { return c.x() < 1.0e-9; }},
      {"tip", [](const Vector3& c) { return c.x() > 0.6 - 1.0e-9; }}};
  write_text(dir + "/meshes/beam.msh", write_msh41(original, groups, "beam"));
  write_text(dir + "/beam.json", R"({
    "name": "file_beam",
    "mesh": { "type": "file", "path": "meshes/beam.msh" },
    "material": { "youngs_modulus": 70e9, "poisson_ratio": 0.3, "density": 2700 },
    "model": { "thickness": 0.01 },
    "boundary_conditions": [ { "fix": ["x", "y"], "region": { "group": "clamp" } } ],
    "load_cases": [ { "name": "tip", "tractions": [
        { "region": { "group": "tip" }, "traction": [0, -1e6] } ] } ]
  })");
  const Configuration config = load_configuration(dir + "/beam.json", true);
  REQUIRE(config.mesh_kind == MeshKind::File);
  REQUIRE_FALSE(is_structured(config.mesh_kind));
  REQUIRE(config.dim() == 2);
  REQUIRE(config.mesh_report.cells == original.num_elements());
  REQUIRE_THAT(config.mesh_file.resolved_path, ContainsSubstring("deck/meshes/beam.msh"));
  REQUIRE_THAT(config.describe_mesh(), ContainsSubstring("Tri3"));

  FemModel model = build_model(config);
  require_same_mesh(model.mesh(), original, 1.0e-15);
  Assembler assembler(model);
  StaticAnalysis analysis(model, assembler, StaticAnalysisOptions());
  const StaticSolution sol = analysis.solve_all().front();
  REQUIRE(sol.equilibrium.applied_force.isApprox(Vector3(0.0, -1.0e6 * 0.15 * 0.01, 0.0),
                                                 1.0e-12));
  REQUIRE(sol.equilibrium.relative_force_error < 1.0e-10);

  // Resolution keys and a missing file are rejected with the fix.
  const auto parse = [&](const std::string& mesh_block) {
    const json::Value doc = json::parse(
        R"({ "mesh": )" + mesh_block + R"(,
             "material": { "youngs_modulus": 70e9, "poisson_ratio": 0.3 },
             "boundary_conditions": [ { "fix": ["x", "y"], "region": { "all": true } } ],
             "load_cases": [ { "name": "c", "prescribed_displacement_only": true } ] })");
    return parse_configuration(doc, "inline", true, dir);
  };
  REQUIRE_THROWS_WITH(parse(R"({ "type": "file", "path": "meshes/beam.msh", "nx": 4 })"),
                      ContainsSubstring("mesher"));
  REQUIRE_THROWS_WITH(parse(R"({ "type": "file", "path": "meshes/none.msh" })"),
                      ContainsSubstring("mesh.path"));
  REQUIRE_THROWS_WITH(parse(R"({ "type": "file", "path": "meshes/beam.msh", "scale": 0 })"),
                      ContainsSubstring("mesh.scale"));
  REQUIRE_THROWS_WITH(parse(R"({ "type": "voronoi" })"), ContainsSubstring("structured_tet"));
  REQUIRE(parse(R"({ "type": "structured_tri", "nx": 4, "ny": 2, "lx": 1, "ly": 0.5 })")
              .dim() == 2);
  const Configuration tet = parse(
      R"({ "type": "structured_tet", "nx": 2, "ny": 2, "nz": 2, "lx": 1, "ly": 1, "lz": 1 })");
  REQUIRE(tet.dim() == 3);
  REQUIRE(build_mesh(tet).num_elements() == 48);
  REQUIRE_THROWS_WITH(parse(R"({ "type": "structured_tri", "nx": 4, "ny": 2, "nz": 2,
                                 "lx": 1, "ly": 0.5 })"),
                      ContainsSubstring("structured_tet"));
}
