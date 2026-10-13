/// \file test_mesh.cpp
/// \brief Mesh generation, validation diagnostics, boundary edges, sub-meshes.
#include "sparlab/core/Exceptions.hpp"
#include "sparlab/fem/Selector.hpp"
#include "sparlab/mesh/StructuredMesh.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <set>

using namespace sparlab;
using Catch::Approx;

TEST_CASE("structured generator produces the documented numbering", "[mesh]") {
  StructuredMeshSpec spec;
  spec.nx = 3;
  spec.ny = 2;
  spec.lx = 3.0;
  spec.ly = 2.0;
  const Mesh mesh = make_structured_quad_mesh(spec);

  REQUIRE(mesh.num_nodes() == 12);
  REQUIRE(mesh.num_elements() == 6);
  REQUIRE(mesh.nodes_per_elem() == 4);

  // node(i, j) = j * (nx + 1) + i
  REQUIRE(mesh.dim() == 2);
  REQUIRE(mesh.node(0).isApprox(Vector3(0.0, 0.0, 0.0)));
  REQUIRE(mesh.node(3).isApprox(Vector3(3.0, 0.0, 0.0)));
  REQUIRE(mesh.node(4).isApprox(Vector3(0.0, 1.0, 0.0)));
  REQUIRE(mesh.node(11).isApprox(Vector3(3.0, 2.0, 0.0)));

  // element(0) is the lower-left cell with counter-clockwise nodes 0,1,5,4
  const Index* e0 = mesh.element_nodes(0);
  REQUIRE(e0[0] == 0);
  REQUIRE(e0[1] == 1);
  REQUIRE(e0[2] == 5);
  REQUIRE(e0[3] == 4);

  for (Index e = 0; e < mesh.num_elements(); ++e) {
    REQUIRE(mesh.element_measure(e) == Approx(1.0));
  }
  REQUIRE(mesh.element_centroid(0).isApprox(Vector3(0.5, 0.5, 0.0)));

  const BoundingBox bb = mesh.bounding_box();
  REQUIRE(bb.lower.x() == Approx(0.0));
  REQUIRE(bb.lower.y() == Approx(0.0));
  REQUIRE(bb.upper.x() == Approx(3.0));
  REQUIRE(bb.upper.y() == Approx(2.0));
  REQUIRE(bb.extent().z() == Approx(0.0));

  REQUIRE(mesh.structured_info().has_value());
  REQUIRE(mesh.structured_info()->uniform);
  REQUIRE(structured_node_index(*mesh.structured_info(), 2, 1) == 6);
  REQUIRE(structured_element_index(*mesh.structured_info(), 2, 1) == 5);
}

TEST_CASE("structured generator rejects degenerate specifications",
          "[mesh][diagnostics]") {
  StructuredMeshSpec spec;
  spec.nx = 0;
  REQUIRE_THROWS_AS(make_structured_quad_mesh(spec), ConfigError);

  spec.nx = 2;
  spec.ny = -1;
  REQUIRE_THROWS_AS(make_structured_quad_mesh(spec), ConfigError);

  spec.ny = 2;
  spec.lx = 0.0;
  REQUIRE_THROWS_AS(make_structured_quad_mesh(spec), ConfigError);

  spec.lx = 1.0;
  spec.ly = -1.0;
  REQUIRE_THROWS_AS(make_structured_quad_mesh(spec), ConfigError);
}

TEST_CASE("mesh validation catches invalid connectivity and geometry",
          "[mesh][diagnostics]") {
  Matrix coords(2, 4);
  coords << 0, 1, 1, 0,
            0, 0, 1, 1;

  SECTION("out-of-range node index") {
    const Mesh mesh(coords, {0, 1, 2, 9}, ElementType::Quad4);
    REQUIRE_THROWS_AS(mesh.validate(), MeshError);
  }

  SECTION("clockwise ordering gives a negative area") {
    const Mesh mesh(coords, {0, 3, 2, 1}, ElementType::Quad4);
    REQUIRE_THROWS_AS(mesh.validate(), MeshError);
  }

  SECTION("repeated node inside one element") {
    const Mesh mesh(coords, {0, 1, 1, 3}, ElementType::Quad4);
    REQUIRE_THROWS_AS(mesh.validate(), MeshError);
  }

  SECTION("connectivity length is not a multiple of the stride") {
    REQUIRE_THROWS_AS(Mesh(coords, {0, 1, 2}, ElementType::Quad4), MeshError);
  }

  SECTION("orphan nodes are reported") {
    Matrix five(2, 5);
    five << 0, 1, 1, 0, 5,
            0, 0, 1, 1, 5;
    const Mesh mesh(five, {0, 1, 2, 3}, ElementType::Quad4);
    REQUIRE_THROWS_AS(mesh.validate(), MeshError);
  }
}

TEST_CASE("boundary edges are the edges owned by one element", "[mesh]") {
  StructuredMeshSpec spec;
  spec.nx = 3;
  spec.ny = 2;
  const Mesh mesh = make_structured_quad_mesh(spec);
  const std::vector<Mesh::BoundaryFace> edges = mesh.boundary_faces();

  // A 3 x 2 grid has 2*(3+2) = 10 boundary edges whose lengths add up to the
  // perimeter of the unit square.
  REQUIRE(edges.size() == 10);
  Scalar perimeter = 0.0;
  for (const Mesh::BoundaryFace& edge : edges) {
    REQUIRE(edge.nodes.size() == 2);
    REQUIRE(edge.element >= 0);
    REQUIRE(edge.element < mesh.num_elements());
    REQUIRE(edge.local_face >= 0);
    REQUIRE(edge.local_face < 4);
    perimeter += mesh.face_measure(edge);
  }
  REQUIRE(perimeter == Approx(2.0 * (spec.lx + spec.ly)));
}

TEST_CASE("perturbed mesh keeps the boundary and stays valid", "[mesh]") {
  StructuredMeshSpec spec;
  spec.nx = 5;
  spec.ny = 4;
  spec.lx = 2.0;
  spec.ly = 1.0;
  const Mesh uniform = make_structured_quad_mesh(spec);
  const Mesh perturbed = make_perturbed_quad_mesh(spec, 0.3, 42u);

  REQUIRE(perturbed.num_nodes() == uniform.num_nodes());
  REQUIRE(perturbed.num_elements() == uniform.num_elements());
  REQUIRE_FALSE(perturbed.structured_info()->uniform);
  REQUIRE_NOTHROW(perturbed.validate());

  // Total area is preserved exactly: only interior nodes moved.
  Scalar area_uniform = 0.0;
  Scalar area_perturbed = 0.0;
  for (Index e = 0; e < uniform.num_elements(); ++e) {
    area_uniform += uniform.element_measure(e);
    area_perturbed += perturbed.element_measure(e);
  }
  REQUIRE(area_perturbed == Approx(area_uniform));

  // The bounding box is unchanged.
  REQUIRE(perturbed.bounding_box().lower.isApprox(uniform.bounding_box().lower));
  REQUIRE(perturbed.bounding_box().upper.isApprox(uniform.bounding_box().upper));

  // Deterministic for a fixed seed.
  const Mesh again = make_perturbed_quad_mesh(spec, 0.3, 42u);
  REQUIRE(again.coordinates().isApprox(perturbed.coordinates()));

  REQUIRE_THROWS_AS(make_perturbed_quad_mesh(spec, 0.5), ConfigError);
  REQUIRE_THROWS_AS(make_perturbed_quad_mesh(spec, -0.1), ConfigError);
}

TEST_CASE("selectors pick the expected nodes and elements", "[selector]") {
  StructuredMeshSpec spec;
  spec.nx = 4;
  spec.ny = 4;
  spec.lx = 1.0;
  spec.ly = 1.0;
  const Mesh mesh = make_structured_quad_mesh(spec);

  SECTION("degenerate box selects a line of nodes") {
    SelectorGroup group;
    group.name = "left";
    Selector box;
    box.kind = SelectorKind::Box;
    box.xmax = 0.0;
    group.members.push_back(box);
    REQUIRE(group.select_nodes(mesh).size() == 5);
  }

  SECTION("circle selects elements by centroid") {
    SelectorGroup group;
    group.name = "hole";
    Selector circle;
    circle.kind = SelectorKind::Circle;
    circle.center = Vector3(0.5, 0.5, 0.0);
    circle.radius = 0.2;
    group.members.push_back(circle);
    const std::vector<Index> elements = group.select_elements(mesh);
    REQUIRE(elements.size() == 4);  // the four cells touching the centre
    for (Index e : elements) {
      REQUIRE((mesh.element_centroid(e) - circle.center).norm() <= 0.2 + 1.0e-9);
    }
  }

  SECTION("annulus excludes the inner disc") {
    SelectorGroup group;
    Selector annulus;
    annulus.kind = SelectorKind::Annulus;
    annulus.center = Vector3(0.5, 0.5, 0.0);
    annulus.inner_radius = 0.2;
    annulus.radius = 0.4;
    group.members.push_back(annulus);
    const std::vector<Index> elements = group.select_elements(mesh);
    REQUIRE_FALSE(elements.empty());
    for (Index e : elements) {
      const Scalar r = (mesh.element_centroid(e) - annulus.center).norm();
      REQUIRE(r >= 0.2 - 1.0e-9);
      REQUIRE(r <= 0.4 + 1.0e-9);
    }
  }

  SECTION("union of two boxes and inversion") {
    SelectorGroup group;
    Selector left;
    left.kind = SelectorKind::Box;
    left.xmax = 0.0;
    Selector right;
    right.kind = SelectorKind::Box;
    right.xmin = 1.0;
    group.members = {left, right};
    REQUIRE(group.select_nodes(mesh).size() == 10);

    group.invert = true;
    REQUIRE(static_cast<Index>(group.select_nodes(mesh).size()) == mesh.num_nodes() - 10);
  }

  SECTION("nearest node picks exactly one") {
    SelectorGroup group;
    Selector nearest;
    nearest.kind = SelectorKind::NearestNode;
    nearest.point = Vector3(0.99, 0.01, 0.0);
    group.members.push_back(nearest);
    const std::vector<Index> nodes = group.select_nodes(mesh);
    REQUIRE(nodes.size() == 1);
    REQUIRE(mesh.node(nodes.front()).isApprox(Vector3(1.0, 0.0, 0.0)));
  }

  SECTION("explicit ids are range-checked") {
    SelectorGroup group;
    group.name = "bad";
    Selector ids;
    ids.kind = SelectorKind::NodeIds;
    ids.ids = {0, 1, 9999};
    group.members.push_back(ids);
    REQUIRE_THROWS_AS(group.select_nodes(mesh), ConfigError);
  }

  SECTION("node-only selectors are rejected for element regions") {
    SelectorGroup group;
    Selector nearest;
    nearest.kind = SelectorKind::NearestNode;
    nearest.point = Vector3::Zero();
    group.members.push_back(nearest);
    REQUIRE_THROWS_AS(group.select_elements(mesh), ConfigError);
  }
}
