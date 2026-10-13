#include "sparlab/mesh/Mesh.hpp"

#include "sparlab/core/Exceptions.hpp"
#include "sparlab/core/Logging.hpp"
#include "sparlab/elements/Hex8.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <utility>

namespace sparlab {

int nodes_per_element(ElementType type) {
  switch (type) {
    case ElementType::Hex8: return 8;
  }
  throw MeshError("unhandled element type");
}

int element_dimension(ElementType type) {
  switch (type) {
    case ElementType::Hex8: return 3;
  }
  throw MeshError("unhandled element type");
}

int face_corner_nodes(ElementType type) {
  switch (type) {
    case ElementType::Hex8: return 4;
  }
  throw MeshError("unhandled element type");
}

std::string to_string(ElementType type) {
  switch (type) {
    case ElementType::Hex8: return "Hex8";
  }
  return "Unknown";
}

const std::vector<std::vector<int>>& element_local_faces(ElementType type) {
  // The Hex8 faces follow the VTK hexahedron convention (bottom, top, front,
  // right, back, left), each wound so that its right-hand normal points out
  // of a positively oriented element.
  static const std::vector<std::vector<int>> hex_faces = {
      {0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}};
  switch (type) {
    case ElementType::Hex8: return hex_faces;
  }
  throw MeshError("unhandled element type");
}

Mesh::Mesh(Matrix coords, std::vector<Index> connectivity, ElementType type)
    : coords_(std::move(coords)),
      connectivity_(std::move(connectivity)),
      type_(type),
      nodes_per_element_(nodes_per_element(type)),
      dim_(element_dimension(type)) {
  if (coords_.rows() != dim_) {
    std::ostringstream os;
    os << "coordinate array has " << coords_.rows() << " rows but " << to_string(type)
       << " elements live in " << dim_ << " dimensions";
    throw MeshError(os.str());
  }
  if (connectivity_.size() % static_cast<std::size_t>(nodes_per_element_) != 0) {
    std::ostringstream os;
    os << "connectivity length " << connectivity_.size() << " is not a multiple of "
       << nodes_per_element_ << " nodes per " << to_string(type) << " element";
    throw MeshError(os.str());
  }
}

Matrix Mesh::element_coordinates(Index e) const {
  Matrix xe(dim_, nodes_per_element_);
  const Index* nodes = element_nodes(e);
  for (int a = 0; a < nodes_per_element_; ++a) xe.col(a) = coords_.col(nodes[a]);
  return xe;
}

Vector3 Mesh::element_centroid(Index e) const {
  Vector3 c = Vector3::Zero();
  const Index* nodes = element_nodes(e);
  for (int a = 0; a < nodes_per_element_; ++a) c.head(dim_) += coords_.col(nodes[a]);
  return c / static_cast<Scalar>(nodes_per_element_);
}

Scalar Mesh::element_measure(Index e) const {
  Scalar min_det = 0.0;
  return hex8_volume(element_coordinates(e), &min_det);
}

MeshQuality Mesh::quality() const {
  MeshQuality q;
  const Index ne = num_elements();
  if (ne == 0) return q;
  static const int hex_corner[8][3] = {{1, 3, 4}, {2, 0, 5}, {3, 1, 6}, {0, 2, 7},
                                       {7, 5, 0}, {4, 6, 1}, {5, 7, 2}, {6, 4, 3}};
  q.metric = "minimum scaled Jacobian at the corners";
  q.min = std::numeric_limits<Scalar>::max();
  Scalar sum = 0.0;
  for (Index e = 0; e < ne; ++e) {
    const Index* n = element_nodes(e);
    Scalar value = std::numeric_limits<Scalar>::max();
    for (int a = 0; a < 8; ++a) {
      const Vector3 p = coords_.col(n[a]);
      const Vector3 e1 = Vector3(coords_.col(n[hex_corner[a][0]])) - p;
      const Vector3 e2 = Vector3(coords_.col(n[hex_corner[a][1]])) - p;
      const Vector3 e3 = Vector3(coords_.col(n[hex_corner[a][2]])) - p;
      const Scalar denom = e1.norm() * e2.norm() * e3.norm();
      const Scalar sj = denom > 0.0 ? e1.dot(e2.cross(e3)) / denom : -1.0;
      value = std::min(value, sj);
    }
    sum += value;
    if (value < q.min) {
      q.min = value;
      q.worst_element = e;
    }
    if (value < q.poor_threshold) ++q.poor_elements;
  }
  q.mean = sum / static_cast<Scalar>(ne);
  return q;
}

void Mesh::set_node_set(const std::string& name, std::vector<Index> nodes) {
  if (name.empty()) throw MeshError("a node set needs a non-empty name");
  std::sort(nodes.begin(), nodes.end());
  nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
  if (!nodes.empty() && (nodes.front() < 0 || nodes.back() >= num_nodes())) {
    std::ostringstream os;
    os << "node set '" << name << "' references node "
       << (nodes.front() < 0 ? nodes.front() : nodes.back()) << " outside [0, "
       << num_nodes() - 1 << "]";
    throw MeshError(os.str());
  }
  node_sets_[name] = std::move(nodes);
}

void Mesh::set_element_set(const std::string& name, std::vector<Index> elements) {
  if (name.empty()) throw MeshError("an element set needs a non-empty name");
  std::sort(elements.begin(), elements.end());
  elements.erase(std::unique(elements.begin(), elements.end()), elements.end());
  if (!elements.empty() && (elements.front() < 0 || elements.back() >= num_elements())) {
    std::ostringstream os;
    os << "element set '" << name << "' references element "
       << (elements.front() < 0 ? elements.front() : elements.back()) << " outside [0, "
       << num_elements() - 1 << "]";
    throw MeshError(os.str());
  }
  element_sets_[name] = std::move(elements);
}

BoundingBox Mesh::bounding_box() const {
  if (num_nodes() == 0) throw MeshError("bounding box requested for an empty mesh");
  BoundingBox bb;
  for (int i = 0; i < dim_; ++i) {
    bb.lower(i) = coords_.row(i).minCoeff();
    bb.upper(i) = coords_.row(i).maxCoeff();
  }
  return bb;
}

void Mesh::validate() const {
  if (num_nodes() == 0) throw MeshError("mesh contains no nodes");
  if (num_elements() == 0) throw MeshError("mesh contains no elements");
  if (!coords_.allFinite()) throw MeshError("nodal coordinates contain NaN or Inf");

  const Index nn = num_nodes();
  for (std::size_t k = 0; k < connectivity_.size(); ++k) {
    const Index node = connectivity_[k];
    if (node < 0 || node >= nn) {
      std::ostringstream os;
      os << "element " << (k / nodes_per_element_) << " references node " << node
         << " which is outside the valid range [0, " << nn - 1 << "]";
      throw MeshError(os.str());
    }
  }

  const Index ne = num_elements();
  const char* unit = " m^3";
  Scalar min_measure = std::numeric_limits<Scalar>::max();
  Scalar max_measure = 0.0;
  for (Index e = 0; e < ne; ++e) {
    // Duplicated node within one element => degenerate topology.
    const Index* nodes = element_nodes(e);
    for (int a = 0; a < nodes_per_element_; ++a) {
      for (int b = a + 1; b < nodes_per_element_; ++b) {
        if (nodes[a] == nodes[b]) {
          std::ostringstream os;
          os << "element " << e << " uses node " << nodes[a] << " at local positions "
             << a << " and " << b << " (collapsed/degenerate element)";
          throw MeshError(os.str());
        }
      }
    }
    // A hexahedron can have positive volume and still be folded at a
    // corner, so the Jacobian is checked at every integration point rather
    // than only through the volume.
    Scalar min_det = 0.0;
    const Scalar volume = hex8_volume(element_coordinates(e), &min_det);
    if (!(min_det > 0.0) || !(volume > 0.0)) {
      std::ostringstream os;
      os << "element " << e << " has volume " << volume
         << " m^3 and a minimum Jacobian determinant of " << min_det
         << " m^3 over its 2 x 2 x 2 Gauss points; nodes must follow the VTK "
            "hexahedron ordering (bottom face counter-clockwise seen from +z, then "
            "the top face) and the element must not be inverted or folded";
      throw MeshError(os.str());
    }
    min_measure = std::min(min_measure, volume);
    max_measure = std::max(max_measure, volume);
  }

  if (max_measure / min_measure > 1.0e8) {
    log::warn("extreme element volume ratio ",
              max_measure / min_measure, " (min ", min_measure, unit, ", max ",
              max_measure, unit, "); conditioning of the stiffness matrix may suffer");
  }

  // Nodes referenced by no element contribute empty rows/columns to K and are a
  // frequent cause of singular systems.
  std::vector<char> used(static_cast<std::size_t>(nn), 0);
  for (Index node : connectivity_) used[static_cast<std::size_t>(node)] = 1;
  const auto orphans = std::count(used.begin(), used.end(), 0);
  if (orphans > 0) {
    std::ostringstream os;
    os << orphans << " of " << nn
       << " nodes are not referenced by any element; their degrees of freedom would "
          "make the stiffness matrix singular";
    throw MeshError(os.str());
  }
}

std::vector<Mesh::BoundaryFace> Mesh::boundary_faces() const {
  const std::vector<std::vector<int>>& local = element_local_faces(type_);
  const std::size_t corners = static_cast<std::size_t>(face_corner_nodes(type_));

  // Faces are keyed by their sorted corner nodes, so orientation does not
  // matter (the corners alone identify a Tet10 face).
  // Sorting one flat array of keys is far cheaper than a map of vectors on a
  // large mesh, and the boundary comes out in ascending key order - the order
  // the earlier map-based search produced, which fixes the summation order of
  // traction loads and the triangle order of an STL file.
  struct Record {
    std::array<Index, 4> key;
    Index element;
    int local_face;
  };
  const Index ne = num_elements();
  // Sized up front and filled by index: besides saving the reallocations,
  // this keeps GCC from assuming a one-record buffer inside std::sort and
  // raising a spurious -Warray-bounds.
  std::vector<Record> records(static_cast<std::size_t>(ne) * local.size());
  std::size_t next = 0;
  for (Index e = 0; e < ne; ++e) {
    const Index* nodes = element_nodes(e);
    for (std::size_t lf = 0; lf < local.size(); ++lf) {
      Record& r = records[next++];
      r.key.fill(-1);
      for (std::size_t a = 0; a < corners; ++a) r.key[a] = nodes[local[lf][a]];
      std::sort(r.key.begin(), r.key.begin() + static_cast<long>(corners));
      r.element = e;
      r.local_face = static_cast<int>(lf);
    }
  }
  std::sort(records.begin(), records.end(), [](const Record& a, const Record& b) {
    if (a.key != b.key) return a.key < b.key;
    if (a.element != b.element) return a.element < b.element;
    return a.local_face < b.local_face;
  });

  std::vector<BoundaryFace> boundary;
  std::size_t i = 0;
  while (i < records.size()) {
    std::size_t j = i + 1;
    while (j < records.size() && records[j].key == records[i].key) ++j;
    if (j - i == 1) {
      BoundaryFace face;
      face.element = records[i].element;
      face.local_face = records[i].local_face;
      const Index* nodes = element_nodes(face.element);
      for (int a : local[static_cast<std::size_t>(face.local_face)]) {
        face.nodes.push_back(nodes[a]);
      }
      boundary.push_back(std::move(face));
    }
    i = j;
  }
  return boundary;
}

Scalar Mesh::face_measure(const BoundaryFace& face) const {
  if (face.nodes.size() == 4 && dim_ == 3) {
    Matrix xf(3, 4);
    for (int a = 0; a < 4; ++a) xf.col(a) = coords_.col(face.nodes[static_cast<std::size_t>(a)]);
    return hex8_face_area(xf);
  }
  throw MeshError("face_measure received a face with an unsupported node count");
}

}  // namespace sparlab
