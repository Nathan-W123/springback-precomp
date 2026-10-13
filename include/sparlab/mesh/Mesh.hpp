/// \file Mesh.hpp
/// \brief Storage-only mesh container plus validation.
///
/// The mesh owns nodal coordinates, element connectivity and optional named
/// node and element sets, and nothing else: materials, loads and degrees of
/// freedom live in separate components. Its cells are Hex8 (coordinates are
/// 3 x n); the dimension is a runtime property read from the coordinate
/// array, and every consumer keys on `dim()` rather than on a compile-time
/// constant. The connectivity is stored as a flat array with an explicit
/// stride.
///
/// Named sets come from mesh files (Abaqus/CalculiX *NSET / *ELSET) and let a
/// deck address a part of the mesh by name where a geometric box would not
/// fit.
#pragma once

#include "sparlab/core/Types.hpp"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace sparlab {

/// Supported element topologies.
enum class ElementType {
  Hex8  ///< Eight-node trilinear isoparametric hexahedron (3-D).
};

/// Number of nodes carried by an element topology.
int nodes_per_element(ElementType type);

/// Spatial dimension of an element topology (3).
int element_dimension(ElementType type);

/// Number of corner nodes on one boundary face (4 for the quadrilateral face
/// of a Hex8); they lead each face list of `element_local_faces`, and they
/// alone identify a face shared by two cells.
int face_corner_nodes(ElementType type);

/// Human-readable name, also used in VTK output.
std::string to_string(ElementType type);

/// Local node lists of the boundary faces of a topology, each wound so its
/// right-hand normal points out of the element: the six quadrilateral faces
/// of a Hex8 (VTK hexahedron face convention).
const std::vector<std::vector<int>>& element_local_faces(ElementType type);

/// Summary of element shape quality. The metric is 1 for an ideal cell and
/// falls towards 0 as the cell degenerates: the minimum scaled Jacobian over
/// the corners of a Hex8 (angle distortion).
struct MeshQuality {
  std::string metric;
  Scalar min = 0.0;
  Scalar mean = 0.0;
  Index worst_element = -1;
  Index poor_elements = 0;   ///< elements below `poor_threshold`
  Scalar poor_threshold = 0.1;
};

/// Metadata retained when a mesh came from the structured generator. Presence
/// of this record allows the assembler to reuse a single element stiffness
/// matrix for the whole grid (all cells are geometrically identical).
struct StructuredGridInfo {
  Index nx = 0;       ///< elements along x
  Index ny = 0;       ///< elements along y
  Index nz = 0;       ///< elements along z (0 for a 2-D grid)
  Scalar lx = 0.0;    ///< domain length along x [m]
  Scalar ly = 0.0;    ///< domain length along y [m]
  Scalar lz = 0.0;    ///< domain length along z [m] (0 for a 2-D grid)
  bool uniform = false;  ///< true when every cell is the same box
};

/// Axis-aligned bounding box. In 2-D the z entries are zero.
struct BoundingBox {
  Vector3 lower = Vector3::Zero();  ///< [m]
  Vector3 upper = Vector3::Zero();  ///< [m]
  Vector3 extent() const { return upper - lower; }
};

/// Node coordinates + element connectivity for a 2-D or 3-D mesh.
class Mesh {
 public:
  Mesh() = default;

  /// \param coords dim x num_nodes matrix of nodal coordinates [m], with dim
  ///        matching the element type (3).
  /// \param connectivity flat, row-major connectivity with
  ///        `nodes_per_element(type)` entries per element, in the VTK
  ///        hexahedron ordering.
  Mesh(Matrix coords, std::vector<Index> connectivity, ElementType type);

  Index num_nodes() const { return static_cast<Index>(coords_.cols()); }
  Index num_elements() const {
    return static_cast<Index>(connectivity_.size()) / nodes_per_element_;
  }
  int nodes_per_elem() const { return nodes_per_element_; }
  /// Spatial dimension: 2 or 3.
  int dim() const { return dim_; }
  ElementType element_type() const { return type_; }

  /// Coordinates of node `n` [m]. In 2-D the z component is zero, so callers
  /// that only need the plane can read `.x()` and `.y()` unchanged.
  Vector3 node(Index n) const {
    Vector3 x = Vector3::Zero();
    x.head(dim_) = coords_.col(n);
    return x;
  }

  /// dim x num_nodes coordinate array [m].
  const Matrix& coordinates() const { return coords_; }

  /// Global node indices of element `e` (pointer to `nodes_per_elem()` values).
  const Index* element_nodes(Index e) const {
    return connectivity_.data() + static_cast<std::size_t>(e) * nodes_per_element_;
  }

  const std::vector<Index>& connectivity() const { return connectivity_; }

  /// dim x nodes_per_elem matrix of the element's nodal coordinates [m].
  Matrix element_coordinates(Index e) const;

  /// Geometric centroid of element `e` (average of its nodes) [m]; z = 0 in 2-D.
  Vector3 element_centroid(Index e) const;

  /// Measure of element `e`: its signed volume [m^3] from the trilinear
  /// Jacobian (positive for correctly ordered cells).
  Scalar element_measure(Index e) const;

  /// Shape-quality statistics of every element (see MeshQuality).
  MeshQuality quality() const;

  /// Axis-aligned bounding box of the mesh [m].
  BoundingBox bounding_box() const;

  void set_structured_info(const StructuredGridInfo& info) { structured_ = info; }
  const std::optional<StructuredGridInfo>& structured_info() const { return structured_; }

  /// Validate connectivity ranges, element measures and node degeneracy.
  /// \throws MeshError with an actionable message on the first problem found.
  void validate() const;

  /// Named node sets (ascending, unique indices), e.g. from a mesh file.
  const std::map<std::string, std::vector<Index>>& node_sets() const { return node_sets_; }
  /// Named element sets (ascending, unique indices).
  const std::map<std::string, std::vector<Index>>& element_sets() const {
    return element_sets_;
  }
  /// Register a named set; indices are sorted and de-duplicated.
  /// \throws MeshError for an out-of-range index or an empty name.
  void set_node_set(const std::string& name, std::vector<Index> nodes);
  void set_element_set(const std::string& name, std::vector<Index> elements);

  /// Boundary face of the mesh, referenced by exactly one element, with its
  /// nodes in the element's local order.
  struct BoundaryFace {
    std::vector<Index> nodes;
    Index element = 0;
    int local_face = 0;  ///< local face id within the element
  };

  /// Faces referenced by exactly one element, i.e. the mesh boundary.
  /// Computed on demand; O(num_elements * faces per element * log).
  std::vector<BoundaryFace> boundary_faces() const;

 private:
  Matrix coords_;
  std::vector<Index> connectivity_;
  ElementType type_ = ElementType::Hex8;
  int nodes_per_element_ = 8;
  int dim_ = 3;
  std::optional<StructuredGridInfo> structured_;
  std::map<std::string, std::vector<Index>> node_sets_;
  std::map<std::string, std::vector<Index>> element_sets_;
};

}  // namespace sparlab
