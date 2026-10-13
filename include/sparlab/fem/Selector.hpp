/// \file Selector.hpp
/// \brief Geometric region selectors for boundary conditions, tool surfaces
///        and material regions.
///
/// Selectors are geometric so that the *same* configuration block can be
/// applied to meshes of different resolution without re-authoring indices.
/// The one non-geometric primitive, `Group`, names a set carried by the mesh
/// itself (an Abaqus *NSET / *ELSET): the way to reach a curved hole wall or
/// a CAD face that no box describes.
/// Every primitive is written for three coordinates; on a 2-D mesh the z
/// coordinate of every node is zero, so the default z bounds (+-infinity) and
/// the default circle axis (z) make a 2-D deck read exactly as before.
///
/// A `SelectorGroup` is the union of its members, optionally complemented.
/// Tolerances are relative to the mesh diagonal so that degenerate boxes
/// (`xmin == xmax`) reliably capture a line (2-D) or plane (3-D) of nodes.
#pragma once

#include "sparlab/core/Types.hpp"
#include "sparlab/mesh/Mesh.hpp"

#include <limits>
#include <string>
#include <vector>

namespace sparlab {

enum class SelectorKind {
  All,          ///< every entity
  Box,          ///< axis-aligned box, inclusive within tolerance
  Circle,       ///< closed disc of radius `radius` about `center`, measured in
                ///< the plane normal to `axis` (a cylinder through a 3-D mesh)
  Annulus,      ///< closed ring with `inner_radius` <= r <= `radius`, same metric
  Sphere,       ///< closed ball of radius `radius` about `center` (3-D distance)
  NodeIds,      ///< explicit node indices (node selection only)
  ElementIds,   ///< explicit element indices (element selection only)
  NearestNode,  ///< the single node closest to `point`
  Group         ///< a named node / element set of the mesh (`group`)
};

/// One geometric primitive.
struct Selector {
  SelectorKind kind = SelectorKind::All;

  // Box bounds [m]; unspecified sides default to +-infinity.
  Scalar xmin = -std::numeric_limits<Scalar>::infinity();
  Scalar xmax = std::numeric_limits<Scalar>::infinity();
  Scalar ymin = -std::numeric_limits<Scalar>::infinity();
  Scalar ymax = std::numeric_limits<Scalar>::infinity();
  Scalar zmin = -std::numeric_limits<Scalar>::infinity();
  Scalar zmax = std::numeric_limits<Scalar>::infinity();

  Vector3 center = Vector3::Zero();  ///< circle / annulus / sphere centre [m]
  Scalar radius = 0.0;               ///< outer radius [m]
  Scalar inner_radius = 0.0;         ///< annulus inner radius [m]
  /// Axis (0 = x, 1 = y, 2 = z) normal to the plane in which a circle or
  /// annulus measures its radius. z by default, which is the in-plane distance
  /// of a 2-D mesh and a through-thickness bolt hole of a 3-D plate.
  int axis = 2;

  Vector3 point = Vector3::Zero();   ///< NearestNode target [m]
  std::vector<Index> ids;            ///< explicit indices
  std::string group;                 ///< Group: name of a mesh node / element set

  /// Absolute geometric tolerance [m]. When <= 0 a tolerance of
  /// `1e-9 * mesh diagonal` is used.
  Scalar tolerance = 0.0;

  bool contains(const Vector3& x, Scalar tol) const;
};

/// Union of selectors with an optional complement, plus a label for diagnostics.
struct SelectorGroup {
  std::string name = "region";
  std::vector<Selector> members;
  bool invert = false;  ///< select the complement of the union

  /// Node indices matched by this group (ascending, unique). A `Group`
  /// primitive selects the named node set, or else the nodes of the named
  /// element set.
  std::vector<Index> select_nodes(const Mesh& mesh) const;

  /// Element indices matched by this group, tested at element centroids
  /// (ascending, unique). A `Group` primitive needs an element set.
  std::vector<Index> select_elements(const Mesh& mesh) const;
};

}  // namespace sparlab
