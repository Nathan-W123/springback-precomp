// Where this fits: see include/sparlab/fem/Selector.hpp

#include "sparlab/fem/Selector.hpp"

#include "sparlab/core/Exceptions.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace sparlab {
namespace {

Scalar default_tolerance(const Mesh& mesh) {
  const BoundingBox bb = mesh.bounding_box();
  const Vector3 extent = bb.extent();
  const Scalar diag = mesh.dim() == 2 ? std::hypot(extent.x(), extent.y())
                                      : std::hypot(extent.x(), extent.y(), extent.z());
  return 1.0e-9 * std::max(diag, 1.0e-30);
}

std::string available_sets(const Mesh& mesh) {
  std::ostringstream os;
  if (mesh.node_sets().empty() && mesh.element_sets().empty()) {
    os << "the mesh carries no named sets (only a mesh read from a file with physical "
          "groups or *NSET / *ELSET cards does)";
    return os.str();
  }
  os << "available node sets:";
  for (const auto& entry : mesh.node_sets()) {
    os << " '" << entry.first << "' (" << entry.second.size() << ")";
  }
  os << "; element sets:";
  for (const auto& entry : mesh.element_sets()) {
    os << " '" << entry.first << "' (" << entry.second.size() << ")";
  }
  return os.str();
}

/// Distance from `center` measured in the plane normal to `axis`.
Scalar radial_distance(const Vector3& x, const Vector3& center, int axis) {
  Vector3 d = x - center;
  d(axis) = 0.0;
  return d.norm();
}

}  // namespace

bool Selector::contains(const Vector3& x, Scalar tol) const {
  const Scalar t = tolerance > 0.0 ? tolerance : tol;
  switch (kind) {
    case SelectorKind::All:
      return true;
    case SelectorKind::Box:
      return x.x() >= xmin - t && x.x() <= xmax + t && x.y() >= ymin - t &&
             x.y() <= ymax + t && x.z() >= zmin - t && x.z() <= zmax + t;
    case SelectorKind::Circle:
      return radial_distance(x, center, axis) <= radius + t;
    case SelectorKind::Annulus: {
      const Scalar r = radial_distance(x, center, axis);
      return r >= inner_radius - t && r <= radius + t;
    }
    case SelectorKind::Sphere:
      return (x - center).norm() <= radius + t;
    case SelectorKind::NodeIds:
    case SelectorKind::ElementIds:
    case SelectorKind::NearestNode:
    case SelectorKind::Group:
      // Handled by the group, which needs mesh-wide information.
      return false;
  }
  return false;
}

std::vector<Index> SelectorGroup::select_nodes(const Mesh& mesh) const {
  const Scalar tol = default_tolerance(mesh);
  std::vector<char> hit(static_cast<std::size_t>(mesh.num_nodes()), 0);

  for (const Selector& sel : members) {
    if (sel.kind == SelectorKind::ElementIds) {
      throw ConfigError("region '" + name +
                        "' uses element_ids where a node region is expected");
    }
    if ((sel.kind == SelectorKind::Circle || sel.kind == SelectorKind::Annulus) &&
        (sel.axis < 0 || sel.axis > 2)) {
      throw ConfigError("region '" + name + "' has a circle/annulus axis outside 0..2");
    }
    if (sel.kind == SelectorKind::NodeIds) {
      for (Index id : sel.ids) {
        if (id < 0 || id >= mesh.num_nodes()) {
          std::ostringstream os;
          os << "region '" << name << "' lists node id " << id
             << " which is outside [0, " << mesh.num_nodes() - 1 << "]";
          throw ConfigError(os.str());
        }
        hit[static_cast<std::size_t>(id)] = 1;
      }
      continue;
    }
    if (sel.kind == SelectorKind::Group) {
      const auto nodes = mesh.node_sets().find(sel.group);
      if (nodes != mesh.node_sets().end()) {
        for (Index n : nodes->second) hit[static_cast<std::size_t>(n)] = 1;
        continue;
      }
      const auto elements = mesh.element_sets().find(sel.group);
      if (elements != mesh.element_sets().end()) {
        for (Index e : elements->second) {
          const Index* en = mesh.element_nodes(e);
          for (int a = 0; a < mesh.nodes_per_elem(); ++a) {
            hit[static_cast<std::size_t>(en[a])] = 1;
          }
        }
        continue;
      }
      throw ConfigError("region '" + name + "' names mesh group '" + sel.group +
                        "', which the mesh does not define; " + available_sets(mesh));
    }
    if (sel.kind == SelectorKind::NearestNode) {
      Index best = -1;
      Scalar best_d2 = std::numeric_limits<Scalar>::max();
      for (Index n = 0; n < mesh.num_nodes(); ++n) {
        const Scalar d2 = (mesh.node(n) - sel.point).squaredNorm();
        if (d2 < best_d2) {
          best_d2 = d2;
          best = n;
        }
      }
      if (best < 0) throw ConfigError("region '" + name + "': mesh has no nodes");
      hit[static_cast<std::size_t>(best)] = 1;
      continue;
    }
    for (Index n = 0; n < mesh.num_nodes(); ++n) {
      if (sel.contains(mesh.node(n), tol)) hit[static_cast<std::size_t>(n)] = 1;
    }
  }

  std::vector<Index> out;
  for (Index n = 0; n < mesh.num_nodes(); ++n) {
    const bool inside = hit[static_cast<std::size_t>(n)] != 0;
    if (inside != invert) out.push_back(n);
  }
  return out;
}

std::vector<Index> SelectorGroup::select_elements(const Mesh& mesh) const {
  const Scalar tol = default_tolerance(mesh);
  std::vector<char> hit(static_cast<std::size_t>(mesh.num_elements()), 0);

  for (const Selector& sel : members) {
    if (sel.kind == SelectorKind::NodeIds || sel.kind == SelectorKind::NearestNode) {
      throw ConfigError("region '" + name +
                        "' uses a node-only selector where an element region is "
                        "expected");
    }
    if ((sel.kind == SelectorKind::Circle || sel.kind == SelectorKind::Annulus) &&
        (sel.axis < 0 || sel.axis > 2)) {
      throw ConfigError("region '" + name + "' has a circle/annulus axis outside 0..2");
    }
    if (sel.kind == SelectorKind::Group) {
      const auto elements = mesh.element_sets().find(sel.group);
      if (elements == mesh.element_sets().end()) {
        const bool node_set = mesh.node_sets().count(sel.group) > 0;
        throw ConfigError(
            "region '" + name + "' names mesh group '" + sel.group + "' as an element " +
            "region, " +
            (node_set ? std::string("but it is a node set (a boundary group); element "
                                    "regions need a group of cells - a physical surface "
                                    "in 2-D or a physical volume in 3-D. ")
                      : std::string("which the mesh does not define; ")) +
            available_sets(mesh));
      }
      for (Index e : elements->second) hit[static_cast<std::size_t>(e)] = 1;
      continue;
    }
    if (sel.kind == SelectorKind::ElementIds) {
      for (Index id : sel.ids) {
        if (id < 0 || id >= mesh.num_elements()) {
          std::ostringstream os;
          os << "region '" << name << "' lists element id " << id
             << " which is outside [0, " << mesh.num_elements() - 1 << "]";
          throw ConfigError(os.str());
        }
        hit[static_cast<std::size_t>(id)] = 1;
      }
      continue;
    }
    for (Index e = 0; e < mesh.num_elements(); ++e) {
      if (sel.contains(mesh.element_centroid(e), tol)) hit[static_cast<std::size_t>(e)] = 1;
    }
  }

  std::vector<Index> out;
  for (Index e = 0; e < mesh.num_elements(); ++e) {
    const bool inside = hit[static_cast<std::size_t>(e)] != 0;
    if (inside != invert) out.push_back(e);
  }
  return out;
}

}  // namespace sparlab
