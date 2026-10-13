#include "sparlab/fem/BoundaryConditions.hpp"

#include "sparlab/core/Exceptions.hpp"
#include "sparlab/core/Logging.hpp"

#include <algorithm>
#include <set>
#include <sstream>

namespace sparlab {
std::vector<Mesh::BoundaryFace> faces_in_region(const Mesh& mesh,
                                                const std::vector<Mesh::BoundaryFace>& boundary,
                                                const SelectorGroup& region) {
  std::vector<char> in_region(static_cast<std::size_t>(mesh.num_nodes()), 0);
  for (Index n : region.select_nodes(mesh)) in_region[static_cast<std::size_t>(n)] = 1;
  std::vector<Mesh::BoundaryFace> out;
  for (const Mesh::BoundaryFace& face : boundary) {
    const bool inside = std::all_of(face.nodes.begin(), face.nodes.end(), [&](Index n) {
      return in_region[static_cast<std::size_t>(n)] != 0;
    });
    if (inside) out.push_back(face);
  }
  return out;
}

void DisplacementConstraint::set(int component, bool fix, Scalar value) {
  switch (component) {
    case 0: fix_x = fix; value_x = value; return;
    case 1: fix_y = fix; value_y = value; return;
    case 2: fix_z = fix; value_z = value; return;
    case 3: fix_rx = fix; value_rx = value; return;
    case 4: fix_ry = fix; value_ry = value; return;
    case 5: fix_rz = fix; value_rz = value; return;
    default: break;
  }
  std::ostringstream os;
  os << "DOF component " << component << " is outside [0, " << kMaxDofsPerNode - 1 << "]";
  throw ModelError(os.str());
}

Index apply_constraints(const Mesh& mesh,
                        const std::vector<DisplacementConstraint>& constraints,
                        DofManager& dofs) {
  const int dim = mesh.dim();
  const int ndpn = dofs.dofs_per_node();
  std::set<Index> touched;
  for (const DisplacementConstraint& bc : constraints) {
    bool any = false;
    for (int k = 0; k < kMaxDofsPerNode; ++k) any = any || bc.fixes(k);
    if (!any) {
      throw ConfigError("boundary condition '" + bc.region.name +
                        "' constrains no component; remove it or list the "
                        "components to fix");
    }
    if (dim == 2 && bc.fix_z) {
      throw ConfigError("boundary condition '" + bc.region.name +
                        "' fixes z, but the model is two-dimensional and has no z "
                        "degree of freedom");
    }
    if (ndpn < kMaxDofsPerNode && bc.fixes_rotation()) {
      throw ConfigError("boundary condition '" + bc.region.name +
                        "' fixes a rotation, but the model's nodes carry translations "
                        "only; rotations are degrees of freedom of shell and beam "
                        "elements");
    }
    const std::vector<Index> nodes = bc.region.select_nodes(mesh);
    if (nodes.empty()) {
      throw ConfigError("boundary condition '" + bc.region.name +
                        "' selected no nodes; check its coordinates against the mesh "
                        "extents (an unconstrained model yields a singular stiffness "
                        "matrix)");
    }
    for (Index n : nodes) {
      for (int k = 0; k < ndpn; ++k) {
        if (!bc.fixes(k)) continue;
        dofs.prescribe(n, k, bc.value(k));
        touched.insert(dofs.dof(n, k));
      }
    }
    if (static_cast<int>(log::level()) <= static_cast<int>(log::Level::Debug)) {
      std::ostringstream fixed;
      for (int k = 0; k < ndpn; ++k) {
        if (bc.fixes(k)) fixed << ' ' << dof_component_name(k);
      }
      log::debug("boundary condition '", bc.region.name, "': ", nodes.size(),
                 " nodes, fixed:", fixed.str());
    }
  }
  return static_cast<Index>(touched.size());
}

}  // namespace sparlab
