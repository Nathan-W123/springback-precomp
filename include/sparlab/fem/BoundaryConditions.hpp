/// \file BoundaryConditions.hpp
/// \brief Displacement constraints and their application to a DofManager,
///        and the load case a model carries.
///
/// Sign convention: displacements are positive along the global +x / +y / +z
/// axes. Vectors are three-component throughout; on a 2-D model `fix_z` is
/// an error.
#pragma once

#include "sparlab/core/Types.hpp"
#include "sparlab/mesh/Mesh.hpp"
#include "sparlab/fem/DofManager.hpp"
#include "sparlab/fem/Selector.hpp"

#include <string>
#include <vector>

namespace sparlab {

/// Prescribed displacement on the x, y and/or z component of a node region,
/// and, on a shell or beam model, prescribed rotations about the global axes.
struct DisplacementConstraint {
  SelectorGroup region;
  bool fix_x = false;
  bool fix_y = false;
  bool fix_z = false;
  bool fix_rx = false;   ///< rotation about x (shell and beam nodes only)
  bool fix_ry = false;
  bool fix_rz = false;
  Scalar value_x = 0.0;  ///< prescribed u_x [m]
  Scalar value_y = 0.0;  ///< prescribed u_y [m]
  Scalar value_z = 0.0;  ///< prescribed u_z [m]
  Scalar value_rx = 0.0;  ///< prescribed rotation about x [rad]
  Scalar value_ry = 0.0;
  Scalar value_rz = 0.0;

  /// Component `k` in node-major DOF order: 0-2 the translations, 3-5 the
  /// rotations.
  bool fixes(int component) const {
    switch (component) {
      case 0: return fix_x;
      case 1: return fix_y;
      case 2: return fix_z;
      case 3: return fix_rx;
      case 4: return fix_ry;
      case 5: return fix_rz;
      default: return false;
    }
  }
  Scalar value(int component) const {
    switch (component) {
      case 0: return value_x;
      case 1: return value_y;
      case 2: return value_z;
      case 3: return value_rx;
      case 4: return value_ry;
      case 5: return value_rz;
      default: return 0.0;
    }
  }
  void set(int component, bool fix, Scalar value = 0.0);
  bool fixes_rotation() const { return fix_rx || fix_ry || fix_rz; }
};

/// The load case of a model. The forming analysis applies no load: its tools
/// and prescribed displacements drive the model, which carries one empty
/// case (named "forming" by the configuration) for the record of mesh.json.
struct LoadCaseSpec {
  std::string name = "load_case";
  Scalar weight = 1.0;  ///< recorded in mesh.json [-]
};

/// The boundary faces of `boundary` whose nodes all lie in
/// `region` (a tool's contact surface), in the order of `boundary`.
std::vector<Mesh::BoundaryFace> faces_in_region(const Mesh& mesh,
                                                const std::vector<Mesh::BoundaryFace>& boundary,
                                                const SelectorGroup& region);

/// Apply every constraint to `dofs`.
/// \return the number of distinct DOFs constrained.
/// \throws ConfigError when a region selects no nodes (almost always an
///         authoring mistake, and silently ignoring it produces a singular
///         system later), when a 2-D model is asked to fix z, or when a
///         model without rotational DOFs is asked to fix a rotation.
Index apply_constraints(const Mesh& mesh,
                        const std::vector<DisplacementConstraint>& constraints,
                        DofManager& dofs);

}  // namespace sparlab
