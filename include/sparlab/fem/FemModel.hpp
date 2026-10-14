/// \file FemModel.hpp
/// \brief Aggregation of everything that defines one discrete structural model.
///
/// The model is a value type: it owns a mesh, the material, the idealisation
/// (plane stress, plane strain or full 3-D elasticity), integration options,
/// the DOF partitioning and the load case. It deliberately performs no
/// numerics — assembly and the analyses consume a `const FemModel&`.
///
/// The spatial dimension is the mesh's. A 2-D mesh takes a plane stress state
/// and a positive thickness; a 3-D mesh takes `StressState::ThreeDimensional`
/// and no thickness (the value must be 1). Either mismatch is rejected in the
/// constructor, so a deck cannot pair a solid mesh with a plane idealisation.
/// The element is that of `integration.formulation` (Element.hpp), and a
/// through-thickness rule (`integration.thickness_points`) is accepted for a
/// Hex8 mesh only.
///
/// Where this fits: Simulations stage - the model built from the deck (mesh, materials,
///   element, DOFs, boundary conditions) that every analysis of sparlab_form reads.
/// Called from: io/Config.cpp:build_model (builds and finalises it), apps/sparlab_form.cpp:main
/// Calls into: mesh/Mesh, elements/Element (make_element), material/Material,
///   fem/BoundaryConditions (apply_constraints)
/// Start reading at: FemModel::FemModel, FemModel::finalize (fem/FemModel.cpp)
#pragma once

#include "sparlab/core/Types.hpp"
#include "sparlab/elements/Element.hpp"
#include "sparlab/fem/BoundaryConditions.hpp"
#include "sparlab/fem/DofManager.hpp"
#include "sparlab/material/Material.hpp"
#include "sparlab/mesh/Mesh.hpp"

#include <memory>
#include <vector>

namespace sparlab {

/// Mass-matrix formulation.
enum class MassType {
  Consistent,  ///< M_e = rho t int N^T N dOmega (default)
  Lumped       ///< row-sum lumping of the consistent matrix (diagonal)
};

/// A finite element model: mesh, element, materials, constraints and DOFs.
class FemModel {
 public:
  FemModel(Mesh mesh, Material material, Scalar thickness,
           StressState stress_state, IntegrationOptions integration);

  const Mesh& mesh() const { return mesh_; }
  /// Spatial dimension of the model: the translations per node (3:
  /// sparlab_form meshes are Hex8).
  int dim() const { return mesh_.dim(); }
  /// Degrees of freedom per node: the `dim()` translations.
  int dofs_per_node() const { return element_->dofs_per_node(); }
  /// The primary material (the deck's `material`); every element uses it
  /// unless `assign_material` gave it another.
  const Material& material() const { return materials_.front(); }
  /// Out-of-plane thickness [m] of a 2-D model; 1 for a 3-D model.
  Scalar thickness() const { return thickness_; }
  StressState stress_state() const { return stress_state_; }
  const IntegrationOptions& integration() const { return integration_; }
  const Element& element() const { return *element_; }

  DofManager& dofs() { return dofs_; }
  const DofManager& dofs() const { return dofs_; }

  /// Constitutive matrix of the primary material [Pa] (3 x 3 or 6 x 6).
  const Matrix& constitutive() const { return d_.front(); }

  /// Give the listed elements `material`, appended to the model's materials.
  /// A later assignment overrides an earlier one for the same elements.
  /// \throws ModelError for an element index out of range.
  void assign_material(const Material& material, const std::vector<Index>& elements);

  const std::vector<Material>& materials() const { return materials_; }
  /// True when every element uses the primary material.
  bool single_material() const { return element_material_.empty(); }
  /// Index into `materials()` of element `e`'s material.
  int element_material(Index e) const {
    return element_material_.empty() ? 0 : element_material_[static_cast<std::size_t>(e)];
  }
  const Material& material_of(Index e) const {
    return materials_[static_cast<std::size_t>(element_material(e))];
  }
  const Matrix& constitutive_of(Index e) const {
    return d_[static_cast<std::size_t>(element_material(e))];
  }

  std::vector<LoadCaseSpec>& load_case_specs() { return load_case_specs_; }
  const std::vector<LoadCaseSpec>& load_case_specs() const { return load_case_specs_; }

  std::vector<DisplacementConstraint>& constraints() { return constraints_; }
  const std::vector<DisplacementConstraint>& constraints() const { return constraints_; }

  /// Apply `constraints()` to the DOF manager. Must be called after the
  /// constraints and load cases are set and before any assembly. Idempotent.
  /// \throws ConfigError for a model with no load case, or a constraint
  ///         that `apply_constraints` refuses.
  void finalize();

  bool finalized() const { return finalized_; }

 private:
  /// Validates the sheet integration of `integration_` (thickness points:
  /// Hex8 only, their range) and warns about elements whose natural thickness
  /// axis is not along the global axis of the same index.
  void check_sheet_integration() const;

  Mesh mesh_;
  std::vector<Material> materials_;
  std::vector<int> element_material_;  ///< empty: every element uses materials_[0]
  Scalar thickness_;
  StressState stress_state_;
  IntegrationOptions integration_;
  std::unique_ptr<Element> element_;
  std::vector<Matrix> d_;              ///< one constitutive matrix per material
  DofManager dofs_;
  std::vector<DisplacementConstraint> constraints_;
  std::vector<LoadCaseSpec> load_case_specs_;
  bool finalized_ = false;
};

}  // namespace sparlab
