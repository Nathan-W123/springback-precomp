/// \file DofManager.hpp
/// \brief Global degree-of-freedom numbering and Dirichlet partitioning.
///
/// DOF numbering is node-major with `dofs_per_node` translational components:
/// node `n` owns DOFs `d n + k` for component `k` (u_x, u_y, u_z).
/// Constraints are handled by *partitioning* rather than by penalty terms: the
/// DOF set is split into free (f) and prescribed (p) blocks and the system
/// \f[
///   \begin{bmatrix} K_{ff} & K_{fp} \\ K_{pf} & K_{pp}\end{bmatrix}
///   \begin{Bmatrix} u_f \\ u_p \end{Bmatrix} =
///   \begin{Bmatrix} f_f \\ f_p + r \end{Bmatrix}
/// \f]
/// is reduced to \f$ K_{ff} u_f = f_f - K_{fp} u_p \f$ with reactions
/// \f$ r = K_{pf} u_f + K_{pp} u_p - f_p \f$.
/// This keeps \f$K_{ff}\f$ symmetric positive definite (no artificial large
/// diagonal entries) and gives exact reaction forces.
///
/// Where this fits: Simulations stage - the global DOF numbering and the free / prescribed
///   split of the model's constraints (forming steps build their own in fem/Forming.cpp).
/// Called from: fem/BoundaryConditions.cpp:apply_constraints (prescribe),
///   fem/Assembler.cpp:Assembler::assemble_stiffness, io/ResultWriter.cpp:ResultWriter::write_mesh
/// Calls into: core/Types and core/Exceptions only
/// Start reading at: DofManager::dof, DofManager::prescribe (fem/DofManager.cpp)
#pragma once

#include "sparlab/core/Types.hpp"

#include <vector>

namespace sparlab {

class DofManager {
 public:
  DofManager() = default;
  /// \param dofs_per_node translational components per node (3: sparlab_form
  ///        meshes are Hex8; 2 is accepted but unreachable).
  explicit DofManager(Index num_nodes, int dofs_per_node = 3);

  int dofs_per_node() const { return dofs_per_node_; }
  Index num_dofs() const { return num_nodes_ * dofs_per_node_; }

  /// Global DOF index of `component` (0 = x, 1 = y, 2 = z) at `node`.
  /// \throws ModelError for out-of-range input.
  Index dof(Index node, int component) const;

  /// Prescribe a displacement value [m] on one nodal component. Re-prescribing
  /// the same DOF overwrites the previous value.
  void prescribe(Index node, int component, Scalar value = 0.0);

  /// Prescribe by global DOF index.
  void prescribe_dof(Index dof, Scalar value = 0.0);

  bool is_constrained(Index dof) const { return constrained_[static_cast<std::size_t>(dof)]; }
  Scalar prescribed_value(Index dof) const { return values_[static_cast<std::size_t>(dof)]; }

  /// Global indices of the free DOFs, ascending.
  const std::vector<Index>& free_dofs() const;
  /// Global indices of the prescribed DOFs, ascending.
  const std::vector<Index>& constrained_dofs() const;

  Index num_free() const { return static_cast<Index>(free_dofs().size()); }
  Index num_constrained() const { return static_cast<Index>(constrained_dofs().size()); }

  /// Position of `dof` inside the reduced (free) system, or -1 if constrained.
  Index reduced_index(Index dof) const;

  /// Fill `out` with the `nodes_per_elem * dofs_per_node` global DOFs of an
  /// element, in node-major order matching the element kernels.
  void element_dofs(const Index* nodes, int nodes_per_elem, Index* out) const;

 private:
  void rebuild() const;

  Index num_nodes_ = 0;
  int dofs_per_node_ = 3;
  std::vector<char> constrained_;
  std::vector<Scalar> values_;

  // Lazily rebuilt partition data.
  mutable bool dirty_ = true;
  mutable std::vector<Index> free_;
  mutable std::vector<Index> fixed_;
  mutable std::vector<Index> reduced_;
};

}  // namespace sparlab
