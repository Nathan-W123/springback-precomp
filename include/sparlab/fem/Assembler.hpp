/// \file Assembler.hpp
/// \brief Sparse global assembly of stiffness and mass matrices.
///
/// Assembly builds a triplet list and compresses it into a CSC matrix:
/// \f[
///   \mathbf{K} = \sum_e s_e\, \mathbf{A}_e^T \mathbf{K}_e^0 \mathbf{A}_e,
///   \qquad
///   \mathbf{M} = \sum_e m_e\, \mathbf{A}_e^T \mathbf{M}_e^0 \mathbf{A}_e,
/// \f]
/// where \f$\mathbf{A}_e\f$ is the (implicit) DOF gather operator, and
/// \f$s_e\f$, \f$m_e\f$ are optional per-element scale factors (1 when
/// omitted).
///
/// On a *uniform structured* mesh every cell has identical geometry, so
/// \f$\mathbf{K}_e^0\f$ and \f$\mathbf{M}_e^0\f$ are computed once and reused.
/// The generic per-element path is always available and the two are checked
/// against each other in the test suite.
///
/// **Cached sparsity.** The pattern of a stiffness (or consistent mass) matrix
/// depends only on the mesh: one dim x dim block for every pair of nodes that
/// share an element. It is built once, together with the position of each
/// element's node blocks in it, and every later assembly scatters the scaled
/// element matrices straight into the value array - no triplet list, no
/// sort, memory proportional to the matrix itself. Contributions reach each
/// entry in element order, exactly as the triplet assembly sums them, so the
/// two paths give bitwise identical matrices. The free-free block is likewise
/// cached as an index map into the full matrix for the current constraint
/// set. An element with a zero scale factor, whose entries a triplet assembly
/// would leave out of the pattern, falls back to the triplet path so the
/// result stays the same.
///
/// Where this fits: Simulations stage - sums element matrices into global sparse matrices:
///   the Newton tangent of the forming analysis (the mass matrices are tested, not used by it).
/// Called from: fem/NonlinearSystem.hpp:NonlinearSystem::evaluate (assemble_elementwise),
///   apps/sparlab_form.cpp:main
/// Calls into: fem/FemModel (mesh, element, materials), fem/DofManager (element DOFs)
/// Start reading at: Assembler::assemble_elementwise, Assembler::scatter (fem/Assembler.cpp)
#pragma once

#include "sparlab/core/Types.hpp"
#include "sparlab/fem/FemModel.hpp"

#include <functional>
#include <vector>

namespace sparlab {

class Assembler {
 public:
  explicit Assembler(const FemModel& model);

  /// Element stiffness matrix of element `e` [N/m].
  const Matrix& element_stiffness(Index e) const;

  /// Consistent element mass matrix of element `e` with its material's
  /// density rho [kg].
  const Matrix& element_mass(Index e) const;

  /// True when a single cached element matrix is reused for the whole mesh.
  bool uses_element_cache() const { return uniform_; }

  /// Assemble the global stiffness matrix [N/m].
  /// \param scale optional per-element stiffness factors (length
  ///        num_elements); `nullptr` means all ones.
  SparseMatrix assemble_stiffness(const Vector* scale = nullptr) const;

  /// Assemble the global mass matrix [kg].
  /// \param type consistent or lumped (row sums).
  /// \param scale optional per-element mass factors; `nullptr` means all ones.
  SparseMatrix assemble_mass(MassType type, const Vector* scale = nullptr) const;

  /// Extract the free-free block \f$K_{ff}\f$ of a full-size matrix.
  SparseMatrix reduce_free_free(const SparseMatrix& full) const;

  /// Assemble \f$\sum_e A_e^T M_e A_e\f$ for element matrices computed on
  /// demand by `element_matrix(e)` (num_dofs x num_dofs of the element, in
  /// its node-major DOF order), for matrices that are neither the stiffness
  /// nor the mass: the tangent of a deformed state, say. Any element matrix
  /// fits the stiffness sparsity pattern, so the cached pattern is used and
  /// the result can be reduced with `reduce_free_free`. Entries may have any
  /// sign.
  SparseMatrix assemble_elementwise(const std::function<Matrix(Index)>& element_matrix) const;

 private:
  struct Pattern {
    bool ready = false;
    Index size = 0;
    std::vector<StorageIndex> outer;
    std::vector<StorageIndex> inner;
    /// Offset of local node a's rows within local node b's columns, per
    /// element: block_offset[(e * npe + a) * npe + b].
    std::vector<Index> block_offset;
  };
  struct Reduction {
    std::vector<Index> key;          ///< constraint partition the map was built for
    Index rows = 0;
    Index cols = 0;
    std::vector<StorageIndex> outer;
    std::vector<StorageIndex> inner;
    std::vector<Index> source;       ///< value index in the full matrix
  };

  void build_cache();
  Matrix compute_element_stiffness(Index e) const;
  Matrix compute_element_mass(Index e) const;
  void build_pattern() const;
  bool matches_pattern(const SparseMatrix& full) const;
  template <typename ElementMatrix>
  SparseMatrix scatter(const ElementMatrix& element_matrix, const Vector* scale) const;
  static SparseMatrix from_reduction(const Reduction& map, const SparseMatrix& full);

  const FemModel& model_;
  bool uniform_ = false;
  Matrix cached_k_;
  Matrix cached_m_;
  mutable std::vector<Matrix> per_element_k_;
  mutable std::vector<Matrix> per_element_m_;
  mutable Pattern pattern_;
  mutable Reduction free_free_;
};

}  // namespace sparlab
