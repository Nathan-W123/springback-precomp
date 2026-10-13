#include "sparlab/fem/DofManager.hpp"

#include "sparlab/core/Exceptions.hpp"

#include <sstream>

namespace sparlab {

DofManager::DofManager(Index num_nodes, int dofs_per_node)
    : num_nodes_(num_nodes), dofs_per_node_(dofs_per_node) {
  if (num_nodes_ < 0) throw ModelError("DofManager requires a non-negative node count");
  if (dofs_per_node_ != 2 && dofs_per_node_ != 3) {
    std::ostringstream os;
    os << "DofManager supports 2 or 3 translational DOFs per node (got " << dofs_per_node_
       << ")";
    throw ModelError(os.str());
  }
  constrained_.assign(static_cast<std::size_t>(num_dofs()), 0);
  values_.assign(static_cast<std::size_t>(num_dofs()), 0.0);
}

Index DofManager::dof(Index node, int component) const {
  if (node < 0 || node >= num_nodes_) {
    std::ostringstream os;
    os << "node index " << node << " is outside [0, " << num_nodes_ - 1 << "]";
    throw ModelError(os.str());
  }
  if (component < 0 || component >= dofs_per_node_) {
    std::ostringstream os;
    os << "DOF component " << component << " is outside [0, " << dofs_per_node_ - 1
       << "] (";
    for (int k = 0; k < dofs_per_node_; ++k) {
      os << (k ? ", " : "") << k << " = " << dof_component_name(k);
    }
    os << ")";
    throw ModelError(os.str());
  }
  return node * dofs_per_node_ + component;
}

void DofManager::prescribe(Index node, int component, Scalar value) {
  prescribe_dof(dof(node, component), value);
}

void DofManager::prescribe_dof(Index d, Scalar value) {
  if (d < 0 || d >= num_dofs()) {
    std::ostringstream os;
    os << "DOF index " << d << " is outside [0, " << num_dofs() - 1 << "]";
    throw ModelError(os.str());
  }
  constrained_[static_cast<std::size_t>(d)] = 1;
  values_[static_cast<std::size_t>(d)] = value;
  dirty_ = true;
}

void DofManager::rebuild() const {
  free_.clear();
  fixed_.clear();
  reduced_.assign(static_cast<std::size_t>(num_dofs()), -1);
  for (Index d = 0; d < num_dofs(); ++d) {
    if (constrained_[static_cast<std::size_t>(d)]) {
      fixed_.push_back(d);
    } else {
      reduced_[static_cast<std::size_t>(d)] = static_cast<Index>(free_.size());
      free_.push_back(d);
    }
  }
  dirty_ = false;
}

const std::vector<Index>& DofManager::free_dofs() const {
  if (dirty_) rebuild();
  return free_;
}

const std::vector<Index>& DofManager::constrained_dofs() const {
  if (dirty_) rebuild();
  return fixed_;
}

Index DofManager::reduced_index(Index dof) const {
  if (dirty_) rebuild();
  if (dof < 0 || dof >= num_dofs()) {
    std::ostringstream os;
    os << "DOF index " << dof << " is outside [0, " << num_dofs() - 1 << "]";
    throw ModelError(os.str());
  }
  return reduced_[static_cast<std::size_t>(dof)];
}

void DofManager::element_dofs(const Index* nodes, int nodes_per_elem, Index* out) const {
  for (int a = 0; a < nodes_per_elem; ++a) {
    for (int k = 0; k < dofs_per_node_; ++k) {
      out[dofs_per_node_ * a + k] = nodes[a] * dofs_per_node_ + k;
    }
  }
}

}  // namespace sparlab
