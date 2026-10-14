// Where this fits: see include/sparlab/fem/FemModel.hpp

#include "sparlab/fem/FemModel.hpp"

#include "sparlab/core/Exceptions.hpp"
#include "sparlab/core/Logging.hpp"
#include "sparlab/elements/Hex8.hpp"
#include "sparlab/elements/Quadrature.hpp"

#include <cmath>
#include <sstream>
#include <utility>

namespace sparlab {

FemModel::FemModel(Mesh mesh, Material material, Scalar thickness,
                   StressState stress_state, IntegrationOptions integration)
    : mesh_(std::move(mesh)),
      materials_{std::move(material)},
      thickness_(thickness),
      stress_state_(stress_state),
      integration_(integration),
      element_(make_element(mesh_.element_type(), integration)),
      d_{materials_.front().constitutive(stress_state)},
      dofs_(mesh_.num_nodes(), element_->dofs_per_node()) {
  if (stress_state_dimension(stress_state_) != mesh_.dim()) {
    std::ostringstream os;
    os << "stress state '" << to_string(stress_state_) << "' belongs to a "
       << stress_state_dimension(stress_state_) << "-D model but the mesh is "
       << mesh_.dim() << "-D (" << to_string(mesh_.element_type()) << " elements); use "
       << (mesh_.dim() == 3 ? "'three_dimensional'" : "'plane_stress' or 'plane_strain'");
    throw ConfigError(os.str());
  }
  if (!(thickness_ > 0.0)) {
    std::ostringstream os;
    os << "model thickness must be positive (got " << thickness_ << " m)";
    throw ConfigError(os.str());
  }
  if (mesh_.dim() == 3 && thickness_ != 1.0) {
    std::ostringstream os;
    os << "a 3-D model has no thickness (got " << thickness_
       << " m); leave model.thickness at its default of 1 for a solid mesh";
    throw ConfigError(os.str());
  }
  mesh_.validate();
  check_sheet_integration();
}

void FemModel::check_sheet_integration() const {
  const IntegrationOptions& opts = integration_;
  // The gradients of the incompatible modes vanish at the element centre, so
  // a one-point rule along any natural axis leaves the modes without
  // stiffness there (K_aa singular).
  if (opts.formulation == ElementFormulation::IncompatibleModes &&
      (opts.stiffness_points == 1 || opts.thickness_points == 1)) {
    throw ConfigError(std::string("the incompatible-mode Hex8 needs at least two integration "
                                  "points along every natural axis: the gradients of its modes "
                                  "vanish at the element centre and a one-point rule leaves "
                                  "them without stiffness; set 'model.integration.") +
                      (opts.stiffness_points == 1 ? "stiffness_points'" : "thickness_points'") +
                      " to 2 or more");
  }
  if (opts.thickness_points == 0) return;
  if (mesh_.element_type() != ElementType::Hex8) {
    throw ConfigError("'model.integration.thickness_points' sets the stiffness rule through "
                      "the thickness of a Hex8 sheet; the mesh is " +
                      to_string(mesh_.element_type()) + ": remove the key");
  }
  if (opts.thickness_points < 1 || opts.thickness_points > kMaxThicknessPoints) {
    std::ostringstream os;
    os << "'model.integration.thickness_points' is " << opts.thickness_points
       << "; it must be 1 to " << kMaxThicknessPoints;
    throw ConfigError(os.str());
  }
  if (opts.thickness_axis < 0 || opts.thickness_axis > 2) {
    std::ostringstream os;
    os << "the thickness axis is " << opts.thickness_axis << "; expected 0, 1 or 2 (x, y, z)";
    throw ConfigError(os.str());
  }
  if (opts.stiffness_points < 1 || opts.stiffness_points > 4) {
    std::ostringstream os;
    os << "'model.integration.stiffness_points' is " << opts.stiffness_points
       << "; it must be 1 to 4";
    throw ConfigError(os.str());
  }
  // The rule refines the element's natural axis thickness_axis, which the
  // structured generator aligns with the global axis of the same index. An
  // element whose natural axis is not the one closest to that global axis
  // (a mesh from a file with another node order) gets its extra points along
  // another direction: still a valid rule, but not the one meant.
  const int axis = opts.thickness_axis;
  const Eigen::Matrix<Scalar, 8, 3> dn = hex8_shape_gradients_natural(0.0, 0.0, 0.0);
  Index misaligned = 0;
  Index first = -1;
  for (Index e = 0; e < mesh_.num_elements(); ++e) {
    const Matrix3 jac = mesh_.element_coordinates(e) * dn;  // column k: dx/dxi_k
    int best = 0;
    Scalar best_cos = -1.0;
    for (int k = 0; k < 3; ++k) {
      const Scalar norm = jac.col(k).norm();
      const Scalar c = norm > 0.0 ? std::abs(jac(axis, k)) / norm : 0.0;
      if (c > best_cos) {
        best_cos = c;
        best = k;
      }
    }
    if (best != axis) {
      if (first < 0) first = e;
      ++misaligned;
    }
  }
  if (misaligned > 0) {
    static const char* const kAxis[3] = {"x", "y", "z"};
    static const char* const kNatural[3] = {"xi", "eta", "zeta"};
    log::warn(misaligned, " of ", mesh_.num_elements(), " Hex8 element(s) (the first is element ",
              first, ") have their ", kNatural[axis], " axis, which carries the ",
              opts.thickness_points, " thickness points, not along ", kAxis[axis],
              ": their extra points refine another direction; order the nodes so that ",
              kNatural[axis], " crosses the sheet, or change thickness_direction");
  }
}

void FemModel::assign_material(const Material& material,
                               const std::vector<Index>& elements) {
  if (finalized_) {
    throw ModelError("materials must be assigned before FemModel::finalize()");
  }
  const Index ne = mesh_.num_elements();
  for (Index e : elements) {
    if (e < 0 || e >= ne) {
      std::ostringstream os;
      os << "material '" << material.name() << "' assigned to element " << e
         << ", outside [0, " << ne - 1 << "]";
      throw ModelError(os.str());
    }
  }
  materials_.push_back(material);
  d_.push_back(material.constitutive(stress_state_));
  if (element_material_.empty()) element_material_.assign(static_cast<std::size_t>(ne), 0);
  const int id = static_cast<int>(materials_.size()) - 1;
  for (Index e : elements) element_material_[static_cast<std::size_t>(e)] = id;
}

void FemModel::finalize() {
  if (finalized_) return;
  if (load_case_specs_.empty()) {
    throw ConfigError("model has no load cases; define at least one");
  }
  const Index constrained = apply_constraints(mesh_, constraints_, dofs_);
  log::info("applied ", constraints_.size(), " boundary condition group(s): ",
            constrained, " of ", dofs_.num_dofs(), " DOFs prescribed, ",
            dofs_.num_free(), " free");
  finalized_ = true;
}

}  // namespace sparlab
