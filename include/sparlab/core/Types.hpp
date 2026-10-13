/// \file Types.hpp
/// \brief Fundamental scalar / matrix aliases and dimensional helpers.
///
/// All quantities in SparLab are expressed in strict SI base units:
///   length [m], force [N], stress and modulus [Pa], density [kg/m^3],
///   mass [kg], frequency [Hz], compliance [J] = [N m].
/// See docs/conventions.md for the full convention table.
///
/// The spatial dimension is a *runtime* property of the mesh (2 for plane
/// problems, 3 for solids), so nothing here fixes it at compile time. The
/// quantities that follow from it - degrees of freedom per node, Voigt
/// components - are obtained from the helpers below.
#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <string>
#include <vector>

namespace sparlab {

/// Floating point type used throughout the library.
using Scalar = double;

/// Integer type used for sparse-matrix indices (must match Eigen's default).
using StorageIndex = int;

/// Global node / element / DOF index type.
using Index = int;

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
using Vector2 = Eigen::Vector2d;
using Vector3 = Eigen::Vector3d;
using Vector6 = Eigen::Matrix<Scalar, 6, 1>;
using Matrix2 = Eigen::Matrix2d;
using Matrix3 = Eigen::Matrix3d;
using Matrix6 = Eigen::Matrix<Scalar, 6, 6>;

using SparseMatrix = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, StorageIndex>;
using Triplet = Eigen::Triplet<Scalar, StorageIndex>;
using TripletList = std::vector<Triplet>;

/// Number of independent stress/strain components in Voigt notation for a
/// spatial dimension: 3 in 2-D (xx, yy, xy) and 6 in 3-D
/// (xx, yy, zz, xy, yz, zx). Shear entries are engineering strains.
constexpr int voigt_components(int dim) { return dim == 3 ? 6 : 3; }

/// Configuration spelling of nodal DOF component `k`: "x", "y", "z".
inline const char* dof_component_name(int k) {
  static const char* const names[3] = {"x", "y", "z"};
  return (k >= 0 && k < 3) ? names[k] : "?";
}

/// Idealisation used to build the constitutive matrix. The two plane states
/// belong to 2-D meshes, `ThreeDimensional` to 3-D meshes; FemModel rejects a
/// mismatch.
enum class StressState {
  PlaneStress,      ///< sigma_zz = 0 (thin sheet). Default 2-D path.
  PlaneStrain,      ///< epsilon_zz = 0 (thick section / plane of symmetry).
  ThreeDimensional  ///< full isotropic elasticity on a solid mesh.
};

/// Spatial dimension a stress state belongs to (2 or 3).
constexpr int stress_state_dimension(StressState state) {
  return state == StressState::ThreeDimensional ? 3 : 2;
}

/// Configuration-file spelling of a stress state.
inline std::string to_string(StressState state) {
  switch (state) {
    case StressState::PlaneStress: return "plane_stress";
    case StressState::PlaneStrain: return "plane_strain";
    case StressState::ThreeDimensional: return "three_dimensional";
  }
  return "unknown";
}

}  // namespace sparlab
