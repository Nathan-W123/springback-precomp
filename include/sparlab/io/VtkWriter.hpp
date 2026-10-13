/// \file VtkWriter.hpp
/// \brief Legacy ASCII VTK (`UNSTRUCTURED_GRID`) export for ParaView.
///
/// The legacy format is used deliberately: it is a single self-describing text
/// file with no XML or compression dependencies, and ParaView, VisIt and
/// pyvista all read it. Hex8 elements are written as VTK_HEXAHEDRON (cell
/// type 12). The forming writer (FormingWriter.hpp) names its fields:
/// point data `displacement` and `displacement_magnitude`, cell data
/// `eq_plastic_strain` and `von_mises`.
#pragma once

#include "sparlab/core/Types.hpp"
#include "sparlab/mesh/Mesh.hpp"

#include <string>
#include <utility>
#include <vector>

namespace sparlab {

/// Accumulates named point/cell arrays and writes one legacy VTK file.
class VtkWriter {
 public:
  explicit VtkWriter(const Mesh& mesh, std::string title = "SparLab field data");

  /// Add a scalar array over nodes (length num_nodes).
  VtkWriter& add_point_scalars(const std::string& name, const Vector& values);

  /// Add a vector array over nodes; written as a 3-vector (z = 0 in 2-D).
  /// \param values length dim*num_nodes in node-major DOF ordering.
  VtkWriter& add_point_vectors(const std::string& name, const Vector& values);

  /// Add a scalar array over elements (length num_elements).
  VtkWriter& add_cell_scalars(const std::string& name, const Vector& values);

  /// Write the file.
  /// \throws IoError on failure.
  void write(const std::string& path) const;

 private:
  const Mesh& mesh_;
  std::string title_;
  std::vector<std::pair<std::string, Vector>> point_scalars_;
  std::vector<std::pair<std::string, Vector>> point_vectors_;
  std::vector<std::pair<std::string, Vector>> cell_scalars_;
};

}  // namespace sparlab
