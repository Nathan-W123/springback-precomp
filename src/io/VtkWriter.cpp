#include "sparlab/io/VtkWriter.hpp"

#include "sparlab/core/Exceptions.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <utility>

namespace sparlab {
namespace {

int vtk_cell_type(ElementType type) {
  switch (type) {
    case ElementType::Hex8: return 12;   // VTK_HEXAHEDRON
  }
  throw IoError("no VTK cell type registered for this element type");
}

void check_length(const std::string& name, Eigen::Index have, Eigen::Index want,
                  const char* entity) {
  if (have != want) {
    std::ostringstream os;
    os << "VTK array '" << name << "' has " << have << " entries but the mesh has "
       << want << " " << entity;
    throw IoError(os.str());
  }
}

}  // namespace

VtkWriter::VtkWriter(const Mesh& mesh, std::string title)
    : mesh_(mesh), title_(std::move(title)) {}

VtkWriter& VtkWriter::add_point_scalars(const std::string& name, const Vector& values) {
  check_length(name, values.size(), mesh_.num_nodes(), "nodes");
  point_scalars_.emplace_back(name, values);
  return *this;
}

VtkWriter& VtkWriter::add_point_vectors(const std::string& name, const Vector& values) {
  check_length(name, values.size(), mesh_.num_nodes() * mesh_.dim(), "DOFs");
  point_vectors_.emplace_back(name, values);
  return *this;
}

VtkWriter& VtkWriter::add_cell_scalars(const std::string& name, const Vector& values) {
  check_length(name, values.size(), mesh_.num_elements(), "elements");
  cell_scalars_.emplace_back(name, values);
  return *this;
}

void VtkWriter::write(const std::string& path) const {
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out) throw IoError("cannot open '" + path + "' for writing");
  out << std::setprecision(9);

  const Index nn = mesh_.num_nodes();
  const Index ne = mesh_.num_elements();
  const int npe = mesh_.nodes_per_elem();
  const int dim = mesh_.dim();

  out << "# vtk DataFile Version 3.0\n";
  out << title_ << "\n";
  out << "ASCII\n";
  out << "DATASET UNSTRUCTURED_GRID\n";

  out << "POINTS " << nn << " double\n";
  for (Index n = 0; n < nn; ++n) {
    const Vector3 x = mesh_.node(n);
    out << x.x() << ' ' << x.y() << ' ';
    if (dim == 3) {
      out << x.z() << '\n';
    } else {
      out << "0\n";
    }
  }

  out << "CELLS " << ne << ' ' << ne * (npe + 1) << "\n";
  for (Index e = 0; e < ne; ++e) {
    out << npe;
    const Index* nodes = mesh_.element_nodes(e);
    for (int a = 0; a < npe; ++a) out << ' ' << nodes[a];
    out << '\n';
  }

  const int cell_type = vtk_cell_type(mesh_.element_type());
  out << "CELL_TYPES " << ne << "\n";
  for (Index e = 0; e < ne; ++e) out << cell_type << '\n';

  if (!point_scalars_.empty() || !point_vectors_.empty()) {
    out << "POINT_DATA " << nn << "\n";
    for (const auto& field : point_scalars_) {
      out << "SCALARS " << field.first << " double 1\nLOOKUP_TABLE default\n";
      for (Index n = 0; n < nn; ++n) out << field.second(n) << '\n';
    }
    for (const auto& field : point_vectors_) {
      out << "VECTORS " << field.first << " double\n";
      for (Index n = 0; n < nn; ++n) {
        out << field.second(n * dim + 0) << ' ' << field.second(n * dim + 1) << ' ';
        if (dim == 3) {
          out << field.second(n * dim + 2) << '\n';
        } else {
          out << "0\n";
        }
      }
    }
  }

  if (!cell_scalars_.empty()) {
    out << "CELL_DATA " << ne << "\n";
    for (const auto& field : cell_scalars_) {
      out << "SCALARS " << field.first << " double 1\nLOOKUP_TABLE default\n";
      for (Index e = 0; e < ne; ++e) out << field.second(e) << '\n';
    }
  }

  out.flush();
  if (!out) throw IoError("failed while writing '" + path + "'");
}

}  // namespace sparlab
