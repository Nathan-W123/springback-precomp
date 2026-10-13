#include "sparlab/io/ResultWriter.hpp"

#include "sparlab/core/Exceptions.hpp"
#include "sparlab/core/Logging.hpp"
#include "sparlab/core/Version.hpp"
#include "sparlab/io/CsvWriter.hpp"

#include <ctime>
#include <fstream>
#include <string>
#include <utility>

namespace sparlab {
namespace {

/// [x, y] on a 2-D model, [x, y, z] on a 3-D one.
json::Value point_json(const Vector3& v, int dim) {
  json::Value out = json::Value::make_array();
  for (int i = 0; i < dim; ++i) out.push_back(json::Value::make_number(v(i)));
  return out;
}

}  // namespace

ResultWriter::ResultWriter(std::string directory, const Configuration& config)
    : directory_(std::move(directory)), config_(config) {
  ensure_directory(directory_);
  log::debug("result directory: ", directory_);
}

std::string ResultWriter::file(const std::string& name) const {
  return path_join(directory_, name);
}

void ResultWriter::write_json(const std::string& file_name,
                              const json::Value& value) const {
  const std::string path = file(file_name);
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out) throw IoError("cannot open '" + path + "' for writing");
  out << json::dump(value, 2) << '\n';
  out.flush();
  if (!out) throw IoError("failed while writing '" + path + "'");
}

void ResultWriter::write_config() const {
  write_json("config.json", config_.document);
}

void ResultWriter::write_mesh(const FemModel& model) const {
  const Mesh& mesh = model.mesh();
  const int dim = mesh.dim();
  json::Value doc = json::Value::make_object();
  doc.set("case", json::Value::make_string(config_.name));
  doc.set("element_type", json::Value::make_string(to_string(mesh.element_type())));
  doc.set("dim", json::Value::make_number(dim));
  doc.set("nodes_per_element", json::Value::make_number(mesh.nodes_per_elem()));

  json::Value nodes = json::Value::make_array();
  for (Index n = 0; n < mesh.num_nodes(); ++n) {
    nodes.push_back(point_json(mesh.node(n), dim));
  }
  doc.set("nodes_m", nodes);

  json::Value elements = json::Value::make_array();
  for (Index e = 0; e < mesh.num_elements(); ++e) {
    json::Value row = json::Value::make_array();
    const Index* en = mesh.element_nodes(e);
    for (int a = 0; a < mesh.nodes_per_elem(); ++a) {
      row.push_back(json::Value::make_number(en[a]));
    }
    elements.push_back(row);
  }
  doc.set("elements", elements);
  if (!model.single_material()) {
    // Index into the summary's "materials" of every element.
    json::Value materials = json::Value::make_array();
    for (Index e = 0; e < mesh.num_elements(); ++e) {
      materials.push_back(json::Value::make_number(model.element_material(e)));
    }
    doc.set("element_materials", materials);
  }

  // Prescribed DOFs, for the boundary-condition figure.
  const int ndpn = model.dofs_per_node();
  doc.set("dofs_per_node", json::Value::make_number(ndpn));
  json::Value constraints = json::Value::make_array();
  for (Index d : model.dofs().constrained_dofs()) {
    json::Value entry = json::Value::make_object();
    const int k = static_cast<int>(d % ndpn);
    entry.set("node", json::Value::make_number(d / ndpn));
    entry.set("component", json::Value::make_string(dof_component_name(k)));
    entry.set("value_m", json::Value::make_number(model.dofs().prescribed_value(d)));
    constraints.push_back(entry);
  }
  doc.set("prescribed_dofs", constraints);

  // The load cases with their applied nodal forces: none, the forming
  // analysis applies no load.
  json::Value cases = json::Value::make_array();
  for (const LoadCaseSpec& spec : model.load_case_specs()) {
    json::Value entry = json::Value::make_object();
    entry.set("name", json::Value::make_string(spec.name));
    entry.set("weight", json::Value::make_number(spec.weight));
    entry.set("nodal_forces", json::Value::make_array());
    cases.push_back(entry);
  }
  doc.set("load_cases", cases);

  write_json("mesh.json", doc);
}

json::Value make_provenance(const Configuration& config) {
  json::Value out = json::Value::make_object();
  out.set("code", json::Value::make_string("SparLab"));
  out.set("version", json::Value::make_string(SPARLAB_VERSION_STRING));
  out.set("build_type", json::Value::make_string(SPARLAB_BUILD_TYPE));
  out.set("compiler", json::Value::make_string(SPARLAB_COMPILER));
  out.set("eigen_version", json::Value::make_string(SPARLAB_EIGEN_VERSION));
  out.set("config_source", json::Value::make_string(config.source_path));
  out.set("case", json::Value::make_string(config.name));
  out.set("description", json::Value::make_string(config.description));
  out.set("units", json::Value::make_string("SI: m, N, Pa, kg, kg/m^3, Hz, J"));

  const std::time_t now = std::time(nullptr);
  char buffer[64] = {0};
  std::tm utc {};
#if defined(_WIN32)
  gmtime_s(&utc, &now);
#else
  gmtime_r(&now, &utc);
#endif
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc);
  out.set("timestamp_utc", json::Value::make_string(buffer));
  return out;
}

}  // namespace sparlab
