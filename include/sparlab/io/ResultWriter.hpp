/// \file ResultWriter.hpp
/// \brief The output directory of a run and its common files.
///
/// Layout produced for a case in `<out>` (the forming files are
/// FormingWriter.hpp's):
/// \code
///   <out>/summary.json              scalars, timings, provenance
///   <out>/config.json               verbatim echo of the input deck
///   <out>/mesh.json                 nodes, connectivity, prescribed DOFs
/// \endcode
/// Every file is plain text.
///
/// Where this fits: Simulations stage - owns the run's output directory and writes
///   config.json, mesh.json and the JSON documents (summary.json) into it.
/// Called from: apps/sparlab_form.cpp:main, io/FormingWriter.cpp:write_forming_step (file),
///   io/FormingWriter.cpp:forming_summary_json (make_provenance)
/// Calls into: io/Json, io/CsvWriter (ensure_directory, path_join), core/Version
/// Start reading at: ResultWriter, make_provenance
#pragma once

#include "sparlab/io/Config.hpp"
#include "sparlab/io/Json.hpp"

#include <string>

namespace sparlab {

class ResultWriter {
 public:
  /// Create (or reuse) the output directory.
  ResultWriter(std::string directory, const Configuration& config);

  const std::string& directory() const { return directory_; }

  /// Echo the input deck so a result directory is self-describing.
  void write_config() const;

  /// Nodes, connectivity, prescribed DOFs and the load cases.
  void write_mesh(const FemModel& model) const;

  /// Write an arbitrary JSON document into the directory.
  void write_json(const std::string& file_name, const json::Value& value) const;

  /// Path of a file inside the output directory.
  std::string file(const std::string& name) const;

 private:
  std::string directory_;
  const Configuration& config_;
};

/// Common provenance block: version, build type, timestamp.
json::Value make_provenance(const Configuration& config);

}  // namespace sparlab
