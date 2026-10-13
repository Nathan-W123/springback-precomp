/// \file CsvWriter.hpp
/// \brief Minimal CSV writer with explicit precision.
///
/// Numerical output is written with 17 significant digits by default so that a
/// value read back into a double is bit-identical to the one computed. Column
/// headers carry units in brackets, e.g. `ux[m]`.
///
/// Where this fits: Simulations stage - the CSV result files (step nodes and elements,
///   tool_forces.csv) and the output-directory helpers.
/// Called from: io/FormingWriter.cpp:write_forming_step, io/FormingWriter.cpp:write_tool_forces,
///   io/ResultWriter.cpp:ResultWriter::ResultWriter (ensure_directory)
/// Calls into: core/Exceptions (IoError)
/// Start reading at: CsvWriter::row, ensure_directory
#pragma once

#include "sparlab/core/Types.hpp"

#include <fstream>
#include <string>
#include <vector>

namespace sparlab {

class CsvWriter {
 public:
  /// Open `path` and write the header row.
  /// \throws IoError when the file cannot be opened.
  CsvWriter(const std::string& path, const std::vector<std::string>& header,
            int precision = 17);

  /// Write a row of an index column and numeric values. The number of fields
  /// must match the header.
  void row(Index index, const std::vector<Scalar>& values);

  /// Write a row of arbitrary strings (already formatted).
  void raw_row(const std::vector<std::string>& fields);

  void close();

  ~CsvWriter();

 private:
  std::ofstream out_;
  std::string path_;
  std::size_t columns_ = 0;
};

/// Ensure a directory exists, creating parents as needed.
/// \throws IoError when the path exists as a file or cannot be created.
void ensure_directory(const std::string& path);

/// Join a directory and a file name with '/'.
std::string path_join(const std::string& directory, const std::string& name);

}  // namespace sparlab
