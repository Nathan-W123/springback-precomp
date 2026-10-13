#include "sparlab/io/CsvWriter.hpp"

#include "sparlab/core/Exceptions.hpp"

#include <sys/stat.h>
#include <sys/types.h>

#include <cerrno>
#include <iomanip>
#include <sstream>

namespace sparlab {

std::string path_join(const std::string& directory, const std::string& name) {
  if (directory.empty()) return name;
  if (directory.back() == '/') return directory + name;
  return directory + "/" + name;
}

void ensure_directory(const std::string& path) {
  if (path.empty()) return;
  // Create every parent in turn so nested output directories work.
  std::string partial;
  std::istringstream stream(path);
  std::string segment;
  const bool absolute = path.front() == '/';
  bool first = true;
  while (std::getline(stream, segment, '/')) {
    if (segment.empty()) {
      if (first && absolute) partial = "/";
      first = false;
      continue;
    }
    first = false;
    if (partial.empty()) {
      partial = segment;
    } else if (partial == "/") {
      partial += segment;
    } else {
      partial += "/" + segment;
    }
    struct stat info {};
    if (stat(partial.c_str(), &info) == 0) {
      if (!S_ISDIR(info.st_mode)) {
        throw IoError("cannot create output directory '" + partial +
                      "': the path exists and is not a directory");
      }
      continue;
    }
    if (mkdir(partial.c_str(), 0775) != 0 && errno != EEXIST) {
      throw IoError("cannot create output directory '" + partial +
                    "': " + std::strerror(errno));
    }
  }
}

CsvWriter::CsvWriter(const std::string& path, const std::vector<std::string>& header,
                     int precision)
    : path_(path), columns_(header.size()) {
  out_.open(path, std::ios::out | std::ios::trunc);
  if (!out_) {
    throw IoError("cannot open '" + path + "' for writing");
  }
  out_ << std::setprecision(precision);
  for (std::size_t i = 0; i < header.size(); ++i) {
    if (i > 0) out_ << ',';
    out_ << header[i];
  }
  out_ << '\n';
}

void CsvWriter::row(Index index, const std::vector<Scalar>& values) {
  if (values.size() + 1 != columns_) {
    std::ostringstream os;
    os << "CSV row for '" << path_ << "' has " << values.size() + 1
       << " fields but the header declares " << columns_;
    throw IoError(os.str());
  }
  out_ << index;
  for (const Scalar v : values) out_ << ',' << v;
  out_ << '\n';
}

void CsvWriter::raw_row(const std::vector<std::string>& fields) {
  if (fields.size() != columns_) {
    std::ostringstream os;
    os << "CSV row for '" << path_ << "' has " << fields.size()
       << " fields but the header declares " << columns_;
    throw IoError(os.str());
  }
  for (std::size_t i = 0; i < fields.size(); ++i) {
    if (i > 0) out_ << ',';
    out_ << fields[i];
  }
  out_ << '\n';
}

void CsvWriter::close() {
  if (out_.is_open()) {
    out_.flush();
    if (!out_) throw IoError("failed while writing '" + path_ + "'");
    out_.close();
  }
}

CsvWriter::~CsvWriter() {
  // Destructors must not throw; a failure here is reported by close() when the
  // caller closes explicitly, which the result writer always does.
  if (out_.is_open()) out_.close();
}

}  // namespace sparlab
