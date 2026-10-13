/// \file AppSupport.hpp
/// \brief Command-line parsing and error reporting for sparlab_form.
///
/// The flags are documented by `sparlab_form --help`.
#pragma once

#include "sparlab/core/Logging.hpp"
#include "sparlab/core/Types.hpp"

#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace sparlab {
namespace app {

/// Parsed command line: flags with values plus boolean switches.
class CommandLine {
 public:
  /// \param argc/argv as received by main.
  /// \param known set of accepted option names (without the leading `--`);
  ///        an unknown option is an error, because a silently ignored flag
  ///        means the run did not do what the user asked.
  CommandLine(int argc, char** argv, const std::vector<std::string>& known);

  bool has(const std::string& name) const;
  std::string value(const std::string& name, const std::string& fallback = "") const;
  std::string require(const std::string& name) const;
  int integer(const std::string& name, int fallback) const;

 private:
  std::map<std::string, std::string> options_;
  std::vector<std::string> known_;
};

/// Apply `--verbosity` if present.
void apply_verbosity(const CommandLine& cli);

/// Print a usage block and return 0 (so apps can `return print_usage(...)`).
int print_usage(const std::string& program, const std::string& synopsis,
                const std::vector<std::pair<std::string, std::string>>& options);

/// Run `body`, converting SparLab exceptions into a diagnostic on stderr and a
/// non-zero exit status. Exit codes: 0 success, 2 configuration/model error,
/// 3 solver error, 4 I/O error, 5 unexpected exception.
int run_guarded(const std::function<int()>& body);

/// Default output directory for a case name.
std::string default_output_directory(const std::string& case_name);

/// Format a scalar with a fixed number of significant digits (for tables).
std::string format(Scalar value, int digits = 6);

}  // namespace app
}  // namespace sparlab
