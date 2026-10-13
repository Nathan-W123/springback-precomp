#include "AppSupport.hpp"

#include "sparlab/core/Exceptions.hpp"

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace sparlab {
namespace app {

CommandLine::CommandLine(int argc, char** argv, const std::vector<std::string>& known)
    : known_(known) {
  // Options that never take a value.
  const std::vector<std::string> switches = {"help", "strict-config", "no-vtk", "version"};

  for (int i = 1; i < argc; ++i) {
    std::string token = argv[i];
    if (token.rfind("--", 0) != 0) {
      throw ConfigError("unexpected positional argument '" + token +
                        "'; all inputs are passed with --option value");
    }
    token = token.substr(2);
    std::string name = token;
    std::string value;
    bool inline_value = false;
    const std::size_t eq = token.find('=');
    if (eq != std::string::npos) {
      name = token.substr(0, eq);
      value = token.substr(eq + 1);
      inline_value = true;
    }

    if (std::find(known_.begin(), known_.end(), name) == known_.end()) {
      std::ostringstream os;
      os << "unknown option '--" << name << "'. Accepted options:";
      for (const std::string& k : known_) os << " --" << k;
      throw ConfigError(os.str());
    }

    const bool is_switch =
        std::find(switches.begin(), switches.end(), name) != switches.end();
    if (is_switch) {
      options_[name] = inline_value ? value : "true";
      continue;
    }
    if (!inline_value) {
      if (i + 1 >= argc) {
        throw ConfigError("option '--" + name + "' requires a value");
      }
      value = argv[++i];
    }
    options_[name] = value;
  }
}

bool CommandLine::has(const std::string& name) const {
  return options_.find(name) != options_.end();
}

std::string CommandLine::value(const std::string& name,
                               const std::string& fallback) const {
  const auto it = options_.find(name);
  return it == options_.end() ? fallback : it->second;
}

std::string CommandLine::require(const std::string& name) const {
  const auto it = options_.find(name);
  if (it == options_.end()) {
    throw ConfigError("option '--" + name + "' is required");
  }
  return it->second;
}

int CommandLine::integer(const std::string& name, int fallback) const {
  if (!has(name)) return fallback;
  const std::string text = value(name);
  try {
    std::size_t consumed = 0;
    const int parsed = std::stoi(text, &consumed);
    if (consumed != text.size()) throw std::invalid_argument("trailing characters");
    return parsed;
  } catch (const std::exception&) {
    throw ConfigError("option '--" + name + "' expects an integer, got '" + text + "'");
  }
}

void apply_verbosity(const CommandLine& cli) {
  if (cli.has("verbosity")) {
    log::set_level(log::parse_level(cli.value("verbosity")));
  }
}

int print_usage(const std::string& program, const std::string& synopsis,
                const std::vector<std::pair<std::string, std::string>>& options) {
  std::cout << "usage: " << program << " " << synopsis << "\n\noptions:\n";
  std::size_t width = 0;
  for (const auto& opt : options) width = std::max(width, opt.first.size());
  for (const auto& opt : options) {
    std::cout << "  " << std::left << std::setw(static_cast<int>(width) + 2) << opt.first
              << opt.second << "\n";
  }
  std::cout << "\nAll quantities are in SI units (m, N, Pa, kg).\n";
  return 0;
}

int run_guarded(const std::function<int()>& body) {
  try {
    return body();
  } catch (const ConfigError& e) {
    std::cerr << "[error] " << e.what() << "\n";
    return 2;
  } catch (const MeshError& e) {
    std::cerr << "[error] " << e.what() << "\n";
    return 2;
  } catch (const ModelError& e) {
    std::cerr << "[error] " << e.what() << "\n";
    return 2;
  } catch (const SolverError& e) {
    std::cerr << "[error] " << e.what() << "\n";
    return 3;
  } catch (const IoError& e) {
    std::cerr << "[error] " << e.what() << "\n";
    return 4;
  } catch (const SparLabError& e) {
    std::cerr << "[error] " << e.what() << "\n";
    return 5;
  } catch (const std::exception& e) {
    std::cerr << "[error] unexpected failure: " << e.what() << "\n";
    return 5;
  }
}

std::string default_output_directory(const std::string& case_name) {
  std::string sanitised;
  for (char c : case_name) {
    const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                    (c >= '0' && c <= '9') || c == '-' || c == '_';
    sanitised.push_back(ok ? c : '_');
  }
  if (sanitised.empty()) sanitised = "case";
  return "results/" + sanitised;
}

std::string format(Scalar value, int digits) {
  std::ostringstream os;
  os << std::setprecision(digits) << value;
  return os.str();
}

}  // namespace app
}  // namespace sparlab
