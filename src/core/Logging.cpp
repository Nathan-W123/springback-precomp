// Where this fits: see include/sparlab/core/Logging.hpp

#include "sparlab/core/Logging.hpp"

#include "sparlab/core/Exceptions.hpp"

#include <algorithm>
#include <cstdlib>

namespace sparlab {
namespace log {
namespace {

Level& mutable_level() {
  static Level current = [] {
    if (const char* env = std::getenv("SPARLAB_LOG_LEVEL")) {
      try {
        return parse_level(env);
      } catch (const ConfigError&) {
        std::cerr << "[warn ] SPARLAB_LOG_LEVEL='" << env
                  << "' is not a valid level; falling back to 'info'\n";
      }
    }
    return Level::Info;
  }();
  return current;
}

const char* tag(Level level) {
  switch (level) {
    case Level::Trace: return "[trace]";
    case Level::Debug: return "[debug]";
    case Level::Info: return "[info ]";
    case Level::Warn: return "[warn ]";
    case Level::Error: return "[error]";
    case Level::Silent: return "[quiet]";
  }
  return "[?????]";
}

}  // namespace

Level level() { return mutable_level(); }

void set_level(Level level) { mutable_level() = level; }

Level parse_level(const std::string& text) {
  std::string lower;
  lower.reserve(text.size());
  std::transform(text.begin(), text.end(), std::back_inserter(lower),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (lower == "trace") return Level::Trace;
  if (lower == "debug") return Level::Debug;
  if (lower == "info") return Level::Info;
  if (lower == "warn" || lower == "warning") return Level::Warn;
  if (lower == "error") return Level::Error;
  if (lower == "silent" || lower == "none" || lower == "off") return Level::Silent;
  throw ConfigError("unknown log level '" + text +
                    "' (expected trace|debug|info|warn|error|silent)");
}

void emit(Level level, const std::string& message) {
  std::cerr << tag(level) << ' ' << message << '\n';
}

}  // namespace log
}  // namespace sparlab
