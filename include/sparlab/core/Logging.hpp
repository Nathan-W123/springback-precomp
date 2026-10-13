/// \file Logging.hpp
/// \brief Minimal leveled logger writing to stderr.
///
/// Diagnostics go to stderr so that stdout can carry machine-readable output.
/// The verbosity is a process-wide setting controlled by the `--verbosity`
/// command-line flag or the `SPARLAB_LOG_LEVEL` environment variable.
///
/// Where this fits: Simulations stage - sparlab_form's diagnostics on stderr, which precomp's
///   fea/runner.py:run_deck saves in sparlab_form.log.
/// Called from: apps/AppSupport.cpp:apply_verbosity (set_level); log::info and log::warn
///   across the library (e.g. fem/Forming.cpp, io/Config.cpp)
/// Calls into: core/Exceptions (parse_level)
/// Start reading at: log::info, log::set_level
#pragma once

#include <iostream>
#include <sstream>
#include <string>

namespace sparlab {
namespace log {

enum class Level : int { Trace = 0, Debug = 1, Info = 2, Warn = 3, Error = 4, Silent = 5 };

/// Current global verbosity threshold; messages below it are discarded.
Level level();
void set_level(Level level);

/// Parse a textual level ("trace", "debug", "info", "warn", "error", "silent").
/// \throws ConfigError if the string is not recognised.
Level parse_level(const std::string& text);

/// Emit a fully formed message. Prefer the variadic helpers below.
void emit(Level level, const std::string& message);

namespace detail {
inline void append(std::ostringstream&) {}

template <typename T, typename... Rest>
void append(std::ostringstream& os, const T& value, const Rest&... rest) {
  os << value;
  append(os, rest...);
}

template <typename... Args>
void write(Level lvl, const Args&... args) {
  if (static_cast<int>(lvl) < static_cast<int>(level())) return;
  std::ostringstream os;
  append(os, args...);
  emit(lvl, os.str());
}
}  // namespace detail

template <typename... Args>
void trace(const Args&... args) {
  detail::write(Level::Trace, args...);
}
template <typename... Args>
void debug(const Args&... args) {
  detail::write(Level::Debug, args...);
}
template <typename... Args>
void info(const Args&... args) {
  detail::write(Level::Info, args...);
}
template <typename... Args>
void warn(const Args&... args) {
  detail::write(Level::Warn, args...);
}

}  // namespace log
}  // namespace sparlab
