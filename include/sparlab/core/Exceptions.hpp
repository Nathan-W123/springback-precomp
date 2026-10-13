/// \file Exceptions.hpp
/// \brief Exception hierarchy used to surface actionable diagnostics.
///
/// SparLab never silently swallows a numerical failure. Every failure mode has
/// a dedicated exception type carrying a message that states *what* failed,
/// *which* quantity was out of range, and *what the user should change*.
#pragma once

#include <stdexcept>
#include <string>

namespace sparlab {

/// Base class for every error raised by SparLab.
class SparLabError : public std::runtime_error {
 public:
  explicit SparLabError(const std::string& what) : std::runtime_error(what) {}
};

/// Malformed, missing or contradictory configuration input.
class ConfigError : public SparLabError {
 public:
  explicit ConfigError(const std::string& what)
      : SparLabError("configuration error: " + what) {}
};

/// Invalid mesh: degenerate or inverted elements, out-of-range connectivity,
/// non-positive Jacobian determinant, duplicated nodes.
class MeshError : public SparLabError {
 public:
  explicit MeshError(const std::string& what)
      : SparLabError("mesh error: " + what) {}
};

/// The discrete model itself is ill-posed, e.g. insufficient displacement
/// boundary conditions leaving rigid-body or mechanism modes.
class ModelError : public SparLabError {
 public:
  explicit ModelError(const std::string& what)
      : SparLabError("model error: " + what) {}
};

/// A linear or eigenvalue solver reported failure, produced a non-finite
/// result, or left a residual above the requested tolerance.
class SolverError : public SparLabError {
 public:
  explicit SolverError(const std::string& what)
      : SparLabError("solver error: " + what) {}
};

/// Input/output failure (unwritable directory, unreadable file).
class IoError : public SparLabError {
 public:
  explicit IoError(const std::string& what)
      : SparLabError("I/O error: " + what) {}
};

}  // namespace sparlab
