/// \file Version.hpp
/// \brief Build provenance recorded in every result summary.
///
/// The build system supplies `SPARLAB_VERSION_STRING` and
/// `SPARLAB_BUILD_TYPE`; the compiler and Eigen strings are derived from
/// standard predefined macros so the header also works in a bare
/// `g++ -Iinclude` compile.
///
/// Where this fits: Simulations stage - the build provenance (version, build type, compiler,
///   Eigen) recorded in each run's summary.json.
/// Called from: io/ResultWriter.cpp:make_provenance
/// Calls into: Eigen/Core only (its version macros)
/// Start reading at: SPARLAB_VERSION_STRING, SPARLAB_COMPILER
#pragma once

#include <Eigen/Core>

#ifndef SPARLAB_VERSION_STRING
#define SPARLAB_VERSION_STRING "unversioned"
#endif

#ifndef SPARLAB_BUILD_TYPE
#define SPARLAB_BUILD_TYPE "unknown"
#endif

#define SPARLAB_STRINGIFY_IMPL(x) #x
#define SPARLAB_STRINGIFY(x) SPARLAB_STRINGIFY_IMPL(x)

#if defined(__clang__)
#define SPARLAB_COMPILER                                           \
  "clang " SPARLAB_STRINGIFY(__clang_major__) "." SPARLAB_STRINGIFY( \
      __clang_minor__) "." SPARLAB_STRINGIFY(__clang_patchlevel__)
#elif defined(__GNUC__)
#define SPARLAB_COMPILER                                     \
  "gcc " SPARLAB_STRINGIFY(__GNUC__) "." SPARLAB_STRINGIFY(  \
      __GNUC_MINOR__) "." SPARLAB_STRINGIFY(__GNUC_PATCHLEVEL__)
#elif defined(_MSC_VER)
#define SPARLAB_COMPILER "msvc " SPARLAB_STRINGIFY(_MSC_VER)
#else
#define SPARLAB_COMPILER "unknown"
#endif

#define SPARLAB_EIGEN_VERSION                                          \
  SPARLAB_STRINGIFY(EIGEN_WORLD_VERSION) "." SPARLAB_STRINGIFY(        \
      EIGEN_MAJOR_VERSION) "." SPARLAB_STRINGIFY(EIGEN_MINOR_VERSION)
