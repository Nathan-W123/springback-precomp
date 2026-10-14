cmake_minimum_required(VERSION 3.16)

project(SparLab
        VERSION 1.0.0
        DESCRIPTION "Finite-element incremental-forming and springback solver"
        LANGUAGES CXX)

# ----------------------------------------------------------------------------
# Options
# ----------------------------------------------------------------------------
option(SPARLAB_BUILD_TESTS     "Build the Catch2 test suite"            ON)
option(SPARLAB_BUILD_APPS      "Build the sparlab_form executable"      ON)
option(SPARLAB_ENABLE_WARNINGS "Enable a strict warning set"            ON)
option(SPARLAB_WARNINGS_AS_ERRORS "Treat compiler warnings as errors"   OFF)
option(SPARLAB_ENABLE_OPENMP   "Use OpenMP where the code supports it"  ON)
# SuiteSparse (CHOLMOD supernodal Cholesky, UMFPACK LU) for the forming
# analysis's tangents: AUTO uses it when found, ON requires it, OFF never
# uses it. Everything builds and passes without it (Eigen's own sparse
# factorisations are the fallback).
set(SPARLAB_WITH_CHOLMOD AUTO CACHE STRING "Use SuiteSparse CHOLMOD/UMFPACK: AUTO, ON or OFF")
set_property(CACHE SPARLAB_WITH_CHOLMOD PROPERTY STRINGS AUTO ON OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  message(STATUS "SparLab: CMAKE_BUILD_TYPE not set, defaulting to Release")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ----------------------------------------------------------------------------
# Dependencies
# ----------------------------------------------------------------------------
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
message(STATUS "SparLab: found Eigen ${Eigen3_VERSION}")

if(SPARLAB_ENABLE_OPENMP)
  find_package(OpenMP QUIET)
  if(OpenMP_CXX_FOUND)
    message(STATUS "SparLab: OpenMP enabled (${OpenMP_CXX_VERSION})")
  else()
    message(STATUS "SparLab: OpenMP not found, building single-threaded")
  endif()
endif()

set(SPARLAB_CHOLMOD_FOUND FALSE)
string(TOUPPER "${SPARLAB_WITH_CHOLMOD}" _sparlab_cholmod_mode)
if(NOT _sparlab_cholmod_mode MATCHES "^(AUTO|ON|OFF)$")
  message(FATAL_ERROR "SparLab: SPARLAB_WITH_CHOLMOD must be AUTO, ON or OFF, got "
                      "'${SPARLAB_WITH_CHOLMOD}'")
endif()
if(NOT _sparlab_cholmod_mode STREQUAL "OFF")
  find_path(SPARLAB_CHOLMOD_INCLUDE_DIR cholmod.h PATH_SUFFIXES suitesparse)
  find_path(SPARLAB_UMFPACK_INCLUDE_DIR umfpack.h PATH_SUFFIXES suitesparse)
  find_library(SPARLAB_CHOLMOD_LIBRARY NAMES cholmod)
  find_library(SPARLAB_UMFPACK_LIBRARY NAMES umfpack)
  find_library(SPARLAB_SUITESPARSECONFIG_LIBRARY NAMES suitesparseconfig)
  find_library(SPARLAB_AMD_LIBRARY NAMES amd)
  find_library(SPARLAB_COLAMD_LIBRARY NAMES colamd)
  if(SPARLAB_CHOLMOD_INCLUDE_DIR AND SPARLAB_UMFPACK_INCLUDE_DIR AND SPARLAB_CHOLMOD_LIBRARY
     AND SPARLAB_UMFPACK_LIBRARY AND SPARLAB_SUITESPARSECONFIG_LIBRARY AND SPARLAB_AMD_LIBRARY
     AND SPARLAB_COLAMD_LIBRARY)
    set(SPARLAB_CHOLMOD_FOUND TRUE)
    message(STATUS "SparLab: SuiteSparse CHOLMOD/UMFPACK enabled (${SPARLAB_CHOLMOD_LIBRARY})")
  elseif(_sparlab_cholmod_mode STREQUAL "ON")
    message(FATAL_ERROR "SparLab: SPARLAB_WITH_CHOLMOD=ON but CHOLMOD/UMFPACK were not found")
  else()
    message(STATUS "SparLab: SuiteSparse not found, the forming analysis uses Eigen's "
                   "factorisations")
  endif()
endif()

# ----------------------------------------------------------------------------
# Core library
# ----------------------------------------------------------------------------
add_library(sparlab STATIC
  src/core/Logging.cpp
  src/mesh/Mesh.cpp
  src/mesh/StructuredMesh.cpp
  src/material/Material.cpp
  src/material/Plasticity.cpp
  src/material/LogarithmicStrain.cpp
  src/elements/Quadrature.cpp
  src/elements/Element.cpp
  src/elements/Hex8.cpp
  src/elements/Hex8Incompatible.cpp
  src/elements/FaceGeometry.cpp
  src/fem/DofManager.cpp
  src/fem/Selector.cpp
  src/fem/BoundaryConditions.cpp
  src/fem/FemModel.cpp
  src/fem/Assembler.cpp
  src/fem/TotalLagrangian.cpp
  src/fem/Elastoplastic.cpp
  src/fem/IncompatibleModes.cpp
  src/fem/RigidTool.cpp
  src/fem/Forming.cpp
  src/fem/ExplicitKernel.cpp
  src/fem/ExplicitDynamics.cpp
  src/io/Json.cpp
  src/io/Config.cpp
  src/io/CsvWriter.cpp
  src/io/VtkWriter.cpp
  src/io/MeshReader.cpp
  src/io/ResultWriter.cpp
  src/io/FormingWriter.cpp
)

target_include_directories(sparlab PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(sparlab PUBLIC Eigen3::Eigen)

if(SPARLAB_CHOLMOD_FOUND)
  target_include_directories(sparlab SYSTEM PRIVATE ${SPARLAB_CHOLMOD_INCLUDE_DIR}
                             ${SPARLAB_UMFPACK_INCLUDE_DIR})
  target_link_libraries(sparlab PUBLIC ${SPARLAB_CHOLMOD_LIBRARY} ${SPARLAB_UMFPACK_LIBRARY}
                        ${SPARLAB_AMD_LIBRARY} ${SPARLAB_COLAMD_LIBRARY}
                        ${SPARLAB_SUITESPARSECONFIG_LIBRARY})
  target_compile_definitions(sparlab PRIVATE SPARLAB_HAVE_CHOLMOD=1)
endif()

if(SPARLAB_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(sparlab PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(sparlab PUBLIC SPARLAB_HAVE_OPENMP=1)
endif()

target_compile_definitions(sparlab PUBLIC
  SPARLAB_VERSION_STRING="${PROJECT_VERSION}"
  SPARLAB_BUILD_TYPE="$<CONFIG>"
)

# Eigen's own asserts are valuable during development but very slow; keep them
# in Debug only.
target_compile_definitions(sparlab PUBLIC
  $<$<NOT:$<CONFIG:Debug>>:EIGEN_NO_DEBUG>
)

if(SPARLAB_ENABLE_WARNINGS)
  if(MSVC)
    target_compile_options(sparlab PRIVATE /W4 /permissive-)
  else()
    target_compile_options(sparlab PRIVATE
      -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor -Wcast-align
      -Wunused -Woverloaded-virtual -Wdouble-promotion -Wformat=2)
  endif()
  if(SPARLAB_WARNINGS_AS_ERRORS)
    if(MSVC)
      target_compile_options(sparlab PRIVATE /WX)
    else()
      target_compile_options(sparlab PRIVATE -Werror)
    endif()
  endif()
endif()

# ----------------------------------------------------------------------------
# Applications
# ----------------------------------------------------------------------------
if(SPARLAB_BUILD_APPS)
  add_library(sparlab_app_support STATIC apps/AppSupport.cpp)
  target_link_libraries(sparlab_app_support PUBLIC sparlab)
  target_include_directories(sparlab_app_support PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/apps)

  add_executable(sparlab_form apps/sparlab_form.cpp)
  target_link_libraries(sparlab_form PRIVATE sparlab_app_support)

  # The git revision `sparlab_form --version` prints, regenerated at every
  # build (a result cache keys on it), "-dirty" with uncommitted changes.
  set(SPARLAB_GIT_SHA_HEADER ${CMAKE_BINARY_DIR}/generated/sparlab_git_sha.h)
  add_custom_target(sparlab_git_sha
    COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
            -DOUTPUT=${SPARLAB_GIT_SHA_HEADER} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/GitSha.cmake
    BYPRODUCTS ${SPARLAB_GIT_SHA_HEADER}
    COMMENT "SparLab: recording the git revision")
  add_dependencies(sparlab_form sparlab_git_sha)
  target_include_directories(sparlab_form PRIVATE ${CMAKE_BINARY_DIR}/generated)
endif()

# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
if(SPARLAB_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
message(STATUS "SparLab configuration summary")
message(STATUS "  build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ std    : ${CMAKE_CXX_STANDARD}")
message(STATUS "  tests      : ${SPARLAB_BUILD_TESTS}")
message(STATUS "  apps       : ${SPARLAB_BUILD_APPS}")
message(STATUS "  SuiteSparse: ${SPARLAB_CHOLMOD_FOUND}")
