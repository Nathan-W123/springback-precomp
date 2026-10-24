# Catch2 v3 is preferred; the system package is used when available and
# otherwise fetched, so a fresh clone can build its tests offline *or* online.
find_package(Catch2 3 QUIET)

if(NOT Catch2_FOUND)
  message(STATUS "SparLab: Catch2 3 not found locally, fetching it")
  include(FetchContent)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.2
  )
  FetchContent_MakeAvailable(Catch2)
endif()

set(SPARLAB_TEST_SOURCES
  test_quadrature_shape.cpp
  test_material.cpp
  test_assembly_solver.cpp
  test_io.cpp
  test_solid.cpp
  test_unstructured.cpp
  test_plasticity.cpp
  test_anisotropy.cpp
  test_logarithmic.cpp
  test_forming.cpp
  test_incompatible_modes.cpp
)

add_executable(sparlab_tests ${SPARLAB_TEST_SOURCES})
target_link_libraries(sparlab_tests PRIVATE sparlab Catch2::Catch2WithMain)
target_include_directories(sparlab_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(SPARLAB_ENABLE_WARNINGS AND NOT MSVC)
  target_compile_options(sparlab_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(Catch)
# `ctest -j N` runs N test processes at once, and each would start an OpenMP
# thread per core: with N near the core count the cores are oversubscribed
# and OpenMP's threads, which spin while they wait at a barrier, take the
# CPU from the ones with work. One thread per test case keeps the cores
# busy with work instead (the forming cases under ctest -j4 on four cores:
# 30 s instead of 78 s in all, the release case 3.9 s instead of 21.6 s).
# The thread-determinism tests set their own thread counts to compare them.
# A Debug build (no optimisation, Eigen's assertions) is about 40 times
# slower: the springback beam and the single-point forming cases take 17 s
# and 12 s in Release but 11 to 16 min in Debug on a GitHub runner, past
# the 900 s that is ample in Release.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(SPARLAB_TEST_TIMEOUT 3600)
else()
  set(SPARLAB_TEST_TIMEOUT 900)
endif()
catch_discover_tests(sparlab_tests
  TEST_PREFIX "sparlab."
  PROPERTIES TIMEOUT ${SPARLAB_TEST_TIMEOUT} ENVIRONMENT "OMP_NUM_THREADS=1"
)
