# SparLab convenience targets.
#
# Each target is a thin wrapper over a CMake or ctest invocation, so nothing
# is hidden and every command can also be typed by hand.
#
#   make configure   configure the CMake build (Release by default)
#   make build       build the library, sparlab_form and the tests
#   make test        run the whole Catch2 suite through ctest
#   make all         build, test
#   make clean       remove the build directory
#   make distclean   also remove results/

BUILD_DIR   ?= build
BUILD_TYPE  ?= Release
JOBS        ?= $(shell nproc 2>/dev/null || echo 4)
GENERATOR   ?= $(shell command -v ninja >/dev/null 2>&1 && echo Ninja || echo "Unix Makefiles")
RESULTS_DIR ?= results

CMAKE_FLAGS ?= -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

.PHONY: all configure build test clean distclean help

help:
	@sed -n '3,11p' Makefile

configure:
	cmake -S . -B $(BUILD_DIR) -G "$(GENERATOR)" $(CMAKE_FLAGS)

build: configure
	cmake --build $(BUILD_DIR) -j $(JOBS)

test: build
	cd $(BUILD_DIR) && ctest --output-on-failure -j $(JOBS)

all: build test

clean:
	rm -rf $(BUILD_DIR)

distclean: clean
	rm -rf $(RESULTS_DIR)
