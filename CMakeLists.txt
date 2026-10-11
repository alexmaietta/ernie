# SPDX-License-Identifier: BSD-3-Clause
cmake_minimum_required(VERSION 3.22)
# Files extracted from URL downloads (Slang, fonts, JSON) get the time of extraction, so a new
# download rebuilds what depends on them.
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

project(
    Ernie
    VERSION 0.1.0
    DESCRIPTION
        "Efficient Renderer for Natural and Interplanetary Environments: a spectral Vulkan renderer"
    LANGUAGES C CXX)
# A pre-release suffix for builds between releases, such as ".dev12" (PEP 440): CI sets it to the
# commits since the last tag on pushes to main; releases, tagged vX.Y.Z, leave it empty. The wheel
# reads it here too (pyproject.toml), and the packages turn it into "~dev12".
# One line, as CI rewrites it and the wheel's version regex reads it.
# cmake-format: off
set(ERNIE_VERSION_SUFFIX "" CACHE STRING "Pre-release suffix of Ernie's version, such as .dev12")
# cmake-format: on
set(ERNIE_VERSION "${PROJECT_VERSION}${ERNIE_VERSION_SUFFIX}")

option(ERNIE_BUILD_TESTS "Build Ernie's tests" OFF)
option(ERNIE_BUILD_PYTHON "Build Ernie's Python module" OFF)
option(ERNIE_BUILD_DEMOS
       "Build a program for each of Ernie's C++ demos" OFF)
option(
    ERNIE_BUILD_APP
    "Build Ernie's app, the GUI viewer (GLFW, Dear ImGui, ImPlot, ImGuizmo, fonts)"
    OFF)
option(
    ERNIE_TRACY
    "Instrument Ernie for the Tracy profiler (CPU and Vulkan GPU zones, on demand)"
    OFF)
option(ERNIE_CCACHE "Compile through ccache when it is installed" ON)
# libernie and the install rules need CMake 3.24's whole-archive linking; Windows needs export
# annotations first.
option(
    ERNIE_INSTALL
    "Build libernie, the shared library, and Ernie's install rules and packages"
    OFF)
if(ERNIE_INSTALL AND CMAKE_VERSION VERSION_LESS 3.24)
    message(FATAL_ERROR "ERNIE_INSTALL needs CMake 3.24 or later")
endif()
if(ERNIE_INSTALL AND WIN32)
    message(
        FATAL_ERROR "ERNIE_INSTALL is not supported on Windows yet")
endif()

set(ERNIE_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}"
    CACHE INTERNAL "Ernie source directory")

# An optimised build with debug info unless a build type is given: Ernie's CPU side (frame setup,
# culling lists, scene sync) is many times slower without optimisation.
get_property(ernie_multi_config GLOBAL
             PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(PROJECT_IS_TOP_LEVEL
   AND NOT ernie_multi_config
   AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE
        RelWithDebInfo
        CACHE STRING "Build type" FORCE)
endif()

if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
endif()

# A compiler cache, so reconfigures, branch switches and clean builds reuse earlier compiles; a
# launcher given on the command line wins.
if(PROJECT_IS_TOP_LEVEL
   AND ERNIE_CCACHE
   AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
    find_program(ERNIE_CCACHE_PROGRAM ccache)
    if(ERNIE_CCACHE_PROGRAM)
        set(CMAKE_C_COMPILER_LAUNCHER "${ERNIE_CCACHE_PROGRAM}")
        set(CMAKE_CXX_COMPILER_LAUNCHER "${ERNIE_CCACHE_PROGRAM}")
    endif()
endif()

# The Python module and libernie link Ernie's static libraries into shared objects.
if(ERNIE_BUILD_PYTHON OR ERNIE_INSTALL)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

include(cmake/third_party/vulkan_headers.cmake)
include(cmake/third_party/volk.cmake)
include(cmake/third_party/vma.cmake)
include(cmake/third_party/slang.cmake)
if(APPLE)
    include(cmake/third_party/moltenvk.cmake)
endif()
include(cmake/third_party/glm.cmake)
include(cmake/third_party/dlpack.cmake)
include(cmake/third_party/stb.cmake)
include(cmake/third_party/openexr.cmake)
include(cmake/third_party/meshoptimizer.cmake)
include(cmake/third_party/tracy.cmake)
include(cmake/third_party/json.cmake)

# Warning and coverage flags: TerrainGen's when built inside it, Ernie's own otherwise.
add_library(ernie_build_options INTERFACE)
if(TARGET terraingen_warnings)
    target_link_libraries(
        ernie_build_options INTERFACE terraingen_warnings
                                      terraingen_coverage)
elseif(MSVC)
    # Standard conformance (two-phase lookup, __cplusplus as the real standard), UTF-8 sources, and
    # room for large objects (the templated bindings and schemas).
    target_compile_options(
        ernie_build_options INTERFACE /W4 /permissive-
                                      /Zc:__cplusplus /utf-8 /bigobj)
    # M_PI and the other math constants, which MSVC's <cmath> declares only on request; and
    # <windows.h> lean and without min and max macros, wherever it comes in.
    target_compile_definitions(
        ernie_build_options INTERFACE _USE_MATH_DEFINES
                                      WIN32_LEAN_AND_MEAN NOMINMAX)
else()
    target_compile_options(
        ernie_build_options
        INTERFACE -Wall
                  -Wextra
                  -Wpedantic
                  -Wshadow
                  -Wconversion
                  -Wsign-conversion
                  -Wdouble-promotion)
endif()

add_subdirectory(ernie)

# The demos' libcurl, found before the packaging so the packages can
# depend on it.
if(TARGET ernie_app)
    include(cmake/third_party/curl.cmake)
endif()

if(ERNIE_INSTALL)
    include(cmake/ErnieInstall.cmake)
    include(cmake/ErniePackaging.cmake)
endif()

# The demos: built into the `ernie` program and the Python module with the app, and each as a
# program of its own with ERNIE_BUILD_DEMOS.
if(TARGET ernie_app)
    add_subdirectory(demos)
endif()

if(ERNIE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(ernie/test)
endif()
