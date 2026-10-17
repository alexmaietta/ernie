# SPDX-License-Identifier: BSD-3-Clause
cmake_minimum_required(VERSION 3.21)

project(
    Ernie
    VERSION 0.2.0
    DESCRIPTION
        "Efficient Renderer for Natural and Interplanetary Environments: a spectral Vulkan ray tracer"
    LANGUAGES C CXX
)

# Pre-release suffix such as .dev12, written by CI.
set(ERNIE_VERSION_SUFFIX "" CACHE STRING "Pre-release suffix of Ernie's version")
set(ERNIE_VERSION "${PROJECT_VERSION}${ERNIE_VERSION_SUFFIX}")

option(ERNIE_BUILD_TESTS "Build Ernie's tests" OFF)
option(ERNIE_BUILD_PYTHON "Build Ernie's Python module" OFF)
option(ERNIE_BUILD_APP "Build the ernie program" ON)
option(ERNIE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(ERNIE_ASSERTS "Keep assertions in optimised builds" OFF)

get_property(ernie_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(PROJECT_IS_TOP_LEVEL AND NOT ernie_multi_config AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
if(ERNIE_BUILD_PYTHON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

include(cmake/third_party/vulkan_headers.cmake)
include(cmake/third_party/volk.cmake)

add_library(ernie_build_options INTERFACE)
if(MSVC)
    target_compile_options(ernie_build_options INTERFACE /W4 /permissive- /Zc:__cplusplus /utf-8)
    target_compile_definitions(ernie_build_options INTERFACE NOMINMAX WIN32_LEAN_AND_MEAN)
else()
    target_compile_options(
        ernie_build_options
        INTERFACE
            -Wall
            -Wextra
            -Wpedantic
            -Wshadow
            -Wconversion
            -Wsign-conversion
            # Designated initializers leave Vulkan structs' other fields zero by design.
            -Wno-missing-field-initializers
    )
endif()
if(ERNIE_WARNINGS_AS_ERRORS)
    target_compile_options(ernie_build_options INTERFACE $<IF:$<CXX_COMPILER_ID:MSVC>,/WX,-Werror>)
endif()
if(ERNIE_ASSERTS)
    target_compile_options(
        ernie_build_options
        INTERFACE $<IF:$<CXX_COMPILER_ID:MSVC>,/UNDEBUG,-UNDEBUG>
    )
endif()

add_subdirectory(ernie)

if(ERNIE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(ernie/test)
endif()

if(PROJECT_IS_TOP_LEVEL AND NOT SKBUILD)
    include(cmake/ErniePackaging.cmake)
endif()
