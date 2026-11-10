# SPDX-License-Identifier: BSD-3-Clause
cmake_minimum_required(VERSION 3.25)
if(POLICY CMP0156)
    cmake_policy(SET CMP0156 NEW)
endif()
set(CMAKE_OSX_DEPLOYMENT_TARGET 26.0 CACHE STRING "")
project(
    Ernie
    VERSION 0.1.0
    DESCRIPTION
        "Efficient Renderer for Natural and Interplanetary Environments: a physically-based, real-time, spectral renderer for planetary robotics and sensing"
    HOMEPAGE_URL "https://github.com/alexmaietta/ernie"
    LANGUAGES
        C
        CXX
)
set(ERNIE_PRERELEASE "")

option(ERNIE_BUILD_TESTS "Build Ernie's tests" OFF)
option(ERNIE_BUILD_BENCHMARKS "Build Ernie's benchmarks" OFF)
option(ERNIE_BUILD_PYTHON "Build Ernie's Python bindings" OFF)
option(ERNIE_BUILD_VIEWER "Build the Ernie viewer" OFF)
option(ERNIE_INSTALL "Install Ernie's library and headers" OFF)
option(ERNIE_COVERAGE "Instrument Ernie for code coverage (GCC or Clang)" OFF)
option(ERNIE_WARNINGS_AS_ERRORS "Fail on compiler and linker warnings in Ernie's own targets" OFF)
option(ERNIE_DEBIAN_LIBDIR "Install the libraries in Debian's lib/<arch>-linux-gnu" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(WIN32)
    add_compile_definitions(
        NOMINMAX
        WIN32_LEAN_AND_MEAN
        _CRT_SECURE_NO_WARNINGS
    )
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
set(ERNIE_GENERATED_INCLUDE_DIR "${PROJECT_BINARY_DIR}/include")

include(FetchContent)
if(ERNIE_DEBIAN_LIBDIR)
    set(CMAKE_INSTALL_LIBDIR
        "lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu"
        CACHE PATH
        "Object code libraries (LIBDIR)"
    )
endif()
include(GNUInstallDirs)

include(cmake/third_party/fmt.cmake)
include(cmake/third_party/vulkan_headers.cmake)
include(cmake/third_party/volk.cmake)
include(cmake/third_party/vma.cmake)
include(cmake/third_party/dlpack.cmake)
include(cmake/third_party/nvtx.cmake)
include(cmake/third_party/slang.cmake)
include(cmake/third_party/mikktspace.cmake)
include(cmake/third_party/stb.cmake)
include(cmake/third_party/fonts.cmake)
include(cmake/third_party/openexr.cmake)
include(cmake/third_party/tinyobjloader.cmake)
include(cmake/third_party/cgltf.cmake)
include(cmake/third_party/tinyusd.cmake)

add_subdirectory(ernie)
add_subdirectory(ernie/app)

if(ERNIE_BUILD_TESTS OR ERNIE_BUILD_BENCHMARKS)
    enable_testing()
    if(ERNIE_BUILD_TESTS)
        include(cmake/third_party/google_test.cmake)
    endif()
    if(ERNIE_BUILD_BENCHMARKS)
        include(cmake/third_party/google_benchmark.cmake)
    endif()
    add_subdirectory(tests)
endif()

if(ERNIE_INSTALL)
    include(cmake/ErnieInstall.cmake)
    include(cmake/ErniePackaging.cmake)
endif()
