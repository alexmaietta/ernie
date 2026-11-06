# SPDX-License-Identifier: BSD-3-Clause
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

set(MADRONA_BENCHMARK_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../external/madrona_benchmark"
    CACHE PATH
    "madrona_benchmark checkout that provides the patched madrona"
)

include("${MADRONA_BENCHMARK_DIR}/external/madrona/cmake/madrona_init.cmake")

project(
    ErnieBenchMadrona
    LANGUAGES
        C
        CXX
)

include(setup)
include(dependencies)

add_subdirectory("${MADRONA_BENCHMARK_DIR}/external" external)
add_subdirectory(src)
