# SPDX-License-Identifier: BSD-3-Clause
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BENCHMARK_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    google_benchmark
    URL
        https://github.com/google/benchmark/archive/refs/tags/v1.9.4.tar.gz
    URL_HASH SHA256=b334658edd35efcf06a99d9be21e4e93e092bd5f95074c1673d5c8705d95c104
)
FetchContent_MakeAvailable(google_benchmark)
