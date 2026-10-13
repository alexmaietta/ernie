# SPDX-License-Identifier: BSD-3-Clause
# ernie_benchmarks: the scenarios, presets and suites of benchmarks/, one program. Each source
# registers what it defines at static initialization, so they link in as objects, not a library.
file(GLOB ernie_benchmark_scenarios CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/scenarios/*.cpp")
add_executable(
    ernie_benchmarks
    main.cpp
    bench.cpp
    batch.cpp
    instrument.cpp
    statistics.cpp
    metrics.cpp
    result.cpp
    images.cpp
    runner.cpp
    autotune.cpp
    compare.cpp
    presets.cpp
    suites.cpp
    ${ernie_benchmark_scenarios})
target_link_libraries(
    ernie_benchmarks PRIVATE ernie ernie_json ernie_stb
                             ernie_build_options)

# The commit and build the results name.
find_package(Git QUIET)
set(ernie_benchmark_commit "unknown")
if(GIT_FOUND)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" describe --always --dirty
        WORKING_DIRECTORY "${ERNIE_SOURCE_DIR}"
        OUTPUT_VARIABLE ernie_benchmark_commit
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()
target_compile_definitions(
    ernie_benchmarks
    PRIVATE ERNIE_BENCH_COMMIT="${ernie_benchmark_commit}"
            ERNIE_BENCH_BUILD_TYPE="$<CONFIG>")
