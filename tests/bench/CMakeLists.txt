# SPDX-License-Identifier: BSD-3-Clause
add_library(
    ernie_bench_main
    STATIC
    main.cpp
    bench.cpp
)
ernie_warnings(ernie_bench_main)
target_link_libraries(
    ernie_bench_main
    PUBLIC
        ernie_device_internal
        benchmark::benchmark
)
target_include_directories(
    ernie_bench_main
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${PROJECT_SOURCE_DIR}/tests"
)
target_compile_definitions(
    ernie_bench_main
    PRIVATE
        ERNIE_BENCH_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
        ERNIE_BENCH_BUILD_TYPE="$<CONFIG>"
)

function(ernie_bench name)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "" "SOURCES;LINK")
    add_executable(${name} ${ARG_SOURCES})
    ernie_warnings(${name})
    target_link_libraries(
        ${name}
        PRIVATE
            ${ARG_LINK}
            ernie_bench_main
    )
    add_test(
        NAME ${name}
        COMMAND
            ${name} --benchmark_dry_run
    )
    set_tests_properties(
        ${name}
        PROPERTIES
            LABELS
                bench
            ENVIRONMENT
                ERNIE_LOG=warning
    )
endfunction()

ernie_bench(
    ernie_algorithms_bench
    LINK ernie_algorithms_internal
    SOURCES
        algorithms/btest_algorithms_radix_sort.cpp
        algorithms/btest_algorithms_scan.cpp
)

ernie_bench(
    ernie_bvh_bench
    LINK ernie_render_internal
    SOURCES
        bvh/btest_bvh_lbvh.cpp
        bvh/btest_bvh_trace.cpp
)

ernie_bench(
    ernie_render_bench
    LINK ernie_render_internal
    SOURCES
        render/btest_render_instancers.cpp
        render/btest_render_overlay.cpp
        render/btest_render_points.cpp
        render/btest_render_renderer.cpp
        render/btest_render_scaling.cpp
        render/btest_render_sensors.cpp
        render/btest_render_throughput.cpp
        render/btest_render_workloads.cpp
)

ernie_bench(
    ernie_scene_bench
    LINK ernie_scene_internal
    SOURCES
        scene/btest_scene_publish.cpp
)
