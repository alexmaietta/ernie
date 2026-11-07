# SPDX-License-Identifier: BSD-3-Clause
if(ERNIE_BUILD_BENCHMARKS)
    add_subdirectory(bench)
endif()

if(NOT ERNIE_BUILD_TESTS)
    return()
endif()

function(ernie_test name)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "" "SOURCES;LINK")
    add_executable(${name} ${ARG_SOURCES})
    target_link_libraries(
        ${name}
        PRIVATE
            ${ARG_LINK}
            GTest::gtest_main
    )
    target_include_directories(${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    ernie_warnings(${name})
    ernie_coverage(${name})
    add_test(NAME ${name} COMMAND ${name})
endfunction()

ernie_test(
    ernie_app_tests
    LINK
        ernie_app
    SOURCES
        unit/app/utest_app_cli.cpp
)

ernie_test(
    ernie_algorithms_tests
    LINK
        ernie_algorithms_internal
    SOURCES
        unit/algorithms/utest_algorithms_radix_sort.cpp
        unit/algorithms/utest_algorithms_scan.cpp
)

ernie_test(
    ernie_bvh_tests
    LINK
        ernie_bvh_internal
    SOURCES
        unit/bvh/utest_bvh_builder.cpp
        unit/bvh/utest_bvh_inspect.cpp
        unit/bvh/utest_bvh_lbvh.cpp
        unit/bvh/utest_bvh_reference.cpp
)

ernie_test(
    ernie_device_tests
    LINK
        ernie_device_internal
    SOURCES
        unit/device/utest_device_buffer.cpp
        unit/device/utest_device_commands.cpp
        unit/device/utest_device_device.cpp
        unit/device/utest_device_hip.cpp
        unit/device/utest_device_program.cpp
        unit/device/utest_device_ref.cpp
        unit/device/utest_device_shared.cpp
        unit/device/utest_device_tensor.cpp
)

ernie_test(
    ernie_import_tests
    LINK
        ernie_import_internal
    SOURCES
        unit/import/utest_import_importer.cpp
)

ernie_test(
    ernie_io_tests
    LINK
        ernie_io_internal
    SOURCES
        unit/io/utest_io_encoding.cpp
        unit/io/utest_io_ies.cpp
        unit/io/utest_io_image.cpp
)

ernie_test(
    ernie_render_tests
    LINK
        ernie_render_internal
    SOURCES
        unit/render/utest_render_camera.cpp
        unit/render/utest_render_lidar.cpp
        unit/render/utest_render_origin.cpp
        unit/render/utest_render_overlay.cpp
        unit/render/utest_render_ray_cast.cpp
        unit/render/utest_render_ray_origin.cpp
        unit/render/utest_render_renderer.cpp
)

ernie_test(
    ernie_scene_tests
    LINK
        ernie_scene_internal
    SOURCES
        unit/scene/utest_scene_access.cpp
        unit/scene/utest_scene_attachment.cpp
        unit/scene/utest_scene_bounds.cpp
        unit/scene/utest_scene_catalog.cpp
        unit/scene/utest_scene_geometry.cpp
        unit/scene/utest_scene_hierarchy.cpp
        unit/scene/utest_scene_instancers.cpp
        unit/scene/utest_scene_instances.cpp
        unit/scene/utest_scene_lights.cpp
        unit/scene/utest_scene_move.cpp
        unit/scene/utest_scene_paths.cpp
        unit/scene/utest_scene_rotation.cpp
        unit/scene/utest_scene_scene.cpp
        unit/scene/utest_scene_semantics.cpp
        unit/scene/utest_scene_store.cpp
        unit/scene/utest_scene_tangents.cpp
        unit/scene/utest_scene_textures.cpp
        unit/scene/utest_scene_timeline.cpp
        unit/scene/utest_scene_tokens.cpp
        unit/scene/utest_scene_updates.cpp
)

ernie_test(
    ernie_texture_tests
    LINK
        ernie_texture_internal
    SOURCES
        unit/texture/utest_texture_cache.cpp
)

ernie_test(
    ernie_util_tests
    LINK
        ernie_util_internal
        ernie_nvtx
    SOURCES
        unit/util/utest_util_disk_cache.cpp
        unit/util/utest_util_environment.cpp
        unit/util/utest_util_error.cpp
        unit/util/utest_util_library.cpp
        unit/util/utest_util_log.cpp
        unit/util/utest_util_profiler.cpp
        unit/util/utest_util_span.cpp
        unit/util/utest_util_text.cpp
        unit/util/utest_util_thread_pool.cpp
        unit/util/utest_util_version.cpp
)
target_compile_definitions(
    ernie_util_tests
    PRIVATE
        ERNIE_TEST_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        ERNIE_TEST_VERSION_MINOR=${PROJECT_VERSION_MINOR}
        ERNIE_TEST_VERSION_PATCH=${PROJECT_VERSION_PATCH}
        ERNIE_TEST_VERSION_FULL="${PROJECT_VERSION}${ERNIE_PRERELEASE}"
)

if(UNIX AND NOT APPLE)
    add_library(ernie_fake_beside SHARED support/fake_beside.cpp)
    set_target_properties(
        ernie_fake_beside
        PROPERTIES
            OUTPUT_NAME
                ernie_beside
            DEBUG_POSTFIX
                ""
            LIBRARY_OUTPUT_DIRECTORY
                "${CMAKE_CURRENT_BINARY_DIR}/beside"
    )
    add_dependencies(ernie_util_tests ernie_fake_beside)
    target_compile_definitions(
        ernie_util_tests
        PRIVATE
            ERNIE_TEST_BESIDE_DIR="$<TARGET_FILE_DIR:ernie_fake_beside>"
    )
endif()

ernie_test(
    ernie_runtime_compiler_tests
    LINK
        ernie_device_internal
        ernie_render_internal
        ernie_io_internal
    SOURCES
        unit/device/utest_device_runtime_compilers.cpp
)
set_tests_properties(
    ernie_runtime_compiler_tests
    PROPERTIES
        LABELS
            kernels
)

if(ERNIE_BUILD_VIEWER)
    ernie_test(
        ernie_viewer_tests
        LINK
            ernie_viewer_internal
        SOURCES
            unit/app/viewer/utest_viewer_editor.cpp
            unit/app/viewer/utest_viewer_history.cpp
            unit/app/viewer/utest_viewer_orbit_camera.cpp
    )
endif()

if(ERNIE_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter REQUIRED)
    set(ernie_python_coverage)
    if(ERNIE_COVERAGE)
        set(ernie_python_coverage
            --cov=ernie
            --cov-report=term
            "--cov-report=xml:${PROJECT_BINARY_DIR}/python-coverage.xml"
        )
    endif()
    add_test(
        NAME ernie_python_tests
        COMMAND
            "${Python_EXECUTABLE}" -m pytest "${CMAKE_CURRENT_SOURCE_DIR}/python" -q -rs -p
            no:cacheprovider ${ernie_python_coverage}
    )
    set_tests_properties(
        ernie_python_tests
        PROPERTIES
            ENVIRONMENT
                "PYTHONPATH=${PROJECT_BINARY_DIR}/python"
    )
endif()
