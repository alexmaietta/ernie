# SPDX-License-Identifier: BSD-3-Clause
include("${ERNIE_SOURCE_DIR}/cmake/ErnieTesting.cmake")

ernie_add_test_suite(
    unit
    SOURCES
    device_shader_compiler_test.cpp
    geometry_cube_sphere_lod_test.cpp
    geometry_frustum_test.cpp
    geometry_lod_quadtree_test.cpp
    geometry_mesh_data_test.cpp
    geometry_primitives_test.cpp
    raster_delegate_test.cpp
    raster_poisson_disk_test.cpp
    raster_sensor_plan_test.cpp
    raster_shadow_cascades_test.cpp
    scene_brdf_test.cpp
    scene_cahvore_test.cpp
    scene_camera_schemas_test.cpp
    scene_camera_test.cpp
    scene_edit_history_test.cpp
    scene_geometry_schemas_test.cpp
    scene_isd_test.cpp
    scene_material_schemas_test.cpp
    scene_orbit_controller_test.cpp
    scene_overlay_test.cpp
    scene_path_test.cpp
    scene_prim_type_test.cpp
    scene_render_schemas_test.cpp
    scene_scene_test.cpp
    scene_json_test.cpp
    scene_schema_test.cpp
    scene_sensor_filters_test.cpp
    scene_terrain_schemas_test.cpp
    scene_wide_view_test.cpp
    scene_xform_test.cpp
    session_delegate_test.cpp
    session_frame_params_test.cpp
    session_gpu_xform_test.cpp
    session_lights_test.cpp
    session_resolve_test.cpp
    session_sampling_test.cpp
    util_image_test.cpp
    util_astronomy_test.cpp
    util_spectrum_test.cpp
    util_tensor_test.cpp
    LIBRARIES
    ernie
    ernie_stb
    ernie_openexr
    DEFINITIONS
    ERNIE_TEST_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
ernie_add_test_suite(
    gpu
    SOURCES
    device_cuda_interop_test.cpp
    device_frame_uploads_test.cpp
    device_pipeline_test.cpp
    device_vulkan_context_test.cpp
    raster_batch_gpu_test.cpp
    raster_culling_gpu_test.cpp
    raster_delegate_gpu_test.cpp
    raster_gpu_meshes_gpu_test.cpp
    raster_gpu_terrain_gpu_test.cpp
    raster_hex_tiling_gpu_test.cpp
    raster_lens_rays_gpu_test.cpp
    raster_lights_gpu_test.cpp
    raster_mesh_drawable_test.cpp
    raster_motion_gpu_test.cpp
    raster_overlays_gpu_test.cpp
    raster_planet_gpu_test.cpp
    raster_points_gpu_test.cpp
    raster_products_gpu_test.cpp
    raster_sensor_filters_gpu_test.cpp
    raster_standard_filters_gpu_test.cpp
    raster_terrain_drawable_test.cpp
    raster_textures_gpu_test.cpp
    raster_zero_copy_gpu_test.cpp
    scene_brdf_vulkan_test.cpp
    support/delegate_scenes.cpp
    LIBRARIES
    ernie
    LABELS
    gpu
    DEFINITIONS
    ERNIE_SHADER_DIR="${ERNIE_SOURCE_DIR}/ernie/kernel"
    ERNIE_TEST_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")

if(TARGET _ernie)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import pytest, numpy"
        RESULT_VARIABLE ernie_python_test_deps_missing
        OUTPUT_QUIET ERROR_QUIET)
    if(ernie_python_test_deps_missing)
        message(WARNING "pytest or numpy not found for ${Python_EXECUTABLE}; skipping Ernie's "
                        "Python tests")
    else()
        add_test(NAME ernie.python.pytest
                 COMMAND "${Python_EXECUTABLE}" -m pytest -q -p no:cacheprovider
                         "${CMAKE_CURRENT_SOURCE_DIR}/python")
        set_tests_properties(
            ernie.python.pytest PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
                                           LABELS "gpu;python")
    endif()
endif()

# An application built against the installed Ernie (find_package, Ernie::ernie).
if(ERNIE_INSTALL)
    add_test(
        NAME ernie.install.consumer
        COMMAND
            "${CMAKE_COMMAND}" "-DBUILD_DIR=${CMAKE_BINARY_DIR}"
            "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/install_test" "-DGENERATOR=${CMAKE_GENERATOR}"
            "-DCONFIG=$<CONFIG>" -P "${CMAKE_CURRENT_SOURCE_DIR}/install/run_install_test.cmake")
    set_tests_properties(ernie.install.consumer PROPERTIES LABELS install)
endif()

if(TARGET ernie_app)
    ernie_add_test_suite(
        app
        SOURCES
        app_file_dialog_test.cpp
        app_panel_test.cpp
        app_prim_schemas_test.cpp
        app_session_test.cpp
        app_theme_test.cpp
        LIBRARIES
        ernie_app
        ernie_stb)
    ernie_add_test_suite(
        app_gpu
        SOURCES
        app_viewer_gpu_test.cpp
        LIBRARIES
        ernie_app
        ernie_stb
        LABELS
        gpu)
endif()
