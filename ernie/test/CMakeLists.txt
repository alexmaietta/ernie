# SPDX-License-Identifier: BSD-3-Clause
include("${ERNIE_SOURCE_DIR}/cmake/ErnieTesting.cmake")

ernie_add_test_suite(
    unit
    SOURCES
    device_shader_compiler_test.cpp
    geometry_cube_sphere_lod_test.cpp
    geometry_ellipsoid_test.cpp
    geometry_frustum_test.cpp
    geometry_lod_quadtree_test.cpp
    geometry_mesh_data_test.cpp
    geometry_primitives_test.cpp
    geometry_sphere_tessellation_test.cpp
    raster_delegate_test.cpp
    raster_poisson_disk_test.cpp
    raster_sensor_plan_test.cpp
    raster_shadow_cascades_test.cpp
    raster_virtual_shadow_test.cpp
    scene_brdf_test.cpp
    scene_cahvore_test.cpp
    scene_dem_test.cpp
    scene_camera_schemas_test.cpp
    scene_camera_test.cpp
    scene_edit_history_test.cpp
    scene_geometry_schemas_test.cpp
    scene_isd_test.cpp
    scene_light_pattern_test.cpp
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
    scene_stage_test.cpp
    scene_terrain_schemas_test.cpp
    scene_volume_schemas_test.cpp
    scene_wide_view_test.cpp
    scene_xform_test.cpp
    session_delegate_test.cpp
    session_frame_params_test.cpp
    session_gpu_xform_test.cpp
    session_light_links_test.cpp
    session_lights_test.cpp
    session_resolve_test.cpp
    session_sampling_test.cpp
    util_image_test.cpp
    util_texture_file_test.cpp
    util_reflectance_library_test.cpp
    util_ies_test.cpp
    util_nanovdb_test.cpp
    util_noise_test.cpp
    util_profile_report_test.cpp
    util_astronomy_test.cpp
    util_atmosphere_test.cpp
    util_scattering_test.cpp
    util_solar_test.cpp
    util_spectrum_test.cpp
    util_tensor_test.cpp
    LIBRARIES
    ernie
    ernie_json
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
    raster_atmosphere_gpu_test.cpp
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
    raster_virtual_shadow_gpu_test.cpp
    raster_volumes_gpu_test.cpp
    raster_zero_copy_gpu_test.cpp
    scene_brdf_vulkan_test.cpp
    support/delegate_scenes.cpp
    LIBRARIES
    ernie
    ernie_stb
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
        message(
            WARNING
                "pytest or numpy not found for ${Python_EXECUTABLE}; skipping Ernie's "
                "Python tests")
    else()
        # A ctest test per test file, and per demo for the demos', so
        # ctest runs them side by side and CI's parts share them out.
        # Each has its own temporary directory, as pytest's shared one
        # is pruned by whichever run starts next; --durations reports
        # the slowest tests.
        function(ernie_add_pytest name)
            cmake_parse_arguments(ARG "" "" "ARGS;ENVIRONMENT"
                                  ${ARGN})
            add_test(
                NAME ernie.python.${name}
                COMMAND
                    "${Python_EXECUTABLE}" -m pytest -q -p
                    no:cacheprovider --durations=10
                    "--basetemp=${CMAKE_CURRENT_BINARY_DIR}/pytest/${name}"
                    ${ARG_ARGS})
            set(environment "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
                            ${ARG_ENVIRONMENT})
            set_tests_properties(
                ernie.python.${name}
                PROPERTIES ENVIRONMENT "${environment}" LABELS
                           "gpu;python")
        endfunction()
        # pytest makes each --basetemp, but not its parent.
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/pytest")
        set(python_tests "${CMAKE_CURRENT_SOURCE_DIR}/python")
        file(GLOB ernie_python_test_files CONFIGURE_DEPENDS
             "${python_tests}/test_*.py")
        foreach(file ${ernie_python_test_files})
            get_filename_component(stem "${file}" NAME_WE)
            if(NOT stem STREQUAL "test_demos")
                ernie_add_pytest(${stem} ARGS "${file}")
            endif()
        endforeach()
        # The demos (demos/<category>/demo_<category>_<name>.{cpp,py},
        # whose id is the file's stem), each in both languages.
        file(GLOB ernie_demo_files CONFIGURE_DEPENDS
             "${PROJECT_SOURCE_DIR}/demos/*/demo_*.cpp"
             "${PROJECT_SOURCE_DIR}/demos/*/demo_*.py")
        set(ernie_demo_ids)
        foreach(file ${ernie_demo_files})
            get_filename_component(id "${file}" NAME_WE)
            list(APPEND ernie_demo_ids ${id})
        endforeach()
        list(REMOVE_DUPLICATES ernie_demo_ids)
        set(demo_test
            test_demo_passes_its_checks_and_twins_render_the_same)
        foreach(id ${ernie_demo_ids})
            ernie_add_pytest(
                ${id}
                ARGS
                "${python_tests}/test_demos.py"
                -k
                "${demo_test}"
                ENVIRONMENT
                "ERNIE_TEST_DEMO=${id}")
        endforeach()
        ernie_add_pytest(
            test_demos ARGS "${python_tests}/test_demos.py" -k
            "not ${demo_test}")
    endif()
endif()

# An application built against the installed Ernie (find_package, Ernie::ernie).
if(ERNIE_INSTALL)
    add_test(
        NAME ernie.install.consumer
        COMMAND
            "${CMAKE_COMMAND}" "-DBUILD_DIR=${CMAKE_BINARY_DIR}"
            "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/install_test"
            "-DGENERATOR=${CMAKE_GENERATOR}" "-DCONFIG=$<CONFIG>" -P
            "${CMAKE_CURRENT_SOURCE_DIR}/install/run_install_test.cmake"
    )
    set_tests_properties(ernie.install.consumer PROPERTIES LABELS
                                                           install)
endif()

if(TARGET ernie_app)
    ernie_add_test_suite(
        app
        SOURCES
        app_demo_test.cpp
        app_file_dialog_test.cpp
        app_panel_test.cpp
        app_prim_schemas_test.cpp
        app_session_test.cpp
        app_theme_test.cpp
        app_timeline_test.cpp
        LIBRARIES
        ernie_app
        ernie_stb)
    ernie_add_test_suite(
        app_gpu
        SOURCES
        app_demo_gpu_test.cpp
        app_viewer_gpu_test.cpp
        LIBRARIES
        ernie_app
        ernie_stb
        LABELS
        gpu)
endif()
