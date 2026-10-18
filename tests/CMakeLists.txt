# SPDX-License-Identifier: BSD-3-Clause
include("${PROJECT_SOURCE_DIR}/cmake/ErnieTesting.cmake")

ernie_add_test_suite(
    util
    SOURCES
        util/algorithm_test.cpp
        util/bytes_test.cpp
        util/disk_cache_test.cpp
        util/file_test.cpp
        util/hash_index_test.cpp
        util/linear_algebra_test.cpp
        util/log_test.cpp
        util/math_test.cpp
        util/noise_test.cpp
        util/parallel_test.cpp
        util/profiler_test.cpp
        util/random_test.cpp
        util/tensor_test.cpp
        util/text_test.cpp
        util/version_test.cpp
        util/worker_pool_test.cpp
    LIBRARIES ernie_static ernie_json
)
ernie_add_test_suite(geometry SOURCES geometry/bounds_test.cpp LIBRARIES ernie_static)
ernie_add_test_suite(
    physics
    SOURCES
        physics/astronomy_test.cpp
        physics/atmosphere_test.cpp
        physics/reflectance_library_test.cpp
        physics/regolith_test.cpp
        physics/scattering_test.cpp
        physics/solar_test.cpp
        physics/spectrum_test.cpp
    LIBRARIES ernie_static
)
# Monte Carlo and Mie checks against analytic results, slower than unit tests.
ernie_add_test_suite(
    physics_accuracy
    LABEL accuracy
    SOURCES physics/cloud_test.cpp
    LIBRARIES ernie_static
)
ernie_add_test_suite(
    io
    SOURCES
        io/ies_test.cpp
        io/image_test.cpp
        io/nanovdb_test.cpp
        io/raster_test.cpp
        io/texture_test.cpp
    LIBRARIES ernie_static ernie_stb ernie_openexr ernie_zstd
)
ernie_add_test_suite(
    scene
    SOURCES
        scene/api_test.cpp
        scene/extent_cache_test.cpp
        scene/holder_index_test.cpp
        scene/inherit_test.cpp
        scene/instancer_test.cpp
        scene/lux_test.cpp
        scene/mesh_test.cpp
        scene/path_test.cpp
        scene/planet_test.cpp
        scene/render_test.cpp
        scene/scene_test.cpp
        scene/schema_test.cpp
        scene/shade_test.cpp
        scene/storage_test.cpp
        scene/vol_test.cpp
        scene/xform_test.cpp
    LIBRARIES ernie_static
)
ernie_add_test_suite(device LABEL gpu SOURCES device/vulkan_devices_test.cpp LIBRARIES ernie_static)

if(TARGET _ernie)
    # The interpreter tests run with; CI passes the one on its PATH.
    set(ERNIE_TEST_PYTHON
        "${Python_EXECUTABLE}"
        CACHE STRING
        "Python interpreter for the Python tests"
    )
    file(GLOB python_tests CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/python/test_*.py")
    foreach(file ${python_tests})
        get_filename_component(name "${file}" NAME_WE)
        add_test(
            NAME ernie.python.${name}
            COMMAND "${ERNIE_TEST_PYTHON}" -m pytest -q "${file}"
        )
        set_tests_properties(
            ernie.python.${name}
            PROPERTIES
                ENVIRONMENT "PYTHONPATH=${PROJECT_BINARY_DIR}/python"
                LABELS python
                TIMEOUT ${ERNIE_TEST_TIMEOUT_python}
        )
    endforeach()
endif()
