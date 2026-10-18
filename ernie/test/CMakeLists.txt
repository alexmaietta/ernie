# SPDX-License-Identifier: BSD-3-Clause
include("${PROJECT_SOURCE_DIR}/cmake/ErnieTesting.cmake")

ernie_add_test_suite(
    core
    SOURCES
        core/disk_cache_test.cpp
        core/linear_algebra_test.cpp
        core/log_test.cpp
        core/mapped_file_test.cpp
        core/math_test.cpp
        core/noise_test.cpp
        core/parallel_test.cpp
        core/profile_report_test.cpp
        core/profiler_test.cpp
        core/random_test.cpp
        core/tensor_test.cpp
        core/version_test.cpp
        core/worker_pool_test.cpp
    LIBRARIES ernie_core ernie_json
)
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
    LIBRARIES ernie_physics
)
# Monte Carlo and Mie checks against analytic results, slower than unit tests.
ernie_add_test_suite(
    physics_accuracy
    LABEL accuracy
    SOURCES physics/cloud_test.cpp
    LIBRARIES ernie_physics
)
ernie_add_test_suite(device LABEL gpu SOURCES device/vulkan_devices_test.cpp LIBRARIES ernie_device)

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
