# SPDX-License-Identifier: BSD-3-Clause
include_guard(GLOBAL)

include("${CMAKE_CURRENT_LIST_DIR}/third_party/googletest.cmake")
include(GoogleTest)

# Hard timeouts per label, to catch hangs; timings are tracked by benchmarks/.
set(ERNIE_TEST_TIMEOUT_unit 10)
set(ERNIE_TEST_TIMEOUT_gpu 60)
set(ERNIE_TEST_TIMEOUT_accuracy 120)
set(ERNIE_TEST_TIMEOUT_python 120)
# Unoptimised, instrumented coverage builds run several times slower.
if(ERNIE_COVERAGE)
    foreach(label unit gpu accuracy python)
        math(EXPR ERNIE_TEST_TIMEOUT_${label} "${ERNIE_TEST_TIMEOUT_${label}} * 10")
    endforeach()
endif()

# Adds the gtest executable ernie_tests_<name> and registers its tests as ernie.<name>.<test>.
function(ernie_add_test_suite name)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "LABEL" "SOURCES;LIBRARIES")
    if(NOT ARG_LABEL)
        set(ARG_LABEL unit)
    endif()
    set(target ernie_tests_${name})
    add_executable(${target} ${ARG_SOURCES})
    target_link_libraries(${target} PRIVATE ${ARG_LIBRARIES} GTest::gtest_main ernie_build_options)
    target_compile_definitions(
        ${target}
        PRIVATE ERNIE_TEST_FIXTURES="${PROJECT_SOURCE_DIR}/tests/fixtures"
    )
    gtest_discover_tests(
        ${target}
        TEST_PREFIX "ernie.${name}."
        DISCOVERY_MODE PRE_TEST
        PROPERTIES LABELS "${ARG_LABEL}" TIMEOUT "${ERNIE_TEST_TIMEOUT_${ARG_LABEL}}"
    )
endfunction()
