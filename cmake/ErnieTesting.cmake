# SPDX-License-Identifier: BSD-3-Clause
# ernie_add_test_suite(<name> SOURCES ... [LIBRARIES ...] [LABELS ...] [DEFINITIONS ...])
#
# Inside a TerrainGen build the suite goes through terraingen_add_test_suite, so it shares
# TerrainGen's GoogleTest, warnings, coverage and test labels; standalone Ernie fetches GoogleTest
# itself.
include_guard(GLOBAL)

if(NOT COMMAND terraingen_add_test_suite AND NOT TARGET
                                             GTest::gtest_main)
    include("${CMAKE_CURRENT_LIST_DIR}/third_party/googletest.cmake")
endif()
include(GoogleTest)

function(ernie_add_test_suite name)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" ""
                          "SOURCES;LIBRARIES;LABELS;DEFINITIONS")
    if(NOT ARG_LABELS)
        set(ARG_LABELS unit)
    endif()
    if(COMMAND terraingen_add_test_suite)
        terraingen_add_test_suite(
            ernie_${name}
            SOURCES
            ${ARG_SOURCES}
            LIBRARIES
            ${ARG_LIBRARIES}
            LABELS
            ${ARG_LABELS})
        set(target terraingen_tests_ernie_${name})
    else()
        set(target ernie_tests_${name})
        add_executable(${target} ${ARG_SOURCES})
        target_link_libraries(
            ${target} PRIVATE ${ARG_LIBRARIES} GTest::gtest_main
                              ernie_build_options)
        gtest_discover_tests(
            ${target}
            TEST_PREFIX "ernie_${name}." DISCOVERY_MODE PRE_TEST
            PROPERTIES LABELS "${ARG_LABELS}")
    endif()
    target_include_directories(
        ${target} PRIVATE "${ERNIE_SOURCE_DIR}/ernie/test")
    target_compile_definitions(${target} PRIVATE ${ARG_DEFINITIONS})
    # Test code is setup and assertions, whose optimisation at -O2 dominates its compile time
    # (several times the parse); -O1 keeps it quick to build, while the libraries under test keep
    # their own optimisation.
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(
            ${target} PRIVATE $<$<CONFIG:Release,RelWithDebInfo>:-O1>)
    endif()
endfunction()
