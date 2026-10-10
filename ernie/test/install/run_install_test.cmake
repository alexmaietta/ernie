# SPDX-License-Identifier: BSD-3-Clause
# cmake -DBUILD_DIR=<Ernie's build> -DWORK_DIR=<scratch> -DGENERATOR=<generator> [-DCONFIG=<config>]
#       -P run_install_test.cmake
#
# Installs Ernie's runtime and dev components from BUILD_DIR into WORK_DIR/prefix, builds the
# consumer project beside this script against it, and runs the consumer.
cmake_minimum_required(VERSION 3.24)

foreach(variable BUILD_DIR WORK_DIR GENERATOR)
    if(NOT DEFINED ${variable})
        message(FATAL_ERROR "run_install_test.cmake needs -D${variable}=...")
    endif()
endforeach()
if(NOT CONFIG)
    set(CONFIG Release)
endif()

set(prefix "${WORK_DIR}/prefix")
set(consumer_build "${WORK_DIR}/consumer")
file(REMOVE_RECURSE "${WORK_DIR}")

function(run)
    execute_process(COMMAND ${ARGN} RESULT_VARIABLE result)
    if(NOT result EQUAL 0)
        list(JOIN ARGN " " command)
        message(FATAL_ERROR "Failed (${result}): ${command}")
    endif()
endfunction()

foreach(component runtime dev)
    run("${CMAKE_COMMAND}" --install "${BUILD_DIR}" --prefix "${prefix}" --config "${CONFIG}"
        --component ${component})
endforeach()

# The README's C++ example: the first ```cpp block.
file(READ "${CMAKE_CURRENT_LIST_DIR}/../../../README.md" readme)
string(FIND "${readme}" "```cpp\n" start)
if(start EQUAL -1)
    message(FATAL_ERROR "README.md has no ```cpp example")
endif()
math(EXPR start "${start} + 7")
string(SUBSTRING "${readme}" ${start} -1 readme)
string(FIND "${readme}" "```" end)
string(SUBSTRING "${readme}" 0 ${end} readme_cpp)
file(WRITE "${WORK_DIR}/readme_example.cpp" "${readme_cpp}")

run("${CMAKE_COMMAND}" -S "${CMAKE_CURRENT_LIST_DIR}" -B "${consumer_build}" -G "${GENERATOR}"
    "-DCMAKE_BUILD_TYPE=${CONFIG}" "-DCMAKE_PREFIX_PATH=${prefix}"
    "-DREADME_CPP=${WORK_DIR}/readme_example.cpp")
run("${CMAKE_COMMAND}" --build "${consumer_build}" --config "${CONFIG}")

# Both render when there is a GPU and only report it otherwise; the README's example throws
# without one, so it runs only when the consumer found a renderer.
foreach(program consumer readme_example)
    find_program(
        ${program}_path ${program}
        PATHS "${consumer_build}" "${consumer_build}/${CONFIG}"
        NO_DEFAULT_PATH NO_CACHE)
endforeach()
execute_process(
    COMMAND "${consumer_path}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output)
message("${output}")
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Failed (${result}): ${consumer_path}")
endif()
if(output MATCHES "renderer: ")
    run("${readme_example_path}")
endif()
