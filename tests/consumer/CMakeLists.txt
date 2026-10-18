# SPDX-License-Identifier: BSD-3-Clause
# A project using an installed Ernie, as end users do: built in CI against `cmake --install` and
# the -dev packages.
cmake_minimum_required(VERSION 3.21)
project(ErnieConsumer LANGUAGES CXX)

find_package(Ernie 0.2 REQUIRED)

# Each installed public header compiles alone (API-008).
file(
    GLOB_RECURSE headers
    RELATIVE "${Ernie_DIR}/../../../include"
    "${Ernie_DIR}/../../../include/ernie/*.h"
)
list(FILTER headers EXCLUDE REGEX "^ernie/third_party/")
foreach(header ${headers})
    string(MAKE_C_IDENTIFIER "${header}" name)
    file(WRITE "${CMAKE_BINARY_DIR}/headers/${name}.cpp" "#include <${header}>\n")
    list(APPEND header_sources "${CMAKE_BINARY_DIR}/headers/${name}.cpp")
endforeach()
add_library(headers OBJECT ${header_sources})
target_link_libraries(headers PRIVATE Ernie::ernie)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE Ernie::ernie)

enable_testing()
add_test(NAME consumer COMMAND consumer)
