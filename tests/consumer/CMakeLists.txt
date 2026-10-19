# SPDX-License-Identifier: BSD-3-Clause
cmake_minimum_required(VERSION 3.21)
project(ErnieConsumer LANGUAGES CXX)

find_package(Ernie 0.2 REQUIRED)

get_target_property(include_dirs Ernie::ernie INTERFACE_INCLUDE_DIRECTORIES)
foreach(directory ${include_dirs})
    if(EXISTS "${directory}/ernie/export.h")
        set(include_dir "${directory}")
    endif()
endforeach()
file(GLOB_RECURSE headers RELATIVE "${include_dir}" "${include_dir}/ernie/*.h")
list(FILTER headers EXCLUDE REGEX "^ernie/third_party/")
if(NOT headers)
    message(FATAL_ERROR "No public headers under ${include_dir}")
endif()
foreach(header ${headers})
    string(MAKE_C_IDENTIFIER "${header}" name)
    file(WRITE "${CMAKE_BINARY_DIR}/headers/${name}.cpp" "#include <${header}>\n")
    list(APPEND header_sources "${CMAKE_BINARY_DIR}/headers/${name}.cpp")
endforeach()
add_library(headers OBJECT ${header_sources})
target_link_libraries(headers PRIVATE Ernie::ernie)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE Ernie::ernie)

# The README's first C++ render as users copy it, its `main` run inside first_render's.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../../README.md" readme)
string(FIND "${readme}" "## Your First Render" section)
string(SUBSTRING "${readme}" ${section} -1 readme)
string(FIND "${readme}" "```cpp\n" begin)
if(section EQUAL -1 OR begin EQUAL -1)
    message(FATAL_ERROR "README.md has no first render in C++")
endif()
math(EXPR begin "${begin} + 7")
string(SUBSTRING "${readme}" ${begin} -1 readme)
string(FIND "${readme}" "```" end)
string(SUBSTRING "${readme}" 0 ${end} example)
string(REPLACE "int main() {" "void ReadmeMain() {" example "${example}")
file(CONFIGURE OUTPUT "${CMAKE_BINARY_DIR}/readme/first_render.inl" CONTENT "${example}" @ONLY)
set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../../README.md"
)

add_executable(first_render first_render.cpp)
target_include_directories(first_render PRIVATE "${CMAKE_BINARY_DIR}/readme")
target_link_libraries(first_render PRIVATE Ernie::ernie)

enable_testing()
add_test(NAME consumer COMMAND consumer)
add_test(NAME first_render COMMAND first_render)
set_tests_properties(first_render PROPERTIES SKIP_RETURN_CODE 77)
