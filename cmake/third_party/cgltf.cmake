# SPDX-License-Identifier: BSD-3-Clause
FetchContent_Declare(
    cgltf
    URL
        https://github.com/jkuhlmann/cgltf/archive/refs/tags/v1.15.tar.gz
    URL_HASH SHA256=84e352092e5cd6aab7f66de62ddb66beb5e6f18d412ca9d12950d7a55bfef25a
    SOURCE_SUBDIR none
)
FetchContent_MakeAvailable(cgltf)

file(
    CONFIGURE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/cgltf.cpp"
    CONTENT "#define CGLTF_IMPLEMENTATION\n#include <cgltf.h>\n"
)
add_library(ernie_cgltf STATIC "${CMAKE_CURRENT_BINARY_DIR}/cgltf.cpp")
target_include_directories(ernie_cgltf SYSTEM PUBLIC "${cgltf_SOURCE_DIR}")
set_target_properties(
    ernie_cgltf
    PROPERTIES
        POSITION_INDEPENDENT_CODE
            ON
)
target_compile_options(ernie_cgltf PRIVATE "$<IF:$<CXX_COMPILER_ID:MSVC>,/w,-w>")
