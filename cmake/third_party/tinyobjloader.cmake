# SPDX-License-Identifier: BSD-3-Clause
FetchContent_Declare(
    tinyobjloader
    URL
        https://github.com/tinyobjloader/tinyobjloader/archive/refs/tags/v2.0.0rc13.tar.gz
    URL_HASH SHA256=0feb92b838f8ce4aa6eb0ccc32dff30cb64a891e0ec3bde837fca49c78d44334
    SOURCE_SUBDIR none
)
FetchContent_MakeAvailable(tinyobjloader)
add_library(ernie_tinyobjloader STATIC "${tinyobjloader_SOURCE_DIR}/tiny_obj_loader.cc")
target_include_directories(ernie_tinyobjloader SYSTEM PUBLIC "${tinyobjloader_SOURCE_DIR}")
set_target_properties(
    ernie_tinyobjloader
    PROPERTIES
        POSITION_INDEPENDENT_CODE
            ON
)
target_compile_options(ernie_tinyobjloader PRIVATE "$<IF:$<CXX_COMPILER_ID:MSVC>,/w,-w>")
