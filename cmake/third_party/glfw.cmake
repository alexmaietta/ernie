# SPDX-License-Identifier: BSD-3-Clause
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    glfw
    URL
        https://github.com/glfw/glfw/archive/refs/tags/3.4.tar.gz
    URL_HASH SHA256=c038d34200234d071fae9345bc455e4a8f2f544ab60150765d7704e08f3dac01
)
FetchContent_MakeAvailable(glfw)
target_compile_options(glfw PRIVATE "$<IF:$<CXX_COMPILER_ID:MSVC>,/w,-w>")
