# SPDX-License-Identifier: BSD-3-Clause
FetchContent_Declare(
    nvtx
    URL
        https://github.com/NVIDIA/NVTX/archive/refs/tags/v3.2.2.tar.gz
    URL_HASH SHA256=76b875507d2501c9431d79e272ae5b6d4131bd0ffa7a7d3b79ac261d35609f7c
    SOURCE_SUBDIR none
)
FetchContent_MakeAvailable(nvtx)

add_library(ernie_nvtx INTERFACE)
target_include_directories(ernie_nvtx SYSTEM INTERFACE "${nvtx_SOURCE_DIR}/c/include")
target_link_libraries(ernie_nvtx INTERFACE ${CMAKE_DL_LIBS})
