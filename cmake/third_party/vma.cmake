# SPDX-License-Identifier: BSD-3-Clause
FetchContent_Declare(
    vma
    URL
        https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/archive/refs/tags/v3.4.0.tar.gz
    URL_HASH SHA256=822aa850c6ce77346ae96a8a1d351d52e77e85929f35363849a0a4e638e0a2a1
    SYSTEM
)
FetchContent_MakeAvailable(vma)

file(
    CONFIGURE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/vma.cpp"
    CONTENT "#define VMA_IMPLEMENTATION\n#include <volk.h>\n#include <vk_mem_alloc.h>\n"
)
add_library(ernie_vma STATIC "${CMAKE_CURRENT_BINARY_DIR}/vma.cpp")
target_link_libraries(
    ernie_vma
    PUBLIC
        volk::volk
        GPUOpen::VulkanMemoryAllocator
)
target_compile_definitions(
    ernie_vma
    PUBLIC
        VMA_STATIC_VULKAN_FUNCTIONS=0
        VMA_DYNAMIC_VULKAN_FUNCTIONS=1
)
target_compile_options(ernie_vma PRIVATE "$<IF:$<CXX_COMPILER_ID:MSVC>,/w,-w>")
