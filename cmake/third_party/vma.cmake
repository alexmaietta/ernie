# SPDX-License-Identifier: BSD-3-Clause
set(ERNIE_DEP_VERSION_vma v3.4.0 CACHE STRING "Tag of Vulkan Memory Allocator to fetch")
mark_as_advanced(ERNIE_DEP_VERSION_vma)

ernie_fetch(
    vma
    URL
        https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/archive/refs/tags/${ERNIE_DEP_VERSION_vma}.tar.gz
    SOURCE_SUBDIR no-build
)

# The allocator is header-only; vma/vma.cpp compiles its implementation once, fetching Vulkan
# functions through volk.
add_library(ernie_vma STATIC "${CMAKE_CURRENT_LIST_DIR}/vma/vma.cpp")
target_include_directories(ernie_vma SYSTEM PUBLIC "${vma_SOURCE_DIR}/include")
target_compile_definitions(
    ernie_vma
    PUBLIC VMA_STATIC_VULKAN_FUNCTIONS=0 VMA_DYNAMIC_VULKAN_FUNCTIONS=1
)
target_link_libraries(ernie_vma PUBLIC ernie_volk)
set_target_properties(ernie_vma PROPERTIES POSITION_INDEPENDENT_CODE ON)
ernie_quiet_third_party(ernie_vma)
