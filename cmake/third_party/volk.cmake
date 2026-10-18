# SPDX-License-Identifier: BSD-3-Clause
set(ERNIE_DEP_VERSION_volk vulkan-sdk-1.4.363.0 CACHE STRING "Tag of volk to fetch")
mark_as_advanced(ERNIE_DEP_VERSION_volk)

ernie_fetch(
    volk
    URL https://github.com/zeux/volk/archive/refs/tags/${ERNIE_DEP_VERSION_volk}.tar.gz
    SOURCE_SUBDIR no-build
)

# Loads the Vulkan loader at run time, so nothing links against libvulkan.
add_library(ernie_volk STATIC "${volk_SOURCE_DIR}/volk.c")
target_include_directories(ernie_volk SYSTEM PUBLIC "${volk_SOURCE_DIR}")
target_compile_definitions(ernie_volk PUBLIC VK_NO_PROTOTYPES)
target_link_libraries(ernie_volk PUBLIC ernie_vulkan_headers ${CMAKE_DL_LIBS})
set_target_properties(ernie_volk PROPERTIES POSITION_INDEPENDENT_CODE ON)
