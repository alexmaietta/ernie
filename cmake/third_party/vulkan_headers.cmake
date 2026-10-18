# SPDX-License-Identifier: BSD-3-Clause
set(ERNIE_DEP_VERSION_vulkan_headers
    vulkan-sdk-1.4.363.0
    CACHE STRING
    "Tag of Vulkan-Headers to fetch"
)
mark_as_advanced(ERNIE_DEP_VERSION_vulkan_headers)

ernie_fetch(
    vulkan_headers
    URL
        https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/${ERNIE_DEP_VERSION_vulkan_headers}.tar.gz
    SOURCE_SUBDIR no-build
)

add_library(ernie_vulkan_headers INTERFACE)
target_include_directories(
    ernie_vulkan_headers
    SYSTEM
    INTERFACE "${vulkan_headers_SOURCE_DIR}/include"
)
