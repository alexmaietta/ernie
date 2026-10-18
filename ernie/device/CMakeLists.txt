# SPDX-License-Identifier: BSD-3-Clause
add_library(ernie_device OBJECT vulkan_devices.cpp)
target_link_libraries(ernie_device PUBLIC ernie_util PRIVATE ernie_volk ernie_build_options)
