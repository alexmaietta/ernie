# SPDX-License-Identifier: BSD-3-Clause
add_library(
    ernie_device
    OBJECT
    arena.cpp
    buffer.cpp
    context.cpp
    instance.cpp
    queue.cpp
    staging.cpp
    vulkan.cpp
)
target_link_libraries(
    ernie_device
    PUBLIC ernie_util
    PRIVATE ernie_volk ernie_vma ernie_build_options
)
