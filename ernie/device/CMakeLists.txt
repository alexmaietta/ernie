# SPDX-License-Identifier: BSD-3-Clause
add_library(
    ernie_device
    OBJECT
    arena.cpp
    bindless.cpp
    buffer.cpp
    column.cpp
    context.cpp
    external.cpp
    image.cpp
    instance.cpp
    kernel.cpp
    queue.cpp
    shader_compiler.cpp
    staging.cpp
    vulkan.cpp
)
target_link_libraries(
    ernie_device
    PUBLIC ernie_util
    PRIVATE ernie_volk ernie_vma ernie_glm slang::slang ernie_build_options
)
