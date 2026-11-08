# SPDX-License-Identifier: BSD-3-Clause
ernie_object_library(
    ernie_render
    render_output.cpp
    renderer.cpp
)
target_link_libraries(
    ernie_render
    PUBLIC
        ernie_scene
        ernie_overlay
)
ernie_embed(
    ernie_render
    LIBRARY kRenderKernels
    USES
        kSceneKernels
    HEADER ernie/render/kernels.h
    ROOTS
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels|ernie/render/kernels/|*.slang"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels|ernie/render/kernels/|*.h"
)

ernie_static_library(
    ernie_render_internal
    ernie_render
    ernie_overlay
    ernie_ray_cast
    ernie_scene_internal
    ernie_fonts
    ernie_stb
)

ernie_public_files(
    render_output.h
    renderer.h
)
