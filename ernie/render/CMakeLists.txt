# SPDX-License-Identifier: BSD-3-Clause
ernie_object_library(
    ernie_render
    overlay/font.cpp
    overlay/overlay.cpp
    ray_cast/ray_cast.cpp
    render_output.cpp
    renderer.cpp
)
target_link_libraries(
    ernie_render
    PUBLIC
        ernie_scene
    PRIVATE
        ernie_fonts
        ernie_stb
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
    ernie_scene_internal
    ernie_fonts
    ernie_stb
)

ernie_public_files(
    kernels/overlay/style.h
    overlay/overlay.h
    ray_cast/ray_cast.h
    render_output.h
    renderer.h
)
