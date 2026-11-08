# SPDX-License-Identifier: BSD-3-Clause
ernie_object_library(
    ernie_overlay
    font.cpp
    overlay.cpp
)
target_link_libraries(
    ernie_overlay
    PUBLIC
        ernie_scene
    PRIVATE
        ernie_fonts
        ernie_stb
)
ernie_embed(
    ernie_overlay
    LIBRARY kOverlayKernels
    USES
        kRenderKernels
    HEADER ernie/overlay/kernels.h
    ROOTS
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels|ernie/overlay/kernels/|*.slang"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels|ernie/overlay/kernels/|*.h"
)

ernie_public_files(
    kernels/style.h
    overlay.h
)
