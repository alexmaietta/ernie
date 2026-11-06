# SPDX-License-Identifier: BSD-3-Clause
ernie_object_library(ernie_texture cache.cpp)
target_link_libraries(ernie_texture PUBLIC ernie_io)
ernie_embed(
    ernie_texture
    LIBRARY kTextureKernels
    USES
        kIoKernels
    HEADER ernie/texture/kernels.h
    ROOTS
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels|ernie/texture/kernels/|*.slang"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels|ernie/texture/kernels/|*.h"
)

ernie_static_library(
    ernie_texture_internal
    ernie_texture
    ernie_io_internal
)
