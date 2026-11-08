# SPDX-License-Identifier: BSD-3-Clause
ernie_object_library(ernie_ray_cast ray_cast.cpp)
target_link_libraries(ernie_ray_cast PUBLIC ernie_render)
ernie_embed(
    ernie_ray_cast
    LIBRARY kRayCastKernels
    USES
        kRenderKernels
    HEADER ernie/ray_cast/kernels.h
    ROOTS
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels|ernie/ray_cast/kernels/|*.slang"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels|ernie/ray_cast/kernels/|*.h"
)

ernie_public_files(ray_cast.h)
