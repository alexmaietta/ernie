# SPDX-License-Identifier: BSD-3-Clause
ernie_object_library(
    ernie_bvh
    builder.cpp
    common.cpp
    hardware_builder.cpp
    inspect.cpp
    lbvh.cpp
    reference.cpp
    software_builder.cpp
)
target_link_libraries(ernie_bvh PUBLIC ernie_algorithms)
ernie_embed(
    ernie_bvh
    LIBRARY kBvhKernels
    USES
        kAlgorithmsKernels
    HEADER ernie/bvh/kernels.h
    ROOTS
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels|ernie/bvh/kernels/|*.slang"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels|ernie/bvh/kernels/|*.h"
)

ernie_static_library(
    ernie_bvh_internal
    ernie_bvh
    ernie_algorithms_internal
)
