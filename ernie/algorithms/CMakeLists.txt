# SPDX-License-Identifier: BSD-3-Clause
ernie_object_library(
    ernie_algorithms
    radix_sort.cpp
    scan.cpp
)
target_link_libraries(ernie_algorithms PUBLIC ernie_device)
ernie_embed(
    ernie_algorithms
    LIBRARY kAlgorithmsKernels
    USES
        kDeviceKernels
    HEADER ernie/algorithms/kernels.h
    ROOTS
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels|ernie/algorithms/kernels/|*.slang"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels|ernie/algorithms/kernels/|*.h"
)

ernie_static_library(
    ernie_algorithms_internal
    ernie_algorithms
    ernie_device_internal
)
