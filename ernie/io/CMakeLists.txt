# SPDX-License-Identifier: BSD-3-Clause
ernie_object_library(
    ernie_io
    ies.cpp
    image.cpp
)
target_link_libraries(
    ernie_io
    PUBLIC
        ernie_device
    PRIVATE
        ernie_stb
        ernie_openexr
)
ernie_embed(
    ernie_io
    LIBRARY kIoKernels
    USES
        kDeviceKernels
    HEADER ernie/io/kernels.h
    ROOTS
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels|ernie/io/kernels/|*.slang"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels|ernie/io/kernels/|*.h"
)

ernie_static_library(
    ernie_io_internal
    ernie_io
    ernie_device_internal
    ernie_stb
    ernie_openexr
)

ernie_public_files(image.h)
