# SPDX-License-Identifier: BSD-3-Clause
set(ERNIE_DEP_VERSION_dlpack v1.3 CACHE STRING "Tag of DLPack to fetch")
mark_as_advanced(ERNIE_DEP_VERSION_dlpack)

ernie_fetch(
    dlpack
    URL https://github.com/dmlc/dlpack/archive/refs/tags/${ERNIE_DEP_VERSION_dlpack}.tar.gz
    SOURCE_SUBDIR no-build
)

# The tensor exchange header, nothing to build.
add_library(ernie_dlpack INTERFACE)
target_include_directories(ernie_dlpack SYSTEM INTERFACE "${dlpack_SOURCE_DIR}/include")
