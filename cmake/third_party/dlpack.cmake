# SPDX-License-Identifier: BSD-3-Clause
set(ERNIE_DEP_VERSION_dlpack v1.3 CACHE STRING "Tag of DLPack to fetch")
mark_as_advanced(ERNIE_DEP_VERSION_dlpack)

include(FetchContent)
FetchContent_Declare(
    dlpack
    GIT_REPOSITORY https://github.com/dmlc/dlpack.git
    GIT_TAG ${ERNIE_DEP_VERSION_dlpack}
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR no-build
)
FetchContent_MakeAvailable(dlpack)

# The tensor exchange header, nothing to build.
add_library(ernie_dlpack INTERFACE)
target_include_directories(ernie_dlpack SYSTEM INTERFACE "${dlpack_SOURCE_DIR}/include")
