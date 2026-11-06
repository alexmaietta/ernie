# SPDX-License-Identifier: BSD-3-Clause
set(ernie_build_info "${CMAKE_CURRENT_BINARY_DIR}/build_info.h")
add_custom_target(
    ernie_build_info
    ALL
    COMMAND
        "${CMAKE_COMMAND}" "-DSOURCE=${PROJECT_SOURCE_DIR}" "-DOUTPUT=${ernie_build_info}"
        "-DVERSION=${PROJECT_VERSION}" -P "${PROJECT_SOURCE_DIR}/cmake/ErnieBuildInfo.cmake"
    BYPRODUCTS
        "${ernie_build_info}"
    VERBATIM
)

ernie_object_library(
    ernie_util
    disk_cache.cpp
    environment.cpp
    error.cpp
    library.cpp
    log.cpp
    profiler.cpp
    text.cpp
    thread_pool.cpp
    version.cpp
)
add_dependencies(ernie_util ernie_build_info)
target_include_directories(ernie_util PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(
    ernie_util
    PRIVATE
        ernie_nvtx
        ${CMAKE_DL_LIBS}
)
ernie_static_library(
    ernie_util_internal
    ernie_util
    ${CMAKE_DL_LIBS}
)

ernie_public_files(
    error.h
    log.h
    profiler.h
    span.h
    version.h
)
