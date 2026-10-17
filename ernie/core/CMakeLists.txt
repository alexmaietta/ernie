# SPDX-License-Identifier: BSD-3-Clause
add_library(
    ernie_core
    STATIC
    disk_cache.cpp
    log.cpp
    mapped_file.cpp
    noise.cpp
    profile_report.cpp
    profiler.cpp
    tensor.cpp
    version.cpp
)
target_include_directories(ernie_core PUBLIC "${PROJECT_SOURCE_DIR}")
target_link_libraries(
    ernie_core
    PUBLIC ernie_glm ernie_dlpack ernie_tracy
    PRIVATE ernie_json ernie_build_options
)
find_package(Threads REQUIRED)
target_link_libraries(ernie_core PUBLIC Threads::Threads)
set_source_files_properties(
    version.cpp
    PROPERTIES
        COMPILE_DEFINITIONS
            "ERNIE_VERSION_STRING=\"${ERNIE_VERSION}\";ERNIE_BUILD_TYPE=\"$<CONFIG>\""
)
