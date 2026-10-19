# SPDX-License-Identifier: BSD-3-Clause
add_library(
    ernie_util
    OBJECT
    disk_cache.cpp
    file.cpp
    library.cpp
    log.cpp
    noise.cpp
    process.cpp
    profiler.cpp
    tensor.cpp
    version.cpp
)
# The sources, and the generated `ernie/export.h`.
target_include_directories(
    ernie_util
    PUBLIC "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}/include"
)
target_link_libraries(
    ernie_util
    PUBLIC ernie_glm ernie_dlpack ernie_tracy
    PRIVATE ernie_json ernie_build_options
)
find_package(Threads REQUIRED)
target_link_libraries(ernie_util PUBLIC Threads::Threads PRIVATE ${CMAKE_DL_LIBS})
if(WIN32)
    # The process's memory counters.
    target_link_libraries(ernie_util PRIVATE psapi)
endif()
set_source_files_properties(
    version.cpp
    PROPERTIES
        COMPILE_DEFINITIONS
            "ERNIE_VERSION_STRING=\"${ERNIE_VERSION}\";ERNIE_BUILD_TYPE=\"$<CONFIG>\""
)
