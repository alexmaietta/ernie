# SPDX-License-Identifier: BSD-3-Clause
set(ERNIE_DEP_VERSION_tracy v0.14.1 CACHE STRING "Tag of Tracy to fetch")
mark_as_advanced(ERNIE_DEP_VERSION_tracy)

# Ernie's profiling zones also go to Tracy when ERNIE_TRACY is on; otherwise this is empty.
add_library(ernie_tracy INTERFACE)
if(NOT ERNIE_TRACY)
    return()
endif()

# Collect only while a profiler is connected, so a profiling build runs at full speed until then.
set(TRACY_ENABLE ON CACHE BOOL "" FORCE)
set(TRACY_ON_DEMAND ON CACHE BOOL "" FORCE)

ernie_fetch(
    tracy
    URL https://github.com/wolfpld/tracy/archive/refs/tags/${ERNIE_DEP_VERSION_tracy}.tar.gz
)
set_target_properties(
    TracyClient
    PROPERTIES POSITION_INDEPENDENT_CODE ON COMPILE_WARNING_AS_ERROR OFF
)

target_link_libraries(ernie_tracy INTERFACE Tracy::TracyClient)
target_compile_definitions(ernie_tracy INTERFACE ERNIE_TRACY=1)
