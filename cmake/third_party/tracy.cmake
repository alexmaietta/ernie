set(ERNIE_DEP_VERSION_tracy
    v0.14.1
    CACHE STRING "Tag of Tracy to fetch.")
mark_as_advanced(ERNIE_DEP_VERSION_tracy)

# Tracy (BSD-3-Clause), a frame profiler: Ernie's CPU zones and Vulkan GPU zones
# (ernie/util/profile.h, ernie/device/gpu_profile.h) when ERNIE_TRACY is on; nothing otherwise.
add_library(ernie_tracy INTERFACE)
if(NOT ERNIE_TRACY)
    return()
endif()

include(FetchContent)

# Collect only while a profiler is connected, so a profiling build runs at full speed until then.
set(TRACY_ENABLE
    ON
    CACHE BOOL "" FORCE)
set(TRACY_ON_DEMAND
    ON
    CACHE BOOL "" FORCE)
# Tracy's install rules have no switch; leaving its directory out of `all` (CMake 3.28+) keeps
# them out of Ernie's install.
set(tracy_exclude_from_all)
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.28)
    set(tracy_exclude_from_all EXCLUDE_FROM_ALL)
endif()
FetchContent_Declare(
    tracy
    GIT_REPOSITORY https://github.com/wolfpld/tracy.git
    GIT_TAG ${ERNIE_DEP_VERSION_tracy}
    GIT_SHALLOW TRUE
    ${tracy_exclude_from_all})
FetchContent_MakeAvailable(tracy)
set_target_properties(TracyClient PROPERTIES POSITION_INDEPENDENT_CODE ON
                                             COMPILE_WARNING_AS_ERROR OFF)

target_link_libraries(ernie_tracy INTERFACE Tracy::TracyClient)
target_compile_definitions(ernie_tracy INTERFACE ERNIE_TRACY=1)
