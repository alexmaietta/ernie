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
FetchContent_Declare(
    tracy
    GIT_REPOSITORY https://github.com/wolfpld/tracy.git
    GIT_TAG ${ERNIE_DEP_VERSION_tracy}
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(tracy)
set_target_properties(TracyClient PROPERTIES POSITION_INDEPENDENT_CODE ON
                                             COMPILE_WARNING_AS_ERROR OFF)

target_link_libraries(ernie_tracy INTERFACE Tracy::TracyClient)
target_compile_definitions(ernie_tracy INTERFACE ERNIE_TRACY=1)
