set(ERNIE_DEP_VERSION_meshoptimizer
    v1.3
    CACHE STRING "Tag of meshoptimizer to fetch.")
mark_as_advanced(ERNIE_DEP_VERSION_meshoptimizer)

# meshoptimizer (MIT), for the simplified levels of detail of instanced meshes.
include(FetchContent)

# Linked into Ernie's libraries, not installed beside them.
set(MESHOPT_INSTALL OFF)

# SYSTEM: its header's own warnings stay out of Ernie's builds.
FetchContent_Declare(
    meshoptimizer
    GIT_REPOSITORY https://github.com/zeux/meshoptimizer.git
    GIT_TAG ${ERNIE_DEP_VERSION_meshoptimizer}
    GIT_SHALLOW TRUE
    SYSTEM)
FetchContent_MakeAvailable(meshoptimizer)
set_target_properties(
    meshoptimizer PROPERTIES POSITION_INDEPENDENT_CODE ON
                             COMPILE_WARNING_AS_ERROR OFF)

add_library(ernie_meshoptimizer INTERFACE)
target_link_libraries(ernie_meshoptimizer INTERFACE meshoptimizer)
