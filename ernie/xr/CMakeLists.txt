# SPDX-License-Identifier: BSD-3-Clause
include("${PROJECT_SOURCE_DIR}/cmake/third_party/openxr.cmake")

ernie_object_library(
    ernie_xr_objects
    EXPORTS ernie_xr_EXPORTS
    anchors.cpp
    locomotion.cpp
    openxr.cpp
    pointers.cpp
    profiles.cpp
    rig.cpp
    session.cpp
    simulated.cpp
    stats.cpp
)
target_link_libraries(
    ernie_xr_objects
    PRIVATE
        ernie
        OpenXR::openxr_loader
        OpenXR::headers
        Vulkan::Headers
)

ernie_shared_library(
    ernie_xr
    ernie_xr_objects
    OpenXR::openxr_loader
)
set_target_properties(
    ernie_xr
    PROPERTIES
        DEFINE_SYMBOL
            ernie_xr_EXPORTS
        EXPORT_NAME
            xr
)
target_link_libraries(ernie_xr PUBLIC ernie)
add_library(Ernie::xr ALIAS ernie_xr)
set(ernie_export_msvc "#ifdef _MSC_VER\n#pragma warning(disable : 4251 4275)\n#endif")
generate_export_header(
    ernie_xr
    BASE_NAME ernie_xr
    EXPORT_MACRO_NAME ERNIE_XR_API
    EXPORT_FILE_NAME "${ERNIE_GENERATED_INCLUDE_DIR}/ernie/xr_export.h"
    CUSTOM_CONTENT_FROM_VARIABLE ernie_export_msvc
)

add_library(ernie_xr_internal STATIC)
target_link_libraries(
    ernie_xr_internal
    PUBLIC
        ernie_xr_objects
        ernie
        OpenXR::openxr_loader
)
target_link_options(
    ernie_xr_internal
    INTERFACE
        "$<$<CXX_COMPILER_ID:MSVC>:/NOIMPLIB;/NOEXP>"
        "$<$<BOOL:${ERNIE_COVERAGE}>:--coverage>"
)
target_compile_definitions(ernie_xr_internal INTERFACE ERNIE_XR_STATIC_DEFINE)

file(
    GLOB ernie_xr_files
    RELATIVE "${PROJECT_SOURCE_DIR}"
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)
ernie_check_includes(ernie/xr ${ernie_xr_files})
