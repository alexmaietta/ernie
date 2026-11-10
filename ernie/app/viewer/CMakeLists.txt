# SPDX-License-Identifier: BSD-3-Clause
include("${PROJECT_SOURCE_DIR}/cmake/third_party/glfw.cmake")
include("${PROJECT_SOURCE_DIR}/cmake/third_party/dear_imgui.cmake")

set(ernie_viewport_source "${CMAKE_CURRENT_SOURCE_DIR}/viewport/viewport.slang")
set(ernie_viewport_spirv "${CMAKE_CURRENT_BINARY_DIR}/viewport.spv")
add_custom_command(
    OUTPUT
        "${ernie_viewport_spirv}"
    COMMAND
        "${slang_SOURCE_DIR}/bin/slangc" "${ernie_viewport_source}" -target spirv -profile spirv_1_5
        -fvk-use-entrypoint-name -entry VertexMain -stage vertex -entry FragmentMain -stage fragment
        -o "${ernie_viewport_spirv}"
    DEPENDS
        "${ernie_viewport_source}"
    VERBATIM
)

ernie_object_library(
    ernie_viewer_objects
    EXPORTS ernie_viewer_EXPORTS
    overlays/hud.cpp
    overlays/toolbar.cpp
    panels/capture.cpp
    panels/help.cpp
    panels/memory.cpp
    panels/profiler.cpp
    panels/properties.cpp
    panels/stage.cpp
    panels/status.cpp
    presenter/vulkan_presenter.cpp
    shell/capture.cpp
    shell/editor.cpp
    theme/theme.cpp
    viewer.cpp
    viewport/orbit_camera.cpp
    viewport/gizmo.cpp
    viewport/viewport.cpp
    widgets/asset_path.cpp
    widgets/file_dialog.cpp
    widgets/widgets.cpp
)
ernie_embed(ernie_viewer_objects FILE "${ernie_viewport_spirv}" SYMBOL ernie_viewport_spirv)
if(ERNIE_BUILD_XR)
    target_sources(ernie_viewer_objects PRIVATE panels/xr.cpp)
    target_compile_definitions(ernie_viewer_objects PRIVATE ERNIE_VIEWER_XR)
    target_link_libraries(ernie_viewer_objects PRIVATE ernie_xr)
endif()
target_compile_definitions(
    ernie_viewer_objects
    PRIVATE
        VK_NO_PROTOTYPES
        GLFW_INCLUDE_NONE
)
target_link_libraries(
    ernie_viewer_objects
    PUBLIC
        ernie_imgui
    PRIVATE
        ernie
        ernie_imgui_backends
        ernie_fonts
        ernie_json
        Vulkan::Headers
)

ernie_shared_library(
    ernie_viewer
    ernie_viewer_objects
    $<$<BOOL:${ERNIE_BUILD_XR}>:ernie_xr>
    ernie_imgui
    ernie_imgui_backends
    ernie_fonts
)
set_target_properties(
    ernie_viewer
    PROPERTIES
        DEFINE_SYMBOL
            ernie_viewer_EXPORTS
        EXPORT_NAME
            viewer
)
target_link_libraries(ernie_viewer PUBLIC ernie INTERFACE "$<BUILD_INTERFACE:ernie_imgui>")
if(NOT ERNIE_IMGUI_TARGET)
    target_include_directories(
        ernie_viewer
        INTERFACE
            "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ernie/third_party/imgui>"
    )
endif()
add_library(Ernie::viewer ALIAS ernie_viewer)
include(GenerateExportHeader)
set(ernie_export_msvc "#ifdef _MSC_VER\n#pragma warning(disable : 4251 4275)\n#endif")
generate_export_header(
    ernie_viewer
    BASE_NAME ernie_viewer
    EXPORT_MACRO_NAME ERNIE_VIEWER_API
    EXPORT_FILE_NAME "${ERNIE_GENERATED_INCLUDE_DIR}/ernie/viewer_export.h"
    CUSTOM_CONTENT_FROM_VARIABLE ernie_export_msvc
)

add_library(ernie_viewer_internal STATIC)
target_link_libraries(
    ernie_viewer_internal
    PUBLIC
        ernie_viewer_objects
        ernie_imgui
        ernie_imgui_backends
        ernie_fonts
        ernie
        $<$<BOOL:${ERNIE_BUILD_XR}>:ernie_xr>
)
target_link_options(
    ernie_viewer_internal
    INTERFACE
        "$<$<CXX_COMPILER_ID:MSVC>:/NOIMPLIB;/NOEXP>"
        "$<$<BOOL:${ERNIE_COVERAGE}>:--coverage>"
)
target_compile_definitions(
    ernie_viewer_internal
    INTERFACE
        ERNIE_VIEWER_STATIC_DEFINE
        VK_NO_PROTOTYPES
        GLFW_INCLUDE_NONE
)
target_link_libraries(ernie_viewer_internal PUBLIC Vulkan::Headers)

file(
    GLOB_RECURSE ernie_viewer_files
    RELATIVE "${PROJECT_SOURCE_DIR}"
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)
if(NOT ERNIE_BUILD_XR)
    list(REMOVE_ITEM ernie_viewer_files ernie/app/viewer/panels/xr.cpp)
endif()
ernie_check_includes(ernie/app ${ernie_viewer_files})

if(ERNIE_BUILD_PYTHON)
    if(SKBUILD)
        set_property(TARGET ernie_viewer PROPERTY VERSION)
        set_property(TARGET ernie_viewer PROPERTY SOVERSION)
    endif()
    install(
        TARGETS
            ernie_viewer
        RUNTIME
            DESTINATION ernie
            COMPONENT python
        LIBRARY
            DESTINATION ernie
            COMPONENT python
            NAMELINK_SKIP
    )
endif()
