# SPDX-License-Identifier: BSD-3-Clause
set(ERNIE_DEP_VERSION_dear_imgui
    v1.92.9b-docking
    CACHE STRING
    "Tag of Dear ImGui to fetch: a docking-branch tag, for dockable panels"
)
mark_as_advanced(ERNIE_DEP_VERSION_dear_imgui)

# Dear ImGui (MIT) has no CMake project; its sources are compiled here.
ernie_fetch(
    dear_imgui
    URL https://github.com/ocornut/imgui/archive/refs/tags/${ERNIE_DEP_VERSION_dear_imgui}.tar.gz
)
add_library(
    ernie_imgui
    STATIC
    "${dear_imgui_SOURCE_DIR}/imgui.cpp"
    "${dear_imgui_SOURCE_DIR}/imgui_demo.cpp"
    "${dear_imgui_SOURCE_DIR}/imgui_draw.cpp"
    "${dear_imgui_SOURCE_DIR}/imgui_tables.cpp"
    "${dear_imgui_SOURCE_DIR}/imgui_widgets.cpp"
    # Vulkan functions come from volk, as in the device layer.
    "${dear_imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp"
)
target_include_directories(
    ernie_imgui
    SYSTEM
    PUBLIC "${dear_imgui_SOURCE_DIR}" "${dear_imgui_SOURCE_DIR}/backends"
)
target_compile_definitions(ernie_imgui PUBLIC IMGUI_IMPL_VULKAN_USE_VOLK)
target_link_libraries(ernie_imgui PUBLIC ernie_volk)
ernie_quiet_third_party(ernie_imgui)

if(TARGET ernie_glfw)
    add_library(ernie_imgui_glfw STATIC "${dear_imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp")
    # No OpenGL header from glfw3.h: the viewer draws with Vulkan.
    target_compile_definitions(ernie_imgui_glfw PRIVATE GLFW_INCLUDE_NONE)
    target_link_libraries(ernie_imgui_glfw PUBLIC ernie_imgui ernie_glfw)
    ernie_quiet_third_party(ernie_imgui_glfw)
endif()
