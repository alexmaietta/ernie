ernie_add_library(
    ernie_app
    SOURCES
    panel.cpp
    properties_panel.cpp
    session.cpp
    stage_panel.cpp
    theme.cpp
    viewer.cpp
    PUBLIC
    ernie_backends
    dear_imgui
    ernie_implot
    PRIVATE
    dear_imgui_glfw_vulkan
    ernie_imguizmo
    ernie_gui_fonts
    ernie_stb)

# The `ernie` executable: the viewer on a default scene.
add_executable(ernie_viewer main.cpp)
set_target_properties(ernie_viewer PROPERTIES OUTPUT_NAME ernie)
target_link_libraries(ernie_viewer PRIVATE ernie_app ernie_build_options)
