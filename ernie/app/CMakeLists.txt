ernie_add_library(
    ernie_app
    SOURCES
    camera_panel.cpp
    create.cpp
    file_dialog.cpp
    log_panel.cpp
    panel.cpp
    prim_icons.cpp
    prim_schemas.cpp
    profiler_panel.cpp
    properties_panel.cpp
    property_fields.cpp
    session.cpp
    settings_panel.cpp
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
target_link_libraries(ernie_viewer PRIVATE ernie_build_options)
target_compile_definitions(ernie_viewer PRIVATE ERNIE_VERSION="${PROJECT_VERSION}")
if(ERNIE_INSTALL)
    # Installed beside libernie, which holds the app: compiled against the app's headers and
    # linked to the shared library rather than a second copy of everything.
    target_link_libraries(ernie_viewer PRIVATE ernie_shared)
    target_include_directories(ernie_viewer SYSTEM
                               PRIVATE $<TARGET_PROPERTY:ernie_app,INTERFACE_INCLUDE_DIRECTORIES>)
    target_compile_definitions(ernie_viewer
                               PRIVATE $<TARGET_PROPERTY:ernie_app,INTERFACE_COMPILE_DEFINITIONS>)
else()
    target_link_libraries(ernie_viewer PRIVATE ernie_app)
endif()
