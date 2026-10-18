# SPDX-License-Identifier: BSD-3-Clause
include("${PROJECT_SOURCE_DIR}/cmake/third_party/glfw.cmake")
include("${PROJECT_SOURCE_DIR}/cmake/third_party/dear_imgui.cmake")

# The viewer: its window, panels and frame loop, a library so tests drive it without a window.
add_library(
    ernie_viewer
    STATIC
    device_panel.cpp
    dock_layout.cpp
    help_windows.cpp
    hud.cpp
    imgui_session.cpp
    imgui_renderer.cpp
    log_panel.cpp
    menu.cpp
    menu_bar.cpp
    message_log.cpp
    navigation.cpp
    offscreen_presenter.cpp
    orbit_camera.cpp
    profiler_panel.cpp
    shell.cpp
    stage_panel.cpp
    theme.cpp
    view.cpp
    viewer.cpp
    widgets.cpp
)
target_link_libraries(
    ernie_viewer
    PUBLIC ernie_static ernie_imgui ernie_volk ernie_vma
    PRIVATE ernie_build_options
)
if(TARGET ernie_glfw)
    target_sources(ernie_viewer PRIVATE swapchain.cpp window.cpp window_presenter.cpp)
    target_link_libraries(ernie_viewer PRIVATE ernie_imgui_glfw)
    target_compile_definitions(ernie_viewer PUBLIC ERNIE_VIEWER_WINDOW=1)
endif()

add_executable(ernie_app main.cpp)
set_target_properties(ernie_app PROPERTIES OUTPUT_NAME ernie)
target_link_libraries(ernie_app PRIVATE ernie_viewer ernie_build_options)
install(TARGETS ernie_app RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT app)

# The licences of what the program carries.
set(ernie_app_notices "dear_imgui|${dear_imgui_SOURCE_DIR}/LICENSE.txt")
if(glfw_SOURCE_DIR)
    list(APPEND ernie_app_notices "glfw|${glfw_SOURCE_DIR}/LICENSE.md")
endif()
foreach(notice ${ernie_app_notices})
    string(REPLACE "|" ";" parts "${notice}")
    list(GET parts 0 name)
    list(GET parts 1 file)
    install(FILES "${file}" DESTINATION "${CMAKE_INSTALL_DOCDIR}/third_party/${name}" COMPONENT app)
endforeach()
