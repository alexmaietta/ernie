# SPDX-License-Identifier: BSD-3-Clause
include("${PROJECT_SOURCE_DIR}/cmake/third_party/glfw.cmake")
include("${PROJECT_SOURCE_DIR}/cmake/third_party/dear_imgui.cmake")

# The viewer: its window, panels and frame loop, a library so tests drive it without a window.
add_library(
    ernie_viewer
    STATIC
    create.cpp
    imgui_backend.cpp
    menu.cpp
    shell_menus.cpp
    ui.cpp
    viewer.cpp
)
target_link_libraries(
    ernie_viewer
    PUBLIC ernie_static ernie_imgui ernie_volk ernie_vma
    PRIVATE ernie_build_options
)
if(TARGET ernie_glfw)
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
