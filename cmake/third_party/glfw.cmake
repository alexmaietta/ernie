# SPDX-License-Identifier: BSD-3-Clause
set(ERNIE_DEP_VERSION_glfw 3.4 CACHE STRING "Tag of GLFW to fetch")
mark_as_advanced(ERNIE_DEP_VERSION_glfw)
option(
    ERNIE_USE_SYSTEM_GLFW
    "Use an installed GLFW 3.4 instead of building one in, which loads its window systems at run time"
    OFF
)

# The viewer's window. `ernie_glfw` is defined only when a window system can be built for; without
# one the viewer still draws offscreen.
if(ERNIE_USE_SYSTEM_GLFW)
    find_package(glfw3 3.4 REQUIRED)
    add_library(ernie_glfw INTERFACE)
    target_link_libraries(ernie_glfw INTERFACE glfw)
    return()
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # GLFW opens X11 and Wayland at run time, but needs their headers to build.
    find_package(X11 QUIET)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(ernie_wayland QUIET wayland-client wayland-cursor wayland-egl xkbcommon)
    endif()
    find_program(ernie_wayland_scanner wayland-scanner)
    set(ernie_x11 OFF)
    if(
        X11_FOUND
        AND X11_Xrandr_INCLUDE_PATH
        AND X11_Xinerama_INCLUDE_PATH
        AND X11_Xkb_INCLUDE_PATH
        AND X11_Xcursor_INCLUDE_PATH
        AND X11_Xi_INCLUDE_PATH
    )
        set(ernie_x11 ON)
    endif()
    set(ernie_wayland OFF)
    if(ernie_wayland_FOUND AND ernie_wayland_scanner)
        set(ernie_wayland ON)
    endif()
    if(NOT ernie_x11 AND NOT ernie_wayland)
        message(
            STATUS
            "No X11 or Wayland development headers: the viewer is built without a window"
        )
        return()
    endif()
    set(GLFW_BUILD_X11 ${ernie_x11} CACHE BOOL "" FORCE)
    set(GLFW_BUILD_WAYLAND ${ernie_wayland} CACHE BOOL "" FORCE)
endif()

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
set(ernie_saved_shared_libs ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF)
ernie_fetch(
    glfw
    URL https://github.com/glfw/glfw/archive/refs/tags/${ERNIE_DEP_VERSION_glfw}.tar.gz
)
set(BUILD_SHARED_LIBS ${ernie_saved_shared_libs})
ernie_quiet_third_party(glfw)

add_library(ernie_glfw INTERFACE)
target_link_libraries(ernie_glfw INTERFACE glfw)
