# SPDX-License-Identifier: BSD-3-Clause
include(GenerateExportHeader)

# The installed library: every layer's objects in one shared library, exporting only the public
# API (ERNIE_API) and hiding its dependencies. Defined first, as `ernie/export.h` is generated from
# it.
add_library(ernie SHARED)
add_library(Ernie::ernie ALIAS ernie)
# MSVC warns that exported classes' standard library members need "dll-interface" (C4251, C4275);
# Ernie's C++ ABI already requires consumers to use the same compiler and runtime, so they match.
set(ernie_export_msvc "#ifdef _MSC_VER\n#pragma warning(disable : 4251 4275)\n#endif")
generate_export_header(
    ernie
    EXPORT_MACRO_NAME ERNIE_API
    EXPORT_FILE_NAME "${PROJECT_BINARY_DIR}/include/ernie/export.h"
    CUSTOM_CONTENT_FROM_VARIABLE ernie_export_msvc
)
set_target_properties(
    ernie
    PROPERTIES VERSION "${PROJECT_VERSION}" SOVERSION "${PROJECT_VERSION_MAJOR}"
)

# One object library per layer; a layer links only the layers below it.
add_subdirectory(util)
add_subdirectory(geometry)
add_subdirectory(physics)
add_subdirectory(io)
add_subdirectory(scene)
add_subdirectory(device)
add_subdirectory(graph)
add_subdirectory(render)
add_subdirectory(frame)

set(ernie_layers
    ernie_util
    ernie_geometry
    ernie_physics
    ernie_io
    ernie_scene
    ernie_device
    ernie_graph
    ernie_render
    ernie_frame
)
target_link_libraries(ernie PRIVATE ${ernie_layers} ernie_build_options)
target_include_directories(
    ernie
    INTERFACE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
# Public headers use glm and DLPack; installed, ErnieConfig.cmake supplies them instead.
target_link_libraries(ernie INTERFACE "$<BUILD_INTERFACE:ernie_glm;ernie_dlpack>")
target_compile_features(ernie INTERFACE cxx_std_20)
# Archives linked in keep their symbols to themselves.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(ernie PRIVATE "LINKER:--exclude-libs,ALL")
endif()

add_library(ernie_static STATIC)
target_link_libraries(ernie_static PUBLIC ${ernie_layers})
target_link_options(ernie_static INTERFACE "$<$<CXX_COMPILER_ID:MSVC>:/NOIMPLIB;/NOEXP>")

if(ERNIE_BUILD_APP)
    add_subdirectory(app)
endif()
if(ERNIE_BUILD_PYTHON)
    add_subdirectory(python)
endif()
