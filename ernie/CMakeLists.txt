# SPDX-License-Identifier: BSD-3-Clause
include(GenerateExportHeader)
include("${PROJECT_SOURCE_DIR}/cmake/ErnieLibraries.cmake")
include("${PROJECT_SOURCE_DIR}/cmake/ErnieSchema.cmake")

set(ernie_python_schemas "")
if(ERNIE_BUILD_PYTHON)
    set(ernie_python_defines "${PROJECT_BINARY_DIR}/ernie/python/defines.cpp")
    set(ernie_python_schemas
        PYTHON
        "${PROJECT_BINARY_DIR}/python/ernie/schemas.py"
        PYTHON_DEFINES
        "${ernie_python_defines}"
    )
endif()
set(ernie_scene_defines "${PROJECT_BINARY_DIR}/ernie/scene/defines.cpp")
ernie_generate_schemas(
    "${ERNIE_GENERATED_INCLUDE_DIR}/ernie/scene/kernels/schema"
    "${CMAKE_CURRENT_SOURCE_DIR}/schema/geom.schema"
    "${CMAKE_CURRENT_SOURCE_DIR}/schema/lux.schema"
    "${CMAKE_CURRENT_SOURCE_DIR}/schema/sensor.schema"
    "${CMAKE_CURRENT_SOURCE_DIR}/schema/render.schema"
    "${CMAKE_CURRENT_SOURCE_DIR}/schema/shade.schema"
    DEFINE_HEADER "${ERNIE_GENERATED_INCLUDE_DIR}/ernie/scene/defines.h"
    DEFINE_SOURCE "${ernie_scene_defines}"
    TYPED_HEADER "${ERNIE_GENERATED_INCLUDE_DIR}/ernie/scene/typed.h"
    ${ernie_python_schemas}
)

add_subdirectory(util)
add_subdirectory(device)
add_subdirectory(algorithms)
add_subdirectory(bvh)
add_subdirectory(io)
add_subdirectory(texture)
add_subdirectory(scene)
add_subdirectory(render)
add_subdirectory(import)

ernie_public_files(ernie.h)

ernie_shared_library(
    ernie
    ernie_util
    ernie_device
    ernie_algorithms
    ernie_bvh
    ernie_texture
    ernie_scene
    ernie_render
    ernie_io
    ernie_import
    slang::slang
)
add_library(Ernie::ernie ALIAS ernie)
set(ernie_export_msvc "#ifdef _MSC_VER\n#pragma warning(disable : 4251 4275)\n#endif")
generate_export_header(
    ernie
    EXPORT_MACRO_NAME ERNIE_API
    EXPORT_FILE_NAME "${ERNIE_GENERATED_INCLUDE_DIR}/ernie/export.h"
    CUSTOM_CONTENT_FROM_VARIABLE ernie_export_msvc
)

ernie_check_public_files()

if(ERNIE_BUILD_PYTHON)
    add_subdirectory(python)
endif()
