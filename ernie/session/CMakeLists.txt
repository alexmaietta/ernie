# SPDX-License-Identifier: BSD-3-Clause
ernie_add_library(
    ernie_session
    SOURCES
    delegate.cpp
    frame_params.cpp
    light_links.cpp
    materialx.cpp
    outputs.cpp
    resolve.cpp
    skinning.cpp
    volumes.cpp
    PUBLIC
    ernie_device
    ernie_geometry
    ernie_scene
    PRIVATE
    ernie_materialx)
# MaterialX's data libraries in a build (materialx.h, MaterialXLibraries).
set_source_files_properties(
    materialx.cpp
    PROPERTIES
        COMPILE_DEFINITIONS
        "ERNIE_MATERIALX_LIBRARIES_BUILD_PATH=\"${ERNIE_MATERIALX_LIBRARIES}\""
)
