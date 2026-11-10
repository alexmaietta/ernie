# SPDX-License-Identifier: BSD-3-Clause
ernie_object_library(
    ernie_import
    gltf.cpp
    importer.cpp
    obj.cpp
    usd.cpp
)
target_link_libraries(
    ernie_import
    PUBLIC
        ernie_scene
    PRIVATE
        ernie_cgltf
        ernie_tinyobjloader
        tinyusd::tinyusd
)

ernie_static_library(
    ernie_import_internal
    ernie_import
    ernie_scene_internal
    ernie_cgltf
    ernie_tinyobjloader
    tinyusd::tinyusd
)

ernie_public_files(importer.h)
