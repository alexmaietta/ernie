# SPDX-License-Identifier: BSD-3-Clause
ernie_object_library(
    ernie_import
    importer.cpp
    obj.cpp
)
target_link_libraries(ernie_import PUBLIC ernie_scene PRIVATE ernie_tinyobjloader)

ernie_static_library(
    ernie_import_internal
    ernie_import
    ernie_scene_internal
    ernie_tinyobjloader
)

ernie_public_files(importer.h)
