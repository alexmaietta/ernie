# SPDX-License-Identifier: BSD-3-Clause
add_library(
    ernie_render
    OBJECT
    bottom_levels.cpp
    device_scene.cpp
    geometry.cpp
    instances.cpp
    prim_rows.cpp
    scene_columns.cpp
    top_levels.cpp
    transform_graph.cpp
)
target_link_libraries(
    ernie_render
    PUBLIC ernie_geometry ernie_scene ernie_graph
    PRIVATE ernie_volk ernie_vma ernie_build_options
)
