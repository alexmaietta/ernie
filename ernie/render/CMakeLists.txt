# SPDX-License-Identifier: BSD-3-Clause
add_library(
    ernie_render
    OBJECT
    bottom_levels.cpp
    device_scene.cpp
    geometry.cpp
    instances.cpp
    prim_rows.cpp
    ray_tracer.cpp
    scene_columns.cpp
    top_levels.cpp
    transform_graph.cpp
)
ernie_add_shaders(
    ernie_render
    shaders/affine.slang
    shaders/compose.slang
    shaders/double_float.slang
    shaders/local.slang
    shaders/ray_tracer.slang
    shaders/shapes.slang
    shaders/world.slang
    shaders/write_instances.slang
)
target_link_libraries(
    ernie_render
    PUBLIC ernie_geometry ernie_scene ernie_graph
    PRIVATE ernie_volk ernie_vma ernie_build_options
)
