# SPDX-License-Identifier: BSD-3-Clause
add_library(
    ernie_render
    OBJECT
    camera_pass.cpp
    device_scene.cpp
    geometry.cpp
    instances.cpp
    prim_rows.cpp
    renderer.cpp
    rt/bottom_levels.cpp
    rt/ray_tracer.cpp
    rt/top_levels.cpp
    scene_columns.cpp
    transform_graph.cpp
)
add_dependencies(ernie_render ernie_shaders)
target_link_libraries(
    ernie_render
    PUBLIC ernie_geometry ernie_scene ernie_graph
    PRIVATE ernie_volk ernie_vma ernie_build_options
)
