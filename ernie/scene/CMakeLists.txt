# SPDX-License-Identifier: BSD-3-Clause
add_library(
    ernie_scene
    OBJECT
    animation.cpp
    edit_history.cpp
    extent_cache.cpp
    holder_index.cpp
    inherit.cpp
    json.cpp
    path.cpp
    reference.cpp
    schema.cpp
    scene.cpp
    storage.cpp
    time_samples.cpp
    usd_plugin.cpp
    validate.cpp
    view.cpp
    xform.cpp
    schema/camera.cpp
    schema/geom.cpp
    schema/instancer.cpp
    schema/lux.cpp
    schema/mesh.cpp
    schema/planet.cpp
    schema/registry.cpp
    schema/render.cpp
    schema/shade.cpp
    schema/vol.cpp
)
target_link_libraries(
    ernie_scene
    PUBLIC ernie_physics ernie_io
    PRIVATE ernie_json ernie_build_options
)
