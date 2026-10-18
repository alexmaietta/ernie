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
    schemas/camera.cpp
    schemas/geom.cpp
    schemas/instancer.cpp
    schemas/laser.cpp
    schemas/lux.cpp
    schemas/mesh.cpp
    schemas/planet.cpp
    schemas/registry.cpp
    schemas/render.cpp
    schemas/shade.cpp
    schemas/vol.cpp
)
target_link_libraries(
    ernie_scene
    PUBLIC ernie_physics ernie_io
    PRIVATE ernie_json ernie_build_options
)
