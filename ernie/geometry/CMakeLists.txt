# SPDX-License-Identifier: BSD-3-Clause
ernie_add_library(
    ernie_geometry
    SOURCES
    cube_sphere_lod.cpp
    ellipsoid.cpp
    frustum.cpp
    lod_quadtree.cpp
    mesh_data.cpp
    planet_heights.cpp
    primitives.cpp
    skinning.cpp
    sphere_tessellation.cpp
    subdivision.cpp
    PUBLIC
    ernie_util
    PRIVATE
    ernie_opensubdiv)
