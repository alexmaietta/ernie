# SPDX-License-Identifier: BSD-3-Clause
ernie_add_library(
    ernie_session
    SOURCES
    delegate.cpp
    frame_params.cpp
    light_links.cpp
    outputs.cpp
    resolve.cpp
    skinning.cpp
    volumes.cpp
    PUBLIC
    ernie_device
    ernie_geometry
    ernie_scene)
