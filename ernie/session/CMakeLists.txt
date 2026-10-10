# SPDX-License-Identifier: BSD-3-Clause
ernie_add_library(
    ernie_session
    SOURCES
    delegate.cpp
    frame_params.cpp
    outputs.cpp
    resolve.cpp
    PUBLIC
    ernie_device
    ernie_geometry
    ernie_scene)
