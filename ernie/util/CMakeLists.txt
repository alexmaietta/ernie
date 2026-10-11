# SPDX-License-Identifier: BSD-3-Clause
ernie_add_library(
    ernie_util
    SOURCES
    astronomy.cpp
    ies.cpp
    image.cpp
    profiler.cpp
    spectrum.cpp
    tensor.cpp
    PUBLIC
    ernie_glm
    ernie_dlpack
    ernie_tracy
    PRIVATE
    ernie_stb
    ernie_openexr)
