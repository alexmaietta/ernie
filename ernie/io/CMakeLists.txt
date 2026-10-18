# SPDX-License-Identifier: BSD-3-Clause
add_library(
    ernie_io
    OBJECT
    ies.cpp
    image.cpp
    nanovdb.cpp
    raster.cpp
    texture.cpp
)
target_link_libraries(
    ernie_io
    PUBLIC ernie_geometry
    PRIVATE
        ernie_build_options
        ernie_basisu
        ernie_bc7enc
        ernie_nanovdb
        ernie_openexr
        ernie_stb
        ernie_zstd
)
