# SPDX-License-Identifier: BSD-3-Clause
add_library(ernie_io STATIC ies.cpp image.cpp raster.cpp)
target_link_libraries(
    ernie_io
    PUBLIC ernie_util
    PRIVATE ernie_build_options ernie_stb ernie_openexr
)
