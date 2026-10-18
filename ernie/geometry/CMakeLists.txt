# SPDX-License-Identifier: BSD-3-Clause
add_library(ernie_geometry OBJECT triangle_mesh.cpp)
target_link_libraries(ernie_geometry PUBLIC ernie_util PRIVATE ernie_build_options)
