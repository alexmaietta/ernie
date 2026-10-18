# SPDX-License-Identifier: BSD-3-Clause
add_library(ernie_scene OBJECT path.cpp storage.cpp)
target_link_libraries(ernie_scene PUBLIC ernie_physics ernie_io PRIVATE ernie_build_options)
