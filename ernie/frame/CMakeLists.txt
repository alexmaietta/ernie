# SPDX-License-Identifier: BSD-3-Clause
add_library(ernie_frame OBJECT frame.cpp)
target_link_libraries(
    ernie_frame
    PUBLIC ernie_render
    PRIVATE ernie_volk ernie_vma ernie_build_options
)
