# SPDX-License-Identifier: BSD-3-Clause
add_library(ernie_graph OBJECT dependencies.cpp graph.cpp pool.cpp)
target_link_libraries(
    ernie_graph
    PUBLIC ernie_device
    PRIVATE ernie_volk ernie_vma ernie_build_options
)
