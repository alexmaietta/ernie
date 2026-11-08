# SPDX-License-Identifier: BSD-3-Clause
ernie_object_library(
    ernie_graph
    graph.cpp
    shape.cpp
)
target_link_libraries(ernie_graph PUBLIC ernie_util)

ernie_static_library(
    ernie_graph_internal
    ernie_graph
    ernie_util_internal
)
