# SPDX-License-Identifier: BSD-3-Clause
ernie_object_library(
    ernie_pipeline
    builtins.cpp
    codegen.cpp
    definition.cpp
    node.cpp
    resolve.cpp
    segments.cpp
)
target_link_libraries(ernie_pipeline PUBLIC ernie_graph)

ernie_static_library(
    ernie_pipeline_internal
    ernie_pipeline
    ernie_graph_internal
)
