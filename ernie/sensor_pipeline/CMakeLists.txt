# SPDX-License-Identifier: BSD-3-Clause
ernie_object_library(
    ernie_sensor_pipeline
    codegen.cpp
    definition.cpp
    node.cpp
    resolve.cpp
)
target_link_libraries(ernie_sensor_pipeline PUBLIC ernie_graph)

ernie_static_library(
    ernie_sensor_pipeline_internal
    ernie_sensor_pipeline
    ernie_graph_internal
)
