# SPDX-License-Identifier: BSD-3-Clause
ernie_object_library(
    ernie_sensor_pipeline
    codegen.cpp
    definition.cpp
    node.cpp
    resolve.cpp
)
target_link_libraries(ernie_sensor_pipeline PUBLIC ernie_graph)
foreach(
    template
    module
    elements
    captures
    reduce_groups
    reduce_partial
)
    ernie_embed(
        ernie_sensor_pipeline
        FILE "${CMAKE_CURRENT_SOURCE_DIR}/templates/${template}.slang.in"
        SYMBOL ernie_sensor_pipeline_${template}
    )
endforeach()

ernie_static_library(
    ernie_sensor_pipeline_internal
    ernie_sensor_pipeline
    ernie_graph_internal
)
