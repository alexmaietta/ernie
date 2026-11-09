# SPDX-License-Identifier: BSD-3-Clause
ernie_object_library(
    ernie_material
    codegen.cpp
    definition.cpp
    node.cpp
    uplift.cpp
)
target_link_libraries(ernie_material PUBLIC ernie_graph)

ernie_static_library(
    ernie_material_internal
    ernie_material
    ernie_graph_internal
)
