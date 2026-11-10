# SPDX-License-Identifier: BSD-3-Clause
ernie_object_library(
    ernie_material
    codegen.cpp
    node.cpp
    uplift.cpp
)
target_link_libraries(ernie_material PUBLIC ernie_graph)
foreach(
    template
    dispatch
    definition
    surface
    standard
    registered
)
    ernie_embed(
        ernie_material
        FILE "${CMAKE_CURRENT_SOURCE_DIR}/templates/${template}.slang.in"
        SYMBOL ernie_material_${template}
    )
endforeach()

ernie_static_library(
    ernie_material_internal
    ernie_material
    ernie_graph_internal
)
