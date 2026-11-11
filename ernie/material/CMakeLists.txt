# SPDX-License-Identifier: BSD-3-Clause
ernie_object_library(
    ernie_material
    codegen.cpp
    node.cpp
    uplift.cpp
)
target_link_libraries(ernie_material PUBLIC ernie_graph)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/uplift.cpp" ernie_uplift_source)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/uplift.h" ernie_uplift_header)
string(SHA256 ernie_uplift_digest "${ernie_uplift_source}${ernie_uplift_header}")
string(SUBSTRING "${ernie_uplift_digest}" 0 16 ernie_uplift_digest)
set_property(
    DIRECTORY
    APPEND
    PROPERTY
        CMAKE_CONFIGURE_DEPENDS
            uplift.cpp
            uplift.h
)
set_property(
    SOURCE
        uplift.cpp
    APPEND
    PROPERTY
        COMPILE_DEFINITIONS
            ERNIE_UPLIFT_DIGEST="${ernie_uplift_digest}"
)
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
