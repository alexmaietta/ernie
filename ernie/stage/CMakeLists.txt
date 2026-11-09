# SPDX-License-Identifier: BSD-3-Clause
ernie_object_library(
    ernie_stage
    columns/array_pool.cpp
    columns/column_store.cpp
    columns/column_table.cpp
    columns/device_heap.cpp
    publish/change_set.cpp
    publish/publisher.cpp
    schema/runtime.cpp
    structure/packing.cpp
    structure/paths.cpp
    structure/prims.cpp
    structure/structure.cpp
    structure/tokens.cpp
    views/attribute.cpp
    views/layout.cpp
    views/mapping.cpp
    views/poses.cpp
    views/view.cpp
)
target_link_libraries(ernie_stage PUBLIC ernie_device)
ernie_embed(
    ernie_stage
    LIBRARY kStageKernels
    USES
        kDeviceKernels
    HEADER ernie/stage/kernels.h
    ROOTS
        "${CMAKE_CURRENT_SOURCE_DIR}/columns/kernels|ernie/stage/columns/kernels/|*.slang"
        "${CMAKE_CURRENT_SOURCE_DIR}/columns/kernels|ernie/stage/columns/kernels/|*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/schema|ernie/stage/schema/|*.h"
        "${ERNIE_GENERATED_INCLUDE_DIR}/ernie/scene/schema|ernie/scene/schema/|*.h"
)

file(GLOB_RECURSE ernie_stage_files CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*")
list(FILTER ernie_stage_files EXCLUDE REGEX "/CMakeLists\\.txt$")
ernie_check_layer(
    ernie_stage
    ALLOW
        "^ernie/device/"
        "^ernie/util/"
        "^ernie/stage/"
        "^ernie/export\\.h$"
        "^ernie/scene/schema/(columns|registry)\\.h$"
    FILES
        ${ernie_stage_files}
)

ernie_static_library(
    ernie_stage_internal
    ernie_stage
    ernie_device_internal
)

ernie_public_files(
    columns/kernels/paged.h
    primvar.h
    schema/ids.h
    schema/primvars.h
    schema/runtime.h
    schema/schema.h
    schema/types.h
    structure/tokens.h
    types.h
    views/attribute.h
    views/layout.h
    views/mapping.h
    views/poses.h
    views/view.h
)
