# SPDX-License-Identifier: BSD-3-Clause
# Vendored dependencies, kept byte for byte as their upstream releases; each folder's README or
# LICENSE says which.

# Basis Universal's transcoder (Apache-2.0): KTX2 textures in Basis's ETC1S and UASTC formats
# become BC blocks. Its zstd calls link against OpenEXR's copy (ernie_zstd).
add_library(ernie_basisu STATIC basisu/transcoder/basisu_transcoder.cpp)
target_include_directories(
    ernie_basisu
    SYSTEM
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/basisu/transcoder"
)
target_compile_definitions(ernie_basisu PUBLIC BASISD_SUPPORT_KTX2=1 BASISD_SUPPORT_KTX2_ZSTD=1)
target_link_libraries(ernie_basisu PRIVATE ernie_zstd)
set_target_properties(ernie_basisu PROPERTIES POSITION_INDEPENDENT_CODE ON)

# PNanoVDB (Apache-2.0), NanoVDB's portable C and shader reader: the C side reads .nvdb grid
# headers, and shaders embed the same header to walk grids on the GPU.
add_library(ernie_nanovdb INTERFACE)
target_include_directories(ernie_nanovdb SYSTEM INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/nanovdb")
