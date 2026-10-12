# PNanoVDB (Apache-2.0), NanoVDB's portable C/HLSL reader, vendored in third_party/nanovdb: the
# C side reads .nvdb files' grid headers, and the shaders embed the same header to walk the grids
# on the GPU.
add_library(ernie_nanovdb INTERFACE)
target_include_directories(
    ernie_nanovdb SYSTEM
    INTERFACE "${ERNIE_SOURCE_DIR}/third_party/nanovdb")
