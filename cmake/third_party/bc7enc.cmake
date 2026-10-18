# SPDX-License-Identifier: BSD-3-Clause
# bc7enc_rdo has no releases, so a commit is pinned.
set(ERNIE_DEP_VERSION_bc7enc
    b9438627eef73a1157e84201b6fa6eb2ffd6d9f0
    CACHE STRING
    "Commit of bc7enc_rdo to fetch"
)
mark_as_advanced(ERNIE_DEP_VERSION_bc7enc)

ernie_fetch(
    bc7enc
    URL https://github.com/richgel999/bc7enc_rdo/archive/${ERNIE_DEP_VERSION_bc7enc}.tar.gz
    SOURCE_SUBDIR no-build
)

# bc7enc_rdo (MIT or public domain): its BC7 encoder (bc7enc), BC1 to BC5 encoders and decoders
# (rgbcx) and BC7 decoder (bc7decomp), for baking and decoding textures.
add_library(
    ernie_bc7enc
    STATIC
    "${bc7enc_SOURCE_DIR}/bc7enc.cpp"
    "${bc7enc_SOURCE_DIR}/rgbcx.cpp"
    "${bc7enc_SOURCE_DIR}/bc7decomp.cpp"
)
target_include_directories(ernie_bc7enc SYSTEM PUBLIC "${bc7enc_SOURCE_DIR}")
set_target_properties(ernie_bc7enc PROPERTIES POSITION_INDEPENDENT_CODE ON)
