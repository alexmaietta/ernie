set(ERNIE_DEP_VERSION_bc7enc
    b9438627eef73a1157e84201b6fa6eb2ffd6d9f0
    CACHE STRING "Commit of bc7enc_rdo to fetch.")
mark_as_advanced(ERNIE_DEP_VERSION_bc7enc)

# bc7enc_rdo (MIT or public domain), its BC7 encoder (bc7enc) and its BC1 to BC5 encoders (rgbcx),
# for baking textures (util/texture_bake). Its own build (the command-line tool, ISPC) is not used.
# It has no release tags, so a commit is pinned, fetched whole.
include(FetchContent)

FetchContent_Declare(
    bc7enc
    GIT_REPOSITORY https://github.com/richgel999/bc7enc_rdo.git
    GIT_TAG ${ERNIE_DEP_VERSION_bc7enc}
    SOURCE_SUBDIR no-build)
FetchContent_MakeAvailable(bc7enc)

add_library(
    ernie_bc7enc STATIC
    "${bc7enc_SOURCE_DIR}/bc7enc.cpp"
    "${bc7enc_SOURCE_DIR}/rgbcx.cpp"
    # Its BC7 decoder, for checking encoders against their sources.
    "${bc7enc_SOURCE_DIR}/bc7decomp.cpp")
target_include_directories(ernie_bc7enc SYSTEM
                           PUBLIC "${bc7enc_SOURCE_DIR}")
set_target_properties(
    ernie_bc7enc PROPERTIES POSITION_INDEPENDENT_CODE ON
                            COMPILE_WARNING_AS_ERROR OFF)
