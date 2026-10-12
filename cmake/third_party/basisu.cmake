set(ERNIE_DEP_VERSION_basisu
    v2_50
    CACHE STRING "Tag of Basis Universal to fetch.")
mark_as_advanced(ERNIE_DEP_VERSION_basisu)

# Basis Universal (Apache-2.0), only its transcoder: KTX2 textures in Basis's ETC1S and UASTC
# formats become BC blocks. Its own build (the encoder and tools) is not used, and its zstd sources
# are not compiled: zstd comes from OpenEXR's copy (ernie_zstd).
include(FetchContent)

FetchContent_Declare(
    basisu
    GIT_REPOSITORY https://github.com/BinomialLLC/basis_universal.git
    GIT_TAG ${ERNIE_DEP_VERSION_basisu}
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR no-build)
FetchContent_MakeAvailable(basisu)

add_library(ernie_basisu STATIC
            "${basisu_SOURCE_DIR}/transcoder/basisu_transcoder.cpp")
target_include_directories(ernie_basisu SYSTEM
                           PUBLIC "${basisu_SOURCE_DIR}/transcoder")
target_compile_definitions(
    ernie_basisu PUBLIC BASISD_SUPPORT_KTX2=1
                        BASISD_SUPPORT_KTX2_ZSTD=1)
target_link_libraries(ernie_basisu PRIVATE ernie_zstd)
set_target_properties(
    ernie_basisu PROPERTIES POSITION_INDEPENDENT_CODE ON
                            COMPILE_WARNING_AS_ERROR OFF)
