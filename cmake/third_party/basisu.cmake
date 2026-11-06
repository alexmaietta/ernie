# SPDX-License-Identifier: BSD-3-Clause
set(basisu_commit 9bebe16726b3a61c8c213eeee3b7cffb462ef34e)
set(basisu_url "https://raw.githubusercontent.com/BinomialLLC/basis_universal/${basisu_commit}")
set(basisu_SOURCE_DIR "${FETCHCONTENT_BASE_DIR}/basisu-src")
set(basisu_files
    LICENSE
    transcoder/basisu.h
    transcoder/basisu_astc_cfgs.inl
    transcoder/basisu_astc_hdr_core.h
    transcoder/basisu_astc_helpers.h
    transcoder/basisu_containers.h
    transcoder/basisu_containers_impl.h
    transcoder/basisu_dds_transcoder.inl
    transcoder/basisu_etc1_mods.inl
    transcoder/basisu_file_headers.h
    transcoder/basisu_idct.h
    transcoder/basisu_transcoder.cpp
    transcoder/basisu_transcoder.h
    transcoder/basisu_transcoder_internal.h
    transcoder/basisu_transcoder_tables_astc.inc
    transcoder/basisu_transcoder_tables_astc_0_255.inc
    transcoder/basisu_transcoder_tables_atc_55.inc
    transcoder/basisu_transcoder_tables_atc_56.inc
    transcoder/basisu_transcoder_tables_bc7_m5_alpha.inc
    transcoder/basisu_transcoder_tables_bc7_m5_color.inc
    transcoder/basisu_transcoder_tables_dxt1_5.inc
    transcoder/basisu_transcoder_tables_dxt1_6.inc
    transcoder/basisu_transcoder_tables_pvrtc2_45.inc
    transcoder/basisu_transcoder_tables_pvrtc2_alpha_33.inc
    transcoder/basisu_transcoder_uastc.h
    transcoder/basisu_xbc7_decoder.h
    transcoder/basisu_xbc7_decoder.inl
)
set(basisu_hashes
    065fcf48d6af21c0b75e23be5ed5753aee75c892e1c2cf178fa6736305614a5c
    2724dd59acbd332d43dde5f15a9b546ed851c45c7c2acbcdcab4a288c58628e0
    2f56d16cc37ddb6a5f361be42e3ab32ecbeefd94fd287d5bd46c15e0e1fdb34e
    4d1e2b68275c7f402623ab0824722d7e8d6e85adc2fc747a44c07e5321b366d4
    f698c23506c0e59f3bd4cb645bd3b4d071b06f1112a0cf6a0de19a19cff837f8
    84a17dcb701dd22a44381ae226e3e9cdc9d361e74efc640d0ccb8bbbb081ee41
    9810baeca8b39497101997cf309c0cee153f096af21f1eb8e12c142410f5ef2a
    18f731ebe27b311efda2a576b3bc76f1e1b541b25e9a790a9759f76effc61833
    ddf9efd40f07209c914f4051b94bbe1821915ae61b6ca39d61aafd0126c6dd0d
    5988f2d27e41a3f36ef5a2e350675dd30bd1ee2c70853e42a1d796a74d8178a9
    0b218ce8a9f72036c1e531387d2a168b950a2af2d5a40204f08a5ec381a2e4ee
    244cf0da60941c0311feed5e9c01b9225d6ca796292391d808a4fa3cf9e406f2
    d60e4d56c7dbc392ec476e484cc5030046ecd378c554a5ffa898ccd0e1ec1b08
    9e12e4543d2872c4c0beb91bf9438d5bed1205d52358fd15671b747d8485369f
    7dd41c648d3ed8e7d511eb4af865669c4dd50b8ecead62a68fca471138c46e1a
    f41acb498be64349c48600e965664f4bec1161781169e3ec3411827c490ef4f8
    4a510a20bffc3738d85d41e44633d44145931aa5734affb390c7836ec8b5dabb
    94714b54f0c13c2f3ba5481fbfb40b974892048d59ab68419b369222f0a68075
    4d713b839594f429d95f329d60f8ad495e932619111b5dfddffb688e3634b495
    e6f63d3ebce0ac84d801ec54614b6a4eeda3be3e994f3abbfc732377cfe5a531
    500e640b5e6858fd10194eb4845670eb36d3ef9dc117a017632c1104ba3eb9d5
    3d203465f9097845eddb821943d312bf8cffb04d8f5ce09c18d4c6df40519ed3
    780711642098a91239ee37b3f9110469ee1c7da8dd9e440a43c7ba70f992ff9d
    250d389235c0213757266441f72aee37cbe138a3c2a3f58b590ed53d6fb746fa
    8796f8e755b868c56967e3d4d53089c2aaac7e1f345b03be1e975016fcbaf11e
    7487d5c43fe2ff97aba5a3cee87cfcdcb433f003b72c47927d4d4aeebde7b2ea
    485b6c4b9297047430ea8124319501d58389d96da38642f59dafd974d88d65d1
)
foreach(basisu_file basisu_hash IN ZIP_LISTS basisu_files basisu_hashes)
    set(basisu_path "${basisu_SOURCE_DIR}/${basisu_file}")
    set(basisu_existing "")
    if(EXISTS "${basisu_path}")
        file(SHA256 "${basisu_path}" basisu_existing)
    endif()
    if(NOT basisu_existing STREQUAL basisu_hash)
        file(
            DOWNLOAD "${basisu_url}/${basisu_file}"
            "${basisu_path}"
            EXPECTED_HASH SHA256=${basisu_hash}
            TLS_VERIFY ON
        )
    endif()
endforeach()

file(
    CONFIGURE
    OUTPUT "${basisu_SOURCE_DIR}/zstd/zstd.h"
    CONTENT "#pragma once\n#include <zstd.h>\n"
)
add_library(ernie_basisu STATIC "${basisu_SOURCE_DIR}/transcoder/basisu_transcoder.cpp")
target_include_directories(ernie_basisu SYSTEM PUBLIC "${basisu_SOURCE_DIR}/transcoder")
target_compile_definitions(
    ernie_basisu
    PUBLIC
        BASISD_SUPPORT_KTX2=1
        BASISD_SUPPORT_KTX2_ZSTD=1
        BASISD_SUPPORT_ATC=0
        BASISD_SUPPORT_PVRTC2=0
        BASISD_SUPPORT_FXT1=0
        BASISD_SUPPORT_ETC2_EAC_RG11=0
        BASISD_SUPPORT_ASTC_HIGHER_OPAQUE_QUALITY=0
)
target_link_libraries(ernie_basisu PRIVATE ernie_zstd)
set_target_properties(
    ernie_basisu
    PROPERTIES
        POSITION_INDEPENDENT_CODE
            ON
)
target_compile_options(
    ernie_basisu
    PRIVATE
        "$<IF:$<CXX_COMPILER_ID:MSVC>,/w,-w>"
        "$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fno-strict-aliasing>"
)
