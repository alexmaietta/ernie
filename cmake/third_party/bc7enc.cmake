# SPDX-License-Identifier: BSD-3-Clause
FetchContent_Declare(
    bc7enc
    URL
        https://github.com/richgel999/bc7enc_rdo/archive/b9438627.tar.gz
    URL_HASH SHA256=bbb33d1dbb6178a3a2c044956c3cc0b8ea9fc2f2903824379dd86f87d53b64c6
    SOURCE_SUBDIR none
)
FetchContent_MakeAvailable(bc7enc)
add_library(
    ernie_bc7enc
    STATIC
    "${bc7enc_SOURCE_DIR}/bc7enc.cpp"
    "${bc7enc_SOURCE_DIR}/rgbcx.cpp"
    "${bc7enc_SOURCE_DIR}/bc7decomp.cpp"
)
target_include_directories(ernie_bc7enc SYSTEM PUBLIC "${bc7enc_SOURCE_DIR}")
set_target_properties(
    ernie_bc7enc
    PROPERTIES
        POSITION_INDEPENDENT_CODE
            ON
)
target_compile_options(ernie_bc7enc PRIVATE "$<IF:$<CXX_COMPILER_ID:MSVC>,/w,-w>")
