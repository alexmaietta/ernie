# SPDX-License-Identifier: BSD-3-Clause
FetchContent_Declare(
    fmt
    URL
        https://github.com/fmtlib/fmt/archive/refs/tags/12.1.0.tar.gz
    URL_HASH SHA256=ea7de4299689e12b6dddd392f9896f08fb0777ac7168897a244a6d6085043fea
    SOURCE_SUBDIR none
)
FetchContent_MakeAvailable(fmt)

add_library(ernie_fmt INTERFACE)
target_include_directories(ernie_fmt SYSTEM INTERFACE "${fmt_SOURCE_DIR}/include")
target_compile_definitions(
    ernie_fmt
    INTERFACE
        ERNIE_FMT
        FMT_HEADER_ONLY
)
target_compile_options(ernie_fmt INTERFACE "$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
