# SPDX-License-Identifier: BSD-3-Clause
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

include(FetchContent)

file(GLOB MADRONA_PATCHES "${CMAKE_CURRENT_SOURCE_DIR}/patches/*.patch")
list(SORT MADRONA_PATCHES)
list(JOIN MADRONA_PATCHES " " MADRONA_PATCHES)

FetchContent_Populate(
    madrona
    GIT_REPOSITORY https://github.com/shacklettbp/madrona.git
    GIT_TAG b908c4326bf0b0877babff33863ecb908d16acad
    GIT_SUBMODULES_RECURSE TRUE
    GIT_PROGRESS TRUE
    PATCH_COMMAND
        sh
        -c
        "git apply --reverse --check ${MADRONA_PATCHES} 2>/dev/null || git apply ${MADRONA_PATCHES}"
)

include("${madrona_SOURCE_DIR}/cmake/madrona_init.cmake")

project(
    ErnieBenchMadrona
    LANGUAGES
        C
        CXX
)

include(setup)
include(dependencies)

add_subdirectory("${madrona_SOURCE_DIR}" madrona EXCLUDE_FROM_ALL)
add_subdirectory(src)
