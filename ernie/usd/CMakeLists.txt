# SPDX-License-Identifier: BSD-3-Clause
# libernie_usd, the USD importer plugin Ernie loads at run time (ernie/scene/usd_import.cpp): the
# only target linking OpenUSD.
include("${PROJECT_SOURCE_DIR}/cmake/third_party/openusd.cmake")

add_library(ernie_usd MODULE usd_plugin.cpp)
target_include_directories(ernie_usd PRIVATE "${PROJECT_SOURCE_DIR}")
target_compile_features(ernie_usd PRIVATE cxx_std_20)
target_link_libraries(ernie_usd PRIVATE ernie_openusd)
set_target_properties(
    ernie_usd
    PROPERTIES PREFIX "lib"
               OUTPUT_NAME "ernie_usd"
               LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
               CXX_VISIBILITY_PRESET hidden
               VISIBILITY_INLINES_HIDDEN ON
               BUILD_RPATH "${ERNIE_USD_RPATH}")
# OpenUSD's headers warn under Ernie's flags; the plugin keeps the defaults. DT_RPATH, unlike
# RUNPATH, also finds what libusd_ms itself links (oneTBB, OpenSubdiv, MaterialX) beside it.
if(NOT MSVC)
    target_compile_options(ernie_usd PRIVATE -Wall -Wno-deprecated)
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(ernie_usd PRIVATE
                        "LINKER:--disable-new-dtags")
endif()
