# SPDX-License-Identifier: BSD-3-Clause
# libernie_usd (ernie_usd.dll on Windows), the USD importer plugin Ernie loads at run time
# (ernie/scene/usd_import.cpp): the only target linking OpenUSD.
include("${PROJECT_SOURCE_DIR}/cmake/third_party/openusd.cmake")

add_library(ernie_usd MODULE usd_plugin.cpp)
target_include_directories(ernie_usd PRIVATE "${PROJECT_SOURCE_DIR}")
target_compile_features(ernie_usd PRIVATE cxx_std_20)
target_link_libraries(ernie_usd PRIVATE ernie_openusd)
# An OpenUSD Ernie built has its headers patched for GCC 13 (openusd_patch.cmake) once it is
# built, before the plugin compiles; a host's (ERNIE_USD_ROOT) stays as it is.
if(NOT ERNIE_USD_ROOT)
    add_custom_target(
        ernie_openusd_patch
        COMMAND
            ${CMAKE_COMMAND} "-DUSD_INSTALL=${ernie_usd_dir}" -P
            "${PROJECT_SOURCE_DIR}/cmake/third_party/openusd_patch.cmake"
        VERBATIM)
    add_dependencies(ernie_openusd_patch ernie_openusd_build)
    add_dependencies(ernie_usd ernie_openusd_patch)
endif()
# oneTBB's headers ask MSVC to link tbb12.lib, which lies beside OpenUSD's.
if(WIN32)
    target_link_directories(ernie_usd PRIVATE "${ERNIE_USD_RPATH}")
endif()
# In lib/ whatever the configuration (the generator expression keeps multi-config generators from
# adding one's folder).
if(WIN32)
    set(ernie_usd_prefix "")
else()
    set(ernie_usd_prefix "lib")
endif()
set_target_properties(
    ernie_usd
    PROPERTIES PREFIX "${ernie_usd_prefix}"
               OUTPUT_NAME "ernie_usd"
               LIBRARY_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/lib>"
               CXX_VISIBILITY_PRESET hidden
               VISIBILITY_INLINES_HIDDEN ON
               BUILD_RPATH "${ERNIE_USD_RPATH}")
# OpenUSD's headers warn under Ernie's flags; the plugin keeps the defaults. DT_RPATH, unlike
# RUNPATH, also finds what libusd_ms itself links (oneTBB, OpenSubdiv, MaterialX) beside it.
if(MSVC)
    # What OpenUSD's own MSVC build defines and turns on (cmake/defaults/msvcdefaults.cmake), so
    # that its headers compile as they do there.
    target_compile_options(
        ernie_usd
        PRIVATE /EHsc
                /permissive-
                /Zc:inline
                /Zc:__cplusplus
                /bigobj
                /wd4003
                /wd4244
                /wd4305
                /wd4267
                /wd4506
                /wd4091
                /wd4273
                /wd4180
                /wd4334)
    target_compile_definitions(
        ernie_usd
        PRIVATE NOMINMAX
                WIN32_LEAN_AND_MEAN
                _CRT_SECURE_NO_WARNINGS
                _CRT_NONSTDC_NO_WARNINGS
                _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING
                BOOST_ALL_NO_LIB)
else()
    target_compile_options(ernie_usd PRIVATE -Wall -Wno-deprecated)
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(ernie_usd PRIVATE
                        "LINKER:--disable-new-dtags")
endif()
# Windows has no rpath: the folders of OpenUSD's DLLs, one a line, beside the plugin
# (ernie_usd.paths), for Ernie to add to the DLL search path before loading it.
if(WIN32)
    string(REPLACE ";" "\n" ernie_usd_dll_dirs
                   "${ERNIE_USD_DLL_DIRS}")
    file(
        GENERATE
        OUTPUT "${CMAKE_BINARY_DIR}/lib/ernie_usd.paths"
        CONTENT "${ernie_usd_dll_dirs}\n")
endif()
