# OpenUSD for the USD importer plugin (ernie/usd), and nothing else: Ernie's core and Python
# module never link it. Built from source by OpenUSD's own build_usd.py (which builds oneTBB,
# OpenSubdiv and MaterialX too) when the plugin is first built, not at configure, into an
# install directory that a CI cache keeps: one library (usd_ms), imaging and UsdImaging (Hydra's
# scene indices) without OpenGL, Metal, Storm, Python or usdview. ERNIE_USD_ROOT names an
# installed OpenUSD instead, such as the one a host process (Isaac Sim, Kit) already loads, since
# a process must not load two.
#
# References:
#   https://github.com/PixarAnimationStudios/OpenUSD/blob/release/BUILDING.md

set(ERNIE_DEP_VERSION_usd
    26.08
    CACHE STRING "OpenUSD release to build.")
set(ERNIE_DEP_HASH_usd
    "SHA256=4bccbb95cddda1dbeef2f74a08b9456352f2aa91bfd4578c0c613009c7950149"
    CACHE STRING "Hash of the OpenUSD release's source archive.")
mark_as_advanced(ERNIE_DEP_VERSION_usd ERNIE_DEP_HASH_usd)
set(ERNIE_USD_ROOT
    ""
    CACHE PATH
          "An installed OpenUSD with imaging (include/pxr, lib/libusd_ms or lib/libusd_*), in place of building one"
)
set(ERNIE_USD_TBB_ROOT
    ""
    CACHE PATH "oneTBB's headers for ERNIE_USD_ROOT's OpenUSD, when not beside it")
set(ERNIE_USD_BUILD_JOBS
    4
    CACHE STRING "Parallel jobs building OpenUSD and its dependencies")

if(WIN32)
    message(FATAL_ERROR "ERNIE_USD: the USD importer plugin builds on Linux and macOS for now")
endif()

add_library(ernie_openusd INTERFACE)

if(ERNIE_USD_ROOT)
    set(ernie_usd_dir "${ERNIE_USD_ROOT}")
    if(ERNIE_USD_TBB_ROOT)
        target_include_directories(ernie_openusd SYSTEM INTERFACE "${ERNIE_USD_TBB_ROOT}/include")
    endif()
    if(NOT EXISTS "${ernie_usd_dir}/include/pxr/usdImaging/usdImaging/sceneIndices.h")
        message(FATAL_ERROR "The OpenUSD in ${ernie_usd_dir} has no UsdImaging (Hydra scene indices)")
    endif()
    # One monolithic library, or the per-module ones a default build (or Kit's) makes.
    set(ernie_usd_libraries "${ernie_usd_dir}/lib/libusd_ms${CMAKE_SHARED_LIBRARY_SUFFIX}")
    if(NOT EXISTS "${ernie_usd_libraries}")
        set(ernie_usd_libraries "")
        foreach(
            module
            arch
            tf
            gf
            js
            trace
            work
            plug
            vt
            ts
            ar
            kind
            sdf
            pcp
            usd
            usdGeom
            usdShade
            usdLux
            usdSkel
            usdVol
            usdRender
            sdr
            hf
            hd
            hdsi
            cameraUtil
            geomUtil
            pxOsd
            usdImaging
            usdSkelImaging)
            set(library "${ernie_usd_dir}/lib/libusd_${module}${CMAKE_SHARED_LIBRARY_SUFFIX}")
            if(EXISTS "${library}")
                list(APPEND ernie_usd_libraries "${library}")
            endif()
        endforeach()
    endif()
else()
    find_package(Python3 3.9 REQUIRED COMPONENTS Interpreter)
    include(ExternalProject)
    set(ernie_usd_args
        --build-variant
        release
        --build-monolithic
        --no-python
        --no-usdview
        --no-tests
        --no-examples
        --no-tutorials
        --no-tools
        --no-docs
        --no-python-docs
        --no-usdValidation
        --usd-imaging
        --materialx
        --onetbb
        --no-ptex
        --no-openvdb
        --no-embree
        --no-prman
        --no-openimageio
        --no-opencolorio
        --no-alembic
        --no-draco
        --no-vulkan
        --no-compiler-cache
        --build-args
        "USD,-DPXR_ENABLE_GL_SUPPORT=OFF -DPXR_ENABLE_METAL_SUPPORT=OFF")
    # The install directory beside the fetched sources (FETCHCONTENT_BASE_DIR, which CI caches),
    # named for the version, options and compiler so that a change builds another.
    if(NOT FETCHCONTENT_BASE_DIR)
        set(FETCHCONTENT_BASE_DIR "${CMAKE_BINARY_DIR}/_deps")
    endif()
    string(SHA1 ernie_usd_options
                "${ERNIE_DEP_VERSION_usd};${ernie_usd_args};${CMAKE_CXX_COMPILER_ID}")
    string(SUBSTRING "${ernie_usd_options}" 0 10 ernie_usd_options)
    set(ernie_usd_dir
        "${FETCHCONTENT_BASE_DIR}/openusd-${ERNIE_DEP_VERSION_usd}-${ernie_usd_options}")
    set(ernie_usd_libraries "${ernie_usd_dir}/lib/libusd_ms${CMAKE_SHARED_LIBRARY_SUFFIX}")
    string(REPLACE ";" "|" ernie_usd_args_joined "${ernie_usd_args}")
    if(EXISTS "${ernie_usd_dir}/.built")
        # Built already (by an earlier build, or restored from a cache).
        add_custom_target(ernie_openusd_build)
    else()
        ExternalProject_Add(
            ernie_openusd_build
            URL "https://github.com/PixarAnimationStudios/OpenUSD/archive/refs/tags/v${ERNIE_DEP_VERSION_usd}.tar.gz"
            URL_HASH ${ERNIE_DEP_HASH_usd}
            DOWNLOAD_DIR "${FETCHCONTENT_BASE_DIR}/openusd-download"
            PREFIX "${FETCHCONTENT_BASE_DIR}/openusd-work"
            CONFIGURE_COMMAND ""
            # Under a lock (openusd_build.cmake): two builds started at once wait for each other.
            BUILD_COMMAND
                ${CMAKE_COMMAND} "-DPYTHON=${Python3_EXECUTABLE}" -DUSD_SOURCE=<SOURCE_DIR>
                -DUSD_WORK=<BINARY_DIR> "-DUSD_INSTALL=${ernie_usd_dir}"
                -DUSD_JOBS=${ERNIE_USD_BUILD_JOBS} "-DUSD_ARGS=${ernie_usd_args_joined}" -P
                "${CMAKE_CURRENT_LIST_DIR}/openusd_build.cmake"
            INSTALL_COMMAND ""
            BUILD_BYPRODUCTS "${ernie_usd_libraries}"
            LOG_BUILD ON
            LOG_OUTPUT_ON_FAILURE ON)
    endif()
    add_dependencies(ernie_openusd ernie_openusd_build)
endif()

target_include_directories(ernie_openusd SYSTEM INTERFACE "${ernie_usd_dir}/include")
target_link_libraries(ernie_openusd INTERFACE ${ernie_usd_libraries})
# Where the plugin finds OpenUSD (its plugInfo files beside it) and what it links (oneTBB,
# OpenSubdiv and MaterialX in the same directory).
set(ERNIE_USD_RPATH "${ernie_usd_dir}/lib")
