# SPDX-License-Identifier: BSD-3-Clause
set(ERNIE_DEP_VERSION_openexr v3.5.2 CACHE STRING "Tag of OpenEXR to fetch")
set(ERNIE_DEP_VERSION_imath v3.2.3 CACHE STRING "Tag of Imath to fetch for OpenEXR")
mark_as_advanced(ERNIE_DEP_VERSION_openexr ERNIE_DEP_VERSION_imath)

set(OPENEXR_BUILD_TOOLS OFF)
set(OPENEXR_BUILD_EXAMPLES OFF)
set(OPENEXR_BUILD_DOCS OFF)
set(OPENEXR_BUILD_PYTHON OFF)
set(OPENEXR_INSTALL OFF)
set(OPENEXR_INSTALL_TOOLS OFF)
set(OPENEXR_INSTALL_PKG_CONFIG OFF)
set(OPENEXR_TEST_LIBRARIES OFF)
set(OPENEXR_TEST_TOOLS OFF)
set(OPENEXR_TEST_PYTHON OFF)
set(OPENEXR_FORCE_INTERNAL_IMATH ON)
set(OPENEXR_FORCE_INTERNAL_DEFLATE ON)
set(OPENEXR_FORCE_INTERNAL_ZSTD ON)
set(OPENEXR_FORCE_INTERNAL_OPENJPH ON)
set(IMATH_INSTALL_PKG_CONFIG OFF)
set(IMATH_INSTALL OFF)
set(BUILD_SHARED_LIBS OFF)

# Their own tests stay out of Ernie's ctest. Imath is made available first, so OpenEXR finds it
# populated and skips its own git clone of it.
set(ernie_saved_build_testing ${BUILD_TESTING})
set(BUILD_TESTING OFF)
ernie_fetch(
    Imath
    URL
        https://github.com/AcademySoftwareFoundation/Imath/archive/refs/tags/${ERNIE_DEP_VERSION_imath}.tar.gz
)
ernie_fetch(
    openexr
    URL
        https://github.com/AcademySoftwareFoundation/openexr/archive/refs/tags/${ERNIE_DEP_VERSION_openexr}.tar.gz
)
set(BUILD_TESTING ${ernie_saved_build_testing})

# Its headers and Imath's are system headers to Ernie, kept out of its warnings.
foreach(
    target
    OpenEXR
    OpenEXRCore
    OpenEXRUtil
    Iex
    IlmThread
    Imath
    ImathConfig
    OpenEXRConfig
)
    if(TARGET ${target})
        ernie_quiet_third_party(${target})
        get_target_property(directories ${target} INTERFACE_INCLUDE_DIRECTORIES)
        if(directories)
            set_property(
                TARGET ${target}
                APPEND
                PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES ${directories}
            )
        endif()
    endif()
endforeach()

add_library(ernie_openexr INTERFACE)
target_link_libraries(ernie_openexr INTERFACE OpenEXR::OpenEXR)

# The zstd built into OpenEXRCore exports its API, so KTX2's zstd supercompression is decoded with
# it rather than a second copy.
add_library(ernie_zstd INTERFACE)
target_include_directories(ernie_zstd SYSTEM INTERFACE "${openexr_SOURCE_DIR}/external/zstd/lib")
target_link_libraries(ernie_zstd INTERFACE OpenEXR::OpenEXRCore)
