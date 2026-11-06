# SPDX-License-Identifier: BSD-3-Clause
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

set(ernie_saved_build_testing ${BUILD_TESTING})
set(BUILD_TESTING OFF)
FetchContent_Declare(
    Imath
    URL
        https://github.com/AcademySoftwareFoundation/Imath/archive/refs/tags/v3.2.3.tar.gz
    URL_HASH SHA256=e10c12b3f21f45bf08e09d4215d9c7691368d747beebd840de0b6fefed2df9f8
)
FetchContent_Declare(
    openexr
    URL
        https://github.com/AcademySoftwareFoundation/openexr/archive/refs/tags/v3.5.2.tar.gz
    URL_HASH SHA256=85a291b9b8563fabef1aa3f6e912d744ce1d4b9fd5a8f978eda194cd51b15ae9
)
FetchContent_MakeAvailable(
    Imath
    openexr
)
set(BUILD_TESTING ${ernie_saved_build_testing})

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
        get_target_property(type ${target} TYPE)
        if(NOT type STREQUAL "INTERFACE_LIBRARY")
            target_compile_options(${target} PRIVATE "$<IF:$<CXX_COMPILER_ID:MSVC>,/w,-w>")
        endif()
        get_target_property(directories ${target} INTERFACE_INCLUDE_DIRECTORIES)
        if(directories)
            set_property(
                TARGET
                    ${target}
                APPEND
                PROPERTY
                    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                        ${directories}
            )
        endif()
    endif()
endforeach()

add_library(ernie_openexr INTERFACE)
target_link_libraries(ernie_openexr INTERFACE OpenEXR::OpenEXR)

add_library(ernie_zstd INTERFACE)
target_include_directories(ernie_zstd SYSTEM INTERFACE "${openexr_SOURCE_DIR}/external/zstd/lib")
target_link_libraries(ernie_zstd INTERFACE OpenEXR::OpenEXRCore)
