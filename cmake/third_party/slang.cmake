# SPDX-License-Identifier: BSD-3-Clause
set(ERNIE_DEP_VERSION_slang
    2026.19
    CACHE STRING
    "Release of Slang to fetch. Changing it requires updating the hashes below."
)
mark_as_advanced(ERNIE_DEP_VERSION_slang)

# Prebuilt Slang: the compiler Ernie links to turn Slang modules into SPIR-V (and WGSL) at run
# time, so user models compile into Ernie's kernels. The Linux builds target old glibc versions,
# so they run wherever manylinux_2_34 wheels do.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(slang_platform linux-x86_64-glibc-2.27)
    set(slang_sha256 ae348a33d906ed29d14c2294f2260746cb2c3eb4aa2c7fc4bfd887a3a298737d)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(slang_platform linux-aarch64-glibc-2.28)
    set(slang_sha256 52b1e6b2aad2d9a119308fe8f99d1699d1854e4ac2d7b579cf0f8f25349bcd17)
elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(slang_platform macos-aarch64)
    set(slang_sha256 f77d89b91ec63698a7a8e5aa4316ce08b70d3211a236640ee4551f55be85d9ed)
elseif(APPLE)
    set(slang_platform macos-x86_64)
    set(slang_sha256 c5e18233c0b90cc651c792ec4c4e4a467a9a101ffb6301643aab89c016b48e42)
elseif(WIN32 AND CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64")
    set(slang_platform windows-x86_64)
    set(slang_sha256 fc922f214b5d115f663932e59e80e0f6b6cc6fae140979cf307da6459b6f8f33)
else()
    message(FATAL_ERROR "No prebuilt Slang for ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
endif()

ernie_fetch(
    slang
    URL
        https://github.com/shader-slang/slang/releases/download/v${ERNIE_DEP_VERSION_slang}/slang-${ERNIE_DEP_VERSION_slang}-${slang_platform}.zip
    URL_HASH SHA256=${slang_sha256}
    SOURCE_SUBDIR no-build
)

# Each release keeps its CMake package in lib/ (Linux x86_64, macOS), lib64/ (Linux aarch64) or
# the top (Windows).
find_package(
    slang
    CONFIG
    REQUIRED
    PATHS "${slang_SOURCE_DIR}"
    PATH_SUFFIXES lib/cmake/slang lib64/cmake/slang cmake
    NO_DEFAULT_PATH
)
set_target_properties(slang::slang PROPERTIES IMPORTED_GLOBAL TRUE)

# The runtime files Ernie's binaries need beside them: the compiler library, and the glslang
# library it loads at run time.
get_target_property(slang_library slang::slang IMPORTED_LOCATION_RELEASE)
if(NOT slang_library)
    get_target_property(slang_library slang::slang IMPORTED_LOCATION)
endif()
get_filename_component(ERNIE_SLANG_RUNTIME_DIR "${slang_library}" DIRECTORY)
file(
    GLOB ERNIE_SLANG_RUNTIME
    "${ERNIE_SLANG_RUNTIME_DIR}/libslang-compiler.so.*"
    "${ERNIE_SLANG_RUNTIME_DIR}/libslang-glslang-*.so"
    "${ERNIE_SLANG_RUNTIME_DIR}/libslang-compiler*.dylib"
    "${ERNIE_SLANG_RUNTIME_DIR}/libslang-glslang-*.dylib"
    "${ERNIE_SLANG_RUNTIME_DIR}/slang-compiler.dll"
    "${ERNIE_SLANG_RUNTIME_DIR}/slang-glslang.dll"
)
# Windows has no rpath: a program finds its DLLs beside it, so the build's programs get Slang's.
if(WIN32 AND CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    file(COPY ${ERNIE_SLANG_RUNTIME} DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
endif()
