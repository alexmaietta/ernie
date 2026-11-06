# SPDX-License-Identifier: BSD-3-Clause
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(slang_platform linux-x86_64-glibc-2.27)
    set(slang_hash 2a92a877c62452d249436cca39f76c419992eb560293d06e7ea94ec80f27380d)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(slang_platform linux-aarch64-glibc-2.28)
    set(slang_hash 8ce90d69a6183f6160f868945e82b84a2729ccf4e7f07cd806dab06733cc0cfa)
elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(slang_platform macos-aarch64)
    set(slang_hash ede1865bf086442c8786c5b7c0470ce5f4fd4f032eefe881356726ff9c445f40)
elseif(APPLE)
    set(slang_platform macos-x86_64)
    set(slang_hash a71d4a499d08b6178325a697a5a5c9fddc9eee5914daaf195451b283189e96ce)
elseif(WIN32 AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(slang_platform windows-x86_64)
    set(slang_hash e6dd32bbddf1720ebb4e679d55e9635ce792865a5bd84f3d2cd9af28647c0174)
elseif(WIN32 AND CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|aarch64")
    set(slang_platform windows-aarch64)
    set(slang_hash 4f18a7e76a59969bc9bf6ea596bd9e496a3c689af173990ae028b18f03e1ac1a)
else()
    message(FATAL_ERROR "No prebuilt Slang for ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
endif()
FetchContent_Declare(
    slang
    URL
        https://github.com/shader-slang/slang/releases/download/v2026.20/slang-2026.20-${slang_platform}.zip
    URL_HASH SHA256=${slang_hash}
)
FetchContent_MakeAvailable(slang)
find_package(
    slang
    CONFIG
    REQUIRED
    PATHS
        "${slang_SOURCE_DIR}"
    PATH_SUFFIXES
        lib/cmake/slang
        lib64/cmake/slang
        cmake
    NO_DEFAULT_PATH
)

get_target_property(slang_library slang::slang IMPORTED_LOCATION_RELEASE)
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
if(WIN32)
    file(COPY ${ERNIE_SLANG_RUNTIME} DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
endif()
