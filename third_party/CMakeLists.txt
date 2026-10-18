# SPDX-License-Identifier: BSD-3-Clause
# Vendored dependencies, kept byte for byte as their upstream releases; each folder's README or
# LICENSE says which. The .cpp files here are Ernie's, compiling header-only libraries once.

# glm (MIT), headers only; Ernie's own glm configuration is in ernie_build_options, so it never
# reaches consumers.
add_library(ernie_glm INTERFACE)
target_include_directories(ernie_glm SYSTEM INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/glm")

# DLPack (Apache-2.0), the tensor exchange header.
add_library(ernie_dlpack INTERFACE)
target_include_directories(
    ernie_dlpack
    SYSTEM
    INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/dlpack/include"
)

# nlohmann/json (MIT), headers only.
add_library(ernie_json INTERFACE)
target_include_directories(ernie_json SYSTEM INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/json/include")

# bc7enc_rdo (MIT or public domain): its BC7 encoder (bc7enc), BC1 to BC5 encoders and decoders
# (rgbcx) and BC7 decoder (bc7decomp), for baking and decoding textures.
add_library(ernie_bc7enc STATIC bc7enc/bc7enc.cpp bc7enc/rgbcx.cpp bc7enc/bc7decomp.cpp)
target_include_directories(ernie_bc7enc SYSTEM PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/bc7enc")
set_target_properties(ernie_bc7enc PROPERTIES POSITION_INDEPENDENT_CODE ON)

# stb_image (PNG, JPEG) and stb_image_write (MIT or public domain).
add_library(ernie_stb STATIC stb.cpp)
target_include_directories(ernie_stb SYSTEM PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/stb")
target_compile_definitions(ernie_stb PUBLIC STBI_ONLY_PNG STBI_ONLY_JPEG)
set_target_properties(ernie_stb PROPERTIES POSITION_INDEPENDENT_CODE ON)

# The Vulkan C headers (Apache-2.0 or MIT).
add_library(ernie_vulkan_headers INTERFACE)
target_include_directories(
    ernie_vulkan_headers
    SYSTEM
    INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/vulkan_headers/include"
)

# volk (MIT) loads the Vulkan loader at run time, so nothing links against libvulkan.
add_library(ernie_volk STATIC volk/volk.c)
target_include_directories(ernie_volk SYSTEM PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/volk")
target_compile_definitions(ernie_volk PUBLIC VK_NO_PROTOTYPES)
target_link_libraries(ernie_volk PUBLIC ernie_vulkan_headers ${CMAKE_DL_LIBS})
set_target_properties(ernie_volk PROPERTIES POSITION_INDEPENDENT_CODE ON)
# Loads the Win32 sharing functions, which only external.cpp declares.
if(WIN32)
    target_compile_definitions(ernie_volk PRIVATE VK_USE_PLATFORM_WIN32_KHR)
endif()

# Vulkan Memory Allocator (MIT) is header-only; vma.cpp compiles its implementation once, fetching
# Vulkan functions through volk.
add_library(ernie_vma STATIC vma.cpp)
target_include_directories(ernie_vma SYSTEM PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/vma/include")
target_compile_definitions(
    ernie_vma
    PUBLIC VMA_STATIC_VULKAN_FUNCTIONS=0 VMA_DYNAMIC_VULKAN_FUNCTIONS=1
)
target_link_libraries(ernie_vma PUBLIC ernie_volk)
set_target_properties(ernie_vma PROPERTIES POSITION_INDEPENDENT_CODE ON)
ernie_quiet_third_party(ernie_vma)

# Basis Universal's transcoder (Apache-2.0): KTX2 textures in Basis's ETC1S and UASTC formats
# become BC blocks. Its zstd calls link against OpenEXR's copy (ernie_zstd).
add_library(ernie_basisu STATIC basisu/transcoder/basisu_transcoder.cpp)
target_include_directories(
    ernie_basisu
    SYSTEM
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/basisu/transcoder"
)
target_compile_definitions(ernie_basisu PUBLIC BASISD_SUPPORT_KTX2=1 BASISD_SUPPORT_KTX2_ZSTD=1)
target_link_libraries(ernie_basisu PRIVATE ernie_zstd)
set_target_properties(ernie_basisu PROPERTIES POSITION_INDEPENDENT_CODE ON)
ernie_quiet_third_party(ernie_basisu)

# PNanoVDB (Apache-2.0), NanoVDB's portable C and shader reader: the C side reads .nvdb grid
# headers, and shaders embed the same header to walk grids on the GPU.
add_library(ernie_nanovdb INTERFACE)
target_include_directories(ernie_nanovdb SYSTEM INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/nanovdb")
