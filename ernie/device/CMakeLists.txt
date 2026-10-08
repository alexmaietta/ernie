ernie_add_library(
    ernie_device
    SOURCES
    buffer.cpp
    compute_interop.cpp
    cuda_interop.cpp
    frame_uploads.cpp
    gpu_timer.cpp
    image.cpp
    offscreen_target.cpp
    pipeline.cpp
    shader_compiler.cpp
    swapchain.cpp
    vulkan_context.cpp
    PUBLIC
    ernie_util
    ernie_volk
    ernie_vma
    PRIVATE
    slang::slang
    ${CMAKE_DL_LIBS})
# Vulkan structs are initialised as `VkFoo info{VK_STRUCTURE_TYPE_FOO}` with the rest zeroed, here
# and in code that uses Ernie's Vulkan types.
target_compile_options(ernie_device
                       PUBLIC $<$<CXX_COMPILER_ID:GNU,Clang>:-Wno-missing-field-initializers>)
if(APPLE)
    target_compile_definitions(ernie_device PRIVATE ERNIE_MOLTENVK_PATH="${ERNIE_MOLTENVK_LIBRARY}")
endif()
