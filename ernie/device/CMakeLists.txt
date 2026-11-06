# SPDX-License-Identifier: BSD-3-Clause
ernie_object_library(
    ernie_device
    buffer.cpp
    commands.cpp
    compute/runtime_compiler.cpp
    cuda/api.cpp
    cuda/adapters.cpp
    cuda/device.cpp
    device.cpp
    hip/api.cpp
    hip/adapters.cpp
    hip/device.cpp
    slang/compiler.cpp
    slang/debug.cpp
    program.cpp
    queue.cpp
    tensor.cpp
    vulkan/api.cpp
    vulkan/adapters.cpp
    vulkan/device.cpp
    vulkan/interop.cpp
    vulkan/native.cpp
    vulkan/queue.cpp
)
target_compile_definitions(ernie_device PRIVATE VK_NO_PROTOTYPES)
target_link_libraries(
    ernie_device
    PUBLIC
        ernie_util
        ernie_dlpack
    PRIVATE
        Vulkan::Headers
        volk::volk
        ernie_vma
        ernie_nvtx
        slang::slang
)

set(ernie_prelude_inputs
    "${slang_SOURCE_DIR}/include/slang-cuda-prelude.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/compute/prelude.cuh"
    "${PROJECT_SOURCE_DIR}/cmake/ErniePrelude.cmake"
)
foreach(backend IN ITEMS cuda hip)
    set(prelude "${CMAKE_CURRENT_BINARY_DIR}/${backend}_prelude.h")
    set(arguments
        "-DPRELUDE=${slang_SOURCE_DIR}/include/slang-cuda-prelude.h"
        "-DCOMPUTE_OVERLAY=${CMAKE_CURRENT_SOURCE_DIR}/compute/prelude.cuh"
        "-DOUTPUT=${prelude}"
    )
    set(inputs ${ernie_prelude_inputs})
    if(backend STREQUAL "hip")
        list(APPEND arguments "-DHIP_OVERLAY=${CMAKE_CURRENT_SOURCE_DIR}/hip/prelude.hip")
        list(APPEND inputs "${CMAKE_CURRENT_SOURCE_DIR}/hip/prelude.hip")
    endif()
    add_custom_command(
        OUTPUT
            "${prelude}"
        COMMAND
            "${CMAKE_COMMAND}" ${arguments} -P "${PROJECT_SOURCE_DIR}/cmake/ErniePrelude.cmake"
        DEPENDS
            ${inputs}
        VERBATIM
    )
    ernie_embed(ernie_device FILE "${prelude}" SYMBOL ernie_${backend}_prelude)
endforeach()
ernie_embed(
    ernie_device
    LIBRARY kDeviceKernels
    ROOTS
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels|ernie/device/kernels/|*.slang"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels|ernie/device/kernels/|*.h"
)

ernie_static_library(
    ernie_device_internal
    ernie_device
    ernie_util_internal
    volk::volk
    ernie_vma
    ernie_nvtx
    slang::slang
)
target_compile_definitions(ernie_device_internal INTERFACE VK_NO_PROTOTYPES)

ernie_public_files(
    adapter.h
    device.h
    kernels/shared.h
    operation.h
    ref.h
    stream.h
    tensor.h
    vulkan/native.h
)
