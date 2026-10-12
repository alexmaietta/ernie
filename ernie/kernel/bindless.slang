// SPDX-License-Identifier: BSD-3-Clause
// The device's bindless heap (`BindlessHeap`), set 1 of every pass: its textures and samplers by
// the index a pass's uniforms give. An index the same across a dispatch reads directly; one that
// varies between threads goes through NonUniformResourceIndex. A texture's view type decides its
// array: 2D views in `heapTextures`, array views (layered images, cube faces) in
// `heapTextureArrays`; read single channels as `.x`.
//
// Buffers are not here: shaders read them through their addresses (Slang pointers, natural
// layout), `GpuBuffer::Address`.
//
// Reference: Khronos, VK_EXT_descriptor_indexing (core in Vulkan 1.2).

[[vk::binding(0, 1)]] public Texture2D<float4> heapTextures[];
[[vk::binding(1, 1)]] public Texture2DArray<float4> heapTextureArrays[];
[[vk::binding(2, 1)]] public SamplerState heapSamplers[];
[[vk::binding(3, 1)]] public SamplerComparisonState heapComparisonSamplers[];
