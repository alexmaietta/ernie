// SPDX-License-Identifier: BSD-3-Clause
/// @file bindless.slang
/// @brief The bindless heap's arrays, as `BindlessHeap` lays out set 0: kernels index them with
/// the `BindlessIndex` values the host passes.
///
///     import bindless;
///     images[NonUniformResourceIndex(parameters.image)][pixel] = colour;

[[vk::binding(0, 0)]] public Texture2D textures[];
[[vk::binding(1, 0)]] public RWTexture2D<float4> images[];
[[vk::binding(2, 0)]] public SamplerState samplers[];
