// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.device.kernels.device;

#include "ernie/device/kernels/texture.h"

using namespace ernie;

enum TextureFilter : uint {
    Nearest,
    Linear,
    Stochastic,
}

struct TextureLookup : IDifferentiable {
    float2 st;
    float lod;
    no_diff TextureFilter filter;
    no_diff float3 u;
}

interface ITexture {
    [Differentiable]
    float4 Sample(TextureLookup lookup);
}
