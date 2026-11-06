// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;
import ernie.io.kernels.encoding;

#include "ernie/texture/kernels/mips.h"
#include "ernie/texture/kernels/texel.h"

using namespace ernie;

uint FindJob(MipParams p, uint item) {
    uint low = 0;
    uint high = p.jobCount - 1;
    while (low < high) {
        const uint middle = (low + high + 1) / 2;
        if (p.jobs[middle].first <= item) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    return low;
}

float4 LoadTexel(MipJob job, uint x, uint y) {
    const uint index = y * job.sourceWidth + x;
    if (job.format == TexelFormat::Rgba16f) {
        const uint low = job.texels[job.source + 2 * index];
        const uint high = job.texels[job.source + 2 * index + 1];
        return float4(HalfToFloat(low & 0xFFFF), HalfToFloat(low >> 16), HalfToFloat(high & 0xFFFF),
                      HalfToFloat(high >> 16));
    }
    const uint word = job.texels[job.source + index];
    float4 texel = float4(Unorm8(word, 0), Unorm8(word, 8), Unorm8(word, 16), Unorm8(word, 24));
    if (job.colorSpace == TexelColorSpace::Srgb) {
        texel.xyz = float3(DecodeSrgb(texel.x), DecodeSrgb(texel.y), DecodeSrgb(texel.z));
    }
    return texel;
}

void StoreTexel(MipJob job, uint x, uint y, float4 texel) {
    const uint index = y * job.width + x;
    if (job.format == TexelFormat::Rgba16f) {
        job.texels[job.target + 2 * index] = FloatToHalf(texel.x) | FloatToHalf(texel.y) << 16;
        job.texels[job.target + 2 * index + 1] = FloatToHalf(texel.z) | FloatToHalf(texel.w) << 16;
        return;
    }
    float3 color = texel.xyz;
    if (job.colorSpace == TexelColorSpace::Srgb) {
        color = float3(EncodeSrgb(color.x), EncodeSrgb(color.y), EncodeSrgb(color.z));
    }
    job.texels[job.target + index] = EncodeUnorm8(color.x) | EncodeUnorm8(color.y) << 8 |
                                     EncodeUnorm8(color.z) << 16 | EncodeUnorm8(texel.w) << 24;
}

[shader("compute")]
[numthreads(kMipGroup, 1, 1)]
void DownsampleLevels(uniform MipParams p) {
    const uint item = LinearThreadIndex();
    if (item >= p.total) {
        return;
    }
    const MipJob job = p.jobs[FindJob(p, item)];
    const uint local = item - job.first;
    const uint x = local % job.width;
    const uint y = local / job.width;
    float4 sum = float4(0);
    for (uint dy = 0; dy < 2; ++dy) {
        for (uint dx = 0; dx < 2; ++dx) {
            sum += LoadTexel(job, min(2 * x + dx, job.sourceWidth - 1),
                             min(2 * y + dy, job.sourceHeight - 1));
        }
    }
    StoreTexel(job, x, y, sum / 4);
}
