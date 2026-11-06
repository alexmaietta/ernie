// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/kernels/shared.h"

#ifndef __SLANG__
#include "ernie/device/kernels/texture.h"
#include "ernie/io/kernels/encoding.h"
#endif

namespace ernie {

struct Texel {
    float value[4];
};

ERNIE_SHARED Texel MakeTexel(float r, float g, float b, float a) {
    Texel texel = {{r, g, b, a}};
    return texel;
}

ERNIE_SHARED float Unorm8(uint word, uint shift) {
    return float((word >> shift) & 0xFF) / 255;
}

ERNIE_SHARED float Unorm16(uint word, uint shift) {
    return float((word >> shift) & 0xFFFF) / 65535;
}

ERNIE_SHARED float Half(uint word, uint shift) {
    return HalfToFloat((word >> shift) & 0xFFFF);
}

ERNIE_SHARED Texel DecodeTexel(TexelFormat format, uint words[4], uint shift) {
    const uint w = words[0];
    switch (format) {
    case TexelFormat::R8:
        return MakeTexel(Unorm8(w, shift), 0, 0, 1);
    case TexelFormat::Rg8:
        return MakeTexel(Unorm8(w, shift), Unorm8(w, shift + 8), 0, 1);
    case TexelFormat::Rgba8:
        return MakeTexel(Unorm8(w, 0), Unorm8(w, 8), Unorm8(w, 16), Unorm8(w, 24));
    case TexelFormat::R16:
        return MakeTexel(Unorm16(w, shift), 0, 0, 1);
    case TexelFormat::Rg16:
        return MakeTexel(Unorm16(w, 0), Unorm16(w, 16), 0, 1);
    case TexelFormat::Rgba16:
        return MakeTexel(Unorm16(w, 0), Unorm16(w, 16), Unorm16(words[1], 0),
                         Unorm16(words[1], 16));
    case TexelFormat::R16f:
        return MakeTexel(Half(w, shift), 0, 0, 1);
    case TexelFormat::Rg16f:
        return MakeTexel(Half(w, 0), Half(w, 16), 0, 1);
    case TexelFormat::Rgba16f:
        return MakeTexel(Half(w, 0), Half(w, 16), Half(words[1], 0), Half(words[1], 16));
    case TexelFormat::R32f:
        return MakeTexel(asfloat(w), 0, 0, 1);
    case TexelFormat::Rg32f:
        return MakeTexel(asfloat(w), asfloat(words[1]), 0, 1);
    case TexelFormat::Rgba32f:
        return MakeTexel(asfloat(w), asfloat(words[1]), asfloat(words[2]), asfloat(words[3]));
    default:
        return MakeTexel(0, 0, 0, 0);
    }
}

ERNIE_SHARED Texel TexelToLinear(Texel texel, TexelColorSpace colorSpace) {
    if (colorSpace == TexelColorSpace::Srgb) {
        for (uint channel = 0; channel < 3; ++channel) {
            texel.value[channel] = DecodeSrgb(texel.value[channel]);
        }
    }
    return texel;
}

}
