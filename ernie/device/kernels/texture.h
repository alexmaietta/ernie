// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/kernels/shared.h"

namespace ernie {

ERNIE_ENUM(TexelFormat){
    R8,
    Rg8,
    Rgba8,
    R16,
    Rg16,
    Rgba16,
    R16f,
    Rg16f,
    Rgba16f,
    R32f,
    Rg32f,
    Rgba32f,
    Bc1,
    Bc3,
    Bc4,
    Bc5,
};

ERNIE_ENUM(TextureWrap){
    Repeat,
    Mirror,
    Clamp,
    Border,
};

ERNIE_ENUM(TexelColorSpace){
    Linear,
    Srgb,
};

static const uint kTextureMaxLevels = 16;

struct TextureDesc {
    Ptr<uint> texels;
    uint width;
    uint height;
    uint numLevels;
    TexelFormat format;
    TexelColorSpace colorSpace;
    TextureWrap wrapS;
    TextureWrap wrapT;
    uint reserved;
    uint levelOffsets[kTextureMaxLevels];
};

ERNIE_CHECK_SIZE(TextureDesc, 104);

struct TextureObjects {
    uint64_t filtered;
    uint64_t texels;
    uint width;
    uint height;
    uint numLevels;
    uint channels;
};

ERNIE_CHECK_SIZE(TextureObjects, 32);

struct UdimTileRecord {
    uint tile;
    uint texture;
};

ERNIE_CHECK_SIZE(UdimTileRecord, 8);

ERNIE_SHARED uint TextureLevelSide(uint side, uint level) {
    return max(side >> level, 1u);
}

ERNIE_SHARED uint TexelBlockSide(TexelFormat format) {
    switch (format) {
    case TexelFormat::Bc1:
    case TexelFormat::Bc3:
    case TexelFormat::Bc4:
    case TexelFormat::Bc5:
        return 4;
    default:
        return 1;
    }
}

ERNIE_SHARED uint TexelBlockBytes(TexelFormat format) {
    switch (format) {
    case TexelFormat::R8:
        return 1;
    case TexelFormat::Rg8:
    case TexelFormat::R16:
    case TexelFormat::R16f:
        return 2;
    case TexelFormat::Rgba8:
    case TexelFormat::Rg16:
    case TexelFormat::Rg16f:
    case TexelFormat::R32f:
        return 4;
    case TexelFormat::Rgba16:
    case TexelFormat::Rgba16f:
    case TexelFormat::Rg32f:
    case TexelFormat::Bc1:
    case TexelFormat::Bc4:
        return 8;
    default:
        return 16;
    }
}

ERNIE_SHARED uint TexelWords(TexelFormat format) {
    return max(TexelBlockBytes(format) / 4, 1u);
}

ERNIE_SHARED uint TextureLevelBytes(TexelFormat format, uint width, uint height, uint level) {
    const uint side = TexelBlockSide(format);
    const uint across = (TextureLevelSide(width, level) + side - 1) / side;
    const uint down = (TextureLevelSide(height, level) + side - 1) / side;
    return across * down * TexelBlockBytes(format);
}

static const uint kTextureWrapBits = 2;
static const uint kTextureSlotShift = 2 * kTextureWrapBits;

static const uint kTextureSamplers = 1u << kTextureSlotShift;

static const uint kTextureBinding = 1;
static const uint kSamplerBinding = 2;

ERNIE_SHARED uint TextureHandle(uint slot, TextureWrap wrapS, TextureWrap wrapT) {
    return (slot << kTextureSlotShift) | (uint(wrapS) << kTextureWrapBits) | uint(wrapT);
}

}
