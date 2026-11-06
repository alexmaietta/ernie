// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.texture.kernels.software;

using namespace ernie;

interface ITextureImage {
    static This FromHandle(uint64_t handle);

    uint3 Size(uint level);

    float4 Texel(uint level, int2 texel);

    float4 SampleLevel(float2 at, float lod);

    float4 SampleGrad(float2 at, float2 dx, float2 dy);
}

[Differentiable]
float2 HardwarePosition<I : ITextureImage>(no_diff I image, uint level, float2 st) {
    const uint3 size = no_diff image.Size(level);
    return float2(st.x, 1 - st.y) * float2(size.xy);
}

[Differentiable]
float4 HardwareBilinear<I : ITextureImage>(no_diff I image, uint level, float2 st) {
    const float2 at = HardwarePosition(image, level, st) - 0.5f;
    const float2 base = no_diff floor(detach(at));
    const float2 t = at - base;
    const int2 i = int2(base);
    const float4 a = no_diff image.Texel(level, i);
    const float4 b = no_diff image.Texel(level, i + int2(1, 0));
    const float4 c = no_diff image.Texel(level, i + int2(0, 1));
    const float4 e = no_diff image.Texel(level, i + int2(1, 1));
    return lerp(lerp(a, b, t.x), lerp(c, e, t.x), t.y);
}

[Differentiable]
float4 EmulatedTrilinear<I : ITextureImage>(no_diff I image, float2 st, float lod) {
    const uint3 size = no_diff image.Size(0);
    const uint levels = size.z;
    const uint fine = uint(detach(lod));
    const float between = lod - float(fine);
    const uint coarse = min(fine + 1, levels - 1);
    const float4 near = HardwareBilinear(image, fine, st);
    if (between == 0 || coarse == fine) {
        return near;
    }
    return lerp(near, HardwareBilinear(image, coarse, st), between);
}

[ForwardDerivative(HardwareTrilinearForward)]
[BackwardDerivative(HardwareTrilinearBackward)]
float4 HardwareTrilinear<I : ITextureImage>(I image, float2 st, float lod) {
    return image.SampleLevel(float2(st.x, 1 - st.y), lod);
}

DifferentialPair<float4> HardwareTrilinearForward<I : ITextureImage>(
    I image, DifferentialPair<float2> st, DifferentialPair<float> lod) {
    const DifferentialPair<float4> emulated = fwd_diff(EmulatedTrilinear<I>)(image, st, lod);
    return diffPair(HardwareTrilinear(image, st.p, lod.p), emulated.d);
}

void HardwareTrilinearBackward<I : ITextureImage>(I image, inout DifferentialPair<float2> st,
                                                  inout DifferentialPair<float> lod,
                                                  float4 dOut) {
    bwd_diff(EmulatedTrilinear<I>)(image, st, lod, dOut);
}

struct HardwareTexture<I : ITextureImage> : ITexture {
    I image;

    __init(I image) {
        this.image = image;
    }

    float4 SampleFootprint(float2 st, float2 major, float minor, float bias, uint maxAnisotropy) {
        const float extent = length(major);
        if (maxAnisotropy <= 1) {
            return SampleLinear(st, log2(max(minor, extent)) + bias);
        }
        const float scale = exp2(bias);
        const float2 across = extent > 0 ? float2(-major.y, major.x) / extent : float2(0, 1);
        const float2 dx = major * scale;
        const float2 dy = across * max(minor, extent / max(maxAnisotropy, 1u)) * scale;
        return image.SampleGrad(float2(st.x, 1 - st.y), float2(dx.x, -dx.y), float2(dy.x, -dy.y));
    }

    float4 SampleLinear(float2 st, float lod) {
        const uint3 size = image.Size(0);
        const float texels = lod + 0.5f * log2(float(size.x) * float(size.y));
        return HardwareTrilinear(image, st, clamp(texels, 0.0f, float(size.z - 1)));
    }

    [Differentiable]
    float4 Sample(TextureLookup lookup) {
        const uint3 size = no_diff image.Size(0);
        const float texels = lookup.lod + 0.5f * log2(float(size.x) * float(size.y));
        const float lod = clamp(texels, 0.0f, float(size.z - 1));
        const uint fine = uint(detach(lod));
        const float between = detach(lod) - float(fine);
        const uint coarse = min(fine + 1, size.z - 1);
        switch (lookup.filter) {
        case TextureFilter::Nearest: {
            const uint level = between < 0.5f ? fine : coarse;
            return no_diff image.Texel(level,
                                       int2(floor(HardwarePosition(image, level, lookup.st))));
        }
        case TextureFilter::Stochastic: {
            const uint level = lookup.u.x < between ? coarse : fine;
            const float2 at = HardwarePosition(image, level, lookup.st) - 0.5f;
            const float2 base = floor(at);
            const float2 toward = at - base;
            return no_diff image.Texel(level, int2(base) + int2(lookup.u.y < toward.x ? 1 : 0,
                                                                lookup.u.z < toward.y ? 1 : 0));
        }
        default:
            return HardwareTrilinear(image, lookup.st, lod);
        }
    }
}

struct HardwareUdimSet<I : ITextureImage> : ITexture {
    UdimTileRecord* tiles;
    uint numTiles;
    uint64_t* handles;

    __init(UdimTileRecord* tiles, uint numTiles, uint64_t* handles) {
        this.tiles = tiles;
        this.numTiles = numTiles;
        this.handles = handles;
    }

    bool Covers(float2 st) {
        return UdimFind(tiles, numTiles, st) != numTiles;
    }

    [Differentiable]
    float4 Sample(TextureLookup lookup) {
        const uint found = no_diff UdimFind(tiles, numTiles, detach(lookup.st));
        if (found == numTiles) {
            return float4(0);
        }
        TextureLookup inTile = lookup;
        inTile.st = lookup.st - floor(lookup.st);
        const HardwareTexture<I> texture =
            HardwareTexture<I>(I.FromHandle(handles[tiles[found].texture]));
        return texture.Sample(inTile);
    }
}

#if ERNIE_HARDWARE_TEXTURES
[[vk::binding(1, 0)]]
Texture2D Textures[];
[[vk::binding(2, 0)]]
SamplerState Samplers[kTextureSamplers];

struct BindlessImage : ITextureImage {
    uint handle;

    static BindlessImage FromHandle(uint64_t handle) {
        BindlessImage image;
        image.handle = uint(handle);
        return image;
    }

    Texture2D Image() {
        return Textures[NonUniformResourceIndex(handle >> kTextureSlotShift)];
    }

    SamplerState Sampler() {
        return Samplers[NonUniformResourceIndex(handle & (kTextureSamplers - 1))];
    }

    uint3 Size(uint level) {
        uint width;
        uint height;
        uint levels;
        Image().GetDimensions(level, width, height, levels);
        return uint3(width, height, levels);
    }

    float4 Texel(uint level, int2 texel) {
        const uint kWrapMask = (1u << kTextureWrapBits) - 1;
        const int2 size = int2(Size(level).xy);
        const int x =
            WrapTexel(texel.x, size.x, TextureWrap((handle >> kTextureWrapBits) & kWrapMask));
        const int y = WrapTexel(texel.y, size.y, TextureWrap(handle & kWrapMask));
        if (x < 0 || y < 0) {
            return float4(0);
        }
        return Image().Load(int3(x, y, int(level)));
    }

    float4 SampleLevel(float2 at, float lod) {
        return Image().SampleLevel(Sampler(), at, lod);
    }

    float4 SampleGrad(float2 at, float2 dx, float2 dy) {
        return Image().SampleGrad(Sampler(), at, dx, dy);
    }
}
#endif

#if ERNIE_OBJECT_TEXTURES
float4 TextureObjectLod(uint64_t object, float2 at, float lod) {
    __target_switch {
    case cuda:
#if ERNIE_HIP
        __intrinsic_asm "tex2DLod<float4>((hipTextureObject_t)($0), ($1).x, ($1).y, $2)";
#else
        __intrinsic_asm "tex2DLod<float4>((cudaTextureObject_t)($0), ($1).x, ($1).y, $2)";
#endif
    }
}

float4 TextureObjectGrad(uint64_t object, float2 at, float2 dx, float2 dy) {
    __target_switch {
    case cuda:
#if ERNIE_HIP
        __intrinsic_asm "tex2DGrad<float4>((hipTextureObject_t)($0), ($1).x, ($1).y, $2, $3)";
#else
        __intrinsic_asm "tex2DGrad<float4>((cudaTextureObject_t)($0), ($1).x, ($1).y, $2, $3)";
#endif
    }
}

extension ernie::TextureObjects : ITextureImage {
    static TextureObjects FromHandle(uint64_t handle) {
        return *Ptr<TextureObjects>(handle);
    }

    float4 Channels(float4 value) {
        switch (channels) {
        case 1:
            return float4(value.x, 0, 0, 1);
        case 2:
            return float4(value.xy, 0, 1);
        default:
            return value;
        }
    }

    uint3 Size(uint level) {
        return uint3(TextureLevelSide(width, level), TextureLevelSide(height, level), numLevels);
    }

    float4 Texel(uint level, int2 texel) {
        const float2 at = (float2(texel) + 0.5f) / float2(Size(level).xy);
        return Channels(TextureObjectLod(texels, at, float(level)));
    }

    float4 SampleLevel(float2 at, float lod) {
        return Channels(TextureObjectLod(filtered, at, lod));
    }

    float4 SampleGrad(float2 at, float2 dx, float2 dy) {
        return Channels(TextureObjectGrad(filtered, at, dx, dy));
    }
}
#endif
