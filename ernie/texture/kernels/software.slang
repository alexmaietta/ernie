// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.texture.kernels.itexture;
import ernie.io.kernels.encoding;

#include "ernie/texture/kernels/texel.h"

using namespace ernie;

static const float kBaseLevel = -64;

int WrapTexel(int texel, int side, TextureWrap wrap) {
    switch (wrap) {
    case TextureWrap::Repeat:
        return ((texel % side) + side) % side;
    case TextureWrap::Mirror: {
        const int period = 2 * side;
        const int at = ((texel % period) + period) % period;
        return at < side ? at : period - 1 - at;
    }
    case TextureWrap::Clamp:
        return clamp(texel, 0, side - 1);
    default:
        return texel >= 0 && texel < side ? texel : -1;
    }
}

int2 WrapPair(int texel, int side, TextureWrap wrap) {
    switch (wrap) {
    case TextureWrap::Repeat: {
        int at = texel % side;
        at = at < 0 ? at + side : at;
        return int2(at, at + 1 == side ? 0 : at + 1);
    }
    case TextureWrap::Clamp:
        return int2(clamp(texel, 0, side - 1), clamp(texel + 1, 0, side - 1));
    default:
        return int2(WrapTexel(texel, side, wrap), WrapTexel(texel + 1, side, wrap));
    }
}

float3 Unpack565(uint color) {
    return float3((color >> 11) & 0x1F, (color >> 5) & 0x3F, color & 0x1F) /
           float3(31, 63, 31);
}

float4 DecodeBc1(uint2 words, uint texel, bool fourColors) {
    const uint first = words.x & 0xFFFF;
    const uint second = words.x >> 16;
    const float3 a = Unpack565(first);
    const float3 b = Unpack565(second);
    const uint index = (words.y >> (2 * texel)) & 3;
    if (fourColors || first > second) {
        const float3 colors[4] = { a, b, (2 * a + b) / 3, (a + 2 * b) / 3 };
        return float4(colors[index], 1);
    }
    const float4 colors[4] = { float4(a, 1), float4(b, 1), float4((a + b) / 2, 1), float4(0) };
    return colors[index];
}

float DecodeBc4(uint2 words, uint texel) {
    const float first = float(words.x & 0xFF) / 255;
    const float second = float((words.x >> 8) & 0xFF) / 255;
    const uint64_t bits = (uint64_t(words.y) << 16) | uint64_t(words.x >> 16);
    const uint index = uint(bits >> (3 * texel)) & 7;
    if (index == 0) {
        return first;
    }
    if (index == 1) {
        return second;
    }
    if (first > second) {
        return (float(8 - index) * first + float(index - 1) * second) / 7;
    }
    if (index == 6) {
        return 0;
    }
    if (index == 7) {
        return 1;
    }
    return (float(6 - index) * first + float(index - 1) * second) / 5;
}

float4 DecodeBlockTexel(TextureDesc* d, uint level, int x, int y, int width) {
    const uint across = (uint(width) + 3) / 4;
    const uint byte = d.levelOffsets[level] +
                      ((uint(y) / 4) * across + uint(x) / 4) * TexelBlockBytes(d.format);
    uint* words = d.texels + byte / 4;
    const uint texel = (uint(y) % 4) * 4 + uint(x) % 4;
    switch (d.format) {
    case TexelFormat::Bc1:
        return DecodeBc1(uint2(words[0], words[1]), texel, false);
    case TexelFormat::Bc3:
        return float4(DecodeBc1(uint2(words[2], words[3]), texel, true).xyz,
                      DecodeBc4(uint2(words[0], words[1]), texel));
    case TexelFormat::Bc4:
        return float4(DecodeBc4(uint2(words[0], words[1]), texel), 0, 0, 1);
    case TexelFormat::Bc5:
        return float4(DecodeBc4(uint2(words[0], words[1]), texel),
                      DecodeBc4(uint2(words[2], words[3]), texel), 0, 1);
    default:
        return float4(0);
    }
}

struct TexelQuad {
    float4 a;
    float4 b;
    float4 c;
    float4 e;
}

struct SoftwareTexture : ITexture {
    TextureDesc* desc;

    __init(TextureDesc* desc) {
        this.desc = desc;
    }

    [Differentiable]
#if !ERNIE_VULKAN
    [noinline]
#endif
    float4 Sample(TextureLookup lookup) {
        TextureDesc* d = desc;
        const float texels = lookup.lod + 0.5f * log2(float(d.width) * float(d.height));
        const float lod = clamp(texels, 0.0f, float(d.numLevels - 1));
        const uint fine = uint(lod);
        const float between = lod - float(fine);
        const uint coarse = min(fine + 1, d.numLevels - 1);
        if (lookup.filter == TextureFilter::Linear) {
            const bool blended = between != 0 && coarse != fine;
            float4 sum = float4(0);
            [loop]
            [MaxIters(2)]
            for (uint k = 0; k < (blended ? 2 : 1); ++k) {
                const float weight = !blended ? 1 : (k == 0 ? 1 - between : between);
                sum += Bilinear(d, k == 0 ? fine : coarse, lookup.st) * weight;
            }
            return sum;
        }
        const bool nearest = lookup.filter == TextureFilter::Nearest;
        const uint level =
            nearest ? (between < 0.5f ? fine : coarse) : (lookup.u.x < between ? coarse : fine);
        const float2 at = Position(d, level, lookup.st) - (nearest ? 0.0f : 0.5f);
        const float2 base = floor(at);
        const float2 toward = at - base;
        const int2 step =
            nearest ? int2(0) : int2(lookup.u.y < toward.x ? 1 : 0, lookup.u.z < toward.y ? 1 : 0);
        return no_diff Fetch(d, level, int2(base) + step);
    }

    float4 SampleLinear(float2 st, float lod) {
        TextureLookup lookup;
        lookup.st = st;
        lookup.lod = lod;
        lookup.filter = TextureFilter::Linear;
        lookup.u = float3(0);
        return Sample(lookup);
    }

    float4 SampleFootprint(float2 st, float2 major, float minor, float bias, uint maxAnisotropy) {
        const float extent = length(major);
        const uint probes =
            minor > 0 ? clamp(uint(ceil(extent / minor)), 1u, max(maxAnisotropy, 1u)) : 1u;
        const float width = max(minor, extent / probes);
        TextureLookup lookup;
        lookup.lod = width > 0 ? log2(width) + bias : kBaseLevel;
        lookup.filter = TextureFilter::Linear;
        lookup.u = float3(0);
        float4 sum = float4(0);
        for (uint p = 0; p < probes; ++p) {
            lookup.st = st + major * ((float(p) + 0.5f) / probes - 0.5f);
            sum += Sample(lookup);
        }
        return sum / probes;
    }

    [Differentiable]
    static float2 Position(TextureDesc* d, uint level, float2 st) {
        const float2 size =
            float2(TextureLevelSide(d.width, level), TextureLevelSide(d.height, level));
        return float2(st.x, 1 - st.y) * size;
    }

    [Differentiable]
    static float4 Bilinear(TextureDesc* d, uint level, float2 st) {
        const float2 at = Position(d, level, st) - 0.5f;
        const float2 base = no_diff floor(detach(at));
        const float2 t = at - base;
        const int2 i = int2(base);
        const int width = int(TextureLevelSide(d.width, level));
        const int height = int(TextureLevelSide(d.height, level));
        const int2 xs = no_diff WrapPair(i.x, width, d.wrapS);
        const int2 ys = no_diff WrapPair(i.y, height, d.wrapT);
        if (d.format == TexelFormat::Rgba8) {
            const TexelQuad quad =
                no_diff GatherRgba8(d.texels + d.levelOffsets[level] / 4, xs, ys, width,
                                    d.colorSpace == TexelColorSpace::Srgb);
            return lerp(lerp(quad.a, quad.b, t.x), lerp(quad.c, quad.e, t.x), t.y);
        }
        float4 sum = float4(0);
        [loop]
        [MaxIters(4)]
        for (uint k = 0; k < 4; ++k) {
            const bool right = (k & 1) != 0;
            const bool below = (k & 2) != 0;
            const float weight = (right ? t.x : 1 - t.x) * (below ? t.y : 1 - t.y);
            sum += no_diff Read(d, level, right ? xs.y : xs.x, below ? ys.y : ys.x, width) * weight;
        }
        return sum;
    }

    static TexelQuad GatherRgba8(uint* texels, int2 xs, int2 ys, int width, bool srgb) {
        TexelQuad quad;
        const bool2 insideX = xs >= 0;
        const bool2 insideY = ys >= 0;
        const uint2 rows = uint2(max(ys, 0)) * uint(width);
        const uint2 columns = uint2(max(xs, 0));
        const uint a = texels[rows.x + columns.x];
        const uint b = texels[rows.x + columns.y];
        const uint c = texels[rows.y + columns.x];
        const uint e = texels[rows.y + columns.y];
        quad.a = insideX.x && insideY.x ? DecodeRgba8(a, srgb) : float4(0);
        quad.b = insideX.y && insideY.x ? DecodeRgba8(b, srgb) : float4(0);
        quad.c = insideX.x && insideY.y ? DecodeRgba8(c, srgb) : float4(0);
        quad.e = insideX.y && insideY.y ? DecodeRgba8(e, srgb) : float4(0);
        return quad;
    }

    static float4 Fetch(TextureDesc* d, uint level, int2 texel) {
        const int width = int(TextureLevelSide(d.width, level));
        const int height = int(TextureLevelSide(d.height, level));
        return Read(d, level, WrapTexel(texel.x, width, d.wrapS),
                     WrapTexel(texel.y, height, d.wrapT), width);
    }

    static float4 Read(TextureDesc* d, uint level, int x, int y, int width) {
        if (x < 0 || y < 0) {
            return float4(0);
        }
        if (TexelBlockSide(d.format) == 4) {
            const float4 decoded = DecodeBlockTexel(d, level, x, y, width);
            Texel texel = { { decoded.x, decoded.y, decoded.z, decoded.w } };
            texel = TexelToLinear(texel, d.colorSpace);
            return float4(texel.value[0], texel.value[1], texel.value[2], texel.value[3]);
        }
        const uint byte = d.levelOffsets[level] +
                          (uint(y) * uint(width) + uint(x)) * TexelBlockBytes(d.format);
        uint* at = d.texels + byte / 4;
        const uint count = TexelWords(d.format);
        uint words[4] = { at[0], count > 1 ? at[1] : 0, count > 2 ? at[2] : 0,
                          count > 3 ? at[3] : 0 };
        const Texel stored = DecodeTexel(d.format, words, (byte % 4) * 8);
        const Texel texel = TexelToLinear(stored, d.colorSpace);
        return float4(texel.value[0], texel.value[1], texel.value[2], texel.value[3]);
    }

    static float4 DecodeRgba8(uint word, bool srgb) {
        const Texel texel = TexelToLinear(
            MakeTexel(Unorm8(word, 0), Unorm8(word, 8), Unorm8(word, 16), Unorm8(word, 24)),
            srgb ? TexelColorSpace::Srgb : TexelColorSpace::Linear);
        return float4(texel.value[0], texel.value[1], texel.value[2], texel.value[3]);
    }
}

uint UdimTile(float2 st) {
    const float2 cell = floor(st);
    if (cell.x < 0 || cell.x >= 10 || cell.y < 0 || cell.y >= 1000) {
        return 0;
    }
    return 1001 + uint(cell.x) + 10 * uint(cell.y);
}

uint UdimFind(UdimTileRecord* tiles, uint numTiles, float2 st) {
    const uint tile = UdimTile(st);
    uint low = 0;
    uint high = numTiles;
    while (low < high) {
        const uint middle = low + (high - low) / 2;
        if (tiles[middle].tile < tile) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return low < numTiles && tiles[low].tile == tile ? low : numTiles;
}

struct UdimSet : ITexture {
    UdimTileRecord* tiles;
    uint numTiles;
    TextureDesc* textures;

    __init(UdimTileRecord* tiles, uint numTiles, TextureDesc* textures) {
        this.tiles = tiles;
        this.numTiles = numTiles;
        this.textures = textures;
    }

    static uint Tile(float2 st) {
        return UdimTile(st);
    }

    uint Find(float2 st) {
        return UdimFind(tiles, numTiles, st);
    }

    bool Covers(float2 st) {
        return Find(st) != numTiles;
    }

    [Differentiable]
    float4 Sample(TextureLookup lookup) {
        const uint found = no_diff Find(detach(lookup.st));
        if (found == numTiles) {
            return float4(0);
        }
        TextureLookup inTile = lookup;
        inTile.st = lookup.st - floor(lookup.st);
        const SoftwareTexture texture = SoftwareTexture(textures + tiles[found].texture);
        return texture.Sample(inTile);
    }
}
