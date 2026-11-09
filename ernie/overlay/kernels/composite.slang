// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.overlay.kernels.common;

#include "ernie/overlay/kernels/composite.h"

using namespace ernie;

float MarkerDistance(uint shape, float2 offset, float radius) {
    const float2 folded = abs(offset);
    const float stroke = max(1.5f, 0.25f * radius);
    switch (shape) {
    case uint(MarkerShape::Ring):
        return 0.5f * stroke - abs(length(offset) - (radius - 0.5f * stroke));
    case uint(MarkerShape::Square):
        return radius - max(folded.x, folded.y);
    case uint(MarkerShape::Diamond):
        return (radius - (folded.x + folded.y)) * 0.7071f;
    case uint(MarkerShape::Cross):
        return min(0.5f * stroke - abs(folded.x - folded.y) * 0.7071f,
                   radius * 0.7071f - max(folded.x, folded.y));
    case uint(MarkerShape::Plus):
        return min(0.5f * stroke - min(folded.x, folded.y), radius - max(folded.x, folded.y));
    default:
        return radius - length(offset);
    }
}

float GlyphCoverage(OverlayRecord record, float2 point, uint64_t font) {
    const float spread = (record.flags & kOverlayHalo) != 0
                             ? float((record.flags >> kOverlayHaloShift) & kOverlayHaloMask) / 4
                             : 0;
    const float2 a = float2(record.ax, record.ay);
    const float2 b = float2(record.bx, record.by);
    if (any(point < a - 1) || any(point > b + 1)) {
        return 0;
    }
    const uint width = (record.flags >> kOverlayFieldShift) & 0xFFu;
    const uint height = record.flags >> 24;
    const uint first = asuint(record.depthB);
    const float2 texel = (point - a) / (b - a) * float2(width, height) - 0.5f;
    const int2 low = int2(floor(texel));
    const float2 blend = texel - float2(low);
    float corners[4];
    for (uint corner = 0; corner < 4; ++corner) {
        const int x = clamp(low.x + int(corner & 1), 0, int(width) - 1);
        const int y = clamp(low.y + int(corner >> 1), 0, int(height) - 1);
        corners[corner] = float(ReadByte(font, first + uint(y) * width + uint(x)));
    }
    const float value = lerp(lerp(corners[0], corners[1], blend.x),
                             lerp(corners[2], corners[3], blend.x), blend.y);
    const float distance = (value - kOverlayFieldEdge) / kOverlayFieldLevels;
    return saturate(distance * (b.x - a.x) / float(width) + spread + 0.5f);
}

float4 SampleImage(OverlayFill fill, float2 uv) {
    const float2 texel = saturate(uv) * float2(fill.width, fill.height) - 0.5f;
    const int2 low = int2(floor(texel));
    const float2 blend = texel - float2(low);
    const Ptr<uint> pixels = Ptr<uint>(fill.pixels);
    float4 corners[4];
    for (uint corner = 0; corner < 4; ++corner) {
        const int x = clamp(low.x + int(corner & 1), 0, int(fill.width) - 1);
        const int y = clamp(low.y + int(corner >> 1), 0, int(fill.height) - 1);
        corners[corner] = DecodeOverlayColor(pixels[uint(y) * fill.rowStride + uint(x)]);
    }
    float4 sampled =
        lerp(lerp(corners[0], corners[1], blend.x), lerp(corners[2], corners[3], blend.x), blend.y);
    if (fill.opaque != 0) {
        sampled.w = 1;
    }
    return sampled;
}

float TriangleCoverage(OverlayRecord record, OverlayFill fill, float2 point, inout float4 color,
                       out float depth) {
    depth = 0;
    const FillTriangle triangle = FillTriangle(fill);
    const uint edges = (record.flags >> kOverlayEdgeShift) & kOverlayEdgeMask;
    float coverage = 1;
    for (uint edge = 0; edge < 3; ++edge) {
        const float inside = triangle.Inside(edge, point);
        if (((edges >> edge) & 1) != 0) {
            coverage = min(coverage, saturate(inside + 0.5f));
        } else if (inside < 0) {
            coverage = 0;
        }
    }
    if (coverage <= 0) {
        return 0;
    }
    const float3 shares = triangle.Shares(point);
    const bool perspective = (record.flags & kOverlayPerspective) != 0;
    const float3 depths = float3(fill.depth[0], fill.depth[1], fill.depth[2]);
    const float3 weights = perspective ? shares / depths : shares;
    const float total = weights.x + weights.y + weights.z;
    depth = perspective ? 1 / total : dot(shares, depths);
    if ((record.flags & kOverlayTextured) != 0) {
        const uint uvs = record.flags >> kOverlayUvShift;
        float2 uv = float2(0);
        for (uint corner = 0; corner < 3; ++corner) {
            const float2 cornerUv = float2((uvs >> (2 * corner)) & 1, (uvs >> (2 * corner + 1)) & 1);
            uv += weights[corner] * cornerUv;
        }
        color *= SampleImage(fill, uv / total);
    }
    return coverage;
}

float4 Shade(OverlayRecord record, float4 color, float2 point, float sceneDepth,
             float missDistance, uint64_t font, Ptr<OverlayFill> fills) {
    const float2 a = float2(record.ax, record.ay);
    const float2 b = float2(record.bx, record.by);
    const uint kind = (record.flags >> kOverlayKindShift) & kOverlayKindMask;
    float t = 0;
    float coverage;
    float fillDepth = 0;
    if (kind == uint(OverlayPiece::Triangle)) {
        coverage = TriangleCoverage(record, fills[asuint(record.depthA)], point, color, fillDepth);
    } else if (kind == uint(OverlayPiece::Glyph)) {
        coverage = GlyphCoverage(record, point, font);
    } else if (kind == uint(OverlayPiece::Box)) {
        const float2 inside = min(point - a, b - point);
        coverage = saturate(min(inside.x, inside.y) + 0.5f);
    } else if (kind == uint(OverlayPiece::Segment)) {
        t = SegmentT(record, point);
        coverage = saturate(RecordRadius(record) + 0.5f - length(point - lerp(a, b, t)));
        const uint dash = (record.flags >> kOverlayDashShift) & kOverlayPatternMask;
        if (dash > 0 && coverage > 0) {
            const float gap = float(max((record.flags >> kOverlayGapShift) & kOverlayPatternMask, 1u));
            const float period = float(dash) + gap;
            const float phase = fmod(t * length(b - a), period);
            const float inside = phase < float(dash) ? min(phase, float(dash) - phase)
                                                     : -min(phase - float(dash), period - phase);
            coverage *= saturate(inside + 0.5f);
        }
    } else {
        const uint shape = (record.flags >> kOverlayShapeShift) & kOverlayShapeMask;
        coverage = saturate(MarkerDistance(shape, point - a, RecordRadius(record)) + 0.5f);
    }
    if (coverage <= 0) {
        return float4(0);
    }
    const bool flat = kind == uint(OverlayPiece::Glyph) || kind == uint(OverlayPiece::Box);
    const float depth = kind == uint(OverlayPiece::Triangle) ? fillDepth
                        : flat ? record.depthA
                        : SegmentDepth(record, t);
    const uint mode = record.flags & kOverlayDepthMask;
    if (mode != uint(OverlayDepth::Always) && sceneDepth != missDistance &&
        depth > sceneDepth * (1 + kDepthSlack)) {
        if (mode != uint(OverlayDepth::XRay)) {
            return float4(0);
        }
        coverage *= kXRayOpacity;
    }
    return float4(color.xyz, color.w * coverage);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void FoldOverlayTiles(uniform OverlayParams p) {
    const uint tiles = p.occupied[0];
    const uint across = min(tiles, kOverlayMostGroups);
    p.occupied[1] = across;
    p.occupied[2] = across == 0 ? 0 : (tiles + across - 1) / across;
    p.occupied[3] = 1;
}

static const float kOpaque = 254.5f / 255;

groupshared OverlayRecord s_records[kOverlayTilePixels];
groupshared float4 s_colors[kOverlayTilePixels];
groupshared float4 s_bounds[kOverlayTilePixels];
groupshared uint s_open;

struct CompositePixel {
    uint index;
    float2 point;
    float sceneDepth;
    bool inside;
    float4 under;
    bool done;

    __init(OverlayView view, uint2 pixel) {
        inside = pixel.x < view.width && pixel.y < view.height;
        const uint2 clamped = min(pixel, uint2(view.width, view.height) - 1);
        index = view.PixelIndex(clamped);
        point = float2(pixel) + 0.5f;
        sceneDepth = view.depth[index];
        under = float4(0);
        done = !inside;
    }

    [mutating]
    void Add(OverlayRecord record, float4 color, float missDistance, uint64_t font,
             Ptr<OverlayFill> fills) {
        if (done) {
            return;
        }
        const float4 shaded = Shade(record, color, point, sceneDepth, missDistance, font, fills);
        AddUnder(under, shaded.xyz, shaded.w);
        done = under.w >= kOpaque;
    }

    void Write(OverlayView view) {
        if (inside && under.w > 0) {
            view.overlay[index] = EncodeOverlayColor(float4(under.xyz / under.w, under.w));
        }
    }
}

[shader("compute")]
[numthreads(kOverlayTile, kOverlayTile, 1)]
void CompositeOverlays(uniform OverlayParams p) {
    const uint listed = LinearGroupIndex();
    if (listed >= p.occupied[0]) {
        return;
    }
    const uint at = kOverlayTileList + kOverlayTileWords * listed;
    const uint tile = p.occupied[at];
    const uint start = p.occupied[at + 2];
    const uint end = p.occupied[at + 3];
    const uint thread = LocalThreadIndex();
    const OverlayView view = p.views[p.occupied[at + 1]];
    CompositePixel pixel = CompositePixel(
        view, view.TileCorner(tile) + uint2(thread % kOverlayTile, thread / kOverlayTile));

    uint chunkEnd = end;
    while (chunkEnd > start) {
        const uint chunkStart = chunkEnd - min(kOverlayTilePixels, chunkEnd - start);
        const uint loaded = chunkEnd - chunkStart;
        if (thread < loaded) {
            const OverlayRecord record = p.records[p.values[chunkStart + thread]];
            s_records[thread] = record;
            s_colors[thread] = DecodeOverlayColor(record.color);
            const uint kind = (record.flags >> kOverlayKindShift) & kOverlayKindMask;
            const float2 a = float2(record.ax, record.ay);
            const float2 b = float2(record.bx, record.by);
            float4 bounds = float4(a - 1, b + 1);
            if (kind == uint(OverlayPiece::Segment)) {
                const float reach = RecordRadius(record) + 1;
                bounds = float4(min(a, b) - reach, max(a, b) + reach);
            } else if (kind == uint(OverlayPiece::Marker)) {
                const float reach = RecordRadius(record) + 1;
                bounds = float4(a - reach, a + reach);
            } else if (kind == uint(OverlayPiece::Triangle)) {
                const OverlayFill fill = p.fills[asuint(record.depthA)];
                const float2 low = min(float2(fill.x[0], fill.y[0]),
                                       min(float2(fill.x[1], fill.y[1]), float2(fill.x[2], fill.y[2])));
                const float2 high = max(float2(fill.x[0], fill.y[0]),
                                        max(float2(fill.x[1], fill.y[1]), float2(fill.x[2], fill.y[2])));
                bounds = float4(low - 1, high + 1);
            }
            s_bounds[thread] = bounds;
        }
        if (thread == 0) {
            s_open = 0;
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint i = loaded; i > 0 && !pixel.done; --i) {
            const float4 bounds = s_bounds[i - 1];
            if (all(pixel.point >= bounds.xy) && all(pixel.point <= bounds.zw)) {
                pixel.Add(s_records[i - 1], s_colors[i - 1], view.missDistance, p.font, p.fills);
            }
        }
        if (chunkStart == start) {
            break;
        }
        if (!pixel.done) {
            InterlockedAdd(s_open, 1u);
        }
        GroupMemoryBarrierWithGroupSync();
        const bool open = s_open > 0;
        GroupMemoryBarrierWithGroupSync();
        if (!open) {
            break;
        }
        chunkEnd = chunkStart;
    }
    pixel.Write(view);
}

[shader("compute")]
[numthreads(kOverlayGroup, 1, 1)]
void BlendOverlay(uniform BlendParams p) {
    const uint pixel = LinearThreadIndex();
    if (pixel >= p.count) {
        return;
    }
    const uint covering = p.overlay[pixel];
    const uint color = p.color[pixel];
    if ((covering >> 24) == 0) {
        p.blended[pixel] = color;
        return;
    }
    const float4 over = DecodeOverlayColor(covering);
    const float4 under = DecodeOverlayColor(color);
    p.blended[pixel] = EncodeOverlayColor(float4(over.xyz * over.w + under.xyz * (1 - over.w),
                                                 over.w + under.w * (1 - over.w)));
}
