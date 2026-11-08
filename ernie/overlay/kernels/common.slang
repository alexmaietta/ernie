// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.device.kernels.device;
__exported import ernie.io.kernels.encoding;
__exported import ernie.render.kernels.sensors.camera.camera;

#include "ernie/overlay/kernels/overlay.h"

using namespace ernie;

static const float kDepthSlack = 2e-3f;
static const float kXRayOpacity = 0.3f;
static const float kFeather = 1.0f;
static const float kTau = 6.28318530718f;

float4 DecodeOverlayColor(uint packed) {
    return float4(DecodeSrgb(float(packed & 0xFF) / 255), DecodeSrgb(float((packed >> 8) & 0xFF) / 255),
                  DecodeSrgb(float((packed >> 16) & 0xFF) / 255), float(packed >> 24) / 255);
}

uint EncodeOverlayColor(float4 color) {
    return PackOverlayColor(color.x, color.y, color.z, color.w);
}

uint FindBatch(OverlayParams p, uint item) {
    uint low = 0;
    uint high = p.batchCount;
    while (high - low > 1) {
        const uint middle = (low + high) / 2;
        if (p.batches[middle].firstItem <= item) {
            low = middle;
        } else {
            high = middle;
        }
    }
    return low;
}

uint FindView(OverlayParams p, uint tile) {
    uint low = 0;
    uint high = p.viewCount;
    while (high - low > 1) {
        const uint middle = (low + high) / 2;
        if (p.views[middle].firstTile <= tile) {
            low = middle;
        } else {
            high = middle;
        }
    }
    return low;
}

float3 ViewOrigin(RenderTables* tables, WorldTransform w, uint world, uint viewWorld) {
    return RenderTranslation(tables, w, viewWorld) + ShownOffset(tables->instances, viewWorld, world);
}

float3 RenderSpace(OverlayParams p, OverlayBatch batch, uint world, float3 point) {
    RenderTables* tables = p.tables;
    if (batch.anchor != kNoPrim) {
        const WorldTransform transform = PrimWorld(tables, batch.anchor);
        return transform.Apply(point) +
               ViewOrigin(tables, transform, tables->prims.world[batch.anchor], world);
    }
    return RenderPoint(tables, point, world);
}

float RecordRadius(OverlayRecord record) {
    return float(record.flags >> kOverlayRadiusShift) / kOverlayRadiusUnits;
}

float3 LoadPoint(Ptr<float> values, uint at) {
    return float3(values[at], values[at + 1], values[at + 2]);
}

struct ShapeRow {
    OverlayBatch batch;
    uint slice;
    uint row;

    float3 Point() {
        return LoadPoint(batch.points, slice * batch.pointWorldStride + row * batch.pointStride);
    }

    float3 End() {
        return LoadPoint(batch.ends, slice * batch.endWorldStride + row * batch.endStride);
    }

    float3 Sample(uint index) {
        return LoadPoint(batch.points, row * batch.pointStride + index * batch.endStride);
    }

    float4 Rotation() {
        const uint at = slice * batch.rotationWorldStride + row * batch.rotationStride;
        const float4 loaded = float4(batch.rotations[at], batch.rotations[at + 1],
                                     batch.rotations[at + 2], batch.rotations[at + 3]);
        return (batch.flags & kOverlayHasRotations) != 0 ? loaded : float4(0, 0, 0, 1);
    }

    float Scalar() {
        const float loaded =
            batch.scalars[slice * batch.scalarWorldStride + row * batch.scalarStride];
        return (batch.flags & kOverlayHasScalars) != 0 ? loaded : batch.length;
    }

    float3 Third() {
        return LoadPoint(batch.rotations,
                         slice * batch.rotationWorldStride + row * batch.rotationStride);
    }

    uint Prim() {
        return batch.prims[row * batch.pointStride];
    }

    uint Extra(uint fallback) {
        const uint loaded = batch.extras[slice * batch.extraWorldStride + row * batch.extraStride];
        return (batch.flags & kOverlayHasExtras) != 0 ? loaded : fallback;
    }
}

bool PrimBounded(RenderTables* tables, uint prim) {
    return PrimAlive(tables, prim) && SchemaIsA(tables->prims.schema[prim], SchemaId::Gprim);
}

float2 Pixel(float2 film, OverlayView view) {
    return float2((film.x + 1) * 0.5f * float(view.width), (1 - film.y) * 0.5f * float(view.height));
}

uint ReadByte(uint64_t address, uint index) {
    const uint64_t at = address + index;
    const Ptr<uint> word = Ptr<uint>(at & ~uint64_t(3));
    return (word[0] >> ((uint(at) & 3u) * 8u)) & 0xFFu;
}

float Cross2(float2 a, float2 b) {
    return a.x * b.y - a.y * b.x;
}

struct FillTriangle {
    float2 corners[3];
    float area;
    float side;

    __init(OverlayFill fill) {
        corners = {float2(fill.x[0], fill.y[0]), float2(fill.x[1], fill.y[1]),
                   float2(fill.x[2], fill.y[2])};
        area = Cross2(corners[1] - corners[0], corners[2] - corners[0]);
        side = area > 0 ? 1 : -1;
    }

    float Inside(uint edge, float2 point) {
        const float2 from = corners[edge];
        const float2 along = corners[(edge + 1) % 3] - from;
        return side * Cross2(along, point - from) / max(length(along), 1e-6f);
    }

    float3 Shares(float2 point) {
        const float first = Cross2(corners[1] - point, corners[2] - point) / area;
        const float second = Cross2(corners[2] - point, corners[0] - point) / area;
        return float3(first, second, 1 - first - second);
    }
}

float SegmentT(OverlayRecord record, float2 point) {
    const float2 a = float2(record.ax, record.ay);
    const float2 along = float2(record.bx, record.by) - a;
    const float lengthSquared = dot(along, along);
    return lengthSquared > 0 ? saturate(dot(point - a, along) / lengthSquared) : 0;
}

float SegmentDepth(OverlayRecord record, float t) {
    return (record.flags & kOverlayPerspective) != 0
               ? 1 / lerp(1 / record.depthA, 1 / record.depthB, t)
               : lerp(record.depthA, record.depthB, t);
}

void AddUnder(inout float4 under, float3 color, float alpha) {
    const float weight = (1 - under.w) * alpha;
    under = float4(under.xyz + color * weight, under.w + weight);
}

namespace ernie {

extension OverlayView {
    uint PixelIndex(uint2 pixel) {
        return (layer * height + pixel.y) * width + pixel.x;
    }

    uint2 TileCorner(uint tile) {
        const uint local = tile - firstTile;
        return uint2((local % columnCount) * kOverlayTile, (local / columnCount) * kOverlayTile);
    }
}

}

bool ResolveRow(OverlayBatch batch, uint viewWorld, uint item, out uint slice, out uint row) {
    slice = 0;
    row = item;
    if ((batch.flags & kOverlayGrouped) != 0) {
        const uint globalStart = batch.groupStarts[0];
        const uint globalLength = batch.groupStarts[1] - globalStart;
        uint at = globalStart + item;
        if (item >= globalLength) {
            const bool placed = viewWorld < batch.groups;
            const uint bucket = placed ? viewWorld + 1 : batch.groups + 1;
            const uint start = batch.groupStarts[bucket];
            const uint length = placed ? batch.groupStarts[bucket + 1] - start : 0;
            if (item - globalLength >= length) {
                return false;
            }
            at = start + item - globalLength;
        }
        row = batch.order[at];
        return true;
    }
    if (batch.worldCount > 0) {
        if (viewWorld == kGlobalWorld || viewWorld >= batch.worldCount) {
            return false;
        }
        slice = viewWorld;
    }
    if (item >= min(batch.itemCounts[slice * batch.itemCountStride], batch.capacity / batch.pieces)) {
        return false;
    }
    const uint loadedWorld = batch.worlds[item * batch.worldStride];
    const uint itemWorld = (batch.flags & kOverlayHasWorlds) != 0 ? loadedWorld : batch.world;
    return batch.worldCount > 0 || itemWorld == kGlobalWorld || itemWorld == viewWorld;
}

struct OverlayCamera {
    CameraModel model;
    float near;
    bool perspective;

    __init(RenderTables* tables, uint prim, uint product) {
        model = CameraModel.Load(tables, prim, product);
        const CameraRecord record = tables->cameras[PrimRow(tables, prim, kSlotCamera)];
        near = record.clipping.x;
        perspective = Projection(record.projection) != Projection::Orthographic;
    }
}

bool Outlined(OverlayOutline outline, uint id) {
    const uint word = id / 32;
    return id != kNoPrim && word < outline.bitWords && ((outline.bits[word] >> (id % 32)) & 1) != 0;
}

uint MaskIndex(OverlayView view, uint2 pixel) {
    return pixel.y * view.width + pixel.x;
}
