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

float3 RenderSpace(OverlayParams p, OverlayBatch batch, uint world, float3 point) {
    RenderTables* tables = p.tables;
    InstanceLayout* layout = p.tables->instances;
    if (batch.anchor != kNoPrim) {
        const WorldTransform transform = PrimWorld(p.tables, batch.anchor);
        return transform.Apply(point) + RenderTranslation(p.tables, transform, world) +
               ShownOffset(layout, world, tables->prims.world[batch.anchor]);
    }
    return RenderPoint(p.tables, point, world);
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

    __init(RenderTables* tables, uint prim) {
        model = CameraModel.Load(tables, prim);
        const CameraRecord record = tables->cameras[PrimRow(tables, prim, kSlotCamera)];
        near = record.clipping.x;
        perspective = Projection(record.projection) != Projection::Orthographic;
    }
}
