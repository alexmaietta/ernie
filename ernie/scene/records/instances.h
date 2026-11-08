// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/kernels/ids.h"
#include "ernie/scene/schema/schema.h"

namespace ernie {

struct SurfaceRecord {
    uint64_t points;
    uint64_t indices;
    uint64_t corners;
    uint64_t faces;
    uint64_t normals;
    uint64_t smooth;
    uint normalsInterpolation;
    uint normalsCount;
    uint meshRow;
    uint padding0;
    ArrayRef primvars;
    ArrayRef instancerPrimvars;
    Float3 displayColor;
    uint padding1;
    int64_t instanceId;
};

ERNIE_CHECK_SIZE(SurfaceRecord, 120);

static const float kRenderOriginCell = 64;

struct RenderOrigin {
    Float3 high;
    Float3 low;
};

ERNIE_CHECK_SIZE(RenderOrigin, 24);

static const float kInfinitePlane = 1e6f;

ERNIE_ENUM(ShapeKind){
    Mesh,
    Sphere,
    Box,
    Plane,
    Cylinder,
    Cone,
    Capsule,
    Disk,
    Tube,
};

static const uint kGeometryMask = 1u;
static const uint kCameraEmitterMask = 2u;
static const uint kEmitterMask = 4u;

struct ShapeRecord {
    uint kind;
    float param;
};

ERNIE_CHECK_SIZE(ShapeRecord, 8);

ERNIE_SHARED bool IsEmitter(uint schema) {
    switch (SchemaId(schema)) {
    case SchemaId::SphereLight:
    case SchemaId::RectLight:
    case SchemaId::DiskLight:
    case SchemaId::CylinderLight:
        return true;
    default:
        return false;
    }
}

static const uint kInstancerSlot = 0x80000000u;
static const uint kNoNested = 0xFFFFFFFFu;
static const uint kMostNesting = 4;

struct InstancerBlock {
    uint instancer;
    uint first;
    uint count;
    uint width;
    uint prototypes;
    uint prototypeCount;
    uint native;
    uint hidden;
    uint lightWidth;
    uint padding0;
    uint padding1;
    uint padding2;
};

ERNIE_CHECK_SIZE(InstancerBlock, 48);

struct PrototypeEntry {
    uint first;
    uint count;
    uint parent;
    uint width;
    uint lights;
    uint padding0;
    uint padding1;
    uint padding2;
};

ERNIE_CHECK_SIZE(PrototypeEntry, 32);

struct PrototypePiece {
    uint prim;
    uint start;
    uint slots;
    uint nested;
    uint lightStart;
    uint lights;
    uint padding0;
    uint padding1;
};

ERNIE_CHECK_SIZE(PrototypePiece, 32);

struct SlotHit {
    uint prim;
    uint instancer;
    uint instance;
    uint offset;
};

ERNIE_CHECK_SIZE(SlotHit, 16);

struct InstanceLayout {
    Ptr<uint64_t> handles;
    Ptr<uint> firstInstances;
    Ptr<uint> instanceCounts;
    Ptr<uint> instancePrims;
    Ptr<ShapeRecord> shapes;
    Ptr<uint> worldOrigins;
    Ptr<SlotHit> hits;
    Ptr<InstancerBlock> blocks;
    Ptr<PrototypeEntry> prototypes;
    Ptr<PrototypePiece> pieces;
    Ptr<uint> hiddenRanges;
    uint64_t hiddenIds;
    uint64_t records;
    Ptr<uint> offsetBases;
    Ptr<Float3> offsets;
    uint worldCount;
    uint displayColorName;
};

ERNIE_CHECK_SIZE(InstanceLayout, 128);

#ifdef __SLANG__

SlotHit Drawn(uint prim, uint instancer, uint instance, uint offset) {
    SlotHit hit;
    hit.prim = prim;
    hit.instancer = instancer;
    hit.instance = instance;
    hit.offset = offset;
    return hit;
}

float3 ShownOffset(InstanceLayout* layout, uint view, uint world) {
    if (view >= layout->worldCount || world >= layout->worldCount) {
        return float3(0);
    }
    const uint base = layout->offsetBases[view];
    if (base == kNoRow) {
        return float3(0);
    }
    const float3 offset = layout->offsets[base + world].Vector();
    return isnan(offset.x) ? float3(0) : offset;
}

bool WorldShown(InstanceLayout* layout, uint view, uint world) {
    if (world == view || world == kGlobalWorld) {
        return true;
    }
    if (view >= layout->worldCount || world >= layout->worldCount) {
        return false;
    }
    const uint base = layout->offsetBases[view];
    return base != kNoRow && !isnan(layout->offsets[base + world].x);
}

float3 RelativeTranslation(RenderOrigin origin, WorldTransform w) {
    return (w.high.Vector() - origin.high.Vector()) + (w.low.Vector() - origin.low.Vector());
}

RenderOrigin SnapOrigin(WorldTransform w) {
    const float3 high = round(w.high.Vector() / kRenderOriginCell) * kRenderOriginCell;
    const float3 rest = (w.high.Vector() - high) + w.low.Vector();
    RenderOrigin origin;
    origin.high = Float3(high);
    origin.low = Float3(round(rest / kRenderOriginCell) * kRenderOriginCell);
    return origin;
}

#endif

}
