// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/kernels/scene.h"
#include "ernie/scene/kernels/shapes.h"

namespace ernie {

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

ERNIE_SHARED uint InstanceTlas(uint world, uint worldCount) {
    return world == kGlobalWorld ? worldCount : world;
}

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

float3 RotateByQuat(float4 q, float3 v) {
    const float3 t = 2 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

float3 SolveAxes(float3 a, float3 b, float3 c, float3 v) {
    const float3 bc = cross(b, c);
    const float det = dot(a, bc);
    if (abs(det) < 1e-30f) {
        return float3(0);
    }
    return float3(dot(bc, v), dot(cross(c, a), v), dot(cross(a, b), v)) / det;
}

struct InstancePlacement {
    bool valid;
    uint chosen;
    float3 position;
    float4 orientation;
    float3 scale;

}

InstancePlacement
PlaceCopy(SceneData* scene, InstancerBlock block, uint instance) {

    InstancePlacement placement;
    placement.valid = true;
    placement.chosen = 0;
    placement.position = float3(0);
    placement.orientation = float4(0, 0, 0, 1);
    placement.scale = float3(1);
    if (block.native != 0) {
        return placement;
    }
    PointInstancerData* d = &scene->schemas.pointInstancer;
    const uint row = PrimRow(scene, block.instancer, kSlotPointInstancer);
    const ArrayRef indices = d.protoIndices[row];
    const int index = instance < indices.count ? Ptr<int>(indices.address)[instance] : -1;
    placement.valid = index >= 0;
    placement.chosen = uint(max(index, 0));
    placement.position = Ptr<Float3>(d.positions[row].address)[instance].Vector();
    const ArrayRef orientations = d.orientations[row];
    if (instance < orientations.count) {
        const float4 q = Ptr<Float4>(orientations.address)[instance].Vector();
        const float size = length(q);
        placement.orientation = size > 0 ? q / size : float4(0, 0, 0, 1);
    }
    const ArrayRef scales = d.scales[row];
    if (instance < scales.count) {
        placement.scale = Ptr<Float3>(scales.address)[instance].Vector();
    }
    return placement;
}

WorldTransform CopyWorld(SceneData* scene, Ptr<WorldTransform> transforms, InstancerBlock block,
                         InstancePlacement placement, WorldTransform g, uint parent) {
    float3 axes[3] = {g.x.Vector(), g.y.Vector(), g.z.Vector()};
    float3 offset = g.Translation();
    if (parent != kNoPrim) {
        const WorldTransform p = PrimWorld(scene, transforms, parent);
        const float3 a = p.x.Vector();
        const float3 b = p.y.Vector();
        const float3 c = p.z.Vector();
        for (uint axis = 0; axis < 3; ++axis) {
            axes[axis] = SolveAxes(a, b, c, axes[axis]);
        }
        offset = SolveAxes(a, b, c,
                           (g.high.Vector() - p.high.Vector()) + (g.low.Vector() - p.low.Vector()));
    }
    for (uint axis = 0; axis < 3; ++axis) {
        axes[axis] = RotateByQuat(placement.orientation, placement.scale * axes[axis]);
    }
    const float3 moved =
        placement.position + RotateByQuat(placement.orientation, placement.scale * offset);
    const WorldTransform outer = PrimWorld(scene, transforms, block.instancer);
    WorldTransform w;
    w.x = Float3(outer.Apply(axes[0]));
    w.y = Float3(outer.Apply(axes[1]));
    w.z = Float3(outer.Apply(axes[2]));
    w.high = outer.high;
    w.low = Float3(outer.low.Vector() + outer.Apply(moved));
    return w;
}

bool HiddenCopy(InstanceLayout* layout, InstancerBlock block, int64_t id) {
    const uint first = layout->hiddenRanges[2 * block.hidden];
    uint low = 0;
    uint high = layout->hiddenRanges[2 * block.hidden + 1];
    Ptr<int64_t> ids = Ptr<int64_t>(layout->hiddenIds) + first;
    while (low < high) {
        const uint middle = (low + high) / 2;
        const int64_t value = ids[middle];
        if (value == id) {
            return true;
        }
        if (value < id) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return false;
}

bool ShownPrim(SceneData* scene, Ptr<ResolvedImageable> imageables, uint prim) {
    return ImageableShown(PrimImageable(scene, imageables, prim));
}

struct ResolvedCopy {
    bool valid;
    bool shown;
    uint prim;
    uint instance;
    uint offset;
    WorldTransform world;

}

ResolvedCopy
ResolveCopy(SceneData* scene, Ptr<WorldTransform> transforms, Ptr<ResolvedImageable> imageables,
            InstanceLayout* layout, uint top, uint offset, bool lights) {

    ResolvedCopy resolved;
    resolved.valid = false;
    resolved.shown = true;
    resolved.prim = kNoPrim;
    resolved.instance = 0;
    resolved.offset = 0;
    InstancerBlock blocks[kMostNesting];
    InstancePlacement placements[kMostNesting];
    uint parents[kMostNesting];
    uint depth = 0;
    uint index = top;
    uint at = offset;
    while (true) {
        const InstancerBlock block = layout->blocks[index];
        const uint width = lights ? block.lightWidth : block.width;
        if (width == 0) {
            return resolved;
        }
        const uint instance = at / width;
        const uint member = at % width;
        if (depth == 0) {
            resolved.instance = instance;
        }
        if (instance >= block.count) {
            return resolved;
        }
        if (block.native == 0) {
            const ArrayRef ids = scene->schemas.pointInstancer
                                     .ids[PrimRow(scene, block.instancer, kSlotPointInstancer)];
            const int64_t id =
                instance < ids.count ? Ptr<int64_t>(ids.address)[instance] : int64_t(instance);
            if (HiddenCopy(layout, block, id)) {
                return resolved;
            }
        }
        const InstancePlacement placement = PlaceCopy(scene, block, instance);
        if (!placement.valid || placement.chosen >= block.prototypeCount) {
            return resolved;
        }
        const PrototypeEntry prototype = layout->prototypes[block.prototypes + placement.chosen];
        if (member >= (lights ? prototype.lights : prototype.width)) {
            return resolved;
        }
        PrototypePiece piece = layout->pieces[prototype.first];
        for (uint i = 0; i < prototype.count; ++i) {
            const PrototypePiece candidate = layout->pieces[prototype.first + i];
            const uint start = lights ? candidate.lightStart : candidate.start;
            const uint size = lights ? candidate.lights : candidate.slots;
            if (member >= start && member < start + size) {
                piece = candidate;
                break;
            }
        }
        resolved.offset += instance * block.width + piece.start;
        resolved.shown = resolved.shown && ShownPrim(scene, imageables, block.instancer);
        blocks[depth] = block;
        placements[depth] = placement;
        parents[depth] = prototype.parent;
        ++depth;
        if (piece.nested == kNoNested) {
            resolved.prim = piece.prim;
            break;
        }
        if (depth >= kMostNesting) {
            return resolved;
        }
        index = piece.nested;
        at = member - (lights ? piece.lightStart : piece.start);
    }
    WorldTransform w = PrimWorld(scene, transforms, resolved.prim);
    for (uint level = depth; level-- > 0;) {
        w = CopyWorld(scene, transforms, blocks[level], placements[level], w, parents[level]);
    }
    resolved.world = w;
    resolved.valid = true;
    return resolved;
}

WorldTransform SlotWorld(SceneData* scene, Ptr<WorldTransform> transforms,
                         Ptr<ResolvedImageable> imageables, InstanceLayout* layout, uint slot) {
    const SlotHit drawn = layout->hits[slot];
    const uint item = layout->instancePrims[slot];
    if ((item & kInstancerSlot) == 0) {
        return PrimWorld(scene, transforms, drawn.prim);
    }
    return ResolveCopy(scene, transforms, imageables, layout, item & ~kInstancerSlot, drawn.offset,
                       false)
        .world;
}

#endif

}
