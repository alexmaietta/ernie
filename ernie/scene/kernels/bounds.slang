// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;

#include "ernie/scene/kernels/bounds.h"
#include "ernie/scene/kernels/shapes.h"

using namespace ernie;

bool Shown(SceneData* scene, uint imageRow) {
    const uint purpose = scene->schemas.imageable.computedPurpose[imageRow];
    return scene->schemas.imageable.computedVisibility[imageRow] != uint(Visibility::Invisible) &&
           purpose != uint(Purpose::Proxy) && purpose != uint(Purpose::Guide);
}

float3 OriginOffset(BoundsParams p, WorldTransform w, uint world) {
    if (world == kGlobalWorld || p.origins[world] == kNoRow) {
        return w.Translation();
    }
    return w.TranslationFrom(p.scene->schemas.xformable.worldTransform[p.origins[world]]);
}

[shader("compute")]
[numthreads(kBoundsGroup, 1, 1)]
void PrimBounds(uniform BoundsParams p) {
    const uint prim = LinearThreadIndex();
    SceneData* scene = p.scene;
    if (prim >= p.primCount || (scene->prims.flags[prim] & kPrimAlive) == 0) {
        return;
    }
    const uint schemaId = scene->prims.schema[prim];
    if (!SchemaIsA(schemaId, SchemaId::Gprim)) {
        return;
    }
    const SchemaId schema = SchemaId(schemaId);
    const WorldTransform w = PrimWorld(scene, prim);
    const UnitShape shape = PlaceShape(scene, prim, w);
    float3 center = float3(0);
    float3 half = float3(1);
    bool bounded = ShapeKind(shape.record.kind) != ShapeKind::Mesh;
    if (schema == SchemaId::Mesh) {
        const ArrayRef extent = scene->schemas.mesh.pointBounds[PrimRow(scene, prim, kSlotMesh)];
        bounded = extent.count >= 2;
        if (bounded) {
            const float3 lower = Ptr<Float3>(extent.address)[0].Vector();
            const float3 upper = Ptr<Float3>(extent.address)[1].Vector();
            center = (lower + upper) / 2;
            half = (upper - lower) / 2;
        }
    }
    const uint boundRow = PrimRow(scene, prim, kSlotBoundable);
    if (!bounded) {
        scene->schemas.boundable.worldMin[boundRow] = Float3(float3(0));
        scene->schemas.boundable.worldMax[boundRow] = Float3(float3(0));
        return;
    }

    const uint world = scene->prims.world[prim];
    const float3x3 axes = shape.axes;
    const float3 middle = OriginOffset(p, w, world) + center.x * axes[0] + center.y * axes[1] +
                          center.z * axes[2];
    const float3 reach = abs(axes[0]) * half.x + abs(axes[1]) * half.y + abs(axes[2]) * half.z;
    const float3 lower = middle - reach;
    const float3 upper = middle + reach;
    scene->schemas.boundable.worldMin[boundRow] = Float3(lower);
    scene->schemas.boundable.worldMax[boundRow] = Float3(upper);

}

void Grow(Ptr<uint> extents, uint lowerAt, uint upperAt, float3 lower, float3 upper) {
    for (uint axis = 0; axis < 3; ++axis) {
        InterlockedMin(extents[lowerAt + axis], OrderedFloat(lower[axis]));
        InterlockedMax(extents[upperAt + axis], OrderedFloat(upper[axis]));
    }
}

[shader("compute")]
[numthreads(kBoundsGroup, 1, 1)]
void SlotBounds(uniform BoundsParams p) {
    const uint slot = LinearThreadIndex();
    if (slot >= p.instanceSlots) {
        return;
    }
    InstanceLayout* layout = p.instances;
    uint low = 0;
    uint high = p.tlasCount - 1;
    while (low < high) {
        const uint middle = (low + high + 1) / 2;
        if (layout->firstInstances[middle] <= slot) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    const uint tlas = low;
    if (slot - layout->firstInstances[tlas] >= layout->instanceCounts[tlas]) {
        return;
    }
    const InstanceRecord record = Ptr<InstanceRecord>(layout->records)[slot];
    if (record.blas == 0 || (record.custom >> 24) != kGeometryMask) {
        return;
    }
    SceneData* scene = p.scene;
    const SlotHit drawn = layout->hits[slot];
    const uint owner = drawn.instancer != kNoPrim ? drawn.instancer : drawn.prim;
    const uint world = scene->prims.world[owner];
    const bool global = tlas == layout->worldCount;
    if ((world == kGlobalWorld) != global) {
        return;
    }
    float3 center = float3(0);
    float3 half = float3(1);
    if (SchemaId(scene->prims.schema[drawn.prim]) == SchemaId::Mesh) {
        const ArrayRef extent =
            scene->schemas.mesh.pointBounds[PrimRow(scene, drawn.prim, kSlotMesh)];
        if (extent.count < 2) {
            return;
        }
        const float3 lower = Ptr<Float3>(extent.address)[0].Vector();
        const float3 upper = Ptr<Float3>(extent.address)[1].Vector();
        center = (lower + upper) / 2;
        half = (upper - lower) / 2;
    }
    const float t[12] = record.transform;
    const float3 x = float3(t[0], t[4], t[8]);
    const float3 y = float3(t[1], t[5], t[9]);
    const float3 z = float3(t[2], t[6], t[10]);
    const float3 middle = float3(t[3], t[7], t[11]) + center.x * x + center.y * y + center.z * z;
    const float3 reach = abs(x) * half.x + abs(y) * half.y + abs(z) * half.z;
    const uint at = WorldSlot(scene, world);
    Grow(p.extents, at * 3, (p.slotCount + at) * 3, middle - reach, middle + reach);
    if (drawn.instancer != kNoPrim) {
        Grow(p.instancerExtents, drawn.instancer * 6, drawn.instancer * 6 + 3, middle - reach,
             middle + reach);
    }
}

[shader("compute")]
[numthreads(kBoundsGroup, 1, 1)]
void InstancerBounds(uniform BoundsParams p) {
    const uint prim = LinearThreadIndex();
    SceneData* scene = p.scene;
    if (prim >= p.primCount || (scene->prims.flags[prim] & kPrimAlive) == 0 ||
        SchemaId(scene->prims.schema[prim]) != SchemaId::PointInstancer) {
        return;
    }
    float3 lower;
    float3 upper;
    for (uint axis = 0; axis < 3; ++axis) {
        lower[axis] = FloatFromOrdered(p.instancerExtents[prim * 6 + axis]);
        upper[axis] = FloatFromOrdered(p.instancerExtents[prim * 6 + 3 + axis]);
    }
    const bool empty = lower.x > upper.x;
    const uint row = PrimRow(scene, prim, kSlotBoundable);
    scene->schemas.boundable.worldMin[row] = Float3(empty ? float3(0) : lower);
    scene->schemas.boundable.worldMax[row] = Float3(empty ? float3(0) : upper);
}

[shader("compute")]
[numthreads(kBoundsGroup, 1, 1)]
void FinishBoxes(uniform BoundsParams p) {
    const uint slot = LinearThreadIndex();
    if (slot >= p.slotCount) {
        return;
    }
    BvhBox box;
    for (uint axis = 0; axis < 3; ++axis) {
        box.lower[axis] = FloatFromOrdered(p.extents[slot * 3 + axis]);
        box.upper[axis] = FloatFromOrdered(p.extents[(p.slotCount + slot) * 3 + axis]);
    }
    p.boxes[slot] = box;
}
