// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;

#include "ernie/scene/tables/kernels/instances.h"

using namespace ernie;

InstanceRecord Inactive(uint local) {
    InstanceRecord instance;
    instance.transform = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    instance.custom = local;
    instance.flags = 0;
    instance.blas = 0;
    return instance;
}

InstanceRecord MakeInstance(InstanceParams p, uint prim, uint world, uint local, uint slot) {
    return PlaceInstance(p, prim, PrimWorld(p.scene, p.transforms, prim), true, world, local, slot);
}

InstanceRecord PlaceInstance(InstanceParams p, uint prim, WorldTransform w, bool shown, uint world,
                             uint local, uint slot) {
    AuthoredData* scene = p.scene;
    const SchemaId schema = SchemaId(scene->prims.schema[prim]);
    const UnitShape shape = PlaceShape(scene, prim, w);
    p.shapes[slot] = shape.record;
    p.slotWorlds[slot] = w;
    uint64_t blas = p.unitBox;
    if (ShapeKind(shape.record.kind) == ShapeKind::Mesh) {
        blas = schema == SchemaId::Mesh ? p.meshes[PrimRow(scene, prim, kSlotMesh)].blas
                                        : 0;
    }
    const bool point =
        (schema == SchemaId::SphereLight &&
         scene->schemas.sphereLight.treatAsPoint[PrimRow(scene, prim, kSlotSphereLight)] != 0) ||
        (schema == SchemaId::CylinderLight &&
         scene->schemas.cylinderLight.treatAsLine[PrimRow(scene, prim, kSlotCylinderLight)] != 0);
    if (blas == 0 || point || !shown || !ImageableShown(PrimImageable(scene, p.imageables, prim))) {
        return Inactive(local);
    }

    const float3 translation =
        RelativeTranslation(p.origins[TlasIndex(world, p.layout->worldCount)], w);
    const float3 x = shape.axes[0];
    const float3 y = shape.axes[1];
    const float3 z = shape.axes[2];
    InstanceRecord instance;
    instance.transform = { x.x, y.x, z.x, translation.x,
                           x.y, y.y, z.y, translation.y,
                           x.z, y.z, z.z, translation.z };
    uint mask = kGeometryMask;
    if (IsEmitter(uint(schema))) {
        mask = kEmitterMask;
        if (scene->schemas.lightAPI.visibleInPrimaryRay[ApiRow(scene, prim, kApiLightAPI)] != 0) {
            mask |= kCameraEmitterMask;
        }
    }
    instance.custom = local | (mask << 24);
    instance.flags = 0;
    instance.blas = blas;
    return instance;
}

InstanceRecord Shift(InstanceRecord record, float3 offset) {
    if (record.blas != 0) {
        record.transform[3] += offset.x;
        record.transform[7] += offset.y;
        record.transform[11] += offset.z;
    }
    return record;
}

InstanceRecord Expand(InstanceParams p, uint item, uint world, uint local, uint slot) {
    InstanceLayout* layout = p.layout;
    const uint index = item & ~kInstancerSlot;
    const InstancerBlock block = layout->blocks[index];
    const ResolvedCopy copy = ResolveCopy(p.scene, p.transforms, p.imageables, layout, index,
                                          local - block.first, false);
    p.hits[slot] = Drawn(copy.valid ? copy.prim : kNoPrim, block.instancer, copy.instance,
                         local - block.first);
    if (!copy.valid) {
        return Inactive(local);
    }
    const InstanceRecord record =
        PlaceInstance(p, copy.prim, copy.world, copy.shown, world, local, slot);
    if (record.blas == 0) {
        p.hits[slot] = Drawn(kNoPrim, block.instancer, copy.instance, local - block.first);
    }
    return record;
}

[shader("compute")]
[numthreads(kInstanceGroup, 1, 1)]
void WriteInstances(uniform InstanceParams p) {
    const uint3 thread = ThreadIndex();
    InstanceLayout* layout = p.layout;
    const uint tlas = p.tlases[thread.y];
    const uint slot = layout->firstInstances[tlas] + thread.x;
    if (slot >= layout->firstInstances[tlas + 1]) {
        return;
    }
    if (thread.x >= layout->instanceCounts[tlas]) {
        p.instances[slot] = Inactive(thread.x);
        p.hits[slot] = Drawn(kNoPrim, kNoPrim, 0, 0);
        return;
    }
    const uint world = TlasWorld(tlas, layout->worldCount);
    const uint item = layout->instancePrims[slot];
    if ((item & kInstancerSlot) != 0) {
        const uint instancer = layout->blocks[item & ~kInstancerSlot].instancer;
        p.instances[slot] = Shift(Expand(p, item, world, thread.x, slot),
                                  ShownOffset(layout, world, p.scene->prims.world[instancer]));
        return;
    }
    p.hits[slot] = Drawn(item, kNoPrim, 0, 0);
    p.instances[slot] = Shift(MakeInstance(p, item, world, thread.x, slot),
                              ShownOffset(layout, world, p.scene->prims.world[item]));
}

[shader("compute")]
[numthreads(kInstanceGroup, 1, 1)]
void GatherViewers(uniform OriginParams p) {
    const uint index = LinearThreadIndex();
    if (index >= p.productCount) {
        return;
    }
    AuthoredData* scene = p.scene;
    InstanceLayout* layout = p.layout;
    const uint product = p.products[index];
    const ArrayRef sensors =
        scene->schemas.renderProduct.camera[PrimRow(scene, product, kSlotRenderProduct)];
    for (uint at = 0; at < sensors.count; ++at) {
        const uint sensor = Ptr<uint>(sensors.address)[at];
        if (sensor < scene->primCount && (scene->prims.flags[sensor] & kPrimAlive) != 0) {
            const uint world = scene->prims.world[sensor];
            InterlockedMin(p.viewers[TlasIndex(world, layout->worldCount)], sensor);
        }
    }
}

[shader("compute")]
[numthreads(kInstanceGroup, 1, 1)]
void PlaceOrigins(uniform OriginParams p) {
    const uint tlas = LinearThreadIndex();
    if (tlas >= p.tlasCount) {
        return;
    }
    AuthoredData* scene = p.scene;
    InstanceLayout* layout = p.layout;
    const uint viewer = p.viewers[tlas];
    p.viewers[tlas] = kNoPrim;
    RenderOrigin origin;
    origin.high = Float3(float3(0));
    origin.low = Float3(float3(0));
    if (viewer != kNoPrim) {
        origin = SnapOrigin(PrimWorld(scene, p.transforms, viewer));
    } else if (tlas < layout->worldCount && layout->worldOrigins[tlas] != kNoRow) {
        const WorldTransform anchor = p.transforms[layout->worldOrigins[tlas]];
        origin.high = anchor.high;
        origin.low = anchor.low;
    }
    p.origins[tlas] = origin;
}
