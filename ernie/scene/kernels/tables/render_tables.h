// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/kernels/instance_layout.h"
#include "ernie/scene/kernels/lights.h"
#include "ernie/scene/kernels/scene.h"
#include "ernie/scene/kernels/semantics.h"
#include "ernie/scene/kernels/tables/bounds.h"
#include "ernie/scene/kernels/tables/geometry.h"
#include "ernie/scene/kernels/tables/hierarchy.h"
#include "ernie/scene/kernels/tables/instances.h"
#include "ernie/scene/kernels/tables/prims.h"
#include "ernie/scene/kernels/tables/sensors.h"
#include "ernie/scene/kernels/tables/tangents.h"
#include "ernie/scene/kernels/tables/textures.h"

namespace ernie {

struct RenderTables {
    PrimTable prims;
    Ptr<WorldTransform> transforms;
    Ptr<ResolvedImageable> imageables;
    Ptr<MeshRecord> meshes;
    Ptr<TangentRecord> tangents;
    Ptr<SurfaceRecord> surfaces;
    Ptr<BoundableBox> primBounds;
    Ptr<TextureRecord> textures;
    Ptr<InstanceLayout> instances;
    Ptr<LightLayout> lights;
    Ptr<RenderOrigin> origins;
    Ptr<CameraRecord> cameras;
    Ptr<LidarRecord> lidars;
    Ptr<RadarRecord> radars;
    SemanticsData semantics;
    uint worldCount;
    uint padding;
};

ERNIE_CHECK_SIZE(RenderTables, 200);

#ifdef __SLANG__

bool PrimAlive(RenderTables* tables, uint prim) {
    return prim < tables->prims.count && (tables->prims.flags[prim] & kPrimAlive) != 0;
}

SchemaId PrimSchema(RenderTables* tables, uint prim) {
    return SchemaId(tables->prims.schema[prim]);
}

uint PrimRow(RenderTables* tables, uint prim, uint slot) {
    return tables->prims.rows[tables->prims.slots[prim] + slot];
}

bool HasApi(RenderTables* tables, uint prim, uint api) {
    const uint64_t apis = tables->prims.apis[prim];
    return ((api < 32 ? uint(apis) >> api : uint(apis >> 32) >> (api - 32)) & 1) != 0;
}

uint ApiRow(RenderTables* tables, uint prim, uint api) {
    const uint64_t apis = tables->prims.apis[prim];
    const uint low = uint(apis);
    const uint below =
        api < 32 ? countbits(low & ((1u << api) - 1))
                 : countbits(low) + countbits(uint(apis >> 32) & ((1u << (api - 32)) - 1));
    const uint typed = kSchemaSlotCounts[tables->prims.schema[prim]];
    return tables->prims.rows[tables->prims.slots[prim] + typed + below];
}

WorldTransform PrimWorld(RenderTables* tables, uint prim) {
    return tables->transforms[PrimRow(tables, prim, kSlotXformable)];
}

ResolvedImageable PrimImageable(RenderTables* tables, uint prim) {
    return tables->imageables[PrimRow(tables, prim, kSlotImageable)];
}

RenderOrigin TlasOrigin(RenderTables* tables, uint world) {
    return tables->origins[TlasIndex(world, tables->worldCount)];
}

float3 RenderTranslation(RenderTables* tables, WorldTransform w, uint world) {
    return RelativeTranslation(TlasOrigin(tables, world), w);
}

float3 RenderPoint(RenderTables* tables, float3 point, uint world) {
    const RenderOrigin origin = TlasOrigin(tables, world);
    return (point - origin.high.Vector()) - origin.low.Vector();
}

float3 RenderShift(SceneData* scene, RenderTables* tables, uint from, uint world) {
    const RenderOrigin origin = TlasOrigin(tables, world);
    InstanceLayout* layout = tables->instances;
    if (from >= tables->worldCount || layout->worldOrigins[from] == kNoRow) {
        return -origin.high.Vector() - origin.low.Vector();
    }
    const WorldTransform anchor = tables->transforms[layout->worldOrigins[from]];
    return (anchor.high.Vector() - origin.high.Vector()) +
           (anchor.low.Vector() - origin.low.Vector());
}

#endif

}
