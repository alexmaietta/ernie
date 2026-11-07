// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/kernels/instance_layout.h"
#include "ernie/scene/kernels/lights.h"
#include "ernie/scene/kernels/origins.h"
#include "ernie/scene/kernels/scene.h"
#include "ernie/scene/kernels/semantics.h"
#include "ernie/scene/kernels/tables/meshes.h"
#include "ernie/scene/kernels/tables/transforms.h"

namespace ernie {

struct RenderTables {
    Ptr<WorldTransform> transforms;
    Ptr<ResolvedImageable> imageables;
    Ptr<MeshRecord> meshes;
    Ptr<TangentRecord> tangents;
    Ptr<InstanceLayout> instances;
    Ptr<LightLayout> lights;
    Ptr<RenderOrigin> origins;
    uint64_t bounds;
    SemanticsData semantics;
    uint worldCount;
    uint padding;
};

ERNIE_CHECK_SIZE(RenderTables, 104);

#ifdef __SLANG__

RenderOrigin TlasOrigin(RenderTables* tables, uint world) {
    return tables->origins[min(world, tables->worldCount)];
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
