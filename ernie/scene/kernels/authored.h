// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/kernels/ids.h"
#include "ernie/scene/kernels/schema/schema.h"
#include "ernie/scene/records/transforms.h"

namespace ernie {

static const uint kPrimAlive = 1u << 0;
static const uint kPrimSession = 1u << 1;

struct PrimData {
    Paged<uint> schema;
    Paged<uint> parent;
    Paged<uint> world;
    Paged<uint> flags;
    Paged<uint> slots;
    Paged<uint64_t> apis;
};

ERNIE_CHECK_SIZE(PrimData, 48);

struct AuthoredData {
    PrimData prims;
    Paged<uint> rows;
    Paged<uint64_t> worldStamps;
    Ptr<uint64_t> textureBindings;
    Schemas schemas;
    uint primCount;
    uint worldCount;
};

ERNIE_CHECK_SIZE(AuthoredData, 80 + sizeof(Schemas));

#ifdef __SLANG__

uint PrimRow(AuthoredData* scene, uint prim, uint slot) {
    return scene->rows[scene->prims.slots[prim] + slot];
}

bool HasApi(AuthoredData* scene, uint prim, uint api) {
    const uint64_t apis = scene->prims.apis[prim];
    return ((api < 32 ? uint(apis) >> api : uint(apis >> 32) >> (api - 32)) & 1) != 0;
}

uint ApiRow(AuthoredData* scene, uint prim, uint api) {
    const uint64_t apis = scene->prims.apis[prim];
    const uint low = uint(apis);
    const uint below =
        api < 32 ? countbits(low & ((1u << api) - 1))
                 : countbits(low) + countbits(uint(apis >> 32) & ((1u << (api - 32)) - 1));
    const uint typed = kSchemaSlotCounts[scene->prims.schema[prim]];
    return scene->rows[scene->prims.slots[prim] + typed + below];
}

uint ApiInstanceRow(AuthoredData* scene, uint prim, uint api, uint instance, Paged<uint> names,
                    Paged<uint> next) {
    if (!HasApi(scene, prim, api)) {
        return kNoRow;
    }
    uint row = ApiRow(scene, prim, api);
    while (row != kNoRow && names[row] != instance) {
        row = next[row];
    }
    return row;
}

WorldTransform PrimWorld(AuthoredData* scene, Ptr<WorldTransform> transforms, uint prim) {
    return transforms[PrimRow(scene, prim, kSlotXformable)];
}

ResolvedImageable PrimImageable(AuthoredData* scene, Ptr<ResolvedImageable> imageables, uint prim) {
    return imageables[PrimRow(scene, prim, kSlotImageable)];
}

bool ImageableShown(ResolvedImageable resolved) {
    return resolved.visibility != uint(Visibility::Invisible) &&
           resolved.purpose != uint(Purpose::Proxy) && resolved.purpose != uint(Purpose::Guide);
}

#endif

}
