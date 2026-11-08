// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/columns/kernels/authored.h"
#include "ernie/scene/schema/types.h"

namespace ernie {

struct ResolvedImageable {
    uint visibility;
    uint purpose;
    uint material;
    uint motion;
};

ERNIE_CHECK_SIZE(ResolvedImageable, 16);

struct MotionRecord {
    Float3 linear;
    uint padding0;
    Float3 angular;
    uint padding1;
};

ERNIE_CHECK_SIZE(MotionRecord, 32);

#ifdef __SLANG__

WorldTransform PrimWorld(AuthoredData* scene, Ptr<WorldTransform> transforms, uint prim) {
    return transforms[PrimRow(scene, prim, kSlotXformable)];
}

ResolvedImageable PrimImageable(AuthoredData* scene, Ptr<ResolvedImageable> imageables, uint prim) {
    return imageables[PrimRow(scene, prim, kSlotImageable)];
}

uint BindingRow(AuthoredData* scene, uint prim) {
    return ApiRow(scene, prim, kApiMaterialBindingAPI);
}

uint BoundMaterial(AuthoredData* scene, uint prim, out bool binds) {
    const ArrayRef binding =
        scene->schemas.materialBindingAPI.materialBinding[BindingRow(scene, prim)];
    binds = binding.count > 0;
    if (!binds) {
        return kNoPrim;
    }
    const uint target = Ptr<uint>(binding.address)[0];
    const bool bound = target < scene->primCount &&
                       (scene->prims.flags[target] & kPrimAlive) != 0 &&
                       SchemaIsA(scene->prims.schema[target], SchemaId::Material);
    return bound ? target : kNoPrim;
}

bool ImageableShown(ResolvedImageable resolved) {
    return resolved.visibility != uint(Visibility::Invisible) &&
           resolved.purpose != uint(Purpose::Proxy) && resolved.purpose != uint(Purpose::Guide);
}

#endif

}
