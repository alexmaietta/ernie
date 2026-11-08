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

bool ImageableShown(ResolvedImageable resolved) {
    return resolved.visibility != uint(Visibility::Invisible) &&
           resolved.purpose != uint(Purpose::Proxy) && resolved.purpose != uint(Purpose::Guide);
}

#endif

}
