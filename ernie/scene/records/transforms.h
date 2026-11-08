// SPDX-License-Identifier: BSD-3-Clause
#pragma once

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

}
