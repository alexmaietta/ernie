// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/schema/types.h"

namespace ernie {

struct BoundableBox {
    Float3 lower;
    Float3 upper;
};

ERNIE_CHECK_SIZE(BoundableBox, 24);

#ifdef __SLANG__

uint OrderedFloat(float value) {
    const uint bits = asuint(value);
    return (bits & 0x80000000u) != 0 ? ~bits : bits | 0x80000000u;
}

float FloatFromOrdered(uint bits) {
    return asfloat((bits & 0x80000000u) != 0 ? bits & 0x7FFFFFFFu : ~bits);
}

#endif

}
