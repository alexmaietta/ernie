// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/kernels/schema/types.h"

namespace ernie {

struct ResolvedImageable {
    uint visibility;
    uint purpose;
    uint material;
    uint motion;
};

ERNIE_CHECK_SIZE(ResolvedImageable, 16);

}
