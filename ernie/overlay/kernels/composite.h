// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/kernels/shared.h"

namespace ernie {

struct BlendParams {
    Ptr<uint> color;
    Ptr<uint> overlay;
    Ptr<uint> blended;
    uint count;
    uint padding;
};

ERNIE_CHECK_SIZE(BlendParams, 32);

}
