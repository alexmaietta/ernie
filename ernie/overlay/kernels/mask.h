// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/kernels/shared.h"

namespace ernie {

struct OverlayOutline {
    Ptr<uint> bits;
    uint bitWords;
    uint color;
    float width;
    uint overlay;
};

ERNIE_CHECK_SIZE(OverlayOutline, 24);

}
