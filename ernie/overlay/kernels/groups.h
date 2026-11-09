// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/kernels/shared.h"

namespace ernie {

struct OverlayGroupParams {
    Ptr<uint> worlds;
    Ptr<uint> itemCounts;
    Ptr<uint> keys;
    Ptr<uint> order;
    Ptr<uint> starts;
    Ptr<uint> needs;
    uint rows;
    uint worldStride;
    uint groups;
    uint need;
};

ERNIE_CHECK_SIZE(OverlayGroupParams, 64);

}
