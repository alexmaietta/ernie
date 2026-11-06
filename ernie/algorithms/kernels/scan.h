// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/kernels/shared.h"

namespace ernie {

static const uint kScanThreads = 256;
static const uint kScanItems = 4;
static const uint kScanTileItems = kScanThreads * kScanItems;

struct ScanParams {
    Ptr<uint> input;
    Ptr<uint> output;
    Ptr<uint> sums;
    Ptr<uint> deviceCount;
    uint count;
    uint hasSums;
};

ERNIE_CHECK_SIZE(ScanParams, 40);

}
