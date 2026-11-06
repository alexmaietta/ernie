// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/kernels/shared.h"

namespace ernie {

static const uint kDigitBits = 8;
static const uint kDigitBins = 1 << kDigitBits;
static const uint kSortThreads = kDigitBins;
static const uint kSortItems = 4;
static const uint kSortTileItems = kSortThreads * kSortItems;

struct SortParams {
    Ptr<uint> keys;
    Ptr<uint> values;
    Ptr<uint> keysOut;
    Ptr<uint> valuesOut;
    Ptr<uint> counts;
    Ptr<uint> deviceCount;
    uint count;
    uint shift;
    uint mask;
    uint tiles;
    uint keyWords;
    uint padding;
};

ERNIE_CHECK_SIZE(SortParams, 72);

}
