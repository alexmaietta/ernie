// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/kernels/shared.h"

namespace ernie {

struct StableId {
    uint words[4];
};

ERNIE_CHECK_SIZE(StableId, 16);

struct SemanticsData {
    Ptr<StableId> stableIds;
    Ptr<uint> semanticIds;
    Ptr<uint> semanticPrims;
    uint stride;
    uint taxonomyCount;
};

ERNIE_CHECK_SIZE(SemanticsData, 32);

}
