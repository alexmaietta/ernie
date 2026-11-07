// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/kernels/ids.h"

namespace ernie {

struct GatherParams {
    Ptr<uint> source;
    Ptr<uint> rows;
    Ptr<uint> out;
    uint words;
    uint count;
};

ERNIE_CHECK_SIZE(GatherParams, 32);

static const uint kReadbackGroup = 64;

}
