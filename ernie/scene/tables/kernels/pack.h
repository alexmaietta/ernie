// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/stage/columns/kernels/paged.h"
#include "ernie/stage/schema/types.h"

namespace ernie {

struct PackEntry {
    uint64_t pages;
    uint row;
    uint words;
    uint destination;
    uint value;
};

ERNIE_CHECK_SIZE(PackEntry, 24);

struct PackParams {
    Ptr<PackEntry> entries;
    Ptr<uint> words;
    uint count;
    uint padding0;
};

ERNIE_CHECK_SIZE(PackParams, 24);

static const uint kPackGroup = 64;

}
