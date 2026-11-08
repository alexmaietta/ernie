// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/kernels/paged.h"

namespace ernie {

struct PrimTable {
    Paged<uint> schema;
    Paged<uint> world;
    Paged<uint> flags;
    Paged<uint> slots;
    Paged<uint64_t> apis;
    Paged<uint> rows;
    uint count;
    uint padding;
};

ERNIE_CHECK_SIZE(PrimTable, 56);

}
