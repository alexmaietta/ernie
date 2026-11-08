// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/columns/kernels/authored.h"
#include "ernie/scene/columns/kernels/paged.h"

namespace ernie {

struct PrimTable {
    Paged<uint> schema;
    Paged<uint> world;
    Paged<uint> flags;
    Paged<uint> slots;
    Paged<uint64_t> apis;
    Paged<uint> rows;
    Paged<uint> parent;
    uint count;
    uint padding;
};

ERNIE_CHECK_SIZE(PrimTable, 64);

}
