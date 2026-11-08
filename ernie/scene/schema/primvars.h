// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/kernels/shared.h"

namespace ernie {

ERNIE_ENUM(PrimvarType) {
    Float,
    Float2,
    Float3,
    Float4,
    Int,
    Int2,
    Int3,
    Int4,
};

struct PrimvarEntry {
    uint name;
    uint type;
    uint interpolation;
    uint elementSize;
    uint values;
    uint valueCount;
    uint indices;
    uint indexCount;
};

ERNIE_CHECK_SIZE(PrimvarEntry, 32);

static const uint kPrimvarEntryWords = 8;

ERNIE_SHARED uint PrimvarComponents(PrimvarType type) {
    return uint(type) % 4 + 1;
}

ERNIE_SHARED bool PrimvarIsInt(PrimvarType type) {
    return uint(type) >= uint(PrimvarType::Int);
}

}
