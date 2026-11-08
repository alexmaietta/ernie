// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/records/materials.h"
#include "ernie/scene/tables/kernels/authored.h"

namespace ernie {

struct MaterialParams {
    Ptr<AuthoredData> scene;
    Ptr<CheckerRecord> checkers;
};

ERNIE_CHECK_SIZE(MaterialParams, 16);

struct PackParams {
    Ptr<PackEntry> entries;
    Ptr<uint> words;
    uint count;
    uint padding0;
};

ERNIE_CHECK_SIZE(PackParams, 24);

static const uint kMaterialGroup = 64;

}
