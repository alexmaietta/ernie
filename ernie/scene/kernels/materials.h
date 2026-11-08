// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/kernels/authored.h"
#include "ernie/scene/tables/materials.h"

namespace ernie {

struct MaterialParams {
    Ptr<AuthoredData> scene;
    Ptr<PreviewRecord> previews;
    Ptr<NonVisualRecord> nonVisuals;
    Ptr<CheckerRecord> checkers;
};

ERNIE_CHECK_SIZE(MaterialParams, 32);

static const uint kMaterialGroup = 64;

}
