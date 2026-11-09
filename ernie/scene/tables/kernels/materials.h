// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/records/materials.h"
#include "ernie/scene/records/prims.h"
#include "ernie/scene/records/transforms.h"

namespace ernie {

struct MaterialParams {
    Ptr<AuthoredData> scene;
    Ptr<CheckerRecord> checkers;
    Ptr<float> temperatures;
};

ERNIE_CHECK_SIZE(MaterialParams, 24);

static const uint kMaterialGroup = 64;

}
