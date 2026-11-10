// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/records/views.h"
#include "ernie/stage/schema/ids.h"

namespace ernie {

static const uint kViewGroup = 64;

struct ViewEntry {
    uint product;
    uint camera;
    uint distortion;
    uint padding;
};

ERNIE_CHECK_SIZE(ViewEntry, 16);

struct ViewParams {
    Ptr<CameraRecord> cameras;
    Ptr<DistortionRecord> distortions;
    Ptr<ProductRecord> products;
    Ptr<ViewEntry> entries;
    Ptr<ViewRecord> views;
    uint count;
    uint padding;
};

ERNIE_CHECK_SIZE(ViewParams, 48);

}
