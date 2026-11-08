// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/schema/types.h"

namespace ernie {

static const uint kNoSpectrum = 0xFFFFFFFF;

struct SpectraRecord {
    Ptr<Float3> rgb;
    Ptr<float> bands;
    Ptr<Uint2> ranges;
    Ptr<Float2> samples;
    uint count;
    uint detectors;
};

ERNIE_CHECK_SIZE(SpectraRecord, 40);

}
