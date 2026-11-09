// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/stage/schema/types.h"

namespace ernie {

static const uint kNoSpectrum = 0xFFFFFFFF;

struct SpectraRecord {
    Ptr<Float3> rgb;
    Ptr<float> bands;
    Ptr<Uint2> ranges;
    Ptr<Float2> samples;
    Ptr<float> sigmoids;
    Ptr<float> sigmoidScales;
    uint count;
    uint detectors;
    uint sigmoidResolution;
    uint padding0;
};

ERNIE_CHECK_SIZE(SpectraRecord, 64);

}
