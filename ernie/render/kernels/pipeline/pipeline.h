// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/records/render_tables.h"

namespace ernie {

static const uint kPipelineGroup = 64;
static const uint kMostSignalChannels = 8;
static const uint kSignalHalf = 1;
static const uint kSignalMosaic = 2;
static const uint kSignalSampled = 4;
static const uint kSignalStrideShift = 8;
static const uint kStatWords = 16;
static const uint kReduceGroups = 64;

struct PipelineValue {
    uint64_t address;
    uint channels;
    uint flags;
};

ERNIE_CHECK_SIZE(PipelineValue, 16);

struct PipelineCapture {
    uint64_t firstElement;
    Ptr<uint> words;
    uint sensor;
    uint detector;
    uint width;
    uint height;
    uint capture;
    uint identity;
    uint firstValue;
    uint camera;
    Ptr<float> state;
};

ERNIE_CHECK_SIZE(PipelineCapture, 56);

struct PipelineParams {
    Ptr<RenderTables> tables;
    Ptr<PipelineCapture> captures;
    Ptr<PipelineValue> values;
    uint64_t elementCount;
    uint captureCount;
    uint padding0;
};

ERNIE_CHECK_SIZE(PipelineParams, 40);

}
