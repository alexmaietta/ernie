// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/schema/types.h"

namespace ernie {

static const float kDefaultPointWidth = 0.01f;

struct PointsRecord {
    uint64_t blas;
    uint64_t points;
    uint64_t widths;
    uint pointCount;
    uint widthCount;
    ArrayRef bounds;
};

ERNIE_CHECK_SIZE(PointsRecord, 48);

#ifdef __SLANG__

float PointRadius(PointsRecord record, uint point) {
    if (record.widthCount == 0) {
        return kDefaultPointWidth / 2;
    }
    const uint at = record.widthCount == record.pointCount ? point : 0;
    return abs(Ptr<float>(record.widths)[at]) / 2;
}

#endif

}
