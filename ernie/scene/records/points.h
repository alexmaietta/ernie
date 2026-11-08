// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/schema/types.h"

namespace ernie {

static const float kDefaultPointWidth = 1.0f;

struct PointsRecord {
    uint64_t blas;
    uint64_t points;
    uint64_t widths;
    uint64_t normals;
    uint64_t ids;
    uint64_t order;
    uint64_t packed;
    uint pointCount;
    uint widthCount;
    uint normalCount;
    uint idCount;
    uint perBox;
    uint padding;
    ArrayRef bounds;
};

ERNIE_CHECK_SIZE(PointsRecord, 96);

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
