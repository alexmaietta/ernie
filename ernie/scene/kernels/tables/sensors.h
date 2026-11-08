// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/kernels/schema/types.h"

namespace ernie {

struct CameraRecord {
    Float2 aperture;
    Float2 offset;
    float focalLength;
    uint projection;
    float apertureRadius;
    float focusDistance;
    Float2 clipping;
    Float2 shutter;
    ArrayRef clippingPlanes;
    float exposure;
    uint padding0;
    uint padding1;
    uint padding2;
};

ERNIE_CHECK_SIZE(CameraRecord, 80);

struct LidarRecord {
    ArrayRef elevations;
    ArrayRef azimuths;
    Float2 sweep;
    Float2 range;
    Float2 divergence;
    uint rotary;
    uint samples;
    uint latticeStep;
    uint returnMode;
    uint padding0;
    uint padding1;
};

ERNIE_CHECK_SIZE(LidarRecord, 80);

struct RadarRecord {
    Float2 fieldOfView;
    Float2 range;
};

ERNIE_CHECK_SIZE(RadarRecord, 16);

}
