// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/records/sensors.h"
#include "ernie/scene/tables/kernels/authored.h"

namespace ernie {

struct SensorParams {
    Ptr<AuthoredData> scene;
    Ptr<CameraRecord> cameras;
    Ptr<LidarRecord> lidars;
    Ptr<RadarRecord> radars;
    uint cameraRows;
    uint lidarRows;
    uint radarRows;
    uint padding;
};

ERNIE_CHECK_SIZE(SensorParams, 48);

static const uint kSensorGroup = 64;
static const float kSensorMillimeters = 0.001f;

}
