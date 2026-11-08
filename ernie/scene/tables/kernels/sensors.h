// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/records/prims.h"
#include "ernie/scene/records/sensors.h"
#include "ernie/scene/records/transforms.h"

namespace ernie {

struct SensorParams {
    Ptr<AuthoredData> scene;
    Ptr<CameraRecord> cameras;
    Ptr<DistortionRecord> distortions;
    Ptr<DetectorRecord> detectors;
    Ptr<LidarRecord> lidars;
    Ptr<RadarRecord> radars;
    uint cameraRows;
    uint distortionRows;
    uint detectorRows;
    uint lidarRows;
    uint radarRows;
    uint padding0;
};

ERNIE_CHECK_SIZE(SensorParams, 72);

static const uint kSensorGroup = 64;
static const float kSensorMillimeters = 0.001f;

}
