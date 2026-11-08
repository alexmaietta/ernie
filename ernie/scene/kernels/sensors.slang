// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;

#include "ernie/scene/kernels/sensors.h"

using namespace ernie;

uint LatticeStep(uint samples) {
    uint step = max(uint(float(samples) * 0.618034f), 1u);
    for (;;) {
        uint a = step;
        uint b = samples;
        while (b != 0) {
            const uint r = a % b;
            a = b;
            b = r;
        }
        if (a == 1) {
            return step;
        }
        ++step;
    }
}

[shader("compute")]
[numthreads(kSensorGroup, 1, 1)]
void WriteCameras(uniform SensorParams p) {
    const uint row = LinearThreadIndex();
    if (row >= p.cameraRows) {
        return;
    }
    CameraData* data = &p.scene->schemas.camera;
    CameraRecord record;
    record.aperture = Float2(float2(data.horizontalAperture[row], data.verticalAperture[row]));
    record.offset =
        Float2(float2(data.horizontalApertureOffset[row], data.verticalApertureOffset[row]));
    record.focalLength = data.focalLength[row];
    record.projection = data.projection[row];
    const float fStop = data.fStop[row];
    record.apertureRadius = Projection(record.projection) == Projection::Orthographic || fStop <= 0
                                ? 0
                                : record.focalLength * kSensorMillimeters / (2 * fStop);
    record.focusDistance = data.focusDistance[row];
    record.clipping = Float2(float2(data.clippingRange[row].x, data.clippingRange[row].y));
    record.shutter = Float2(float2(data.shutterOpen[row], data.shutterClose[row]));
    record.clippingPlanes = data.clippingPlanes[row];
    const float exposureFStop = data.exposureFStop[row];
    const float sensitivity = data.exposureResponsivity[row] * (data.exposureIso[row] / 100);
    const float light = data.exposureTime[row] / (exposureFStop * exposureFStop);
    record.exposure = sensitivity * light * exp2(data.exposure[row]);
    record.padding0 = 0;
    record.padding1 = 0;
    record.padding2 = 0;
    p.cameras[row] = record;
}

[shader("compute")]
[numthreads(kSensorGroup, 1, 1)]
void WriteLidars(uniform SensorParams p) {
    const uint row = LinearThreadIndex();
    if (row >= p.lidarRows) {
        return;
    }
    LidarData* data = &p.scene->schemas.lidar;
    LidarRecord record;
    record.elevations = data.emitterElevations[row];
    record.azimuths = data.emitterAzimuths[row];
    record.sweep = Float2(radians(float2(data.azimuthRange[row].x, data.azimuthRange[row].y)));
    record.range = Float2(float2(data.range[row].x, data.range[row].y));
    record.divergence =
        Float2(radians(float2(data.beamDivergence[row].x, data.beamDivergence[row].y)));
    record.rotary = ScanType(data.scanType[row]) == ScanType::Rotary ? 1 : 0;
    record.samples = max(data.beamSamples[row], 1u);
    record.latticeStep = LatticeStep(record.samples);
    record.returnMode = data.returnMode[row];
    record.padding0 = 0;
    record.padding1 = 0;
    p.lidars[row] = record;
}

[shader("compute")]
[numthreads(kSensorGroup, 1, 1)]
void WriteRadars(uniform SensorParams p) {
    const uint row = LinearThreadIndex();
    if (row >= p.radarRows) {
        return;
    }
    RadarData* data = &p.scene->schemas.radar;
    RadarRecord record;
    record.fieldOfView = Float2(radians(float2(data.fieldOfView[row].x, data.fieldOfView[row].y)));
    record.range = Float2(float2(data.range[row].x, data.range[row].y));
    p.radars[row] = record;
}
