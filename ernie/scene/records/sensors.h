// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/stage/schema/types.h"

namespace ernie {

static const uint kMostLidarReturns = 4;

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
    float readout;
    float fNumber;
    float exposureTime;
    float iso;
    uint padding0;
    uint padding1;
    uint padding2;
};

ERNIE_CHECK_SIZE(CameraRecord, 96);

struct DistortionRecord {
    Float4 radial;
    Float4 thinPrism;
    ArrayRef fTheta;
    Float3 radialDivisor;
    float maxAngle;
    Float2 tangential;
    uint model;
    uint padding0;
};

ERNIE_CHECK_SIZE(DistortionRecord, 80);

static const uint kDetectorBands = 8;
static const uint kMostDetectorColors = 3;

struct BandSet {
    Float4 fromRgb[6];
    Float4 toRgb[6];
    float first;
    float width;
    uint padding0;
    uint padding1;
};

ERNIE_CHECK_SIZE(BandSet, 208);

struct DetectorRecord {
    Float4 qe[2 * kMostDetectorColors];
    BandSet bands;
    uint pattern;
    uint channels;
    float fullWell;
    float readNoise;
    float darkCurrent;
    float darkCurrentDoubling;
    float temperature;
    float prnu;
    float dsnu;
    float systemGain;
    uint bitDepth;
    float blackLevel;
    uint seed;
    uint padding0;
    uint padding1;
    uint padding2;
};

ERNIE_CHECK_SIZE(DetectorRecord, 368);

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
    float wavelength;
    uint padding0;
};

ERNIE_CHECK_SIZE(LidarRecord, 80);

struct RadarRecord {
    Float2 fieldOfView;
    Float2 range;
    float frequency;
    uint padding0;
};

ERNIE_CHECK_SIZE(RadarRecord, 24);

}
