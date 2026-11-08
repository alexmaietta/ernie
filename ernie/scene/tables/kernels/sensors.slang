// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;
import ernie.scene.tables.kernels.uplift;

#include "ernie/scene/tables/kernels/sensors.h"

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
    record.readout = max(data.shutterReadout[row], 0.0f);
    record.fNumber = exposureFStop > 0 ? exposureFStop : fStop > 0 ? fStop : 1;
    record.exposureTime = max(data.exposureTime[row], 0.0f);
    record.iso = max(data.exposureIso[row], 0.0f);
    record.padding0 = 0;
    record.padding1 = 0;
    record.padding2 = 0;
    p.cameras[row] = record;
}

[shader("compute")]
[numthreads(kSensorGroup, 1, 1)]
void WriteDistortions(uniform SensorParams p) {
    const uint row = LinearThreadIndex();
    if (row >= p.distortionRows) {
        return;
    }
    LensDistortionAPIData* data = &p.scene->schemas.lensDistortionAPI;
    DistortionRecord record;
    record.radial = data.distortionRadial[row];
    record.thinPrism = data.distortionThinPrism[row];
    record.fTheta = data.distortionFTheta[row];
    record.radialDivisor = data.distortionRadialDivisor[row];
    record.maxAngle = radians(clamp(data.distortionMaxFieldOfView[row], 0.0f, 360.0f)) * 0.5f;
    record.tangential = data.distortionTangential[row];
    record.model = data.distortionModel[row];
    record.padding0 = 0;
    p.distortions[row] = record;
}

enum class QeColor : uint {
    Red,
    Green,
    Blue,
    Clear,
}

static const float2 kGenericQe[] = {
    float2(560, 0.00f), float2(580, 0.05f), float2(590, 0.25f), float2(600, 0.45f),
    float2(620, 0.50f), float2(640, 0.45f), float2(660, 0.30f), float2(680, 0.12f),
    float2(700, 0.02f), float2(720, 0.00f),

    float2(460, 0.02f), float2(480, 0.10f), float2(500, 0.35f), float2(520, 0.52f),
    float2(540, 0.55f), float2(560, 0.48f), float2(580, 0.30f), float2(600, 0.12f),
    float2(620, 0.04f), float2(650, 0.01f), float2(680, 0.00f),

    float2(380, 0.05f), float2(400, 0.25f), float2(430, 0.42f), float2(450, 0.46f),
    float2(470, 0.42f), float2(490, 0.28f), float2(510, 0.12f), float2(530, 0.04f),
    float2(560, 0.01f), float2(600, 0.00f),

    float2(350, 0.05f), float2(400, 0.30f), float2(450, 0.50f), float2(500, 0.60f),
    float2(550, 0.62f), float2(600, 0.60f), float2(650, 0.55f), float2(700, 0.48f),
    float2(750, 0.40f), float2(800, 0.32f), float2(850, 0.24f), float2(900, 0.16f),
    float2(950, 0.08f), float2(1000, 0.03f), float2(1100, 0.00f),
};

static const uint kGenericQeStarts[] = { 0, 10, 21, 31, 46 };

static const float kRgbBasisFirstNm = 402.5f;
static const float kRgbBasisStepNm = 45;
static const float3 kRgbBasis[] = {
    float3(0.05f, 0.00f, 0.60f), float3(0.00f, 0.05f, 1.00f), float3(0.00f, 0.45f, 0.70f),
    float3(0.05f, 1.00f, 0.15f), float3(0.45f, 0.85f, 0.00f), float3(1.00f, 0.35f, 0.00f),
    float3(0.80f, 0.05f, 0.00f), float3(0.25f, 0.00f, 0.00f),
};

float3 RgbBasis(float nanometers) {
    const float at = clamp((nanometers - kRgbBasisFirstNm) / kRgbBasisStepNm, 0.0f, 7.0f);
    const uint below = min(uint(at), 6u);
    return lerp(kRgbBasis[below], kRgbBasis[below + 1], at - float(below));
}

BandSet WriteBandSet(float first, float width) {
    BandSet bands;
    bands.first = first;
    bands.width = width;
    float3 basis[kDetectorBands];
    float3 totals = float3(0);
    for (uint band = 0; band < kDetectorBands; ++band) {
        basis[band] = RgbBasis(first + (float(band) + 0.5f) * width);
        totals += basis[band];
    }
    for (uint color = 0; color < 3; ++color) {
        float4 from[2];
        float4 to[2];
        for (uint band = 0; band < kDetectorBands; ++band) {
            const float center = first + (float(band) + 0.5f) * width;
            const float sum = dot(basis[band], float3(1));
            from[band / 4][band % 4] = center < kVisibleFirstNm || center > kVisibleLastNm
                                           ? LinearUplift(center)[color]
                                       : sum > 0 ? basis[band][color] / sum
                                                 : 1.0f / 3;
            to[band / 4][band % 4] = totals[color] > 0 ? basis[band][color] / totals[color] : 0;
        }
        bands.fromRgb[2 * color] = Float4(from[0]);
        bands.fromRgb[2 * color + 1] = Float4(from[1]);
        bands.toRgb[2 * color] = Float4(to[0]);
        bands.toRgb[2 * color + 1] = Float4(to[1]);
    }
    bands.padding0 = 0;
    bands.padding1 = 0;
    return bands;
}

float SegmentArea(float2 start, float2 end, float low, float high) {
    const float from = max(low, start.x);
    const float to = min(high, end.x);
    if (!(to > from) || !(end.x > start.x)) {
        return 0;
    }
    const float2 t = (float2(from, to) - start.x) / (end.x - start.x);
    return (to - from) * 0.5f * dot(lerp(float2(start.y), float2(end.y), t), float2(1));
}

float BandQe(ArrayRef authored, QeColor color, float low, float high) {
    float area = 0;
    if (authored.count > 0) {
        const Ptr<float2> points = Ptr<float2>(authored.address);
        for (uint i = 1; i < authored.count; ++i) {
            area += SegmentArea(points[i - 1], points[i], low, high);
        }
    } else {
        for (uint i = kGenericQeStarts[uint(color)] + 1; i < kGenericQeStarts[uint(color) + 1];
             ++i) {
            area += SegmentArea(kGenericQe[i - 1], kGenericQe[i], low, high);
        }
    }
    return area / (high - low);
}

[shader("compute")]
[numthreads(kSensorGroup, 1, 1)]
void WriteDetectors(uniform SensorParams p) {
    const uint row = LinearThreadIndex();
    if (row >= p.detectorRows) {
        return;
    }
    DetectorAPIData* data = &p.scene->schemas.detectorAPI;
    DetectorRecord record;
    const float2 range = float2(data.detectorSpectralRange[row].x, data.detectorSpectralRange[row].y);
    const float first = max(range.x, 0.0f);
    const float width = max((range.y - first) / kDetectorBands, 1e-3f);
    record.bands = WriteBandSet(first, width);
    record.pattern = data.detectorPattern[row];
    const ArrayRef curves[] = { data.detectorQeRed[row], data.detectorQeGreen[row],
                                data.detectorQeBlue[row], data.detectorQeClear[row] };
    QeColor colors[kMostDetectorColors] = { QeColor::Red, QeColor::Green, QeColor::Blue };
    switch (DetectorPattern(record.pattern)) {
    case DetectorPattern::Mono:
    case DetectorPattern::Bands:
        colors[0] = QeColor::Clear;
        record.channels = DetectorPattern(record.pattern) == DetectorPattern::Mono ? 1 : kDetectorBands;
        break;
    case DetectorPattern::Rccb:
        colors[1] = QeColor::Clear;
        record.channels = 3;
        break;
    case DetectorPattern::Rccc:
        colors[1] = QeColor::Clear;
        record.channels = 2;
        break;
    default:
        record.channels = 3;
        break;
    }
    const uint colorCount = min(record.channels, kMostDetectorColors);
    for (uint channel = 0; channel < kMostDetectorColors; ++channel) {
        float4 low = float4(0);
        float4 high = float4(0);
        if (channel < colorCount) {
            const QeColor color = colors[channel];
            for (uint band = 0; band < 4; ++band) {
                const float start = first + band * width;
                low[band] = BandQe(curves[uint(color)], color, start, start + width);
                high[band] = BandQe(curves[uint(color)], color, start + 4 * width, start + 5 * width);
            }
        }
        record.qe[2 * channel] = Float4(low);
        record.qe[2 * channel + 1] = Float4(high);
    }
    record.fullWell = max(data.detectorFullWell[row], 0.0f);
    record.readNoise = max(data.detectorReadNoise[row], 0.0f);
    record.darkCurrent = max(data.detectorDarkCurrent[row], 0.0f);
    record.darkCurrentDoubling = max(data.detectorDarkCurrentDoubling[row], 0.0f);
    record.temperature = max(data.detectorTemperature[row], 0.0f);
    record.prnu = max(data.detectorPrnu[row], 0.0f);
    record.dsnu = max(data.detectorDsnu[row], 0.0f);
    record.systemGain = max(data.detectorSystemGain[row], 0.0f);
    record.bitDepth = uint(clamp(data.detectorBitDepth[row], 1, 32));
    record.blackLevel = max(data.detectorBlackLevel[row], 0.0f);
    record.seed = uint(data.detectorSeed[row]);
    record.padding0 = 0;
    record.padding1 = 0;
    record.padding2 = 0;
    p.detectors[row] = record;
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
