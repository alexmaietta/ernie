// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.sensors.ipattern;
import ernie.render.kernels.sensors.frame;

using namespace ernie;

struct LidarPattern : IPattern {
    SensorFrame frame;
    bool rotary;
    float azimuthStart;
    float azimuthStep;
    ArrayRef elevations;
    ArrayRef azimuths;
    float2 range;
    float2 divergence;
    uint samples;
    uint latticeStep;

    static LidarPattern Load(SensorContext context) {
        SceneData* scene = context.params.scene;
        Capture* capture = context.capture;
        LidarData* data = &scene->schemas.lidar;
        const uint row = PrimRow(scene, capture->sensor, kSlotLidar);
        const float2 sweep = radians(float2(data.azimuthRange[row].x, data.azimuthRange[row].y));
        LidarPattern pattern;
        pattern.frame = LoadSensorFrame(scene, context.params.tables, capture->sensor);
        pattern.rotary = ScanType(data.scanType[row]) == ScanType::Rotary;
        pattern.azimuthStart = sweep.x;
        pattern.azimuthStep = (sweep.y - sweep.x) / float(max(capture->width, 1u));
        pattern.elevations = data.emitterElevations[row];
        pattern.azimuths = data.emitterAzimuths[row];
        pattern.range = float2(data.range[row].x, data.range[row].y);
        pattern.divergence =
            radians(float2(data.beamDivergence[row].x, data.beamDivergence[row].y));
        pattern.samples = max(data.beamSamples[row], 1u);
        pattern.latticeStep = max(uint(float(pattern.samples) * 0.618034f), 1u);
        for (;;) {
            uint a = pattern.latticeStep;
            uint b = pattern.samples;
            while (b != 0) {
                const uint r = a % b;
                a = b;
                b = r;
            }
            if (a == 1) {
                break;
            }
            ++pattern.latticeStep;
        }
        return pattern;
    }

    uint SampleCount() {
        return samples;
    }

    SensorRay Generate(CapturePixel pixel, uint sample) {
        const uint emitter = pixel.coord.y;
        float azimuth =
            radians(emitter < azimuths.count ? Ptr<float>(azimuths.address)[emitter] : 0.0f);
        if (rotary) {
            azimuth += azimuthStart + azimuthStep * float(pixel.coord.x);
        }
        float elevation = radians(Ptr<float>(elevations.address)[emitter]);
        if (samples > 1) {
            const float2 u = (float2(sample, sample * latticeStep % samples) + 0.5f) /
                             float(samples);
            azimuth += (u.x - 0.5f) * divergence.x;
            elevation += (u.y - 0.5f) * divergence.y;
        }
        SensorRay ray;
        ray.ray = Ray(frame.position, normalize(frame.ToWorld(SphericalDirection(azimuth, elevation))),
                      range.x, range.y, 0);
        ray.weight = 1;
        ray.spread = 0;
        return ray;
    }
}
