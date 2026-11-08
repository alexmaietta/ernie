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
        RenderTables* tables = context.params.tables;
        Capture* capture = context.capture;
        const LidarRecord record = tables->lidars[PrimRow(tables, capture->sensor, kSlotLidar)];
        LidarPattern pattern;
        pattern.frame = LoadSensorFrame(tables, capture->sensor);
        pattern.rotary = record.rotary != 0;
        pattern.azimuthStart = record.sweep.x;
        pattern.azimuthStep = (record.sweep.y - record.sweep.x) / float(max(capture->width, 1u));
        pattern.elevations = record.elevations;
        pattern.azimuths = record.azimuths;
        pattern.range = record.range.Vector();
        pattern.divergence = record.divergence.Vector();
        pattern.samples = record.samples;
        pattern.latticeStep = record.latticeStep;
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
