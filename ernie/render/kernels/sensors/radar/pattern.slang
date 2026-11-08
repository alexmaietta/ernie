// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.sensors.ipattern;

using namespace ernie;

struct RadarPattern : IPattern {
    SensorFrame frame;
    float2 fieldOfView;
    float2 resolution;
    float2 range;

    static RadarPattern Load(SensorContext context) {
        RenderTables* tables = context.params.tables;
        Capture* capture = context.capture;
        const RadarRecord record = tables->radars[PrimRow(tables, capture->sensor, kSlotRadar)];
        RadarPattern pattern;
        pattern.frame = LoadSensorFrame(tables, capture->sensor);
        pattern.fieldOfView = record.fieldOfView.Vector();
        pattern.resolution = float2(capture->width, capture->height);
        pattern.range = record.range.Vector();
        return pattern;
    }

    uint SampleCount() {
        return 1;
    }

    SensorRay Generate(CapturePixel pixel, uint sample) {
        const float2 cell = (float2(pixel.coord) + 0.5f) / resolution;
        const float azimuth = (0.5f - cell.x) * fieldOfView.x;
        const float elevation = (0.5f - cell.y) * fieldOfView.y;
        SensorRay ray;
        ray.ray = Ray(frame.position, normalize(frame.ToWorld(SphericalDirection(azimuth, elevation))),
                      range.x, range.y, 0);
        ray.weight = 1;
        ray.spread = 0;
        return ray;
    }
}
