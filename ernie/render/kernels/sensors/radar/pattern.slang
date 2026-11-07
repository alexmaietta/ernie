// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.sensors.ipattern;

using namespace ernie;

struct RadarPattern : IPattern {
    SensorFrame frame;
    float2 fieldOfView;
    float2 resolution;
    float2 range;

    static RadarPattern Load(SensorContext context) {
        SceneData* scene = context.params.scene;
        Capture* capture = context.capture;
        RadarData* data = &scene->schemas.radar;
        const uint row = PrimRow(scene, capture->sensor, kSlotRadar);
        RadarPattern pattern;
        pattern.frame = LoadSensorFrame(scene, context.params.layout, capture->sensor);
        pattern.fieldOfView =
            radians(float2(data.fieldOfView[row].x, data.fieldOfView[row].y));
        pattern.resolution = float2(capture->width, capture->height);
        pattern.range = float2(data.range[row].x, data.range[row].y);
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
