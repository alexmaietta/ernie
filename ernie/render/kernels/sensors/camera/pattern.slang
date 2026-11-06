// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.sensors.camera.camera;
__exported import ernie.render.kernels.sensors.ipattern;

using namespace ernie;

struct CameraPattern : IPattern {
    ICamera camera;
    float2 resolution;

    static CameraPattern Load(RenderParams params, Capture* capture) {
        CameraPattern pattern;
        pattern.camera = LoadCamera(params.scene, params.layout, capture->sensor);
        pattern.resolution = float2(capture->width, capture->height);
        return pattern;
    }

    uint SampleCount() {
        return 1;
    }

    SensorRay Generate(CapturePixel target, uint sample) {
        const float2 ndc = (float2(target.pixel) + 0.5f) / resolution * 2.0f - 1.0f;
        const CameraRay cameraRay =
            camera.GenerateRay(CameraSample(float2(ndc.x, -ndc.y), float2(0.5f), 0.5f));
        SensorRay ray;
        ray.ray = cameraRay.ray;
        ray.weight = cameraRay.weight;
        ray.spread = camera.PixelSpread(2 / resolution);
        return ray;
    }
}
