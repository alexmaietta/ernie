// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.sensors.camera.camera;
__exported import ernie.render.kernels.sensors.ipattern;

using namespace ernie;

extension CameraModel : IPattern {
    static CameraModel Load(SensorContext context) {
        return CameraModel.Load(context.params.tables,
                                context.capture->plan.sensor);
    }

    uint SampleCount() {
        return 1;
    }

    SensorRay Generate(CapturePixel pixel, uint sample) {
        const float2 resolution = float2(pixel.capture->plan.width, pixel.capture->plan.height);
        const float2 ndc = (float2(pixel.coord) + 0.5f) / resolution * 2.0f - 1.0f;
        const CameraRay cameraRay =
            GenerateRay(CameraSample(float2(ndc.x, -ndc.y), float2(0.5f), 0.5f));
        SensorRay ray;
        ray.ray = cameraRay.ray;
        ray.weight = cameraRay.weight;
        ray.spread = PixelSpread(2 / resolution);
        return ray;
    }
}
