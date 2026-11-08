// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.sensors.camera.camera;
__exported import ernie.render.kernels.sensors.ipattern;
import ernie.render.kernels.random;

using namespace ernie;

static const float2 kR2Steps = float2(0.7548776662f, 0.5698402910f);

struct PixelSampler {
    uint samples;
    uint seed;

    float2 Stratified(uint sample, uint dimension) {
        const uint shift = Hash(seed ^ Hash(dimension));
        const float2 offset = float2(shift >> 8, Hash(shift) >> 8) * (1.0f / 16777216.0f);
        return frac(offset + 0.5f + float(sample) * kR2Steps);
    }

    CameraSample Sample(float2 coord, float2 resolution, uint sample) {
        if (samples <= 1) {
            const float2 ndc = (coord + 0.5f) / resolution * 2.0f - 1.0f;
            return CameraSample(float2(ndc.x, -ndc.y), float2(0.5f), 0.5f,
                                coord.y / max(resolution.y - 1, 1.0f));
        }
        const float2 ndc = (coord + Stratified(sample, 0)) / resolution * 2.0f - 1.0f;
        const float time = frac(Stratified(sample, 2).x + float(sample) / float(samples));
        return CameraSample(float2(ndc.x, -ndc.y), Stratified(sample, 1), time,
                            coord.y / max(resolution.y - 1, 1.0f));
    }
}

struct CameraPattern : IPattern {
    CameraModel camera;
    uint samples;

    static CameraPattern Load(SensorContext context) {
        RenderTables* tables = context.params.tables;
        const PlannedCapture plan = context.capture->plan;
        const ProductRecord product = tables->products[PrimRow(tables, plan.product, kSlotRenderProduct)];
        CameraPattern pattern;
        pattern.camera = CameraModel.Load(tables, plan.sensor, plan.product);
        pattern.samples =
            RenderMode(product.renderMode) == RenderMode::RealTime ? max(product.pixelSamples, 1u) : 1;
        return pattern;
    }

    uint SampleCount() {
        return samples;
    }

    SensorRay Generate(CapturePixel pixel, uint sample) {
        const float2 resolution = float2(pixel.capture->plan.width, pixel.capture->plan.height);
        const PixelSampler sampler = { samples, Hash(pixel.index ^ Hash(pixel.capture->plan.sensor)) };
        const CameraRay cameraRay =
            camera.GenerateRay(sampler.Sample(float2(pixel.coord), resolution, sample));
        SensorRay ray;
        ray.ray = cameraRay.ray;
        ray.weight = cameraRay.weight;
        ray.spread = camera.PixelSpread(2 / resolution);
        return ray;
    }
}
