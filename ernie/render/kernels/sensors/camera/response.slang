// SPDX-License-Identifier: BSD-3-Clause

import ernie.render.kernels.background.background;
__exported import ernie.render.kernels.render.output;
__exported import ernie.render.kernels.sensors.camera.camera;
__exported import ernie.render.kernels.sensors.iresponse;
import ernie.render.kernels.random;

using namespace ernie;

struct CameraResponse<R : IRender> : IResponse {
    R render;
    HitQuery query;
    SceneData* scene;
    uint product;
    float exposure;
    PathSample surface;
    float3 color;
    uint count;

    static CameraResponse<R> Load(RenderParams params, Capture* capture, HitQuery query) {
        CameraResponse<R> response;
        response.render = R.Load(params, capture, query);
        response.query = query;
        response.scene = params.scene;
        response.product = capture->product;
        response.exposure =
            LoadCameraExposure(params.scene, PrimRow(params.scene, capture->sensor, kSlotCamera));
        response.surface = PathSample.Miss();
        response.color = float3(0);
        response.count = 0;
        return response;
    }

    [mutating]
    void Add(SensorRay ray, inout Rng rng) {
        PathSample sample = PathSample.Miss();
        if (ray.weight > 0) {
            sample = render.TracePath(query, ray.ray, ray.spread, rng);
        }
        if (!sample.hit) {
            const float3 display =
                LoadBackground(scene, product).Radiance(normalize(ray.ray.direction));
            sample.radiance = sample.radiance + DisplayRadiance(display) / max(exposure, 1e-20f);
        }
        const bool occluded = sample.testShadow && query.Occluded(sample.shadow);
        color += (sample.radiance + (occluded ? sample.lit * sample.shadowVisibility : sample.lit)) *
                 exposure;
        if (count == 0) {
            surface = sample;
        }
        ++count;
    }

    void Write(RenderParams params, CapturePixel target) {
        WriteSurface(params, target, surface);
        WriteColor(target, color / float(max(count, 1u)), surface.hit);
    }
}
