// SPDX-License-Identifier: BSD-3-Clause

import ernie.render.kernels.background.background;
__exported import ernie.render.kernels.integrator.iintegrator;
__exported import ernie.render.kernels.render.outputs;
__exported import ernie.render.kernels.sensors.camera.pattern;
__exported import ernie.render.kernels.sensors.iresponse;
import ernie.render.kernels.random;

using namespace ernie;

struct CameraResponse<I : IIntegrator> : IResponse<CameraModel> {
    I integrator;
    Background background;
    float exposure;
    SurfacePoint first;
    float3 albedo;
    float3 color;
    uint count;

    static CameraResponse<I> Load(SensorContext context, CameraModel camera) {
        CameraResponse<I> response;
        response.integrator = I.Load(context);
        response.background = Background.Load(context.query.tables, context.capture->product);
        response.exposure = camera.exposure;
        response.first = SurfacePoint.Miss();
        response.albedo = float3(0);
        response.color = float3(0);
        response.count = 0;
        return response;
    }

    [mutating]
    void Add(SensorRay ray, inout Rng rng) {
        PathResult<I.Spectrum> result = PathResult<I.Spectrum>.Miss();
        if (ray.weight > 0) {
            result = integrator.Trace(ray.ray, ray.spread, rng);
        }
        float3 radiance = result.radiance.ToRgb();
        if (!result.Hit()) {
            const float3 display = background.Radiance(normalize(ray.ray.direction));
            radiance += DisplayRadiance(display) / max(exposure, 1e-20f);
        }
        color += radiance * exposure;
        if (count == 0) {
            first = result.point;
            albedo = result.albedo;
        }
        ++count;
    }

    void Write(SensorContext context, CameraModel camera, CapturePixel pixel) {
        const bool hit = first.prim != kNoPrim;
        WriteSurface(context.params, camera, pixel, hit);
        WriteColor(pixel, color / float(max(count, 1u)), hit);
    }

    void WriteSurface(RenderParams params, CameraModel camera, CapturePixel pixel, bool hit) {
        WritePrimId(pixel, hit ? first.prim : kNoPrim);
        if (!hit) {
            Write(pixel, RenderVarSource::DistanceToCamera, pixel.capture->missDistance);
            Write(pixel, RenderVarSource::DistanceToImagePlane, pixel.capture->missDistance);
            WriteNormal(pixel, float3(0));
            Write(pixel, RenderVarSource::DiffuseAlbedo, PackSrgb(float3(0), false));
            WriteSegmentation(params, pixel, kNoPrim);
            WriteInstance(params, pixel, first.slot, false);
            return;
        }
        Write(pixel, RenderVarSource::DistanceToCamera, first.distance);
        Write(pixel, RenderVarSource::DistanceToImagePlane,
              camera.ImagePlaneDistance(first.position));
        WriteNormal(pixel, first.shadingNormal);
        Write(pixel, RenderVarSource::DiffuseAlbedo, PackSrgb(albedo, true));
        WriteSegmentation(params, pixel, first.prim);
        WriteInstance(params, pixel, first.slot, true);
    }
}
