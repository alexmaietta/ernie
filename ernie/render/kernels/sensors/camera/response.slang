// SPDX-License-Identifier: BSD-3-Clause

import ernie.render.kernels.background.background;
__exported import ernie.render.kernels.integrator.iintegrator;
__exported import ernie.render.kernels.render.outputs;
__exported import ernie.render.kernels.sensors.camera.channels;
__exported import ernie.render.kernels.sensors.camera.pattern;
__exported import ernie.render.kernels.sensors.iresponse;
import ernie.render.kernels.random;

using namespace ernie;

struct CameraResponse<I : IIntegrator> : IResponse<CameraPattern> {
    I integrator;
    Background background;
    float exposure;
    SurfacePoint first;
    float3 albedo;
    float3 color;
    uint count;
    RenderTables* tables;
    Capture* capture;
    BandValues signal;
    bool rated;

    static CameraResponse<I> Load(SensorContext context, CameraPattern pattern) {
        RenderTables* tables = context.params.tables;
        const uint detector = ApiRow(tables, context.capture->plan.sensor, kApiDetectorAPI);
        UseBands(tables, detector);
        CameraResponse<I> response;
        response.integrator = I.Load(context);
        response.background = Background.Load(context.query.tables, context.capture->plan.product);
        response.exposure = pattern.camera.exposure;
        response.first = SurfacePoint.Miss();
        response.albedo = float3(0);
        response.color = float3(0);
        response.count = 0;
        response.rated = context.capture->vars[uint(RenderVarSource::ElectronRate)] != 0;
        response.signal.low = float4(0);
        response.signal.high = float4(0);
        response.tables = tables;
        response.capture = context.capture;
        return response;
    }

    [mutating]
    void Add(SensorRay ray, inout Rng rng) {
        PathResult<I.Spectrum> result = PathResult<I.Spectrum>.Miss();
        if (ray.weight > 0) {
            result = integrator.Trace(ray.ray, ray.spread, rng);
        }
        float3 radiance = result.radiance.ToRgb();
        if (rated) {
            const BandValues bands = result.radiance.Bands();
            const float3 forward = -LoadSensorFrame(tables, capture->plan.sensor).z;
            const float falloff = CameraFalloff(ray.ray.direction, forward) * ray.weight;
            signal.low += bands.low * falloff;
            signal.high += bands.high * falloff;
        }
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

    void Write(SensorContext context, CameraPattern pattern, CapturePixel pixel) {
        const bool hit = first.prim != kNoPrim;
        WriteSurface(context.params, pattern.camera, pixel, hit);
        WriteColor(pixel, color / float(max(count, 1u)), hit);
        if (rated) {
            const PlannedCapture plan = context.capture->plan;
            const ChannelResponse channels = ChannelResponse.Load(
                context.params.tables, plan.sensor, pattern.camera, float2(plan.width, plan.height));
            BandValues mean;
            mean.low = signal.low / float(max(count, 1u));
            mean.high = signal.high / float(max(count, 1u));
            Ptr<float> rates = Ptr<float>(pixel.capture->vars[uint(RenderVarSource::ElectronRate)]);
            const uint count = min(channels.channels, kMostDetectorChannels);
            for (uint channel = 0; channel < count; ++channel) {
                rates[pixel.index * count + channel] = channels.Rate(mean, channel);
            }
        }
    }

    void WriteSurface(RenderParams params, CameraModel camera, CapturePixel pixel, bool hit) {
        WritePrimId(pixel, hit ? first.prim : kNoPrim);
        if (!hit) {
            Write(pixel, RenderVarSource::DistanceToCamera, pixel.capture->plan.missDistance);
            Write(pixel, RenderVarSource::DistanceToImagePlane, pixel.capture->plan.missDistance);
            WriteNormal(pixel, float3(0));
            Write(pixel, RenderVarSource::DiffuseAlbedo, PackSrgb(float3(0), false));
            WriteSegmentation(params, pixel, kNoPrim);
            WriteInstance(params, pixel, first, false);
            return;
        }
        Write(pixel, RenderVarSource::DistanceToCamera, first.distance);
        Write(pixel, RenderVarSource::DistanceToImagePlane,
              camera.ImagePlaneDistance(first.position));
        WriteNormal(pixel, first.shadingNormal);
        Write(pixel, RenderVarSource::DiffuseAlbedo, PackSrgb(albedo, true));
        WriteSegmentation(params, pixel, first.prim);
        WriteInstance(params, pixel, first, true);
    }
}
