// SPDX-License-Identifier: BSD-3-Clause

import ernie.render.kernels.background.background;
import ernie.render.kernels.basis;
import ernie.render.kernels.integrator.iintegrator;
import ernie.render.kernels.light.distant;
import ernie.render.kernels.light.scatter;
import ernie.render.kernels.material.dispatch;
import ernie.render.kernels.material.thermal;
import ernie.render.kernels.render.outputs;
import ernie.render.kernels.sensors.camera.channels;
import ernie.render.kernels.sensors.camera.pattern;
import ernie.render.kernels.spectrum.band;
import ernie.render.kernels.surface.compute;
import ernie.render.kernels.surface.primvars;
import ernie.render.kernels.random;

using namespace ernie;

static const uint kMinimalAnisotropy = 1;

struct MinimalShade {
    float3 albedo;
    float3 unlit;
    float3 lit;
    bool testShadow;
    Ray shadow;
}

struct MinimalLighting {
    DistantLight<RgbSpectrum> light;
    bool headlight;
    bool castShadows;
    float shadowVisibility;
    float3 ambient;
    float3 ground;
    bool hemispheric;

    static MinimalLighting Load(HitQuery query, uint row) {
        ProductRecord* product = &query.tables->products[row];
        MinimalLighting lighting;
        lighting.castShadows = product.castShadows != 0;
        lighting.shadowVisibility = product.shadowVisibility;
        lighting.ambient = product.ambient.Vector();
        lighting.ground = product.ground.Vector();
        lighting.hemispheric = product.hemispheric != 0;
        const MinimalLightRecord record = query.tables->minimalLights[query.tlas];
        lighting.headlight = record.headlight != 0;
        lighting.light =
            lighting.headlight
                ? DistantLight<RgbSpectrum>(float3(0, 0, 1),
                                            RgbSpectrum.Constant(kHeadlightIrradiance), 1, 1)
                : DistantLight<RgbSpectrum>(record.direction.Vector(),
                                            RgbSpectrum(record.radiance.Vector()), record.diffuse,
                                            record.specular);
        return lighting;
    }

    MinimalShade Shade(Ray ray, SurfacePoint point, OpticalClosure<RgbSpectrum> closure) {
        const Basis frame = Basis(point.shadingNormal);
        const float3 surrounding =
            hemispheric ? lerp(ground, ambient, 0.5f * (point.shadingNormal.z + 1)) : ambient;
        MinimalShade shade;
        shade.albedo = closure.albedo.ToRgb();
        shade.unlit = surrounding * shade.albedo + closure.emission.ToRgb();
        DistantLight<RgbSpectrum> source = light;
        if (headlight) {
            source.direction = -ray.direction;
        }
        const LightSample<RgbSpectrum> toward = source.Sample(point.position, float2(0.5f));
        shade.lit = Scattered<RgbSpectrum, OpticalClosure<RgbSpectrum>>(closure, frame,
                                                                        -ray.direction, toward)
                        .ToRgb();
        shade.testShadow = castShadows && !headlight && any(shade.lit > 0);
        if (shade.testShadow) {
            shade.shadow =
                SpawnRay(point, toward.direction, min(toward.distance, kBvhFar), ray.time);
        }
        return shade;
    }

}

void WriteRates(RenderTables* tables, CapturePixel pixel, BandValues signal) {
    const PlannedCapture plan = pixel.capture->plan;
    const CameraModel camera = CameraModel.Load(tables, plan.sensor, plan.product);
    const ChannelResponse channels =
        ChannelResponse.Load(tables, plan.sensor, camera, float2(plan.width, plan.height));
    Ptr<float> rates = Ptr<float>(pixel.capture->vars[uint(RenderVarSource::ElectronRate)]);
    const uint count = min(channels.channels, kMostDetectorChannels);
    for (uint channel = 0; channel < count; ++channel) {
        rates[pixel.index * count + channel] = channels.Rate(signal, channel);
    }
}

void RenderMinimalPixel<let Mode : MinimalMode>(RenderParams params, CapturePixel pixel) {
    RenderTables* tables = params.tables;
    const PlannedCapture plan = pixel.capture->plan;
    const uint row = PrimRow(tables, plan.product, kSlotRenderProduct);
    const ViewRecord view = tables->views[tables->viewStarts[row] + plan.layer];
    const float2 resolution = float2(plan.width, plan.height);
    CameraRay cameraRay;
    float spread;
    float exposure;
    if ((view.flags & kViewPinhole) != 0) {
        cameraRay =
            PinholeRay(LoadSensorFrame(tables, plan.sensor), view, float2(pixel.coord), resolution);
        spread = view.spread;
        exposure = view.exposure;
    } else {
        const CameraModel camera = CameraModel.Load(tables, plan.sensor, plan.product);
        const PixelSampler sampler = { 1, Hash(pixel.index ^ Hash(plan.sensor)) };
        cameraRay = camera.GenerateRay(sampler.Sample(float2(pixel.coord), resolution, 0));
        spread = camera.PixelSpread(2 / resolution);
        exposure = camera.exposure;
    }
    const Ray ray = cameraRay.ray;
    const bool rated = Wants(pixel, RenderVarSource::ElectronRate);
    if (rated) {
        UseBands(tables, ApiRow(tables, plan.sensor, kApiDetectorAPI));
    }
    const HitQuery query = HitQuery(tables, tables->prims.world[plan.sensor]);
    PrimHit primHit;
    primHit.hit = Miss(ray);
    if (cameraRay.weight > 0) {
        primHit = query.Intersect(ray, kGeometryMask | kTraceCullBackFaces);
    }
    if (!primHit.IsHit()) {
        WritePrimId(pixel, kNoPrim);
        Write(pixel, RenderVarSource::DistanceToCamera, plan.missDistance);
        Write(pixel, RenderVarSource::DistanceToImagePlane, plan.missDistance);
        WriteNormal(pixel, float3(0));
        Write(pixel, RenderVarSource::DiffuseAlbedo, PackSrgb(float3(0), false));
        WriteSegmentation(params, pixel, kNoPrim);
        WriteInstance(params, pixel, SurfacePoint.Miss(), false);
        if (Mode != MinimalMode::Data) {
            const float3 display =
                Background.Load(tables, plan.product).Radiance(normalize(ray.direction));
            WriteColor(pixel, DisplayRadiance(display) / max(exposure, 1e-20f) * exposure, false);
        }
        if (rated) {
            BandValues dark;
            dark.low = float4(0);
            dark.high = float4(0);
            WriteRates(tables, pixel, dark);
        }
        return;
    }
    const SurfacePoint point = ComputeSurfacePoint(query, ray, primHit, spread);
    WritePrimId(pixel, point.prim);
    Write(pixel, RenderVarSource::DistanceToCamera, point.distance);
    if (Wants(pixel, RenderVarSource::DistanceToImagePlane)) {
        Write(pixel, RenderVarSource::DistanceToImagePlane,
              -LoadSensorFrame(tables, plan.sensor).ToLocal(point.position).z);
    }
    WriteNormal(pixel, point.shadingNormal);
    WriteSegmentation(params, pixel, point.part);
    WriteInstance(params, pixel, point, true);
    if (Mode == MinimalMode::Data) {
        Write(pixel, RenderVarSource::DiffuseAlbedo, PackSrgb(float3(0), true));
        WriteColor(pixel, float3(0), true);
        return;
    }

    const ProductRecord product = tables->products[row];
    const MinimalLightRecord sun = tables->minimalLights[query.tlas];
    MinimalLighting lighting;
    lighting.castShadows = product.castShadows != 0;
    lighting.shadowVisibility = product.shadowVisibility;
    lighting.ambient = product.ambient.Vector();
    lighting.ground = product.ground.Vector();
    lighting.hemispheric = product.hemispheric != 0;
    lighting.headlight = sun.headlight != 0;
    lighting.light =
        lighting.headlight
            ? DistantLight<RgbSpectrum>(float3(0, 0, 1), RgbSpectrum.Constant(kHeadlightIrradiance),
                                        1, 1)
            : DistantLight<RgbSpectrum>(sun.direction.Vector(), RgbSpectrum(sun.radiance.Vector()),
                                        sun.diffuse, sun.specular);
    MinimalShade shade;
    if (Mode == MinimalMode::DisplayColor) {
        const float3 color = SurfaceDisplayColor(tables, point.slot, point.element);
        shade = lighting.Shade(ray, point, OpticalClosure<RgbSpectrum>.Diffuse(RgbSpectrum(color)));
    } else {
        MaterialQuery materials = MaterialQuery.Load(query, plan.product);
        materials.minimal = true;
        materials.maxAnisotropy = min(materials.maxAnisotropy, kMinimalAnisotropy);
        ShadeContext context = ShadeContext(materials, point.material, point, -ray.direction);
        if (Mode == MinimalMode::TextureDiffuse) {
            const RgbSpectrum albedo = AlbedoDefinition<RgbSpectrum>(context);
            shade = lighting.Shade(ray, context.point, OpticalClosure<RgbSpectrum>.Diffuse(albedo));
        } else {
            OpticalClosure<RgbSpectrum> closure = OpticalClosure<RgbSpectrum>();
            ShadeDefinition<RgbSpectrum>(context, closure);
            AddThermalEmission<RgbSpectrum>(tables, context.point, closure);
            shade = lighting.Shade(ray, context.point, closure);
        }
    }
    Write(pixel, RenderVarSource::DiffuseAlbedo, PackSrgb(shade.albedo, true));
    float3 lit = shade.lit;
    if (shade.testShadow && query.Occluded(shade.shadow)) {
        lit *= lighting.shadowVisibility;
    }
    const float3 radiance = shade.unlit + lit;
    WriteColor(pixel, radiance * exposure, true);
    if (rated) {
        const float3 forward = -LoadSensorFrame(tables, plan.sensor).z;
        const float falloff = CameraFalloff(ray.direction, forward) * cameraRay.weight;
        BandValues signal = RgbSpectrum(radiance).Bands();
        signal.low *= falloff;
        signal.high *= falloff;
        WriteRates(tables, pixel, signal);
    }
}
