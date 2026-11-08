// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.render.outputs;
__exported import ernie.render.kernels.sensors.radar.pattern;
__exported import ernie.render.kernels.sensors.iresponse;
import ernie.render.kernels.sensors.points;
import ernie.render.kernels.random;
import ernie.render.kernels.surface.compute;
import ernie.render.kernels.material.dispatch;

using namespace ernie;

struct RadarResponse : IResponse<RadarPattern> {
    HitQuery query;
    RenderTables* tables;
    float3 sensorVelocity;
    float frequency;
    bool hit;
    float3 position;
    float distance;
    float intensity;
    float radialVelocity;
    float3 normal;
    uint prim;
    int64_t instance;

    static RadarResponse Load(SensorContext context, RadarPattern pattern) {
        RenderParams params = context.params;
        RadarResponse response;
        response.query = context.query;
        response.tables = params.tables;
        response.sensorVelocity =
            PointVelocity(params.tables, context.capture->plan.sensor, pattern.frame.position);
        response.frequency =
            params.tables->radars[PrimRow(params.tables, context.capture->plan.sensor, kSlotRadar)]
                .frequency;
        response.hit = false;
        response.position = float3(0);
        response.distance = 0;
        response.intensity = 0;
        response.radialVelocity = 0;
        response.normal = float3(0);
        response.prim = kNoPrim;
        response.instance = -1;
        return response;
    }

    float Backscatter(SurfacePoint point, float3 direction) {
        MaterialQuery materials = MaterialQuery(query, 0, 1);
        materials.minimal = true;
        ShadeContext context =
            ShadeContext(materials, point.material, point, -direction);
        OpticalClosure<MonoSpectrum> closure = OpticalClosure<MonoSpectrum>();
        ShadeDefinition<MonoSpectrum>(context, closure);
        RadioClosure radio = closure.radio;
        if (context.material != kNoPrim) {
            const RadioRecord record = tables->radios[context.material];
            if (record.valid != 0) {
                radio.electrical.kind = record.kind;
                radio.electrical.data = record.data.Vector();
                radio.rmsHeight = record.rmsHeight;
                radio.correlationLength = record.correlationLength;
            }
        }
        return closure.radarBackscatter *
               radio.Backscatter(frequency, abs(dot(point.geometricNormal, direction)));
    }

    [mutating]
    void Add(SensorRay ray, inout Rng rng) {
        const PrimHit primHit = query.Intersect(ray.ray);
        if (!primHit.IsHit()) {
            return;
        }
        const SurfacePoint point = ComputeSurfacePoint(query, ray.ray, primHit, ray.spread);
        const float3 velocity = PointVelocity(tables, point.prim, point.position);
        hit = true;
        position = point.position;
        distance = point.distance;
        intensity = Backscatter(point, ray.ray.direction);
        radialVelocity = dot(velocity - sensorVelocity, ray.ray.direction);
        normal = point.shadingNormal;
        prim = point.part;
        instance = HitInstanceId(tables, point);
    }

    void Write(SensorContext context, RadarPattern pattern, CapturePixel pixel) {
        RenderParams params = context.params;
        WritePoint(pixel, hit, pattern.frame, position, distance, intensity, radialVelocity, prim,
                   instance);
        if (!hit) {
            Write(pixel, RenderVarSource::DistanceToCamera, pixel.capture->plan.missDistance);
            Write(pixel, RenderVarSource::Intensity, 0.0f);
            Write(pixel, RenderVarSource::RadialVelocity, 0.0f);
            WriteNormal(pixel, float3(0));
            WriteSegmentation(params, pixel, kNoPrim);
            return;
        }
        Write(pixel, RenderVarSource::DistanceToCamera, distance);
        Write(pixel, RenderVarSource::Intensity, intensity);
        Write(pixel, RenderVarSource::RadialVelocity, radialVelocity);
        WriteNormal(pixel, normal);
        WriteSegmentation(params, pixel, prim);
    }
}
