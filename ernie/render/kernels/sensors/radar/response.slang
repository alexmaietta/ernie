// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.render.outputs;
__exported import ernie.render.kernels.sensors.radar.pattern;
__exported import ernie.render.kernels.sensors.iresponse;
import ernie.render.kernels.sensors.points;
import ernie.render.kernels.random;
import ernie.render.kernels.surface.compute;

using namespace ernie;

struct RadarResponse : IResponse<RadarPattern> {
    HitQuery query;
    SceneData* scene;
    RenderTables* tables;
    float3 sensorVelocity;
    bool hit;
    float3 position;
    float distance;
    float intensity;
    float radialVelocity;
    float3 normal;
    uint prim;

    static RadarResponse Load(SensorContext context, RadarPattern pattern) {
        RenderParams params = context.params;
        RadarResponse response;
        response.query = context.query;
        response.scene = params.scene;
        response.tables = params.tables;
        response.sensorVelocity = PointVelocity(params.tables,
                                                context.capture->sensor, pattern.frame.position);
        response.hit = false;
        response.position = float3(0);
        response.distance = 0;
        response.intensity = 0;
        response.radialVelocity = 0;
        response.normal = float3(0);
        response.prim = kNoPrim;
        return response;
    }

    float Backscatter(uint prim) {
        const uint material = PrimImageable(tables, prim).material;
        if (material != kNoPrim && HasApi(tables, material, kApiNonVisualMaterialAPI)) {
            return tables->nonVisuals[ApiRow(tables, material, kApiNonVisualMaterialAPI)]
                .radarBackscatter;
        }
        return 1;
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
        intensity = Backscatter(point.prim) * abs(dot(point.geometricNormal, ray.ray.direction));
        radialVelocity = dot(velocity - sensorVelocity, ray.ray.direction);
        normal = point.shadingNormal;
        prim = point.prim;
    }

    void Write(SensorContext context, RadarPattern pattern, CapturePixel pixel) {
        RenderParams params = context.params;
        WritePoint(pixel, hit, pattern.frame, position, distance, intensity, radialVelocity, prim);
        if (!hit) {
            Write(pixel, RenderVarSource::DistanceToCamera, pixel.capture->missDistance);
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
