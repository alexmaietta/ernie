// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.render.output;
__exported import ernie.render.kernels.sensors.iresponse;
import ernie.render.kernels.sensors.points;
import ernie.render.kernels.random;

using namespace ernie;

struct RadarResponse : IResponse {
    HitQuery query;
    SceneData* scene;
    InstanceLayout* layout;
    SensorFrame frame;
    float3 sensorVelocity;
    bool hit;
    float3 position;
    float distance;
    float intensity;
    float radialVelocity;
    float3 normal;
    uint prim;

    static RadarResponse Load(RenderParams params, Capture* capture, HitQuery query) {
        RadarResponse response;
        response.query = query;
        response.scene = params.scene;
        response.layout = params.layout;
        response.frame = LoadSensorFrame(params.scene, params.layout, capture->sensor);
        response.sensorVelocity =
            PointVelocity(params.scene, params.layout, capture->sensor, response.frame.position);
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
        const uint material =
            scene->schemas.imageable.computedMaterial[PrimRow(scene, prim, kSlotImageable)];
        if (material != kNoPrim && HasApi(scene, material, kApiNonVisualMaterialAPI)) {
            return scene->schemas.nonVisualMaterialAPI
                .nonvisualRadarBackscatter[ApiRow(scene, material, kApiNonVisualMaterialAPI)];
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
        const float3 velocity = PointVelocity(scene, layout, point.prim, point.position);
        hit = true;
        position = point.position;
        distance = point.distance;
        intensity = Backscatter(point.prim) * abs(dot(point.geometricNormal, ray.ray.direction));
        radialVelocity = dot(velocity - sensorVelocity, ray.ray.direction);
        normal = point.shadingNormal;
        prim = point.prim;
    }

    void Write(RenderParams params, CapturePixel target) {
        WritePoint(target, hit, frame, position, distance, intensity, radialVelocity, prim);
        if (!hit) {
            Write(target, RenderVarSource::DistanceToCamera, target.capture->missDistance);
            Write(target, RenderVarSource::Intensity, 0.0f);
            Write(target, RenderVarSource::RadialVelocity, 0.0f);
            Write(target, RenderVarSource::Normal, float4(0));
            WriteSegmentation(params, target, kNoPrim);
            return;
        }
        Write(target, RenderVarSource::DistanceToCamera, distance);
        Write(target, RenderVarSource::Intensity, intensity);
        Write(target, RenderVarSource::RadialVelocity, radialVelocity);
        Write(target, RenderVarSource::Normal, float4(normal, 1));
        WriteSegmentation(params, target, prim);
    }
}
