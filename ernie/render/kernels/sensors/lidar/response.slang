// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.render.output;
__exported import ernie.render.kernels.sensors.iresponse;
import ernie.render.kernels.sensors.points;
import ernie.render.kernels.random;

using namespace ernie;

struct LidarReturn {
    float3 position;
    float distance;
    float intensity;
    float3 normal;
    uint prim;
}

struct LidarResponse : IResponse {
    HitQuery query;
    SceneData* scene;
    SensorFrame frame;
    ReturnMode mode;
    uint stride;
    uint returns;
    uint found;
    LidarReturn kept[kMostLidarReturns];
    uint samples;
    uint hits;
    float3 positionSum;
    float distanceSum;
    float intensitySum;
    float3 normalSum;
    uint firstPrim;

    static LidarResponse Load(RenderParams params, Capture* capture, HitQuery query) {
        SceneData* scene = params.scene;
        const uint row = PrimRow(scene, capture->sensor, kSlotLidar);
        LidarResponse response;
        response.query = query;
        response.scene = scene;
        response.frame = LoadSensorFrame(scene, params.layout, capture->sensor);
        response.mode = ReturnMode(scene->schemas.lidar.returnMode[row]);
        response.stride = max(capture->returns, 1u);
        response.returns = min(response.stride, kMostLidarReturns);
        response.found = 0;
        for (uint i = 0; i < kMostLidarReturns; ++i) {
            response.kept[i] = { float3(0), 0, 0, float3(0), kNoPrim };
        }
        response.samples = 0;
        response.hits = 0;
        response.positionSum = float3(0);
        response.distanceSum = 0;
        response.intensitySum = 0;
        response.normalSum = float3(0);
        response.firstPrim = kNoPrim;
        return response;
    }

    float Intensity(uint prim, float3 normal, float3 direction) {
        float reflectance = 1;
        bool retroreflective = false;
        const uint material =
            scene->schemas.imageable.computedMaterial[PrimRow(scene, prim, kSlotImageable)];
        if (material != kNoPrim && HasApi(scene, material, kApiNonVisualMaterialAPI)) {
            const uint row = ApiRow(scene, material, kApiNonVisualMaterialAPI);
            reflectance = scene->schemas.nonVisualMaterialAPI.nonvisualLidarReflectance[row];
            retroreflective = scene->schemas.nonVisualMaterialAPI.nonvisualRetroreflective[row] != 0;
        }
        return retroreflective ? reflectance : reflectance * abs(dot(normal, direction));
    }

    bool Before(LidarReturn a, LidarReturn b) {
        switch (mode) {
        case ReturnMode::First:
            return a.distance < b.distance;
        case ReturnMode::Last:
            return a.distance > b.distance;
        default:
            return a.intensity > b.intensity;
        }
    }

    [mutating]
    void Keep(LidarReturn candidate) {
        for (uint i = 0; i < kMostLidarReturns; ++i) {
            if (i >= returns) {
                return;
            }
            if (i >= found) {
                kept[i] = candidate;
                ++found;
                return;
            }
            if (Before(candidate, kept[i])) {
                const LidarReturn displaced = kept[i];
                kept[i] = candidate;
                candidate = displaced;
            }
        }
    }

    [mutating]
    void Add(SensorRay ray, inout Rng rng) {
        ++samples;
        const PrimHit hit = query.Intersect(ray.ray);
        if (!hit.IsHit()) {
            return;
        }
        const SurfacePoint point = ComputeSurfacePoint(query, ray.ray, hit, ray.spread);
        LidarReturn candidate;
        candidate.position = point.position;
        candidate.distance = point.distance;
        candidate.intensity = Intensity(point.prim, point.geometricNormal, ray.ray.direction);
        candidate.normal = point.shadingNormal;
        candidate.prim = point.prim;
        if (mode == ReturnMode::Mean) {
            ++hits;
            positionSum += candidate.position;
            distanceSum += candidate.distance;
            intensitySum += candidate.intensity;
            normalSum += candidate.normal;
            firstPrim = firstPrim == kNoPrim ? candidate.prim : firstPrim;
            return;
        }
        Keep(candidate);
    }

    void Write(RenderParams params, CapturePixel target) {
        LidarReturn results[kMostLidarReturns] = kept;
        uint count = found;
        if (mode == ReturnMode::Mean) {
            count = hits > 0 ? 1 : 0;
            const float spread = float(max(hits, 1u));
            results[0] = { positionSum / spread, distanceSum / spread,
                           intensitySum / float(max(samples, 1u)), normalize(normalSum), firstPrim };
        }
        for (uint i = 0; i < stride; ++i) {
            CapturePixel at = target;
            at.index = target.index * stride + i;
            const bool returned = i < count && i < kMostLidarReturns;
            const LidarReturn point = results[min(i, kMostLidarReturns - 1)];
            WritePoint(at, returned, frame, point.position, point.distance, point.intensity, 0,
                       point.prim);
            if (returned) {
                const LidarReturn result = results[i];
                Write(at, RenderVarSource::DistanceToCamera, result.distance);
                Write(at, RenderVarSource::Intensity, result.intensity);
                Write(at, RenderVarSource::Normal, float4(result.normal, 1));
                WriteSegmentation(params, at, result.prim);
            } else {
                Write(at, RenderVarSource::DistanceToCamera, target.capture->missDistance);
                Write(at, RenderVarSource::Intensity, 0.0f);
                Write(at, RenderVarSource::Normal, float4(0));
                WriteSegmentation(params, at, kNoPrim);
            }
        }
    }
}
