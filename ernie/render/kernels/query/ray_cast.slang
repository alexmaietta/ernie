// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;
import ernie.render.kernels.surface.compute;
#include "ernie/render/kernels/query/ray_cast.h"

using namespace ernie;

float3 Load3(Ptr<float> values, uint ray) {
    return float3(values[3 * ray], values[3 * ray + 1], values[3 * ray + 2]);
}

void Store3(Ptr<float> values, uint ray, float3 value) {
    values[3 * ray] = value.x;
    values[3 * ray + 1] = value.y;
    values[3 * ray + 2] = value.z;
}

struct WorldOffset {
    float3 high;
    float3 low;

    float3 ToWorld(float3 local) {
        return (local + low) + high;
    }
}

bool LoadRay(RayCastParams params, uint index, out HitQuery query, out Ray ray,
             out WorldOffset offset) {
    InstanceLayout* layout = params.tables->instances;
    const uint world = params.worlds != nullptr ? params.worlds[index] : params.world;
    query = HitQuery(params.tables, world == kGlobalWorld || world < layout->worldCount
                                       ? world
                                       : kGlobalWorld);
    const RenderOrigin origin = TlasOrigin(params.tables, query.world);
    offset.high = origin.high.Vector();
    offset.low = origin.low.Vector();
    const float3 start = Load3(params.origins, index);
    const float tMin = params.tMins != nullptr ? params.tMins[index] : params.minDistance;
    const float tMax = params.tMaxes != nullptr ? params.tMaxes[index] : params.maxDistance;
    ray = Ray((start - offset.high) - offset.low, Load3(params.directions, index), tMin, tMax, 0);
    return world == kGlobalWorld || world < layout->worldCount;
}

[shader("compute")]
[numthreads(kRayCastGroup, 1, 1)]
void CastClosest(uniform RayCastParams params) {
    const uint index = LinearThreadIndex();
    if (index >= params.count) {
        return;
    }
    HitQuery query;
    Ray ray;
    WorldOffset offset;
    PrimHit found;
    found.prim = kNoPrim;
    if (LoadRay(params, index, query, ray, offset)) {
        found = query.Intersect(ray);
    }
    const bool hit = found.prim != kNoPrim;
    params.hits[index] = hit ? 1 : 0;
    if (params.distances != nullptr) {
        params.distances[index] = hit ? found.hit.t : kInfinity;
    }
    if (params.prims != nullptr) {
        params.prims[index] = found.prim;
    }
    if (params.normals == nullptr && params.positions == nullptr) {
        return;
    }
    SurfacePoint point = SurfacePoint.Miss();
    if (hit) {
        point = ComputeSurfacePoint(query, ray, found, 0);
    }
    if (params.normals != nullptr) {
        Store3(params.normals, index, point.geometricNormal);
    }
    if (params.positions != nullptr) {
        Store3(params.positions, index, hit ? offset.ToWorld(point.position) : float3(0));
    }
}

[shader("compute")]
[numthreads(kRayCastGroup, 1, 1)]
void CastAny(uniform RayCastParams params) {
    const uint index = LinearThreadIndex();
    if (index >= params.count) {
        return;
    }
    HitQuery query;
    Ray ray;
    WorldOffset offset;
    const bool valid = LoadRay(params, index, query, ray, offset);
    params.hits[index] = valid && query.Occluded(ray) ? 1 : 0;
}
