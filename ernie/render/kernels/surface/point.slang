// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.surface.element;
import ernie.device.kernels.device;

using namespace ernie;

struct SurfacePoint : IDifferentiable {
    float3 position;
    float3 shadingNormal;
    no_diff float3 geometricNormal;
    no_diff float distance;
    no_diff uint prim;
    no_diff uint slot;
    no_diff SurfaceElement element;
    no_diff float footprint;
    no_diff float3 stretch;

    static SurfacePoint Miss() {
        SurfacePoint point;
        point.position = float3(0);
        point.shadingNormal = float3(0);
        point.geometricNormal = float3(0);
        point.distance = kInfinity;
        point.prim = kNoPrim;
        point.slot = 0;
        point.element = SurfaceElement.Shape(0);
        point.footprint = 0;
        point.stretch = float3(0);
        return point;
    }
}

Ray SpawnRay(SurfacePoint point, float3 direction, float tMax, float time) {
    const float side = dot(direction, point.geometricNormal) < 0 ? -1.0f : 1.0f;
    const float epsilon = 1e-4f * max(1.0f, length(point.position));
    return Ray(point.position + side * epsilon * point.geometricNormal, direction, 0, tMax, time);
}
