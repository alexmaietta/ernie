// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.sensors.camera.projection.iprojection;

using namespace ernie;

struct PerspectiveProjection : IProjection {
    float focalLength;

    static PerspectiveProjection Load(RenderTables* tables, uint row) {
        PerspectiveProjection projection;
        projection.focalLength = tables->cameras[row].focalLength;
        return projection;
    }

    [Differentiable]
    LensRay Trace(float2 film) {
        LensRay ray;
        ray.origin = float3(0);
        ray.direction = float3(film / focalLength, -1);
        ray.visible = true;
        return ray;
    }

    [Differentiable]
    float2 Image(float3 local) {
        return local.xy * (focalLength / -local.z);
    }

    float Distance(float3 local) {
        return length(local);
    }

    float Spread(float2 filmStep) {
        return filmStep.y / focalLength;
    }
}
