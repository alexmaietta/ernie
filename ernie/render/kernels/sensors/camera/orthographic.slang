// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.sensors.camera.iprojection;

using namespace ernie;

static const float kOrthographicApertureUnits = 0.1f;

struct OrthographicProjection : IProjection {
    static OrthographicProjection Load(SceneData* scene, uint row) {
        return OrthographicProjection();
    }

    [Differentiable]
    LensRay Trace(float2 film) {
        LensRay ray;
        ray.origin = float3(film * kOrthographicApertureUnits, 0);
        ray.direction = float3(0, 0, -1);
        return ray;
    }

    [Differentiable]
    float2 Image(float3 local) {
        return local.xy / kOrthographicApertureUnits;
    }

    float Distance(float3 local) {
        return -local.z;
    }

    float Spread(float2 filmStep) {
        return 0;
    }
}
