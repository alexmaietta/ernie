// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.sensors.camera.projection.iprojection;

using namespace ernie;

static const int kAngularSteps = 8;

interface IAngularMap : IDifferentiable {
    [Differentiable]
    float Radius(float angle);

    [Differentiable]
    float Slope(float angle);

    float MaxAngle();
}

struct AngularLens<M : IAngularMap> : IDifferentiable {
    M map;
    no_diff float focalLength;

    [Differentiable]
    float Angle(float radius) {
        float angle = radius / focalLength;
        [MaxIters(kAngularSteps)]
        for (int step = 0; step < kAngularSteps; ++step) {
            angle -= (map.Radius(angle) - radius) / max(map.Slope(angle), 1e-6f);
        }
        return angle;
    }

    [Differentiable]
    LensRay Trace(float2 film) {
        const float radius = length(film);
        const float angle = Angle(radius);
        const float2 across = radius > 0 ? film / radius : float2(0);
        LensRay ray;
        ray.origin = float3(0);
        ray.direction = float3(across * sin(angle), -cos(angle));
        ray.visible = detach(angle) <= map.MaxAngle();
        return ray;
    }

    [Differentiable]
    float2 Image(float3 local) {
        const float across = length(local.xy);
        const float angle = atan2(across, -local.z);
        return across > 0 ? local.xy / across * map.Radius(angle) : float2(0);
    }

    bool Sees(float3 local) {
        return atan2(length(local.xy), -local.z) <= map.MaxAngle();
    }
}
