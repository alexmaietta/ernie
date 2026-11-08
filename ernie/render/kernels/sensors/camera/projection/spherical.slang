// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.sensors.camera.projection.iprojection;

using namespace ernie;

struct SphericalProjection : IProjection {
    float2 aperture;
    float2 offset;

    static SphericalProjection Load(RenderTables* tables, uint row) {
        const CameraRecord record = tables->cameras[row];
        SphericalProjection projection;
        projection.aperture = record.aperture.Vector();
        projection.offset = record.offset.Vector();
        return projection;
    }

    [Differentiable]
    LensRay Trace(float2 film) {
        const float2 angles = (film - offset) / (0.5f * aperture) * float2(kPi, 0.5f * kPi);
        LensRay ray;
        ray.origin = float3(0);
        ray.direction = float3(sin(angles.x) * cos(angles.y), sin(angles.y),
                               -cos(angles.x) * cos(angles.y));
        ray.visible = true;
        return ray;
    }

    [Differentiable]
    float2 Image(float3 local) {
        const float azimuth = atan2(local.x, -local.z);
        const float elevation = atan2(local.y, length(local.xz));
        return float2(azimuth / kPi, elevation / (0.5f * kPi)) * 0.5f * aperture + offset;
    }

    float Distance(float3 local) {
        return length(local);
    }

    float Spread(float2 filmStep) {
        return filmStep.y * kPi / aperture.y;
    }
}
