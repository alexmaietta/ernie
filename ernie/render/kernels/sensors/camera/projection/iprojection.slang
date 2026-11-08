// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.sensors.frame;

using namespace ernie;

struct LensRay : IDifferentiable {
    float3 origin;
    float3 direction;
    no_diff bool visible;
}

interface IProjection : IDifferentiable {
    static This Load(RenderTables* tables, uint row);

    [Differentiable]
    LensRay Trace(float2 film);

    [Differentiable]
    float2 Image(float3 local);

    float Distance(float3 local);

    float Spread(float2 filmStep);
}
