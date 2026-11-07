// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.material.ibsdf;
__exported import ernie.render.kernels.material.imedium;
__exported import ernie.render.kernels.query.material;
__exported import ernie.render.kernels.surface.point;

interface IMaterial<Spectrum : ISpectrum> : IDifferentiable {
    associatedtype Bsdf : IBsdf<Spectrum>;
    associatedtype Medium : IMedium<Spectrum>;

    static This Load(MaterialQuery query, uint material, inout SurfacePoint point);

    [Differentiable]
    Spectrum Albedo(SurfacePoint point);

    [Differentiable]
    Bsdf Surface(SurfacePoint point);

    [Differentiable]
    Spectrum Emission(SurfacePoint point, no_diff float3 outgoing);

    [Differentiable]
    float Opacity(SurfacePoint point);

    [Differentiable]
    Medium Volume(SurfacePoint point);
}
