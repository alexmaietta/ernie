// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.query.hit;
__exported import ernie.render.kernels.spectrum.spectrum;

using namespace ernie;

struct LightContext {
    RenderTables* tables;
    LightLayout* lights;
    uint tlas;

    __init(RenderTables* tables, LightLayout* lights, uint tlas) {
        this.tables = tables;
        this.lights = lights;
        this.tlas = tlas;
    }
}

struct LightSample<Spectrum : ISpectrum> : IDifferentiable {
    float3 direction;
    no_diff float distance;
    Spectrum radiance;
    no_diff float pdf;
    no_diff bool delta;
    float diffuse;
    float specular;
}

interface ILight<Spectrum : ISpectrum> : IDifferentiable {
    static This Load(LightContext context, LightRecord record);

    [Differentiable]
    LightSample<Spectrum> Sample(float3 position, no_diff float2 u);

    [Differentiable]
    Spectrum Radiance(float3 position, float3 direction);

    float Pdf(float3 position, float3 direction);
}
