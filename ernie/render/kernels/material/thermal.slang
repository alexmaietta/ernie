// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.material.closure;
import ernie.render.kernels.surface.primvars;
import ernie.render.kernels.surface.point;

using namespace ernie;

static const float kRendererRadiancePerSpectralRadiance = 683.0f * 106.857f;

float SurfaceTemperature(RenderTables* tables, SurfacePoint point) {
    if (point.prim == kNoPrim) {
        return 0;
    }
    float temperature = tables->temperatures[point.prim];
    if (temperature <= 0) {
        return 0;
    }
    float4 value;
    if (ReadPrimvar(FindPrimvar(tables, point.slot, tables->temperaturePrimvar), point.element,
                    value)) {
        temperature = value.x;
    }
    return temperature;
}

void AddThermalEmission<Spectrum : ISpectrum>(RenderTables* tables, SurfacePoint point,
                                             inout OpticalClosure<Spectrum> closure) {
    const float temperature = SurfaceTemperature(tables, point);
    if (temperature <= 0) {
        return;
    }
    const Spectrum emissivity = closure.emissivityAuthored
                                    ? closure.emissivity
                                    : Spectrum.Constant(1) - closure.albedo;
    closure.emission = closure.emission +
                       emissivity.Multiply(Spectrum.Blackbody(temperature))
                           .Scale(kRendererRadiancePerSpectralRadiance);
}
