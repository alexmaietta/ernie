// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.material.closure;
__exported import ernie.render.kernels.material.context;

interface IMaterialDefinition<Spectrum : ISpectrum> {
    static void Shade(inout ShadeContext context, inout OpticalClosure<Spectrum> closure);

    static Spectrum Albedo(inout ShadeContext context);
}
