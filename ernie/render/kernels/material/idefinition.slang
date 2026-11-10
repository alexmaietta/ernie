// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.material.closure;
__exported import ernie.render.kernels.material.context;

using namespace ernie;

interface IMaterialDefinition<Spectrum : ISpectrum> {
    static void Shade(inout ShadeContext context, inout OpticalClosure<Spectrum> closure);

    static Spectrum Albedo(inout ShadeContext context);
}

Spectrum ClosureAlbedo<Spectrum : ISpectrum, D : IMaterialDefinition<Spectrum>>(
    inout ShadeContext context) {
    OpticalClosure<Spectrum> closure = OpticalClosure<Spectrum>();
    D.Shade(context, closure);
    return closure.albedo;
}

interface ILinkedDefinitions {
    static bool Linked(uint definition);
}

void ChooseDefinition<L : ILinkedDefinitions>(inout ShadeContext context) {
    if (L.Linked(context.definition)) {
        return;
    }
    const MaterialRecord record = context.materials.Record(context.material);
    if (L.Linked(record.previousDefinition)) {
        context.definition = record.previousDefinition;
        context.words = record.previousWords;
    } else {
        context.definition = kDefinitionPlaceholder;
    }
}
