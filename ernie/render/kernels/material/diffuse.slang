// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.material.imaterial;
__exported import ernie.render.kernels.material.lambert;
import ernie.render.kernels.material.mapping;
import ernie.render.kernels.surface.primvars;

using namespace ernie;

struct DiffuseMaterial<Spectrum : ISpectrum> : IMaterial<Spectrum> {
    typealias Bsdf = Lambert<Spectrum>;
    typealias Medium = Vacuum<Spectrum>;

    Spectrum color;

    [Differentiable]
    __init(Spectrum color) {
        this.color = color;
    }

    static DiffuseMaterial<Spectrum> Load(MaterialQuery query, uint material,
                                          inout SurfacePoint point) {
        const float3 display = SurfaceDisplayColor(query.hit.tables, point.slot);
        if (PrimSchema(query.hit.tables, point.prim) != SchemaId::Plane) {
            return DiffuseMaterial<Spectrum>(Spectrum.FromRgb(display));
        }
        return DiffuseMaterial<Spectrum>(Spectrum.FromRgb(CheckerColor(query, point, display)));
    }

    [Differentiable]
    Spectrum Albedo(SurfacePoint point) {
        return color;
    }

    [Differentiable]
    Bsdf Surface(SurfacePoint point) {
        return Bsdf(color);
    }

    [Differentiable]
    Spectrum Emission(SurfacePoint point, no_diff float3 outgoing) {
        return Spectrum.Constant(0);
    }

    [Differentiable]
    float Opacity(SurfacePoint point) {
        return 1;
    }

    [Differentiable]
    Medium Volume(SurfacePoint point) {
        return Medium();
    }
}
