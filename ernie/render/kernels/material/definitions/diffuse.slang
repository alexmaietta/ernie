// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.material.standard;
import ernie.render.kernels.material.mapping;
import ernie.render.kernels.surface.primvars;

using namespace ernie;

struct DiffuseMaterial : IStandardMaterial {
    static float3 Color(ShadeContext context) {
        if (context.definition == kDefinitionPlaceholder) {
            const int3 cell = int3(floor(context.point.position * 4));
            return ((cell.x + cell.y + cell.z) & 1) != 0 ? float3(0.8f, 0, 0.8f) : float3(0.05f);
        }
        RenderTables* tables = context.materials.hit.tables;
        const float3 display = SurfaceDisplayColor(tables, context.point.slot, context.point.element);
        if (PrimSchema(tables, context.point.prim) != SchemaId::Plane) {
            return display;
        }
        return CheckerColor(context.materials, context.point, display);
    }

    static void Material(inout ShadeContext context, inout SurfaceInputs inputs) {
        inputs.baseColor = SpectralValue.Rgb(Color(context));
        inputs.specularWeight = 0;
    }

    static SpectralValue Albedo(inout ShadeContext context) {
        return SpectralValue.Rgb(Color(context));
    }
}
