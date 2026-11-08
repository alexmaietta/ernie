// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.background.color;
__exported import ernie.render.kernels.background.gradient;

using namespace ernie;

struct Background {
    SceneData* scene;
    uint prim;
    SchemaId kind;
    float3 fallback;

    static Background Load(SceneData* scene, RenderTables* tables, uint product) {
        ProductRecord* record = &tables->products[PrimRow(tables, product, kSlotRenderProduct)];
        Background background;
        background.scene = scene;
        background.prim = kNoPrim;
        background.kind = SchemaId::None;
        background.fallback = record.backgroundColor.Vector();
        if (PrimAlive(tables, record.background)) {
            background.prim = record.background;
            background.kind = PrimSchema(tables, record.background);
        }
        return background;
    }

    float3 Radiance(float3 direction) {
        switch (kind) {
        case SchemaId::GradientBackground:
            return GradientBackground.Load(scene, prim).Radiance(direction);
        case SchemaId::ColorBackground:
            return ColorBackground.Load(scene, prim).Radiance(direction);
        default:
            return fallback;
        }
    }
}
