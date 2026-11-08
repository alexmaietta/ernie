// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.background.color;
__exported import ernie.render.kernels.background.gradient;

using namespace ernie;

struct Background {
    ProductRecord* product;
    SchemaId kind;
    float3 fallback;

    static Background Load(RenderTables* tables, uint product) {
        Background background;
        background.product = &tables->products[PrimRow(tables, product, kSlotRenderProduct)];
        background.kind = SchemaId(background.product.backgroundKind);
        background.fallback = background.product.backgroundColor.Vector();
        return background;
    }

    float3 Radiance(float3 direction) {
        switch (kind) {
        case SchemaId::GradientBackground:
            return GradientBackground.Load(product).Radiance(direction);
        case SchemaId::ColorBackground:
            return ColorBackground.Load(product).Radiance(direction);
        default:
            return fallback;
        }
    }
}
