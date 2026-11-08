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
        BackgroundAPIData* data = &scene->schemas.backgroundAPI;
        const uint row = ApiRow(scene, product, kApiBackgroundAPI);
        const ArrayRef targets = data.background[row];

        Background background;
        background.scene = scene;
        background.prim = kNoPrim;
        background.kind = SchemaId::None;
        background.fallback = data.backgroundColor[row].Vector();

        if (targets.count > 0) {
            const uint prim = Ptr<uint>(targets.address)[0];
            if (PrimAlive(tables, prim)) {
                background.prim = prim;
                background.kind = PrimSchema(tables, prim);
            }
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
