// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.background.color;
__exported import ernie.render.kernels.background.gradient;

using namespace ernie;

IBackground LoadBackground(SceneData* scene, uint product) {
    const uint row = ApiRow(scene, product, kApiBackgroundAPI);
    BackgroundAPIData* data = &scene->schemas.backgroundAPI;
    const ArrayRef targets = data.background[row];
    const uint prim = targets.count > 0 ? Ptr<uint>(targets.address)[0] : kNoPrim;
    const bool alive = prim < scene->primCount && (scene->prims.flags[prim] & kPrimAlive) != 0;
    switch (alive ? SchemaId(scene->prims.schema[prim]) : SchemaId::None) {
    case SchemaId::GradientBackground:
        return GradientBackground.Load(scene, prim);
    case SchemaId::ColorBackground:
        return ColorBackground.Load(scene, prim);
    default:
        return ColorBackground(data.backgroundColor[row].Vector());
    }
}
