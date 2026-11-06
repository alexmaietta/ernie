// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.background.ibackground;

using namespace ernie;

struct ColorBackground : IBackground {
    float3 color;

    __init(float3 color) {
        this.color = color;
    }

    static ColorBackground Load(SceneData* scene, uint prim) {
        const uint row = PrimRow(scene, prim, kSlotColorBackground);
        return ColorBackground(scene->schemas.colorBackground.color[row].Vector());
    }

    float3 Radiance(float3 direction) {
        return color;
    }
}
