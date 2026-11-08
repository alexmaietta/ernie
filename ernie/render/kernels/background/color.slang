// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.background.ibackground;

using namespace ernie;

struct ColorBackground : IBackground {
    float3 color;

    __init(float3 color) {
        this.color = color;
    }

    static ColorBackground Load(ProductRecord* product) {
        return ColorBackground(product.backgroundLow.Vector());
    }

    float3 Radiance(float3 direction) {
        return color;
    }
}
