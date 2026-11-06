// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.background.ibackground;

using namespace ernie;

struct GradientBackground : IBackground {
    float3 horizon;
    float3 zenith;

    __init(float3 horizon, float3 zenith) {
        this.horizon = horizon;
        this.zenith = zenith;
    }

    static GradientBackground Load(SceneData* scene, uint prim) {
        GradientBackgroundData* data = &scene->schemas.gradientBackground;
        const uint row = PrimRow(scene, prim, kSlotGradientBackground);
        return GradientBackground(data.horizonColor[row].Vector(), data.zenithColor[row].Vector());
    }

    float3 Radiance(float3 direction) {
        return lerp(horizon, zenith, saturate(direction.z));
    }
}
