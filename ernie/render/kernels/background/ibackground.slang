// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.scene.kernels.scene;

using namespace ernie;

interface IBackground {
    static This Load(SceneData* scene, uint prim);

    float3 Radiance(float3 direction);
}
