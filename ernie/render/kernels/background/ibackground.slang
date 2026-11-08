// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.scene.kernels.scene;

using namespace ernie;

interface IBackground {
    static This Load(ProductRecord* product);

    float3 Radiance(float3 direction);
}
