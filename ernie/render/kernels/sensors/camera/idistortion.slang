// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.sensors.camera.iprojection;

using namespace ernie;

interface IDistortion : IDifferentiable {
    static This Load(SceneData* scene, uint row);

    [Differentiable]
    float2 Distort(float2 ideal) {
        return ideal;
    }

    [Differentiable]
    float2 Undistort(float2 film) {
        return film;
    }
}

struct NoDistortion : IDistortion {
    static NoDistortion Load(SceneData* scene, uint row) {
        return NoDistortion();
    }
}
