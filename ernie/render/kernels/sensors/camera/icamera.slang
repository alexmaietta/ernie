// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.scene.kernels.scene;
__exported import ernie.render.kernels.trace.ray;

using namespace ernie;

struct CameraSample : IDifferentiable {
    float2 film;
    no_diff float2 lens;
    no_diff float time;

    [Differentiable]
    __init(float2 film, no_diff float2 lens, no_diff float time) {
        this.film = film;
        this.lens = lens;
        this.time = time;
    }
}

struct CameraRay : IDifferentiable {
    Ray ray;
    no_diff float weight;
}

struct FilmPoint : IDifferentiable {
    float2 film;
    no_diff bool visible;
}

interface ICamera : IDifferentiable {
    static This Load(SceneData* scene, InstanceLayout* layout, uint prim);

    [Differentiable]
    CameraRay GenerateRay(CameraSample sample);

    [Differentiable]
    FilmPoint Project(float3 position);

    float ImagePlaneDistance(float3 position);

    float PixelSpread(float2 filmStep);

    float Exposure();
}
