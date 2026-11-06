// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.surface.compute;
import ernie.render.kernels.random;

#include "ernie/render/kernels/render/render.h"

using namespace ernie;

static const float kHeadlightIrradiance = kPi;

struct PathSample {
    float3 radiance;
    float3 albedo;
    SurfacePoint point;
    bool hit;
    float3 lit;
    Ray shadow;
    bool testShadow;
    float shadowVisibility;

    static PathSample Miss() {
        PathSample sample;
        sample.radiance = float3(0);
        sample.albedo = float3(0);
        sample.point = SurfacePoint.Miss();
        sample.hit = false;
        sample.lit = float3(0);
        sample.testShadow = false;
        sample.shadowVisibility = 0;
        return sample;
    }
}

interface IRender {
    static This Load(RenderParams params, Capture* capture, HitQuery query);

    PathSample TracePath(HitQuery query, Ray ray, float spread, inout Rng rng);
}
