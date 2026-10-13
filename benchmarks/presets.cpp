// SPDX-License-Identifier: BSD-3-Clause
// The registered presets and output sets. A preset sets what it
// changes on the `kPaths` prims; add one with ERNIE_BENCH_PRESET.

#include "bench.h"

#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/tokens.h"

#include <string>

namespace ernie::bench {

namespace {

void RgbBands(Scene& scene) {
    Camera(scene, kPaths.camera)
        .SetSpectralMode(std::string(tokens::kRgb));
}

/// One cascade of hard shadows, as a shadow ray casts.
void HardShadows(Scene& scene) {
    RenderSettings(scene, kPaths.settings)
        .SetShadowCascades(1)
        .SetShadowResolution(1024)
        .SetShadowSoftness(0.0);
}

}  // namespace

ERNIE_BENCH_PRESET(
    "minimal",
    "One forward pass of diffuse textures lit by the sun, no shadows",
    [](Scene& scene) {
        RgbBands(scene);
        RenderProduct(scene, kPaths.product)
            .SetMinimalMode(std::string(tokens::kTexturedDiffuse));
    });

ERNIE_BENCH_PRESET(
    "lit",
    "The full pipeline in three RGB bands with one cascade of hard "
    "sun shadows",
    [](Scene& scene) {
        RgbBands(scene);
        HardShadows(scene);
    });

ERNIE_BENCH_PRESET(
    "lit-noshadow", "lit without shadows, for their cost",
    [](Scene& scene) {
        RgbBands(scene);
        DistantLight(scene, kPaths.sun).SetShadowEnable(false);
    });

ERNIE_BENCH_PRESET(
    "native",
    "The defaults: 16 spectral bands, soft cascaded shadows, sensor "
    "noise",
    [](Scene& scene) {
        Camera(scene, kPaths.camera)
            .SetSpectralBands(16)
            .SetSensorNoise(true);
    });

ERNIE_BENCH_OUTPUTS("depth", "Distance to the camera",
                    {"DistanceToCamera"});
ERNIE_BENCH_OUTPUTS("rgb", "8-bit colour", {"LdrColor"});
ERNIE_BENCH_OUTPUTS("rgbd", "8-bit colour and distance",
                    {"LdrColor", "DistanceToCamera"});
ERNIE_BENCH_OUTPUTS(
    "gbuffer", "Colour, distance, normals, prim and instance ids",
    {"LdrColor", "DistanceToCamera", "Normal", "PrimId",
     "InstanceId"});
ERNIE_BENCH_OUTPUTS("hdr", "Float colour", {"HdrColor"});

}  // namespace ernie::bench
