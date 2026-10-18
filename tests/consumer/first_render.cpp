// SPDX-License-Identifier: BSD-3-Clause
/// @file first_render.cpp
/// @brief The README's first C++ render against an installed libernie, as a user would write it;
/// it reports a skip (77) on a machine with no device that ray traces.
#include <ernie/render/renderer.h>
#include <ernie/scene/schema/camera.h>
#include <ernie/scene/schema/geom.h>
#include <ernie/scene/schema/lux.h>
#include <ernie/scene/schema/planet.h>
#include <ernie/scene/schema/render.h>
#include <ernie/scene/tokens.h>
#include <ernie/scene/xform.h>

#include <cstdio>
#include <memory>
#include <stdexcept>

int main() {
    using namespace ernie;
    Scene scene;
    GroundPlane::Define(scene, "/World/Ground");
    Sphere::Define(scene, "/World/Rock");
    SetXform(scene, "/World/Rock", {.translate = glm::dvec3(0.0, 0.0, 1.0)});  // 1 m up
    DistantLight::Define(scene, "/World/Sun");
    scene.Set("/World/Sun", tokens::kTransform, DistantLightTransform({0.4, -0.6, 0.7}));
    Camera::Define(scene, "/World/Camera");
    scene.Set("/World/Camera", tokens::kTransform,
              LookAtTransform({0.0, -8.0, 3.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, 1.0}));
    RenderSettings::Define(scene, "/Render/Settings")
        .SetCamera("/World/Camera")
        .SetResolution({640, 480});

    std::unique_ptr<Renderer> renderer;
    try {
        renderer = std::make_unique<Renderer>();
    } catch (const std::runtime_error& error) {
        std::printf("Skipped: %s\n", error.what());
        return 77;
    }
    Tensor image = renderer->Render(scene);  // 480 x 640 x 4 bytes of LdrColor on the CPU
    if (image.Shape() != std::vector<std::int64_t>{480, 640, 4}) {
        std::fprintf(stderr, "The image is not 480 x 640 x 4\n");
        return 1;
    }
    const auto* bytes = static_cast<const std::uint8_t*>(image.Data());
    if (bytes[4 * (240 * 640 + 320)] < 100) {
        std::fprintf(stderr, "The rock at the image's centre is not lit\n");
        return 1;
    }
    std::printf("Rendered the README's first scene\n");
    return 0;
}
