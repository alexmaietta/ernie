// SPDX-License-Identifier: BSD-3-Clause
/// @file renderer_test.cpp
/// @brief The README's first render, as written there: a rock on the ground under the sun, seen
/// by a camera the render settings name.
#include "ernie/render/renderer.h"
#include "ernie/scene/schema/camera.h"
#include "ernie/scene/schema/geom.h"
#include "ernie/scene/schema/lux.h"
#include "ernie/scene/schema/planet.h"
#include "ernie/scene/schema/render.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"

#include "tests/device/common.h"

#include <gtest/gtest.h>

#include <array>
#include <memory>

namespace ernie {
namespace {

/// @return A renderer, or none without a device that ray traces; no Vulkan at all fails where CI
/// requires it.
std::unique_ptr<Renderer> MakeRenderer() {
    const std::unique_ptr<DeviceContext> probe = test::MakeContext();
    if (!probe || !probe->Capabilities().rayQuery) {
        return nullptr;
    }
    return std::make_unique<Renderer>(RendererOptions{.validation = true});
}

/// @return The RGBA bytes at a pixel of an image `width` pixels across.
std::array<std::uint8_t, 4> Pixel(const Tensor& image, std::int64_t x, std::int64_t y) {
    const auto* bytes = static_cast<const std::uint8_t*>(image.Data());
    const std::int64_t at = 4 * (y * image.Shape()[1] + x);
    return {bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]};
}

TEST(Renderer, RendersTheReadmesFirstScene) {
    const std::unique_ptr<Renderer> renderer = MakeRenderer();
    if (!renderer) {
        GTEST_SKIP() << "No Vulkan device that ray traces";
    }
    // As the README's C++ example has it.
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

    const Tensor image = renderer->Render(scene);
    ASSERT_EQ(image.Shape(), (std::vector<std::int64_t>{480, 640, 4}));
    EXPECT_EQ(image.Type(), DType::UInt8);
    // The camera looks 14 degrees down at the rock's centre, its view 17 degrees high: the ground
    // fills the rest, lit more steeply near the camera than far off.
    const std::array<std::uint8_t, 4> rock = Pixel(image, 320, 240);
    const std::array<std::uint8_t, 4> near = Pixel(image, 320, 470);
    const std::array<std::uint8_t, 4> far = Pixel(image, 320, 5);
    EXPECT_GT(rock[0], 100) << "the rock, lit";
    EXPECT_GT(near[0], far[0]) << "the ground, darker where the headlight grazes it";
    EXPECT_GT(far[0], 0) << "still the ground at the top";
    EXPECT_EQ(rock[3], 255);
    test::ExpectClean(renderer->Context());

    // Again, after the rock moves out of view: only what changed is redone.
    SetXform(scene, "/World/Rock", {.translate = glm::dvec3(50.0, 0.0, 1.0)});
    const std::array<std::uint8_t, 4> moved = Pixel(renderer->Render(scene), 320, 240);
    EXPECT_NE(moved, rock) << "the ground behind where the rock was";
    test::ExpectClean(renderer->Context());
}

TEST(Renderer, AsksForRenderSettingsWithACamera) {
    const std::unique_ptr<Renderer> renderer = MakeRenderer();
    if (!renderer) {
        GTEST_SKIP() << "No Vulkan device that ray traces";
    }
    Scene scene;
    EXPECT_THROW(renderer->Render(scene), std::invalid_argument);
    RenderSettings::Define(scene, "/Render/Settings");
    EXPECT_THROW(renderer->Render(scene), std::invalid_argument);
}

}  // namespace
}  // namespace ernie
