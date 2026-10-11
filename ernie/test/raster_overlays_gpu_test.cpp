// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/offscreen_target.h"
#include "ernie/raster/delegate.h"
#include "ernie/scene/camera.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/overlay.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/sensor_filters.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/wide_view.h"
#include "support/delegate_scenes.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

using ernie::MarkerShape;
using ernie::OverlayDepth;
using ernie::OverlayStyle;
using ernie::RasterDelegate;
using ernie::Scene;
using ernie::Tensor;
using namespace ernie::testing;

namespace {

/// MeshScene drawing the overlays `overlays` (`ernie:overlays`) over
/// its image.
Scene OverlayScene(const std::string& overlays = "*") {
    Scene scene = MeshScene();
    ernie::RenderSettings(scene, "/Render/Settings")
        .SetOverlays(overlays);
    return scene;
}

/// The RGB of the texel showing world point `p`.
glm::ivec3 Rgb(const Scene& scene, const Tensor& image, glm::vec3 p) {
    const std::uint8_t* texel = TexelAt(scene, image, p);
    return {texel[0], texel[1], texel[2]};
}

/// A white line across the empty space below MeshScene's meshes.
void DrawCrossing(Scene& scene, const std::string& name) {
    OverlayStyle style;
    style.width = 3.0f;
    scene.GetOverlay(name).Line({-20.0, -15.0, 0.0},
                                {20.0, -15.0, 0.0}, style);
}

}  // namespace

TEST_F(RasterDelegateTest,
       OverlaysDrawOnlyIntoProductsThatAskForThem) {
    RasterDelegate delegate(*context_);
    Scene plain = MeshScene();
    const std::vector<std::uint8_t> expected =
        Bytes(delegate.Render(plain));
    DrawCrossing(plain, "debug");
    EXPECT_EQ(Bytes(delegate.Render(plain)), expected)
        << "no ernie:overlays: an observation";

    Scene scene = OverlayScene();
    DrawCrossing(scene, "debug");
    const Tensor image = delegate.Render(scene);
    EXPECT_EQ(Rgb(scene, image, {0.0f, -15.0f, 0.0f}),
              glm::ivec3(255));
    EXPECT_EQ(Rgb(scene, image, {0.0f, 15.0f, 0.0f}), glm::ivec3(0))
        << "away from the line";

    scene.GetOverlay("debug").SetVisible(false);
    EXPECT_EQ(Bytes(delegate.Render(scene)), expected) << "hidden";
    scene.GetOverlay("debug").SetVisible(true).Clear();
    EXPECT_EQ(Bytes(delegate.Render(scene)), expected) << "cleared";
}

TEST_F(RasterDelegateTest, ProductsChooseOverlaysByName) {
    RasterDelegate delegate(*context_);
    Scene scene = OverlayScene("other, wanted");
    DrawCrossing(scene, "wanted");
    OverlayStyle style;
    style.width = 3.0f;
    scene.GetOverlay("unwanted")
        .Line({-20.0, 15.0, 0.0}, {20.0, 15.0, 0.0}, style);
    const Tensor image = delegate.Render(scene);
    EXPECT_EQ(Rgb(scene, image, {0.0f, -15.0f, 0.0f}),
              glm::ivec3(255));
    EXPECT_EQ(Rgb(scene, image, {0.0f, 15.0f, 0.0f}), glm::ivec3(0));
}

TEST_F(RasterDelegateTest, OverlaysMeetSurfacesByTheirDepthMode) {
    // A line 1 below the red quad (x -14 to -10, at z 0), along its
    // middle.
    RasterDelegate delegate(*context_);
    const Scene plain = MeshScene();
    const glm::ivec3 quad =
        Rgb(plain, delegate.Render(plain), {-12.0f, 0.0f, 0.0f});
    const auto through = [&](OverlayDepth depth) {
        Scene scene = OverlayScene();
        OverlayStyle style;
        style.width = 3.0f;
        style.depth = depth;
        scene.GetOverlay("debug").Line({-17.0, 0.0, -1.0},
                                       {-7.0, 0.0, -1.0}, style);
        const Tensor image = delegate.Render(scene);
        return std::pair(Rgb(scene, image, {-12.0f, 0.0f, 0.0f}),
                         Rgb(scene, image, {-8.0f, 0.0f, 0.0f}));
    };
    const auto [hidden, outside] = through(OverlayDepth::Test);
    EXPECT_EQ(hidden, quad) << "behind the quad";
    EXPECT_EQ(outside, glm::ivec3(255))
        << "past the quad, over the background";
    const auto [xray, xrayOutside] = through(OverlayDepth::XRay);
    EXPECT_GT(xray.g, quad.g + 40);
    EXPECT_LT(xray.g, 200) << "faded behind the quad";
    EXPECT_EQ(xrayOutside, glm::ivec3(255));
    const auto [always, alwaysOutside] =
        through(OverlayDepth::Always);
    EXPECT_EQ(always, glm::ivec3(255));
    EXPECT_EQ(alwaysOutside, glm::ivec3(255));
}

TEST_F(RasterDelegateTest, OverlaysFollowACubeCamerasLens) {
    // MeshScene through an equidistant fisheye reaching 100 degrees
    // off axis, on cube faces: a line 60 below the camera from 68
    // degrees off axis on one side to 68 on the other, across the
    // faces' seams, lands where the lens puts each of its points,
    // bending as the lens does.
    RasterDelegate delegate(*context_);
    Scene scene = OverlayScene();
    const double aperture = 20.0;
    const double corner = std::sqrt(2.0) * 0.5 * kSize;
    ernie::Camera(scene, "/World/Camera")
        .SetLensProjection(ernie::tokens::kFisheye)
        .SetVerticalAperture(aperture)
        .SetHorizontalAperture(aperture)
        .SetFocalLength(aperture * corner / (100.0 * M_PI / 180.0) /
                        kSize);
    const ernie::CameraParams camera =
        ernie::CameraParamsOf(scene, "/World/Camera", kSize, kSize);
    ASSERT_EQ(ernie::CubeFacesOf(camera), ernie::kCubeFaces);
    OverlayStyle style;
    style.width = 3.0f;
    style.depth = OverlayDepth::Always;
    scene.GetOverlay("debug").Line({-150.0, -15.0, 0.0},
                                   {150.0, -15.0, 0.0}, style);
    const Tensor image = delegate.Render(scene);
    // The camera looks down -z from 60 with +y up: its frame is the
    // world's.
    const double f = camera.FocalLengthPixels();
    const glm::dvec2 centre = camera.PrincipalPoint();
    const auto pixelOf = [&](glm::dvec3 p) {
        const glm::dvec3 v = p - glm::dvec3(0.0, 0.0, 60.0);
        const double theta = std::acos(-v.z / glm::length(v));
        const glm::dvec2 across =
            glm::normalize(glm::dvec2(v.x, v.y));
        return centre + theta * f * glm::dvec2(across.x, -across.y);
    };
    const auto texel = [&](glm::dvec2 pixel) {
        const auto col = static_cast<std::size_t>(pixel.x);
        const auto row = static_cast<std::size_t>(pixel.y);
        return image.As<std::uint8_t>() + 4 * (row * kSize + col);
    };
    int lit = 0;
    for (double x = -140.0; x <= 140.0; x += 10.0) {
        const glm::dvec2 on = pixelOf({x, -15.0, 0.0});
        EXPECT_GT(texel(on)[0], 200)
            << "x = " << x << " at " << on.x << ", " << on.y;
        lit += texel(on)[0] > 200 ? 1 : 0;
        // Its mirror image across the axis stays dark.
        const glm::dvec2 off = pixelOf({x, 15.0, 0.0});
        EXPECT_LT(texel(off)[0], 60) << "x = " << x;
    }
    EXPECT_GT(lit, 20);
}

TEST_F(RasterDelegateTest,
       CubeCamerasOverlaysDrawOverAnyPresentedVar) {
    // As OverlaysFollowACubeCamerasLens, presenting DistanceToCamera
    // to UNORM and sRGB targets: the line lands where the lens puts
    // it in the colour it has on LdrColor, and leaves the rest of the
    // presented image as it was.
    Scene scene = OverlayScene();
    const double aperture = 20.0;
    const double corner = std::sqrt(2.0) * 0.5 * kSize;
    ernie::Camera(scene, "/World/Camera")
        .SetLensProjection(ernie::tokens::kFisheye)
        .SetVerticalAperture(aperture)
        .SetHorizontalAperture(aperture)
        .SetFocalLength(aperture * corner / (100.0 * M_PI / 180.0) /
                        kSize);
    const ernie::CameraParams camera =
        ernie::CameraParamsOf(scene, "/World/Camera", kSize, kSize);
    ASSERT_EQ(ernie::CubeFacesOf(camera), ernie::kCubeFaces);
    OverlayStyle style;
    style.width = 3.0f;
    style.depth = OverlayDepth::Always;
    style.color = {0.2f, 0.6f, 1.0f, 1.0f};
    scene.GetOverlay("debug").Line({-150.0, -15.0, 0.0},
                                   {150.0, -15.0, 0.0}, style);
    const double f = camera.FocalLengthPixels();
    const glm::dvec2 centre = camera.PrincipalPoint();
    const auto pixelOf = [&](glm::dvec3 p) {
        const glm::dvec3 v = p - glm::dvec3(0.0, 0.0, 60.0);
        const double theta = std::acos(-v.z / glm::length(v));
        const glm::dvec2 across =
            glm::normalize(glm::dvec2(v.x, v.y));
        return centre + theta * f * glm::dvec2(across.x, -across.y);
    };
    const auto offset = [](glm::dvec2 pixel) {
        return 4 * (static_cast<std::size_t>(pixel.y) * kSize +
                    static_cast<std::size_t>(pixel.x));
    };
    RasterDelegate ldr(*context_);
    const std::vector<std::uint8_t> onLdr = Bytes(ldr.Render(scene));
    for (const VkFormat format :
         {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SRGB}) {
        RasterDelegate delegate(*context_, format);
        const auto present = [&](bool overlays) {
            scene.GetOverlay("debug").SetVisible(overlays);
            delegate.Update(scene, "/Render/Settings",
                            {kSize, kSize});
            ernie::OffscreenTarget target(*context_, kSize, kSize,
                                          format);
            context_->Submit([&](VkCommandBuffer commands) {
                delegate.Record(
                    commands, target.Prepare(commands),
                    ernie::render_vars::kDistanceToCamera);
            });
            return target.ReadColor();
        };
        const std::vector<std::uint8_t> without = present(false);
        const std::vector<std::uint8_t> with = present(true);
        int lit = 0;
        for (double x = -140.0; x <= 140.0; x += 10.0) {
            const std::size_t on = offset(pixelOf({x, -15.0, 0.0}));
            for (std::size_t c = 0; c < 3; ++c) {
                EXPECT_NEAR(with[on + c], onLdr[on + c], 3)
                    << format << " x = " << x;
            }
            lit += with[on + 2] > 200 && with[on] < 80 ? 1 : 0;
            const std::size_t off = offset(pixelOf({x, 15.0, 0.0}));
            for (std::size_t c = 0; c < 4; ++c) {
                EXPECT_EQ(with[off + c], without[off + c])
                    << format << " x = " << x;
            }
        }
        EXPECT_GT(lit, 20) << format;
    }
}

TEST_F(RasterDelegateTest, OverlaysDrawMarkersLabelsAndShapes) {
    RasterDelegate delegate(*context_);
    Scene scene = OverlayScene();
    const std::vector<std::uint8_t> expected =
        Bytes(delegate.Render(scene));
    OverlayStyle blue;
    blue.color = {0.0f, 0.0f, 1.0f, 1.0f};
    blue.size = 8.0f;
    scene.GetOverlay("markers").Marker({0.0, 15.0, 0.0}, blue,
                                       MarkerShape::Square);
    Tensor image = delegate.Render(scene);
    EXPECT_EQ(Rgb(scene, image, {0.0f, 15.0f, 0.0f}),
              glm::ivec3(0, 0, 255));

    scene.GetOverlay("markers").Clear();
    scene.GetOverlay("labels").Label({0.0, 0.0, 0.0}, "Ernie");
    image = delegate.Render(scene);
    EXPECT_GT(BytesDiffering(Bytes(image), expected), 20u)
        << "the label's text and backing";

    scene.GetOverlay("labels").Clear();
    OverlayStyle arrow;
    arrow.width = 3.0f;
    scene.GetOverlay("shapes").Arrow({0.0, -20.0, 0.0},
                                     {0.0, -5.0, 0.0}, arrow, 4.0);
    image = delegate.Render(scene);
    EXPECT_EQ(Rgb(scene, image, {0.0f, -15.0f, 0.0f}),
              glm::ivec3(255))
        << "the shaft";
    EXPECT_EQ(Rgb(scene, image, {0.0f, -7.0f, 0.0f}), glm::ivec3(255))
        << "the head";
}
