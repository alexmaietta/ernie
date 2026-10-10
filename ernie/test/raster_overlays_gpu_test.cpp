// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/raster/delegate.h"
#include "ernie/scene/overlay.h"
#include "ernie/scene/tokens.h"
#include "support/delegate_scenes.h"

#include <gtest/gtest.h>

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
namespace tokens = ernie::tokens;

namespace {

/// MeshScene drawing the overlays `overlays` (`ernie:overlays`) over its image.
Scene OverlayScene(const std::string& overlays = "*") {
    Scene scene = MeshScene();
    scene.Set("/Render/Settings", tokens::kOverlays, overlays);
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
    scene.GetOverlay(name).Line({-20.0, -15.0, 0.0}, {20.0, -15.0, 0.0}, style);
}

}  // namespace

TEST_F(RasterDelegateTest, OverlaysDrawOnlyIntoProductsThatAskForThem) {
    RasterDelegate delegate(*context_);
    Scene plain = MeshScene();
    const std::vector<std::uint8_t> expected = Bytes(delegate.Render(plain));
    DrawCrossing(plain, "debug");
    EXPECT_EQ(Bytes(delegate.Render(plain)), expected) << "no ernie:overlays: an observation";

    Scene scene = OverlayScene();
    DrawCrossing(scene, "debug");
    const Tensor image = delegate.Render(scene);
    EXPECT_EQ(Rgb(scene, image, {0.0f, -15.0f, 0.0f}), glm::ivec3(255));
    EXPECT_EQ(Rgb(scene, image, {0.0f, 15.0f, 0.0f}), glm::ivec3(0)) << "away from the line";

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
    scene.GetOverlay("unwanted").Line({-20.0, 15.0, 0.0}, {20.0, 15.0, 0.0}, style);
    const Tensor image = delegate.Render(scene);
    EXPECT_EQ(Rgb(scene, image, {0.0f, -15.0f, 0.0f}), glm::ivec3(255));
    EXPECT_EQ(Rgb(scene, image, {0.0f, 15.0f, 0.0f}), glm::ivec3(0));
}

TEST_F(RasterDelegateTest, OverlaysMeetSurfacesByTheirDepthMode) {
    // A line 1 below the red quad (x -14 to -10, at z 0), along its middle.
    RasterDelegate delegate(*context_);
    const Scene plain = MeshScene();
    const glm::ivec3 quad = Rgb(plain, delegate.Render(plain), {-12.0f, 0.0f, 0.0f});
    const auto through = [&](OverlayDepth depth) {
        Scene scene = OverlayScene();
        OverlayStyle style;
        style.width = 3.0f;
        style.depth = depth;
        scene.GetOverlay("debug").Line({-17.0, 0.0, -1.0}, {-7.0, 0.0, -1.0}, style);
        const Tensor image = delegate.Render(scene);
        return std::pair(Rgb(scene, image, {-12.0f, 0.0f, 0.0f}),
                         Rgb(scene, image, {-8.0f, 0.0f, 0.0f}));
    };
    const auto [hidden, outside] = through(OverlayDepth::Test);
    EXPECT_EQ(hidden, quad) << "behind the quad";
    EXPECT_EQ(outside, glm::ivec3(255)) << "past the quad, over the background";
    const auto [xray, xrayOutside] = through(OverlayDepth::XRay);
    EXPECT_GT(xray.g, quad.g + 40);
    EXPECT_LT(xray.g, 200) << "faded behind the quad";
    EXPECT_EQ(xrayOutside, glm::ivec3(255));
    const auto [always, alwaysOutside] = through(OverlayDepth::Always);
    EXPECT_EQ(always, glm::ivec3(255));
    EXPECT_EQ(alwaysOutside, glm::ivec3(255));
}

TEST_F(RasterDelegateTest, OverlaysDrawMarkersLabelsAndShapes) {
    RasterDelegate delegate(*context_);
    Scene scene = OverlayScene();
    const std::vector<std::uint8_t> expected = Bytes(delegate.Render(scene));
    OverlayStyle blue;
    blue.color = {0.0f, 0.0f, 1.0f, 1.0f};
    blue.size = 8.0f;
    scene.GetOverlay("markers").Marker({0.0, 15.0, 0.0}, blue, MarkerShape::Square);
    Tensor image = delegate.Render(scene);
    EXPECT_EQ(Rgb(scene, image, {0.0f, 15.0f, 0.0f}), glm::ivec3(0, 0, 255));

    scene.GetOverlay("markers").Clear();
    scene.GetOverlay("labels").Label({0.0, 0.0, 0.0}, "Ernie");
    image = delegate.Render(scene);
    EXPECT_GT(BytesDiffering(Bytes(image), expected), 20u) << "the label's text and backing";

    scene.GetOverlay("labels").Clear();
    OverlayStyle arrow;
    arrow.width = 3.0f;
    scene.GetOverlay("shapes").Arrow({0.0, -20.0, 0.0}, {0.0, -5.0, 0.0}, arrow, 4.0);
    image = delegate.Render(scene);
    EXPECT_EQ(Rgb(scene, image, {0.0f, -15.0f, 0.0f}), glm::ivec3(255)) << "the shaft";
    EXPECT_EQ(Rgb(scene, image, {0.0f, -7.0f, 0.0f}), glm::ivec3(255)) << "the head";
}
