// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/offscreen_target.h"
#include "ernie/raster/delegate.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/overlay.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/sensor_filters.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/wide_view.h"
#include "ernie/scene/xform.h"
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

TEST_F(RasterDelegateTest,
       ScreenItemsSitWhereTheyArePlacedOnTheImage) {
    // A red square in the top left corner and a blue one in the
    // bottom right, 10 pixels each, through a frame camera and a
    // fisheye on cube faces alike: the lens bends the world, not the
    // screen.
    for (const bool fisheye : {false, true}) {
        RasterDelegate delegate(*context_);
        Scene scene = OverlayScene();
        if (fisheye) {
            // As OverlaysFollowACubeCamerasLens: 100 degrees off axis
            // at the corners.
            const double aperture = 20.0;
            const double corner = std::sqrt(2.0) * 0.5 * kSize;
            ernie::Camera(scene, "/World/Camera")
                .SetLensProjection(ernie::tokens::kFisheye)
                .SetVerticalAperture(aperture)
                .SetHorizontalAperture(aperture)
                .SetFocalLength(aperture * corner /
                                (100.0 * M_PI / 180.0) / kSize);
        }
        const std::vector<std::uint8_t> plain =
            Bytes(delegate.Render(scene));
        OverlayStyle red;
        red.color = {1.0f, 0.0f, 0.0f, 1.0f};
        OverlayStyle blue;
        blue.color = {0.0f, 0.0f, 1.0f, 1.0f};
        ernie::ScreenPlacement topLeft;
        ernie::ScreenPlacement bottomRight;
        bottomRight.at = {1.0f, 1.0f};
        scene.GetOverlay("hud")
            .Rect(topLeft, {10.0f, 10.0f}, red)
            .Rect(bottomRight, {10.0f, 10.0f}, blue);
        const std::vector<std::uint8_t> image =
            Bytes(delegate.Render(scene));
        const auto rgb = [&](std::size_t x, std::size_t y) {
            const std::size_t at = 4 * (y * kSize + x);
            return glm::ivec3(image[at], image[at + 1],
                              image[at + 2]);
        };
        EXPECT_EQ(rgb(0, 0), glm::ivec3(255, 0, 0)) << fisheye;
        EXPECT_EQ(rgb(9, 9), glm::ivec3(255, 0, 0)) << fisheye;
        EXPECT_EQ(rgb(kSize - 1, kSize - 1), glm::ivec3(0, 0, 255))
            << fisheye;
        EXPECT_EQ(rgb(kSize - 10, kSize - 10), glm::ivec3(0, 0, 255))
            << fisheye;
        for (const auto& [x, y] :
             {std::pair<std::size_t, std::size_t>{10, 10},
              {kSize / 2, kSize / 2},
              {kSize - 11, kSize - 11}}) {
            const std::size_t at = 4 * (y * kSize + x);
            EXPECT_EQ(image[at], plain[at]) << fisheye << " " << x;
        }
    }
}

TEST_F(RasterDelegateTest, PicksFindOverlayItemsInTheWorld) {
    // A host marker with an id and a batch of two markers, picked
    // where the view draws them, as the viewer's clicks do.
    RasterDelegate delegate(*context_);
    Scene scene = OverlayScene();
    OverlayStyle host;
    host.id = 5;
    host.size = 8.0f;
    host.depth = OverlayDepth::Always;
    scene.GetOverlay("debug").Marker({0.0, 15.0, 0.0}, host,
                                     MarkerShape::Square);
    ernie::OverlayBatch markers;
    markers.kind = ernie::OverlayBatchKind::Markers;
    markers.points = Tensor::Copy<float>(
        std::vector<float>{-15.0f, -15.0f, 0.0f, 15.0f, -15.0f, 0.0f},
        {2, 3});
    markers.shape = MarkerShape::Square;
    markers.style.size = 8.0f;
    markers.style.depth = OverlayDepth::Always;
    scene.GetOverlay("contacts").Batch(markers);
    // Drawn into the viewer's own targets, which picks read.
    delegate.Update(scene, "/Render/Settings", {kSize, kSize});
    ernie::OffscreenTarget target(*context_, kSize, kSize,
                                  VK_FORMAT_R8G8B8A8_UNORM);
    context_->Submit([&](VkCommandBuffer commands) {
        delegate.Record(commands, target.Prepare(commands),
                        ernie::render_vars::kLdrColor);
    });
    const Tensor image = delegate.Render(scene);
    const auto pixel = [&](glm::vec3 p) {
        const auto at = static_cast<std::size_t>(
            (TexelAt(scene, image, p) -
             static_cast<const std::uint8_t*>(image.Data())) /
            4);
        return std::pair<std::uint32_t, std::uint32_t>(
            static_cast<std::uint32_t>(at % kSize),
            static_cast<std::uint32_t>(at / kSize));
    };
    const auto pick = [&](glm::vec3 p) {
        const auto [x, y] = pixel(p);
        return delegate.PickOverlay(scene, x, y);
    };
    const std::optional<ernie::OverlayHit> marker =
        pick({0.0f, 15.0f, 0.0f});
    ASSERT_TRUE(marker.has_value());
    EXPECT_EQ(marker->overlay, "debug");
    EXPECT_EQ(marker->batch, -1);
    EXPECT_EQ(marker->item, 5u);
    const std::optional<ernie::OverlayHit> second =
        pick({15.0f, -15.0f, 0.0f});
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(second->overlay, "contacts");
    EXPECT_EQ(second->batch, 0);
    EXPECT_EQ(second->item, 1u);
    EXPECT_FALSE(pick({0.0f, -5.0f, 0.0f}).has_value())
        << "between them";
}

TEST_F(RasterDelegateTest, ImagesDrawOnQuadsAndTheScreen) {
    // A 2 x 2 image of red, green, blue and white texels, on the
    // screen at its corners' pixels, and on a quad in the world.
    RasterDelegate delegate(*context_);
    Scene scene = OverlayScene();
    const std::vector<std::uint8_t> texels{
        255, 0, 0,   255, 0,   255, 0,   255,
        0,   0, 255, 255, 255, 255, 255, 255};
    const Tensor image =
        Tensor::Copy<std::uint8_t>(texels, {2, 2, 4});
    ernie::ScreenPlacement at;
    at.offset = {0.0f, 0.0f};
    scene.GetOverlay("pictures")
        .ScreenImage(at, image, {}, {20.0f, 20.0f});
    const std::array<glm::dvec3, 4> corners{
        glm::dvec3(5, 20, 0), glm::dvec3(25, 20, 0),
        glm::dvec3(25, 0, 0), glm::dvec3(5, 0, 0)};
    OverlayStyle world;
    world.depth = OverlayDepth::Always;
    scene.GetOverlay("pictures").Image(corners, image, world);
    const std::vector<std::uint8_t> rendered =
        Bytes(delegate.Render(scene));
    const auto rgb = [&](std::size_t x, std::size_t y) {
        const std::size_t texel = 4 * (y * kSize + x);
        return glm::ivec3(rendered[texel], rendered[texel + 1],
                          rendered[texel + 2]);
    };
    // Each quarter of the screen image near its own texel's colour.
    EXPECT_EQ(rgb(1, 1), glm::ivec3(255, 0, 0));
    EXPECT_EQ(rgb(18, 1), glm::ivec3(0, 255, 0));
    EXPECT_EQ(rgb(1, 18), glm::ivec3(0, 0, 255));
    EXPECT_EQ(rgb(18, 18), glm::ivec3(255, 255, 255));
    // The world quad's top left quarter red, its bottom right white.
    const Tensor again = delegate.Render(scene);
    EXPECT_EQ(Rgb(scene, again, {6.5f, 18.5f, 0.0f}),
              glm::ivec3(255, 0, 0));
    EXPECT_EQ(Rgb(scene, again, {23.5f, 1.5f, 0.0f}),
              glm::ivec3(255, 255, 255));
}

TEST_F(RasterDelegateTest, PolylinesBlendTheirPointsColours) {
    RasterDelegate delegate(*context_);
    Scene scene = OverlayScene();
    OverlayStyle style;
    style.width = 4.0f;
    style.depth = OverlayDepth::Always;
    const std::vector<glm::dvec3> points{{-20.0, -15.0, 0.0},
                                         {20.0, -15.0, 0.0}};
    const std::vector<glm::vec4> colors{{1.0f, 0.0f, 0.0f, 1.0f},
                                        {0.0f, 0.0f, 1.0f, 1.0f}};
    scene.GetOverlay("debug").Polyline(points, style, false, colors);
    const Tensor image = delegate.Render(scene);
    const glm::ivec3 start =
        Rgb(scene, image, {-18.0f, -15.0f, 0.0f});
    const glm::ivec3 end = Rgb(scene, image, {18.0f, -15.0f, 0.0f});
    EXPECT_GT(start.r, 200);
    EXPECT_LT(start.b, 60);
    EXPECT_GT(end.b, 200);
    EXPECT_LT(end.r, 60);
}

TEST_F(RasterDelegateTest, MetricMarkersKeepTheirSizeInTheWorld) {
    // A marker 16 m across and one 16 pixels across, probed between
    // their half widths in metres: only the larger covers the probe.
    RasterDelegate delegate(*context_);
    const Scene plain = OverlayScene();
    const Tensor reference = delegate.Render(plain);
    const auto column = [&](float x) {
        return (TexelAt(plain, reference, {x, 15.0f, 0.0f}) -
                static_cast<const std::uint8_t*>(reference.Data())) /
               4 % static_cast<std::ptrdiff_t>(kSize);
    };
    const double pixelsPerMetre =
        static_cast<double>(column(10.0f) - column(0.0f)) / 10.0;
    ASSERT_GT(pixelsPerMetre, 0.0);
    const double metricHalf = 8.0;                  // metres
    const double pixelHalf = 8.0 / pixelsPerMetre;  // metres
    const double probe = 0.5 * (metricHalf + pixelHalf);
    for (const bool metric : {true, false}) {
        Scene scene = OverlayScene();
        OverlayStyle style;
        style.size = 16.0f;
        style.metric = metric;
        style.depth = OverlayDepth::Always;
        style.color = {0.0f, 1.0f, 0.0f, 1.0f};
        scene.GetOverlay("debug").Marker({0.0, 15.0, 0.0}, style,
                                         MarkerShape::Square);
        const Tensor image = delegate.Render(scene);
        const glm::ivec3 rim = Rgb(
            scene, image, {static_cast<float>(probe), 15.0f, 0.0f});
        const bool covers = (metric ? metricHalf : pixelHalf) > probe;
        EXPECT_EQ(rim == glm::ivec3(0, 255, 0), covers) << metric;
    }
}

TEST_F(RasterDelegateTest, HoveredButtonsTakeTheirHoverColour) {
    RasterDelegate delegate(*context_);
    Scene scene = OverlayScene();
    OverlayStyle style;
    style.id = 3;
    style.background = {0.0f, 0.0f, 0.0f, 1.0f};
    style.hover = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
    ernie::ScreenPlacement at;
    at.offset = {2.0f, 2.0f};
    scene.GetOverlay("menu").Button(at, "", style, {20.0f, 12.0f});
    const auto corner = [&] {
        const std::vector<std::uint8_t> image =
            Bytes(delegate.Render(scene));
        const std::size_t texel = 4 * (4 * kSize + 4);
        return glm::ivec3(image[texel], image[texel + 1],
                          image[texel + 2]);
    };
    EXPECT_EQ(corner(), glm::ivec3(0)) << "its background";
    scene.GetOverlay("menu").SetHovered(3);
    EXPECT_EQ(corner(), glm::ivec3(255, 0, 0)) << "hovered";
    scene.GetOverlay("menu").SetHovered(0);
    EXPECT_EQ(corner(), glm::ivec3(0));
}

TEST_F(RasterDelegateTest, DashedLinesLeaveGaps) {
    // DrawCrossing's line dashed every 4 pixels, from the host and
    // from a batch: along it, about as many texels lit as not.
    RasterDelegate delegate(*context_);
    for (const bool batched : {false, true}) {
        Scene scene = OverlayScene();
        OverlayStyle style;
        style.width = 3.0f;
        style.dash = 4.0f;
        if (batched) {
            ernie::OverlayBatch lines;
            lines.kind = ernie::OverlayBatchKind::Lines;
            lines.points = Tensor::Copy<float>(
                std::vector<float>{-20.0f, -15.0f, 0.0f}, {1, 3});
            lines.ends = Tensor::Copy<float>(
                std::vector<float>{20.0f, -15.0f, 0.0f}, {1, 3});
            lines.style = style;
            scene.GetOverlay("debug").Batch(lines);
        } else {
            scene.GetOverlay("debug").Line({-20.0, -15.0, 0.0},
                                           {20.0, -15.0, 0.0}, style);
        }
        const Tensor image = delegate.Render(scene);
        int lit = 0;
        int dark = 0;
        for (float x = -18.0f; x <= 18.0f; x += 0.25f) {
            const glm::ivec3 rgb =
                Rgb(scene, image, {x, -15.0f, 0.0f});
            lit += rgb == glm::ivec3(255) ? 1 : 0;
            dark += rgb == glm::ivec3(0) ? 1 : 0;
        }
        EXPECT_GT(lit, 30) << batched;
        EXPECT_GT(dark, 30) << batched;
    }
}

TEST_F(RasterDelegateTest, ItemsPosedUnderAPrimFollowIt) {
    // A line through a tracker prim, in its space: it draws where the
    // tracker is each frame, though the overlay never changes.
    RasterDelegate delegate(*context_);
    Scene scene = OverlayScene();
    ernie::Xform::Define(scene, "/World/Tracker");
    ernie::SetXform(scene, "/World/Tracker",
                    {.translate = glm::dvec3(0.0, -15.0, 0.0)});
    OverlayStyle style;
    style.width = 3.0f;
    style.anchor = ernie::Path("/World/Tracker");
    scene.GetOverlay("tracked").Line({-20.0, 0.0, 0.0},
                                     {20.0, 0.0, 0.0}, style);
    style.size = 8.0f;
    style.color = {0.0f, 0.0f, 1.0f, 1.0f};
    scene.GetOverlay("tracked").Marker({10.0, 0.0, 0.0}, style,
                                       MarkerShape::Square);
    const std::uint64_t version =
        scene.GetOverlay("tracked").Version();
    Tensor image = delegate.Render(scene);
    EXPECT_EQ(Rgb(scene, image, {0.0f, -15.0f, 0.0f}),
              glm::ivec3(255));
    EXPECT_EQ(Rgb(scene, image, {10.0f, -15.0f, 0.0f}),
              glm::ivec3(0, 0, 255));
    EXPECT_EQ(Rgb(scene, image, {0.0f, 15.0f, 0.0f}), glm::ivec3(0));

    ernie::SetXform(scene, "/World/Tracker",
                    {.translate = glm::dvec3(0.0, 15.0, 0.0)});
    image = delegate.Render(scene);
    EXPECT_EQ(scene.GetOverlay("tracked").Version(), version);
    EXPECT_EQ(Rgb(scene, image, {0.0f, 15.0f, 0.0f}), glm::ivec3(255))
        << "moved with the tracker";
    EXPECT_EQ(Rgb(scene, image, {10.0f, 15.0f, 0.0f}),
              glm::ivec3(0, 0, 255));
    EXPECT_EQ(Rgb(scene, image, {0.0f, -15.0f, 0.0f}), glm::ivec3(0));
}

TEST_F(RasterDelegateTest, NumbersPrintOnTheGpuAsTextDoesOnTheHost) {
    // Values formatted on the GPU, a line each from the placement,
    // draw the very pixels of the same text from the host.
    RasterDelegate delegate(*context_);
    const std::vector<float> values{12.5f, -3.25f, 0.004f};
    OverlayStyle style;
    style.size = 6.0f;
    style.color = {1.0f, 0.9f, 0.2f, 1.0f};
    ernie::ScreenPlacement placement;
    placement.at = {0.0f, 0.0f};
    placement.offset = {2.0f, 2.0f};

    Scene host = OverlayScene();
    const char* const lines[] = {"t=12.50 s", "t=-3.25 s",
                                 "t=0.00 s"};
    const float line = style.size + 2.0f * 0.25f * style.size;
    for (std::size_t i = 0; i < values.size(); ++i) {
        ernie::ScreenPlacement at = placement;
        at.offset.y += static_cast<float>(i) * line;
        host.GetOverlay("hud").Text(at, lines[i], style);
    }
    const std::vector<std::uint8_t> expected =
        Bytes(delegate.Render(host));

    Scene gpu = OverlayScene();
    ernie::OverlayBatch numbers;
    numbers.kind = ernie::OverlayBatchKind::Numbers;
    numbers.values = Tensor::Copy<float>(
        values, {static_cast<std::int64_t>(values.size())});
    numbers.format.prefix = "t=";
    numbers.format.suffix = " s";
    numbers.format.decimals = 2;
    numbers.screen = true;
    numbers.placement = placement;
    numbers.style = style;
    gpu.GetOverlay("hud").Batch(numbers);
    const std::vector<std::uint8_t> image =
        Bytes(delegate.Render(gpu));
    EXPECT_GT(BytesDiffering(expected,
                             Bytes(delegate.Render(OverlayScene()))),
              50u)
        << "the text draws";
    EXPECT_EQ(BytesDiffering(image, expected), 0u);
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
