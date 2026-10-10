// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/overlay.h"
#include "ernie/scene/scene.h"

#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

using ernie::MarkerShape;
using ernie::Overlay;
using ernie::OverlayDepth;
using ernie::OverlayStyle;
using ernie::Scene;

TEST(Overlay, ShapesBecomeSegmentsTrianglesAndSprites) {
    Overlay overlay;
    OverlayStyle style;
    style.color = {1.0f, 0.0f, 0.0f, 1.0f};
    style.depth = OverlayDepth::XRay;
    style.world = 3;
    overlay.Line({0, 0, 0}, {1, 0, 0}, style);
    ASSERT_EQ(overlay.Segments().size(), 1u);
    EXPECT_EQ(overlay.Segments()[0].item.depth, OverlayDepth::XRay);
    EXPECT_EQ(overlay.Segments()[0].item.world, 3);

    overlay.Clear().Box({-1, -1, -1}, {1, 1, 1});
    EXPECT_EQ(overlay.Segments().size(), 12u);

    const glm::dmat4 moved = glm::translate(glm::dmat4(1.0), glm::dvec3(10.0, 0.0, 0.0));
    overlay.Clear().Box({0, 0, 0}, {1, 1, 1}, {}, moved);
    for (const auto& segment : overlay.Segments()) {
        EXPECT_GE(segment.a.x, 10.0);
        EXPECT_GE(segment.b.x, 10.0);
    }

    overlay.Clear().Sphere({0, 0, 0}, 2.0);
    EXPECT_EQ(overlay.Segments().size(), 3u * 48u);
    for (const auto& segment : overlay.Segments()) {
        EXPECT_NEAR(glm::length(segment.a), 2.0, 1e-9);
    }

    overlay.Clear().Arrow({0, 0, 0}, {0, 0, 10});
    EXPECT_EQ(overlay.Segments().size(), 1u);
    EXPECT_NEAR(overlay.Segments()[0].b.z, 8.0, 1e-9) << "the shaft stops at the head";
    EXPECT_EQ(overlay.Fills().size(), 16u);

    overlay.Clear().Axes(glm::dmat4(1.0), 2.0);
    ASSERT_EQ(overlay.Segments().size(), 3u);
    EXPECT_EQ(overlay.Segments()[2].b, glm::dvec3(0, 0, 2));
}

TEST(Overlay, WireMeshesDrawSharedEdgesOnce) {
    Overlay overlay;
    const std::array<glm::dvec3, 4> points{glm::dvec3(0, 0, 0), glm::dvec3(1, 0, 0),
                                           glm::dvec3(1, 1, 0), glm::dvec3(0, 1, 0)};
    const std::array<std::uint32_t, 6> quad{0, 1, 2, 0, 2, 3};
    overlay.WireMesh(points, quad);
    EXPECT_EQ(overlay.Segments().size(), 5u);
    overlay.Triangles(points, quad);
    EXPECT_EQ(overlay.Fills().size(), 2u);
    const std::array<std::uint32_t, 3> bad{0, 1, 7};
    EXPECT_THROW(overlay.WireMesh(points, bad), std::invalid_argument);
    EXPECT_THROW(overlay.Triangles(points, std::span(quad).first(4)), std::invalid_argument);
}

TEST(Overlay, BulkLinesAndMarkersTakeColoursPerItem) {
    Overlay overlay;
    const std::vector<glm::dvec3> endpoints{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}};
    const std::vector<glm::vec4> colors{{1, 0, 0, 1}, {0, 1, 0, 1}};
    overlay.Lines(endpoints, {}, colors);
    ASSERT_EQ(overlay.Segments().size(), 2u);
    EXPECT_EQ(overlay.Segments()[1].item.color, colors[1]);
    EXPECT_THROW(overlay.Lines(std::span(endpoints).first(3)), std::invalid_argument);
    EXPECT_THROW(overlay.Lines(endpoints, {}, std::span(colors).first(1)), std::invalid_argument);

    overlay.Markers(std::span(endpoints).first(2), {}, MarkerShape::Cross, colors);
    ASSERT_EQ(overlay.Sprites().size(), 2u);
    EXPECT_EQ(overlay.Sprites()[0].shape, MarkerShape::Cross);
    EXPECT_EQ(overlay.Sprites()[0].rect, glm::vec4(-6, -6, 6, 6));
}

TEST(Overlay, LabelsBecomeTextQuadsOverABackingBox) {
    Overlay overlay;
    OverlayStyle style;
    overlay.Label({0, 0, 0}, "Hi", style);
    ASSERT_GT(overlay.Sprites().size(), 3u);
    EXPECT_EQ(overlay.Sprites()[0].item.color, style.background) << "the backing first";
    for (const auto& sprite : overlay.Sprites()) {
        EXPECT_GT(sprite.rect.x, 0.0f) << "right of the point";
        EXPECT_LT(sprite.rect.w, 0.0f) << "above it";
    }
    const std::size_t withBacking = overlay.Sprites().size();
    style.background.a = 0.0f;
    overlay.Clear().Label({0, 0, 0}, "Hi", style);
    EXPECT_EQ(overlay.Sprites().size(), withBacking - 1);
}

TEST(Overlay, VersionsChangeWithEveryEdit) {
    Overlay a;
    Overlay b;
    EXPECT_NE(a.Version(), b.Version());
    const std::uint64_t before = a.Version();
    a.Line({0, 0, 0}, {1, 1, 1});
    EXPECT_NE(a.Version(), before);
    const std::uint64_t drawn = a.Version();
    a.SetVisible(true);
    EXPECT_EQ(a.Version(), drawn) << "no change";
    a.SetVisible(false);
    EXPECT_NE(a.Version(), drawn);
}

TEST(Scene, HoldsNamedOverlays) {
    Scene scene;
    EXPECT_EQ(scene.FindOverlay("debug"), nullptr);
    scene.GetOverlay("debug").Line({0, 0, 0}, {1, 0, 0});
    scene.GetOverlay("contacts");
    ASSERT_NE(scene.FindOverlay("debug"), nullptr);
    EXPECT_EQ(scene.FindOverlay("debug")->Segments().size(), 1u);
    EXPECT_EQ(scene.OverlayNames(), (std::vector<std::string>{"contacts", "debug"}));
    const std::uint64_t version = scene.Version();
    scene.GetOverlay("debug").Clear();
    EXPECT_EQ(scene.Version(), version) << "overlays are not prims";
    EXPECT_TRUE(scene.RemoveOverlay("debug"));
    EXPECT_FALSE(scene.RemoveOverlay("debug"));
}

TEST(SelectOverlays, TakesAllOrNamesInOrder) {
    const std::vector<std::string> available{"a", "b", "c"};
    EXPECT_EQ(ernie::SelectOverlays("*", available), available);
    EXPECT_EQ(ernie::SelectOverlays("c, a", available), (std::vector<std::string>{"a", "c"}));
    EXPECT_EQ(ernie::SelectOverlays("b missing", available), (std::vector<std::string>{"b"}));
    EXPECT_TRUE(ernie::SelectOverlays("", available).empty());
}
