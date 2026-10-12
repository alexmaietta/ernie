// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/overlay.h"
#include "ernie/scene/scene.h"

#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using ernie::MarkerShape;
using ernie::Overlay;
using ernie::OverlayDepth;
using ernie::OverlayStyle;
using ernie::Scene;
using ernie::Tensor;

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

    const glm::dmat4 moved =
        glm::translate(glm::dmat4(1.0), glm::dvec3(10.0, 0.0, 0.0));
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
    EXPECT_NEAR(overlay.Segments()[0].b.z, 8.0, 1e-9)
        << "the shaft stops at the head";
    EXPECT_EQ(overlay.Fills().size(), 16u);

    overlay.Clear().Axes(glm::dmat4(1.0), 2.0);
    ASSERT_EQ(overlay.Segments().size(), 3u);
    EXPECT_EQ(overlay.Segments()[2].b, glm::dvec3(0, 0, 2));
}

TEST(Overlay, WireMeshesDrawSharedEdgesOnce) {
    Overlay overlay;
    const std::array<glm::dvec3, 4> points{
        glm::dvec3(0, 0, 0), glm::dvec3(1, 0, 0), glm::dvec3(1, 1, 0),
        glm::dvec3(0, 1, 0)};
    const std::array<std::uint32_t, 6> quad{0, 1, 2, 0, 2, 3};
    overlay.WireMesh(points, quad);
    EXPECT_EQ(overlay.Segments().size(), 5u);
    overlay.Triangles(points, quad);
    EXPECT_EQ(overlay.Fills().size(), 2u);
    const std::array<std::uint32_t, 3> bad{0, 1, 7};
    EXPECT_THROW(overlay.WireMesh(points, bad),
                 std::invalid_argument);
    EXPECT_THROW(overlay.Triangles(points, std::span(quad).first(4)),
                 std::invalid_argument);
}

TEST(Overlay, BulkLinesAndMarkersTakeColoursPerItem) {
    Overlay overlay;
    const std::vector<glm::dvec3> endpoints{
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}};
    const std::vector<glm::vec4> colors{{1, 0, 0, 1}, {0, 1, 0, 1}};
    overlay.Lines(endpoints, {}, colors);
    ASSERT_EQ(overlay.Segments().size(), 2u);
    EXPECT_EQ(overlay.Segments()[1].item.color, colors[1]);
    EXPECT_THROW(overlay.Lines(std::span(endpoints).first(3)),
                 std::invalid_argument);
    EXPECT_THROW(
        overlay.Lines(endpoints, {}, std::span(colors).first(1)),
        std::invalid_argument);

    overlay.Markers(std::span(endpoints).first(2), {},
                    MarkerShape::Cross, colors);
    ASSERT_EQ(overlay.Sprites().size(), 2u);
    EXPECT_EQ(overlay.Sprites()[0].shape, MarkerShape::Cross);
    EXPECT_EQ(overlay.Sprites()[0].rect, glm::vec4(-6, -6, 6, 6));
}

TEST(Overlay, LabelsBecomeACharacterEachOverABackingBox) {
    Overlay overlay;
    OverlayStyle style;
    overlay.Label({0, 0, 0}, "H i", style);
    ASSERT_EQ(overlay.Sprites().size(), 3u)
        << "the backing and two characters, the space none";
    EXPECT_EQ(overlay.Sprites()[0].item.color, style.background)
        << "the backing first";
    EXPECT_EQ(overlay.Sprites()[0].glyph, 0);
    EXPECT_EQ(overlay.Sprites()[1].glyph, 'H');
    EXPECT_EQ(overlay.Sprites()[2].glyph, 'i');
    EXPECT_GT(overlay.Sprites()[2].rect.x,
              overlay.Sprites()[1].rect.z)
        << "left to right";
    for (const auto& sprite : overlay.Sprites()) {
        EXPECT_GT(sprite.rect.x, 0.0f) << "right of the point";
        EXPECT_LT(sprite.rect.w, 0.0f) << "above it";
        EXPECT_FALSE(sprite.screen);
    }
    style.background.a = 0.0f;
    overlay.Clear().Label({0, 0, 0}, "H i", style);
    EXPECT_EQ(overlay.Sprites().size(), 2u);
}

TEST(Overlay, ArcsAndCurvesBecomeSegmentsOrSectors) {
    Overlay overlay;
    OverlayStyle style;
    style.dash = 4.0f;
    // A quarter turn about z from x: from (2, 0, 0) to (0, 2, 0).
    overlay.Arc({0, 0, 0}, {0, 0, 1}, {1, 0, 0.5}, 2.0, M_PI / 2,
                style, false, 8);
    ASSERT_EQ(overlay.Segments().size(), 8u);
    EXPECT_NEAR(glm::distance(overlay.Segments().front().a,
                              glm::dvec3(2, 0, 0)),
                0.0, 1e-12);
    EXPECT_NEAR(glm::distance(overlay.Segments().back().b,
                              glm::dvec3(0, 2, 0)),
                0.0, 1e-12);
    EXPECT_EQ(overlay.Segments().front().dash, 4.0f);
    overlay.Arc({0, 0, 0}, {0, 0, 1}, {1, 0, 0}, 2.0, M_PI, style,
                true, 6);
    EXPECT_EQ(overlay.Fills().size(), 6u) << "a filled sector";
    EXPECT_EQ(overlay.Fills()[0].points[0], glm::dvec3(0.0));
    EXPECT_THROW(
        overlay.Arc({0, 0, 0}, {0, 0, 1}, {0, 0, 3}, 1.0, 1.0, style),
        std::invalid_argument)
        << "a start along the normal";

    overlay.Clear().Curve({0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0},
                          style, 10);
    ASSERT_EQ(overlay.Segments().size(), 10u);
    EXPECT_EQ(overlay.Segments().front().a, glm::dvec3(0, 0, 0));
    EXPECT_NEAR(glm::distance(overlay.Segments().back().b,
                              glm::dvec3(1, 0, 0)),
                0.0, 1e-12);
    // Halfway, the curve's apex: (0.5, 0.75, 0).
    EXPECT_NEAR(glm::distance(overlay.Segments()[4].b,
                              glm::dvec3(0.5, 0.75, 0)),
                0.0, 1e-12);
}

TEST(Overlay, TheFontCoversPrintableAscii) {
    const auto glyphs = ernie::OverlayGlyphs();
    ASSERT_EQ(glyphs.size(), 95u);
    const ernie::OverlayGlyph& space = glyphs[0];
    EXPECT_EQ(space.box[2], space.box[0]) << "nothing to draw";
    EXPECT_GT(space.advance, 0);
    const ernie::OverlayGlyph& a = glyphs['A' - 32];
    EXPECT_GT(a.box[2], a.box[0]);
    EXPECT_GT(a.box[3], a.box[1]);
    // The box holds every bit set, and the bits fill it.
    std::uint16_t any = 0;
    for (int y = 0; y < ernie::kOverlayGlyphCell; ++y) {
        if (y < a.box[1] || y >= a.box[3]) {
            EXPECT_EQ(a.rows[y], 0) << y;
        }
        any |= a.rows[y];
    }
    EXPECT_EQ(any & ((1u << a.box[0]) - 1u), 0u);
    EXPECT_EQ(any >> a.box[2], 0);
    const glm::vec2 size = ernie::OverlayTextSize("ab\ncd", 24.0f);
    EXPECT_FLOAT_EQ(size.y, 2.0f * (24.0f - 6.0f) + 6.0f)
        << "two lines but the last's spacing";
    EXPECT_GT(size.x, 0.0f);
}

TEST(Overlay, ScreenTextSitsByItsPlacement) {
    Overlay overlay;
    OverlayStyle style;
    style.background.a = 0.0f;
    // The top right corner, 10 pixels in from each edge.
    ernie::ScreenPlacement corner;
    corner.at = {1.0f, 0.0f};
    corner.offset = {-10.0f, 10.0f};
    overlay.Text(corner, "t = 1.50 s", style);
    ASSERT_FALSE(overlay.Sprites().empty());
    const glm::vec2 size =
        ernie::OverlayTextSize("t = 1.50 s", 12.0f);
    float right = -1e9f;
    float top = 1e9f;
    for (const auto& sprite : overlay.Sprites()) {
        EXPECT_TRUE(sprite.screen);
        EXPECT_EQ(sprite.anchor, glm::dvec3(1.0, 0.0, 0.0));
        EXPECT_EQ(sprite.item.depth, OverlayDepth::Always);
        EXPECT_GE(sprite.rect.x, -10.0f - size.x - 1e-3f);
        right = std::max(right, sprite.rect.z);
        top = std::min(top, sprite.rect.y);
    }
    EXPECT_LE(right, -10.0f + 1e-3f) << "inside the right edge";
    EXPECT_GE(top, 10.0f - 1e-3f) << "below the top";
    // Centred on the image's centre.
    ernie::ScreenPlacement centre;
    centre.at = {0.5f, 0.5f};
    overlay.Clear().Rect(centre, {40.0f, 20.0f}, style);
    ASSERT_EQ(overlay.Sprites().size(), 1u);
    EXPECT_EQ(overlay.Sprites()[0].rect,
              glm::vec4(-20.0f, -10.0f, 20.0f, 10.0f));
}

TEST(Overlay, AnchoredItemsShareTheirAnchors) {
    Overlay overlay;
    OverlayStyle style;
    style.anchor = ernie::Path("/World/Robot/Link");
    overlay.Line({0, 0, 0}, {1, 0, 0}, style)
        .Label({0, 0, 1}, "link", style);
    style.anchor = ernie::Path("/World/Ball");
    overlay.Marker({0, 0, 0}, style);
    ASSERT_EQ(overlay.Anchors().size(), 2u);
    EXPECT_EQ(overlay.Segments()[0].item.anchor, 1u);
    for (std::size_t s = 0; s + 1 < overlay.Sprites().size(); ++s) {
        EXPECT_EQ(overlay.Sprites()[s].item.anchor, 1u);
    }
    EXPECT_EQ(overlay.Sprites().back().item.anchor, 2u);
    overlay.Line({0, 0, 0}, {1, 1, 1});
    EXPECT_EQ(overlay.Segments().back().item.anchor, 0u)
        << "the default style: world coordinates";
    overlay.Clear();
    EXPECT_TRUE(overlay.Anchors().empty());
}

TEST(Overlay, BatchesTakeOneAnchorOrOnePerItem) {
    Overlay overlay;
    ernie::OverlayBatch batch;
    batch.kind = ernie::OverlayBatchKind::Markers;
    batch.points = Tensor::Zeros(ernie::DType::Float32, {3, 3});
    batch.style.anchor = ernie::Path("/A");
    overlay.Batch(batch);
    EXPECT_EQ(overlay.Batches()[0].anchors,
              std::vector<ernie::Path>{ernie::Path("/A")})
        << "the style's";
    batch.anchors = {ernie::Path("/A"), ernie::Path("/B")};
    EXPECT_THROW(overlay.Batch(batch), std::invalid_argument);
    batch.anchors.emplace_back("/C");
    overlay.Batch(batch);
    EXPECT_EQ(overlay.Batches()[1].anchors.size(), 3u);
}

TEST(Overlay, NumbersGoBesidePointsOrOnTheScreen) {
    Overlay overlay;
    ernie::OverlayBatch numbers;
    numbers.kind = ernie::OverlayBatchKind::Numbers;
    numbers.values = Tensor::Zeros(ernie::DType::Float32, {4});
    EXPECT_THROW(overlay.Batch(numbers), std::invalid_argument)
        << "in the world, points";
    numbers.screen = true;
    overlay.Batch(numbers);
    EXPECT_EQ(overlay.Batches()[0].Count(), 4u)
        << "one a line, from the placement";
    numbers.points = Tensor::Zeros(ernie::DType::Float32, {3, 3});
    EXPECT_THROW(overlay.Batch(numbers), std::invalid_argument)
        << "a point per value";
    numbers.points = Tensor();
    numbers.format.decimals = 7;
    EXPECT_THROW(overlay.Batch(numbers), std::invalid_argument);
    ernie::OverlayBatch markers;
    markers.kind = ernie::OverlayBatchKind::Markers;
    markers.points = Tensor::Zeros(ernie::DType::Float32, {1, 3});
    markers.screen = true;
    EXPECT_THROW(overlay.Batch(markers), std::invalid_argument);
}

TEST(Overlay, BatchArraysAreReplacedInPlace) {
    Overlay overlay;
    ernie::OverlayBatch batch;
    batch.kind = ernie::OverlayBatchKind::Markers;
    batch.points = Tensor::Zeros(ernie::DType::Float32, {2, 3});
    overlay.Batch(batch);
    const std::uint64_t version = overlay.Version();
    overlay.ReplaceBatchArray(
        0, "points", Tensor::Zeros(ernie::DType::Float32, {2, 3}));
    EXPECT_EQ(overlay.Version(), version) << "the layout stays";
    EXPECT_THROW(overlay.ReplaceBatchArray(
                     0, "points",
                     Tensor::Zeros(ernie::DType::Float32, {3, 3})),
                 std::invalid_argument);
    EXPECT_THROW(
        overlay.ReplaceBatchArray(
            0, "ends", Tensor::Zeros(ernie::DType::Float32, {2, 3})),
        std::invalid_argument)
        << "absent";
    EXPECT_THROW(overlay.ReplaceBatchArray(
                     1, "points",
                     Tensor::Zeros(ernie::DType::Float32, {2, 3})),
                 std::invalid_argument);
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
    EXPECT_EQ(scene.OverlayNames(),
              (std::vector<std::string>{"contacts", "debug"}));
    const std::uint64_t version = scene.Version();
    scene.GetOverlay("debug").Clear();
    EXPECT_EQ(scene.Version(), version) << "overlays are not prims";
    EXPECT_TRUE(scene.RemoveOverlay("debug"));
    EXPECT_FALSE(scene.RemoveOverlay("debug"));
}

TEST(SelectOverlays, TakesAllOrNamesInOrder) {
    const std::vector<std::string> available{"a", "b", "c"};
    EXPECT_EQ(ernie::SelectOverlays("*", available), available);
    EXPECT_EQ(ernie::SelectOverlays("c, a", available),
              (std::vector<std::string>{"a", "c"}));
    EXPECT_EQ(ernie::SelectOverlays("b missing", available),
              (std::vector<std::string>{"b"}));
    EXPECT_TRUE(ernie::SelectOverlays("", available).empty());
}

TEST(Overlay, BatchesKeepTheirArraysAndCheckThem) {
    using ernie::OverlayBatch;
    using ernie::OverlayBatchKind;
    using ernie::Tensor;
    const std::array<float, 6> points{0, 0, 0, 1, 2, 3};
    const std::array<float, 6> ends{1, 0, 0, 1, 2, 4};
    Overlay overlay;
    const std::uint64_t version = overlay.Version();
    OverlayBatch arrows;
    arrows.kind = OverlayBatchKind::Arrows;
    arrows.points = Tensor::Copy<float>(points, {2, 3});
    arrows.ends = Tensor::Copy<float>(ends, {2, 3});
    const std::array<std::int32_t, 2> worlds{0, 1};
    arrows.worlds = Tensor::Copy<std::int32_t>(worlds, {2});
    overlay.Batch(arrows);
    ASSERT_EQ(overlay.Batches().size(), 1u);
    EXPECT_EQ(overlay.Batches()[0].Count(), 2u);
    EXPECT_EQ(overlay.Batches()[0].points.Data(),
              arrows.points.Data())
        << "kept, not copied";
    EXPECT_TRUE(overlay.Segments().empty());
    EXPECT_FALSE(overlay.Empty());
    EXPECT_NE(overlay.Version(), version);

    OverlayBatch missing = arrows;
    missing.ends = Tensor();
    EXPECT_THROW(overlay.Batch(missing), std::invalid_argument);
    OverlayBatch rows = arrows;
    rows.ends = Tensor::Copy<float>(std::span(ends).first(3), {1, 3});
    EXPECT_THROW(overlay.Batch(rows), std::invalid_argument);
    OverlayBatch floats = arrows;
    const std::array<float, 2> notInts{0, 1};
    floats.worlds = Tensor::Copy<float>(notInts, {2});
    EXPECT_THROW(overlay.Batch(floats), std::invalid_argument);
    OverlayBatch spheres;
    spheres.kind = OverlayBatchKind::Spheres;
    spheres.points = arrows.points;
    EXPECT_THROW(overlay.Batch(spheres), std::invalid_argument)
        << "spheres need radii";

    overlay.Clear();
    EXPECT_TRUE(overlay.Empty());
}
