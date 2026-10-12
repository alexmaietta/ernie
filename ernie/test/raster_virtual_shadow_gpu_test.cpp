// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/pipeline.h"
#include "ernie/device/shader_compiler.h"
#include "ernie/raster/delegate.h"
#include "ernie/raster/virtual_shadow_map.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/terrain_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"
#include "support/delegate_scenes.h"
#include "support/vulkan_fixture.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

using ernie::GpuBuffer;
using ernie::GpuImage;
using ernie::VirtualShadowMap;
using namespace ernie::testing;

namespace {

class VirtualShadowMapTest : public VulkanTest {};

/// Every uint of a buffer.
std::vector<std::uint32_t> WordsOf(const GpuBuffer& buffer) {
    std::vector<std::uint32_t> words(buffer.Size() /
                                     sizeof(std::uint32_t));
    buffer.Read(std::as_writable_bytes(std::span(words)));
    return words;
}

/// A depth image of `depth` everywhere, ready to sample.
GpuImage FlatDepth(const ernie::VulkanContext& context,
                   std::uint32_t width, std::uint32_t height,
                   float depth) {
    GpuImage image(
        context,
        ernie::ImageDesc{width, height, 1, 1, VK_FORMAT_R32_SFLOAT,
                         VK_IMAGE_USAGE_SAMPLED_BIT |
                             VK_IMAGE_USAGE_TRANSFER_DST_BIT});
    context.Submit([&](VkCommandBuffer commands) {
        image.Transition(commands,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkClearColorValue value{};
        value.float32[0] = depth;
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT,
                                            0, 1, 0, 1};
        vkCmdClearColorImage(commands, image.Handle(),
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             &value, 1, &range);
        image.Transition(commands,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    return image;
}

/// A depth image of `depth` in its top `rows` rows and the sky's 0
/// below, ready to sample.
GpuImage TopRowsDepth(const ernie::VulkanContext& context,
                      std::uint32_t width, std::uint32_t height,
                      std::uint32_t rows, float depth) {
    GpuImage image(
        context,
        ernie::ImageDesc{width, height, 1, 1, VK_FORMAT_R32_SFLOAT,
                         VK_IMAGE_USAGE_SAMPLED_BIT |
                             VK_IMAGE_USAGE_TRANSFER_DST_BIT});
    std::vector<float> texels(std::size_t{width} * height, 0.0f);
    std::fill_n(texels.begin(), std::size_t{width} * rows, depth);
    GpuBuffer staging(context, texels.size() * sizeof(float),
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      ernie::BufferMemory::Upload);
    staging.Write(std::as_bytes(std::span(texels)));
    context.Submit([&](VkCommandBuffer commands) {
        image.Transition(commands,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0,
                                   1};
        region.imageExtent = {width, height, 1};
        vkCmdCopyBufferToImage(
            commands, staging.Handle(), image.Handle(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        image.Transition(commands,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    return image;
}

/// The page table index of the page holding `p` (origin-relative) in
/// level `index` of `levels`.
std::size_t
PageOf(const std::vector<ernie::VirtualShadowLevel>& levels,
       std::size_t index, const glm::dvec3& p) {
    const glm::dvec4 clip =
        glm::dmat4(levels[index].viewProjection) * glm::dvec4(p, 1.0);
    const double texels = VirtualShadowMap::kVirtualTexels;
    const auto x =
        static_cast<std::size_t>((clip.x * 0.5 + 0.5) * texels);
    const auto y =
        static_cast<std::size_t>((0.5 - clip.y * 0.5) * texels);
    const std::size_t across = VirtualShadowMap::kPagesAcross;
    const std::size_t page = VirtualShadowMap::kPageTexels;
    return (index * across + y / page) * across + x / page;
}

std::uint32_t Backed(const std::vector<std::uint32_t>& table) {
    std::uint32_t backed = 0;
    for (const std::uint32_t entry : table) {
        backed += (entry >> 31) != 0 ? 1u : 0u;
    }
    return backed;
}

}  // namespace

// The depth buffer's top rows see the top of the view, clip y near
// +1, as the views draw them with the viewport flipped: they mark the
// pages there, not those of the bottom of the view.
TEST_F(VirtualShadowMapTest, TheTopRowsMarkTheTopOfTheView) {
    ernie::ShaderCompiler compiler;
    VirtualShadowMap map(*context_, compiler, -2, 12, 256);
    const glm::dvec3 eye(0.0, 0.0, 5.0);
    const auto levels = VirtualShadowMap::FitLevels(
        glm::normalize(glm::dvec3(0.3, 0.2, 1.0)), eye, eye, -2, 12);
    // Points 10 to 15 m away in level 5 and its 2 m pages.
    map.SetLevels(levels, 2.0f);
    // Clip x, y at depth 0.5 to the ground 5 m below the eye, 10 m
    // out.
    glm::mat4 inverse(0.0f);
    inverse[0][0] = 10.0f;
    inverse[1][1] = 10.0f;
    inverse[3][2] = -5.0f;
    inverse[3][3] = 1.0f;
    const GpuImage depth = TopRowsDepth(*context_, 64, 48, 4, 0.5f);
    context_->Submit([&](VkCommandBuffer commands) {
        map.RecordMark(commands, depth.View(), 64, 48, inverse,
                       glm::vec3(0.0f));
        map.RecordAllocate(commands);
    });
    const std::vector<std::uint32_t> table = WordsOf(map.PageTable());
    const std::size_t level5 = 5 + 2;
    EXPECT_NE(table[PageOf(levels, level5, {0.0, 9.2, -5.0})] >> 31,
              0u)
        << "the top of the view";
    EXPECT_EQ(table[PageOf(levels, level5, {0.0, -9.2, -5.0})] >> 31,
              0u)
        << "not its bottom";
}

// A 20 m square of ground seen from 5 m above its centre: its points
// mark pages in the levels suiting 5 to 15 m and the next four, plus
// the coarsest level's four about the camera; allocation backs each
// once, in distinct physical pages, and clearing zeroes exactly
// those.
TEST_F(VirtualShadowMapTest, MarksBacksAndClearsTheSeenPages) {
    ernie::ShaderCompiler compiler;
    VirtualShadowMap map(*context_, compiler, -2, 12, 256);
    const glm::dvec3 eye(0.0, 0.0, 5.0);
    const auto levels = VirtualShadowMap::FitLevels(
        glm::normalize(glm::dvec3(0.3, 0.2, 1.0)), eye, eye, -2, 12);
    // Texels about 3 cm at 5 m: what a coarse depth buffer suits; the
    // four coarser levels a wide light's penumbrae filter in.
    map.SetLevels(levels, 4.0f, nullptr, 4);
    // Clip x, y in [-1, 1] at depth 0.5 to the ground below the eye,
    // relative to the origin (the eye).
    glm::mat4 inverse(0.0f);
    inverse[0][0] = 10.0f;
    inverse[1][1] = 10.0f;
    inverse[3][2] = -5.0f;
    inverse[3][3] = 1.0f;
    const GpuImage depth = FlatDepth(*context_, 64, 48, 0.5f);
    // Junk in the pool, to see the clear.
    context_->Submit([&](VkCommandBuffer commands) {
        vkCmdFillBuffer(commands, map.Pool().Handle(), 0,
                        VK_WHOLE_SIZE, 7);
        ernie::ComputeBarrier(commands);
        map.RecordMark(commands, depth.View(), 64, 48, inverse,
                       glm::vec3(0.0f));
        map.RecordAllocate(commands);
        map.RecordClear(commands);
    });
    std::uint32_t used = 0;
    map.Counter().Read(std::as_writable_bytes(std::span(&used, 1)));
    const std::vector<std::uint32_t> table = WordsOf(map.PageTable());
    EXPECT_EQ(Backed(table), used);
    EXPECT_GT(used, 4u);
    EXPECT_LT(used, 256u);
    // Only levels 6 and 7 (5 to 15 m, biased by 4), the four coarser
    // that wide filters read, and the coarsest have pages.
    const std::size_t layer = 128 * 128;
    for (std::uint32_t l = 0; l < map.LevelCount(); ++l) {
        const std::vector<std::uint32_t> one(
            table.begin() + static_cast<std::ptrdiff_t>(l * layer),
            table.begin() +
                static_cast<std::ptrdiff_t>((l + 1) * layer));
        const int level = -2 + static_cast<int>(l);
        if (level >= 6 && level <= 11) {
            EXPECT_GT(Backed(one), 0u) << "level " << level;
        } else if (level == 12) {
            EXPECT_EQ(Backed(one), 4u) << "level " << level;
        } else {
            EXPECT_EQ(Backed(one), 0u) << "level " << level;
        }
    }
    // Distinct physical pages, each cleared; the rest keep the junk.
    std::vector<int> seen(256, 0);
    const std::vector<std::uint32_t> pool = WordsOf(map.Pool());
    for (const std::uint32_t entry : table) {
        if ((entry >> 31) == 0) {
            continue;
        }
        const std::uint32_t page = entry & 0xfffff;
        ASSERT_LT(page, 256u);
        ++seen[page];
        EXPECT_EQ(pool[page * 128 * 128 + 5 * 128 + 9], 0u);
    }
    for (std::uint32_t p = 0; p < 256; ++p) {
        EXPECT_LE(seen[p], 1) << "page " << p;
    }
    if (used < 256) {
        EXPECT_EQ(pool[255 * 128 * 128], 7u);
    }
}

// More pages wanted than the pool holds: the counter counts them all,
// only the pool's back pages.
TEST_F(VirtualShadowMapTest, AnOverflowingPoolBacksWhatItHolds) {
    ernie::ShaderCompiler compiler;
    VirtualShadowMap map(*context_, compiler, -2, 8, 4);
    const glm::dvec3 eye(0.0, 0.0, 5.0);
    map.SetLevels(VirtualShadowMap::FitLevels(
                      glm::dvec3(0.0, 0.0, 1.0), eye, eye, -2, 8),
                  0.0f);
    glm::mat4 inverse(0.0f);
    inverse[0][0] = 10.0f;
    inverse[1][1] = 10.0f;
    inverse[3][2] = -5.0f;
    inverse[3][3] = 1.0f;
    const GpuImage depth = FlatDepth(*context_, 64, 48, 0.5f);
    context_->Submit([&](VkCommandBuffer commands) {
        map.RecordMark(commands, depth.View(), 64, 48, inverse,
                       glm::vec3(0.0f));
        map.RecordAllocate(commands);
    });
    std::uint32_t used = 0;
    map.Counter().Read(std::as_writable_bytes(std::span(&used, 1)));
    EXPECT_GT(used, 4u);
    EXPECT_EQ(Backed(WordsOf(map.PageTable())), 4u);
}

namespace {

/// Pixels of `image` (RGBA8) darker than half its median green: the
/// shadow on a bright floor.
std::size_t DarkPixels(const ernie::Tensor& image) {
    const auto* pixels = image.As<std::uint8_t>();
    const auto count = static_cast<std::size_t>(image.Elements() / 4);
    std::vector<int> green;
    for (std::size_t i = 0; i < count; ++i) {
        green.push_back(pixels[i * 4 + 1]);
    }
    std::vector<int> sorted = green;
    std::nth_element(sorted.begin(), sorted.begin() + count / 2,
                     sorted.end());
    const int median = sorted[count / 2];
    std::size_t dark = 0;
    for (const int g : green) {
        dark += g < median / 2 ? 1u : 0u;
    }
    return dark;
}

}  // namespace

// A sphere on a floor under a sun 30 degrees off overhead, seen from
// above: its shadow through the virtual shadow map about as large as
// through the cascades.
TEST_F(VirtualShadowMapTest,
       ASphereCastsItsShadowThroughTheVirtualMap) {
    using namespace ernie;
    namespace tokens = ernie::tokens;
    Scene scene;
    GroundPlane::Define(scene, "/World/Floor")
        .SetMaterialBinding("/Looks/Grey");
    Material::Define(scene, "/Looks/Grey").SetAlbedo(glm::dvec3(0.5));
    Sphere::Define(scene, "/World/Ball").SetRadius(2.0);
    scene.Set(
        "/World/Ball", tokens::kTransform,
        glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 0.0, 2.0)));
    DistantLight::Define(scene, "/World/Sun");
    scene.Set("/World/Sun", tokens::kTransform,
              DistantLightTransform({0.5, 0.0, 0.866}));
    Camera::Define(scene, "/World/Camera").SetSensorNoise(false);
    scene.Set("/World/Camera", tokens::kTransform,
              LookAtTransform({0.0, -8.0, 24.0}, {-1.0, 0.0, 0.0},
                              {0.0, 0.0, 1.0}));
    RenderSettings settings =
        RenderSettings::Define(scene, "/Render/Settings");
    settings.SetCamera("/World/Camera")
        .SetResolution(glm::dvec2(64.0, 64.0))
        .SetAmbient(0.0)
        .SetShadowVirtual(false);
    RasterDelegate delegate(*context_);
    const std::size_t cascades = DarkPixels(delegate.Render(scene));
    ASSERT_GT(cascades, 20u);
    settings.SetShadowVirtual(true);
    const std::size_t virtualMap = DarkPixels(delegate.Render(scene));
    EXPECT_GT(virtualMap, cascades / 2);
    EXPECT_LT(virtualMap, cascades * 2);

    // A pool of 8 pages, fewer than the frame asks for, grows to hold
    // them by the next frame, which shadows as with room to spare.
    settings.SetShadowVirtualPages(8);
    RasterDelegate small(*context_);
    small.Render(scene);
    const std::size_t grown = DarkPixels(small.Render(scene));
    ASSERT_NE(small.Renderer().VirtualShadow(), nullptr);
    EXPECT_GT(small.Renderer().VirtualShadow()->Pages(), 8u);
    EXPECT_EQ(grown, virtualMap);
}

// As on a floor, on an Earth-sized planet Terrain, which casts into
// the virtual map too.
TEST_F(VirtualShadowMapTest, ASphereCastsItsShadowOnAPlanet) {
    using namespace ernie;
    namespace tokens = ernie::tokens;
    constexpr double kEarth = 6371008.8;
    Scene scene;
    Terrain::Define(scene, "/World/Earth")
        .SetHeights(Tensor::Copy<float>(
            std::vector<float>(6 * 65 * 65, 0.0f), {6, 65, 65}))
        .SetPlanetRadius(kEarth)
        .SetMaterialBinding("/Looks/Grey");
    scene.Set("/World/Earth", tokens::kTransform,
              glm::translate(glm::dmat4(1.0),
                             glm::dvec3(0.0, 0.0, -kEarth)));
    Material::Define(scene, "/Looks/Grey").SetAlbedo(glm::dvec3(0.5));
    Sphere::Define(scene, "/World/Ball").SetRadius(2.0);
    scene.Set(
        "/World/Ball", tokens::kTransform,
        glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 0.0, 2.0)));
    DistantLight::Define(scene, "/World/Sun");
    scene.Set("/World/Sun", tokens::kTransform,
              DistantLightTransform({0.5, 0.0, 0.866}));
    Camera::Define(scene, "/World/Camera").SetSensorNoise(false);
    scene.Set("/World/Camera", tokens::kTransform,
              LookAtTransform({0.0, -8.0, 24.0}, {-1.0, 0.0, 0.0},
                              {0.0, 0.0, 1.0}));
    RenderSettings settings =
        RenderSettings::Define(scene, "/Render/Settings");
    settings.SetCamera("/World/Camera")
        .SetResolution(glm::dvec2(64.0, 64.0))
        .SetAmbient(0.0)
        .SetShadowVirtual(false);
    RasterDelegate delegate(*context_);
    const std::size_t cascades = DarkPixels(delegate.Render(scene));
    ASSERT_GT(cascades, 20u);
    settings.SetShadowVirtual(true);
    const std::size_t virtualMap = DarkPixels(delegate.Render(scene));
    EXPECT_GT(virtualMap, cascades / 2);
    EXPECT_LT(virtualMap, cascades * 2);
}

// The first frame picks its levels as every later one does: by the
// camera's own width (a pixel's footprint), not one the renderer
// learns later in the frame; a 1 m sphere's shadow on a planet the
// same from the first frame on.
TEST_F(VirtualShadowMapTest, TheFirstFrameShadowsAsTheNext) {
    using namespace ernie;
    namespace tokens = ernie::tokens;
    constexpr double kEarth = 6371008.8;
    Scene scene;
    Terrain::Define(scene, "/World/Earth")
        .SetHeights(Tensor::Copy<float>(
            std::vector<float>(6 * 65 * 65, 0.0f), {6, 65, 65}))
        .SetPlanetRadius(kEarth)
        .SetMaterialBinding("/Looks/Grey");
    scene.Set("/World/Earth", tokens::kTransform,
              glm::translate(glm::dmat4(1.0),
                             glm::dvec3(0.0, 0.0, -kEarth)));
    Material::Define(scene, "/Looks/Grey").SetAlbedo(glm::dvec3(0.5));
    Sphere::Define(scene, "/World/Ball").SetRadius(1.0);
    scene.Set(
        "/World/Ball", tokens::kTransform,
        glm::translate(glm::dmat4(1.0), glm::dvec3(-12.0, 0.0, 1.0)));
    DistantLight::Define(scene, "/World/Sun");
    scene.Set("/World/Sun", tokens::kTransform,
              DistantLightTransform({0.5, 0.0, 0.866}));
    Camera::Define(scene, "/World/Camera").SetSensorNoise(false);
    scene.Set("/World/Camera", tokens::kTransform,
              LookAtTransform({-12.6, 9.0, 4.0}, {-12.6, 0.0, 0.3},
                              {0.0, 0.0, 1.0}));
    RenderSettings::Define(scene, "/Render/Settings")
        .SetCamera("/World/Camera")
        .SetResolution(glm::dvec2(160.0, 90.0))
        .SetAmbient(0.0)
        .SetShadowVirtual(true);
    RasterDelegate delegate(*context_);
    const std::size_t first = DarkPixels(delegate.Render(scene));
    const std::size_t next = DarkPixels(delegate.Render(scene));
    EXPECT_GT(first, 0u);
    EXPECT_EQ(first, next);
}

namespace {

/// Pixels of `image` (RGBA8) in a shadow's penumbra: green between a
/// quarter and three quarters of the image's brightest.
std::size_t PenumbraPixels(const ernie::Tensor& image) {
    const auto* pixels = image.As<std::uint8_t>();
    const auto count = static_cast<std::size_t>(image.Elements() / 4);
    int lit = 0;
    for (std::size_t i = 0; i < count; ++i) {
        lit = std::max<int>(lit, pixels[i * 4 + 1]);
    }
    std::size_t penumbra = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const int g = pixels[i * 4 + 1];
        penumbra += g > lit / 4 && g < 3 * lit / 4 ? 1u : 0u;
    }
    return penumbra;
}

/// A 2 m square roof `height` metres over a floor, under a sun of
/// angular diameter `angle` degrees straight above, seen from above
/// through the virtual shadow map.
ernie::Tensor RoofShadow(ernie::RasterDelegate& delegate,
                         double height, double angle) {
    using namespace ernie;
    namespace tokens = ernie::tokens;
    Scene scene;
    GroundPlane::Define(scene, "/World/Floor")
        .SetMaterialBinding("/Looks/Grey");
    Material::Define(scene, "/Looks/Grey").SetAlbedo(glm::dvec3(0.5));
    ernie::testing::DefineQuad(scene, "/World/Roof", 1.0f);
    scene.Set("/World/Roof", tokens::kTransform,
              glm::translate(glm::dmat4(1.0),
                             glm::dvec3(0.0, 0.0, height)));
    DistantLight::Define(scene, "/World/Sun").SetAngle(angle);
    scene.Set("/World/Sun", tokens::kTransform,
              DistantLightTransform({0.0, 0.0, 1.0}));
    Camera::Define(scene, "/World/Camera")
        .SetSensorNoise(false)
        .SetClippingRange(glm::dvec2(0.1, 1000.0));
    // Below the roof's height, looking up under it at the floor would
    // hide nothing: from the side and above, the floor beside it.
    scene.Set("/World/Camera", tokens::kTransform,
              LookAtTransform({0.0, -6.0, height + 2.0},
                              {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}));
    RenderSettings::Define(scene, "/Render/Settings")
        .SetCamera("/World/Camera")
        .SetResolution(glm::dvec2(96.0, 96.0))
        .SetAmbient(0.0)
        .SetShadowVirtual(true);
    return delegate.Render(scene);
}

}  // namespace

// The sun's angular size softens the virtual map's shadows as it
// does the cascades': the penumbra widens with the caster's height
// over the receiver, and a point sun's stays a texel or two.
TEST_F(VirtualShadowMapTest, TheSunsSizeWidensThePenumbraWithHeight) {
    ernie::RasterDelegate delegate(*context_);
    const std::size_t low =
        PenumbraPixels(RoofShadow(delegate, 0.5, 5.0));
    const std::size_t high =
        PenumbraPixels(RoofShadow(delegate, 3.0, 5.0));
    const std::size_t point =
        PenumbraPixels(RoofShadow(delegate, 3.0, 0.0));
    EXPECT_GT(high, 2 * low);
    EXPECT_GT(high, 3 * point);
}

namespace {

/// The shadow's dark pixels through the cascades and then through the
/// virtual map, for a scene seen from above under a sun 30 degrees
/// off overhead.
std::pair<std::size_t, std::size_t>
CascadesThenVirtual(const ernie::VulkanContext& context,
                    ernie::Scene& scene) {
    using namespace ernie;
    namespace tokens = ernie::tokens;
    DistantLight::Define(scene, "/World/Sun");
    scene.Set("/World/Sun", tokens::kTransform,
              DistantLightTransform({0.5, 0.0, 0.866}));
    Camera::Define(scene, "/World/Camera").SetSensorNoise(false);
    scene.Set("/World/Camera", tokens::kTransform,
              LookAtTransform({0.0, -8.0, 24.0}, {-1.0, 0.0, 0.0},
                              {0.0, 0.0, 1.0}));
    RenderSettings settings =
        RenderSettings::Define(scene, "/Render/Settings");
    settings.SetCamera("/World/Camera")
        .SetResolution(glm::dvec2(64.0, 64.0))
        .SetAmbient(0.0);
    RasterDelegate delegate(context);
    const std::size_t cascades = DarkPixels(delegate.Render(scene));
    settings.SetShadowVirtual(true);
    return {cascades, DarkPixels(delegate.Render(scene))};
}

}  // namespace

// A point cloud's spheres cast into the virtual map as into the
// cascades.
TEST_F(VirtualShadowMapTest, PointsCastThroughTheVirtualMap) {
    using namespace ernie;
    Scene scene;
    GroundPlane::Define(scene, "/World/Floor")
        .SetMaterialBinding("/Looks/Grey");
    Material::Define(scene, "/Looks/Grey").SetAlbedo(glm::dvec3(0.5));
    const std::vector<float> points{0.0f, 0.0f, 3.0f};
    Points::Define(scene, "/World/Grains")
        .SetPoints(Tensor::Copy<float>(points, {1, 3}))
        .SetWidths({4.0f});
    const auto [cascades, virtualMap] =
        CascadesThenVirtual(*context_, scene);
    ASSERT_GT(cascades, 20u);
    EXPECT_GT(virtualMap, cascades / 2);
    EXPECT_LT(virtualMap, cascades * 2);
}

// A terrain's block shadows its own ground through the virtual map as
// through the cascades.
TEST_F(VirtualShadowMapTest, TerrainCastsThroughTheVirtualMap) {
    using namespace ernie;
    using ernie::testing::TestRaster;
    TestRaster raster = TestRaster::Centred(33, 33, 1.0);
    for (std::size_t row = 14; row < 19; ++row) {
        for (std::size_t col = 14; col < 19; ++col) {
            raster(col, row) = 4.0f;
        }
    }
    Scene scene;
    Terrain::Define(scene, "/World/Terrain")
        .SetHeights(ernie::testing::HeightsOf(raster))
        .SetOrigin(glm::dvec2(raster.originX, raster.originY))
        .SetCellSize(1.0)
        .SetMaterialBinding("/Looks/Grey");
    Material::Define(scene, "/Looks/Grey").SetAlbedo(glm::dvec3(0.5));
    const auto [cascades, virtualMap] =
        CascadesThenVirtual(*context_, scene);
    ASSERT_GT(cascades, 20u);
    EXPECT_GT(virtualMap, cascades / 2);
    EXPECT_LT(virtualMap, cascades * 2);
}
