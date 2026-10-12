// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/shader_compiler.h"
#include "ernie/raster/virtual_shadow_map.h"
#include "support/vulkan_fixture.h"

#include <glm/glm.hpp>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using ernie::GpuBuffer;
using ernie::GpuImage;
using ernie::VirtualShadowMap;
using namespace ernie::testing;

namespace {

class VirtualShadowMapTest : public VulkanTest {};

/// Every texel of an R32_UINT image, layer after layer.
std::vector<std::uint32_t>
TexelsOf(const ernie::VulkanContext& context,
         const GpuImage& source) {
    auto& image = const_cast<GpuImage&>(source);  // only relaid out
    const ernie::ImageDesc& desc = image.Desc();
    const VkDeviceSize count =
        VkDeviceSize{desc.width} * desc.height * desc.layers;
    GpuBuffer buffer(context, count * sizeof(std::uint32_t),
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     ernie::BufferMemory::Readback);
    context.Submit([&](VkCommandBuffer commands) {
        image.Transition(commands,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0,
                                   desc.layers};
        region.imageExtent = {desc.width, desc.height, 1};
        vkCmdCopyImageToBuffer(commands, image.Handle(),
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               buffer.Handle(), 1, &region);
    });
    std::vector<std::uint32_t> texels(count);
    buffer.Read(std::as_writable_bytes(std::span(texels)));
    return texels;
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

std::uint32_t Backed(const std::vector<std::uint32_t>& table) {
    std::uint32_t backed = 0;
    for (const std::uint32_t entry : table) {
        backed += (entry >> 31) != 0 ? 1u : 0u;
    }
    return backed;
}

}  // namespace

// A 20 m square of ground seen from 5 m above its centre: its points
// mark pages in the levels suiting 5 to 15 m, plus the coarsest
// level's four about the camera; allocation backs each once, in
// distinct physical pages, and clearing zeroes exactly those.
TEST_F(VirtualShadowMapTest, MarksBacksAndClearsTheSeenPages) {
    ernie::ShaderCompiler compiler;
    VirtualShadowMap map(*context_, compiler, -2, 12, 256);
    const glm::dvec3 eye(0.0, 0.0, 5.0);
    const auto levels = VirtualShadowMap::FitLevels(
        glm::normalize(glm::dvec3(0.3, 0.2, 1.0)), eye, eye, -2, 12);
    // Texels about 3 cm at 5 m: what a coarse depth buffer suits.
    map.SetLevels(levels, 4.0f);
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
        auto& pool = const_cast<GpuImage&>(map.Pool());
        pool.Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
        VkClearColorValue junk{};
        junk.uint32[0] = 7;
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT,
                                            0, 1, 0, 1};
        vkCmdClearColorImage(commands, pool.Handle(),
                             VK_IMAGE_LAYOUT_GENERAL, &junk, 1,
                             &range);
        map.RecordMark(commands, depth.View(), 64, 48, inverse,
                       glm::vec3(0.0f));
        map.RecordAllocate(commands);
        map.RecordClear(commands);
    });
    std::uint32_t used = 0;
    map.Counter().Read(std::as_writable_bytes(std::span(&used, 1)));
    const std::vector<std::uint32_t> table =
        TexelsOf(*context_, map.PageTable());
    EXPECT_EQ(Backed(table), used);
    EXPECT_GT(used, 4u);
    EXPECT_LT(used, 256u);
    // Only levels 6 and 7 (5 to 15 m, biased by 4) and the coarsest
    // have pages.
    const std::size_t layer = 128 * 128;
    for (std::uint32_t l = 0; l < map.LevelCount(); ++l) {
        const std::vector<std::uint32_t> one(
            table.begin() + static_cast<std::ptrdiff_t>(l * layer),
            table.begin() +
                static_cast<std::ptrdiff_t>((l + 1) * layer));
        const int level = -2 + static_cast<int>(l);
        if (level == 6 || level == 7) {
            EXPECT_GT(Backed(one), 0u) << "level " << level;
        } else if (level == 12) {
            EXPECT_EQ(Backed(one), 4u) << "level " << level;
        } else {
            EXPECT_EQ(Backed(one), 0u) << "level " << level;
        }
    }
    // Distinct physical pages, each cleared; the rest keep the junk.
    std::vector<int> seen(256, 0);
    const std::vector<std::uint32_t> pool =
        TexelsOf(*context_, map.Pool());
    const std::uint32_t width = map.RowPages() * 128;
    for (const std::uint32_t entry : table) {
        if ((entry >> 31) == 0) {
            continue;
        }
        const std::uint32_t x = entry & 0x3ff;
        const std::uint32_t y = (entry >> 10) & 0x3ff;
        ++seen[y * map.RowPages() + x];
        EXPECT_EQ(pool[(y * 128 + 5) * width + x * 128 + 9], 0u);
    }
    for (std::uint32_t p = 0; p < 256; ++p) {
        EXPECT_LE(seen[p], 1) << "page " << p;
    }
    const std::uint32_t last = 255;
    if (used <= last) {
        EXPECT_EQ(pool[(last / map.RowPages() * 128) * width +
                       last % map.RowPages() * 128],
                  7u);
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
    EXPECT_EQ(Backed(TexelsOf(*context_, map.PageTable())), 4u);
}
