// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/buffer.h"
#include "ernie/device/pipeline.h"
#include "ernie/raster/virtual_texture.h"
#include "support/vulkan_fixture.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

using ernie::BufferMemory;
using ernie::GpuBuffer;
using ernie::SparseTile;
using ernie::TextureData;
using ernie::TextureFormat;
using ernie::VirtualTexture;

namespace {

/// An RGBA8 texture of `width` x `height` with every mip, each texel
/// holding its column, row and mip: (x & 0xff, y & 0xff, mip, 255).
std::shared_ptr<TextureData> Labelled(std::uint32_t width,
                                      std::uint32_t height) {
    auto texture = std::make_shared<TextureData>();
    texture->format = TextureFormat::Rgba8;
    texture->width = width;
    texture->height = height;
    std::uint32_t mips = 1;
    for (std::uint32_t side = std::max(width, height); side > 1;
         side >>= 1) {
        ++mips;
    }
    texture->mips = mips;
    for (std::uint32_t mip = 0; mip < mips; ++mip) {
        const std::uint32_t w = std::max(1u, width >> mip);
        const std::uint32_t h = std::max(1u, height >> mip);
        texture->offsets.push_back(texture->bytes.size());
        for (std::uint32_t y = 0; y < h; ++y) {
            for (std::uint32_t x = 0; x < w; ++x) {
                for (const std::uint32_t value :
                     {x & 0xffu, y & 0xffu, mip, 255u}) {
                    texture->bytes.push_back(
                        static_cast<std::byte>(value));
                }
            }
        }
    }
    return texture;
}

class VirtualTextureTest : public ernie::testing::VulkanTest {
  protected:
    void SetUp() override {
        VulkanTest::SetUp();
        if (!context_->SupportsSparseResidency()) {
            GTEST_SKIP() << "no sparse image residency";
        }
    }

    /// Texel (`x`, `y`) of `tile`, read back from the sparse image,
    /// as its four bytes.
    std::uint32_t TexelOf(const VirtualTexture& texture,
                          const SparseTile& tile, std::uint32_t x,
                          std::uint32_t y) const {
        const ernie::SparseImage& image = texture.Image();
        GpuBuffer texel(*context_, 4,
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        BufferMemory::Readback);
        context_->Submit([&](VkCommandBuffer commands) {
            ernie::TransitionImage(
                commands, image.Handle(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            VkBufferImageCopy region{};
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT,
                                       tile.mip, 0, 1};
            region.imageOffset = {
                static_cast<std::int32_t>(
                    tile.x * image.TileTexels().width + x),
                static_cast<std::int32_t>(
                    tile.y * image.TileTexels().height + y),
                0};
            region.imageExtent = {1, 1, 1};
            vkCmdCopyImageToBuffer(
                commands, image.Handle(),
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, texel.Handle(),
                1, &region);
            ernie::TransitionImage(
                commands, image.Handle(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        });
        std::uint32_t value = 0;
        texel.Read(std::as_writable_bytes(std::span(&value, 1)));
        return value;
    }
};

std::uint32_t Pack(std::uint32_t x, std::uint32_t y,
                   std::uint32_t mip) {
    return (x & 0xffu) | ((y & 0xffu) << 8) | (mip << 16) |
           (255u << 24);
}

}  // namespace

TEST_F(VirtualTextureTest, RequestsLoadTilesWithTheirAncestors) {
    VirtualTexture texture(*context_, Labelled(2048, 1024),
                           VK_FORMAT_R8G8B8A8_UNORM);
    const ernie::SparseImage& image = texture.Image();
    const std::uint32_t tail = image.TailMip();
    ASSERT_GE(tail, 2u) << "tiles in mips 0 and 1 at least";
    EXPECT_EQ(texture.ResidentTiles(), 0u) << "only the tail";
    EXPECT_EQ(texture.FinestResident(0, 0), tail);

    const SparseTile tile{0, 3, 2};
    texture.Request(tile);
    texture.BeginFrame(0, 1);
    // The tile and one ancestor in each coarser mip above the tail.
    EXPECT_EQ(texture.ResidentTiles(), tail);
    EXPECT_TRUE(image.Bound(tile));
    EXPECT_TRUE(image.Bound({1, 1, 1}));
    EXPECT_EQ(texture.FinestResident(3, 2), 0u);
    EXPECT_EQ(texture.FinestResident(2, 2), 1u) << "its mip-1 parent";
    EXPECT_EQ(texture.FinestResident(15, 7), tail)
        << "far from it, the tail";

    // The tiles hold the source's texels where they lie: the label of
    // texel (5, 7) of tile (3, 2) is its mip-0 position.
    const std::uint32_t x = 3 * image.TileTexels().width + 5;
    const std::uint32_t y = 2 * image.TileTexels().height + 7;
    EXPECT_EQ(TexelOf(texture, tile, 5, 7), Pack(x, y, 0));
    EXPECT_EQ(
        TexelOf(texture, {1, 1, 1}, 0, 0),
        Pack(image.TileTexels().width, image.TileTexels().height, 1));

    // The same frame again does nothing; a frame without requests
    // keeps what is resident.
    texture.BeginFrame(0, 1);
    texture.BeginFrame(1, 2);
    EXPECT_EQ(texture.ResidentTiles(), tail);
    EXPECT_THROW(texture.Request({0, 1000, 0}), std::out_of_range);
}

TEST_F(VirtualTextureTest, TheBudgetEvictsTheLeastRecentlyWanted) {
    // Room for three tiles, in a mip of 2 x 2 tiles whose requests
    // also keep their one parent tile wanted.
    auto source = Labelled(2048, 2048);
    VkDeviceSize tile = 0;
    {
        VirtualTexture probe(*context_, source,
                             VK_FORMAT_R8G8B8A8_UNORM);
        tile = probe.Image().TileMemoryBytes();
    }
    VirtualTexture texture(*context_, source,
                           VK_FORMAT_R8G8B8A8_UNORM, {}, 3 * tile);
    const ernie::SparseImage& image = texture.Image();
    const std::uint32_t mip = image.TailMip() - 2;
    ASSERT_EQ(image.TileCount(mip).width, 2u);
    texture.Request({mip, 0, 0});
    texture.BeginFrame(0, 1);
    EXPECT_EQ(texture.ResidentTiles(), 2u)
        << "the tile and its parent";
    texture.Request({mip, 1, 0});
    texture.BeginFrame(1, 2);
    EXPECT_EQ(texture.ResidentTiles(), 3u);
    texture.Request({mip, 0, 1});
    texture.BeginFrame(0, 3);
    EXPECT_EQ(texture.ResidentTiles(), 3u);
    // The least recently wanted tile leaves the maps at once, but
    // stays bound until the frame before, which may still sample it,
    // is done.
    const std::uint32_t x = 1u
                            << mip;  // Under mip `mip`'s tile (1, 0).
    EXPECT_GT(texture.FinestResident(0, 0), mip)
        << "no longer resident";
    EXPECT_EQ(texture.FinestResident(x, 0), mip)
        << "its neighbour is";
    EXPECT_TRUE(image.Bound({mip, 0, 0})) << "still bound";
    texture.BeginFrame(1, 4);
    EXPECT_FALSE(image.Bound({mip, 0, 0})) << "unbound a frame later";
    EXPECT_EQ(texture.ResidentTiles(), 3u);
    EXPECT_TRUE(image.Bound({mip, 1, 0}));
    EXPECT_TRUE(image.Bound({mip, 0, 1}));
    EXPECT_TRUE(image.Bound({mip + 1, 0, 0}))
        << "wanted by every request";
}
