// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/buffer.h"
#include "ernie/device/pipeline.h"
#include "ernie/device/sparse_image.h"
#include "support/vulkan_fixture.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

using ernie::BufferMemory;
using ernie::GpuBuffer;
using ernie::SparseImage;
using ernie::SparseImageDesc;
using ernie::SparseTile;

namespace {

class SparseImageTest : public ernie::testing::VulkanTest {
  protected:
    void SetUp() override {
        VulkanTest::SetUp();
        if (!context_->SupportsSparseResidency()) {
            GTEST_SKIP() << "no sparse image residency";
        }
    }

    /// The RGBA8 texels of `tile`, read back from the image.
    std::vector<std::uint32_t>
    ReadTile(SparseImage& image, const SparseTile& tile) const {
        const VkExtent2D extent = image.TileExtent(tile);
        GpuBuffer texels(*context_, image.TileDataBytes(tile),
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
                static_cast<std::int32_t>(tile.x *
                                          image.TileTexels().width),
                static_cast<std::int32_t>(tile.y *
                                          image.TileTexels().height),
                0};
            region.imageExtent = {extent.width, extent.height, 1};
            vkCmdCopyImageToBuffer(
                commands, image.Handle(),
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, texels.Handle(),
                1, &region);
            ernie::TransitionImage(
                commands, image.Handle(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        });
        std::vector<std::uint32_t> values(texels.Size() / 4);
        texels.Read(std::as_writable_bytes(std::span(values)));
        return values;
    }
};

std::vector<std::uint32_t> Filled(std::size_t count,
                                  std::uint32_t value) {
    return std::vector<std::uint32_t>(count, value);
}

}  // namespace

TEST_F(SparseImageTest, TilesAreBoundWrittenAndReleasedOneByOne) {
    // 1000 x 600 RGBA8 with every mip: tiles of mip 0 overhang its
    // right and bottom edges.
    SparseImage image(
        *context_,
        SparseImageDesc{1000, 600, 10, VK_FORMAT_R8G8B8A8_UNORM});
    const VkExtent2D tile = image.TileTexels();
    ASSERT_GT(tile.width, 0u);
    const VkExtent2D count = image.TileCount(0);
    EXPECT_EQ(count.width, (1000 + tile.width - 1) / tile.width);
    EXPECT_EQ(count.height, (600 + tile.height - 1) / tile.height);
    const SparseTile corner{0, count.width - 1, count.height - 1};
    EXPECT_EQ(image.TileExtent(corner).width,
              1000 - (count.width - 1) * tile.width);
    EXPECT_EQ(image.TileExtent(corner).height,
              600 - (count.height - 1) * tile.height);
    EXPECT_LE(image.TailMip(), 10u);
    EXPECT_EQ(image.BoundTiles(), 0u);
    const VkDeviceSize tailOnly = image.Bytes();

    const SparseTile first{0, 0, 0};
    const std::vector<std::uint32_t> a =
        Filled(image.TileDataBytes(first) / 4, 0x11223344u);
    const std::vector<std::uint32_t> b =
        Filled(image.TileDataBytes(corner) / 4, 0x55667788u);
    const SparseTile tiles[] = {first, corner};
    const std::span<const std::byte> texels[] = {
        std::as_bytes(std::span(a)), std::as_bytes(std::span(b))};
    image.Write(tiles, texels);
    EXPECT_TRUE(image.Bound(first));
    EXPECT_TRUE(image.Bound(corner));
    EXPECT_FALSE(image.Bound({0, 1, 0}));
    EXPECT_EQ(image.Bytes(), tailOnly + 2 * image.TileMemoryBytes());
    EXPECT_EQ(ReadTile(image, first), a);
    EXPECT_EQ(ReadTile(image, corner), b);

    // Rewriting a bound tile keeps its memory.
    const std::vector<std::uint32_t> c =
        Filled(a.size(), 0x99aabbccu);
    const std::span<const std::byte> rewrite[] = {
        std::as_bytes(std::span(c))};
    image.Write(std::span(tiles, 1), rewrite);
    EXPECT_EQ(image.BoundTiles(), 2u);
    EXPECT_EQ(ReadTile(image, first), c);

    image.Release(std::span(tiles + 1, 1));
    EXPECT_FALSE(image.Bound(corner));
    EXPECT_EQ(image.Bytes(), tailOnly + image.TileMemoryBytes());
    EXPECT_EQ(ReadTile(image, first), c) << "the other tile stays";
}

TEST_F(SparseImageTest, TheMipTailIsWrittenWhole) {
    SparseImage image(
        *context_,
        SparseImageDesc{512, 512, 10, VK_FORMAT_R8G8B8A8_UNORM});
    if (image.TailMip() >= 10u) {
        // Lavapipe tiles every mip, leaving no tail.
        GTEST_SKIP() << "the device puts no mips in a tail";
    }
    std::vector<std::vector<std::uint32_t>> mips;
    std::vector<std::span<const std::byte>> spans;
    for (std::uint32_t mip = image.TailMip(); mip < 10; ++mip) {
        const std::uint32_t side = std::max(1u, 512u >> mip);
        mips.push_back(
            Filled(std::size_t{side} * side, 0xff000000u + mip));
    }
    for (const auto& mip : mips) {
        spans.push_back(std::as_bytes(std::span(mip)));
    }
    EXPECT_NO_THROW(image.WriteTail(spans));
    spans.pop_back();
    EXPECT_THROW(image.WriteTail(spans), std::invalid_argument);
    EXPECT_THROW(image.TileCount(image.TailMip()), std::out_of_range);
}

TEST_F(SparseImageTest, BlockCompressedTilesTakeWholeBlocks) {
    if (!context_->SupportsTextureCompressionBc()) {
        GTEST_SKIP() << "no BC textures";
    }
    SparseImage image(
        *context_,
        SparseImageDesc{2048, 1024, 12, VK_FORMAT_BC7_UNORM_BLOCK});
    const SparseTile tile{0, 0, 0};
    const VkExtent2D extent = image.TileExtent(tile);
    EXPECT_EQ(extent.width % 4, 0u);
    EXPECT_EQ(extent.height % 4, 0u);
    // 16 bytes a 4 x 4 block.
    EXPECT_EQ(image.TileDataBytes(tile),
              VkDeviceSize{extent.width / 4} * (extent.height / 4) *
                  16);
    const std::vector<std::byte> blocks(image.TileDataBytes(tile));
    const std::span<const std::byte> texels[] = {blocks};
    image.Write(std::span(&tile, 1), texels);
    EXPECT_TRUE(image.Bound(tile));

    const std::vector<std::byte> short_(blocks.size() - 16);
    const std::span<const std::byte> wrong[] = {short_};
    EXPECT_THROW(image.Write(std::span(&tile, 1), wrong),
                 std::invalid_argument);
}
