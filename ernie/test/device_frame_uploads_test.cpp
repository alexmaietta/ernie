#include "ernie/device/buffer.h"
#include "ernie/device/frame_uploads.h"
#include "support/vulkan_fixture.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <span>
#include <vector>

using ernie::BufferMemory;
using ernie::FrameUploads;
using ernie::GpuBuffer;

namespace {

class FrameUploadsTest : public ernie::testing::VulkanTest {
  protected:
    std::vector<std::uint32_t> Contents(const GpuBuffer& buffer) const {
        std::vector<std::uint32_t> values(buffer.Size() / sizeof(std::uint32_t));
        buffer.Read(std::as_writable_bytes(std::span(values)));
        return values;
    }

    void Record(FrameUploads& uploads) const {
        context_->Submit([&](VkCommandBuffer commands) { uploads.Record(commands); });
    }
};

std::span<const std::byte> Bytes(const std::vector<std::uint32_t>& values) {
    return std::as_bytes(std::span(values));
}

}  // namespace

TEST_F(FrameUploadsTest, WritesLandInOrderOnlyWhenRecorded) {
    FrameUploads uploads(*context_);
    GpuBuffer buffer(*context_, 4 * sizeof(std::uint32_t), 0, BufferMemory::Device);
    buffer.Write(Bytes({0, 0, 0, 0}));
    uploads.Begin(0);
    uploads.Write(buffer, Bytes({1, 2, 3, 4}));
    uploads.Write(buffer, Bytes({9}), 2 * sizeof(std::uint32_t));  // over the first write
    EXPECT_EQ(uploads.Pending(), 5 * sizeof(std::uint32_t));
    EXPECT_EQ(Contents(buffer), (std::vector<std::uint32_t>{0, 0, 0, 0})) << "not yet recorded";
    Record(uploads);
    EXPECT_EQ(uploads.Pending(), 0u);
    EXPECT_EQ(Contents(buffer), (std::vector<std::uint32_t>{1, 2, 9, 4}));
    EXPECT_THROW(uploads.Write(buffer, Bytes({1, 2, 3, 4, 5})), std::out_of_range);
}

TEST_F(FrameUploadsTest, UnrecordedWritesMoveToTheNextSlot) {
    FrameUploads uploads(*context_);
    GpuBuffer buffer(*context_, 2 * sizeof(std::uint32_t), 0, BufferMemory::Device);
    uploads.Begin(0);
    uploads.Write(buffer, Bytes({7}));
    uploads.Begin(1);  // slot 0's staging may now be reused
    uploads.Write(buffer, Bytes({8}), sizeof(std::uint32_t));
    uploads.Begin(0);
    uploads.Write(buffer, Bytes({5}), sizeof(std::uint32_t));  // after 8
    Record(uploads);
    EXPECT_EQ(Contents(buffer), (std::vector<std::uint32_t>{7, 5}));
}

TEST_F(FrameUploadsTest, WritesToReplacedBuffersAreDropped) {
    FrameUploads uploads(*context_);
    GpuBuffer buffer(*context_, sizeof(std::uint32_t), 0, BufferMemory::Device);
    uploads.Begin(0);
    uploads.Write(buffer, Bytes({3}));
    buffer = GpuBuffer(*context_, sizeof(std::uint32_t), 0, BufferMemory::Device);
    buffer.Write(Bytes({4}));
    {
        GpuBuffer gone(*context_, sizeof(std::uint32_t), 0, BufferMemory::Device);
        uploads.Write(gone, Bytes({6}));
    }
    Record(uploads);
    EXPECT_EQ(Contents(buffer), (std::vector<std::uint32_t>{4}));
}

TEST_F(FrameUploadsTest, StagingGrowsPastItsFirstChunk) {
    FrameUploads uploads(*context_);
    std::vector<std::uint32_t> large(1 << 19);  // 2 MiB, past the first 1 MiB chunk
    for (std::size_t i = 0; i < large.size(); ++i) {
        large[i] = static_cast<std::uint32_t>(i);
    }
    GpuBuffer small(*context_, sizeof(std::uint32_t), 0, BufferMemory::Device);
    GpuBuffer buffer(*context_, large.size() * sizeof(std::uint32_t), 0, BufferMemory::Device);
    for (const std::uint32_t slot : {0u, 1u, 0u}) {
        uploads.Begin(slot);
        uploads.Write(small, Bytes({slot + 1}));
        uploads.Write(buffer, Bytes(large));
        Record(uploads);
        EXPECT_EQ(Contents(buffer), large);
        EXPECT_EQ(Contents(small), (std::vector<std::uint32_t>{slot + 1}));
    }
}
