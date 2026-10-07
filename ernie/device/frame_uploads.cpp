#include "ernie/device/frame_uploads.h"

#include "ernie/device/vulkan_context.h"
#include "ernie/util/profile.h"

#include <algorithm>
#include <stdexcept>

namespace ernie {

namespace {

/// Staging grows in chunks of at least this many bytes.
constexpr VkDeviceSize kMinChunk = VkDeviceSize{1} << 20;

/// Staged ranges start at multiples of this, so typed reads of a chunk stay aligned.
constexpr VkDeviceSize kAlignment = 16;

/// Every earlier command's memory access before every later command's.
void FullBarrier(VkCommandBuffer commands) {
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commands, &dependency);
}

}  // namespace

FrameUploads::FrameUploads(const VulkanContext& context) : context_(context) {}

void FrameUploads::Begin(std::uint32_t slot) {
    ERNIE_ZONE("FrameUploads::Begin");
    if (slot >= kSlots) {
        throw std::out_of_range("FrameUploads::Begin past the last slot");
    }
    // Writes not yet recorded stage in another slot's memory, which the next frame to use that
    // slot reuses: read them back before moving them here.
    std::vector<std::pair<Copy, std::vector<std::byte>>> unrecorded;
    for (const Copy& copy : copies_) {
        std::vector<std::byte> bytes(static_cast<std::size_t>(copy.region.size));
        copy.source->Read(bytes, copy.region.srcOffset);
        unrecorded.emplace_back(copy, std::move(bytes));
    }
    copies_.clear();
    slot_ = slot;
    Staging& next = slots_[slot_];
    // Its frame has finished, so its staging is free: one chunk as large as all of it was.
    if (next.chunks.size() > 1) {
        VkDeviceSize total = 0;
        for (const std::unique_ptr<GpuBuffer>& chunk : next.chunks) {
            total += chunk->Size();
        }
        next.chunks.clear();
        next.chunks.push_back(
            std::make_unique<GpuBuffer>(context_, total, 0, BufferMemory::Upload));
    }
    next.used = 0;
    for (auto& [copy, bytes] : unrecorded) {
        const auto [chunk, offset] = Stage(bytes);
        copy.source = chunk;
        copy.region.srcOffset = offset;
        copies_.push_back(std::move(copy));
    }
}

std::pair<GpuBuffer*, VkDeviceSize> FrameUploads::Stage(std::span<const std::byte> bytes) {
    Staging& slot = slots_[slot_];
    VkDeviceSize offset = (slot.used + kAlignment - 1) / kAlignment * kAlignment;
    if (slot.chunks.empty() || offset + bytes.size() > slot.chunks.back()->Size()) {
        const VkDeviceSize grown = slot.chunks.empty() ? 0 : 2 * slot.chunks.back()->Size();
        const VkDeviceSize size = std::max({kMinChunk, grown, VkDeviceSize{bytes.size()}});
        slot.chunks.push_back(std::make_unique<GpuBuffer>(context_, size, 0, BufferMemory::Upload));
        offset = 0;
    }
    GpuBuffer& chunk = *slot.chunks.back();
    chunk.Write(bytes, offset);
    slot.used = offset + bytes.size();
    return {&chunk, offset};
}

void FrameUploads::Write(const GpuBuffer& destination, std::span<const std::byte> bytes,
                         VkDeviceSize offset) {
    if (offset > destination.Size() || bytes.size() > destination.Size() - offset) {
        throw std::out_of_range("FrameUploads::Write past the end of the buffer");
    }
    if (bytes.empty()) {
        return;
    }
    const auto [chunk, staged] = Stage(bytes);
    copies_.push_back({destination.Handle(), destination.Lifetime(), chunk,
                       VkBufferCopy{staged, offset, bytes.size()}});
}

void FrameUploads::Record(VkCommandBuffer commands) {
    ERNIE_ZONE("FrameUploads::Record");
    FullBarrier(commands);
    // Consecutive copies between the same pair of buffers go in one command.
    for (std::size_t first = 0; first < copies_.size();) {
        std::size_t last = first + 1;
        while (last < copies_.size() && copies_[last].source == copies_[first].source &&
               copies_[last].destination == copies_[first].destination) {
            ++last;
        }
        if (!copies_[first].lifetime.expired()) {
            std::vector<VkBufferCopy> regions;
            for (std::size_t c = first; c < last; ++c) {
                regions.push_back(copies_[c].region);
            }
            vkCmdCopyBuffer(commands, copies_[first].source->Handle(), copies_[first].destination,
                            static_cast<std::uint32_t>(regions.size()), regions.data());
        }
        first = last;
    }
    copies_.clear();
    FullBarrier(commands);
}

VkDeviceSize FrameUploads::Pending() const {
    VkDeviceSize bytes = 0;
    for (const Copy& copy : copies_) {
        bytes += copy.region.size;
    }
    return bytes;
}

}  // namespace ernie
