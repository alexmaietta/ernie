// SPDX-License-Identifier: BSD-3-Clause
/// @file staging.cpp
/// @brief Staged uploads in batched copy regions, and readbacks.
#include "ernie/device/staging.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace ernie {
namespace {

/// Orders the copies after earlier work (`before`) or before later work.
void MemoryBarrier(VkCommandBuffer commands, bool before) {
    const VkPipelineStageFlags2 all = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    const VkAccessFlags2 any = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    const VkMemoryBarrier2 barrier{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                                   .srcStageMask = before ? all : VK_PIPELINE_STAGE_2_COPY_BIT,
                                   .srcAccessMask = before ? VK_ACCESS_2_MEMORY_WRITE_BIT
                                                           : VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                   .dstStageMask = before ? VK_PIPELINE_STAGE_2_COPY_BIT : all,
                                   .dstAccessMask = before ? VK_ACCESS_2_TRANSFER_READ_BIT : any};
    const VkDependencyInfo dependency{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                      .memoryBarrierCount = 1,
                                      .pMemoryBarriers = &barrier};
    vkCmdPipelineBarrier2(commands, &dependency);
}

}  // namespace

Uploader::Uploader(DeviceContext& context, Queue& queue, std::uint64_t chunk)
    : context_(context), queue_(queue), chunk_(chunk) {}

Uploader::~Uploader() {
    Flush();
}

void Uploader::Write(const Buffer& destination, std::uint64_t offset,
                     std::span<const std::byte> bytes) {
    if (offset > destination.Size() || bytes.size() > destination.Size() - offset) {
        throw std::out_of_range("Writing past the end of a buffer");
    }
    if (bytes.empty()) {
        return;
    }
    context_.CountUpload(bytes.size());
    if (!destination.Mapped().empty()) {
        std::memcpy(destination.Mapped().data() + offset, bytes.data(), bytes.size());
        destination.Flush(offset, bytes.size());
        return;
    }
    if (staging_.empty() || used_ + bytes.size() > staging_.back()->Size()) {
        staging_.push_back(std::make_shared<Buffer>(
            context_, std::max<std::uint64_t>(chunk_, bytes.size()), MemoryUse::Upload, "staging"));
        used_ = 0;
    }
    const Buffer& staging = *staging_.back();
    std::memcpy(staging.Mapped().data() + used_, bytes.data(), bytes.size());
    staging.Flush(used_, bytes.size());
    copies_.push_back({staging_.size() - 1, destination.Handle(), {used_, offset, bytes.size()}});
    used_ += bytes.size();
    pending_ += bytes.size();
}

Ticket Uploader::Flush(std::span<const Ticket> waits) {
    if (copies_.empty()) {
        return {};
    }
    const Ticket ticket = queue_.Submit(
        [&](VkCommandBuffer commands) {
            // Regions run to the same buffers in one call each.
            std::size_t first = 0;
            while (first < copies_.size()) {
                std::size_t last = first + 1;
                std::vector<VkBufferCopy> regions{copies_[first].region};
                while (last < copies_.size() && copies_[last].staging == copies_[first].staging &&
                       copies_[last].destination == copies_[first].destination) {
                    regions.push_back(copies_[last++].region);
                }
                vkCmdCopyBuffer(commands, staging_[copies_[first].staging]->Handle(),
                                copies_[first].destination,
                                static_cast<std::uint32_t>(regions.size()), regions.data());
                first = last;
            }
            MemoryBarrier(commands, false);
        },
        waits);
    for (std::shared_ptr<Buffer>& staging : staging_) {
        context_.Retire(std::move(staging));
    }
    staging_.clear();
    copies_.clear();
    used_ = 0;
    pending_ = 0;
    return ticket;
}

Readback::Readback(DeviceContext& context, Queue& queue, const Buffer& source, std::uint64_t offset,
                   std::uint64_t size, std::span<const Ticket> waits)
    : context_(context), host_(std::make_shared<Buffer>(context, std::max<std::uint64_t>(size, 1),
                                                        MemoryUse::Readback, "readback")) {
    if (offset > source.Size() || size > source.Size() - offset) {
        throw std::out_of_range("Reading past the end of a buffer");
    }
    context.CountReadback(size);
    ticket_ = queue.Submit(
        [&](VkCommandBuffer commands) {
            MemoryBarrier(commands, true);
            const VkBufferCopy region{offset, 0, size};
            if (size > 0) {
                vkCmdCopyBuffer(commands, source.Handle(), host_->Handle(), 1, &region);
            }
            MemoryBarrier(commands, false);
        },
        waits);
}

Readback::~Readback() {
    context_.Retire(std::move(host_));
}

std::span<const std::byte> Readback::Bytes() {
    ticket_.Wait();
    host_->Invalidate(0, host_->Size());
    return host_->Mapped();
}

}  // namespace ernie
