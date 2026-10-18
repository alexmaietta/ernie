// SPDX-License-Identifier: BSD-3-Clause
/// @file staging.cpp
/// @brief Staged uploads in batched copy regions, and readbacks.
#include "ernie/device/staging.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace ernie {
namespace {

/// Orders copies after earlier work that reads or writes what they touch.
void BeforeCopies(VkCommandBuffer commands) {
    Barrier(commands, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COPY_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
}

}  // namespace

Uploader::Uploader(DeviceContext& context, Queue& queue, std::uint64_t chunk)
    : context_(context), queue_(queue), chunk_(chunk) {}

Uploader::~Uploader() {
    Quietly([this] { Flush(); });
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
        FlushStaging();
        staging_.push_back(std::make_shared<Buffer>(
            context_, std::max<std::uint64_t>(chunk_, bytes.size()), MemoryUse::Upload, "staging"));
        used_ = 0;
    }
    const Buffer& staging = *staging_.back();
    std::memcpy(staging.Mapped().data() + used_, bytes.data(), bytes.size());
    copies_.push_back({staging_.size() - 1, destination.Handle(), {used_, offset, bytes.size()}});
    used_ += bytes.size();
    pending_ += bytes.size();
}

void Uploader::FlushStaging() {
    // Once a staging buffer is full or about to be copied from, not once a write.
    if (!staging_.empty() && used_ > 0) {
        staging_.back()->Flush(0, used_);
    }
}

void Uploader::WriteRows(const Buffer& destination, std::uint64_t offset, std::uint64_t rowBytes,
                         std::span<const std::byte> host, std::span<const std::uint32_t> rows) {
    for (std::size_t first = 0; first < rows.size();) {
        std::size_t last = first + 1;
        while (last < rows.size() && rows[last] == rows[last - 1] + 1) {
            ++last;
        }
        const std::uint64_t begin = rows[first];
        const std::uint64_t count = rows[last - 1] + 1 - begin;
        Write(destination, offset + begin * rowBytes,
              host.subspan(begin * rowBytes, count * rowBytes));
        first = last;
    }
}

Ticket Uploader::Flush(std::span<const Ticket> waits) {
    // Staging of earlier flushes goes once its copies are done.
    context_.Collect();
    if (copies_.empty()) {
        return {};
    }
    ERNIE_ZONE("Upload");
    ERNIE_PLOT("Uploaded bytes", pending_);
    FlushStaging();
    const Ticket ticket = queue_.Submit(
        [&](VkCommandBuffer commands) {
            BeforeCopies(commands);
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
            Barrier(commands, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, kAnyAccess);
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
            BeforeCopies(commands);
            const VkBufferCopy region{offset, 0, size};
            if (size > 0) {
                vkCmdCopyBuffer(commands, source.Handle(), host_->Handle(), 1, &region);
            }
            // Visible to the host's reads once the ticket is reached.
            Barrier(commands, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
        },
        waits);
}

Readback::~Readback() {
    context_.Retire(std::move(host_));
}

std::span<const std::byte> Readback::Bytes() {
    ERNIE_ZONE("Read back");
    ticket_.Wait();
    host_->Invalidate(0, host_->Size());
    return host_->Mapped();
}

}  // namespace ernie
