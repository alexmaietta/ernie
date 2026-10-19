// SPDX-License-Identifier: BSD-3-Clause
/// @file commands.cpp
/// @brief Recording copies and buffer writes on Vulkan.
#include "ernie/device/commands.h"

#include "ernie/device/buffer.h"

#include <algorithm>
#include <vector>

namespace ernie {
namespace {

/// The most `vkCmdUpdateBuffer` writes at once.
constexpr std::uint64_t kUpdateBytes = 65536;

}  // namespace

void CommandList::Copy(std::span<const BufferRegions> copies) const {
    std::vector<std::vector<VkBufferCopy>> regions;
    std::vector<BufferCopies> vulkan;
    for (const BufferRegions& copy : copies) {
        std::vector<VkBufferCopy>& converted = regions.emplace_back();
        for (const CopyRegion& region : copy.regions) {
            converted.push_back({region.from, region.to, region.bytes});
        }
        vulkan.push_back({copy.source->Handle(), copy.destination->Handle(), converted});
    }
    CopyBuffers(commands_, vulkan);
}

void CommandList::Update(const Buffer& buffer, std::uint64_t offset,
                         std::span<const std::byte> bytes) const {
    for (std::uint64_t at = 0; at < bytes.size(); at += kUpdateBytes) {
        const std::uint64_t size = std::min<std::uint64_t>(kUpdateBytes, bytes.size() - at);
        vkCmdUpdateBuffer(commands_, buffer.Handle(), offset + at, size, bytes.data() + at);
    }
    Barrier(commands_, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, kAnyAccess);
}

}  // namespace ernie
