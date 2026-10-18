// SPDX-License-Identifier: BSD-3-Clause
/// @file column.cpp
/// @brief Allocating device columns and growing them by device copies.
#include "ernie/device/column.h"

#include <algorithm>
#include <functional>
#include <numeric>

namespace ernie {
namespace {

void Barrier(VkCommandBuffer commands, VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
             VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
    const VkMemoryBarrier2 barrier{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                                   .srcStageMask = srcStage,
                                   .srcAccessMask = srcAccess,
                                   .dstStageMask = dstStage,
                                   .dstAccessMask = dstAccess};
    const VkDependencyInfo dependency{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                      .memoryBarrierCount = 1,
                                      .pMemoryBarriers = &barrier};
    vkCmdPipelineBarrier2(commands, &dependency);
}

}  // namespace

DeviceColumn::DeviceColumn(Arenas& arenas, std::string_view name, DType type,
                           std::vector<std::int64_t> element, std::uint64_t rows)
    : arenas_(arenas), name_(name), type_(type), element_(std::move(element)),
      rowBytes_(ByteSize(type) * std::accumulate(element_.begin(), element_.end(), std::uint64_t{1},
                                                 std::multiplies<>())),
      rows_(rows), range_(Allocate(rows)) {}

DeviceColumn::~DeviceColumn() {
    arenas_.Free(range_);
}

BufferRange DeviceColumn::Allocate(std::uint64_t rows) {
    // Arenas hand out no empty ranges.
    return arenas_.Allocate(std::max<std::uint64_t>(rowBytes_ * rows, 1));
}

Ticket DeviceColumn::Resize(std::uint64_t rows, Queue& queue, std::span<const Ticket> waits) {
    const BufferRange old = range_;
    range_ = Allocate(rows);
    const std::uint64_t kept = rowBytes_ * std::min(rows, rows_);
    rows_ = rows;
    Ticket ticket;
    if (kept > 0) {
        ticket = queue.Submit(
            [&](VkCommandBuffer commands) {
                const VkAccessFlags2 any =
                    VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
                Barrier(commands, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                        VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                        VK_ACCESS_2_TRANSFER_READ_BIT);
                const VkBufferCopy region{old.offset, range_.offset, kept};
                vkCmdCopyBuffer(commands, arenas_.Storage(old).Handle(),
                                arenas_.Storage(range_).Handle(), 1, &region);
                Barrier(commands, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, any);
            },
            waits);
    }
    arenas_.Free(old);
    return ticket;
}

}  // namespace ernie
