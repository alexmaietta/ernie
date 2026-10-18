// SPDX-License-Identifier: BSD-3-Clause
/// @file arena.cpp
/// @brief Sub-allocating arenas with VMA's virtual blocks.
#include "ernie/device/arena.h"

#include <algorithm>
#include <optional>

namespace ernie {

Arenas::Arena::~Arena() {
    if (block != VK_NULL_HANDLE) {
        vmaClearVirtualBlock(block);
        vmaDestroyVirtualBlock(block);
    }
}

Arenas::Arenas(DeviceContext& context, MemoryUse use, std::string_view name,
               std::uint64_t arenaSize)
    : context_(context), use_(use), name_(name),
      arenaSize_(std::min(arenaSize, context.MaxBufferSize())) {}

Arenas::~Arenas() {
    Quietly([this] { context_.WaitIdle(); });
}

BufferRange Arenas::Allocate(std::uint64_t size, std::uint64_t alignment) {
    if (size > context_.MaxBufferSize()) {
        throw DeviceError(DeviceErrorKind::Unsupported,
                          std::to_string(size) + " bytes of " + name_ +
                              " exceed the largest buffer the device can address");
    }
    size = std::max<std::uint64_t>(size, 1);
    const VmaVirtualAllocationCreateInfo info{.size = size, .alignment = alignment};
    const auto take = [&](std::uint32_t index) -> std::optional<BufferRange> {
        Arena& arena = *arenas_[index];
        VmaVirtualAllocation allocation = VK_NULL_HANDLE;
        VkDeviceSize offset = 0;
        if (vmaVirtualAllocate(arena.block, &info, &allocation, &offset) != VK_SUCCESS) {
            return std::nullopt;
        }
        arena.allocations.emplace(offset, allocation);
        arena.used += size;
        return BufferRange{index, offset, size};
    };
    const auto takeAny = [&]() -> std::optional<BufferRange> {
        for (std::uint32_t index = 0; index < arenas_.size(); ++index) {
            if (const std::optional<BufferRange> range = take(index)) {
                return range;
            }
        }
        return std::nullopt;
    };
    {
        const std::lock_guard lock(mutex_);
        if (const std::optional<BufferRange> range = takeAny()) {
            return *range;
        }
    }
    // Ranges freed earlier come back once their work is done; collected without the lock, as
    // giving them back takes it.
    context_.Collect();
    const std::lock_guard lock(mutex_);
    if (const std::optional<BufferRange> range = takeAny()) {
        return *range;
    }
    // A range larger than the arena size gets an arena of its own.
    const std::uint64_t own = std::max(size, arenaSize_);
    if (const std::optional<BufferRange> range = take(AddArena(own))) {
        return *range;
    }
    throw DeviceError(DeviceErrorKind::Failed, "Allocating " + std::to_string(size) + " bytes of " +
                                                   name_ + " at alignment " +
                                                   std::to_string(alignment));
}

const Buffer& Arenas::Storage(const BufferRange& range) const {
    const std::lock_guard lock(mutex_);
    return arenas_[range.arena]->buffer;
}

void Arenas::Free(const BufferRange& range) {
    context_.Retire([this, range] { Release(range); });
}

std::uint64_t Arenas::Used() const {
    const std::lock_guard lock(mutex_);
    std::uint64_t used = 0;
    for (const auto& arena : arenas_) {
        used += arena->used;
    }
    return used;
}

std::uint32_t Arenas::AddArena(std::uint64_t size) {
    auto arena = std::make_unique<Arena>();
    arena->buffer =
        Buffer(context_, size, use_, name_ + " arena " + std::to_string(arenas_.size()));
    const VmaVirtualBlockCreateInfo block{.size = size};
    CheckVk(vmaCreateVirtualBlock(&block, &arena->block), "Creating " + name_ + "'s arena");
    arenas_.push_back(std::move(arena));
    return static_cast<std::uint32_t>(arenas_.size() - 1);
}

void Arenas::Release(const BufferRange& range) {
    const std::lock_guard lock(mutex_);
    Arena& arena = *arenas_[range.arena];
    const auto found = arena.allocations.find(range.offset);
    if (found != arena.allocations.end()) {
        vmaVirtualFree(arena.block, found->second);
        arena.allocations.erase(found);
        arena.used -= range.size;
    }
}

}  // namespace ernie
