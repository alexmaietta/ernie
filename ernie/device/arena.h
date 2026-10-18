// SPDX-License-Identifier: BSD-3-Clause
/// @file arena.h
/// @brief Device memory as a few large buffers sub-allocated by offset. Shaders reach data as
/// (arena, offset): a buffer device address on Vulkan, a bound storage buffer and offset on
/// WebGPU, which has no pointers.
#pragma once

#include "ernie/device/buffer.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace ernie {

/// @brief Where data lives on the device: a range of one arena.
struct BufferRange {
    std::uint32_t arena = 0;
    std::uint64_t offset = 0;
    std::uint64_t size = 0;
    bool operator==(const BufferRange&) const = default;
};

/// @brief The arenas of one memory use. Ranges are handed out by offset; a full arena is joined
/// by another, and a range larger than the arena size gets an arena of its own. No arena is
/// larger than the device's largest addressable buffer.
class Arenas {
  public:
    /// @param arenaSize Bytes per arena, capped at `DeviceContext::MaxBufferSize`.
    Arenas(DeviceContext& context, MemoryUse use, std::string_view name,
           std::uint64_t arenaSize = std::uint64_t{256} << 20);
    /// Waits for the device, so ranges freed late are back before the arenas go.
    ~Arenas();
    Arenas(const Arenas&) = delete;
    Arenas& operator=(const Arenas&) = delete;

    /// @brief Takes a range of `size` bytes at a multiple of `alignment`.
    /// @throws DeviceError when `size` exceeds the largest buffer the device can address, or the
    /// memory runs out.
    BufferRange Allocate(std::uint64_t size, std::uint64_t alignment = 16);
    /// @brief Gives a range back once every queue's work submitted so far is done.
    void Free(const BufferRange& range);

    /// @return The buffer of a range's arena: its handle, and its mapping for host-visible uses.
    const Buffer& Storage(const BufferRange& range) const {
        return arenas_[range.arena]->buffer;
    }
    /// @return The range's device address, as shaders read it on Vulkan.
    VkDeviceAddress Address(const BufferRange& range) const {
        return Storage(range).Address() + range.offset;
    }
    std::size_t Count() const {
        return arenas_.size();
    }
    /// @return The bytes handed out and not yet given back.
    std::uint64_t Used() const;

  private:
    struct Arena {
        Buffer buffer;
        VmaVirtualBlock block = VK_NULL_HANDLE;
        /// Each range's allocation in `block`, by offset.
        std::unordered_map<std::uint64_t, VmaVirtualAllocation> allocations;
        std::uint64_t used = 0;
        ~Arena();
    };

    std::uint32_t AddArena(std::uint64_t size);
    void Release(const BufferRange& range);

    DeviceContext& context_;
    MemoryUse use_;
    std::string name_;
    std::uint64_t arenaSize_;
    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<Arena>> arenas_;
};

}  // namespace ernie
