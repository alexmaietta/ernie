// SPDX-License-Identifier: BSD-3-Clause
/// @file commands.h
/// @brief What device calls record into, and the recording code above `device/` may do without
/// naming a graphics API: copy between buffers, and write bytes into one.
///
///     kernel.Dispatch(pass.commands, groups, parameters);
///     pass.commands.Update(buffer, offset, bytes);
#pragma once

#include "ernie/device/vulkan.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace ernie {

class Buffer;

/// @brief A range copied from one buffer offset to another, in bytes.
struct CopyRegion {
    std::uint64_t from = 0;
    std::uint64_t to = 0;
    std::uint64_t bytes = 0;
};

/// @brief Regions copied from one buffer to another, or within one.
struct BufferRegions {
    const Buffer* source = nullptr;
    const Buffer* destination = nullptr;
    std::span<const CopyRegion> regions;
};

/// @brief Commands being recorded for a queue. On Vulkan a command buffer, which converts to one;
/// another backend's encoder would take its place.
class CommandList {
  public:
    CommandList(VkCommandBuffer commands) : commands_(commands) {}

    VkCommandBuffer Handle() const {
        return commands_;
    }

    /// @brief Records copies between buffers, ordered after earlier work on them and before later
    /// work.
    void Copy(std::span<const BufferRegions> copies) const;
    /// @brief Records writing `bytes` at `offset` of a buffer, seen by the commands after it.
    void Update(const Buffer& buffer, std::uint64_t offset, std::span<const std::byte> bytes) const;

  private:
    VkCommandBuffer commands_;
};

}  // namespace ernie
