// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/buffer.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ernie {

class VulkanContext;

/// Host data bound for device buffers, staged for one frame in flight
/// and copied by that frame's own commands (`Record`), so the host
/// can prepare a frame while the device still runs earlier ones that
/// read the same buffers: the copies wait on the device for that
/// earlier work, in queue order, rather than the host waiting for it.
/// Writes keep their order, so a buffer written in part, or twice,
/// ends as if written in place.
///
/// Each of `kSlots` slots has its own staging memory. A frame takes a
/// slot with `Begin` once the frame that last used that slot has
/// finished; writes until its `Record` stage there.
///
/// References:
/// - Khronos, "Synchronization Examples", transfer then use in a
/// later submission.
///   https://github.com/KhronosGroup/Vulkan-Docs/wiki/Synchronization-Examples
/// - A. Kapoulkine, "niagara", per-frame staging of uploads.
/// https://github.com/zeux/niagara
class FrameUploads {
  public:
    /// Frames that may be in flight at once.
    static constexpr std::uint32_t kSlots = 2;

    explicit FrameUploads(const VulkanContext& context);

    /// Starts staging into slot `slot` (below `kSlots`), whose
    /// previous frame must have finished on the device. Writes staged
    /// since the last `Record` move into it.
    void Begin(std::uint32_t slot);

    /// The slot of the last `Begin`.
    std::uint32_t Slot() const {
        return slot_;
    }

    /// Stages `bytes` for `destination` at `offset`; the copy is
    /// recorded by the next `Record`, and dropped if `destination` is
    /// destroyed or replaced first.
    /// @throws std::out_of_range if the range does not fit the
    /// buffer.
    void Write(const GpuBuffer& destination,
               std::span<const std::byte> bytes,
               VkDeviceSize offset = 0);

    /// Records the staged copies, in order, between barriers: before,
    /// against every earlier command's access (including earlier
    /// submissions'), and after, making the copies visible to every
    /// later command. Empties the staged list.
    void Record(VkCommandBuffer commands);

    /// Bytes staged since the last `Record`.
    VkDeviceSize Pending() const;

  private:
    struct Copy {
        VkBuffer destination = VK_NULL_HANDLE;
        std::weak_ptr<const void> lifetime;
        const GpuBuffer* source =
            nullptr;  ///< A chunk of the slot's staging.
        VkBufferCopy region{};
    };
    struct Staging {
        std::vector<std::unique_ptr<GpuBuffer>> chunks;
        VkDeviceSize used = 0;  ///< Bytes used in the last chunk.
    };
    /// Staging for `bytes` more bytes in the current slot: its chunk
    /// and offset.
    std::pair<GpuBuffer*, VkDeviceSize>
    Stage(std::span<const std::byte> bytes);

    const VulkanContext& context_;
    std::array<Staging, kSlots> slots_;
    std::uint32_t slot_ = 0;
    std::vector<Copy> copies_;
};

}  // namespace ernie
