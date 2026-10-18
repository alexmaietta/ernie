// SPDX-License-Identifier: BSD-3-Clause
/// @file acceleration.h
/// @brief Hardware acceleration structures for ray queries: bottom levels over triangles or
/// procedural boxes read in place, top levels over instances a kernel writes, kept in a pool, built
/// in batches and compacted without host waits.
///
///     AccelerationPool pool(context, "scene");
///     AccelerationStructure mesh(pool, Sizes(context, triangles, kStatic), "mesh");
///     const BottomBuild build{.target = &mesh, .triangles = triangles};
///     // In a pass, after the inputs' writers and before the trace:
///     RecordBuilds(commands, std::span(&build, 1), {}, scratch);
///
/// Barriers between passes are the frame graph's: a pass that builds writes its targets, one that
/// builds a top level reads its bottom levels, and one that traces reads the top level.
///
/// References:
/// - Khronos, "VK_KHR_acceleration_structure" and "VK_KHR_ray_query", the Vulkan specification.
/// - NVIDIA, nvpro_core2's nvvk::AccelerationStructureBuilder (batched, budgeted, compacted
///   builds) and vk_lod_clusters (sizes read back without waiting), Apache-2.0.
#pragma once

#include "ernie/device/arena.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ernie {

/// @brief A place in a buffer an acceleration structure reads its input from.
struct BufferView {
    const Buffer* buffer = nullptr;
    std::uint64_t offset = 0;
};

/// @brief Triangles: float x, y, z positions `stride` bytes apart, three uint32 indices each.
struct TriangleGeometry {
    BufferView vertices;
    std::uint64_t stride = 3 * sizeof(float);
    std::uint32_t vertexCount = 0;
    BufferView indices;
    std::uint32_t triangleCount = 0;
};

/// @brief Procedural primitives as axis-aligned boxes, min then max corner in floats, which ray
/// queries report as candidates for the shader to intersect exactly.
struct BoxGeometry {
    BufferView boxes;
    std::uint64_t stride = 6 * sizeof(float);
    std::uint32_t count = 0;
};

/// @brief How a structure is built: tuned to trace or to build fast, and whether it may be
/// compacted once built or refitted in place.
struct BuildFlags {
    bool fastTrace = true;
    bool compactable = false;
    bool updatable = false;
};

/// Bottom levels that rarely change: traced fast and compacted.
inline constexpr BuildFlags kStatic{.fastTrace = true, .compactable = true};
/// Bottom levels that deform: built fast and refitted.
inline constexpr BuildFlags kDeforming{.fastTrace = false, .updatable = true};
/// Top levels rebuilt each step.
inline constexpr BuildFlags kRebuilt{.fastTrace = false};

/// @brief One instance of a top level, as the device reads it (VkAccelerationStructureInstanceKHR):
/// a row-major 3x4 transform, a 24-bit index shaders see and an 8-bit mask, a 24-bit offset and
/// 8 bits of flags, and the bottom level's address.
struct AccelerationInstance {
    float transform[3][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}};
    std::uint32_t indexAndMask = 0xff000000u;
    std::uint32_t offsetAndFlags = 0;
    std::uint64_t bottom = 0;
};
static_assert(sizeof(AccelerationInstance) == 64);

enum class AccelerationKind : std::uint8_t { Bottom, Top };

/// @brief What a structure needs: its storage, scratch for a build and for a refit, and the most
/// primitives (or instances) of each geometry its builds may hold.
struct AccelerationSizes {
    AccelerationKind kind = AccelerationKind::Bottom;
    BuildFlags flags;
    std::uint64_t storage = 0;
    std::uint64_t buildScratch = 0;
    std::uint64_t updateScratch = 0;
    std::vector<std::uint32_t> primitives;
};

/// @return What a bottom level over `triangles` or `boxes` (one kind only) needs.
/// @throws std::invalid_argument for no geometry, both kinds, or triangles without vertices.
/// @throws DeviceError on devices without ray queries.
AccelerationSizes Sizes(const DeviceContext& context, std::span<const TriangleGeometry> triangles,
                        BuildFlags flags);
AccelerationSizes Sizes(const DeviceContext& context, std::span<const BoxGeometry> boxes,
                        BuildFlags flags);
/// @return What a top level of up to `capacity` instances needs.
AccelerationSizes Sizes(const DeviceContext& context, std::uint32_t capacity, BuildFlags flags);

/// @brief Memory acceleration structures live in: large arenas handed out in 256-byte-aligned
/// ranges, so thousands of bottom levels share a few buffers. Ranges come back once the device is
/// done with them.
class AccelerationPool {
  public:
    /// @param arenaSize Bytes per arena; a larger structure gets an arena of its own.
    AccelerationPool(DeviceContext& context, std::string_view name,
                     std::uint64_t arenaSize = std::uint64_t{128} << 20);
    AccelerationPool(const AccelerationPool&) = delete;
    AccelerationPool& operator=(const AccelerationPool&) = delete;

    /// @brief Bytes in use and structures alive, by kind, for memory reports.
    struct Statistics {
        std::uint64_t bottomBytes = 0;
        std::uint64_t topBytes = 0;
        std::uint32_t bottoms = 0;
        std::uint32_t tops = 0;
    };
    const Statistics& Usage() const {
        return usage_;
    }
    DeviceContext& Context() const {
        return context_;
    }
    const std::string& Name() const {
        return name_;
    }
    /// @return The alignment every build's scratch needs.
    std::uint64_t ScratchAlignment() const {
        return scratchAlignment_;
    }

  private:
    friend class AccelerationStructure;
    BufferRange Allocate(AccelerationKind kind, std::uint64_t bytes);
    void Free(AccelerationKind kind, const BufferRange& range);

    DeviceContext& context_;
    std::string name_;
    Arenas arenas_;
    std::uint64_t scratchAlignment_;
    Statistics usage_;
};

/// @brief A bottom or top level in a pool. Destroying one retires it: the device may still trace
/// it, and its range and handle go once the work submitted so far is done.
class AccelerationStructure {
  public:
    /// @throws DeviceError when the pool cannot hold it.
    AccelerationStructure(AccelerationPool& pool, const AccelerationSizes& sizes,
                          std::string_view name);
    ~AccelerationStructure();
    AccelerationStructure(const AccelerationStructure&) = delete;
    AccelerationStructure& operator=(const AccelerationStructure&) = delete;

    AccelerationKind Kind() const {
        return sizes_.kind;
    }
    const AccelerationSizes& Sizes() const {
        return sizes_;
    }
    /// @return The address instances name a bottom level by, and kernels trace a top level by.
    std::uint64_t Address() const {
        return address_;
    }
    /// @return The buffer it lives in and its range there: what the frame graph tracks.
    const Buffer& Storage() const;
    const BufferRange& Range() const {
        return range_;
    }
    const std::string& Name() const {
        return name_;
    }
    const AccelerationPool& Pool() const {
        return pool_;
    }
    VkAccelerationStructureKHR Handle() const {
        return structure_;
    }

  private:
    AccelerationPool& pool_;
    AccelerationSizes sizes_;
    std::string name_;
    BufferRange range_;
    VkAccelerationStructureKHR structure_ = VK_NULL_HANDLE;
    std::uint64_t address_ = 0;
};

/// @brief A bottom level's build: from scratch, or a refit in place of one built `updatable`.
struct BottomBuild {
    const AccelerationStructure* target = nullptr;
    std::span<const TriangleGeometry> triangles;
    std::span<const BoxGeometry> boxes;
    bool update = false;
};

/// @brief A top level's build over `count` instances at `instances`, 16-byte aligned.
struct TopBuild {
    const AccelerationStructure* target = nullptr;
    BufferView instances;
    std::uint32_t count = 0;
    bool update = false;
};

/// @return The scratch `RecordBuilds` needs for these builds, each at its own aligned offset.
std::uint64_t ScratchBytes(std::span<const BottomBuild> bottoms, std::span<const TopBuild> tops);

/// @brief Records every build in one call, so the device runs them together. The builds must not
/// depend on one another: bottom levels a top level reads are built in an earlier call, with a
/// barrier between.
/// @param scratch At least `ScratchBytes`, used only by these builds until they finish.
/// @throws std::invalid_argument for a build larger than its target was sized for, an update of
/// a structure not built updatable, or a target of the wrong kind.
void RecordBuilds(VkCommandBuffer commands, std::span<const BottomBuild> bottoms,
                  std::span<const TopBuild> tops, BufferView scratch);

/// @brief The sizes structures compact to, written after their builds and read back without
/// waiting once the device has written them.
class CompactionQuery {
  public:
    CompactionQuery(DeviceContext& context, std::uint32_t capacity);
    ~CompactionQuery();
    CompactionQuery(const CompactionQuery&) = delete;
    CompactionQuery& operator=(const CompactionQuery&) = delete;

    /// @brief Records writing the compacted sizes of `built`, after a barrier from their builds.
    /// @throws std::invalid_argument for more than the capacity, or a structure not compactable.
    void Record(VkCommandBuffer commands, std::span<const AccelerationStructure* const> built);
    /// @return The sizes in `Record`'s order once the device has written them; none before.
    std::optional<std::vector<std::uint64_t>> Sizes() const;

  private:
    VkDevice device_;
    VkQueryPool pool_ = VK_NULL_HANDLE;
    std::uint32_t capacity_;
    std::uint32_t count_ = 0;
};

/// @brief Records copying `source` into `compacted`, sized from a `CompactionQuery`.
void RecordCompaction(VkCommandBuffer commands, const AccelerationStructure& source,
                      const AccelerationStructure& compacted);

}  // namespace ernie
