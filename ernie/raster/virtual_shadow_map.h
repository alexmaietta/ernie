// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/buffer.h"
#include "ernie/device/image.h"
#include "ernie/device/pipeline.h"
#include "ernie/device/shader_compiler.h"
#include "ernie/device/vulkan_context.h"

#include <glm/glm.hpp>

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ernie {

/// One level of a distant light's virtual shadow clipmap: a 16384²
/// virtual map of 128² pages over a square of light space 4 R across,
/// R = 2^(level + 1) metres, centred on the camera.
struct VirtualShadowLevel {
    std::int32_t level = 0;
    double radius = 0.0;  ///< R, metres.
    /// Relative to the frame's origin to clip space: x and y over the
    /// level's square, z reversed (nearer the light larger, 0.5 at
    /// the camera's depth along the light).
    glm::mat4 viewProjection{1.0f};
    /// Metres per texel of the virtual map: 4 R / 16384.
    double texelWorldSize = 0.0;
};

/// The virtual shadow maps of distant lights, as Unreal Engine 5's
/// (Epic Games, "Virtual Shadow Maps", Unreal Engine 5
/// documentation): a clipmap of levels whose 16384² virtual maps are
/// paged, the pages the frame's pixels need (marked from the depth
/// buffer) backed by a pool of physical 128² pages that the shadow
/// casters write with atomic max. About one shadow texel per screen
/// pixel from centimetres to the horizon, where cascades spread a few
/// thousand texels over kilometres.
///
/// Layout (virtual_shadow.slang): the pool is one R32_UINT image 128
/// pages wide holding reversed-Z depth as float bits (0 far), the
/// page table and the request flags one R32_UINT 128² layer per
/// level; a page table entry is the physical page's x in bits 0-9,
/// its y in 10-19 and bit 31 set when backed.
///
/// Each frame: `RecordMark` flags the pages the depth buffer's points
/// fall in, `RecordAllocate` backs them from the pool in order (no
/// caching yet), `RecordClear` clears the backed pages; the casters
/// then draw each level as a shadow view whose fragments write
/// through the page table, and shading looks the pages up.
///
/// References:
/// - Epic Games, "Virtual Shadow Maps in Unreal Engine",
///   https://dev.epicgames.com/documentation/en-us/unreal-engine/virtual-shadow-maps-in-unreal-engine
/// - A. Lefohn, S. Sengupta and J. D. Owens, "Resolution-Matched
///   Shadow Maps", ACM Transactions on Graphics 26(4):20, 2007.
///   https://doi.org/10.1145/1289603.1289611
class VirtualShadowMap {
  public:
    static constexpr std::uint32_t kPageTexels = 128;
    static constexpr std::uint32_t kPagesAcross = 128;
    static constexpr std::uint32_t kVirtualTexels =
        kPageTexels * kPagesAcross;
    static constexpr std::uint32_t kMaxLevels = 32;

    /// A clipmap of levels `firstLevel` to `lastLevel` (radii
    /// 2^(level + 1) m) backed by `pages` physical pages.
    /// @throws std::invalid_argument for no levels, more than
    /// `kMaxLevels`, or no pages.
    VirtualShadowMap(const VulkanContext& context,
                     const ShaderCompiler& compiler,
                     std::int32_t firstLevel, std::int32_t lastLevel,
                     std::uint32_t pages);

    /// The levels for a light shining from `toLight` (unit, world)
    /// seen by a camera at `eye` (world), relative to `origin`: each
    /// centred on the camera in light space, snapped to its radius
    /// in world coordinates so its pages hold still as the camera
    /// and the origin move.
    static std::vector<VirtualShadowLevel>
    FitLevels(const glm::dvec3& toLight, const glm::dvec3& eye,
              const glm::dvec3& origin, std::int32_t firstLevel,
              std::int32_t lastLevel);

    /// The level whose texels suit a receiver `distance` metres from
    /// the camera: floor(log2 distance + bias), as Unreal's clipmap
    /// level selection.
    static std::int32_t LevelFor(double distance, double bias);

    std::int32_t FirstLevel() const {
        return firstLevel_;
    }
    std::uint32_t LevelCount() const {
        return levelCount_;
    }
    std::uint32_t Pages() const {
        return pages_;
    }

    /// Uploads `levels` (one per level of the clipmap) and the bias
    /// that picks a receiver's level.
    void SetLevels(std::span<const VirtualShadowLevel> levels,
                   float bias);

    /// Flags the pages of every second pixel's point in `depth`
    /// (reversed-Z, 0 far; sampled as `VK_IMAGE_LAYOUT_SHADER_READ_
    /// ONLY_OPTIMAL`), `inverseViewProjection` taking clip space to
    /// origin-relative world, the camera at `eye` (origin-relative).
    void RecordMark(VkCommandBuffer commands, VkImageView depth,
                    std::uint32_t width, std::uint32_t height,
                    const glm::mat4& inverseViewProjection,
                    const glm::vec3& eye) const;

    /// Backs the flagged pages from the pool and writes the page
    /// table; clears the flags for the next frame.
    void RecordAllocate(VkCommandBuffer commands) const;

    /// Clears the backed pages to the far depth, ready for casters.
    void RecordClear(VkCommandBuffer commands) const;

    /// The pool, page table and levels for the casters' fragment
    /// shaders and shading (virtual_shadow.slang).
    const GpuImage& Pool() const {
        return *pool_;
    }
    const GpuImage& PageTable() const {
        return *pageTable_;
    }
    const GpuBuffer& Levels() const {
        return *levels_;
    }
    /// The physical pages the last allocation used (the count can
    /// exceed `Pages()` when the pool overflowed), read on the host
    /// after the frame.
    const GpuBuffer& Counter() const {
        return *counter_;
    }

    /// Physical pages along each row of the pool.
    std::uint32_t RowPages() const {
        return rowPages_;
    }

  private:
    /// virtual_shadow.slang's `VirtualConstants`.
    struct Constants {
        float inverseViewProjection[16];
        float eye[4];
        std::int32_t sizes[4];
        std::uint32_t pool[4];
    };

    const VulkanContext& context_;
    const ShaderCompiler& compiler_;
    std::int32_t firstLevel_ = 0;
    std::uint32_t levelCount_ = 0;
    std::uint32_t pages_ = 0;
    std::uint32_t rowPages_ = 0;
    float bias_ = 0.0f;
    PipelineLayout layout_;
    std::unique_ptr<GpuImage> pool_;
    std::unique_ptr<GpuImage> pageTable_;
    std::unique_ptr<GpuImage> requests_;
    std::unique_ptr<GpuImage> noDepth_;
    std::unique_ptr<GpuBuffer> levels_;
    std::unique_ptr<GpuBuffer> counter_;
    mutable bool cleared_ = false;
    mutable std::unique_ptr<ComputePipeline> markPipeline_;
    mutable std::unique_ptr<ComputePipeline> allocatePipeline_;
    mutable std::unique_ptr<ComputePipeline> clearPipeline_;

    void CreatePipelines() const;
    /// Pipelines made, images in the general layout, flags cleared
    /// once.
    void Prepare(VkCommandBuffer commands) const;
    void Push(VkCommandBuffer commands, VkImageView depth,
              const Constants& constants) const;
    Constants ConstantsOf(std::uint32_t width, std::uint32_t height,
                          const glm::mat4& inverseViewProjection,
                          const glm::vec3& eye) const;
};

}  // namespace ernie
