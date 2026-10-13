// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/image.h"
#include "ernie/device/pipeline.h"
#include "ernie/device/shader_compiler.h"

#include <glm/glm.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ernie {

/// The share of distant lights' discs that their long-range views'
/// casters (`FitLongRangeView`) leave the receivers, such as moons
/// eclipsing the sun: per view, its casters' coverage down a mip
/// chain, then a visibility map over the view integrating the light's
/// limb-darkened disc, at `kWavelengths` wavelengths, a layer of
/// `Visibility` (long_range.slang).
class LongRangeShadows {
  public:
    /// The wavelengths (nm) the visibility is written at, evenly in
    /// the limb darkening exponent (Hestroffer and Magnan's
    /// -0.023 + 292 / wavelength), between which shading
    /// interpolates.
    static constexpr std::array<float, 4> kWavelengths = {
        300.0f, 450.0f, 800.0f, 2400.0f};

    /// One long-range view to integrate: its layer of the light
    /// shadow maps and texels across there (in the layer's corner),
    /// the tangent of its light's angular radius in map widths per
    /// unit of map depth, and its receivers' nearest and farthest
    /// depth in the map (above 1, past its far plane), for each of
    /// which the visibility is written. The square of its map the
    /// visibility
    /// map spans: the receivers' box, `region` xy its corner nearest
    /// the origin and z its side, in map coordinates (0 to 1).
    struct View {
        std::uint32_t layer = 0;
        std::uint32_t texels = 0;
        float tanUv = 0.0f;
        float receiversNear = 1.0f;
        float receiversFar = 1.0f;
        glm::vec3 region{0.0f, 0.0f, 1.0f};
    };

    /// Texels across each visibility map, over the receivers' box: as
    /// fine as a penumbra a few km wide over tens of km needs.
    static constexpr std::uint32_t kVisibilityTexels = 512;

    LongRangeShadows(const VulkanContext& context,
                     const ShaderCompiler& compiler);

    /// Makes the images as large as `views` need, before shading
    /// takes `Visibility`'s slot: rarely, as most frames' views are
    /// as many and as large as the last's.
    void Prepare(std::span<const View> views);

    /// Records each of `views` (as given to `Prepare`)' coverage and
    /// visibility from `shadowMaps` (in its read-only layout), the
    /// visibility's layers in their order.
    void Record(VkCommandBuffer commands, const GpuImage& shadowMaps,
                std::span<const View> views) const;

    /// The visibility maps (an array, two layers per view of the last
    /// `Record`, at its receivers' nearest and farthest depths,
    /// RGBA16F at `kWavelengths`), for shading; two layers of 1
    /// before any view.
    const GpuImage& Visibility() const {
        return *visibility_;
    }

  private:
    void Resize(std::uint32_t layers, std::uint32_t texels);

    const VulkanContext& context_;
    PipelineLayout coverageLayout_;
    PipelineLayout mipLayout_;
    PipelineLayout visibilityLayout_;
    std::unique_ptr<ComputePipeline> coverage_;
    std::unique_ptr<ComputePipeline> mips_;
    std::unique_ptr<ComputePipeline> integrate_;
    /// The coverage of one view at a time (RG32F, a full mip chain),
    /// the visibility of all.
    std::unique_ptr<GpuImage> coverageImage_;
    std::unique_ptr<GpuImage> visibility_;
};

}  // namespace ernie
