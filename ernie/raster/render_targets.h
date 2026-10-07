#pragma once

#include "ernie/device/image.h"
#include "ernie/device/pipeline.h"
#include "ernie/device/vulkan_context.h"
#include "ernie/scene/sensor_filters.h"
#include "ernie/session/outputs.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ernie {

/// The images one frame renders into: the G-buffer and depth of the raster passes, the spectral
/// radiance of the shading pass, and every render var the sensor filters write. Render var images
/// stay in the general layout between frames; with `shared` set the single-view ones live in
/// memory CUDA can map in place.
///
/// For a batch of `views` views every image has one layer per view (spectral images one group of
/// `bands` layers per view, `GpuImage::GroupView`), so the views of a frame render into one set
/// of images: `LayerView(view)` is a view's G-buffer, depth or render var.
class RenderTargets : public FrameImages {
  public:
    /// G-buffer attachment formats: normal and visibility, albedo scale and modulation, ids.
    static constexpr std::array<VkFormat, 3> kGBufferFormats{
        VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32G32B32A32_UINT};
    static constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;

    /// @throws std::runtime_error if allocation fails, or std::invalid_argument for a zero extent,
    /// a band count outside 1 to 64, no views, or more layers than the device allows
    /// (`MaxViews`).
    /// Holds the render vars `vars` (`RequiredVars`; empty for every var); the others get
    /// placeholders of one texel per view that passes bind but do not write.
    RenderTargets(const VulkanContext& context, VkExtent2D extent, int bands, bool shared,
                  std::uint32_t views = 1, std::span<const std::string_view> vars = {});

    /// The most views one set of targets with `bands` bands can hold on `context`'s device: its
    /// `maxImageArrayLayers` over the bands.
    static std::uint32_t MaxViews(const VulkanContext& context, int bands);

    VkExtent2D Extent() const override {
        return extent_;
    }

    int Bands() const {
        return bands_;
    }

    std::uint32_t Views() const override {
        return views_;
    }

    bool Shared() const {
        return shared_;
    }

    /// Records the transitions that make the G-buffer and depth attachments.
    void PrepareGBuffer(VkCommandBuffer commands);

    /// View `view`'s G-buffer and depth attachments for `BeginRendering`, clearing the ids to
    /// "background", or keeping what an earlier pass drew with `load`.
    RenderingTarget GBufferTarget(std::uint32_t view, bool load = false) const;

    /// The views' hierarchical depth (Hi-Z) for occlusion culling, created on first use: per view,
    /// mips of the farthest depth below each texel, level 0 the largest power-of-two size within
    /// the image, so a texel of level `l` bounds 2^l x 2^l of it at most twice over.
    GpuImage& Hzb();

    /// Records the transitions that let compute passes read the G-buffer, depth and raster
    /// radiance and write the radiance and the render vars.
    void PrepareCompute(VkCommandBuffer commands);

    GpuImage& GBuffer(std::size_t index) {
        return *gbuffer_.at(index);
    }

    GpuImage& Depth() {
        return *depth_;
    }

    /// Spectral radiance of the raster image, before the lens filter.
    GpuImage& RasterRadiance() {
        return *rasterRadiance_;
    }

    /// The image of render var `name`, or null for an unknown name or one the targets do not
    /// hold.
    GpuImage* Var(std::string_view name) override;

    /// The image passes bind for render var `name`: its own, or its placeholder.
    /// @throws std::invalid_argument for an unknown name.
    GpuImage& Bound(std::string_view name);

    /// The render vars the targets hold.
    const std::vector<std::string_view>& Vars() const {
        return held_;
    }

    /// Every render var name, in binding order of the filters.
    static std::span<const std::string_view> VarNames();

    std::uint64_t GpuBytes() const;

  private:
    const VulkanContext& context_;
    VkExtent2D extent_;
    int bands_;
    std::uint32_t views_;
    bool shared_;
    std::array<std::unique_ptr<GpuImage>, 3> gbuffer_;
    std::unique_ptr<GpuImage> depth_;
    std::unique_ptr<GpuImage> rasterRadiance_;
    std::unique_ptr<GpuImage> hzb_;
    std::vector<std::unique_ptr<GpuImage>> vars_;
    std::vector<bool>
        holds_;  ///< Per var of `VarNames`, whether `vars_` holds it or a placeholder.
    std::vector<std::string_view> held_;
};

}  // namespace ernie
