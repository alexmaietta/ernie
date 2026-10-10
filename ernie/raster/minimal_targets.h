// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/buffer.h"
#include "ernie/device/image.h"
#include "ernie/device/pipeline.h"
#include "ernie/session/outputs.h"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ernie {

/// What a frame drawn with minimal shading (`MinimalMode`) renders into: layered attachments for
/// a chunk of its views at a time (sRGB colour, ids and normal, depth), and each render var it
/// holds as one linear buffer over every view (`FrameImages::Linear`), which the minimal resolve
/// pass writes chunk by chunk in the frame's own submission. With `shared` the buffers live in
/// memory a compute API views in place, so mapping them copies nothing.
///
/// Holds LdrColor, DistanceToCamera, DistanceToImagePlane, Normal, PrimId and InstanceId.
class MinimalTargets final : public FrameImages {
  public:
    /// Attachment formats: colour (sRGB-encoded RGBA8) and ids (prim id, instance id, octahedral
    /// normal, flags).
    static constexpr std::array<VkFormat, 2> kColorFormats{VK_FORMAT_R8G8B8A8_UNORM,
                                                           VK_FORMAT_R32G32B32A32_UINT};
    static constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;

    /// The render vars a minimal frame can hold.
    static bool Holds(std::string_view name);

    /// The vars a frame asked for `vars` holds (all it can for none), in `Vars` order.
    static std::vector<std::string> Held(std::span<const std::string_view> vars);

    /// Holds `vars` (empty for all it can) for `views` views of `extent`, `chunk` views drawn at
    /// a time (at most the device's array layers).
    /// @throws std::invalid_argument for a zero extent or views, a chunk outside 1 to the
    /// device's array layers, or a var a minimal frame cannot hold.
    MinimalTargets(const VulkanContext& context, VkExtent2D extent, std::uint32_t views,
                   std::uint32_t chunk, std::span<const std::string_view> vars, bool shared);

    VkExtent2D Extent() const override {
        return extent_;
    }

    std::uint32_t Views() const override {
        return views_;
    }

    /// None: a minimal frame holds its vars as linear buffers.
    GpuImage* Var(std::string_view /*name*/) override {
        return nullptr;
    }

    LinearVar Linear(std::string_view name) const override;

    /// The views the attachments hold at a time.
    std::uint32_t Chunk() const {
        return chunk_;
    }

    bool Shared() const {
        return shared_;
    }

    /// The held vars in `kRenderVarNames` order.
    const std::vector<std::string>& Vars() const {
        return names_;
    }

    /// Records the transitions that make the attachments.
    void PrepareDraw(VkCommandBuffer commands);

    /// The attachments for a layered `BeginRendering` of `views` views (at most `Chunk`), each
    /// view drawing into its layer: colour cleared to `clear` (linear RGB, encoded as sRGB), ids
    /// to "background", depth to the far plane.
    RenderingTarget Target(std::uint32_t views, const float clear[3]) const;

    /// Records the transitions that let the resolve pass read the attachments.
    void PrepareResolve(VkCommandBuffer commands);

    GpuImage& Color() {
        return *color_;
    }

    GpuImage& Ids() {
        return *ids_;
    }

    GpuImage& Depth() {
        return *depth_;
    }

    std::uint64_t GpuBytes() const;

  private:
    VkExtent2D extent_;
    std::uint32_t views_;
    std::uint32_t chunk_;
    bool shared_;
    std::unique_ptr<GpuImage> color_;
    std::unique_ptr<GpuImage> ids_;
    std::unique_ptr<GpuImage> depth_;
    std::vector<std::string> names_;
    std::vector<LinearVar> linear_;
};

}  // namespace ernie
