#include "ernie/raster/render_targets.h"

#include "ernie/util/spectrum.h"

#include <algorithm>
#include <bit>
#include <stdexcept>
#include <string>

namespace ernie {

namespace {

constexpr const auto& kVarNames = kRenderVarNames;

constexpr std::array<VkFormat, 10> kVarFormats{
    VK_FORMAT_R32_SFLOAT,        VK_FORMAT_R32G32B32A32_SFLOAT,
    VK_FORMAT_R16G16B16A16_UINT, VK_FORMAT_R32G32B32A32_SFLOAT,
    VK_FORMAT_R8G8B8A8_UNORM,    VK_FORMAT_R32_SFLOAT,
    VK_FORMAT_R32_SFLOAT,        VK_FORMAT_R32G32B32A32_SFLOAT,
    VK_FORMAT_R32_UINT,          VK_FORMAT_R32_UINT};

constexpr VkImageUsageFlags kStorage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

}  // namespace

std::uint32_t RenderTargets::MaxViews(const VulkanContext& context, int bands) {
    return context.Limits().maxImageArrayLayers / static_cast<std::uint32_t>(std::max(bands, 1));
}

RenderTargets::RenderTargets(const VulkanContext& context, VkExtent2D extent, int bands,
                             bool shared, std::uint32_t views,
                             std::span<const std::string_view> vars)
    : context_(context), extent_(extent), bands_(bands), views_(views), shared_(shared) {
    if (extent.width == 0 || extent.height == 0) {
        throw std::invalid_argument("Render targets need a non-zero extent");
    }
    SpectralBands{380.0, 780.0, bands}.Check();
    if (views == 0 || views > MaxViews(context, bands)) {
        throw std::invalid_argument("Render targets hold 1 to " +
                                    std::to_string(MaxViews(context, bands)) + " views of " +
                                    std::to_string(bands) + " bands, not " + std::to_string(views));
    }
    for (std::size_t i = 0; i < gbuffer_.size(); ++i) {
        gbuffer_[i] = std::make_unique<GpuImage>(
            context, ImageDesc{extent.width, extent.height, views, 1, kGBufferFormats[i],
                               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT});
    }
    // An array even for one view, so a depth pyramid builds from every view's layer at once.
    depth_ = std::make_unique<GpuImage>(
        context, ImageDesc{extent.width, extent.height, views, 1, kDepthFormat,
                           VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                           false, true});
    const auto bandLayers = static_cast<std::uint32_t>(bands);
    rasterRadiance_ = std::make_unique<GpuImage>(
        context, ImageDesc{extent.width, extent.height, views * bandLayers, 1, VK_FORMAT_R32_SFLOAT,
                           kStorage, false, true, bandLayers});
    held_ = RequiredVars(vars);
    for (std::size_t i = 0; i < kVarNames.size(); ++i) {
        const bool spectral = kVarNames[i] == render_vars::kSpectralRadiance;
        const bool held = std::find(held_.begin(), held_.end(), kVarNames[i]) != held_.end();
        holds_.push_back(held);
        // A placeholder has a texel per view (a group of them per view for SpectralRadiance), so
        // passes bind it like the image it stands for.
        const std::uint32_t width = held ? extent.width : 1;
        const std::uint32_t height = held ? extent.height : 1;
        vars_.push_back(std::make_unique<GpuImage>(
            context, ImageDesc{width, height, spectral ? views * bandLayers : views, 1,
                               kVarFormats[i], kStorage, held && shared && !spectral && views == 1,
                               spectral, spectral ? bandLayers : 0u}));
    }
}

void RenderTargets::PrepareGBuffer(VkCommandBuffer commands) {
    for (const std::unique_ptr<GpuImage>& image : gbuffer_) {
        image->Transition(commands, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    }
    depth_->Transition(commands, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
}

GpuImage& RenderTargets::Hzb() {
    if (hzb_ == nullptr) {
        // Each texel of level 0 bounds at most 2 x 2 depth texels: the largest powers of two
        // within the image.
        const std::uint32_t width = std::bit_floor(extent_.width);
        const std::uint32_t height = std::bit_floor(extent_.height);
        const auto mips = static_cast<std::uint32_t>(std::bit_width(std::max(width, height)));
        hzb_ = std::make_unique<GpuImage>(
            context_,
            ImageDesc{width, height, views_, mips, VK_FORMAT_R32_SFLOAT, kStorage, false, true});
    }
    return *hzb_;
}

RenderingTarget RenderTargets::GBufferTarget(std::uint32_t view, bool load) const {
    RenderingTarget target;
    target.loadColor = load;
    target.loadDepth = load;
    for (const std::unique_ptr<GpuImage>& image : gbuffer_) {
        target.colors.push_back(image->LayerView(view));
    }
    VkClearColorValue background{};
    background.uint32[0] = 0xFFFFFFFFu;
    target.clearColors = {VkClearColorValue{}, VkClearColorValue{}, background};
    target.depth = depth_->LayerView(view);
    target.extent = extent_;
    return target;
}

void RenderTargets::PrepareCompute(VkCommandBuffer commands) {
    for (const std::unique_ptr<GpuImage>& image : gbuffer_) {
        image->Transition(commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    depth_->Transition(commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    rasterRadiance_->Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
    for (const std::unique_ptr<GpuImage>& image : vars_) {
        image->Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
    }
}

GpuImage* RenderTargets::Var(std::string_view name) {
    for (std::size_t i = 0; i < kVarNames.size(); ++i) {
        if (kVarNames[i] == name) {
            return holds_[i] ? vars_[i].get() : nullptr;
        }
    }
    return nullptr;
}

GpuImage& RenderTargets::Bound(std::string_view name) {
    for (std::size_t i = 0; i < kVarNames.size(); ++i) {
        if (kVarNames[i] == name) {
            return *vars_[i];
        }
    }
    throw std::invalid_argument("Unknown render var " + std::string(name));
}

std::span<const std::string_view> RenderTargets::VarNames() {
    return kVarNames;
}

std::uint64_t RenderTargets::GpuBytes() const {
    std::uint64_t bytes =
        depth_->Bytes() + rasterRadiance_->Bytes() + (hzb_ != nullptr ? hzb_->Bytes() : 0);
    for (const std::unique_ptr<GpuImage>& image : gbuffer_) {
        bytes += image->Bytes();
    }
    for (const std::unique_ptr<GpuImage>& image : vars_) {
        bytes += image->Bytes();
    }
    return bytes;
}

}  // namespace ernie
