// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/raster/render_targets.h"

#include "ernie/util/spectrum.h"

#include <algorithm>
#include <bit>
#include <stdexcept>
#include <string>

namespace ernie {

namespace {

constexpr const auto& kVarNames = kRenderVarNames;

constexpr std::array<VkFormat, 14> kVarFormats{
    VK_FORMAT_R32_SFLOAT,        VK_FORMAT_R32G32B32A32_SFLOAT,
    VK_FORMAT_R16G16B16A16_UINT, VK_FORMAT_R32G32B32A32_SFLOAT,
    VK_FORMAT_R8G8B8A8_UNORM,    VK_FORMAT_R32_SFLOAT,
    VK_FORMAT_R32_SFLOAT,        VK_FORMAT_R32G32B32A32_SFLOAT,
    VK_FORMAT_R32_UINT,          VK_FORMAT_R32_UINT,
    VK_FORMAT_R32G32_SFLOAT,     VK_FORMAT_R32_SFLOAT,
    VK_FORMAT_R32_SFLOAT,        VK_FORMAT_R32G32B32A32_SFLOAT};

constexpr VkImageUsageFlags kStorage =
    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

}  // namespace

std::uint32_t RenderTargets::MaxViews(const VulkanContext& context,
                                      int bands,
                                      std::uint32_t faces) {
    return context.Limits().maxImageArrayLayers /
           (static_cast<std::uint32_t>(std::max(bands, 1)) *
            std::max(faces, 1u));
}

RenderTargets::RenderTargets(const VulkanContext& context,
                             VkExtent2D extent, int bands,
                             bool shared, std::uint32_t views,
                             std::span<const std::string_view> vars,
                             std::span<const std::string> aovs,
                             std::uint32_t faces)
    : context_(context), extent_(extent), bands_(bands),
      views_(views), faces_(faces), shared_(shared),
      aovNames_(aovs.begin(), aovs.end()) {
    if (extent.width == 0 || extent.height == 0) {
        throw std::invalid_argument(
            "Render targets need a non-zero extent");
    }
    SpectralBands{380.0, 780.0, bands}.Check();
    if (faces == 0) {
        throw std::invalid_argument(
            "Render targets need at least one raster layer per view");
    }
    if (views == 0 || views > MaxViews(context, bands, faces)) {
        throw std::invalid_argument(
            "Render targets hold 1 to " +
            std::to_string(MaxViews(context, bands, faces)) +
            " views of " + std::to_string(bands) + " bands and " +
            std::to_string(faces) + " faces, not " +
            std::to_string(views));
    }
    const std::uint32_t layers = views * faces;
    for (std::size_t i = 0; i < gbuffer_.size(); ++i) {
        // Arrays even for one view, so shading reads every view's
        // layer in one dispatch.
        gbuffer_[i] = std::make_unique<GpuImage>(
            context, ImageDesc{extent.width, extent.height, layers, 1,
                               kGBufferFormats[i],
                               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                   VK_IMAGE_USAGE_SAMPLED_BIT,
                               false, true});
    }
    // An array even for one view, so a depth pyramid builds from
    // every view's layer at once.
    depth_ = std::make_unique<GpuImage>(
        context,
        ImageDesc{extent.width, extent.height, layers, 1,
                  kDepthFormat,
                  VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                      VK_IMAGE_USAGE_SAMPLED_BIT,
                  false, true});
    const auto bandLayers = static_cast<std::uint32_t>(bands);
    rasterRadiance_ = std::make_unique<GpuImage>(
        context,
        ImageDesc{extent.width, extent.height, layers * bandLayers, 1,
                  VK_FORMAT_R32_SFLOAT, kStorage, false, true,
                  bandLayers});
    held_ = RequiredVars(vars);
    for (std::size_t i = 0; i < kVarNames.size(); ++i) {
        const bool spectral =
            kVarNames[i] == render_vars::kSpectralRadiance;
        const bool held = std::find(held_.begin(), held_.end(),
                                    kVarNames[i]) != held_.end();
        holds_.push_back(held);
        // A placeholder has a texel per view (a group of them per
        // view for SpectralRadiance), so passes bind it like the
        // image it stands for; each is an array, as the filters write
        // every view's layer in one dispatch.
        const std::uint32_t width = held ? extent.width : 1;
        const std::uint32_t height = held ? extent.height : 1;
        // LdrColor is also drawn into, by the overlays.
        const VkImageUsageFlags usage =
            kVarNames[i] == render_vars::kLdrColor
                ? kStorage | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                : kStorage;
        vars_.push_back(std::make_unique<GpuImage>(
            context,
            ImageDesc{width, height,
                      spectral ? views * bandLayers : views, 1,
                      kVarFormats[i], usage,
                      held && shared && !spectral && views == 1, true,
                      spectral ? bandLayers : 0u}));
    }
    // The AOV slots, float RGBA, held or one-texel placeholders as
    // the vars.
    for (std::size_t slot = 0; slot < kMaxAovs; ++slot) {
        const bool held =
            std::find(held_.begin(), held_.end(),
                      kAovSlotNames[slot]) != held_.end();
        holdsAov_[slot] = held;
        aovImages_[slot] = std::make_unique<GpuImage>(
            context,
            ImageDesc{held ? extent.width : 1,
                      held ? extent.height : 1, views, 1,
                      VK_FORMAT_R32G32B32A32_SFLOAT, kStorage,
                      held && shared && views == 1, true});
    }
}

void RenderTargets::PrepareGBuffer(VkCommandBuffer commands) {
    for (const std::unique_ptr<GpuImage>& image : gbuffer_) {
        image->Transition(commands,
                          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    }
    depth_->Transition(commands,
                       VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
}

GpuImage& RenderTargets::Hzb() {
    if (hzb_ == nullptr) {
        // Each texel of level 0 bounds at most 2 x 2 depth texels:
        // the largest powers of two within the image.
        const std::uint32_t width = std::bit_floor(extent_.width);
        const std::uint32_t height = std::bit_floor(extent_.height);
        const auto mips = static_cast<std::uint32_t>(
            std::bit_width(std::max(width, height)));
        hzb_ = std::make_unique<GpuImage>(
            context_,
            ImageDesc{width, height, RasterLayers(), mips,
                      VK_FORMAT_R32_SFLOAT, kStorage, false, true});
    }
    return *hzb_;
}

GpuImage& RenderTargets::RadianceSum() {
    if (radianceSum_ == nullptr) {
        const auto bandLayers = static_cast<std::uint32_t>(bands_);
        radianceSum_ = std::make_unique<GpuImage>(
            context_, ImageDesc{extent_.width, extent_.height,
                                StokesOffset() * (stokes_ ? 3u : 1u),
                                1, VK_FORMAT_R32_SFLOAT, kStorage,
                                false, true, bandLayers});
    }
    return *radianceSum_;
}

void RenderTargets::SetStokes(bool stokes) {
    if (stokes == stokes_) {
        return;
    }
    const std::uint32_t layers = StokesOffset() * (stokes ? 3u : 1u);
    if (layers > context_.Limits().maxImageArrayLayers) {
        throw std::invalid_argument(
            "A polarizer's Stokes planes need " +
            std::to_string(layers) +
            " layers of raster radiance, more than the device's " +
            std::to_string(context_.Limits().maxImageArrayLayers) +
            ": render fewer views or bands");
    }
    vkDeviceWaitIdle(context_.Device());
    stokes_ = stokes;
    const auto bandLayers = static_cast<std::uint32_t>(bands_);
    rasterRadiance_ = std::make_unique<GpuImage>(
        context_, ImageDesc{extent_.width, extent_.height, layers, 1,
                            VK_FORMAT_R32_SFLOAT, kStorage, false,
                            true, bandLayers});
    radianceSum_.reset();
}

GpuImage& RenderTargets::Raster(bool full) {
    std::unique_ptr<GpuImage>& image =
        full ? raster_ : rasterPlaceholder_;
    if (image == nullptr) {
        // Drawn into by the drawables' motion passes too.
        image = std::make_unique<GpuImage>(
            context_,
            ImageDesc{
                full ? extent_.width : 1, full ? extent_.height : 1,
                full ? RasterLayers() : 1, 1,
                VK_FORMAT_R32G32B32A32_SFLOAT,
                kStorage |
                    (full ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT : 0u),
                false, true});
    }
    return *image;
}

GpuImage& RenderTargets::SensorImage(bool spectrum, std::uint32_t set,
                                     bool full, bool pyramid) {
    if (!full) {
        std::unique_ptr<GpuImage>& placeholder =
            sensorPlaceholders_[spectrum ? 0 : 1];
        if (placeholder == nullptr) {
            placeholder = std::make_unique<GpuImage>(
                context_,
                ImageDesc{1, 1, 1, 1,
                          spectrum ? VK_FORMAT_R32_SFLOAT
                                   : VK_FORMAT_R32G32B32A32_SFLOAT,
                          kStorage, false, true});
        }
        return *placeholder;
    }
    std::unique_ptr<GpuImage>& image =
        sensorImages_[(spectrum ? 0 : 2) + (set & 1u)];
    const std::uint32_t mips =
        pyramid && !spectrum
            ? static_cast<std::uint32_t>(std::bit_width(
                  std::max(extent_.width, extent_.height)))
            : 1u;
    if (image == nullptr || image->Desc().mips < mips) {
        const auto bands = static_cast<std::uint32_t>(bands_);
        image = std::make_unique<GpuImage>(
            context_,
            ImageDesc{extent_.width, extent_.height,
                      views_ * (spectrum ? bands : 4u), mips,
                      spectrum ? VK_FORMAT_R32_SFLOAT
                               : VK_FORMAT_R32G32B32A32_SFLOAT,
                      kStorage, false, true});
    }
    return *image;
}

RenderingTarget RenderTargets::GBufferTarget(bool load) const {
    RenderingTarget target;
    target.loadColor = load;
    target.loadDepth = load;
    for (const std::unique_ptr<GpuImage>& image : gbuffer_) {
        target.colors.push_back(image->View());
    }
    VkClearColorValue background{};
    background.uint32[0] = 0xFFFFFFFFu;
    target.clearColors = {VkClearColorValue{}, VkClearColorValue{},
                          background};
    target.depth = depth_->View();
    target.clearDepth = kDepthClear;
    target.extent = extent_;
    target.layers = RasterLayers();
    return target;
}

void RenderTargets::PrepareCompute(VkCommandBuffer commands) {
    for (const std::unique_ptr<GpuImage>& image : gbuffer_) {
        image->Transition(commands,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    depth_->Transition(commands,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    rasterRadiance_->Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
    for (const std::unique_ptr<GpuImage>& image : vars_) {
        image->Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
    }
    for (const std::unique_ptr<GpuImage>& image : aovImages_) {
        image->Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
    }
}

GpuImage* RenderTargets::Var(std::string_view name) {
    for (std::size_t i = 0; i < kVarNames.size(); ++i) {
        if (kVarNames[i] == name) {
            return holds_[i] ? vars_[i].get() : nullptr;
        }
    }
    // AOVs, by slot name or as "aov:<output>".
    for (std::size_t slot = 0; slot < kMaxAovs; ++slot) {
        const bool named =
            name == kAovSlotNames[slot] ||
            (name.starts_with(kAovPrefix) &&
             slot < aovNames_.size() &&
             name.substr(kAovPrefix.size()) == aovNames_[slot]);
        if (named) {
            return holdsAov_[slot] ? aovImages_[slot].get() : nullptr;
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
    throw std::invalid_argument("Unknown render var " +
                                std::string(name));
}

std::span<const std::string_view> RenderTargets::VarNames() {
    return kVarNames;
}

std::shared_ptr<RenderTargets::LinearVars>
RenderTargets::MakeLinearVars(const VulkanContext& context,
                              VkExtent2D extent, std::uint32_t views,
                              int bands,
                              std::span<const std::string> names) {
    auto linear = std::make_shared<LinearVars>();
    std::array<VkDeviceAddress, LinearVars::kVars> addresses{};
    for (std::size_t i = 0; i < LinearVars::kVars; ++i) {
        if (std::find(names.begin(), names.end(), kVarNames[i]) ==
            names.end()) {
            continue;
        }
        // SpectralRadiance a layer per band, the others one, per
        // view.
        const VkDeviceSize layers =
            VkDeviceSize{views} *
            (i == 0 ? static_cast<VkDeviceSize>(bands) : 1u);
        const VkDeviceSize bytes =
            layers * extent.width * extent.height *
            GpuImage::TexelBytes(kVarFormats[i]);
        linear->buffers[i] = std::make_shared<GpuBuffer>(
            context, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            BufferMemory::Shared);
        addresses[i] = linear->buffers[i]->Address();
        linear->held |= 1u << i;
    }
    linear->table = std::make_unique<GpuBuffer>(
        context, sizeof(addresses),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Device);
    linear->table->Write(std::as_bytes(std::span(addresses)));
    return linear;
}

FrameImages::LinearVar
RenderTargets::Linear(std::string_view name) const {
    if (linear_ == nullptr) {
        return {};
    }
    for (std::size_t i = 0; i < LinearVars::kVars; ++i) {
        if (kVarNames[i] == name &&
            ((linear_->written >> i) & 1u) != 0) {
            return {linear_->buffers[i], kVarFormats[i]};
        }
    }
    return {};
}

std::uint64_t RenderTargets::GpuBytes() const {
    std::uint64_t bytes =
        depth_->Bytes() + rasterRadiance_->Bytes() +
        (hzb_ != nullptr ? hzb_->Bytes() : 0) +
        (radianceSum_ != nullptr ? radianceSum_->Bytes() : 0) +
        (raster_ != nullptr ? raster_->Bytes() : 0);
    for (const std::unique_ptr<GpuImage>& image : gbuffer_) {
        bytes += image->Bytes();
    }
    for (const std::unique_ptr<GpuImage>& image : vars_) {
        bytes += image->Bytes();
    }
    for (const std::unique_ptr<GpuImage>& image : sensorImages_) {
        bytes += image != nullptr ? image->Bytes() : 0;
    }
    for (const std::unique_ptr<GpuImage>& image : aovImages_) {
        bytes += image->Bytes();
    }
    return bytes;
}

}  // namespace ernie
