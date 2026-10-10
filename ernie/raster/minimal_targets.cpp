// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/raster/minimal_targets.h"

#include "ernie/scene/sensor_filters.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ernie {

namespace {

/// The vars a minimal frame holds and their texel formats, as the full pipeline's.
struct MinimalVar {
    std::string_view name;
    VkFormat format;
};

constexpr std::array<MinimalVar, 6> kMinimalVars{{
    {render_vars::kLdrColor, VK_FORMAT_R8G8B8A8_UNORM},
    {render_vars::kDistanceToImagePlane, VK_FORMAT_R32_SFLOAT},
    {render_vars::kDistanceToCamera, VK_FORMAT_R32_SFLOAT},
    {render_vars::kNormal, VK_FORMAT_R32G32B32A32_SFLOAT},
    {render_vars::kPrimId, VK_FORMAT_R32_UINT},
    {render_vars::kInstanceId, VK_FORMAT_R32_UINT},
}};

VkDeviceSize TexelBytes(VkFormat format) {
    return format == VK_FORMAT_R32G32B32A32_SFLOAT ? 16 : 4;
}

float SrgbEncode(float c) {
    c = std::clamp(c, 0.0f, 1.0f);
    return c <= 0.0031308f ? 12.92f * c : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
}

}  // namespace

bool MinimalTargets::Holds(std::string_view name) {
    return std::any_of(kMinimalVars.begin(), kMinimalVars.end(),
                       [&](const MinimalVar& var) { return var.name == name; });
}

std::vector<std::string> MinimalTargets::Held(std::span<const std::string_view> vars) {
    std::vector<std::string> held;
    for (const MinimalVar& var : kMinimalVars) {
        if (vars.empty() || std::find(vars.begin(), vars.end(), var.name) != vars.end()) {
            held.emplace_back(var.name);
        }
    }
    return held;
}

MinimalTargets::MinimalTargets(const VulkanContext& context, VkExtent2D extent,
                               std::uint32_t views, std::uint32_t chunk,
                               std::span<const std::string_view> vars, bool shared)
    : extent_(extent), views_(views), chunk_(std::min(chunk, views)), shared_(shared) {
    if (extent.width == 0 || extent.height == 0 || views == 0) {
        throw std::invalid_argument("Minimal targets need a non-zero extent and views");
    }
    if (chunk == 0 || chunk > context.Limits().maxImageArrayLayers) {
        throw std::invalid_argument("Minimal targets draw 1 to " +
                                    std::to_string(context.Limits().maxImageArrayLayers) +
                                    " views at a time, not " + std::to_string(chunk));
    }
    for (const std::string_view name : vars) {
        if (!Holds(name)) {
            throw std::invalid_argument("Minimal shading renders no " + std::string(name) +
                                        ": only LdrColor, DistanceToImagePlane, "
                                        "DistanceToCamera, Normal, PrimId and InstanceId");
        }
    }
    const auto attachment = [&](VkFormat format, VkImageUsageFlags usage) {
        return std::make_unique<GpuImage>(
            context, ImageDesc{extent.width, extent.height, chunk_, 1, format,
                               usage | VK_IMAGE_USAGE_SAMPLED_BIT, false, true});
    };
    color_ = attachment(kColorFormats[0], VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    ids_ = attachment(kColorFormats[1], VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    depth_ = attachment(kDepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
    const VkDeviceSize pixels = VkDeviceSize{extent.width} * extent.height * views;
    for (const MinimalVar& var : kMinimalVars) {
        const bool held = vars.empty() || std::find(vars.begin(), vars.end(), var.name) !=
                                              vars.end();
        LinearVar linear;
        if (held) {
            linear.buffer = std::make_shared<GpuBuffer>(
                context, pixels * TexelBytes(var.format), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                shared ? BufferMemory::Shared : BufferMemory::Device);
            linear.format = var.format;
            names_.emplace_back(var.name);
        }
        linear_.push_back(std::move(linear));
    }
}

FrameImages::LinearVar MinimalTargets::Linear(std::string_view name) const {
    for (std::size_t i = 0; i < kMinimalVars.size(); ++i) {
        if (kMinimalVars[i].name == name) {
            return linear_[i];
        }
    }
    return {};
}

void MinimalTargets::PrepareDraw(VkCommandBuffer commands) {
    color_->Transition(commands, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    ids_->Transition(commands, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    depth_->Transition(commands, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
}

RenderingTarget MinimalTargets::Target(std::uint32_t views, const float clear[3]) const {
    RenderingTarget target;
    target.colors = {color_->View(), ids_->View()};
    VkClearColorValue background{};
    background.float32[0] = SrgbEncode(clear[0]);
    background.float32[1] = SrgbEncode(clear[1]);
    background.float32[2] = SrgbEncode(clear[2]);
    background.float32[3] = 1.0f;
    VkClearColorValue none{};
    none.uint32[0] = 0xFFFFFFFFu;
    target.clearColors = {background, none};
    target.depth = depth_->View();
    target.extent = extent_;
    target.layers = std::min(views, chunk_);
    return target;
}

void MinimalTargets::PrepareResolve(VkCommandBuffer commands) {
    color_->Transition(commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    ids_->Transition(commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    depth_->Transition(commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

std::uint64_t MinimalTargets::GpuBytes() const {
    std::uint64_t bytes = color_->Bytes() + ids_->Bytes() + depth_->Bytes();
    for (const LinearVar& linear : linear_) {
        bytes += linear.buffer != nullptr ? linear.buffer->Size() : 0;
    }
    return bytes;
}

}  // namespace ernie
