// SPDX-License-Identifier: BSD-3-Clause
/// @file swapchain.cpp
/// @brief Choosing the surface's format and present mode, and acquiring and presenting images.
#include "ernie/app/swapchain.h"

#include "ernie/util/on_failure.h"

#include <algorithm>
#include <array>
#include <limits>

namespace ernie::app {
namespace {

template <typename T, typename Query> std::vector<T> Enumerate(Query&& query) {
    std::uint32_t count = 0;
    query(&count, nullptr);
    std::vector<T> items(count);
    query(&count, items.data());
    return items;
}

/// @return An 8-bit UNORM format: the UI's colours are sRGB-encoded already.
VkSurfaceFormatKHR ChooseFormat(VkPhysicalDevice physical, VkSurfaceKHR surface) {
    const std::vector<VkSurfaceFormatKHR> formats =
        Enumerate<VkSurfaceFormatKHR>([&](std::uint32_t* count, VkSurfaceFormatKHR* items) {
            vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, count, items);
        });
    if (formats.empty()) {
        throw DeviceError(DeviceErrorKind::Unsupported, "The window's surface has no formats");
    }
    for (const VkFormat wanted : {VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM}) {
        const auto found = std::ranges::find_if(formats, [&](const VkSurfaceFormatKHR& format) {
            return format.format == wanted &&
                   format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
        });
        if (found != formats.end()) {
            return *found;
        }
    }
    return formats.front();
}

VkPresentModeKHR ChoosePresentMode(VkPhysicalDevice physical, VkSurfaceKHR surface, bool vsync) {
    if (vsync) {
        return VK_PRESENT_MODE_FIFO_KHR;
    }
    const std::vector<VkPresentModeKHR> modes =
        Enumerate<VkPresentModeKHR>([&](std::uint32_t* count, VkPresentModeKHR* items) {
            vkGetPhysicalDeviceSurfacePresentModesKHR(physical, surface, count, items);
        });
    for (const VkPresentModeKHR wanted :
         {VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR}) {
        if (std::ranges::find(modes, wanted) != modes.end()) {
            return wanted;
        }
    }
    // The one mode every surface supports.
    return VK_PRESENT_MODE_FIFO_KHR;
}

bool Matches(VkResult result) {
    return result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR;
}

}  // namespace

Swapchain::Swapchain(DeviceContext& context, VkSurfaceKHR surface, glm::uvec2 extent, bool vsync)
    : context_(context), surface_(surface), vsync_(vsync) {
    const OnFailure destroySurface(
        [&] { vkDestroySurfaceKHR(context_.Instance(), surface_, nullptr); });
    // The device context enables only the extensions Ernie's capabilities name.
    if (vkGetDeviceProcAddr(context.Device(), "vkCreateSwapchainKHR") == nullptr) {
        throw DeviceError(DeviceErrorKind::Unsupported,
                          "The device context does not enable VK_KHR_swapchain, which presenting "
                          "to a window needs; --offscreen draws without one");
    }
    VkBool32 presents = VK_FALSE;
    CheckVk(vkGetPhysicalDeviceSurfaceSupportKHR(context.Physical(), context.Main().Family(),
                                                 surface, &presents),
            "vkGetPhysicalDeviceSurfaceSupportKHR");
    if (presents == VK_FALSE) {
        throw DeviceError(DeviceErrorKind::Unsupported,
                          "The device's main queue cannot present to the window");
    }
    format_ = ChooseFormat(context.Physical(), surface);
    const VkFenceCreateInfo fence{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    CheckVk(vkCreateFence(context.Device(), &fence, nullptr, &acquired_), "vkCreateFence");
    const OnFailure destroyHandles([&] {
        DestroyViews();
        vkDestroySwapchainKHR(context_.Device(), swapchain_, nullptr);
        vkDestroyFence(context_.Device(), acquired_, nullptr);
    });
    Create(extent);
}

Swapchain::~Swapchain() {
    Quietly([this] { context_.Main().Last().Wait(); });
    DestroyViews();
    vkDestroySwapchainKHR(context_.Device(), swapchain_, nullptr);
    vkDestroyFence(context_.Device(), acquired_, nullptr);
    vkDestroySurfaceKHR(context_.Instance(), surface_, nullptr);
}

void Swapchain::Create(glm::uvec2 extent) {
    VkSurfaceCapabilitiesKHR capabilities{};
    CheckVk(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(context_.Physical(), surface_, &capabilities),
            "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
    if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max()) {
        extent_ = capabilities.currentExtent;
    } else {
        extent_ = {std::clamp(extent.x, capabilities.minImageExtent.width,
                              capabilities.maxImageExtent.width),
                   std::clamp(extent.y, capabilities.minImageExtent.height,
                              capabilities.maxImageExtent.height)};
    }
    std::uint32_t count = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0) {
        count = std::min(count, capabilities.maxImageCount);
    }
    const VkSwapchainKHR old = swapchain_;
    const VkSwapchainCreateInfoKHR info{
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = surface_,
        .minImageCount = count,
        .imageFormat = format_.format,
        .imageColorSpace = format_.colorSpace,
        .imageExtent = extent_,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .preTransform = capabilities.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = ChoosePresentMode(context_.Physical(), surface_, vsync_),
        .clipped = VK_TRUE,
        .oldSwapchain = old};
    CheckVk(vkCreateSwapchainKHR(context_.Device(), &info, nullptr, &swapchain_),
            "vkCreateSwapchainKHR");
    if (old != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(context_.Device(), old, nullptr);
    }
    images_ = Enumerate<VkImage>([&](std::uint32_t* n, VkImage* items) {
        vkGetSwapchainImagesKHR(context_.Device(), swapchain_, n, items);
    });
    for (const VkImage image : images_) {
        const VkImageViewCreateInfo view{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = format_.format,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        CheckVk(vkCreateImageView(context_.Device(), &view, nullptr, &views_.emplace_back()),
                "vkCreateImageView");
    }
}

void Swapchain::DestroyViews() {
    for (const VkImageView view : views_) {
        vkDestroyImageView(context_.Device(), view, nullptr);
    }
    views_.clear();
}

void Swapchain::Resize(glm::uvec2 extent) {
    context_.Main().Last().Wait();
    DestroyViews();
    Create(extent);
}

std::optional<Swapchain::Image> Swapchain::Acquire() {
    std::uint32_t index = 0;
    const VkResult result = vkAcquireNextImageKHR(context_.Device(), swapchain_,
                                                  std::numeric_limits<std::uint64_t>::max(),
                                                  VK_NULL_HANDLE, acquired_, &index);
    if (!Matches(result)) {
        if (result != VK_ERROR_OUT_OF_DATE_KHR) {
            CheckVk(result, "vkAcquireNextImageKHR");
        }
        return std::nullopt;
    }
    CheckVk(vkWaitForFences(context_.Device(), 1, &acquired_, VK_TRUE,
                            std::numeric_limits<std::uint64_t>::max()),
            "vkWaitForFences");
    CheckVk(vkResetFences(context_.Device(), 1, &acquired_), "vkResetFences");
    return Image{index, images_[index], views_[index]};
}

bool Swapchain::Present(const Image& image, const Ticket& drawn) {
    drawn.Wait();
    const VkPresentInfoKHR info{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                                .swapchainCount = 1,
                                .pSwapchains = &swapchain_,
                                .pImageIndices = &image.index};
    const VkResult result = vkQueuePresentKHR(context_.Main().Handle(), &info);
    if (result != VK_ERROR_OUT_OF_DATE_KHR && !Matches(result)) {
        CheckVk(result, "vkQueuePresentKHR");
    }
    return result == VK_SUCCESS;
}

}  // namespace ernie::app
