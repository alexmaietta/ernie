// SPDX-License-Identifier: BSD-3-Clause
/// @file profile.cpp
/// @brief Tracy's host-calibrated Vulkan context, and its zones around commands.
#include "ernie/device/profile.h"

#include "ernie/device/context.h"

#if ERNIE_TRACY
#include <tracy/TracyVulkan.hpp>

#include <cstring>
#endif

namespace ernie {

std::unique_ptr<GpuProfiler> GpuProfiler::Create(const DeviceContext& context) {
#if ERNIE_TRACY
    if (!context.Capabilities().calibratedTimestamps) {
        return nullptr;
    }
    tracy::VkCtx* tracy = tracy::CreateVkContext(
        context.Physical(), context.Device(), vkResetQueryPool,
        vkGetPhysicalDeviceCalibrateableTimeDomainsEXT, vkGetCalibratedTimestampsEXT);
    const std::string& name = context.Info().name;
    tracy->Name(name.c_str(), static_cast<std::uint16_t>(name.size()));
    return std::unique_ptr<GpuProfiler>(new GpuProfiler(tracy));
#else
    static_cast<void>(context);
    return nullptr;
#endif
}

GpuProfiler::~GpuProfiler() {
#if ERNIE_TRACY
    tracy::DestroyVkContext(static_cast<tracy::VkCtx*>(tracy_));
#endif
}

void GpuProfiler::Collect() {
#if ERNIE_TRACY
    static_cast<tracy::VkCtx*>(tracy_)->Collect(VK_NULL_HANDLE);
#endif
}

GpuZone::GpuZone(const DeviceContext& context, VkCommandBuffer commands, std::string_view name)
    : zone_(nullptr, [](void*) {}) {
#if ERNIE_TRACY
    GpuProfiler* profiler = context.Profiler();
    if (profiler == nullptr) {
        return;
    }
    // A transient zone: its name is copied, so it may be built at run time.
    static constexpr char kFunction[] = "GpuZone";
    zone_ = {new tracy::VkCtxScope(static_cast<tracy::VkCtx*>(profiler->Tracy()), __LINE__,
                                   __FILE__, std::strlen(__FILE__), kFunction,
                                   sizeof(kFunction) - 1, name.data(), name.size(), commands, true),
             [](void* zone) { delete static_cast<tracy::VkCtxScope*>(zone); }};
#else
    static_cast<void>(context);
    static_cast<void>(commands);
    static_cast<void>(name);
#endif
}

GpuZone::~GpuZone() = default;

}  // namespace ernie
