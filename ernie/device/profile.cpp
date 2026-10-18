// SPDX-License-Identifier: BSD-3-Clause
/// @file profile.cpp
/// @brief Tracy's host-calibrated Vulkan context, and its zones around commands.
#include "ernie/device/profile.h"

#include "ernie/device/context.h"

#if ERNIE_TRACY
#include <tracy/TracyVulkan.hpp>

#include <cstring>
#include <new>
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
    // Tracy's context collects from one thread at a time; a busy one is collecting already.
    const std::unique_lock lock(collecting_, std::try_to_lock);
    if (lock.owns_lock()) {
        static_cast<tracy::VkCtx*>(tracy_)->Collect(VK_NULL_HANDLE);
    }
#endif
}

GpuZone::GpuZone(const DeviceContext& context, VkCommandBuffer commands, const char* name)
    : context_(context), commands_(commands) {
    context.BeginLabel(commands, name);
#if ERNIE_TRACY
    static_assert(sizeof(tracy::VkCtxScope) <= sizeof(zone_) &&
                  alignof(tracy::VkCtxScope) <= alignof(std::max_align_t));
    GpuProfiler* profiler = context.Profiler();
    if (profiler == nullptr) {
        return;
    }
    // A transient zone: its name is copied, so it may be built at run time.
    static constexpr char kFunction[] = "GpuZone";
    new (zone_) tracy::VkCtxScope(static_cast<tracy::VkCtx*>(profiler->Tracy()), __LINE__, __FILE__,
                                  std::strlen(__FILE__), kFunction, sizeof(kFunction) - 1, name,
                                  std::strlen(name), commands, true);
    timed_ = true;
#endif
}

GpuZone::~GpuZone() {
#if ERNIE_TRACY
    if (timed_) {
        std::launder(reinterpret_cast<tracy::VkCtxScope*>(zone_))->~VkCtxScope();
    }
#endif
    context_.EndLabel(commands_);
}

}  // namespace ernie
