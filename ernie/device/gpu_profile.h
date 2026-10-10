// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// GPU zones: the time a command buffer spends between a zone's start and end on the device,
// recorded by Ernie's own profiler (ernie/util/profiler.h, through the context's `GpuTimer`) for
// the viewer's Profiler panel while it records, and in a profiling build (ERNIE_TRACY) also sent
// to Tracy on the same timeline as the CPU zones of ernie/util/profile.h.
//
//     ERNIE_GPU_ZONE(context, commands, "Shadow passes");  // until the end of the scope
//
// Finished zones reach the profilers when later command buffers collect them
// (`VulkanContext::CollectGpuProfile`, which `Submit` and `SubmitAsync` do).
//
// References:
// - B. Taudul, "Tracy Profiler", the Vulkan GPU zones (TracyVulkan.hpp).
//   https://github.com/wolfpld/tracy

#include "ernie/device/vulkan_context.h"
#include "ernie/util/profile.h"

namespace ernie {

/// An `ERNIE_GPU_ZONE` for Ernie's profiler: timestamps around the rest of the scope's commands.
class ScopedGpuZone {
  public:
    ScopedGpuZone(const VulkanContext& context, VkCommandBuffer commands, const char* name)
        : timer_(context.Timer()), commands_(commands), zone_(timer_.Begin(commands, name)) {}
    ~ScopedGpuZone() {
        timer_.End(commands_, zone_);
    }
    ScopedGpuZone(const ScopedGpuZone&) = delete;
    ScopedGpuZone& operator=(const ScopedGpuZone&) = delete;

  private:
    GpuTimer& timer_;
    VkCommandBuffer commands_;
    std::int64_t zone_;
};

}  // namespace ernie

#define ERNIE_OWN_GPU_ZONE(context, commands, name)                                                \
    const ::ernie::ScopedGpuZone ERNIE_PROFILE_CONCAT(ernieOwnGpuZone, __LINE__)((context),        \
                                                                                 (commands), name)

#if ERNIE_TRACY
// volk first: TracyVulkan.hpp calls Vulkan through volk's function pointers.
#include <volk.h>

#include <tracy/TracyVulkan.hpp>

/// The Tracy Vulkan context of a `VulkanContext`, or null.
#define ERNIE_GPU_PROFILER(context) static_cast<tracy::VkCtx*>((context).GpuProfiler())
#define ERNIE_GPU_ZONE(context, commands, name)                                                    \
    TracyVkNamedZone(ERNIE_GPU_PROFILER(context), TracyConcat(ernieGpuZone, TracyLine), commands,  \
                     name, ERNIE_GPU_PROFILER(context) != nullptr);                                \
    ERNIE_OWN_GPU_ZONE(context, commands, name)
#else
#define ERNIE_GPU_ZONE(context, commands, name) ERNIE_OWN_GPU_ZONE(context, commands, name)
#endif
