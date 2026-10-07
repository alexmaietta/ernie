#pragma once

// GPU zones for the Tracy profiler in a profiling build (ERNIE_TRACY): the time a command buffer
// spends between a zone's start and end on the device, on the same timeline as the CPU zones of
// ernie/util/profile.h. Without ERNIE_TRACY, or if the device's Tracy context could not be made,
// the macros record nothing.
//
//     ERNIE_GPU_ZONE(context, commands, "Shadow passes");  // until the end of the scope
//
// Finished zones reach the profiler when later command buffers collect them
// (`VulkanContext::CollectGpuProfile`, which `Submit` and `SubmitAsync` do).
//
// References:
// - B. Taudul, "Tracy Profiler", the Vulkan GPU zones (TracyVulkan.hpp).
//   https://github.com/wolfpld/tracy

#include "ernie/util/profile.h"

#if ERNIE_TRACY
// volk first: TracyVulkan.hpp calls Vulkan through volk's function pointers.
#include <volk.h>

#include <tracy/TracyVulkan.hpp>

/// The Tracy Vulkan context of a `VulkanContext`, or null.
#define ERNIE_GPU_PROFILER(context) static_cast<tracy::VkCtx*>((context).GpuProfiler())
#define ERNIE_GPU_ZONE(context, commands, name)                                                    \
    TracyVkNamedZone(ERNIE_GPU_PROFILER(context), TracyConcat(ernieGpuZone, TracyLine), commands,  \
                     name, ERNIE_GPU_PROFILER(context) != nullptr)
#else
#define ERNIE_GPU_ZONE(context, commands, name)
#endif
