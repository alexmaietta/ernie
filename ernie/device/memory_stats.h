// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <string>
#include <vector>

/// The device memory Ernie's buffers and images hold, for the
/// viewer's Profiler panel: each live `GpuBuffer` and `GpuImage`
/// registers its allocation under a category from its usage (render
/// targets, textures, storage buffers, ...) until it is released.
namespace ernie::memory {

/// One live allocation.
struct Allocation {
    const char* category = "";  ///< A string literal.
    std::string detail;  ///< Its extent and format, or its size.
    std::uint64_t bytes = 0;
    bool host =
        false;  ///< In host-visible memory (upload, readback).
};

/// Registers the allocation of the buffer or image with Vulkan
/// handle `handle`, replacing any registered under it.
void Track(std::uint64_t handle, Allocation allocation);

/// Forgets the allocation of `handle`, if registered.
void Untrack(std::uint64_t handle);

/// The live allocations, in no particular order.
std::vector<Allocation> Live();

}  // namespace ernie::memory
