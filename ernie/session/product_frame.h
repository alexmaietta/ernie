// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/buffer.h"
#include "ernie/device/compute_interop.h"
#include "ernie/session/outputs.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace ernie {

/// A rendered frame of a render product: its render targets and how
/// to reach them.
struct ProductFrame {
    const VulkanContext* context = nullptr;
    ComputeInterop* compute =
        nullptr;  ///< Null without compute interop.
    std::unique_ptr<FrameImages>
        images;  ///< What the backend rendered the frame into.
    bool batched = false;  ///< Whether the product renders a batch of
                           ///< views (leading dimension).
    std::uint64_t submission =
        0;  ///< The `SubmitAsync` submission that rendered it.
    int arrayMaps =
        0;  ///< Outstanding `MapDevice::GpuArray` mappings.
    /// Linear GPU outputs, owned by the delegate; a buffer is reused
    /// once no tensor holds it.
    std::vector<std::shared_ptr<GpuBuffer>>* linearPool = nullptr;
    /// The product's vars mapped to linear GPU tensors, owned by the
    /// delegate: a var mapped so by copying joins it, and the
    /// product's later frames hold it as a linear buffer instead
    /// (`FrameImages::Linear`).
    std::set<std::string, std::less<>>* linearVars = nullptr;
};

}  // namespace ernie
