#pragma once

#include "ernie/device/buffer.h"
#include "ernie/device/cuda_interop.h"
#include "ernie/session/outputs.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace ernie {

/// A rendered frame of a render product: its render targets and how to reach them.
struct ProductFrame {
    const VulkanContext* context = nullptr;
    CudaInterop* cuda = nullptr;          ///< Null without CUDA interop.
    std::unique_ptr<FrameImages> images;  ///< What the backend rendered the frame into.
    bool batched = false;  ///< Whether the product renders a batch of views (leading dimension).
    std::uint64_t submission = 0;  ///< The `SubmitAsync` submission that rendered it.
    int arrayMaps = 0;             ///< Outstanding `MapDevice::CudaArray` mappings.
    /// Linear CUDA outputs, owned by the delegate; a buffer is reused once no tensor holds it.
    std::vector<std::shared_ptr<GpuBuffer>>* linearPool = nullptr;
};

}  // namespace ernie
