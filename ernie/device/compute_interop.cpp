// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/compute_interop.h"

#include "ernie/device/cuda_interop.h"

namespace ernie {

std::unique_ptr<ComputeInterop> ComputeInterop::Create(const VulkanContext& context,
                                                       std::string* why) {
    std::string cudaWhy;
    if (auto cuda = CudaInterop::Create(context, &cudaWhy)) {
        return cuda;
    }
    if (why != nullptr) {
        *why = "CUDA: " + cudaWhy;
    }
    return nullptr;
}

}  // namespace ernie
