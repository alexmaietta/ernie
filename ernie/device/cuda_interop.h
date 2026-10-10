// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/compute_interop.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ernie {

/// The view of an image `CudaInterop::MapImage` returns (`array` is a `CUarray`).
using CudaArrayView = ComputeArrayView;

/// `ComputeInterop` for the CUDA device behind a Vulkan device.
///
/// The CUDA driver (libcuda) is loaded at run time, so Ernie has no build dependency on the CUDA
/// toolkit; without an NVIDIA driver `Create` reports CUDA unavailable. CUDA work runs in the
/// device's primary context, the one CUDA frameworks use, on the legacy default stream, so it is
/// ordered after work that frameworks hand over through DLPack. Signals to Vulkan all go on the
/// legacy stream, after an event recorded on the stream the work ran on, so values arrive in order
/// however many streams write.
///
/// Reference: NVIDIA, CUDA Driver API, "External Resource Interoperability",
/// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXTRES__INTEROP.html
class CudaInterop final : public ComputeInterop {
  public:
    /// Connects `context`'s device to the CUDA device with the same UUID.
    /// @return Null if the CUDA driver cannot be loaded, no CUDA device matches, or the Vulkan
    /// device cannot export memory; `why`, if given, says which.
    static std::unique_ptr<CudaInterop> Create(const VulkanContext& context,
                                               std::string* why = nullptr);
    ~CudaInterop() override;

    DeviceKind Kind() const override {
        return DeviceKind::Cuda;
    }
    int Device() const override;
    void CopyToBuffer(const Tensor& source, GpuBuffer& buffer, VkDeviceSize offset = 0,
                      std::uintptr_t stream = 1) override;
    void CopyColumnsToBuffer(const Tensor& source, std::span<const std::size_t> order,
                             GpuBuffer& buffer, VkDeviceSize offset = 0,
                             std::uintptr_t stream = 1) override;
    Tensor View(std::shared_ptr<GpuBuffer> buffer, DType type, std::vector<std::int64_t> shape,
                VkDeviceSize offset = 0) override;
    void Copy(const Tensor& source, const Tensor& destination, std::uintptr_t stream = 1) override;
    Tensor Upload(const Tensor& host) override;
    ComputeArrayView MapImage(GpuImage& image) override;
    Tensor DownloadArray(const ComputeArrayView& view) override;
    Tensor Download(const Tensor& device, std::uintptr_t stream = 1) override;
    TimelinePoint VulkanWait() const override;
    TimelinePoint NextVulkanSignal() override;
    void SignalVulkan(std::uintptr_t stream = 1) override;
    void WaitForVulkan(std::uintptr_t stream = 1) override;
    void WaitForVulkan(std::uintptr_t stream, std::uint64_t value) override;
    void Synchronize() override;

    /// The loaded driver and device, opaque outside the implementation.
    struct State;

  private:
    /// Signals the next CUDA-to-Vulkan timeline value after the work queued on `stream`, with
    /// the CUDA context current.
    void SignalVulkanAfter(std::uintptr_t stream);

    explicit CudaInterop(std::shared_ptr<State> state);
    std::uint64_t DevicePointer(GpuBuffer& buffer);
    void CheckOnDevice(const Tensor& tensor) const;

    std::shared_ptr<State> state_;
};

}  // namespace ernie
