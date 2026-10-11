// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/buffer.h"
#include "ernie/device/image.h"
#include "ernie/device/vulkan_context.h"
#include "ernie/util/tensor.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ernie {

/// A Vulkan image seen by a compute API in place, as an opaque array
/// handle (a `CUarray` for CUDA, a `hipArray_t` for HIP): read it
/// with that API's texture or surface objects or 2D array copies. It
/// is valid while the image lives.
struct ComputeArrayView {
    std::uintptr_t array = 0;  ///< The array handle.
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t channels = 0;
    DType type = DType::UInt8;  ///< Type of each channel.
    DeviceKind kind =
        DeviceKind::Cuda;  ///< The compute API the handle belongs to.
    int device = 0;        ///< Device ordinal in that API.
};

/// Zero-copy sharing between a Vulkan device and the compute device
/// behind it (CUDA or HIP), so arrays from compute frameworks
/// (PyTorch, JAX, CuPy, Warp, simulators) reach Ernie's buffers, and
/// Ernie's results reach them, without passing through host memory.
///
/// Vulkan memory is exported as POSIX file descriptors
/// (`BufferMemory::Shared`) and imported into the compute API as
/// external memory; the compute driver is loaded at run time, so
/// Ernie has no build dependency on any GPU toolkit. The two sides
/// order their work on the GPU, without blocking the host, through
/// two timeline semaphores created in Vulkan and imported into the
/// compute API, one per direction so each side signals only its own:
/// - compute to Vulkan: `CopyToBuffer` queues its copy and signals
///   the next value; Vulkan submissions that read the buffer wait for
///   `VulkanWait`.
/// - Vulkan to compute: a Vulkan submission whose writes the compute
///   side reads signals `NextVulkanSignal`; compute streams wait for
///   it with `WaitForVulkan`, and `Download` waits for it itself.
///
/// Streams are numbered as in DLPack for the backend's device kind
/// (for CUDA: 1 the legacy default stream, 2 the per-thread default
/// stream, anything else a stream handle).
///
/// Reference: Khronos, VK_KHR_external_memory_fd,
/// https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_external_memory_fd.html
class ComputeInterop {
  public:
    /// Connects `context`'s device to the compute device with the
    /// same UUID, trying the available backends (CUDA).
    /// @return Null if no backend connects; `why`, if given, says why
    /// each could not.
    static std::unique_ptr<ComputeInterop>
    Create(const VulkanContext& context, std::string* why = nullptr);

    virtual ~ComputeInterop() = default;
    ComputeInterop(const ComputeInterop&) = delete;
    ComputeInterop& operator=(const ComputeInterop&) = delete;

    /// The kind of device tensors on this backend live on
    /// (`DeviceKind::Cuda` or `Rocm`).
    virtual DeviceKind Kind() const = 0;

    /// The device ordinal in the compute API.
    virtual int Device() const = 0;

    /// Where tensors on this backend's device live.
    ernie::Device Location() const {
        return {Kind(), Device()};
    }

    /// Queues a device-to-device copy of a contiguous tensor on this
    /// device into `buffer` at byte `offset` on stream `stream`, so
    /// it follows the work writing the tensor there and the Vulkan
    /// work submitted so far (which may still read the buffer,
    /// `NextVulkanSignal`), and returns without waiting; Vulkan work
    /// reading the buffer must wait for `VulkanWait`. Later writes to
    /// the tensor on that stream follow the copy.
    /// @throws std::invalid_argument if the tensor is not a
    /// contiguous tensor on this device, the buffer is not
    /// `BufferMemory::Shared`, or the bytes do not fit.
    virtual void CopyToBuffer(const Tensor& source, GpuBuffer& buffer,
                              VkDeviceSize offset = 0,
                              std::uintptr_t stream = 1) = 0;

    /// As `CopyToBuffer`, rearranging the columns of a contiguous
    /// (rows, width) tensor: column `order[j]` of each row lands at
    /// column `j` of the buffer's rows, each `order.size()` elements,
    /// such as quaternions with w first stored with w last. Each run
    /// of consecutive columns goes as one strided copy.
    /// @throws std::invalid_argument as `CopyToBuffer`, or for a
    /// tensor that is not 2D or a column out of range.
    virtual void
    CopyColumnsToBuffer(const Tensor& source,
                        std::span<const std::size_t> order,
                        GpuBuffer& buffer, VkDeviceSize offset = 0,
                        std::uintptr_t stream = 1) = 0;

    /// A tensor of `type` and `shape` on this device over `buffer`'s
    /// memory from byte `offset`, which keeps the buffer alive. The
    /// buffer's Vulkan context must outlive the tensor.
    /// @throws std::invalid_argument if the buffer is not
    /// `BufferMemory::Shared` or too small.
    virtual Tensor View(std::shared_ptr<GpuBuffer> buffer, DType type,
                        std::vector<std::int64_t> shape,
                        VkDeviceSize offset = 0) = 0;

    /// A tensor of `type`, `shape` and `strides` (in elements, as
    /// DLPack's) on this device over the texels of the linear shared
    /// image `image` (`ImageDesc::linear` and `shared`), from its
    /// first texel: writing it writes the image itself, which shaders
    /// then sample with no copy. Keeps the image alive. Work writing
    /// it must first wait for the Vulkan work reading the image
    /// (`WaitForVulkan`), and Vulkan work reading it afterwards for
    /// `SignalVulkan`.
    /// @throws std::invalid_argument for an image that is not linear
    /// and shared, or a view past its end.
    virtual Tensor ViewImage(std::shared_ptr<GpuImage> image,
                             DType type,
                             std::vector<std::int64_t> shape,
                             std::vector<std::int64_t> strides) = 0;

    /// Queues a device-to-device copy of the contiguous image or
    /// batch `source`, (rows, cols), (rows, cols, channels) or
    /// (views, rows, cols, channels), into the linear shared image
    /// `image`, the views side by side from its left edge, on
    /// `stream`, after the Vulkan work submitted so far (which may
    /// still read the image, `NextVulkanSignal`), without waiting;
    /// Vulkan work reading the image must wait for `VulkanWait`. A
    /// texel of `source` must be one of the image.
    /// @throws std::invalid_argument for a source not on this device
    /// or not contiguous, an image that is not linear and shared, or
    /// views that do not fit.
    virtual void CopyToImage(const Tensor& source, GpuImage& image,
                             std::uintptr_t stream = 1) = 0;

    /// Queues a device-to-device copy of contiguous tensor `source`
    /// into contiguous tensor `destination` of the same byte size on
    /// `stream`, without waiting.
    /// @throws std::invalid_argument if either is not a contiguous
    /// tensor on this device or their sizes differ.
    virtual void Copy(const Tensor& source, const Tensor& destination,
                      std::uintptr_t stream = 1) = 0;

    /// A contiguous copy of a CPU tensor in device memory that the
    /// tensor owns.
    /// @throws std::invalid_argument for a tensor that is not on the
    /// CPU.
    virtual Tensor Upload(const Tensor& host) = 0;

    /// The shared image `image` as an array, without copying,
    /// importing its memory on first use. Supports RGBA8, RGBA16
    /// unsigned, R32 unsigned and 32-bit float colour formats; work
    /// reading it must first wait for the Vulkan work writing it
    /// (`WaitForVulkan`).
    /// @throws std::invalid_argument for an image that is not shared,
    /// has several layers or mips, or has another format.
    virtual ComputeArrayView MapImage(GpuImage& image) = 0;

    /// A CPU copy of an array, rows top to bottom, after the latest
    /// Vulkan signal.
    virtual Tensor DownloadArray(const ComputeArrayView& view) = 0;

    /// A contiguous CPU copy of a tensor on this device, after the
    /// latest Vulkan signal and the work queued on stream `stream`,
    /// which the copy runs on.
    /// @throws std::invalid_argument for a tensor that is not a
    /// contiguous tensor on this device.
    virtual Tensor Download(const Tensor& device,
                            std::uintptr_t stream = 1) = 0;

    /// The wait Vulkan submissions that read buffers written by
    /// `CopyToBuffer` need; a null semaphore before the first copy.
    virtual TimelinePoint VulkanWait() const = 0;

    /// A signal to add to the next Vulkan submission whose writes the
    /// compute side will read.
    virtual TimelinePoint NextVulkanSignal() = 0;

    /// Queues on `stream` a signal that Vulkan waits for
    /// (`VulkanWait`) before its next work, for compute work that
    /// reads Vulkan memory and must finish before Vulkan writes it
    /// again.
    virtual void SignalVulkan(std::uintptr_t stream = 1) = 0;

    /// Makes stream `stream` wait on the GPU, without blocking the
    /// host, for the latest `NextVulkanSignal`.
    virtual void WaitForVulkan(std::uintptr_t stream = 1) = 0;

    /// Makes stream `stream` wait on the GPU for Vulkan timeline
    /// value `value` from `NextVulkanSignal`; nothing for 0.
    virtual void WaitForVulkan(std::uintptr_t stream,
                               std::uint64_t value) = 0;

    /// Blocks the host until every copy queued so far is done.
    virtual void Synchronize() = 0;

  protected:
    ComputeInterop() = default;
};

}  // namespace ernie
