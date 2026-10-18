// SPDX-License-Identifier: BSD-3-Clause
/// @file cuda.h
/// @brief Zero-copy interop with CUDA, and through it PyTorch, Warp and any DLPack consumer:
/// Ernie's shared memory mapped into CUDA in place, and CUDA streams ordered with Ernie's queues
/// through shared timelines, so neither the host nor a copy sits between them. The CUDA driver
/// is loaded at run time: Ernie builds without the CUDA toolkit and runs without a CUDA driver.
#pragma once

#include "ernie/device/column.h"
#include "ernie/util/library.h"
#include "ernie/util/tensor.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>

namespace ernie {

/// @brief The CUDA driver API's handles, declared here so no CUDA header is needed.
using CUresult = int;
using CUdevice = int;
using CUdeviceptr = std::uint64_t;
using CUcontext = struct CUctx_st*;
using CUstream = struct CUstream_st*;
using CUexternalMemory = struct CUextMemory_st*;
using CUexternalSemaphore = struct CUextSemaphore_st*;
using CUmemGenericAllocationHandle = unsigned long long;

/// @brief The CUDA driver API functions Ernie uses, by their ABI names.
struct CudaDriver {
    SharedLibrary library;
    CUresult (*init)(unsigned int) = nullptr;
    CUresult (*deviceGetCount)(int*) = nullptr;
    CUresult (*deviceGet)(CUdevice*, int) = nullptr;
    CUresult (*deviceGetUuid)(unsigned char*, CUdevice) = nullptr;
    CUresult (*primaryCtxRetain)(CUcontext*, CUdevice) = nullptr;
    CUresult (*primaryCtxRelease)(CUdevice) = nullptr;
    CUresult (*ctxPushCurrent)(CUcontext) = nullptr;
    CUresult (*ctxPopCurrent)(CUcontext*) = nullptr;
    CUresult (*ctxSynchronize)() = nullptr;
    CUresult (*importExternalMemory)(CUexternalMemory*, const void*) = nullptr;
    CUresult (*externalMemoryGetMappedBuffer)(CUdeviceptr*, CUexternalMemory,
                                              const void*) = nullptr;
    CUresult (*destroyExternalMemory)(CUexternalMemory) = nullptr;
    CUresult (*importExternalSemaphore)(CUexternalSemaphore*, const void*) = nullptr;
    CUresult (*signalExternalSemaphoresAsync)(const CUexternalSemaphore*, const void*, unsigned int,
                                              CUstream) = nullptr;
    CUresult (*waitExternalSemaphoresAsync)(const CUexternalSemaphore*, const void*, unsigned int,
                                            CUstream) = nullptr;
    CUresult (*destroyExternalSemaphore)(CUexternalSemaphore) = nullptr;
    CUresult (*memFree)(CUdeviceptr) = nullptr;
    CUresult (*memAlloc)(CUdeviceptr*, std::size_t) = nullptr;
    CUresult (*memcpyDtoDAsync)(CUdeviceptr, CUdeviceptr, std::size_t, CUstream) = nullptr;
    CUresult (*memcpyDtoH)(void*, CUdeviceptr, std::size_t) = nullptr;
    CUresult (*memsetD32Async)(CUdeviceptr, unsigned int, std::size_t, CUstream) = nullptr;
    CUresult (*streamSynchronize)(CUstream) = nullptr;
    CUresult (*streamCreate)(CUstream*, unsigned int) = nullptr;
    CUresult (*streamDestroy)(CUstream) = nullptr;
    // Virtual memory management, whose allocations can be exported.
    CUresult (*memCreate)(CUmemGenericAllocationHandle*, std::size_t, const void*,
                          unsigned long long) = nullptr;
    CUresult (*memRelease)(CUmemGenericAllocationHandle) = nullptr;
    CUresult (*memAddressReserve)(CUdeviceptr*, std::size_t, std::size_t, CUdeviceptr,
                                  unsigned long long) = nullptr;
    CUresult (*memAddressFree)(CUdeviceptr, std::size_t) = nullptr;
    CUresult (*memMap)(CUdeviceptr, std::size_t, std::size_t, CUmemGenericAllocationHandle,
                       unsigned long long) = nullptr;
    CUresult (*memUnmap)(CUdeviceptr, std::size_t) = nullptr;
    CUresult (*memSetAccess)(CUdeviceptr, std::size_t, const void*, std::size_t) = nullptr;
    CUresult (*memGetAllocationGranularity)(std::size_t*, const void*, int) = nullptr;
    CUresult (*memRetainAllocationHandle)(CUmemGenericAllocationHandle*, void*) = nullptr;
    CUresult (*memGetAllocationPropertiesFromHandle)(void*, CUmemGenericAllocationHandle) = nullptr;
    CUresult (*memExportToShareableHandle)(void*, CUmemGenericAllocationHandle, int,
                                           unsigned long long) = nullptr;
    CUresult (*memGetAddressRange)(CUdeviceptr*, std::size_t*, CUdeviceptr) = nullptr;
    CUresult (*getErrorString)(CUresult, const char**) = nullptr;

    /// @brief Exportable device memory, as `AllocateShareable` makes it.
    struct Shareable {
        CUdeviceptr pointer = 0;
        CUmemGenericAllocationHandle handle = 0;
        std::size_t size = 0;
    };

    /// @return The driver, loaded and initialized once per process; null when there is none.
    static const CudaDriver* Get();
    /// @brief Allocates at least `bytes` on device `ordinal` through virtual memory management,
    /// with handles another API imports in place. Needs a current context.
    /// @return The allocation, or none on failure.
    std::optional<Shareable> AllocateShareable(int ordinal, std::size_t bytes) const;
    void FreeShareable(const Shareable& allocation) const;
    /// @throws DeviceError when `result` is an error.
    void Check(CUresult result, std::string_view what) const;
};

/// @brief The CUDA device paired with a context's device by UUID. Columns in
/// `MemoryUse::Shared` arenas are viewed as CUDA tensors in place; each arena is mapped once.
/// Streams are CUDA streams (`torch.cuda.current_stream().cuda_stream`, `wp.get_stream()`), 0
/// for the default one.
class CudaInterop {
  public:
    /// @return The interop, or null when there is no CUDA driver, no CUDA device with the
    /// context's UUID, or the device shares no memory or timelines.
    static std::unique_ptr<CudaInterop> Create(DeviceContext& context);
    ~CudaInterop();
    CudaInterop(const CudaInterop&) = delete;
    CudaInterop& operator=(const CudaInterop&) = delete;

    /// @return The CUDA device ordinal, as `cuda:N` names it.
    int Ordinal() const {
        return ordinal_;
    }
    const CudaDriver& Driver() const {
        return driver_;
    }

    /// @brief A CUDA tensor over the column's rows, in place, shaped (rows, element...). It is
    /// valid while the column keeps its range; order access with `Wait` and `Signal`.
    /// @throws DeviceError unless the column's arenas are `MemoryUse::Shared`.
    Tensor View(const DeviceColumn& column);
    /// @brief Makes work enqueued later on `stream` wait for `ticket`, without the host.
    void Wait(CUstream stream, const Ticket& ticket);
    /// @return A ticket reached once the work enqueued on `stream` so far is done, for Ernie's
    /// queues to wait on.
    Ticket Signal(CUstream stream);
    /// @brief Makes `tensor`'s values the column's rows, after the work on `stream` so far: in
    /// place, with no copy, when `tensor` is the column's own view; otherwise by one copy on the
    /// device, counted in `DeviceContext::Transfers`.
    /// @return The ticket Ernie's queues wait on before reading the rows.
    /// @throws std::invalid_argument unless `tensor` is a contiguous CUDA tensor on this device
    /// with the column's byte size.
    Ticket Assign(const DeviceColumn& column, const Tensor& tensor, CUstream stream);
    /// @brief A device column over a CUDA tensor's own memory, in place, when that memory is
    /// exportable: allocated through CUDA's virtual memory management with shareable handles, as
    /// `CudaAllocate` and PyTorch's expandable segments allocate. Ordinary `cudaMalloc` memory is
    /// not; assign such tensors to an Ernie column instead.
    /// @return The column, shaped (rows, element...) as the tensor is and holding it alive; null
    /// when its memory cannot be shared.
    std::unique_ptr<DeviceColumn> Wrap(const Tensor& tensor, std::string_view name);
    /// @brief Allocates `bytes` of exportable device memory, which `Wrap` shares in place.
    /// @return Its address, or 0 on failure.
    CUdeviceptr Allocate(std::size_t bytes);
    /// @brief Frees what `Allocate` returned.
    void Free(CUdeviceptr pointer);

    /// @brief Makes the device's primary context current on this thread for a scope, for calls
    /// to the driver directly.
    class Current {
      public:
        explicit Current(const CudaInterop& interop);
        ~Current();
        Current(const Current&) = delete;
        Current& operator=(const Current&) = delete;

      private:
        const CudaDriver& driver_;
    };

  private:
    struct Mapping {
        CUexternalMemory memory = nullptr;
        CUdeviceptr pointer = 0;
    };

    /// A stream's own timeline, which only that stream signals, so its values only increase.
    struct StreamTimeline {
        std::unique_ptr<Timeline> timeline;
        CUexternalSemaphore semaphore = nullptr;
        std::uint64_t signalled = 0;
    };

    CudaInterop(DeviceContext& context, const CudaDriver& driver, CUdevice device, int ordinal);
    CUdeviceptr Map(const Buffer& buffer);
    /// Imports one of the context's timelines into CUDA, once. Takes `mutex_`'s lock as held.
    CUexternalSemaphore Import(const Timeline& timeline);

    DeviceContext& context_;
    const CudaDriver& driver_;
    CUdevice device_;
    int ordinal_;
    CUcontext cuda_ = nullptr;
    std::mutex mutex_;
    /// Mapped buffers by `Buffer::Id`, which no later buffer reuses.
    std::unordered_map<std::uint64_t, Mapping> mappings_;
    std::unordered_map<const Timeline*, CUexternalSemaphore> semaphores_;
    std::unordered_map<CUstream, StreamTimeline> streams_;
    std::unordered_map<CUdeviceptr, CudaDriver::Shareable> allocations_;
};

}  // namespace ernie
