// SPDX-License-Identifier: BSD-3-Clause
/// @file cuda.cpp
/// @brief Loading the CUDA driver, pairing devices by UUID, and importing Ernie's memory and
/// timelines into CUDA.
#include "ernie/device/cuda.h"

#include "ernie/util/on_failure.h"
#include "ernie/util/text.h"

#include <cstring>
#include <stdexcept>

namespace ernie {
namespace {

// The driver API's descriptor layouts, as cuda.h's _v1 structs declare them.
struct MemoryHandleDesc {
    int type;
    union {
        int fd;
        struct {
            void* handle;
            const void* name;
        } win32;
        const void* nvSciBufObject;
    } handle;
    unsigned long long size;
    unsigned int flags;
    unsigned int reserved[16];
};
struct BufferDesc {
    unsigned long long offset;
    unsigned long long size;
    unsigned int flags;
    unsigned int reserved[16];
};
struct SemaphoreHandleDesc {
    int type;
    union {
        int fd;
        struct {
            void* handle;
            const void* name;
        } win32;
        const void* nvSciSyncObj;
    } handle;
    unsigned int flags;
    unsigned int reserved[16];
};
struct SemaphoreSignalParams {
    struct {
        struct {
            unsigned long long value;
        } fence;
        union {
            void* fence;
            unsigned long long reserved;
        } nvSciSync;
        struct {
            unsigned long long key;
        } keyedMutex;
        unsigned int reserved[12];
    } params;
    unsigned int flags;
    unsigned int reserved[16];
};
struct SemaphoreWaitParams {
    struct {
        struct {
            unsigned long long value;
        } fence;
        union {
            void* fence;
            unsigned long long reserved;
        } nvSciSync;
        struct {
            unsigned long long key;
            unsigned int timeoutMs;
        } keyedMutex;
        unsigned int reserved[10];
    } params;
    unsigned int flags;
    unsigned int reserved[16];
};

// Virtual memory management's layouts, as cuda.h's _v1 structs declare them.
struct MemoryLocation {
    int type;
    int id;
};
struct AllocationProperties {
    int type;
    int requestedHandleTypes;
    MemoryLocation location;
    void* win32HandleMetaData;
    struct {
        unsigned char compressionType;
        unsigned char gpuDirectRDMACapable;
        unsigned short usage;
        unsigned char reserved[4];
    } allocFlags;
};
struct AccessDescription {
    MemoryLocation location;
    int flags;
};
constexpr int kPinned = 1;          // CU_MEM_ALLOCATION_TYPE_PINNED
constexpr int kOnDevice = 1;        // CU_MEM_LOCATION_TYPE_DEVICE
constexpr int kReadWrite = 3;       // CU_MEM_ACCESS_FLAGS_PROT_READWRITE
constexpr int kMinimumGranule = 0;  // CU_MEM_ALLOC_GRANULARITY_MINIMUM

#if defined(_WIN32)
constexpr int kShareableHandle = 2;   // CU_MEM_HANDLE_TYPE_WIN32
constexpr int kMemoryHandleType = 2;  // CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32
constexpr int kSemaphoreHandleType =
    10;  // CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32
#else
constexpr int kShareableHandle = 1;      // CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
constexpr int kMemoryHandleType = 1;     // CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD
constexpr int kSemaphoreHandleType = 9;  // CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD
#endif
constexpr unsigned int kDedicated = 1;  // CUDA_EXTERNAL_MEMORY_DEDICATED

/// Fills a descriptor's handle from an exported one, which CUDA takes where the platform does.
template <typename Desc> void SetHandle(Desc& desc, const ExternalHandle& handle) {
#if defined(_WIN32)
    desc.handle.win32.handle = reinterpret_cast<void*>(handle.Get());
#else
    desc.handle.fd = static_cast<int>(handle.Get());
#endif
}

bool LoadDriver(CudaDriver& d) {
    if (!d.library.Open({"libcuda.so.1", "libcuda.so", "nvcuda.dll"})) {
        return false;
    }
    const SharedLibrary& l = d.library;
    return l.Load(d.init, "cuInit") && l.Load(d.deviceGetCount, "cuDeviceGetCount") &&
           l.Load(d.deviceGet, "cuDeviceGet") && l.Load(d.deviceGetUuid, "cuDeviceGetUuid_v2") &&
           l.Load(d.primaryCtxRetain, "cuDevicePrimaryCtxRetain") &&
           l.Load(d.primaryCtxRelease, "cuDevicePrimaryCtxRelease_v2") &&
           l.Load(d.ctxPushCurrent, "cuCtxPushCurrent_v2") &&
           l.Load(d.ctxPopCurrent, "cuCtxPopCurrent_v2") &&
           l.Load(d.ctxSynchronize, "cuCtxSynchronize") &&
           l.Load(d.importExternalMemory, "cuImportExternalMemory") &&
           l.Load(d.externalMemoryGetMappedBuffer, "cuExternalMemoryGetMappedBuffer") &&
           l.Load(d.destroyExternalMemory, "cuDestroyExternalMemory") &&
           l.Load(d.importExternalSemaphore, "cuImportExternalSemaphore") &&
           l.Load(d.signalExternalSemaphoresAsync, "cuSignalExternalSemaphoresAsync") &&
           l.Load(d.waitExternalSemaphoresAsync, "cuWaitExternalSemaphoresAsync") &&
           l.Load(d.destroyExternalSemaphore, "cuDestroyExternalSemaphore") &&
           l.Load(d.memFree, "cuMemFree_v2") && l.Load(d.memAlloc, "cuMemAlloc_v2") &&
           l.Load(d.memcpyDtoDAsync, "cuMemcpyDtoDAsync_v2") &&
           l.Load(d.memcpyDtoH, "cuMemcpyDtoH_v2") &&
           l.Load(d.memsetD32Async, "cuMemsetD32Async") &&
           l.Load(d.streamSynchronize, "cuStreamSynchronize") &&
           l.Load(d.streamCreate, "cuStreamCreate") &&
           l.Load(d.streamDestroy, "cuStreamDestroy_v2") && l.Load(d.memCreate, "cuMemCreate") &&
           l.Load(d.memRelease, "cuMemRelease") &&
           l.Load(d.memAddressReserve, "cuMemAddressReserve") &&
           l.Load(d.memAddressFree, "cuMemAddressFree") && l.Load(d.memMap, "cuMemMap") &&
           l.Load(d.memUnmap, "cuMemUnmap") && l.Load(d.memSetAccess, "cuMemSetAccess") &&
           l.Load(d.memGetAllocationGranularity, "cuMemGetAllocationGranularity") &&
           l.Load(d.memRetainAllocationHandle, "cuMemRetainAllocationHandle") &&
           l.Load(d.memGetAllocationPropertiesFromHandle,
                  "cuMemGetAllocationPropertiesFromHandle") &&
           l.Load(d.memExportToShareableHandle, "cuMemExportToShareableHandle") &&
           l.Load(d.memGetAddressRange, "cuMemGetAddressRange_v2") &&
           l.Load(d.getErrorString, "cuGetErrorString") && d.init(0) == 0;
}

}  // namespace

const CudaDriver* CudaDriver::Get() {
    static const std::unique_ptr<CudaDriver> driver = [] {
        auto loaded = std::make_unique<CudaDriver>();
        return LoadDriver(*loaded) ? std::move(loaded) : nullptr;
    }();
    return driver.get();
}

void CudaDriver::Check(CUresult result, std::string_view what) const {
    if (result != 0) {
        const char* message = nullptr;
        getErrorString(result, &message);
        throw DeviceError(DeviceErrorKind::Failed,
                          std::string(what) + ": " + (message ? message : "CUDA error"));
    }
}

CudaInterop::Current::Current(const CudaInterop& interop) : driver_(interop.driver_) {
    driver_.Check(driver_.ctxPushCurrent(interop.cuda_), "Making the CUDA context current");
}

CudaInterop::Current::~Current() {
    CUcontext popped = nullptr;
    driver_.ctxPopCurrent(&popped);
}

std::unique_ptr<CudaInterop> CudaInterop::Create(DeviceContext& context) {
    const CudaDriver* driver = CudaDriver::Get();
    const DeviceCapabilities& capabilities = context.Capabilities();
    if (driver == nullptr || !capabilities.externalMemory || !capabilities.externalTimeline) {
        return nullptr;
    }
    int count = 0;
    driver->Check(driver->deviceGetCount(&count), "Counting CUDA devices");
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        CUdevice device = 0;
        unsigned char uuid[16] = {};
        if (driver->deviceGet(&device, ordinal) == 0 && driver->deviceGetUuid(uuid, device) == 0 &&
            Hex(uuid, sizeof(uuid)) == context.Info().uuid) {
            return std::unique_ptr<CudaInterop>(new CudaInterop(context, *driver, device, ordinal));
        }
    }
    return nullptr;
}

CudaInterop::CudaInterop(DeviceContext& context, const CudaDriver& driver, CUdevice device,
                         int ordinal)
    : context_(context), driver_(driver), device_(device), ordinal_(ordinal), allocations_(driver) {
    // The primary context, which PyTorch, Warp and the runtime API share.
    driver_.Check(driver_.primaryCtxRetain(&cuda_, device_), "Retaining the CUDA context");
}

CudaInterop::~CudaInterop() {
    {
        // Neither API may still use what goes: CUDA's streams finish, then Ernie's queues, which
        // may wait on the streams' timelines.
        const Current current(*this);
        driver_.ctxSynchronize();
        Quietly([this] { context_.WaitIdle(); });
        for (const auto& [id, mapping] : mappings_) {
            driver_.memFree(mapping.pointer);
            driver_.destroyExternalMemory(mapping.memory);
        }
        // The context's queue timelines and the streams' own.
        for (const auto& [timeline, semaphore] : semaphores_) {
            driver_.destroyExternalSemaphore(semaphore);
        }
        allocations_.FreeAll();
    }
    driver_.primaryCtxRelease(device_);
}

CUdeviceptr CudaInterop::Map(const Buffer& buffer) {
    const std::lock_guard lock(mutex_);
    const auto found = mappings_.find(buffer.Id());
    if (found != mappings_.end()) {
        return found->second.pointer;
    }
    ERNIE_ZONE("Map memory into CUDA");
    ExternalHandle handle = buffer.Export();
    MemoryHandleDesc memory{.type = kMemoryHandleType,
                            .handle = {},
                            .size = buffer.MemorySize(),
                            .flags = kDedicated,
                            .reserved = {}};
    SetHandle(memory, handle);
    const Current current(*this);
    Mapping mapping;
    driver_.Check(driver_.importExternalMemory(&mapping.memory, &memory),
                  "Importing memory into CUDA");
    handle.Imported();
    const BufferDesc whole{.offset = 0, .size = buffer.Size(), .flags = 0, .reserved = {}};
    const CUresult mapped =
        driver_.externalMemoryGetMappedBuffer(&mapping.pointer, mapping.memory, &whole);
    if (mapped != 0) {
        driver_.destroyExternalMemory(mapping.memory);
        driver_.Check(mapped, "Mapping memory in CUDA");
    }
    mappings_.emplace(buffer.Id(), mapping);
    return mapping.pointer;
}

CUexternalSemaphore CudaInterop::Import(const Timeline& timeline) {
    const auto found = semaphores_.find(&timeline);
    if (found != semaphores_.end()) {
        return found->second;
    }
    ExternalHandle handle = timeline.Export();
    SemaphoreHandleDesc desc{
        .type = kSemaphoreHandleType, .handle = {}, .flags = 0, .reserved = {}};
    SetHandle(desc, handle);
    const Current current(*this);
    CUexternalSemaphore semaphore = nullptr;
    driver_.Check(driver_.importExternalSemaphore(&semaphore, &desc),
                  "Importing " + timeline.Name() + " into CUDA");
    handle.Imported();
    semaphores_.emplace(&timeline, semaphore);
    return semaphore;
}

Tensor CudaInterop::View(const DeviceColumn& column) {
    const CUdeviceptr base = Map(column.Storage());
    return Tensor(reinterpret_cast<void*>(base + column.Range().offset), column.Type(),
                  column.Shape(), {}, Device{DeviceKind::Cuda, ordinal_});
}

void CudaInterop::CheckOnDevice(const Tensor& tensor, std::string_view what) const {
    if (tensor.Location() != Device{DeviceKind::Cuda, ordinal_} || !tensor.IsContiguous() ||
        tensor.Rank() == 0) {
        throw std::invalid_argument(
            std::string(what) + " takes a contiguous tensor on cuda:" + std::to_string(ordinal_));
    }
}

void CudaInterop::Wait(CUstream stream, const Ticket& ticket) {
    if (ticket.timeline != nullptr && ticket.timeline->Device() != context_.Device()) {
        throw DeviceError(DeviceErrorKind::Failed,
                          "CUDA cannot wait on another device's " + ticket.timeline->Name());
    }
    if (ticket.Done()) {
        return;
    }
    SemaphoreWaitParams wait{};
    wait.params.fence.value = ticket.value;
    const std::lock_guard lock(mutex_);
    const CUexternalSemaphore semaphore = Import(*ticket.timeline);
    const Current current(*this);
    driver_.Check(driver_.waitExternalSemaphoresAsync(&semaphore, &wait, 1, stream),
                  "Waiting on " + ticket.timeline->Name() + " in CUDA");
}

Ticket CudaInterop::Signal(CUstream stream) {
    const std::lock_guard lock(mutex_);
    StreamTimeline& timeline = streams_[stream];
    if (!timeline.timeline) {
        auto made = std::make_unique<Timeline>(context_.Device(), "a CUDA stream", true);
        timeline.semaphore = Import(*made);
        timeline.timeline = std::move(made);
    }
    SemaphoreSignalParams signal{};
    signal.params.fence.value = timeline.signalled + 1;
    const Current current(*this);
    driver_.Check(driver_.signalExternalSemaphoresAsync(&timeline.semaphore, &signal, 1, stream),
                  "Signalling from CUDA");
    return timeline.timeline->At(++timeline.signalled);
}

Ticket CudaInterop::Assign(const DeviceColumn& column, const Tensor& tensor, CUstream stream) {
    const std::uint64_t bytes = column.Bytes();
    CheckOnDevice(tensor, "Assigning " + column.Name());
    if (tensor.Bytes() != bytes) {
        throw std::invalid_argument("Assigning " + column.Name() + " takes a tensor of its size");
    }
    const CUdeviceptr rows = Map(column.Storage()) + column.Range().offset;
    const auto source = reinterpret_cast<CUdeviceptr>(tensor.Data());
    if (source != rows) {
        // After Ernie's work that may still read the rows.
        for (const Queue* queue : context_.Queues()) {
            Wait(stream, queue->Last());
        }
        const Current current(*this);
        driver_.Check(driver_.memcpyDtoDAsync(rows, source, bytes, stream),
                      "Copying a tensor into " + column.Name());
        context_.CountCopy(bytes);
    }
    return Signal(stream);
}

std::unique_ptr<DeviceColumn> CudaInterop::Wrap(const Tensor& tensor, std::string_view name) {
    ERNIE_ZONE("Wrap a CUDA tensor");
    CheckOnDevice(tensor, "Wrapping " + std::string(name));
    const auto pointer = reinterpret_cast<CUdeviceptr>(tensor.Data());
    const std::uint64_t bytes = tensor.Bytes();
    const Current current(*this);
    // Only virtual memory allocations have handles, and only those asked to be shareable export.
    CUmemGenericAllocationHandle allocation = 0;
    if (driver_.memRetainAllocationHandle(&allocation, reinterpret_cast<void*>(pointer)) != 0) {
        return nullptr;
    }
    AllocationProperties properties{};
    CUdeviceptr base = 0;
    std::size_t size = 0;
    ExternalHandle exported;
    if (driver_.memGetAllocationPropertiesFromHandle(&properties, allocation) == 0 &&
        (properties.requestedHandleTypes & kShareableHandle) != 0 &&
        driver_.memGetAddressRange(&base, &size, pointer) == 0 && pointer + bytes <= base + size) {
#if defined(_WIN32)
        void* handle = nullptr;
        if (driver_.memExportToShareableHandle(&handle, allocation, kShareableHandle, 0) == 0) {
            exported = ExternalHandle(reinterpret_cast<std::intptr_t>(handle));
        }
#else
        int handle = -1;
        if (driver_.memExportToShareableHandle(&handle, allocation, kShareableHandle, 0) == 0) {
            exported = ExternalHandle(handle);
        }
#endif
    }
    driver_.memRelease(allocation);
    if (!exported) {
        return nullptr;
    }
    auto storage = std::make_shared<const Buffer>(context_, std::move(exported), size, name,
                                                  SharedFrom::OtherApi);
    std::vector<std::int64_t> element(tensor.Shape().begin() + 1, tensor.Shape().end());
    return std::make_unique<DeviceColumn>(
        context_, std::move(storage), pointer - base, name, tensor.Type(), std::move(element),
        static_cast<std::uint64_t>(tensor.Shape().front()), std::make_shared<const Tensor>(tensor));
}

std::optional<CudaDriver::Shareable> CudaDriver::AllocateShareable(int ordinal,
                                                                   std::size_t bytes) const {
    const AllocationProperties properties{.type = kPinned,
                                          .requestedHandleTypes = kShareableHandle,
                                          .location = {kOnDevice, ordinal},
                                          .win32HandleMetaData = nullptr,
                                          .allocFlags = {}};
    std::size_t granule = 0;
    if (memGetAllocationGranularity(&granule, &properties, kMinimumGranule) != 0) {
        return std::nullopt;
    }
    Shareable allocation;
    allocation.size = (std::max<std::size_t>(bytes, 1) + granule - 1) / granule * granule;
    if (memCreate(&allocation.handle, allocation.size, &properties, 0) != 0) {
        return std::nullopt;
    }
    if (memAddressReserve(&allocation.pointer, allocation.size, 0, 0, 0) != 0) {
        memRelease(allocation.handle);
        return std::nullopt;
    }
    const AccessDescription access{.location = {kOnDevice, ordinal}, .flags = kReadWrite};
    if (memMap(allocation.pointer, allocation.size, 0, allocation.handle, 0) != 0 ||
        memSetAccess(allocation.pointer, allocation.size, &access, 1) != 0) {
        FreeShareable(allocation);
        return std::nullopt;
    }
    return allocation;
}

void CudaDriver::FreeShareable(const Shareable& allocation) const {
    memUnmap(allocation.pointer, allocation.size);
    memAddressFree(allocation.pointer, allocation.size);
    memRelease(allocation.handle);
}

CUdeviceptr ShareableAllocations::Allocate(int ordinal, std::size_t bytes) {
    const std::optional<CudaDriver::Shareable> allocation =
        driver_.AllocateShareable(ordinal, bytes);
    if (!allocation) {
        return 0;
    }
    const std::lock_guard lock(mutex_);
    live_.emplace(allocation->pointer, *allocation);
    return allocation->pointer;
}

void ShareableAllocations::Free(CUdeviceptr pointer) {
    CudaDriver::Shareable allocation;
    {
        const std::lock_guard lock(mutex_);
        const auto found = live_.find(pointer);
        if (found == live_.end()) {
            return;
        }
        allocation = found->second;
        live_.erase(found);
    }
    driver_.FreeShareable(allocation);
}

void ShareableAllocations::FreeAll() {
    const std::lock_guard lock(mutex_);
    for (const auto& [pointer, allocation] : live_) {
        driver_.FreeShareable(allocation);
    }
    live_.clear();
}

CUdeviceptr CudaInterop::Allocate(std::size_t bytes) {
    const Current current(*this);
    return allocations_.Allocate(ordinal_, bytes);
}

void CudaInterop::Free(CUdeviceptr pointer) {
    const Current current(*this);
    allocations_.Free(pointer);
}

}  // namespace ernie
