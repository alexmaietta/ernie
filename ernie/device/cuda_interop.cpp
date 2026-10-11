// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/cuda_interop.h"

#include "ernie/util/profile.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <algorithm>
#include <cstring>
#include <map>
#include <stdexcept>
#include <utility>

namespace ernie {

namespace {

/// The subset of the CUDA driver API Ernie calls, declared from
/// cuda.h (CUDA 12) so Ernie needs no CUDA headers; the driver ABI is
/// stable across versions.
namespace cu {

using Result = int;
using Device = int;
using Context = struct Context_st*;
using Stream = struct Stream_st*;
using Event = struct Event_st*;
using ExternalMemory = struct ExternalMemory_st*;
using ExternalSemaphore = struct ExternalSemaphore_st*;
using Array = struct Array_st*;
using MipmappedArray = struct MipmappedArray_st*;

/// CUDA_ARRAY3D_DESCRIPTOR.
struct Array3dDescriptor {
    std::size_t width;
    std::size_t height;
    std::size_t depth;
    int format;
    unsigned int numChannels;
    unsigned int flags;
};
static_assert(sizeof(Array3dDescriptor) == 40,
              "Must match CUDA_ARRAY3D_DESCRIPTOR");

/// CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC.
struct ExternalMemoryMipmappedArrayDesc {
    unsigned long long offset;
    Array3dDescriptor arrayDesc;
    unsigned int numLevels;
    unsigned int reserved[16];
};
static_assert(sizeof(ExternalMemoryMipmappedArrayDesc) == 120,
              "Must match CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC");

/// CUDA_MEMCPY2D.
struct Memcpy2d {
    std::size_t srcXInBytes;
    std::size_t srcY;
    int srcMemoryType;
    const void* srcHost;
    unsigned long long srcDevice;
    Array srcArray;
    std::size_t srcPitch;
    std::size_t dstXInBytes;
    std::size_t dstY;
    int dstMemoryType;
    void* dstHost;
    unsigned long long dstDevice;
    Array dstArray;
    std::size_t dstPitch;
    std::size_t widthInBytes;
    std::size_t height;
};
static_assert(sizeof(Memcpy2d) == 128, "Must match CUDA_MEMCPY2D");

constexpr int kFormatUnsignedInt8 = 0x01;
constexpr int kFormatUnsignedInt16 = 0x02;
constexpr int kFormatUnsignedInt32 = 0x03;
constexpr int kFormatFloat = 0x20;
constexpr unsigned int kArrayColorAttachment = 0x20;
constexpr int kMemoryHost = 0x01;
constexpr int kMemoryDevice = 0x02;
constexpr int kMemoryArray = 0x03;
using DevicePtr = unsigned long long;

struct Uuid {
    char bytes[16];
};

/// CUDA_EXTERNAL_MEMORY_HANDLE_DESC.
struct ExternalMemoryHandleDesc {
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
static_assert(sizeof(ExternalMemoryHandleDesc) == 104,
              "Must match CUDA_EXTERNAL_MEMORY_HANDLE_DESC");

/// CUDA_EXTERNAL_MEMORY_BUFFER_DESC.
struct ExternalMemoryBufferDesc {
    unsigned long long offset;
    unsigned long long size;
    unsigned int flags;
    unsigned int reserved[16];
};
static_assert(sizeof(ExternalMemoryBufferDesc) == 88,
              "Must match CUDA_EXTERNAL_MEMORY_BUFFER_DESC");

/// CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC.
struct ExternalSemaphoreHandleDesc {
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
static_assert(sizeof(ExternalSemaphoreHandleDesc) == 96,
              "Must match CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC");

/// CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS.
struct ExternalSemaphoreSignalParams {
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
static_assert(sizeof(ExternalSemaphoreSignalParams) == 144,
              "Must match CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS");

/// CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS.
struct ExternalSemaphoreWaitParams {
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
static_assert(sizeof(ExternalSemaphoreWaitParams) == 144,
              "Must match CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS");

constexpr Result kSuccess = 0;
// The handle types Vulkan exports here (ExternalHandle):
// CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_ TIMELINE_SEMAPHORE_WIN32 and
// CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 on Windows, their _FD
// counterparts elsewhere.
#if defined(_WIN32)
constexpr int kSemaphoreTimeline = 10;
constexpr int kMemoryOpaque = 2;
#else
constexpr int kSemaphoreTimeline = 9;
constexpr int kMemoryOpaque = 1;
#endif

/// Puts an exported handle in a CUDA handle description's union.
template <typename Desc>
void SetHandle(Desc& desc, ExternalHandle handle) {
#if defined(_WIN32)
    desc.handle.win32.handle = handle;
#else
    desc.handle.fd = handle;
#endif
}

/// After an import: CUDA owns a file descriptor it imported, but
/// never a Win32 handle.
void ReleaseAfterImport(ExternalHandle handle, bool imported) {
#if defined(_WIN32)
    (void)imported;
    CloseExternalHandle(handle);
#else
    if (!imported) {
        CloseExternalHandle(handle);
    }
#endif
}
constexpr unsigned int kDedicated = 1;
const Stream kLegacyStream = reinterpret_cast<Stream>(0x1);
constexpr unsigned int kEventDisableTiming = 0x2;

struct Driver {
    Result (*init)(unsigned int);
    Result (*deviceGetCount)(int*);
    Result (*deviceGet)(Device*, int);
    Result (*deviceGetUuid)(Uuid*, Device);
    Result (*primaryCtxRetain)(Context*, Device);
    Result (*primaryCtxRelease)(Device);
    Result (*ctxPushCurrent)(Context);
    Result (*ctxPopCurrent)(Context*);
    Result (*importExternalMemory)(ExternalMemory*,
                                   const ExternalMemoryHandleDesc*);
    Result (*externalMemoryGetMappedBuffer)(
        DevicePtr*, ExternalMemory, const ExternalMemoryBufferDesc*);
    Result (*destroyExternalMemory)(ExternalMemory);
    Result (*memAlloc)(DevicePtr*, std::size_t);
    Result (*memFree)(DevicePtr);
    Result (*memcpyDtoDAsync)(DevicePtr, DevicePtr, std::size_t,
                              Stream);
    Result (*memcpyHtoDAsync)(DevicePtr, const void*, std::size_t,
                              Stream);
    Result (*memcpyDtoHAsync)(void*, DevicePtr, std::size_t, Stream);
    Result (*streamSynchronize)(Stream);
    Result (*getErrorString)(Result, const char**);
    Result (*importExternalSemaphore)(
        ExternalSemaphore*, const ExternalSemaphoreHandleDesc*);
    Result (*signalExternalSemaphoresAsync)(
        const ExternalSemaphore*,
        const ExternalSemaphoreSignalParams*, unsigned int, Stream);
    Result (*waitExternalSemaphoresAsync)(
        const ExternalSemaphore*, const ExternalSemaphoreWaitParams*,
        unsigned int, Stream);
    Result (*destroyExternalSemaphore)(ExternalSemaphore);
    Result (*externalMemoryGetMappedMipmappedArray)(
        MipmappedArray*, ExternalMemory,
        const ExternalMemoryMipmappedArrayDesc*);
    Result (*mipmappedArrayGetLevel)(Array*, MipmappedArray,
                                     unsigned int);
    Result (*mipmappedArrayDestroy)(MipmappedArray);
    Result (*memcpy2DAsync)(const Memcpy2d*, Stream);
    Result (*eventCreate)(Event*, unsigned int);
    Result (*eventRecord)(Event, Stream);
    Result (*eventDestroy)(Event);
    Result (*streamWaitEvent)(Stream, Event, unsigned int);
};

}  // namespace cu

// The CUDA driver, loaded at run time: nvcuda.dll on Windows,
// libcuda.so.1 elsewhere.
#if defined(_WIN32)
constexpr const char* kCudaDriver = "nvcuda.dll";
void* OpenLibrary(const char* name) {
    return reinterpret_cast<void*>(LoadLibraryA(name));
}
void* LibrarySymbol(void* library, const char* name) {
    return reinterpret_cast<void*>(
        GetProcAddress(static_cast<HMODULE>(library), name));
}
void CloseLibrary(void* library) {
    FreeLibrary(static_cast<HMODULE>(library));
}
#else
constexpr const char* kCudaDriver = "libcuda.so.1";
void* OpenLibrary(const char* name) {
    return dlopen(name, RTLD_NOW | RTLD_LOCAL);
}
void* LibrarySymbol(void* library, const char* name) {
    return dlsym(library, name);
}
void CloseLibrary(void* library) {
    dlclose(library);
}
#endif

template <typename Function>
bool LoadSymbol(void* library, const char* name, Function& function) {
    function =
        reinterpret_cast<Function>(LibrarySymbol(library, name));
    return function != nullptr;
}

}  // namespace

/// The loaded driver, the device and its retained primary context;
/// shared by everything that holds CUDA resources, so it outlives
/// them.
struct CudaInterop::State {
    void* library = nullptr;
    cu::Driver driver{};
    cu::Device device = 0;
    int ordinal = 0;
    cu::Context context = nullptr;

    /// The shared timelines, CUDA to Vulkan and Vulkan to CUDA, in
    /// both APIs, and the last value each side has signalled.
    VkDevice vulkan = VK_NULL_HANDLE;
    VkSemaphore toVulkan = VK_NULL_HANDLE;
    VkSemaphore toCuda = VK_NULL_HANDLE;
    cu::ExternalSemaphore toVulkanCuda = nullptr;
    cu::ExternalSemaphore toCudaCuda = nullptr;
    std::uint64_t cudaSignalled = 0;
    std::uint64_t vulkanSignalled = 0;
    /// One event per CUDA stream that signals Vulkan, for the legacy
    /// stream to wait on.
    std::map<std::uintptr_t, cu::Event> events;

    /// Destroys the shared timelines. `CudaInterop` does so when it
    /// goes, before the Vulkan device its owner destroys; the state
    /// itself lives on while tensors it allocated do, which only need
    /// CUDA to free their memory.
    void ReleaseVulkan() {
        if (toVulkanCuda != nullptr) {
            driver.destroyExternalSemaphore(toVulkanCuda);
            toVulkanCuda = nullptr;
        }
        if (toCudaCuda != nullptr) {
            driver.destroyExternalSemaphore(toCudaCuda);
            toCudaCuda = nullptr;
        }
        if (vulkan != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(vulkan);
            vkDestroySemaphore(vulkan, toVulkan, nullptr);
            vkDestroySemaphore(vulkan, toCuda, nullptr);
            vulkan = VK_NULL_HANDLE;
        }
    }

    ~State() {
        ReleaseVulkan();
        for (const auto& [stream, event] : events) {
            driver.eventDestroy(event);
        }
        if (context != nullptr) {
            driver.primaryCtxRelease(device);
        }
        if (library != nullptr) {
            CloseLibrary(library);
        }
    }

    void Check(cu::Result result, const char* what) const {
        if (result == cu::kSuccess) {
            return;
        }
        const char* message = nullptr;
        if (driver.getErrorString != nullptr) {
            driver.getErrorString(result, &message);
        }
        throw std::runtime_error(
            std::string(what) + " failed: " +
            (message != nullptr ? message : std::to_string(result)));
    }

    /// Makes the primary context current for the scope.
    struct Scope {
        const State& state;

        explicit Scope(const State& s) : state(s) {
            state.Check(state.driver.ctxPushCurrent(state.context),
                        "cuCtxPushCurrent");
        }

        ~Scope() {
            cu::Context popped = nullptr;
            state.driver.ctxPopCurrent(&popped);
        }
    };
};

namespace {

/// A CUDA import of one shared buffer's or image's memory, attached
/// to it.
struct Import {
    std::shared_ptr<CudaInterop::State> state;
    cu::ExternalMemory memory = nullptr;
    cu::DevicePtr pointer = 0;
    cu::MipmappedArray mipmapped = nullptr;
    cu::Array array = nullptr;

    ~Import() {
        if (mipmapped != nullptr) {
            state->driver.mipmappedArrayDestroy(mipmapped);
        }
        if (memory != nullptr) {
            state->driver.destroyExternalMemory(memory);
        }
    }
};

}  // namespace

std::unique_ptr<CudaInterop>
CudaInterop::Create(const VulkanContext& context, std::string* why) {
    const auto fail =
        [why](std::string reason) -> std::unique_ptr<CudaInterop> {
        if (why != nullptr) {
            *why = std::move(reason);
        }
        return nullptr;
    };
    if (!context.SupportsExternalMemory()) {
        return fail("the Vulkan device cannot export memory and "
                    "semaphores to other APIs");
    }
    auto state = std::make_shared<State>();
    state->library = OpenLibrary(kCudaDriver);
    if (state->library == nullptr) {
        return fail(std::string("the CUDA driver (") + kCudaDriver +
                    ") is not installed");
    }
    cu::Driver& d = state->driver;
    void* lib = state->library;
    const bool loaded =
        LoadSymbol(lib, "cuInit", d.init) &&
        LoadSymbol(lib, "cuDeviceGetCount", d.deviceGetCount) &&
        LoadSymbol(lib, "cuDeviceGet", d.deviceGet) &&
        (LoadSymbol(lib, "cuDeviceGetUuid_v2", d.deviceGetUuid) ||
         LoadSymbol(lib, "cuDeviceGetUuid", d.deviceGetUuid)) &&
        LoadSymbol(lib, "cuDevicePrimaryCtxRetain",
                   d.primaryCtxRetain) &&
        LoadSymbol(lib, "cuDevicePrimaryCtxRelease_v2",
                   d.primaryCtxRelease) &&
        LoadSymbol(lib, "cuCtxPushCurrent_v2", d.ctxPushCurrent) &&
        LoadSymbol(lib, "cuCtxPopCurrent_v2", d.ctxPopCurrent) &&
        LoadSymbol(lib, "cuImportExternalMemory",
                   d.importExternalMemory) &&
        LoadSymbol(lib, "cuExternalMemoryGetMappedBuffer",
                   d.externalMemoryGetMappedBuffer) &&
        LoadSymbol(lib, "cuDestroyExternalMemory",
                   d.destroyExternalMemory) &&
        LoadSymbol(lib, "cuMemAlloc_v2", d.memAlloc) &&
        LoadSymbol(lib, "cuMemFree_v2", d.memFree) &&
        LoadSymbol(lib, "cuMemcpyDtoDAsync_v2", d.memcpyDtoDAsync) &&
        LoadSymbol(lib, "cuMemcpyHtoDAsync_v2", d.memcpyHtoDAsync) &&
        LoadSymbol(lib, "cuMemcpyDtoHAsync_v2", d.memcpyDtoHAsync) &&
        LoadSymbol(lib, "cuStreamSynchronize", d.streamSynchronize) &&
        LoadSymbol(lib, "cuGetErrorString", d.getErrorString) &&
        LoadSymbol(lib, "cuImportExternalSemaphore",
                   d.importExternalSemaphore) &&
        LoadSymbol(lib, "cuSignalExternalSemaphoresAsync",
                   d.signalExternalSemaphoresAsync) &&
        LoadSymbol(lib, "cuWaitExternalSemaphoresAsync",
                   d.waitExternalSemaphoresAsync) &&
        LoadSymbol(lib, "cuDestroyExternalSemaphore",
                   d.destroyExternalSemaphore) &&
        LoadSymbol(lib, "cuExternalMemoryGetMappedMipmappedArray",
                   d.externalMemoryGetMappedMipmappedArray) &&
        LoadSymbol(lib, "cuMipmappedArrayGetLevel",
                   d.mipmappedArrayGetLevel) &&
        LoadSymbol(lib, "cuMipmappedArrayDestroy",
                   d.mipmappedArrayDestroy) &&
        LoadSymbol(lib, "cuMemcpy2DAsync_v2", d.memcpy2DAsync) &&
        LoadSymbol(lib, "cuEventCreate", d.eventCreate) &&
        LoadSymbol(lib, "cuEventRecord", d.eventRecord) &&
        LoadSymbol(lib, "cuEventDestroy_v2", d.eventDestroy) &&
        LoadSymbol(lib, "cuStreamWaitEvent", d.streamWaitEvent);
    if (!loaded) {
        return fail("the CUDA driver lacks external memory interop "
                    "(too old)");
    }
    if (d.init(0) != cu::kSuccess) {
        return fail("cuInit failed; no usable NVIDIA device");
    }
    int count = 0;
    d.deviceGetCount(&count);
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        cu::Device device = 0;
        cu::Uuid uuid{};
        if (d.deviceGet(&device, ordinal) != cu::kSuccess ||
            d.deviceGetUuid(&uuid, device) != cu::kSuccess) {
            continue;
        }
        if (std::memcmp(uuid.bytes, context.DeviceUuid().data(),
                        sizeof(uuid.bytes)) != 0) {
            continue;
        }
        state->device = device;
        state->ordinal = ordinal;
        if (d.primaryCtxRetain(&state->context, device) !=
            cu::kSuccess) {
            state->context = nullptr;
            return fail(
                "cannot retain the CUDA device's primary context");
        }
        try {
            state->vulkan = context.Device();
            state->toVulkan =
                context.CreateTimelineSemaphore(0, true);
            state->toCuda = context.CreateTimelineSemaphore(0, true);
            const State::Scope scope(*state);
            for (const auto& [semaphore, imported] :
                 {std::pair(state->toVulkan, &state->toVulkanCuda),
                  std::pair(state->toCuda, &state->toCudaCuda)}) {
                const ExternalHandle exported =
                    context.ExportSemaphore(semaphore);
                cu::ExternalSemaphoreHandleDesc handle{};
                handle.type = cu::kSemaphoreTimeline;
                cu::SetHandle(handle, exported);
                const bool ok =
                    d.importExternalSemaphore(imported, &handle) ==
                    cu::kSuccess;
                cu::ReleaseAfterImport(exported, ok);
                if (!ok) {
                    return fail("CUDA cannot import Vulkan timeline "
                                "semaphores");
                }
            }
        } catch (const std::runtime_error& error) {
            return fail(error.what());
        }
        return std::unique_ptr<CudaInterop>(
            new CudaInterop(std::move(state)));
    }
    return fail("no CUDA device matches the Vulkan device " +
                context.DeviceName());
}

CudaInterop::CudaInterop(std::shared_ptr<State> state)
    : state_(std::move(state)) {}

CudaInterop::~CudaInterop() {
    if (state_ != nullptr) {
        state_->ReleaseVulkan();
    }
}

int CudaInterop::Device() const {
    return state_->ordinal;
}

std::uint64_t CudaInterop::DevicePointer(GpuBuffer& buffer) {
    if (buffer.Memory() != BufferMemory::Shared) {
        throw std::invalid_argument(
            "CUDA can only reach shared buffers");
    }
    if (auto* import =
            static_cast<Import*>(buffer.Attached().get())) {
        return import->pointer;
    }
    auto import = std::make_shared<Import>();
    import->state = state_;
    const State::Scope scope(*state_);
    const ExternalHandle exported = buffer.ExportHandle();
    cu::ExternalMemoryHandleDesc handle{};
    handle.type = cu::kMemoryOpaque;
    cu::SetHandle(handle, exported);
    handle.size = buffer.AllocationSize();
    handle.flags = cu::kDedicated;
    const cu::Result result =
        state_->driver.importExternalMemory(&import->memory, &handle);
    cu::ReleaseAfterImport(exported, result == cu::kSuccess);
    state_->Check(result, "cuImportExternalMemory");
    cu::ExternalMemoryBufferDesc mapped{};
    mapped.size = buffer.Size();
    state_->Check(state_->driver.externalMemoryGetMappedBuffer(
                      &import->pointer, import->memory, &mapped),
                  "cuExternalMemoryGetMappedBuffer");
    const std::uint64_t pointer = import->pointer;
    buffer.Attach(std::move(import));
    return pointer;
}

std::uint64_t CudaInterop::DevicePointer(GpuImage& image) {
    if (image.AllocationSize() == 0 || !image.Desc().linear) {
        throw std::invalid_argument(
            "CUDA can only address linear shared images");
    }
    if (auto* import = static_cast<Import*>(image.Attached().get())) {
        return import->pointer + image.TexelOffset();
    }
    auto import = std::make_shared<Import>();
    import->state = state_;
    const State::Scope scope(*state_);
    const ExternalHandle exported = image.ExportHandle();
    cu::ExternalMemoryHandleDesc handle{};
    handle.type = cu::kMemoryOpaque;
    cu::SetHandle(handle, exported);
    handle.size = image.AllocationSize();
    handle.flags = cu::kDedicated;
    const cu::Result result =
        state_->driver.importExternalMemory(&import->memory, &handle);
    cu::ReleaseAfterImport(exported, result == cu::kSuccess);
    state_->Check(result, "cuImportExternalMemory");
    // The linear image's memory is its rows, read as a plain buffer.
    cu::ExternalMemoryBufferDesc mapped{};
    mapped.size = image.AllocationSize();
    state_->Check(state_->driver.externalMemoryGetMappedBuffer(
                      &import->pointer, import->memory, &mapped),
                  "cuExternalMemoryGetMappedBuffer");
    const std::uint64_t pointer =
        import->pointer + image.TexelOffset();
    image.Attach(std::move(import));
    return pointer;
}

Tensor CudaInterop::ViewImage(std::shared_ptr<GpuImage> image,
                              DType type,
                              std::vector<std::int64_t> shape,
                              std::vector<std::int64_t> strides) {
    const Tensor shaped(nullptr, type, shape, strides);
    // The farthest byte the view reaches, from the first texel.
    std::int64_t last = 0;
    for (std::size_t d = 0; d < shape.size(); ++d) {
        last += (shape[d] - 1) * shaped.Strides()[d];
    }
    const VkDeviceSize end =
        image->TexelOffset() +
        static_cast<VkDeviceSize>(last + 1) * SizeOf(type);
    const std::uint64_t pointer = DevicePointer(*image);
    if (shaped.Elements() > 0 && end > image->AllocationSize()) {
        throw std::invalid_argument(
            "CudaInterop::ViewImage past the end of the image");
    }
    return Tensor(reinterpret_cast<void*>(pointer), type,
                  std::move(shape), std::move(strides),
                  {DeviceKind::Cuda, state_->ordinal},
                  std::move(image));
}

void CudaInterop::CopyToImage(const Tensor& source, GpuImage& image,
                              std::uintptr_t stream) {
    ERNIE_ZONE("CUDA copy to image");
    CheckOnDevice(source);
    const std::vector<std::int64_t>& shape = source.Shape();
    if (shape.size() < 2 || shape.size() > 4) {
        throw std::invalid_argument(
            "CudaInterop::CopyToImage needs (rows, cols), (rows, "
            "cols, channels) or "
            "(views, rows, cols, channels)");
    }
    const bool batch = shape.size() == 4;
    const auto views = static_cast<std::size_t>(batch ? shape[0] : 1);
    const auto rows = static_cast<std::size_t>(shape[batch ? 1 : 0]);
    const auto cols = static_cast<std::size_t>(shape[batch ? 2 : 1]);
    const std::size_t rowBytes =
        static_cast<std::size_t>(source.Elements()) *
        SizeOf(source.Type()) /
        std::max<std::size_t>(views * rows, 1);
    const ImageDesc& desc = image.Desc();
    if (rows > desc.height || cols * views > desc.width ||
        rowBytes * views > image.RowPitch()) {
        throw std::invalid_argument("CudaInterop::CopyToImage: the "
                                    "views do not fit the image");
    }
    if (rowBytes == 0 || rows == 0) {
        return;
    }
    const std::uint64_t destination = DevicePointer(image);
    WaitForVulkan(stream);
    const State::Scope scope(*state_);
    const auto onStream = reinterpret_cast<cu::Stream>(stream);
    for (std::size_t view = 0; view < views; ++view) {
        cu::Memcpy2d copy{};
        copy.srcMemoryType = cu::kMemoryDevice;
        copy.srcDevice =
            reinterpret_cast<cu::DevicePtr>(source.Data()) +
            view * rows * rowBytes;
        copy.srcPitch = rowBytes;
        copy.dstMemoryType = cu::kMemoryDevice;
        copy.dstDevice = destination + view * rowBytes;
        copy.dstPitch = image.RowPitch();
        copy.widthInBytes = rowBytes;
        copy.height = rows;
        state_->Check(state_->driver.memcpy2DAsync(&copy, onStream),
                      "cuMemcpy2DAsync");
    }
    SignalVulkanAfter(stream);
}

CudaArrayView CudaInterop::MapImage(GpuImage& image) {
    const ImageDesc& desc = image.Desc();
    if (image.AllocationSize() == 0) {
        throw std::invalid_argument(
            "CUDA can only reach shared images");
    }
    if (desc.linear) {
        throw std::invalid_argument("A linear image maps as a tensor "
                                    "(ViewImage), not an array");
    }
    if (desc.layers != 1 || desc.mips != 1) {
        throw std::invalid_argument(
            "CUDA arrays of images need one layer and one mip");
    }
    CudaArrayView view;
    view.width = desc.width;
    view.height = desc.height;
    view.kind = DeviceKind::Cuda;
    view.device = state_->ordinal;
    int format = 0;
    switch (desc.format) {
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R8G8B8A8_SRGB:
        format = cu::kFormatUnsignedInt8;
        view.channels = 4;
        view.type = DType::UInt8;
        break;
    case VK_FORMAT_R32_SFLOAT:
        format = cu::kFormatFloat;
        view.channels = 1;
        view.type = DType::Float32;
        break;
    case VK_FORMAT_R32G32_SFLOAT:
        format = cu::kFormatFloat;
        view.channels = 2;
        view.type = DType::Float32;
        break;
    case VK_FORMAT_R32G32B32A32_SFLOAT:
        format = cu::kFormatFloat;
        view.channels = 4;
        view.type = DType::Float32;
        break;
    case VK_FORMAT_R16G16B16A16_UINT:
        format = cu::kFormatUnsignedInt16;
        view.channels = 4;
        view.type = DType::UInt16;
        break;
    case VK_FORMAT_R32_UINT:
        format = cu::kFormatUnsignedInt32;
        view.channels = 1;
        view.type = DType::UInt32;
        break;
    default:
        throw std::invalid_argument(
            "CUDA arrays support RGBA8, RGBA16 and R32 unsigned, and "
            "float32 images");
    }
    if (auto* import = static_cast<Import*>(image.Attached().get())) {
        view.array = reinterpret_cast<std::uintptr_t>(import->array);
        return view;
    }
    auto import = std::make_shared<Import>();
    import->state = state_;
    const State::Scope scope(*state_);
    const ExternalHandle exported = image.ExportHandle();
    cu::ExternalMemoryHandleDesc handle{};
    handle.type = cu::kMemoryOpaque;
    cu::SetHandle(handle, exported);
    handle.size = image.AllocationSize();
    handle.flags = cu::kDedicated;
    const cu::Result result =
        state_->driver.importExternalMemory(&import->memory, &handle);
    cu::ReleaseAfterImport(exported, result == cu::kSuccess);
    state_->Check(result, "cuImportExternalMemory");
    cu::ExternalMemoryMipmappedArrayDesc mapped{};
    mapped.arrayDesc.width = desc.width;
    mapped.arrayDesc.height = desc.height;
    mapped.arrayDesc.format = format;
    mapped.arrayDesc.numChannels = view.channels;
    mapped.arrayDesc.flags = cu::kArrayColorAttachment;
    mapped.numLevels = 1;
    state_->Check(
        state_->driver.externalMemoryGetMappedMipmappedArray(
            &import->mipmapped, import->memory, &mapped),
        "cuExternalMemoryGetMappedMipmappedArray");
    state_->Check(state_->driver.mipmappedArrayGetLevel(
                      &import->array, import->mipmapped, 0),
                  "cuMipmappedArrayGetLevel");
    view.array = reinterpret_cast<std::uintptr_t>(import->array);
    image.Attach(std::move(import));
    return view;
}

Tensor CudaInterop::DownloadArray(const CudaArrayView& view) {
    Tensor host = Tensor::Zeros(
        view.type, {static_cast<std::int64_t>(view.height),
                    static_cast<std::int64_t>(view.width),
                    static_cast<std::int64_t>(view.channels)});
    WaitForVulkan();
    const State::Scope scope(*state_);
    cu::Memcpy2d copy{};
    copy.srcMemoryType = cu::kMemoryArray;
    copy.srcArray = reinterpret_cast<cu::Array>(view.array);
    copy.dstMemoryType = cu::kMemoryHost;
    copy.dstHost = host.Data();
    copy.dstPitch =
        std::size_t{view.width} * view.channels * SizeOf(view.type);
    copy.widthInBytes = copy.dstPitch;
    copy.height = view.height;
    state_->Check(
        state_->driver.memcpy2DAsync(&copy, cu::kLegacyStream),
        "cuMemcpy2DAsync");
    state_->Check(state_->driver.streamSynchronize(cu::kLegacyStream),
                  "cuStreamSynchronize");
    return host;
}

void CudaInterop::CheckOnDevice(const Tensor& tensor) const {
    if (tensor.Location() !=
        ernie::Device{DeviceKind::Cuda, state_->ordinal}) {
        throw std::invalid_argument(
            "Expected a tensor on CUDA device " +
            std::to_string(state_->ordinal));
    }
    if (!tensor.IsContiguous()) {
        throw std::invalid_argument(
            "Expected a contiguous CUDA tensor");
    }
}

void CudaInterop::CopyToBuffer(const Tensor& source,
                               GpuBuffer& buffer, VkDeviceSize offset,
                               std::uintptr_t stream) {
    ERNIE_ZONE("CUDA copy to buffer");
    CheckOnDevice(source);
    const auto bytes = static_cast<VkDeviceSize>(source.Elements()) *
                       SizeOf(source.Type());
    if (offset > buffer.Size() || bytes > buffer.Size() - offset) {
        throw std::invalid_argument(
            "CudaInterop::CopyToBuffer past the end of the buffer");
    }
    const std::uint64_t destination = DevicePointer(buffer) + offset;
    if (bytes == 0) {
        return;
    }
    // Vulkan work already submitted may still read the buffer (frames
    // in flight): the copy follows it on the GPU.
    WaitForVulkan(stream);
    const State::Scope scope(*state_);
    const auto onStream = reinterpret_cast<cu::Stream>(stream);
    state_->Check(state_->driver.memcpyDtoDAsync(
                      destination,
                      reinterpret_cast<cu::DevicePtr>(source.Data()),
                      bytes, onStream),
                  "cuMemcpyDtoDAsync");
    SignalVulkanAfter(stream);
}

void CudaInterop::CopyColumnsToBuffer(
    const Tensor& source, std::span<const std::size_t> order,
    GpuBuffer& buffer, VkDeviceSize offset, std::uintptr_t stream) {
    ERNIE_ZONE("CUDA copy columns to buffer");
    CheckOnDevice(source);
    if (source.Rank() != 2) {
        throw std::invalid_argument(
            "CudaInterop::CopyColumnsToBuffer needs a (rows, width) "
            "tensor");
    }
    const auto rows = static_cast<std::size_t>(source.Shape()[0]);
    const auto width = static_cast<std::size_t>(source.Shape()[1]);
    for (const std::size_t column : order) {
        if (column >= width) {
            throw std::invalid_argument(
                "CudaInterop::CopyColumnsToBuffer column out of "
                "range");
        }
    }
    const std::size_t element = SizeOf(source.Type());
    const VkDeviceSize bytes = rows * order.size() * element;
    if (offset > buffer.Size() || bytes > buffer.Size() - offset) {
        throw std::invalid_argument(
            "CudaInterop::CopyColumnsToBuffer past the end of the "
            "buffer");
    }
    if (bytes == 0) {
        return;
    }
    WaitForVulkan(stream);
    const State::Scope scope(*state_);
    const auto onStream = reinterpret_cast<cu::Stream>(stream);
    for (std::size_t first = 0; first < order.size();) {
        std::size_t run = 1;
        while (first + run < order.size() &&
               order[first + run] == order[first] + run) {
            ++run;
        }
        cu::Memcpy2d copy{};
        copy.srcMemoryType = cu::kMemoryDevice;
        copy.srcDevice =
            reinterpret_cast<cu::DevicePtr>(source.Data()) +
            order[first] * element;
        copy.srcPitch = width * element;
        copy.dstMemoryType = cu::kMemoryDevice;
        copy.dstDevice =
            DevicePointer(buffer) + offset + first * element;
        copy.dstPitch = order.size() * element;
        copy.widthInBytes = run * element;
        copy.height = rows;
        state_->Check(state_->driver.memcpy2DAsync(&copy, onStream),
                      "cuMemcpy2DAsync");
        first += run;
    }
    SignalVulkanAfter(stream);
}

void CudaInterop::SignalVulkan(std::uintptr_t stream) {
    const State::Scope scope(*state_);
    SignalVulkanAfter(stream);
}

void CudaInterop::SignalVulkanAfter(std::uintptr_t stream) {
    // Every signal goes on the legacy stream, so the timeline's
    // values arrive in order whatever streams the work ran on; other
    // streams' work is ordered before it through an event.
    if (stream != 1) {
        cu::Event& event = state_->events[stream];
        if (event == nullptr) {
            state_->Check(state_->driver.eventCreate(
                              &event, cu::kEventDisableTiming),
                          "cuEventCreate");
        }
        state_->Check(
            state_->driver.eventRecord(
                event, reinterpret_cast<cu::Stream>(stream)),
            "cuEventRecord");
        state_->Check(state_->driver.streamWaitEvent(
                          cu::kLegacyStream, event, 0),
                      "cuStreamWaitEvent");
    }
    cu::ExternalSemaphoreSignalParams signal{};
    signal.params.fence.value = state_->cudaSignalled + 1;
    state_->Check(
        state_->driver.signalExternalSemaphoresAsync(
            &state_->toVulkanCuda, &signal, 1, cu::kLegacyStream),
        "cuSignalExternalSemaphoresAsync");
    ++state_->cudaSignalled;
}

void CudaInterop::Copy(const Tensor& source,
                       const Tensor& destination,
                       std::uintptr_t stream) {
    CheckOnDevice(source);
    CheckOnDevice(destination);
    const auto bytes = static_cast<std::size_t>(source.Elements()) *
                       SizeOf(source.Type());
    if (bytes != static_cast<std::size_t>(destination.Elements()) *
                     SizeOf(destination.Type())) {
        throw std::invalid_argument(
            "CudaInterop::Copy between tensors of different sizes");
    }
    if (bytes == 0) {
        return;
    }
    const State::Scope scope(*state_);
    state_->Check(
        state_->driver.memcpyDtoDAsync(
            reinterpret_cast<cu::DevicePtr>(destination.Data()),
            reinterpret_cast<cu::DevicePtr>(source.Data()), bytes,
            reinterpret_cast<cu::Stream>(stream)),
        "cuMemcpyDtoDAsync");
}

TimelinePoint CudaInterop::VulkanWait() const {
    if (state_->cudaSignalled == 0) {
        return {};
    }
    return {state_->toVulkan, state_->cudaSignalled};
}

TimelinePoint CudaInterop::NextVulkanSignal() {
    return {state_->toCuda, ++state_->vulkanSignalled};
}

void CudaInterop::WaitForVulkan(std::uintptr_t stream) {
    WaitForVulkan(stream, state_->vulkanSignalled);
}

void CudaInterop::WaitForVulkan(std::uintptr_t stream,
                                std::uint64_t value) {
    if (value == 0) {
        return;
    }
    const State::Scope scope(*state_);
    cu::ExternalSemaphoreWaitParams wait{};
    wait.params.fence.value = value;
    state_->Check(state_->driver.waitExternalSemaphoresAsync(
                      &state_->toCudaCuda, &wait, 1,
                      reinterpret_cast<cu::Stream>(stream)),
                  "cuWaitExternalSemaphoresAsync");
}

void CudaInterop::Synchronize() {
    const State::Scope scope(*state_);
    state_->Check(state_->driver.streamSynchronize(cu::kLegacyStream),
                  "cuStreamSynchronize");
}

Tensor CudaInterop::View(std::shared_ptr<GpuBuffer> buffer,
                         DType type, std::vector<std::int64_t> shape,
                         VkDeviceSize offset) {
    const Tensor shaped(nullptr, type, shape);
    const auto bytes =
        static_cast<VkDeviceSize>(shaped.Elements()) * SizeOf(type);
    if (offset > buffer->Size() || bytes > buffer->Size() - offset) {
        throw std::invalid_argument(
            "CudaInterop::View larger than the buffer");
    }
    const std::uint64_t pointer = DevicePointer(*buffer) + offset;
    return Tensor(
        reinterpret_cast<void*>(pointer), type, std::move(shape), {},
        {DeviceKind::Cuda, state_->ordinal}, std::move(buffer));
}

Tensor CudaInterop::Upload(const Tensor& host) {
    ERNIE_ZONE("CUDA upload (waits)");
    if (host.Location().kind != DeviceKind::Cpu) {
        throw std::invalid_argument(
            "CudaInterop::Upload needs a CPU tensor");
    }
    const Tensor packed = host.Contiguous();
    const std::size_t bytes =
        static_cast<std::size_t>(packed.Elements()) *
        SizeOf(packed.Type());
    const State::Scope scope(*state_);
    cu::DevicePtr pointer = 0;
    state_->Check(state_->driver.memAlloc(
                      &pointer, std::max<std::size_t>(bytes, 1)),
                  "cuMemAlloc");
    std::shared_ptr<void> owner(
        reinterpret_cast<void*>(pointer), [state = state_](void* p) {
            const State::Scope freeing(*state);
            state->driver.memFree(reinterpret_cast<cu::DevicePtr>(p));
        });
    if (bytes > 0) {
        state_->Check(
            state_->driver.memcpyHtoDAsync(pointer, packed.Data(),
                                           bytes, cu::kLegacyStream),
            "cuMemcpyHtoDAsync");
        state_->Check(
            state_->driver.streamSynchronize(cu::kLegacyStream),
            "cuStreamSynchronize");
    }
    return Tensor(reinterpret_cast<void*>(pointer), packed.Type(),
                  packed.Shape(), {},
                  {DeviceKind::Cuda, state_->ordinal},
                  std::move(owner));
}

Tensor CudaInterop::Download(const Tensor& device,
                             std::uintptr_t stream) {
    ERNIE_ZONE("CUDA download (waits)");
    CheckOnDevice(device);
    Tensor host = Tensor::Zeros(device.Type(), device.Shape());
    const std::size_t bytes =
        static_cast<std::size_t>(device.Elements()) *
        SizeOf(device.Type());
    WaitForVulkan(stream);
    if (bytes > 0) {
        const State::Scope scope(*state_);
        const auto onStream = reinterpret_cast<cu::Stream>(stream);
        state_->Check(
            state_->driver.memcpyDtoHAsync(
                host.Data(),
                reinterpret_cast<cu::DevicePtr>(device.Data()), bytes,
                onStream),
            "cuMemcpyDtoHAsync");
        state_->Check(state_->driver.streamSynchronize(onStream),
                      "cuStreamSynchronize");
    }
    return host;
}

}  // namespace ernie
