// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/vulkan/interop.h"

#include <condition_variable>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

#if !defined(_WIN32)
#include <unistd.h>
#endif

#include "ernie/device/cuda/api.h"
#include "ernie/device/cuda/device.h"
#include "ernie/device/hip/api.h"
#include "ernie/device/hip/device.h"
#include "ernie/util/error.h"
#include "ernie/util/log.h"

namespace ernie::device::vulkan {

namespace {

constexpr uint32_t kVendorNvidia = 0x10de;
constexpr uint32_t kVendorAmd = 0x1002;

int ExportFd(VulkanDevice& device, VkSemaphore semaphore) {
#if defined(_WIN32)
    (void)device;
    (void)semaphore;
    throw UnsupportedError("Vulkan's semaphores aren't shared with CUDA or HIP on Windows yet");
#else
    VkSemaphoreGetFdInfoKHR info{VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
    info.semaphore = semaphore;
    info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    int fd = -1;
    Check(device.Functions().vkGetSemaphoreFdKHR(device.Handle(), &info, &fd),
          "vkGetSemaphoreFdKHR");
    return fd;
#endif
}

void CloseFd(int fd) {
#if !defined(_WIN32)
    close(fd);
#else
    (void)fd;
#endif
}

struct CudaDevice {
    cuda::CUdevice device = 0;
    int32_t ordinal = 0;
};

CudaDevice FindCudaDevice(const cuda::Api& cuda, const VulkanDevice& device) {
    int count = 0;
    cuda.Check(cuda.cuDeviceGetCount(&count), "cuDeviceGetCount");
    CudaDevice found;
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        cuda::CUuuid uuid{};
        cuda.Check(cuda.cuDeviceGet(&found.device, ordinal), "cuDeviceGet");
        cuda.Check(cuda.cuDeviceGetUuid(&uuid, found.device), "cuDeviceGetUuid");
        if (std::memcmp(uuid.bytes, device.Adapter().uuid.data(), 16) == 0) {
            found.ordinal = ordinal;
            return found;
        }
    }
    throw UnsupportedError("CUDA has no GPU of the UUID of {}", device.Adapter().alias);
}

int FindHipDevice(const hip::Api& hip, const VulkanDevice& device) {
    int count = 0;
    hip.Check(hip.hipGetDeviceCount(&count), "hipGetDeviceCount");
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        hip::hipUUID uuid{};
        hip.Check(hip.hipDeviceGetUuid(&uuid, ordinal), "hipDeviceGetUuid");
        if (std::memcmp(uuid.bytes, device.Adapter().uuid.data(), 16) == 0) {
            return ordinal;
        }
    }
    throw UnsupportedError("HIP has no GPU of the UUID of {}", device.Adapter().alias);
}

class CudaContext {
  public:
    CudaContext(const cuda::Api& cuda, cuda::CUcontext context) : m_cuda(cuda) {
        m_cuda.Check(m_cuda.cuCtxPushCurrent(context), "cuCtxPushCurrent");
    }

    ~CudaContext() {
        cuda::CUcontext popped = nullptr;
        m_cuda.Warn(m_cuda.cuCtxPopCurrent(&popped), "cuCtxPopCurrent");
    }

    CudaContext(const CudaContext&) = delete;
    CudaContext& operator=(const CudaContext&) = delete;

  private:
    const cuda::Api& m_cuda;
};

class CudaInterop : public StreamInterop {
  public:
    CudaInterop(VulkanDevice& device, VkSemaphore timeline, VkSemaphore follow)
        : m_cuda(cuda::GetApi()) {
        m_device = FindCudaDevice(m_cuda, device).device;
        m_cuda.Check(m_cuda.cuDevicePrimaryCtxRetain(&m_context, m_device),
                     "cuDevicePrimaryCtxRetain");
        const CudaContext scope(m_cuda, m_context);
        m_stream = cuda::Traits::CreateStream(m_cuda);
        m_timeline = Import(device, timeline);
        m_follow = Import(device, follow);
    }

    ~CudaInterop() override {
        {
            const CudaContext scope(m_cuda, m_context);
            if (m_follow) {
                m_cuda.Warn(m_cuda.cuDestroyExternalSemaphore(m_follow),
                            "cuDestroyExternalSemaphore");
            }
            if (m_timeline) {
                m_cuda.Warn(m_cuda.cuDestroyExternalSemaphore(m_timeline),
                            "cuDestroyExternalSemaphore");
            }
            if (m_stream) {
                cuda::Traits::DestroyStream(m_cuda, m_stream);
            }
        }
        m_cuda.Warn(m_cuda.cuDevicePrimaryCtxRelease(m_device), "cuDevicePrimaryCtxRelease");
    }

    void Follow(void* stream, uint64_t value) override {
        const CudaContext scope(m_cuda, m_context);
        const cuda::CUevent event = cuda::Traits::CreateEvent(m_cuda, false);
        cuda::Traits::Record(m_cuda, event, cuda::Traits::ApplicationStream(stream), false);
        cuda::Traits::WaitEvent(m_cuda, m_stream, event);
        cuda::Traits::DestroyEvent(m_cuda, event);
        cuda::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS params{};
        params.params.fence.value = value;
        m_cuda.Check(m_cuda.cuSignalExternalSemaphoresAsync(&m_follow, &params, 1, m_stream),
                     "cuSignalExternalSemaphoresAsync");
    }

    void Wait(void* stream, uint64_t value) override {
        const CudaContext scope(m_cuda, m_context);
        cuda::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS params{};
        params.params.fence.value = value;
        m_cuda.Check(m_cuda.cuWaitExternalSemaphoresAsync(&m_timeline, &params, 1,
                                                          cuda::Traits::ApplicationStream(stream)),
                     "cuWaitExternalSemaphoresAsync");
    }

  private:
    cuda::CUexternalSemaphore Import(VulkanDevice& device, VkSemaphore semaphore) {
        const int fd = ExportFd(device, semaphore);
        cuda::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC desc{};
        desc.type = cuda::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD;
        desc.handle.fd = fd;
        cuda::CUexternalSemaphore imported = nullptr;
        const cuda::CUresult result = m_cuda.cuImportExternalSemaphore(&imported, &desc);
        if (result != 0) {
            CloseFd(fd);
        }
        m_cuda.Check(result, "cuImportExternalSemaphore");
        return imported;
    }

    const cuda::Api& m_cuda;
    cuda::CUdevice m_device = 0;
    cuda::CUcontext m_context = nullptr;
    cuda::CUstream m_stream = nullptr;
    cuda::CUexternalSemaphore m_timeline = nullptr;
    cuda::CUexternalSemaphore m_follow = nullptr;
};

class HipInterop : public StreamInterop {
  public:
    HipInterop(VulkanDevice& device, VkSemaphore timeline, VkSemaphore follow)
        : m_hip(hip::GetApi()) {
        m_device = FindHipDevice(m_hip, device);
        Enter();
        m_stream = hip::Traits::CreateStream(m_hip);
        m_timeline = Import(device, timeline);
        m_follow = Import(device, follow);
    }

    ~HipInterop() override {
        m_hip.Warn(m_hip.hipSetDevice(m_device), "hipSetDevice");
        if (m_follow) {
            m_hip.Warn(m_hip.hipDestroyExternalSemaphore(m_follow), "hipDestroyExternalSemaphore");
        }
        if (m_timeline) {
            m_hip.Warn(m_hip.hipDestroyExternalSemaphore(m_timeline),
                       "hipDestroyExternalSemaphore");
        }
        if (m_stream) {
            hip::Traits::DestroyStream(m_hip, m_stream);
        }
    }

    void Follow(void* stream, uint64_t value) override {
        Enter();
        const hip::hipEvent_t event = hip::Traits::CreateEvent(m_hip, false);
        hip::Traits::Record(m_hip, event, hip::Traits::ApplicationStream(stream), false);
        hip::Traits::WaitEvent(m_hip, m_stream, event);
        hip::Traits::DestroyEvent(m_hip, event);
        hip::hipExternalSemaphoreSignalParams params{};
        params.params.fence.value = value;
        m_hip.Check(m_hip.hipSignalExternalSemaphoresAsync(&m_follow, &params, 1, m_stream),
                    "hipSignalExternalSemaphoresAsync");
    }

    void Wait(void* stream, uint64_t value) override {
        Enter();
        hip::hipExternalSemaphoreWaitParams params{};
        params.params.fence.value = value;
        m_hip.Check(m_hip.hipWaitExternalSemaphoresAsync(&m_timeline, &params, 1,
                                                         hip::Traits::ApplicationStream(stream)),
                    "hipWaitExternalSemaphoresAsync");
    }

  private:
    void Enter() const {
        m_hip.Check(m_hip.hipSetDevice(m_device), "hipSetDevice");
    }

    hip::hipExternalSemaphore_t Import(VulkanDevice& device, VkSemaphore semaphore) {
        const int fd = ExportFd(device, semaphore);
        hip::hipExternalSemaphoreHandleDesc desc{};
        desc.type = hip::hipExternalSemaphoreHandleTypeTimelineSemaphoreFd;
        desc.handle.fd = fd;
        hip::hipExternalSemaphore_t imported = nullptr;
        const hip::hipError_t result = m_hip.hipImportExternalSemaphore(&imported, &desc);
        if (result != 0) {
            CloseFd(fd);
        }
        m_hip.Check(result, "hipImportExternalSemaphore");
        return imported;
    }

    const hip::Api& m_hip;
    int m_device = 0;
    hip::hipStream_t m_stream = nullptr;
    hip::hipExternalSemaphore_t m_timeline = nullptr;
    hip::hipExternalSemaphore_t m_follow = nullptr;
};

class HostInterop : public StreamInterop {
  public:
    HostInterop(VulkanDevice& device, VkSemaphore timeline, VkSemaphore follow)
        : m_device(device), m_timeline(timeline), m_follow(follow) {}

    void Follow(void* stream, uint64_t value) override {
        void* event = Record(stream);
        {
            const std::lock_guard lock(m_mutex);
            m_pending.push_back({event, value});
            if (!m_worker.joinable()) {
                m_worker = std::thread([this] { Run(); });
            }
        }
        m_wake.notify_one();
    }

    void Wait(void* stream, uint64_t value) override {
        auto* waiting = new Waiting{this, value};
        {
            const std::lock_guard lock(m_mutex);
            ++m_waiting;
        }
        try {
            Launch(stream, WaitTimeline, waiting);
        } catch (...) {
            delete waiting;
            Done();
            throw;
        }
    }

  protected:
    virtual void* Record(void* stream) = 0;
    virtual void Synchronize(void* event) = 0;
    virtual void Launch(void* stream, void (*function)(void*), void* data) = 0;

    void Stop() {
        {
            std::unique_lock lock(m_mutex);
            m_stopping = true;
            m_idle.wait(lock, [this] { return m_waiting == 0; });
        }
        m_wake.notify_one();
        if (m_worker.joinable()) {
            m_worker.join();
        }
    }

  private:
    struct Pending {
        void* event = nullptr;
        uint64_t value = 0;
    };

    struct Waiting {
        HostInterop* interop = nullptr;
        uint64_t value = 0;
    };

    static void WaitTimeline(void* data) {
        const std::unique_ptr<Waiting> waiting(static_cast<Waiting*>(data));
        HostInterop& interop = *waiting->interop;
        VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        wait.semaphoreCount = 1;
        wait.pSemaphores = &interop.m_timeline;
        wait.pValues = &waiting->value;
        const VkResult result = interop.m_device.Functions().vkWaitSemaphores(
            interop.m_device.Handle(), &wait, UINT64_MAX);
        if (result != VK_SUCCESS) {
            Log(LogLevel::Warning, "vkWaitSemaphores failed with {} while a stream waited",
                int(result));
        }
        interop.Done();
    }

    void Done() {
        const std::lock_guard lock(m_mutex);
        --m_waiting;
        m_idle.notify_all();
    }

    void Run() {
        std::unique_lock lock(m_mutex);
        while (true) {
            m_wake.wait(lock, [this] { return m_stopping || !m_pending.empty(); });
            if (m_pending.empty()) {
                return;
            }
            const Pending next = m_pending.front();
            m_pending.pop_front();
            lock.unlock();
            try {
                Synchronize(next.event);
            } catch (const Error& error) {
                Log(LogLevel::Warning, "A followed stream failed: {}", error.what());
            }
            VkSemaphoreSignalInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
            signal.semaphore = m_follow;
            signal.value = next.value;
            const VkResult result =
                m_device.Functions().vkSignalSemaphore(m_device.Handle(), &signal);
            if (result != VK_SUCCESS) {
                Log(LogLevel::Warning, "vkSignalSemaphore failed with {}", int(result));
            }
            lock.lock();
        }
    }

    VulkanDevice& m_device;
    VkSemaphore m_timeline;
    VkSemaphore m_follow;
    std::mutex m_mutex;
    std::condition_variable m_wake;
    std::condition_variable m_idle;
    std::deque<Pending> m_pending;
    uint32_t m_waiting = 0;
    bool m_stopping = false;
    std::thread m_worker;
};

class CudaHostInterop : public HostInterop {
  public:
    CudaHostInterop(VulkanDevice& device, VkSemaphore timeline, VkSemaphore follow)
        : HostInterop(device, timeline, follow), m_cuda(cuda::GetApi()) {
        m_device = FindCudaDevice(m_cuda, device).device;
        m_cuda.Check(m_cuda.cuDevicePrimaryCtxRetain(&m_context, m_device),
                     "cuDevicePrimaryCtxRetain");
    }

    ~CudaHostInterop() override {
        Stop();
        m_cuda.Warn(m_cuda.cuDevicePrimaryCtxRelease(m_device), "cuDevicePrimaryCtxRelease");
    }

  protected:
    void* Record(void* stream) override {
        const CudaContext scope(m_cuda, m_context);
        const cuda::CUevent event = cuda::Traits::CreateEvent(m_cuda, false);
        try {
            cuda::Traits::Record(m_cuda, event, cuda::Traits::ApplicationStream(stream), false);
        } catch (...) {
            cuda::Traits::DestroyEvent(m_cuda, event);
            throw;
        }
        return event;
    }

    void Synchronize(void* event) override {
        const CudaContext scope(m_cuda, m_context);
        const auto recorded = static_cast<cuda::CUevent>(event);
        cuda::Traits::Synchronize(m_cuda, recorded);
        cuda::Traits::DestroyEvent(m_cuda, recorded);
    }

    void Launch(void* stream, void (*function)(void*), void* data) override {
        const CudaContext scope(m_cuda, m_context);
        m_cuda.Check(
            m_cuda.cuLaunchHostFunc(cuda::Traits::ApplicationStream(stream), function, data),
            "cuLaunchHostFunc");
    }

  private:
    const cuda::Api& m_cuda;
    cuda::CUdevice m_device = 0;
    cuda::CUcontext m_context = nullptr;
};

class HipHostInterop : public HostInterop {
  public:
    HipHostInterop(VulkanDevice& device, VkSemaphore timeline, VkSemaphore follow)
        : HostInterop(device, timeline, follow), m_hip(hip::GetApi()),
          m_device(FindHipDevice(m_hip, device)) {}

    ~HipHostInterop() override {
        Stop();
    }

  protected:
    void* Record(void* stream) override {
        Enter();
        const hip::hipEvent_t event = hip::Traits::CreateEvent(m_hip, false);
        try {
            hip::Traits::Record(m_hip, event, hip::Traits::ApplicationStream(stream), false);
        } catch (...) {
            hip::Traits::DestroyEvent(m_hip, event);
            throw;
        }
        return event;
    }

    void Synchronize(void* event) override {
        Enter();
        const auto recorded = static_cast<hip::hipEvent_t>(event);
        hip::Traits::Synchronize(m_hip, recorded);
        hip::Traits::DestroyEvent(m_hip, recorded);
    }

    void Launch(void* stream, void (*function)(void*), void* data) override {
        Enter();
        m_hip.Check(m_hip.hipLaunchHostFunc(hip::Traits::ApplicationStream(stream), function, data),
                    "hipLaunchHostFunc");
    }

  private:
    void Enter() const {
        m_hip.Check(m_hip.hipSetDevice(m_device), "hipSetDevice");
    }

    const hip::Api& m_hip;
    int m_device = 0;
};

class CudaMemoryInterop : public MemoryInterop {
  public:
    explicit CudaMemoryInterop(VulkanDevice& device) : m_cuda(cuda::GetApi()) {
        const CudaDevice found = FindCudaDevice(m_cuda, device);
        m_device = found.device;
        m_ordinal = found.ordinal;
        m_cuda.Check(m_cuda.cuDevicePrimaryCtxRetain(&m_context, m_device),
                     "cuDevicePrimaryCtxRetain");
        const CudaContext scope(m_cuda, m_context);
        m_stream = cuda::Traits::CreateStream(m_cuda);
    }

    ~CudaMemoryInterop() override {
        {
            const CudaContext scope(m_cuda, m_context);
            if (m_stream) {
                cuda::Traits::DestroyStream(m_cuda, m_stream);
            }
        }
        m_cuda.Warn(m_cuda.cuDevicePrimaryCtxRelease(m_device), "cuDevicePrimaryCtxRelease");
    }

    DLDevice Device() const override {
        return {kDLCUDA, m_ordinal};
    }

    ImportedMemory Import(const ExportedMemory& memory) override {
        const CudaContext scope(m_cuda, m_context);
        cuda::CUDA_EXTERNAL_MEMORY_HANDLE_DESC desc{};
        desc.type = cuda::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
        desc.handle.fd = memory.fd;
        desc.size = memory.memoryBytes;
        desc.flags = cuda::CUDA_EXTERNAL_MEMORY_DEDICATED;
        cuda::CUexternalMemory external = nullptr;
        const cuda::CUresult imported = m_cuda.cuImportExternalMemory(&external, &desc);
        if (imported != cuda::CUDA_SUCCESS) {
            CloseFd(memory.fd);
        }
        m_cuda.Check(imported, "cuImportExternalMemory");

        cuda::CUDA_EXTERNAL_MEMORY_BUFFER_DESC buffer{};
        buffer.offset = memory.offset;
        buffer.size = memory.bytes;
        cuda::CUdeviceptr address = 0;
        const cuda::CUresult mapped =
            m_cuda.cuExternalMemoryGetMappedBuffer(&address, external, &buffer);
        if (mapped != cuda::CUDA_SUCCESS) {
            m_cuda.Warn(m_cuda.cuDestroyExternalMemory(external), "cuDestroyExternalMemory");
        }
        m_cuda.Check(mapped, "cuExternalMemoryGetMappedBuffer");
        return {external, address};
    }

    void Release(ImportedMemory& memory) noexcept override {
        if (memory.external == nullptr) {
            return;
        }
        try {
            const CudaContext scope(m_cuda, m_context);
            m_cuda.Warn(m_cuda.cuMemFree(memory.address), "cuMemFree");
            m_cuda.Warn(m_cuda.cuDestroyExternalMemory(
                            static_cast<cuda::CUexternalMemory>(memory.external)),
                        "cuDestroyExternalMemory");
        } catch (...) {
        }
        memory = {};
    }

    void* Stream() const override {
        return m_stream;
    }

    void Copy(uint64_t dst, uint64_t src, uint64_t bytes, void* stream) override {
        const CudaContext scope(m_cuda, m_context);
        const cuda::CUstream on = stream ? cuda::Traits::ApplicationStream(stream) : m_stream;
        m_cuda.Check(m_cuda.cuMemcpyAsync(dst, src, bytes, on), "cuMemcpyAsync");
        if (stream == nullptr) {
            m_cuda.Check(m_cuda.cuStreamSynchronize(m_stream), "cuStreamSynchronize");
        }
    }

  private:
    const cuda::Api& m_cuda;
    cuda::CUdevice m_device = 0;
    int32_t m_ordinal = 0;
    cuda::CUcontext m_context = nullptr;
    cuda::CUstream m_stream = nullptr;
};

class HipMemoryInterop : public MemoryInterop {
  public:
    explicit HipMemoryInterop(VulkanDevice& device) : m_hip(hip::GetApi()) {
        m_device = FindHipDevice(m_hip, device);
        Enter();
        m_stream = hip::Traits::CreateStream(m_hip);
    }

    ~HipMemoryInterop() override {
        m_hip.Warn(m_hip.hipSetDevice(m_device), "hipSetDevice");
        if (m_stream) {
            hip::Traits::DestroyStream(m_hip, m_stream);
        }
    }

    DLDevice Device() const override {
        return {kDLROCM, m_device};
    }

    ImportedMemory Import(const ExportedMemory& memory) override {
        Enter();
        hip::hipExternalMemoryHandleDesc desc{};
        desc.type = hip::hipExternalMemoryHandleTypeOpaqueFd;
        desc.handle.fd = memory.fd;
        desc.size = memory.memoryBytes;
        desc.flags = hip::hipExternalMemoryDedicated;
        hip::hipExternalMemory_t external = nullptr;
        const hip::hipError_t imported = m_hip.hipImportExternalMemory(&external, &desc);
        if (imported != hip::hipSuccess) {
            CloseFd(memory.fd);
        }
        m_hip.Check(imported, "hipImportExternalMemory");

        hip::hipExternalMemoryBufferDesc buffer{};
        buffer.offset = memory.offset;
        buffer.size = memory.bytes;
        void* address = nullptr;
        const hip::hipError_t mapped =
            m_hip.hipExternalMemoryGetMappedBuffer(&address, external, &buffer);
        if (mapped != hip::hipSuccess) {
            m_hip.Warn(m_hip.hipDestroyExternalMemory(external), "hipDestroyExternalMemory");
        }
        m_hip.Check(mapped, "hipExternalMemoryGetMappedBuffer");
        return {external, uint64_t(address)};
    }

    void Release(ImportedMemory& memory) noexcept override {
        if (memory.external == nullptr) {
            return;
        }
        try {
            Enter();
            m_hip.Warn(m_hip.hipFree(reinterpret_cast<void*>(memory.address)), "hipFree");
            m_hip.Warn(m_hip.hipDestroyExternalMemory(memory.external), "hipDestroyExternalMemory");
        } catch (...) {
        }
        memory = {};
    }

    void* Stream() const override {
        return m_stream;
    }

    void Copy(uint64_t dst, uint64_t src, uint64_t bytes, void* stream) override {
        Enter();
        const hip::hipStream_t on = stream ? hip::Traits::ApplicationStream(stream) : m_stream;
        m_hip.Check(m_hip.hipMemcpyAsync(reinterpret_cast<void*>(dst),
                                         reinterpret_cast<const void*>(src), bytes,
                                         hip::hipMemcpyDefault, on),
                    "hipMemcpyAsync");
        if (stream == nullptr) {
            m_hip.Check(m_hip.hipStreamSynchronize(m_stream), "hipStreamSynchronize");
        }
    }

  private:
    void Enter() const {
        m_hip.Check(m_hip.hipSetDevice(m_device), "hipSetDevice");
    }

    const hip::Api& m_hip;
    int m_device = 0;
    hip::hipStream_t m_stream = nullptr;
};

}

std::unique_ptr<StreamInterop> CreateStreamInterop(VulkanDevice& device, VkSemaphore timeline,
                                                   VkSemaphore follow) {
    if (device.VendorId() == kVendorNvidia && cuda::Available()) {
        if (device.SharesSemaphores()) {
            return std::make_unique<CudaInterop>(device, timeline, follow);
        }
        return std::make_unique<CudaHostInterop>(device, timeline, follow);
    }
    if (device.VendorId() == kVendorAmd && hip::Available()) {
        if (device.SharesSemaphores()) {
            return std::make_unique<HipInterop>(device, timeline, follow);
        }
        return std::make_unique<HipHostInterop>(device, timeline, follow);
    }
    throw UnsupportedError("{} shares no streams with CUDA or HIP: its GPU is of neither here",
                           device.Adapter().alias);
}

std::unique_ptr<MemoryInterop> CreateMemoryInterop(VulkanDevice& device) {
    if (!device.SharesExportedMemory()) {
        throw UnsupportedError("{} exports no memory", device.Adapter().alias);
    }
    if (device.VendorId() == kVendorNvidia && cuda::Available()) {
        return std::make_unique<CudaMemoryInterop>(device);
    }
    if (device.VendorId() == kVendorAmd && hip::Available()) {
        return std::make_unique<HipMemoryInterop>(device);
    }
    throw UnsupportedError("{} shares no memory with CUDA or HIP: its GPU is of neither here",
                           device.Adapter().alias);
}

}
