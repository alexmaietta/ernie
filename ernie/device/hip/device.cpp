// SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "ernie/device/compute/compute.h"
#include "ernie/device/hip/device.h"
#include "ernie/device/queue.h"

extern "C" {
extern const unsigned char ernie_hip_prelude[];
extern const std::size_t ernie_hip_prelude_size;
}

namespace ernie::device::hip {

DeviceScope::DeviceScope(const Api& hip, hipDevice_t device) : m_hip(hip) {
    hip.Check(hip.hipGetDevice(&m_previous), "hipGetDevice");
    hip.Check(hip.hipSetDevice(device), "hipSetDevice");
}

DeviceScope::~DeviceScope() {
    m_hip.Warn(m_hip.hipSetDevice(m_previous), "hipSetDevice");
}

HipDevice::HipDevice(const AdapterInfo& adapter, const DeviceDesc& desc, Traversal traversal)
    : Device(adapter, desc, traversal), m_hip(GetApi()), m_device(int(adapter.ordinal)) {
    m_queue = std::make_unique<compute::ComputeQueue<Traits>>(*this);
}

Ref<Buffer> HipDevice::NewBuffer(const BufferDesc& desc) {
    return compute::ComputeBuffer<Traits>::Create(*this, desc);
}

Ref<Texture> HipDevice::NewTexture(const TextureDesc& desc) {
    return Ref<Texture>(new compute::ComputeTexture<Traits>(*this, desc));
}

void HipDevice::ExternalCopy(BufferRange dst, uint64_t source, Stream producer) {
    compute::ExternalCopy<Traits>(*this, dst, source, producer);
}

std::unique_ptr<Kernel> HipDevice::NewKernel() {
    return std::make_unique<compute::ComputeKernel<Traits>>(*this);
}

std::vector<MemoryHeap> HipDevice::Heaps() const {
    const DeviceScope scope = Traits::Enter(*this);
    size_t free = 0;
    size_t total = 0;
    if (Functions().hipMemGetInfo(&free, &total) != 0) {
        return {};
    }
    return {{.usage = total - free, .budget = total, .size = total, .deviceLocal = true}};
}

DeviceScope Traits::Enter(const HipDevice& device) {
    return DeviceScope(device.Functions(), device.Ordinal());
}

uint64_t Traits::Allocate(const Api& hip, uint64_t bytes) {
    void* pointer = nullptr;
    hip.Check(hip.hipMalloc(&pointer, bytes), "hipMalloc");
    return reinterpret_cast<uint64_t>(pointer);
}

void Traits::Free(const Api& hip, uint64_t address) {
    hip.Warn(hip.hipFree(reinterpret_cast<void*>(address)), "hipFree");
}

hipMemPool_t Traits::CreatePool(const HipDevice& device, uint64_t keptBytes) {
    const Api& hip = device.Functions();
    int pools = 0;
    hip.Check(
        hip.hipDeviceGetAttribute(&pools, hipDeviceAttributeMemoryPoolsSupported, device.Ordinal()),
        "hipDeviceGetAttribute");
    if (pools == 0) {
        return nullptr;
    }

    hipMemPoolProps properties{};
    properties.allocType = hipMemAllocationTypePinned;
    properties.location = {hipMemLocationTypeDevice, device.Ordinal()};
    hipMemPool_t pool = nullptr;
    hip.Check(hip.hipMemPoolCreate(&pool, &properties), "hipMemPoolCreate");

    const hipError_t set =
        hip.hipMemPoolSetAttribute(pool, hipMemPoolAttrReleaseThreshold, &keptBytes);
    if (set != hipSuccess) {
        DestroyPool(hip, pool);
        hip.Check(set, "hipMemPoolSetAttribute");
    }
    return pool;
}

void Traits::DestroyPool(const Api& hip, hipMemPool_t pool) {
    hip.Warn(hip.hipMemPoolDestroy(pool), "hipMemPoolDestroy");
}

uint64_t Traits::AllocateFrom(const Api& hip, hipMemPool_t pool, uint64_t bytes,
                              hipStream_t stream) {
    void* pointer = nullptr;
    hip.Check(hip.hipMallocFromPoolAsync(&pointer, bytes, pool, stream), "hipMallocFromPoolAsync");
    return reinterpret_cast<uint64_t>(pointer);
}

void Traits::FreeTo(const Api& hip, uint64_t address, hipStream_t stream) {
    hip.Warn(hip.hipFreeAsync(reinterpret_cast<void*>(address), stream), "hipFreeAsync");
}

void Traits::SynchronizeStream(const Api& hip, hipStream_t stream) {
    hip.Check(hip.hipStreamSynchronize(stream), "hipStreamSynchronize");
}

void* Traits::AllocateHost(const Api& hip, uint64_t bytes, bool writeCombined) {
    void* host = nullptr;
    const unsigned flags = hipHostMallocMapped | (writeCombined ? hipHostMallocWriteCombined : 0);
    hip.Check(hip.hipHostMalloc(&host, bytes, flags), "hipHostMalloc");
    return host;
}

uint64_t Traits::DeviceAddress(const Api& hip, void* host) {
    void* pointer = nullptr;
    hip.Check(hip.hipHostGetDevicePointer(&pointer, host, 0), "hipHostGetDevicePointer");
    return reinterpret_cast<uint64_t>(pointer);
}

void Traits::FreeHost(const Api& hip, void* host) {
    hip.Warn(hip.hipHostFree(host), "hipHostFree");
}

hipStream_t Traits::CreateStream(const Api& hip) {
    hipStream_t stream = nullptr;
    hip.Check(hip.hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
              "hipStreamCreateWithFlags");
    return stream;
}

void Traits::DestroyStream(const Api& hip, hipStream_t stream) {
    hip.Warn(hip.hipStreamDestroy(stream), "hipStreamDestroy");
}

hipEvent_t Traits::CreateEvent(const Api& hip, bool timed) {
    hipEvent_t event = nullptr;
    hip.Check(hip.hipEventCreateWithFlags(&event, timed ? 0 : hipEventDisableTiming),
              "hipEventCreateWithFlags");
    return event;
}

void Traits::DestroyEvent(const Api& hip, hipEvent_t event) {
    hip.Warn(hip.hipEventDestroy(event), "hipEventDestroy");
}

void Traits::Record(const Api& hip, hipEvent_t event, hipStream_t stream, bool node) {
    hip.Check(hip.hipEventRecordWithFlags(event, stream, node ? hipEventRecordExternal : 0),
              "hipEventRecordWithFlags");
}

void Traits::WaitEvent(const Api& hip, hipStream_t stream, hipEvent_t event) {
    hip.Check(hip.hipStreamWaitEvent(stream, event, 0), "hipStreamWaitEvent");
}

hipStream_t Traits::ApplicationStream(void* handle) {
    return static_cast<hipStream_t>(handle);
}

uint64_t Traits::ElapsedNs(const Api& hip, hipEvent_t from, hipEvent_t to) {
    float milliseconds = 0;
    hip.Check(hip.hipEventElapsedTime(&milliseconds, from, to), "hipEventElapsedTime");
    return uint64_t(double(milliseconds) * 1e6);
}

bool Traits::Query(const Api& hip, hipEvent_t event) {
    const hipError_t error = hip.hipEventQuery(event);
    if (error == hipErrorNotReady) {
        return false;
    }
    hip.Check(error, "hipEventQuery");
    return true;
}

void Traits::Synchronize(const Api& hip, hipEvent_t event) {
    hip.Check(hip.hipEventSynchronize(event), "hipEventSynchronize");
}

void Traits::Copy(const Api& hip, uint64_t dst, uint64_t src, uint64_t bytes, hipStream_t stream) {
    hip.Check(hip.hipMemcpyAsync(reinterpret_cast<void*>(dst), reinterpret_cast<void*>(src), bytes,
                                 hipMemcpyDefault, stream),
              "hipMemcpyAsync");
}

void Traits::Fill(const Api& hip, uint64_t dst, uint32_t value, uint64_t bytes,
                  hipStream_t stream) {
    hip.Check(hip.hipMemsetD32Async(reinterpret_cast<void*>(dst), int(value), bytes / 4, stream),
              "hipMemsetD32Async");
}

void Traits::Upload(const Api& hip, uint64_t dst, const void* host, uint64_t bytes,
                    hipStream_t stream) {
    hip.Check(hip.hipMemcpyHtoDAsync(reinterpret_cast<void*>(dst), host, bytes, stream),
              "hipMemcpyHtoDAsync");
}

void* Traits::CreateTextureArray(const Api& hip, const ArrayDesc& desc, uint32_t levels) {
    void* array = nullptr;
    hip.Check(hip.hipMipmappedArrayCreate(&array, &desc, levels), "hipMipmappedArrayCreate");
    return array;
}

void Traits::DestroyTextureArray(const Api& hip, void* array) {
    hip.Warn(hip.hipMipmappedArrayDestroy(array), "hipMipmappedArrayDestroy");
}

void* Traits::TextureLevel(const Api& hip, void* array, uint32_t level) {
    void* found = nullptr;
    hip.Check(hip.hipMipmappedArrayGetLevel(&found, array, level), "hipMipmappedArrayGetLevel");
    return found;
}

uint64_t Traits::CreateTextureObject(const Api& hip, const ResourceDesc& resource,
                                     const SamplerDesc& sampler) {
    void* object = nullptr;
    hip.Check(hip.hipTexObjectCreate(&object, &resource, &sampler, nullptr), "hipTexObjectCreate");
    return reinterpret_cast<uint64_t>(object);
}

void Traits::DestroyTextureObject(const Api& hip, uint64_t object) {
    hip.Warn(hip.hipTexObjectDestroy(reinterpret_cast<void*>(object)), "hipTexObjectDestroy");
}

void Traits::CopyToArray(const Api& hip, void* level, uint64_t source, uint64_t rowBytes,
                         uint32_t rows, hipStream_t stream) {
    const hip_Memcpy2D copy{.srcMemoryType = hipMemoryTypeDevice,
                            .srcDevice = reinterpret_cast<void*>(source),
                            .srcPitch = rowBytes,
                            .dstMemoryType = hipMemoryTypeArray,
                            .dstArray = level,
                            .WidthInBytes = rowBytes,
                            .Height = rows};
    hip.Check(hip.hipMemcpyParam2DAsync(&copy, stream), "hipMemcpyParam2DAsync");
}

void Traits::BeginCapture(const Api& hip, hipStream_t stream) {
    hip.Check(hip.hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal),
              "hipStreamBeginCapture");
}

hipGraphExec_t Traits::EndCapture(const Api& hip, hipStream_t stream) {
    hipGraph_t graph = nullptr;
    hip.Check(hip.hipStreamEndCapture(stream, &graph), "hipStreamEndCapture");
    hipGraphExec_t exec = nullptr;
    const auto instantiated = hip.hipGraphInstantiateWithFlags(&exec, graph, 0);
    hip.Warn(hip.hipGraphDestroy(graph), "hipGraphDestroy");
    hip.Check(instantiated, "hipGraphInstantiateWithFlags");
    return exec;
}

void Traits::LaunchGraph(const Api& hip, hipGraphExec_t graph, hipStream_t stream) {
    hip.Check(hip.hipGraphLaunch(graph, stream), "hipGraphLaunch");
}

void Traits::DestroyGraph(const Api& hip, hipGraphExec_t graph) {
    hip.Warn(hip.hipGraphExecDestroy(graph), "hipGraphExecDestroy");
}

const compute::RuntimeCompiler& Traits::RuntimeCompiler() {
    return GetHiprtc();
}

std::string_view Traits::Prelude() {
    return {reinterpret_cast<const char*>(ernie_hip_prelude), ernie_hip_prelude_size};
}

std::vector<std::string> Traits::CompileOptions(const std::string& architecture, Size3 groupSize,
                                                bool lineInfo) {
    const uint32_t threads = groupSize.x * groupSize.y * groupSize.z;
    std::vector<std::string> options = {"--offload-arch=" + architecture, "-O3", "-std=c++17",
                                        "--gpu-max-threads-per-block=" + std::to_string(threads),
                                        "-DSLANG_CUDA_ENABLE_FAST_MATH=1"};
    if (lineInfo) {
        options.emplace_back("-gline-tables-only");
    }
    return options;
}

hipModule_t Traits::LoadModule(const Api& hip, const std::string& binary) {
    hipModule_t module = nullptr;
    hip.Check(hip.hipModuleLoadData(&module, binary.data()), "hipModuleLoadData");
    return module;
}

void Traits::UnloadModule(const Api& hip, hipModule_t module) {
    hip.Warn(hip.hipModuleUnload(module), "hipModuleUnload");
}

std::pair<uint64_t, size_t> Traits::Symbol(const Api& hip, hipModule_t module, const char* name) {
    hipDeviceptr_t pointer = nullptr;
    size_t bytes = 0;
    const hipError_t pending = hip.hipPeekAtLastError();
    const hipError_t found = hip.hipModuleGetGlobal(&pointer, &bytes, module, name);
    if (found == hipErrorNotFound) {
        if (pending == hipSuccess) {
            hip.hipGetLastError();
        }
        return {0, 0};
    }
    hip.Check(found, "hipModuleGetGlobal");
    return {reinterpret_cast<uint64_t>(pointer), bytes};
}

void Traits::CopyToDevice(const Api& hip, uint64_t dst, const void* host, uint64_t bytes) {
    hip.Check(hip.hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(dst), host, bytes),
              "hipMemcpyHtoD");
}

hipFunction_t Traits::GetFunction(const Api& hip, hipModule_t module, const char* name) {
    hipFunction_t function = nullptr;
    hip.Check(hip.hipModuleGetFunction(&function, module, name), "hipModuleGetFunction");
    return function;
}

KernelStats Traits::Stats(const Api& hip, hipFunction_t function) {
    const auto attribute = [&](int which) {
        int value = 0;
        hip.Check(hip.hipFuncGetAttribute(&value, which, function), "hipFuncGetAttribute");
        return uint32_t(value);
    };
    return {.registers = attribute(HIP_FUNC_ATTRIBUTE_NUM_REGS),
            .spillBytes = attribute(HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES),
            .sharedBytes = attribute(HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)};
}

void Traits::Launch(const Api& hip, hipFunction_t function, Size3 groups, Size3 threads,
                    void** extra, hipStream_t stream) {
    hip.Check(hip.hipModuleLaunchKernel(function, groups.x, groups.y, groups.z, threads.x,
                                        threads.y, threads.z, 0, stream, nullptr, extra),
              "hipModuleLaunchKernel");
}

}
