// SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "ernie/device/compute/compute.h"
#include "ernie/device/cuda/device.h"
#include "ernie/device/queue.h"

extern "C" {
extern const unsigned char ernie_cuda_prelude[];
extern const std::size_t ernie_cuda_prelude_size;
}

namespace ernie::device::cuda {

ContextScope::ContextScope(const Api& cuda, CUcontext context) : m_cuda(cuda) {
    cuda.Check(cuda.cuCtxPushCurrent(context), "cuCtxPushCurrent");
}

ContextScope::~ContextScope() {
    CUcontext previous = nullptr;
    m_cuda.Warn(m_cuda.cuCtxPopCurrent(&previous), "cuCtxPopCurrent");
}

CudaDevice::CudaDevice(const AdapterInfo& adapter, const DeviceDesc& desc, Traversal traversal)
    : Device(adapter, desc, traversal), m_cuda(GetApi()) {
    m_cuda.Check(m_cuda.cuDeviceGet(&m_device, int(adapter.ordinal)), "cuDeviceGet");
    m_cuda.Check(m_cuda.cuDevicePrimaryCtxRetain(&m_context, m_device), "cuDevicePrimaryCtxRetain");
    try {
        m_queue = std::make_unique<compute::ComputeQueue<Traits>>(*this);
    } catch (...) {
        m_cuda.Warn(m_cuda.cuDevicePrimaryCtxRelease(m_device), "cuDevicePrimaryCtxRelease");
        throw;
    }
}

CudaDevice::~CudaDevice() {
    m_modules.Clear();
    m_cuda.Warn(m_cuda.cuDevicePrimaryCtxRelease(m_device), "cuDevicePrimaryCtxRelease");
}

Ref<Buffer> CudaDevice::NewBuffer(const BufferDesc& desc) {
    return compute::ComputeBuffer<Traits>::Create(*this, desc);
}

Ref<Texture> CudaDevice::NewTexture(const TextureDesc& desc) {
    return Ref<Texture>(new compute::ComputeTexture<Traits>(*this, desc));
}

void CudaDevice::ExternalCopy(BufferRange dst, uint64_t source, Stream producer) {
    compute::ExternalCopy<Traits>(*this, dst, source, producer);
}

std::unique_ptr<Kernel> CudaDevice::NewKernel() {
    return std::make_unique<compute::ComputeKernel<Traits>>(*this);
}

std::vector<MemoryHeap> CudaDevice::Heaps() const {
    const ContextScope scope = Traits::Enter(*this);
    size_t free = 0;
    size_t total = 0;
    if (Functions().cuMemGetInfo(&free, &total) != 0) {
        return {};
    }
    return {{.usage = total - free, .budget = total, .size = total, .deviceLocal = true}};
}

ContextScope Traits::Enter(const CudaDevice& device) {
    return ContextScope(device.Functions(), device.Context());
}

uint64_t Traits::Allocate(const Api& cuda, uint64_t bytes) {
    CUdeviceptr pointer = 0;
    cuda.Check(cuda.cuMemAlloc(&pointer, bytes), "cuMemAlloc");
    return pointer;
}

void Traits::Free(const Api& cuda, uint64_t address) {
    cuda.Warn(cuda.cuMemFree(address), "cuMemFree");
}

CUmemoryPool Traits::CreatePool(const CudaDevice& device, uint64_t keptBytes) {
    const Api& cuda = device.Functions();
    int pools = 0;
    cuda.Check(cuda.cuDeviceGetAttribute(&pools, CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
                                         device.Handle()),
               "cuDeviceGetAttribute");
    if (pools == 0) {
        return nullptr;
    }

    CUmemPoolProps properties{};
    properties.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    properties.location = {CU_MEM_LOCATION_TYPE_DEVICE, device.Handle()};
    CUmemoryPool pool = nullptr;
    cuda.Check(cuda.cuMemPoolCreate(&pool, &properties), "cuMemPoolCreate");

    const CUresult set =
        cuda.cuMemPoolSetAttribute(pool, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &keptBytes);
    if (set != CUDA_SUCCESS) {
        DestroyPool(cuda, pool);
        cuda.Check(set, "cuMemPoolSetAttribute");
    }
    return pool;
}

void Traits::DestroyPool(const Api& cuda, CUmemoryPool pool) {
    cuda.Warn(cuda.cuMemPoolDestroy(pool), "cuMemPoolDestroy");
}

uint64_t Traits::AllocateFrom(const Api& cuda, CUmemoryPool pool, uint64_t bytes, CUstream stream) {
    CUdeviceptr pointer = 0;
    cuda.Check(cuda.cuMemAllocFromPoolAsync(&pointer, bytes, pool, stream),
               "cuMemAllocFromPoolAsync");
    return pointer;
}

void Traits::FreeTo(const Api& cuda, uint64_t address, CUstream stream) {
    cuda.Warn(cuda.cuMemFreeAsync(address, stream), "cuMemFreeAsync");
}

void Traits::SynchronizeStream(const Api& cuda, CUstream stream) {
    cuda.Check(cuda.cuStreamSynchronize(stream), "cuStreamSynchronize");
}

void* Traits::AllocateHost(const Api& cuda, uint64_t bytes, bool writeCombined) {
    void* host = nullptr;
    const unsigned flags =
        CU_MEMHOSTALLOC_DEVICEMAP | (writeCombined ? CU_MEMHOSTALLOC_WRITECOMBINED : 0);
    cuda.Check(cuda.cuMemHostAlloc(&host, bytes, flags), "cuMemHostAlloc");
    return host;
}

uint64_t Traits::DeviceAddress(const Api& cuda, void* host) {
    CUdeviceptr pointer = 0;
    cuda.Check(cuda.cuMemHostGetDevicePointer(&pointer, host, 0), "cuMemHostGetDevicePointer");
    return pointer;
}

void Traits::FreeHost(const Api& cuda, void* host) {
    cuda.Warn(cuda.cuMemFreeHost(host), "cuMemFreeHost");
}

CUstream Traits::CreateStream(const Api& cuda) {
    CUstream stream = nullptr;
    cuda.Check(cuda.cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
    return stream;
}

void Traits::DestroyStream(const Api& cuda, CUstream stream) {
    cuda.Warn(cuda.cuStreamDestroy(stream), "cuStreamDestroy");
}

CUevent Traits::CreateEvent(const Api& cuda, bool timed) {
    CUevent event = nullptr;
    cuda.Check(cuda.cuEventCreate(&event, timed ? 0 : CU_EVENT_DISABLE_TIMING), "cuEventCreate");
    return event;
}

void Traits::DestroyEvent(const Api& cuda, CUevent event) {
    cuda.Warn(cuda.cuEventDestroy(event), "cuEventDestroy");
}

void Traits::Record(const Api& cuda, CUevent event, CUstream stream, bool node) {
    cuda.Check(cuda.cuEventRecordWithFlags(event, stream, node ? CU_EVENT_RECORD_EXTERNAL : 0),
               "cuEventRecordWithFlags");
}

void Traits::WaitEvent(const Api& cuda, CUstream stream, CUevent event) {
    cuda.Check(cuda.cuStreamWaitEvent(stream, event, 0), "cuStreamWaitEvent");
}

CUstream Traits::ApplicationStream(void* handle) {
    return static_cast<CUstream>(handle);
}

uint64_t Traits::ElapsedNs(const Api& cuda, CUevent from, CUevent to) {
    float milliseconds = 0;
    cuda.Check(cuda.cuEventElapsedTime(&milliseconds, from, to), "cuEventElapsedTime");
    return uint64_t(double(milliseconds) * 1e6);
}

bool Traits::Query(const Api& cuda, CUevent event) {
    const CUresult result = cuda.cuEventQuery(event);
    if (result == CUDA_ERROR_NOT_READY) {
        return false;
    }
    cuda.Check(result, "cuEventQuery");
    return true;
}

void Traits::Synchronize(const Api& cuda, CUevent event) {
    cuda.Check(cuda.cuEventSynchronize(event), "cuEventSynchronize");
}

void Traits::Copy(const Api& cuda, uint64_t dst, uint64_t src, uint64_t bytes, CUstream stream) {
    cuda.Check(cuda.cuMemcpyAsync(dst, src, bytes, stream), "cuMemcpyAsync");
}

void Traits::Fill(const Api& cuda, uint64_t dst, uint32_t value, uint64_t bytes, CUstream stream) {
    cuda.Check(cuda.cuMemsetD32Async(dst, value, bytes / 4, stream), "cuMemsetD32Async");
}

void Traits::Upload(const Api& cuda, uint64_t dst, const void* host, uint64_t bytes,
                    CUstream stream) {
    cuda.Check(cuda.cuMemcpyHtoDAsync(dst, host, bytes, stream), "cuMemcpyHtoDAsync");
}

void* Traits::CreateTextureArray(const Api& cuda, const ArrayDesc& desc, uint32_t levels) {
    void* array = nullptr;
    cuda.Check(cuda.cuMipmappedArrayCreate(&array, &desc, levels), "cuMipmappedArrayCreate");
    return array;
}

void Traits::DestroyTextureArray(const Api& cuda, void* array) {
    cuda.Warn(cuda.cuMipmappedArrayDestroy(array), "cuMipmappedArrayDestroy");
}

void* Traits::TextureLevel(const Api& cuda, void* array, uint32_t level) {
    void* found = nullptr;
    cuda.Check(cuda.cuMipmappedArrayGetLevel(&found, array, level), "cuMipmappedArrayGetLevel");
    return found;
}

uint64_t Traits::CreateTextureObject(const Api& cuda, const ResourceDesc& resource,
                                     const SamplerDesc& sampler) {
    unsigned long long object = 0;
    cuda.Check(cuda.cuTexObjectCreate(&object, &resource, &sampler, nullptr), "cuTexObjectCreate");
    return object;
}

void Traits::DestroyTextureObject(const Api& cuda, uint64_t object) {
    cuda.Warn(cuda.cuTexObjectDestroy(object), "cuTexObjectDestroy");
}

void Traits::CopyToArray(const Api& cuda, void* level, uint64_t source, uint64_t rowBytes,
                         uint32_t rows, CUstream stream) {
    const CUDA_MEMCPY2D copy{.srcMemoryType = CU_MEMORYTYPE_DEVICE,
                             .srcDevice = source,
                             .srcPitch = rowBytes,
                             .dstMemoryType = CU_MEMORYTYPE_ARRAY,
                             .dstArray = level,
                             .WidthInBytes = rowBytes,
                             .Height = rows};
    cuda.Check(cuda.cuMemcpy2DAsync(&copy, stream), "cuMemcpy2DAsync");
}

void Traits::BeginCapture(const Api& cuda, CUstream stream) {
    cuda.Check(cuda.cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL),
               "cuStreamBeginCapture");
}

CUgraphExec Traits::EndCapture(const Api& cuda, CUstream stream) {
    CUgraph graph = nullptr;
    cuda.Check(cuda.cuStreamEndCapture(stream, &graph), "cuStreamEndCapture");
    CUgraphExec exec = nullptr;
    const auto instantiated = cuda.cuGraphInstantiateWithFlags(&exec, graph, 0);
    cuda.Warn(cuda.cuGraphDestroy(graph), "cuGraphDestroy");
    cuda.Check(instantiated, "cuGraphInstantiateWithFlags");
    return exec;
}

void Traits::LaunchGraph(const Api& cuda, CUgraphExec graph, CUstream stream) {
    cuda.Check(cuda.cuGraphLaunch(graph, stream), "cuGraphLaunch");
}

void Traits::DestroyGraph(const Api& cuda, CUgraphExec graph) {
    cuda.Warn(cuda.cuGraphExecDestroy(graph), "cuGraphExecDestroy");
}

const compute::RuntimeCompiler& Traits::RuntimeCompiler() {
    return GetNvrtc();
}

std::string_view Traits::Prelude() {
    return {reinterpret_cast<const char*>(ernie_cuda_prelude), ernie_cuda_prelude_size};
}

std::vector<std::string> Traits::CompileOptions(const std::string& architecture, Size3,
                                                uint32_t maxRegisters, bool lineInfo) {
    std::vector<std::string> options = {
        "--gpu-architecture=" + architecture, "--std=c++17",     "-default-device",
        "-DSLANG_CUDA_ENABLE_FAST_MATH=1",    "-prec-div=false", "-prec-sqrt=false"};
    if (maxRegisters > 0) {
        options.push_back("-maxrregcount=" + std::to_string(maxRegisters));
    }
    if (lineInfo) {
        options.emplace_back("-lineinfo");
    }
    return options;
}

CUmodule Traits::LoadModule(const Api& cuda, const std::string& binary) {
    CUmodule module = nullptr;
    cuda.Check(cuda.cuModuleLoadData(&module, binary.data()), "cuModuleLoadData");
    return module;
}

void Traits::UnloadModule(const Api& cuda, CUmodule module) {
    cuda.Warn(cuda.cuModuleUnload(module), "cuModuleUnload");
}

std::pair<uint64_t, size_t> Traits::Symbol(const Api& cuda, CUmodule module, const char* name) {
    CUdeviceptr pointer = 0;
    size_t bytes = 0;
    const CUresult found = cuda.cuModuleGetGlobal(&pointer, &bytes, module, name);
    if (found == CUDA_ERROR_NOT_FOUND) {
        return {0, 0};
    }
    cuda.Check(found, "cuModuleGetGlobal");
    return {pointer, bytes};
}

void Traits::CopyToDevice(const Api& cuda, uint64_t dst, const void* host, uint64_t bytes) {
    cuda.Check(cuda.cuMemcpyHtoD(dst, host, bytes), "cuMemcpyHtoD");
}

CUfunction Traits::GetFunction(const Api& cuda, CUmodule module, const char* name) {
    CUfunction function = nullptr;
    cuda.Check(cuda.cuModuleGetFunction(&function, module, name), "cuModuleGetFunction");
    return function;
}

KernelStats Traits::Stats(const Api& cuda, CUfunction function) {
    const auto attribute = [&](int which) {
        int value = 0;
        cuda.Check(cuda.cuFuncGetAttribute(&value, which, function), "cuFuncGetAttribute");
        return uint32_t(value);
    };
    return {.registers = attribute(CU_FUNC_ATTRIBUTE_NUM_REGS),
            .spillBytes = attribute(CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES),
            .sharedBytes = attribute(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)};
}

void Traits::Launch(const Api& cuda, CUfunction function, Size3 groups, Size3 threads, void** extra,
                    CUstream stream) {
    cuda.Check(cuda.cuLaunchKernel(function, groups.x, groups.y, groups.z, threads.x, threads.y,
                                   threads.z, 0, stream, nullptr, extra),
               "cuLaunchKernel");
}

}
