// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ernie/device/cuda/api.h"
#include "ernie/device/device.h"
#include "ernie/device/slang/compiler.h"

namespace ernie::device::cuda {

constexpr Version kMinimumComputeCapability{7, 0};

std::vector<AdapterInfo> ListAdapters();

class ContextScope {
  public:
    ContextScope(const Api& cuda, CUcontext context);
    ~ContextScope();
    ContextScope(const ContextScope&) = delete;
    ContextScope& operator=(const ContextScope&) = delete;

  private:
    const Api& m_cuda;
};

class CudaDevice : public Device {
  public:
    CudaDevice(const AdapterInfo& adapter, const DeviceDesc& desc, Traversal traversal);
    ~CudaDevice() override;

    const Api& Functions() const {
        return m_cuda;
    }

    CUdevice Handle() const {
        return m_device;
    }

    CUcontext Context() const {
        return m_context;
    }

    compute::ModuleCache& Modules() {
        return m_modules;
    }

  protected:
    Ref<Buffer> NewBuffer(const BufferDesc& desc) override;
    Ref<Texture> NewTexture(const TextureDesc& desc) override;
    void ExternalCopy(BufferRange dst, uint64_t source, Stream producer) override;
    std::unique_ptr<Kernel> NewKernel() override;
    std::vector<MemoryHeap> Heaps() const override;

  private:
    const Api& m_cuda;
    CUdevice m_device = 0;
    CUcontext m_context = nullptr;
    compute::ModuleCache m_modules;
};

struct Traits {
    using Device = CudaDevice;
    using Api = cuda::Api;
    using Stream = CUstream;
    using Event = CUevent;
    using Graph = CUgraphExec;
    using Pool = CUmemoryPool;
    using Module = CUmodule;
    using Function = CUfunction;
    using ArrayDesc = CUDA_ARRAY3D_DESCRIPTOR;
    using ResourceDesc = CUDA_RESOURCE_DESC;
    using SamplerDesc = CUDA_TEXTURE_DESC;

    static ContextScope Enter(const CudaDevice& device);

    static uint64_t Allocate(const Api& cuda, uint64_t bytes);
    static void Free(const Api& cuda, uint64_t address);
    static CUmemoryPool CreatePool(const CudaDevice& device, uint64_t keptBytes);
    static void DestroyPool(const Api& cuda, CUmemoryPool pool);
    static uint64_t AllocateFrom(const Api& cuda, CUmemoryPool pool, uint64_t bytes,
                                 CUstream stream);
    static void FreeTo(const Api& cuda, uint64_t address, CUstream stream);
    static void SynchronizeStream(const Api& cuda, CUstream stream);
    static void* AllocateHost(const Api& cuda, uint64_t bytes, bool writeCombined);
    static uint64_t DeviceAddress(const Api& cuda, void* host);
    static void FreeHost(const Api& cuda, void* host);

    static CUstream CreateStream(const Api& cuda);
    static void DestroyStream(const Api& cuda, CUstream stream);
    static CUevent CreateEvent(const Api& cuda, bool timed);
    static void DestroyEvent(const Api& cuda, CUevent event);
    static void Record(const Api& cuda, CUevent event, CUstream stream, bool node);
    static void WaitEvent(const Api& cuda, CUstream stream, CUevent event);
    static CUstream ApplicationStream(void* handle);
    static uint64_t ElapsedNs(const Api& cuda, CUevent from, CUevent to);
    static bool Query(const Api& cuda, CUevent event);
    static void Synchronize(const Api& cuda, CUevent event);
    static void Copy(const Api& cuda, uint64_t dst, uint64_t src, uint64_t bytes, CUstream stream);
    static void Fill(const Api& cuda, uint64_t dst, uint32_t value, uint64_t bytes,
                     CUstream stream);
    static void Upload(const Api& cuda, uint64_t dst, const void* host, uint64_t bytes,
                       CUstream stream);
    static void* CreateTextureArray(const Api& cuda, const ArrayDesc& desc, uint32_t levels);
    static void DestroyTextureArray(const Api& cuda, void* array);
    static void* TextureLevel(const Api& cuda, void* array, uint32_t level);
    static uint64_t CreateTextureObject(const Api& cuda, const ResourceDesc& resource,
                                        const SamplerDesc& sampler);
    static void DestroyTextureObject(const Api& cuda, uint64_t object);
    static void CopyToArray(const Api& cuda, void* level, uint64_t source, uint64_t rowBytes,
                            uint32_t rows, CUstream stream);
    static void BeginCapture(const Api& cuda, CUstream stream);
    static CUgraphExec EndCapture(const Api& cuda, CUstream stream);
    static void LaunchGraph(const Api& cuda, CUgraphExec graph, CUstream stream);
    static void DestroyGraph(const Api& cuda, CUgraphExec graph);

    static const compute::RuntimeCompiler& RuntimeCompiler();
    static std::string_view Prelude();
    static std::vector<std::string> CompileOptions(const std::string& architecture, Size3 groupSize,
                                                   bool lineInfo);
    static CUmodule LoadModule(const Api& cuda, const std::string& binary);
    static void UnloadModule(const Api& cuda, CUmodule module);
    static std::pair<uint64_t, size_t> Symbol(const Api& cuda, CUmodule module, const char* name);
    static void CopyToDevice(const Api& cuda, uint64_t dst, const void* host, uint64_t bytes);
    static CUfunction GetFunction(const Api& cuda, CUmodule module, const char* name);
    static KernelStats Stats(const Api& cuda, CUfunction function);
    static void Launch(const Api& cuda, CUfunction function, Size3 groups, Size3 threads,
                       void** extra, CUstream stream);
    static constexpr uintptr_t kParamBuffer = CU_LAUNCH_PARAM_BUFFER_POINTER;
    static constexpr uintptr_t kParamBufferSize = CU_LAUNCH_PARAM_BUFFER_SIZE;
    static constexpr uintptr_t kParamEnd = CU_LAUNCH_PARAM_END;
};

}
