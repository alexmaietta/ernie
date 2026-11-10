// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ernie/device/device.h"
#include "ernie/device/hip/api.h"
#include "ernie/device/slang/compiler.h"

namespace ernie::device::hip {

std::vector<AdapterInfo> ListAdapters();

class DeviceScope {
  public:
    DeviceScope(const Api& hip, hipDevice_t device);
    ~DeviceScope();
    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;

  private:
    const Api& m_hip;
    hipDevice_t m_previous = 0;
};

class HipDevice : public Device {
  public:
    HipDevice(const AdapterInfo& adapter, const DeviceDesc& desc, Traversal traversal);

    const Api& Functions() const {
        return m_hip;
    }

    hipDevice_t Ordinal() const {
        return m_device;
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
    const Api& m_hip;
    hipDevice_t m_device;
    compute::ModuleCache m_modules;
};

struct Traits {
    using Device = HipDevice;
    using Api = hip::Api;
    using Stream = hipStream_t;
    using Event = hipEvent_t;
    using Graph = hipGraphExec_t;
    using Pool = hipMemPool_t;
    using Module = hipModule_t;
    using Function = hipFunction_t;
    using ArrayDesc = HIP_ARRAY3D_DESCRIPTOR;
    using ResourceDesc = HIP_RESOURCE_DESC;
    using SamplerDesc = HIP_TEXTURE_DESC;

    static DeviceScope Enter(const HipDevice& device);

    static uint64_t Allocate(const Api& hip, uint64_t bytes);
    static void Free(const Api& hip, uint64_t address);
    static hipMemPool_t CreatePool(const HipDevice& device, uint64_t keptBytes);
    static void DestroyPool(const Api& hip, hipMemPool_t pool);
    static uint64_t AllocateFrom(const Api& hip, hipMemPool_t pool, uint64_t bytes,
                                 hipStream_t stream);
    static void FreeTo(const Api& hip, uint64_t address, hipStream_t stream);
    static void SynchronizeStream(const Api& hip, hipStream_t stream);
    static void* AllocateHost(const Api& hip, uint64_t bytes, bool writeCombined);
    static uint64_t DeviceAddress(const Api& hip, void* host);
    static void FreeHost(const Api& hip, void* host);

    static hipStream_t CreateStream(const Api& hip);
    static void DestroyStream(const Api& hip, hipStream_t stream);
    static hipEvent_t CreateEvent(const Api& hip, bool timed);
    static void DestroyEvent(const Api& hip, hipEvent_t event);
    static void Record(const Api& hip, hipEvent_t event, hipStream_t stream, bool node);
    static void WaitEvent(const Api& hip, hipStream_t stream, hipEvent_t event);
    static hipStream_t ApplicationStream(void* handle);
    static uint64_t ElapsedNs(const Api& hip, hipEvent_t from, hipEvent_t to);
    static bool Query(const Api& hip, hipEvent_t event);
    static void Synchronize(const Api& hip, hipEvent_t event);
    static void Copy(const Api& hip, uint64_t dst, uint64_t src, uint64_t bytes,
                     hipStream_t stream);
    static void Fill(const Api& hip, uint64_t dst, uint32_t value, uint64_t bytes,
                     hipStream_t stream);
    static void Upload(const Api& hip, uint64_t dst, const void* host, uint64_t bytes,
                       hipStream_t stream);
    static void* CreateTextureArray(const Api& hip, const ArrayDesc& desc, uint32_t levels);
    static void DestroyTextureArray(const Api& hip, void* array);
    static void* TextureLevel(const Api& hip, void* array, uint32_t level);
    static uint64_t CreateTextureObject(const Api& hip, const ResourceDesc& resource,
                                        const SamplerDesc& sampler);
    static void DestroyTextureObject(const Api& hip, uint64_t object);
    static void CopyToArray(const Api& hip, void* level, uint64_t source, uint64_t rowBytes,
                            uint32_t rows, hipStream_t stream);
    static void BeginCapture(const Api& hip, hipStream_t stream);
    static hipGraphExec_t EndCapture(const Api& hip, hipStream_t stream);
    static void LaunchGraph(const Api& hip, hipGraphExec_t graph, hipStream_t stream);
    static void DestroyGraph(const Api& hip, hipGraphExec_t graph);

    static const compute::RuntimeCompiler& RuntimeCompiler();
    static std::string_view Prelude();
    static std::vector<std::string> CompileOptions(const std::string& architecture, Size3 groupSize,
                                                   uint32_t maxRegisters, bool lineInfo);
    static hipModule_t LoadModule(const Api& hip, const std::string& binary);
    static void UnloadModule(const Api& hip, hipModule_t module);
    static std::pair<uint64_t, size_t> Symbol(const Api& hip, hipModule_t module, const char* name);
    static void CopyToDevice(const Api& hip, uint64_t dst, const void* host, uint64_t bytes);
    static hipFunction_t GetFunction(const Api& hip, hipModule_t module, const char* name);
    static KernelStats Stats(const Api& hip, hipFunction_t function);
    static void Launch(const Api& hip, hipFunction_t function, Size3 groups, Size3 threads,
                       void** extra, hipStream_t stream);
    static constexpr uintptr_t kParamBuffer = HIP_LAUNCH_PARAM_BUFFER_POINTER;
    static constexpr uintptr_t kParamBufferSize = HIP_LAUNCH_PARAM_BUFFER_SIZE;
    static constexpr uintptr_t kParamEnd = HIP_LAUNCH_PARAM_END;
};

}
