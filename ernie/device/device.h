// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <dlpack/dlpack.h>

#include "ernie/device/adapter.h"
#include "ernie/device/operation.h"
#include "ernie/device/ref.h"
#include "ernie/device/stream.h"
#include "ernie/device/tensor.h"
#include "ernie/export.h"
#include "ernie/util/span.h"

namespace ernie {
struct DeviceGlobals;
}

namespace ernie::device {

struct BuildProgress {
    uint64_t queued = 0;
    uint64_t finished = 0;
    uint64_t cached = 0;
    std::vector<std::string> building;

    bool Done() const {
        return finished >= queued;
    }
};

ERNIE_API BuildProgress CurrentBuilds();

class Accel;
class Buffer;
class BufferRange;
class DebugRecords;
class Kernel;
class Program;
class Queue;
class Resource;
class Texture;
struct AccelBoxes;
struct AccelInstances;
struct AccelSizes;
struct AccelTriangles;
struct BufferDesc;
struct ProgramDesc;
struct TextureDesc;
enum class AccelLevel : uint8_t;

using AccelInput = std::variant<AccelTriangles, AccelBoxes, AccelInstances>;

namespace vulkan {
struct Creator;
}

struct DeviceDesc {
    std::optional<Backend> backend;
    std::string adapter;
    Traversal traversal = Traversal::Auto;
    Sampling sampling = Sampling::Auto;
    bool validation = false;
    bool kernelDebugInfo = false;
    bool profiling = true;
    bool synchronous = false;
    bool presentation = false;
    std::filesystem::path cacheDirectory;
    std::shared_ptr<const vulkan::Creator> creator;
};

struct MemoryUse {
    std::string label;
    uint32_t buffers = 0;
    uint32_t textures = 0;
    uint64_t bytes = 0;
};

struct MemoryHeap {
    uint64_t usage = 0;
    uint64_t budget = 0;
    uint64_t size = 0;
    bool deviceLocal = false;
};

struct MemoryStats {
    std::vector<MemoryUse> uses;
    std::vector<MemoryHeap> heaps;
    uint64_t deviceBytes = 0;
    uint64_t textureBytes = 0;
    uint64_t uploadBytes = 0;
    uint64_t readbackBytes = 0;
    uint64_t sharedBytes = 0;
    uint64_t reservedBytes = 0;
};

struct TrafficStats {
    uint64_t uploadBytes = 0;
    uint64_t readbackBytes = 0;
    uint64_t inlineBytes = 0;
    uint64_t copyBytes = 0;
    uint64_t submissions = 0;
    uint64_t hostWaits = 0;

    TrafficStats& operator+=(const TrafficStats& other) {
        uploadBytes += other.uploadBytes;
        readbackBytes += other.readbackBytes;
        inlineBytes += other.inlineBytes;
        copyBytes += other.copyBytes;
        submissions += other.submissions;
        hostWaits += other.hostWaits;
        return *this;
    }
};

inline TrafficStats operator-(const TrafficStats& a, const TrafficStats& b) {
    TrafficStats result;
    result.uploadBytes = a.uploadBytes - b.uploadBytes;
    result.readbackBytes = a.readbackBytes - b.readbackBytes;
    result.inlineBytes = a.inlineBytes - b.inlineBytes;
    result.copyBytes = a.copyBytes - b.copyBytes;
    result.submissions = a.submissions - b.submissions;
    result.hostWaits = a.hostWaits - b.hostWaits;
    return result;
}

class ERNIE_API Device : public RefCounted {
  public:
    static Ref<Device> Create(const DeviceDesc& desc = {});

    ~Device() override;

    const AdapterInfo& Adapter() const {
        return m_adapter;
    }

    Traversal GetTraversal() const {
        return m_traversal;
    }

    Sampling GetSampling() const {
        return m_sampling;
    }

    bool Presents() const {
        return m_presents;
    }

    const std::filesystem::path& CacheDirectory() const {
        return m_cacheDirectory;
    }

    const std::filesystem::path& KernelCacheDirectory() const {
        return m_kernelCacheDirectory;
    }

    Ref<Buffer> CreateBuffer(const BufferDesc& desc);
    Ref<Buffer> Borrow(DLDevice source, uint64_t address, uint64_t bytes,
                       std::string label = "borrowed");
    Ref<Buffer> CreateBuffer(BufferDesc desc, Span<const std::byte> data);
    template <class T> Ref<Buffer> CreateBuffer(BufferDesc desc, Span<const T> values);
#ifdef __cpp_lib_span
    template <class T> Ref<Buffer> CreateBuffer(BufferDesc desc, std::span<const T> values);
#endif
    template <class T> Ref<Buffer> CreateBuffer(BufferDesc desc, const std::vector<T>& values);

    AccelSizes AccelBuildSizes(const AccelInput& input) const;

    Ref<Accel> CreateAccel(const AccelInput& input, std::string label = "accel");

    Ref<Texture> CreateTexture(const TextureDesc& desc);

    Tensor CreateTensor(DType type, std::vector<int64_t> shape, std::string label = "tensor");

    Operation Copy(const Tensor& src, const Tensor& dst, const Operation& after = {});
    Tensor Import(const DLTensor& source, Stream producer = Stream::None(),
                  std::string label = "imported");
    Operation Follow(Stream producer);

    Ref<Program> CreateProgram(const ProgramDesc& desc);

    MemoryStats MemoryUsage() const;

    TrafficStats Traffic() const;

    Queue& GetQueue() {
        return *m_queue;
    }

    void Require(const Resource& resource, const char* what) const;

    virtual std::optional<DLDevice> SharedDevice();

    bool SharesMemory(DLDeviceType api) {
        const std::optional<DLDevice> shared = SharedDevice();
        return shared && shared->device_type == api;
    }

    bool ReadsDirectly(DLDevice source) const;
    void CopyIn(BufferRange dst, uint64_t source, Stream producer);
    Ref<Buffer> Stage(uint64_t source, uint64_t bytes, Stream producer, const std::string& label);

  protected:
    Device(AdapterInfo adapter, const DeviceDesc& desc, Traversal traversal);

    virtual Ref<Buffer> NewBuffer(const BufferDesc& desc) = 0;
    virtual void ExternalCopy(BufferRange dst, uint64_t source, Stream producer);
    virtual Stream CopyStream();

    virtual AccelSizes HardwareBuildSizes(const AccelInput& input) const;

    virtual Ref<Accel> NewAccel(Ref<Buffer> storage, AccelLevel level, AccelSizes sizes,
                                std::string label);

    virtual Ref<Texture> NewTexture(const TextureDesc& desc);

    virtual std::unique_ptr<Kernel> NewKernel() = 0;

    virtual std::vector<MemoryHeap> Heaps() const;

    DebugRecords& Debug() {
        return *m_debug;
    }

    static void CloseQueue(Queue& queue) noexcept;

    DeviceGlobals Globals() const;

    Buffer& GlobalsBuffer() const {
        return *m_globals;
    }

    std::unique_ptr<Queue> m_queue;

  private:
    friend class Buffer;
    friend class Texture;
    friend class CommandEncoder;
    friend class Program;
    friend class Queue;
    friend class Resource;
    struct Counters;
    struct AccelSizeCache;
    struct ProgramCache;

    void CountTraffic(const TrafficStats& traffic);

    void CreateGlobals();

    void LastReleased() noexcept override;

    void Retire(Resource& resource) noexcept;

    static void Adopt(Resource& resource);

    void CountMemory(Buffer& buffer, int sign);
    void CountMemory(Texture& texture, int sign);

    AdapterInfo m_adapter;
    Traversal m_traversal;
    Sampling m_sampling;
    std::filesystem::path m_cacheDirectory;
    std::filesystem::path m_kernelCacheDirectory;
    bool m_profiling;
    bool m_synchronous;
    bool m_kernelDebugInfo;
    bool m_presents;
    std::unique_ptr<DebugRecords> m_debug;
    Ref<Buffer> m_globals;
    std::unique_ptr<Counters> m_counters;
    std::unique_ptr<AccelSizeCache> m_accelSizes;
    std::shared_ptr<ProgramCache> m_programs;
};

}
