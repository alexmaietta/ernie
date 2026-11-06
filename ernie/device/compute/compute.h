// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <deque>
#include <format>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "ernie/device/buffer.h"
#include "ernie/device/commands.h"
#include "ernie/device/compute/runtime_compiler.h"
#include "ernie/device/device.h"
#include "ernie/device/queue.h"
#include "ernie/device/slang/compiler.h"
#include "ernie/device/texture.h"
#include "ernie/util/disk_cache.h"
#include "ernie/util/error.h"
#include "ernie/util/profiler.h"

namespace ernie::device::compute {

constexpr uint32_t kMaxParamBytes = 4096;
constexpr unsigned kMaxAnisotropy = 16;

struct ComputeDevice {
    Backend backend;
    uint32_t ordinal;
    std::string name;
    std::array<uint8_t, 16> uuid;
    uint64_t memory;
    bool integrated;
    uint32_t warpSize;
    uint32_t maxThreads;
    bool textures;
};

inline AdapterInfo DescribeCompute(const ComputeDevice& device) {
    AdapterInfo adapter;
    adapter.backend = device.backend;
    adapter.ordinal = device.ordinal;
    adapter.name = device.name;
    adapter.uuid = device.uuid;
    adapter.memoryBytes = device.memory;
    adapter.type = device.integrated ? AdapterType::Integrated : AdapterType::Discrete;

    Features& features = adapter.features;
    features.unifiedMemory = device.integrated;
    features.hostMappedMemory = device.integrated;
    features.externalMemory = true;
    features.kernelStatistics = true;
    features.hardwareTextures = device.textures;
    features.anisotropicFiltering = device.textures;
    features.float64 = true;
    features.atomicFloat32Add = true;
    features.atomicInt64 = true;
    features.shaderClock = true;
    features.waveSizeMin = features.waveSizeMax = device.warpSize;

    adapter.limits.maxBufferBytes = device.memory;
    adapter.limits.maxGroupSize = device.maxThreads;
    adapter.limits.maxParamBytes = kMaxParamBytes;
    adapter.limits.maxAnisotropy = device.textures ? float(kMaxAnisotropy) : 1.0f;
    adapter.limits.maxReservedBytes = adapter.limits.maxBufferBytes;
    return adapter;
}

constexpr uint64_t kPoolKeptBytes = 256ull * 1024 * 1024;

template <class T>
concept ComputeTraits =
    requires(const T::Device& device, const T::Api& api, T::Stream stream, T::Event event,
             T::Graph graph, T::Pool pool, uint64_t address, void* host, const void* source,
             uint64_t bytes, uint32_t value, bool flag, T::Module module, T::Function function,
             const std::string& binary, Size3 groups, const T::ArrayDesc& array,
             const T::ResourceDesc& resource, const T::SamplerDesc& sampler) {
        { device.Functions() } -> std::same_as<const typename T::Api&>;
        T::Enter(device);

        { T::Allocate(api, bytes) } -> std::same_as<uint64_t>;
        { T::Free(api, address) } -> std::same_as<void>;
        { T::CreatePool(device, bytes) } -> std::same_as<typename T::Pool>;
        { T::DestroyPool(api, pool) } -> std::same_as<void>;
        { T::AllocateFrom(api, pool, bytes, stream) } -> std::same_as<uint64_t>;
        { T::FreeTo(api, address, stream) } -> std::same_as<void>;
        { T::SynchronizeStream(api, stream) } -> std::same_as<void>;
        { T::AllocateHost(api, bytes, flag) } -> std::same_as<void*>;
        { T::DeviceAddress(api, host) } -> std::same_as<uint64_t>;
        { T::FreeHost(api, host) } -> std::same_as<void>;

        { T::CreateStream(api) } -> std::same_as<typename T::Stream>;
        { T::DestroyStream(api, stream) } -> std::same_as<void>;
        { T::CreateEvent(api, flag) } -> std::same_as<typename T::Event>;
        { T::DestroyEvent(api, event) } -> std::same_as<void>;
        { T::Record(api, event, stream, flag) } -> std::same_as<void>;
        { T::WaitEvent(api, stream, event) } -> std::same_as<void>;
        { T::ApplicationStream(host) } -> std::same_as<typename T::Stream>;
        { T::ElapsedNs(api, event, event) } -> std::same_as<uint64_t>;
        { T::Query(api, event) } -> std::same_as<bool>;
        { T::Synchronize(api, event) } -> std::same_as<void>;
        { T::Copy(api, address, address, bytes, stream) } -> std::same_as<void>;
        { T::Fill(api, address, value, bytes, stream) } -> std::same_as<void>;
        { T::Upload(api, address, source, bytes, stream) } -> std::same_as<void>;
        { T::CreateTextureArray(api, array, value) } -> std::same_as<void*>;
        { T::DestroyTextureArray(api, host) } -> std::same_as<void>;
        { T::TextureLevel(api, host, value) } -> std::same_as<void*>;
        { T::CreateTextureObject(api, resource, sampler) } -> std::same_as<uint64_t>;
        { T::DestroyTextureObject(api, address) } -> std::same_as<void>;
        { T::CopyToArray(api, host, address, bytes, value, stream) } -> std::same_as<void>;
        T::BeginCapture(api, stream);
        { T::EndCapture(api, stream) } -> std::same_as<typename T::Graph>;
        { T::LaunchGraph(api, graph, stream) } -> std::same_as<void>;
        { T::DestroyGraph(api, graph) } -> std::same_as<void>;

        { T::RuntimeCompiler() } -> std::same_as<const RuntimeCompiler&>;
        { T::Prelude() } -> std::same_as<std::string_view>;
        { T::CompileOptions(binary, groups, flag) } -> std::same_as<std::vector<std::string>>;
        { T::LoadModule(api, binary) } -> std::same_as<typename T::Module>;
        { T::UnloadModule(api, module) } -> std::same_as<void>;
        { T::Symbol(api, module, binary.c_str()) } -> std::same_as<std::pair<uint64_t, size_t>>;
        { T::CopyToDevice(api, address, source, bytes) } -> std::same_as<void>;
        { T::GetFunction(api, module, binary.c_str()) } -> std::same_as<typename T::Function>;
        { T::Stats(api, function) } -> std::same_as<KernelStats>;
        { T::Launch(api, function, groups, groups, &host, stream) } -> std::same_as<void>;
        { T::kParamBuffer } -> std::convertible_to<uintptr_t>;
        { T::kParamBufferSize } -> std::convertible_to<uintptr_t>;
        { T::kParamEnd } -> std::convertible_to<uintptr_t>;
    };

template <ComputeTraits Traits>
void ExternalCopy(const typename Traits::Device& device, BufferRange dst, uint64_t source,
                  Stream producer) {
    const auto scope = Traits::Enter(device);
    const typename Traits::Api& api = device.Functions();
    if (producer.IsCompute()) {
        Traits::Copy(api, dst.Address(), source, dst.Bytes(),
                     Traits::ApplicationStream(producer.Handle()));
        return;
    }
    const typename Traits::Stream stream = Traits::CreateStream(api);
    Traits::Copy(api, dst.Address(), source, dst.Bytes(), stream);
    Traits::SynchronizeStream(api, stream);
    Traits::DestroyStream(api, stream);
}

template <ComputeTraits Traits> class ComputeQueue;

template <ComputeTraits Traits> class ComputeBuffer : public Buffer {
  public:
    using Device = Traits::Device;

    static Ref<Buffer> Create(Device& device, const BufferDesc& desc) {
        const auto scope = Traits::Enter(device);
        Ref<ComputeBuffer> buffer(new ComputeBuffer(device, desc));
        const auto& api = buffer->Functions();
        switch (desc.memory) {
        case Memory::Device:
            buffer->m_queue = &static_cast<ComputeQueue<Traits>&>(device.GetQueue());
            buffer->m_address = buffer->m_queue->Allocate(desc.bytes);
            break;
        default:
            buffer->m_mapped = Traits::AllocateHost(api, desc.bytes, desc.memory == Memory::Upload);
            buffer->m_address = Traits::DeviceAddress(api, buffer->m_mapped);
            break;
        }
        return buffer;
    }

    ~ComputeBuffer() override {
        const auto scope = Traits::Enter(Owner());
        if (m_mapped != nullptr) {
            Traits::FreeHost(Functions(), m_mapped);
        } else if (m_address != 0) {
            m_queue->Free(m_address);
        }
    }

  private:
    ComputeBuffer(Device& device, const BufferDesc& desc) : Buffer(device, desc) {}

    const ComputeQueue<Traits>* m_queue = nullptr;

    const Device& Owner() const {
        return static_cast<const Device&>(GetDevice());
    }

    const Traits::Api& Functions() const {
        return Owner().Functions();
    }
};

struct ArrayFormat {
    int format;
    unsigned channels;
};

inline ArrayFormat TextureArrayFormat(const TextureDesc& desc) {
    constexpr int kUnsigned8 = 0x01;
    constexpr int kUnsigned16 = 0x02;
    constexpr int kHalf = 0x10;
    constexpr int kFloat = 0x20;
    constexpr ArrayFormat kFormats[] = {{kUnsigned8, 1},  {kUnsigned8, 2},  {kUnsigned8, 4},
                                        {kUnsigned16, 1}, {kUnsigned16, 2}, {kUnsigned16, 4},
                                        {kHalf, 1},       {kHalf, 2},       {kHalf, 4},
                                        {kFloat, 1},      {kFloat, 2},      {kFloat, 4}};
    if (TexelBlockSide(desc.format) != 1) {
        throw UnsupportedError(
            "texture \"{}\" is block compressed, which texture objects can't hold", desc.label);
    }
    return kFormats[uint32_t(desc.format)];
}

template <ComputeTraits Traits> class ComputeTexture : public Texture {
  public:
    using Device = Traits::Device;

    ComputeTexture(Device& device, const TextureDesc& desc)
        : Texture(device, desc), m_format(TextureArrayFormat(desc)) {
        const auto scope = Traits::Enter(device);
        m_records = Traits::Allocate(Functions(), sizeof(m_objects));
        try {
            m_array = Traits::CreateTextureArray(Functions(),
                                                 {.Width = desc.width,
                                                  .Height = desc.height,
                                                  .Format = m_format.format,
                                                  .NumChannels = m_format.channels},
                                                 desc.levels);
        } catch (...) {
            Traits::Free(Functions(), m_records);
            throw;
        }
    }

    ~ComputeTexture() override {
        const auto scope = Traits::Enter(Owner());
        for (const TextureObjects& objects : m_objects) {
            if (objects.filtered != 0) {
                Traits::DestroyTextureObject(Functions(), objects.filtered);
                Traits::DestroyTextureObject(Functions(), objects.texels);
            }
        }
        Traits::Free(Functions(), m_records);
        Traits::DestroyTextureArray(Functions(), m_array);
    }

    uint64_t KernelHandle(TextureWrap wrapS, TextureWrap wrapT) override {
        const uint32_t index = TextureHandle(0, wrapS, wrapT);
        const uint64_t address = m_records + uint64_t(index) * sizeof(TextureObjects);
        const std::lock_guard lock(m_mutex);
        TextureObjects& objects = m_objects[index];
        if (objects.filtered == 0) {
            const auto scope = Traits::Enter(Owner());
            const TextureDesc& desc = Desc();
            constexpr int kMipmappedArray = 1;
            typename Traits::ResourceDesc resource{.resType = kMipmappedArray};
            resource.res.mipmap.hMipmappedArray = m_array;
            objects = {.filtered = Traits::CreateTextureObject(Functions(), resource,
                                                               Sampler(wrapS, wrapT, true)),
                       .texels = Traits::CreateTextureObject(Functions(), resource,
                                                             Sampler(wrapS, wrapT, false)),
                       .width = desc.width,
                       .height = desc.height,
                       .numLevels = desc.levels,
                       .channels = m_format.channels};
            Traits::CopyToDevice(Functions(), address, &objects, sizeof(objects));
        }
        return address;
    }

    void Copy(uint32_t level, uint64_t source, Traits::Stream stream) const {
        const TextureDesc& desc = Desc();
        Traits::CopyToArray(Functions(), Traits::TextureLevel(Functions(), m_array, level), source,
                            uint64_t(TextureLevelSide(desc.width, level)) *
                                TexelBlockBytes(desc.format),
                            TextureLevelSide(desc.height, level), stream);
    }

  private:
    ArrayFormat m_format;
    void* m_array = nullptr;
    uint64_t m_records = 0;
    std::array<TextureObjects, kTextureSamplers> m_objects{};
    std::mutex m_mutex;

    const Device& Owner() const {
        return static_cast<const Device&>(GetDevice());
    }

    const Traits::Api& Functions() const {
        return Owner().Functions();
    }

    Traits::SamplerDesc Sampler(TextureWrap wrapS, TextureWrap wrapT, bool filtered) const {
        constexpr int kPoint = 0;
        constexpr int kLinear = 1;
        constexpr unsigned kNormalizedCoordinates = 0x02;
        constexpr unsigned kSrgb = 0x10;
        const auto address = [](TextureWrap wrap) {
            constexpr int kModes[] = {0, 2, 1, 3};
            return kModes[uint32_t(wrap)];
        };
        const TextureDesc& desc = Desc();
        return {.addressMode = {address(wrapS), address(wrapT), address(wrapT)},
                .filterMode = filtered ? kLinear : kPoint,
                .flags = kNormalizedCoordinates |
                         (desc.colorSpace == TexelColorSpace::Srgb ? kSrgb : 0u),
                .maxAnisotropy = filtered ? kMaxAnisotropy : 1,
                .mipmapFilterMode = filtered ? kLinear : kPoint,
                .maxMipmapLevelClamp = float(desc.levels - 1)};
    }
};

template <ComputeTraits Traits> struct LoadedModule {
    Traits::Module module{};
    uint64_t dispatchArgs = 0;
    uint64_t counts = 0;
};

template <ComputeTraits Traits> class ComputeKernel : public Kernel {
  public:
    using Device = Traits::Device;

    explicit ComputeKernel(Device& owner) : m_owner(owner) {}

    void Launch(const Traits::Api& api, Size3 groups, uint64_t counts, const void* params,
                uint32_t bytes, Traits::Stream stream) const {
        uint64_t& written = m_module->counts;
        if (counts != written) {
            Traits::Fill(api, m_module->dispatchArgs, uint32_t(counts), 4, stream);
            Traits::Fill(api, m_module->dispatchArgs + 4, uint32_t(counts >> 32), 4, stream);
            written = counts;
        }
        size_t size = bytes;
        void* extra[] = {reinterpret_cast<void*>(Traits::kParamBuffer), const_cast<void*>(params),
                         reinterpret_cast<void*>(Traits::kParamBufferSize), &size,
                         reinterpret_cast<void*>(Traits::kParamEnd)};
        Traits::Launch(api, m_function, groups, GroupSize(), bytes > 0 ? extra : nullptr, stream);
    }

    uint64_t& Counts() const {
        return m_module->counts;
    }

    static constexpr uint64_t kUnknownCounts = ~uint64_t(0);

  protected:
    KernelStats Load(const KernelCode& code, const DiskCache& cache,
                     const DeviceGlobals& globals) override {
        const RuntimeCompiler& compiler = Traits::RuntimeCompiler();
        const std::string& architecture = m_owner.Adapter().architecture;
        const std::vector<std::string> options =
            Traits::CompileOptions(architecture, code.shape.groupSize, code.debugInfo);
        const SourceHeader prelude{kPreludeHeader, Traits::Prelude()};
        const RuntimeCompiler::Compiled binary =
            compiler.CompileCached(code.code, {&prelude, 1}, options,
                                   std::format("kernel {} of program \"{}\" for {}", Name(),
                                               GetProgram().Label(), architecture),
                                   cache);

        const auto scope = Traits::Enter(m_owner);
        const auto& api = m_owner.Functions();
        m_module =
            std::static_pointer_cast<LoadedModule<Traits>>(m_owner.Modules().Find(binary.code, [&] {
                Device& owner = m_owner;
                const std::shared_ptr<LoadedModule<Traits>> loaded(
                    new LoadedModule<Traits>{}, [&owner](LoadedModule<Traits>* unloading) {
                        if (unloading->module != Module{}) {
                            const auto unloadScope = Traits::Enter(owner);
                            Traits::UnloadModule(owner.Functions(), unloading->module);
                        }
                        delete unloading;
                    });
                loaded->module = Traits::LoadModule(api, binary.code);
                const auto [globalsAddress, globalsBytes] =
                    Traits::Symbol(api, loaded->module, "SLANG_globalParams");
                if (globalsAddress != 0) {
                    Traits::CopyToDevice(api, globalsAddress, &globals,
                                         std::min<uint64_t>(globalsBytes, sizeof(globals)));
                }
                loaded->dispatchArgs =
                    Traits::Symbol(api, loaded->module, "ernie_dispatch_args").first;
                return std::shared_ptr<void>(loaded);
            }));
        m_function = Traits::GetFunction(api, m_module->module, Name().c_str());

        KernelStats stats = Traits::Stats(api, m_function);
        stats.cached = binary.cached;
        return stats;
    }

  private:
    using Module = Traits::Module;

    Device& m_owner;
    std::shared_ptr<LoadedModule<Traits>> m_module;
    Traits::Function m_function{};
};

template <ComputeTraits Traits> class ComputeQueue : public Queue {
  public:
    using Device = Traits::Device;

    explicit ComputeQueue(Device& device) : Queue(device) {
        const auto scope = Traits::Enter(device);
        m_stream = Traits::CreateStream(Functions());
        try {
            m_pool = Traits::CreatePool(device, kPoolKeptBytes);
            if (m_pool != Pool{}) {
                m_allocationStream = Traits::CreateStream(Functions());
            }
        } catch (...) {
            DestroyStreams();
            throw;
        }
    }

    ~ComputeQueue() override {
        try {
            const auto scope = Traits::Enter(Owner());
            for (const Pending& pending : m_pending) {
                Traits::DestroyEvent(Functions(), pending.event);
            }
            for (const Event event : m_free) {
                Traits::DestroyEvent(Functions(), event);
            }
            if (m_anchorStream != Stream{}) {
                Traits::DestroyEvent(Functions(), m_anchor);
                Traits::DestroyStream(Functions(), m_anchorStream);
            }
            DestroyStreams();
        } catch (...) {  // NOLINT(bugprone-empty-catch)
        }
    }

    uint64_t Allocate(uint64_t bytes) const {
        if (m_pool == Pool{}) {
            return Traits::Allocate(Functions(), bytes);
        }
        const uint64_t address =
            Traits::AllocateFrom(Functions(), m_pool, bytes, m_allocationStream);
        Traits::SynchronizeStream(Functions(), m_allocationStream);
        return address;
    }

    void Free(uint64_t address) const {
        if (m_pool == Pool{}) {
            Traits::Free(Functions(), address);
        } else {
            Traits::FreeTo(Functions(), address, m_allocationStream);
            Traits::SynchronizeStream(Functions(), m_allocationStream);
        }
    }

  protected:
    void Execute(const CommandBuffer& queued, const CommandBuffer& commands,
                 const Timestamps* timestamps, uint64_t value) override {
        const auto scope = Traits::Enter(Owner());
        const auto* events = static_cast<const EventTimestamps*>(timestamps);
        Run(queued, {});
        if (!commands.Reusable()) {
            Run(commands, {.timestamps = events});
        } else {
            Replay(commands, events);
        }

        const Event event = TakeEvent();
        Traits::Record(Functions(), event, m_stream, false);
        const std::lock_guard lock(m_mutex);
        m_pending.push_back({value, event});
    }

    std::shared_ptr<Timestamps> CreateTimestamps(uint32_t count) override {
        return std::make_shared<EventTimestamps>(*this, count);
    }

    bool SharesStreams() override {
        return true;
    }

    void* NativeStream() override {
        return m_stream;
    }

    void FollowStream(void* stream) override {
        const auto scope = Traits::Enter(Owner());
        const auto& api = Functions();
        const Event event = Traits::CreateEvent(api, false);
        Traits::Record(api, event, Traits::ApplicationStream(stream), false);
        Traits::WaitEvent(api, m_stream, event);
        Traits::DestroyEvent(api, event);
    }

    void StreamWait(void* stream, uint64_t) override {
        const auto scope = Traits::Enter(Owner());
        const auto& api = Functions();
        const Event event = Traits::CreateEvent(api, false);
        Traits::Record(api, event, m_stream, false);
        Traits::WaitEvent(api, Traits::ApplicationStream(stream), event);
        Traits::DestroyEvent(api, event);
    }

    uint64_t Finished() const override {
        const auto scope = Traits::Enter(Owner());
        const std::lock_guard lock(m_mutex);
        while (!m_pending.empty() && Traits::Query(Functions(), m_pending.front().event)) {
            m_finished = m_pending.front().value;
            m_free.push_back(m_pending.front().event);
            m_pending.pop_front();
        }
        return m_finished;
    }

    void WaitFor(uint64_t value) const override {
        Event event{};
        {
            const std::lock_guard lock(m_mutex);
            if (m_finished >= value) {
                return;
            }
            for (const Pending& pending : m_pending) {
                if (pending.value >= value) {
                    event = pending.event;
                    break;
                }
            }
        }
        if (event != Event{}) {
            const auto scope = Traits::Enter(Owner());
            Traits::Synchronize(Functions(), event);
        }
        Finished();
    }

  private:
    using Stream = Traits::Stream;
    using Event = Traits::Event;
    using Pool = Traits::Pool;

    struct Pending {
        uint64_t value;
        Event event;
    };

    const Device& Owner() const {
        return static_cast<const Device&>(GetDevice());
    }

    const Traits::Api& Functions() const {
        return Owner().Functions();
    }

    struct EventTimestamps : Timestamps {
        EventTimestamps(const ComputeQueue& owner, size_t count) : queue(owner) {
            const auto scope = Traits::Enter(owner.Owner());
            events.reserve(count);
            for (size_t i = 0; i < count; ++i) {
                events.push_back(Traits::CreateEvent(owner.Functions(), true));
            }
        }

        ~EventTimestamps() override {
            try {
                const auto scope = Traits::Enter(queue.Owner());
                for (const Event event : events) {
                    Traits::DestroyEvent(queue.Functions(), event);
                }
            } catch (...) {  // NOLINT(bugprone-empty-catch)
            }
        }

        EventTimestamps(const EventTimestamps&) = delete;
        EventTimestamps& operator=(const EventTimestamps&) = delete;

        bool Read(std::span<uint64_t> ns) const override {
            const auto scope = Traits::Enter(queue.Owner());
            const auto [anchor, anchorNs] = queue.Anchor();
            for (size_t i = 0; i < ns.size(); ++i) {
                ns[i] = anchorNs - Traits::ElapsedNs(queue.Functions(), events[i], anchor);
            }
            return true;
        }

        const ComputeQueue& queue;
        std::vector<Event> events;
    };

    struct Graph {
        Traits::Graph graph{};
        uint64_t data = 0;
        std::vector<std::pair<const ComputeKernel<Traits>*, uint64_t>> counts;
    };

    struct RunDesc {
        uint64_t data = 0;
        const EventTimestamps* timestamps = nullptr;
        bool capturing = false;
    };

    void Run(const CommandBuffer& list, const RunDesc& desc) const {
        const auto& api = Functions();
        for (const Command& command : list.Commands()) {
            std::visit(
                Overloaded{
                    [&](const CopyCommand& c) {
                        Traits::Copy(api, c.dst->Address() + c.dstOffset,
                                     c.src->Address() + c.srcOffset, c.bytes, m_stream);
                    },
                    [&](const FillCommand& c) {
                        Traits::Fill(api, c.dst->Address() + c.offset, c.value, c.bytes, m_stream);
                    },
                    [&](const UpdateCommand& c) {
                        if (desc.data != 0) {
                            Traits::Copy(api, c.dst->Address() + c.offset, desc.data + c.data,
                                         c.bytes, m_stream);
                        } else {
                            Traits::Upload(api, c.dst->Address() + c.offset, list.Data(c), c.bytes,
                                           m_stream);
                        }
                    },
                    [&](const BarrierCommand&) {},
                    [&](const ZoneCommand& c) {
                        if (desc.timestamps != nullptr) {
                            Traits::Record(api,
                                           desc.timestamps->events[2 * c.index + (c.begin ? 0 : 1)],
                                           m_stream, desc.capturing);
                        }
                    },
                    [&](const BuildAccelCommand&) {},
                    [&](const TextureCopyCommand& c) {
                        static_cast<const ComputeTexture<Traits>&>(*c.dst).Copy(
                            c.level, c.src->Address() + c.srcOffset, m_stream);
                    },
                    [&](const DispatchCommand& c) {
                        const uint64_t counts =
                            c.counts != nullptr ? c.counts->Address() + c.countsOffset : 0;
                        Kernel(c).Launch(api, c.groups, counts, list.Data(c), c.paramBytes,
                                         m_stream);
                    },
                },
                command);
        }
    }

    static const ComputeKernel<Traits>& Kernel(const DispatchCommand& dispatch) {
        return static_cast<const ComputeKernel<Traits>&>(*dispatch.kernel);
    }

    void Replay(const CommandBuffer& commands, const EventTimestamps* timestamps) const {
        std::shared_ptr<void>& recording = Recording(commands);
        if (!recording) {
            recording = Capture(commands, timestamps);
        }
        const Graph& graph = *static_cast<const Graph*>(recording.get());
        Traits::LaunchGraph(Functions(), graph.graph, m_stream);
        for (const auto& [kernel, counts] : graph.counts) {
            kernel->Counts() = counts;
        }
    }

    Event TakeEvent() {
        {
            const std::lock_guard lock(m_mutex);
            if (!m_free.empty()) {
                const Event event = m_free.back();
                m_free.pop_back();
                return event;
            }
        }
        return Traits::CreateEvent(Functions(), false);
    }

    std::shared_ptr<Graph> Capture(const CommandBuffer& commands,
                                   const EventTimestamps* timestamps) const {
        const auto& api = Functions();
        std::shared_ptr<Graph> graph(new Graph, [this](Graph* freed) {
            try {
                const auto scope = Traits::Enter(Owner());
                if (freed->graph != typename Traits::Graph{}) {
                    Traits::DestroyGraph(Functions(), freed->graph);
                }
                if (freed->data != 0) {
                    Free(freed->data);
                }
            } catch (...) {  // NOLINT(bugprone-empty-catch)
            }
            delete freed;
        });

        const std::span<const std::byte> data = commands.Data();
        const bool updates = std::ranges::any_of(commands.Commands(), [](const Command& command) {
            return std::holds_alternative<UpdateCommand>(command);
        });
        if (updates) {
            graph->data = Allocate(data.size());
            Traits::CopyToDevice(api, graph->data, data.data(), data.size());
        }

        for (const Command& command : commands.Commands()) {
            if (const auto* dispatch = std::get_if<DispatchCommand>(&command)) {
                Kernel(*dispatch).Counts() = ComputeKernel<Traits>::kUnknownCounts;
            }
        }
        graph->graph = RecordGraph(commands, graph->data, timestamps);
        graph->counts = KernelCounts(commands);
        return graph;
    }

    Traits::Graph RecordGraph(const CommandBuffer& commands, uint64_t data,
                              const EventTimestamps* timestamps) const {
        const auto& api = Functions();
        Traits::BeginCapture(api, m_stream);
        try {
            Run(commands, {.data = data, .timestamps = timestamps, .capturing = true});
        } catch (...) {
            try {
                Traits::DestroyGraph(api, Traits::EndCapture(api, m_stream));
            } catch (...) {  // NOLINT(bugprone-empty-catch)
            }
            throw;
        }
        return Traits::EndCapture(api, m_stream);
    }

    static std::vector<std::pair<const ComputeKernel<Traits>*, uint64_t>>
    KernelCounts(const CommandBuffer& commands) {
        std::vector<std::pair<const ComputeKernel<Traits>*, uint64_t>> counts;
        for (const Command& command : commands.Commands()) {
            if (const auto* dispatch = std::get_if<DispatchCommand>(&command)) {
                const ComputeKernel<Traits>* kernel = &Kernel(*dispatch);
                if (std::ranges::none_of(counts,
                                         [&](const auto& held) { return held.first == kernel; })) {
                    counts.emplace_back(kernel, kernel->Counts());
                }
            }
        }
        return counts;
    }

    std::pair<Event, uint64_t> Anchor() const {
        const auto& api = Functions();
        if (m_anchorStream == Stream{}) {
            m_anchorStream = Traits::CreateStream(api);
            m_anchor = Traits::CreateEvent(api, true);
        }
        const uint64_t before = profiler::Now();
        Traits::Record(api, m_anchor, m_anchorStream, false);
        Traits::Synchronize(api, m_anchor);
        const uint64_t after = profiler::Now();
        return {m_anchor, before + (after - before) / 2};
    }

    void DestroyStreams() {
        if (m_allocationStream != Stream{}) {
            Traits::SynchronizeStream(Functions(), m_allocationStream);
            Traits::DestroyStream(Functions(), m_allocationStream);
        }
        if (m_pool != Pool{}) {
            Traits::DestroyPool(Functions(), m_pool);
        }
        Traits::DestroyStream(Functions(), m_stream);
    }

    Stream m_stream{};
    Pool m_pool{};
    Stream m_allocationStream{};
    mutable Stream m_anchorStream{};
    mutable Event m_anchor{};
    mutable std::mutex m_mutex;
    mutable std::deque<Pending> m_pending;
    mutable std::vector<Event> m_free;
    mutable uint64_t m_finished = 0;
};

}
