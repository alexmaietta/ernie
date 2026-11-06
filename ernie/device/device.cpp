// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/device.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <format>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "ernie/device/cuda/device.h"
#include "ernie/device/hip/device.h"
#include "ernie/device/kernels/shared.h"
#include "ernie/device/queue.h"
#include "ernie/device/slang/compiler.h"
#include "ernie/device/slang/debug.h"
#include "ernie/device/vulkan/device.h"
#include "ernie/util/disk_cache.h"
#include "ernie/util/environment.h"
#include "ernie/util/error.h"
#include "ernie/util/log.h"
#include "ernie/util/span.h"
#include "ernie/util/text.h"

namespace ernie::device {
namespace {

constexpr uint64_t kKernelCacheBytes = uint64_t(1) << 30;

BufferRange ContiguousRange(const Tensor& tensor, const char* role) {
    if (tensor.Empty()) {
        throw Error("Copying needs a {} tensor, but it's empty", role);
    }
    const Span<const int64_t> shape = tensor.Shape();
    const Span<const int64_t> strides = tensor.Strides();
    int64_t packed = 1;
    for (size_t axis = shape.size(); axis-- > 0;) {
        if (!strides.empty() && shape[axis] > 1 && strides[axis] != packed) {
            throw Error("Copying needs a contiguous {} tensor", role);
        }
        packed *= shape[axis];
    }
    return tensor.GetBuffer().Slice(tensor.Offset(),
                                    uint64_t(tensor.Elements()) * ElementBytes(tensor.Type()));
}

std::optional<ExternalAddress> SharedAddress(Buffer& buffer, Device& reader) {
    const std::optional<DLDevice> api = reader.SharedDevice();
    if (!api) {
        return std::nullopt;
    }
    try {
        const ExternalAddress external = buffer.External(api->device_type);
        if (reader.ReadsDirectly(external.device)) {
            return external;
        }
    } catch (const UnsupportedError&) {
    }
    return std::nullopt;
}

constexpr uint64_t kFullyBackedPageBytes = uint64_t(64) * 1024;

constexpr Choice<Traversal> kTraversals[] = {{"auto", Traversal::Auto},
                                             {"hardware", Traversal::Hardware},
                                             {"software", Traversal::Software}};

constexpr Choice<Sampling> kSamplings[] = {
    {"auto", Sampling::Auto}, {"hardware", Sampling::Hardware}, {"software", Sampling::Software}};

struct Implementation {
    std::string_view name;
    Backend value;
    const char* title;
    std::vector<AdapterInfo> (*list)() = nullptr;
    Ref<Device> (*create)(const AdapterInfo&, const DeviceDesc&, Traversal) = nullptr;
};

class BorrowedBuffer : public Buffer {
  public:
    BorrowedBuffer(Device& device, const BufferDesc& desc, uint64_t address)
        : Buffer(device, desc) {
        m_address = address;
    }
};

template <class BackendDevice>
Ref<Device> Create(const AdapterInfo& adapter, const DeviceDesc& desc, Traversal traversal) {
    return Ref<Device>(new BackendDevice(adapter, desc, traversal));
}

constexpr Implementation kBackends[] = {
    {"vulkan", Backend::Vulkan, "Vulkan", vulkan::ListAdapters, Create<vulkan::VulkanDevice>},
    {"cuda", Backend::Cuda, "CUDA", cuda::ListAdapters, Create<cuda::CudaDevice>},
    {"hip", Backend::Hip, "HIP", hip::ListAdapters, Create<hip::HipDevice>},
    {"metal", Backend::Metal, "Metal"},
    {"cpu", Backend::Cpu, "CPU"},
};

DeviceDesc WithEnvironment(DeviceDesc desc) {
    if (const std::optional<Backend> backend = EnvironmentChoice("ERNIE_BACKEND", kBackends)) {
        Log(LogLevel::Info, "ERNIE_BACKEND chooses {}", BackendName(*backend));
        desc.backend = backend;
    }
    if (std::string adapter = Environment("ERNIE_ADAPTER"); !adapter.empty()) {
        Log(LogLevel::Info, "ERNIE_ADAPTER chooses \"{}\"", adapter);
        desc.adapter = std::move(adapter);
    }
    if (const std::optional<Traversal> traversal =
            EnvironmentChoice("ERNIE_TRAVERSAL", kTraversals)) {
        Log(LogLevel::Info, "ERNIE_TRAVERSAL chooses {} traversal", Environment("ERNIE_TRAVERSAL"));
        desc.traversal = *traversal;
    }
    if (const std::optional<Sampling> sampling = EnvironmentChoice("ERNIE_SAMPLING", kSamplings)) {
        Log(LogLevel::Info, "ERNIE_SAMPLING chooses {} texture sampling",
            Environment("ERNIE_SAMPLING"));
        desc.sampling = *sampling;
    }
    if (Environment("ERNIE_VALIDATION") == "1") {
        Log(LogLevel::Info, "ERNIE_VALIDATION validates Vulkan calls");
        desc.validation = true;
    }
    if (Environment("ERNIE_DEVICE_SYNC") == "1") {
        Log(LogLevel::Info, "ERNIE_DEVICE_SYNC runs each command alone and waits for it");
        desc.synchronous = true;
    }
    if (Environment("ERNIE_KERNEL_DEBUG_INFO") == "1") {
        Log(LogLevel::Info, "ERNIE_KERNEL_DEBUG_INFO compiles kernels with line info");
        desc.kernelDebugInfo = true;
    }
    return desc;
}

std::optional<Backend> AliasBackend(std::string_view adapter) {
    const size_t colon = adapter.find(':');
    for (const Implementation& implementation : kBackends) {
        if (colon != std::string_view::npos && adapter.substr(0, colon) == implementation.name) {
            return implementation.value;
        }
    }
    return std::nullopt;
}

const Implementation& Implement(Backend backend) {
    for (const Implementation& implementation : kBackends) {
        if (implementation.value == backend) {
            if (implementation.list == nullptr) {
                throw UnsupportedError("Ernie has no {} backend yet", implementation.title);
            }
            return implementation;
        }
    }
    throw UnsupportedError("Ernie has no such backend");
}

void AddAdapters(std::vector<AdapterInfo>& adapters, const Implementation& implementation,
                 bool required) {
    std::vector<AdapterInfo> found;
    try {
        found = implementation.list();
    } catch (const UnsupportedError&) {
        if (required) {
            throw;
        }
        return;
    } catch (const Error& error) {
        if (required) {
            throw;
        }
        Log(LogLevel::Warning, "{} adapters could not be listed: {}", implementation.title,
            error.what());
        return;
    }

    for (AdapterInfo& adapter : found) {
        adapter.alias = std::format("{}:{}", implementation.name, adapter.ordinal);
        if (!adapter.unsupported.empty()) {
            Log(LogLevel::Debug, "Ernie can't run on {} ({}): {}", adapter.alias, adapter.name,
                adapter.unsupported);
        }
        adapters.push_back(std::move(adapter));
    }
}

bool Matches(const AdapterInfo& adapter, std::string_view wanted) {
    return adapter.alias == wanted ||
           Lowercase(adapter.name).find(Lowercase(wanted)) != std::string::npos;
}

std::tuple<int, int, int> Preference(const AdapterInfo& adapter) {
    const int type = int(adapter.type);
    switch (adapter.backend) {
    case Backend::Vulkan:
        if (adapter.features.hardwareTraversal && adapter.type == AdapterType::Discrete) {
            return {0, 0, 0};
        }
        return {4, adapter.features.hardwareTraversal ? 0 : 1, type};
    case Backend::Metal:
        return {1, 0, type};
    case Backend::Cuda:
        return {2, 0, type};
    case Backend::Hip:
        return {3, 0, type};
    case Backend::Cpu:
        return {5, 0, type};
    }
    return {6, 0, type};
}

std::string Describe(const std::vector<AdapterInfo>& adapters) {
    if (adapters.empty()) {
        return "Ernie found no adapters";
    }
    std::string text = "Ernie found:";
    for (const AdapterInfo& adapter : adapters) {
        text += std::format("\n  {}: {}", adapter.alias, adapter.name);
        if (!adapter.unsupported.empty()) {
            text += std::format(" (unsupported: {})", adapter.unsupported);
        }
    }
    return text;
}

const AdapterInfo& ChooseAdapter(const std::vector<AdapterInfo>& adapters, const DeviceDesc& desc) {
    std::vector<const AdapterInfo*> matching;
    for (const AdapterInfo& adapter : adapters) {
        if ((!desc.backend || adapter.backend == *desc.backend) && Matches(adapter, desc.adapter)) {
            matching.push_back(&adapter);
        }
    }

    if (matching.empty()) {
        std::string wanted = "adapter";
        if (!desc.adapter.empty()) {
            wanted = std::format("adapter \"{}\"", desc.adapter);
        } else if (desc.backend) {
            wanted = std::format("{} adapter", BackendName(*desc.backend));
        }
        throw UnsupportedError("there is no {}. {}", wanted, Describe(adapters));
    }

    const auto reason = [&](const AdapterInfo& adapter) -> std::string {
        if (!adapter.unsupported.empty()) {
            return adapter.unsupported;
        }
        if (desc.presentation && !adapter.features.presentation) {
            return "it can't present to windows";
        }
        return {};
    };
    std::vector<const AdapterInfo*> supported;
    std::copy_if(matching.begin(), matching.end(), std::back_inserter(supported),
                 [&](const AdapterInfo* adapter) { return reason(*adapter).empty(); });
    if (supported.empty()) {
        const AdapterInfo& first = *matching.front();
        throw UnsupportedError("Ernie can't run on {} ({}): {}", first.alias, first.name,
                               reason(first));
    }

    return **std::min_element(
        supported.begin(), supported.end(),
        [](const AdapterInfo* a, const AdapterInfo* b) { return Preference(*a) < Preference(*b); });
}

Traversal ResolveTraversal(const AdapterInfo& adapter, Traversal wanted) {
    if (wanted == Traversal::Hardware && !adapter.features.hardwareTraversal) {
        throw UnsupportedError("{} ({}) has no hardware ray traversal", adapter.alias,
                               adapter.name);
    }
    if (wanted == Traversal::Auto) {
        if (!adapter.features.hardwareTraversal) {
            Log(LogLevel::Debug, "{} has no hardware ray traversal, so rays use software BVHs",
                adapter.alias);
            return Traversal::Software;
        }
        if (adapter.type == AdapterType::Cpu) {
            Log(LogLevel::Debug, "{} emulates ray traversal on the CPU, so rays use software BVHs",
                adapter.alias);
            return Traversal::Software;
        }
        return Traversal::Hardware;
    }
    return wanted;
}

Sampling ResolveSampling(const AdapterInfo& adapter, Sampling wanted) {
    if (wanted == Sampling::Hardware && !adapter.features.hardwareTextures) {
        throw UnsupportedError("{} ({}) has no hardware texture sampling", adapter.alias,
                               adapter.name);
    }
    if (wanted == Sampling::Auto) {
        if (!adapter.features.hardwareTextures) {
            Log(LogLevel::Debug,
                "{} has no hardware texture sampling, so textures are sampled in software",
                adapter.alias);
            return Sampling::Software;
        }
        return Sampling::Hardware;
    }
    return wanted;
}

std::vector<AdapterInfo> FindAdapters(const DeviceDesc& desc) {
    const std::optional<Backend> backend = desc.backend ? desc.backend : AliasBackend(desc.adapter);
    if (!backend) {
        return EnumerateAdapters();
    }
    std::vector<AdapterInfo> adapters;
    AddAdapters(adapters, Implement(*backend), true);
    return adapters;
}

}

std::string_view BackendName(Backend backend) {
    for (const Implementation& implementation : kBackends) {
        if (implementation.value == backend) {
            return implementation.name;
        }
    }
    return "unknown";
}

std::vector<AdapterInfo> EnumerateAdapters() {
    std::vector<AdapterInfo> adapters;
    for (const Implementation& implementation : kBackends) {
        if (implementation.list != nullptr) {
            AddAdapters(adapters, implementation, false);
        }
    }
    return adapters;
}

Ref<Device> Device::Create(const DeviceDesc& requested) {
    const DeviceDesc desc = WithEnvironment(requested);
    const std::vector<AdapterInfo> adapters = FindAdapters(desc);
    const AdapterInfo& adapter = ChooseAdapter(adapters, desc);
    const Traversal traversal = ResolveTraversal(adapter, desc.traversal);
    DeviceDesc resolved = desc;
    resolved.sampling = ResolveSampling(adapter, desc.sampling);

    Ref<Device> device = Implement(adapter.backend).create(adapter, resolved, traversal);
    device->CreateGlobals();

    Log(LogLevel::Info, "using {} ({}), with {} ray traversal and {} texture sampling",
        adapter.alias, adapter.name, traversal == Traversal::Hardware ? "hardware" : "software",
        resolved.sampling == Sampling::Hardware ? "hardware" : "software");
    return device;
}

struct Device::Counters {
    std::mutex mutex;
    std::map<std::string, MemoryUse> uses;
    MemoryStats totals;
    TrafficStats traffic;
};

struct Device::AccelSizeCache {
    std::mutex mutex;
    std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, bool, bool>, AccelSizes> sizes;
};

Device::Device(AdapterInfo adapter, const DeviceDesc& desc, Traversal traversal)
    : m_adapter(std::move(adapter)), m_traversal(traversal), m_sampling(desc.sampling),
      m_cacheDirectory(desc.cacheDirectory.empty() ? UserCacheDirectory() : desc.cacheDirectory),
      m_kernelCacheDirectory(ernie::KernelCacheDirectory(m_cacheDirectory)),
      m_profiling(desc.profiling), m_synchronous(desc.synchronous),
      m_kernelDebugInfo(desc.kernelDebugInfo), m_presents(desc.presentation),
      m_counters(std::make_unique<Counters>()), m_accelSizes(std::make_unique<AccelSizeCache>()) {
    static std::mutex trimming;
    static std::set<std::filesystem::path> trimmed;
    const std::lock_guard lock(trimming);
    if (trimmed.insert(m_cacheDirectory).second) {
        TrimKernelCache(m_cacheDirectory, kKernelCacheBytes);
    }
}

Device::~Device() = default;

Ref<Buffer> Device::CreateBuffer(const BufferDesc& requested) {
    const bool reserve = requested.memory == Memory::Virtual;
    const char* kind = reserve ? "virtual buffer" : "buffer";
    const uint64_t limit =
        reserve ? m_adapter.limits.maxReservedBytes : m_adapter.limits.maxBufferBytes;
    if (requested.bytes == 0) {
        throw Error("{} \"{}\" has no bytes", kind, requested.label);
    }
    if (requested.bytes > limit) {
        throw Error("{} \"{}\" has {} bytes; {} takes at most {}", kind, requested.label,
                    requested.bytes, m_adapter.alias, limit);
    }

    BufferDesc desc = requested;
    if (desc.memory == Memory::Shared && !m_adapter.features.hostMappedMemory) {
        desc.memory = Memory::Device;
    }

    Ref<Buffer> buffer;
    if (reserve && !m_adapter.features.virtualMemory) {
        desc.memory = Memory::Device;
        desc.bytes = RoundUp(desc.bytes, kFullyBackedPageBytes);
        buffer = NewBuffer(desc);
        buffer->m_memory = Memory::Virtual;
        buffer->m_pageBytes = kFullyBackedPageBytes;
    } else {
        buffer = NewBuffer(desc);
    }

    CountMemory(*buffer, 1);
    return buffer;
}

Tensor Device::CreateTensor(DType type, std::vector<int64_t> shape, std::string label) {
    uint64_t bytes = ElementBytes(type);
    for (const int64_t side : shape) {
        if (side < 0) {
            throw Error("Tensor \"{}\" has a side of {}", label, side);
        }
        bytes *= uint64_t(side);
    }
    const BufferUsage usage = SharedDevice() ? BufferUsage::Interop : BufferUsage::None;
    Ref<Buffer> buffer = CreateBuffer(
        {.bytes = std::max<uint64_t>(bytes, 4), .usage = usage, .label = std::move(label)});
    return Tensor(std::move(buffer), 0, type, std::move(shape));
}

Operation Device::Follow(Stream producer) {
    m_queue->Follow(producer);
    return m_queue->Mark();
}

Tensor Device::Import(const DLTensor& source, Stream producer, std::string label) {
    const DLDataType type = source.dtype;
    const std::optional<DType> dtype =
        type.lanes == 0                            ? std::nullopt
        : type.code == kDLFloat && type.bits == 32 ? std::optional(DType::Float32)
        : type.code == kDLFloat && type.bits == 64 ? std::optional(DType::Float64)
        : type.code == kDLInt && type.bits == 32   ? std::optional(DType::Int32)
        : type.code == kDLInt && type.bits == 64   ? std::optional(DType::Int64)
        : type.code == kDLUInt && type.bits == 8   ? std::optional(DType::UInt8)
        : type.code == kDLUInt && type.bits == 32  ? std::optional(DType::UInt32)
        : type.code == kDLUInt && type.bits == 64  ? std::optional(DType::UInt64)
                                                   : std::nullopt;
    if (!dtype) {
        throw UnsupportedError("{} imports no tensors of {}-bit type code {} with {} lanes",
                               m_adapter.alias, type.bits, type.code, type.lanes);
    }
    std::vector<int64_t> shape(source.shape, source.shape + source.ndim);
    int64_t elements = 1;
    for (int axis = source.ndim; axis-- > 0;) {
        if (source.strides != nullptr && shape[size_t(axis)] > 1 &&
            source.strides[axis] != elements) {
            throw UnsupportedError("\"{}\" is imported from contiguous tensors only", label);
        }
        elements *= shape[size_t(axis)];
    }
    if (type.lanes > 1) {
        shape.push_back(type.lanes);
        elements *= type.lanes;
    }
    const uint64_t bytes = uint64_t(elements) * ElementBytes(*dtype);
    const uint64_t address = uint64_t(source.data) + source.byte_offset;
    Ref<Buffer> buffer;
    const bool host =
        source.device.device_type == kDLCPU || source.device.device_type == kDLCUDAHost ||
        source.device.device_type == kDLROCMHost || source.device.device_type == kDLCUDAManaged;
    if (host) {
        if (producer.IsCompute()) {
            m_queue->Follow(producer);
            m_queue->Mark().Wait();
        }
        buffer =
            CreateBuffer({.label = label},
                         Span<const std::byte>(reinterpret_cast<const std::byte*>(address), bytes));
    } else if (bytes > 0 && ReadsDirectly(source.device)) {
        buffer = Borrow(source.device, address, bytes, label);
        m_queue->Follow(producer);
    } else {
        buffer = Stage(address, bytes, producer, label);
    }
    return Tensor(std::move(buffer), 0, *dtype, std::move(shape));
}

Operation Device::Copy(const Tensor& src, const Tensor& dst, const Operation& after) {
    const BufferRange from = ContiguousRange(src, "source");
    const BufferRange to = ContiguousRange(dst, "destination");
    if (&dst.GetDevice() != this) {
        throw Error("{} copies only into its own tensors, but the destination is on {}",
                    m_adapter.alias, dst.GetDevice().Adapter().alias);
    }
    if (from.Bytes() != to.Bytes()) {
        throw Error("Copying a tensor of {} bytes into one of {} bytes", from.Bytes(), to.Bytes());
    }
    Device& source = src.GetDevice();
    const auto copy = [](Queue& queue, BufferRange read, BufferRange written) {
        CommandEncoder encoder = queue.CreateEncoder();
        encoder.Copy(read, written);
        return queue.Submit(encoder.Finish());
    };

    if (&source == this) {
        m_queue->Follow(after);
        return copy(*m_queue, from, to);
    }
    if (const std::optional<ExternalAddress> shared = SharedAddress(src.GetBuffer(), *this)) {
        const Ref<Buffer> borrowed =
            Borrow(shared->device, shared->address, src.GetBuffer().Bytes(), "copy source");
        m_queue->Follow(after);
        const Operation copied = copy(*m_queue, borrowed->Slice(src.Offset(), from.Bytes()), to);
        m_queue->Retire(std::make_shared<Tensor>(src), copied);
        return copied;
    }
    if (const std::optional<ExternalAddress> shared = SharedAddress(dst.GetBuffer(), source)) {
        const Ref<Buffer> borrowed = source.Borrow(shared->device, shared->address,
                                                   dst.GetBuffer().Bytes(), "copy destination");
        Queue& queue = source.GetQueue();
        queue.Follow(after);
        const Operation copied = copy(queue, from, borrowed->Slice(dst.Offset(), to.Bytes()));
        queue.Retire(std::make_shared<Tensor>(dst), copied);
        m_queue->Follow(copied);
        return m_queue->Mark();
    }
    after.Wait();
    const std::vector<std::byte> bytes = src.ToHostBytes();
    m_queue->Write(to, bytes.data());
    return m_queue->Mark();
}

Ref<Buffer> Device::Borrow(DLDevice source, uint64_t address, uint64_t bytes, std::string label) {
    if (!ReadsDirectly(source)) {
        throw UnsupportedError("{} can't read memory of device type {} directly", m_adapter.alias,
                               int(source.device_type));
    }
    if (address == 0 || bytes == 0 || address % 4 != 0) {
        throw Error("Borrowed memory \"{}\" needs a word-aligned address and bytes", label);
    }
    return Ref<Buffer>(new BorrowedBuffer(
        *this, {.bytes = bytes, .memory = Memory::Device, .label = std::move(label)}, address));
}

Ref<Buffer> Device::CreateBuffer(BufferDesc desc, Span<const std::byte> data) {
    if (desc.bytes == 0) {
        desc.bytes = std::max<uint64_t>(data.size(), 4);
    }
    const Ref<Buffer> buffer = CreateBuffer(desc);
    if (!data.empty()) {
        GetQueue().Write(buffer->Slice(0, data.size()), data.data());
    }
    return buffer;
}

AccelSizes Device::AccelBuildSizes(const AccelInput& input) const {
    if (m_traversal != Traversal::Hardware) {
        throw UnsupportedError("AccelBuildSizes on {}, which traverses rays in software",
                               m_adapter.alias);
    }
    const auto key = std::visit(
        Overloaded{
            [](const AccelTriangles& t) {
                return std::tuple(0u, t.count, t.vertexCount, t.stride, t.indices.has_value(),
                                  t.refittable);
            },
            [](const AccelBoxes& b) {
                return std::tuple(1u, b.count, 0u, 0u, false, b.refittable);
            },
            [](const AccelInstances& i) { return std::tuple(2u, i.count, 0u, 0u, false, false); },
        },
        input);
    const std::lock_guard lock(m_accelSizes->mutex);
    const auto [found, added] = m_accelSizes->sizes.try_emplace(key);
    if (added) {
        found->second = HardwareBuildSizes(input);
    }
    return found->second;
}

Ref<Accel> Device::CreateAccel(const AccelInput& input, std::string label) {
    const AccelSizes sizes = AccelBuildSizes(input);
    Ref<Buffer> storage = CreateBuffer({.bytes = std::max<uint64_t>(sizes.storage, 256),
                                        .usage = BufferUsage::AccelStorage,
                                        .label = label});
    return NewAccel(std::move(storage), GetLevel(input), sizes, std::move(label));
}

AccelSizes Device::HardwareBuildSizes(const AccelInput&) const {
    throw UnsupportedError("{} builds no hardware acceleration structures", m_adapter.alias);
}

// NOLINTNEXTLINE(performance-unnecessary-value-param)
Ref<Accel> Device::NewAccel(Ref<Buffer>, AccelLevel, AccelSizes, std::string) {
    throw UnsupportedError("{} builds no hardware acceleration structures", m_adapter.alias);
}

Ref<Texture> Device::CreateTexture(const TextureDesc& desc) {
    if (m_sampling != Sampling::Hardware) {
        throw UnsupportedError("CreateTexture on {}, which samples textures in software",
                               m_adapter.alias);
    }
    if (desc.width == 0 || desc.height == 0) {
        throw Error("texture \"{}\" is {}x{}", desc.label, desc.width, desc.height);
    }
    const uint32_t levels = std::bit_width(std::max(desc.width, desc.height));
    if (desc.levels == 0 || desc.levels > levels) {
        throw Error("texture \"{}\" has {} levels; {}x{} takes 1 to {}", desc.label, desc.levels,
                    desc.width, desc.height, levels);
    }
    if (TexelBlockSide(desc.format) == 4 && !m_adapter.features.blockCompression) {
        throw UnsupportedError("texture \"{}\" is block compressed, which {} can't sample",
                               desc.label, m_adapter.alias);
    }
    Ref<Texture> texture = NewTexture(desc);
    CountMemory(*texture, 1);
    return texture;
}

Texture::~Texture() {
    if (m_counted) {
        GetDevice().CountMemory(*this, -1);
    }
}

Ref<Texture> Device::NewTexture(const TextureDesc& desc) {
    throw UnsupportedError("{} creates no hardware textures, so \"{}\" can't be made",
                           m_adapter.alias, desc.label);
}

std::optional<DLDevice> Device::SharedDevice() {
    switch (m_adapter.backend) {
    case Backend::Cuda:
        return DLDevice{kDLCUDA, int32_t(m_adapter.ordinal)};
    case Backend::Hip:
        return DLDevice{kDLROCM, int32_t(m_adapter.ordinal)};
    default:
        return std::nullopt;
    }
}

bool Device::ReadsDirectly(DLDevice source) const {
    const bool api = (m_adapter.backend == Backend::Cuda && source.device_type == kDLCUDA) ||
                     (m_adapter.backend == Backend::Hip && source.device_type == kDLROCM);
    return api && uint32_t(source.device_id) == m_adapter.ordinal;
}

void Device::CopyIn(BufferRange dst, uint64_t source, Stream producer) {
    if (dst.Bytes() == 0) {
        return;
    }
    if (!SharedDevice()) {
        throw UnsupportedError("{} shares no memory with CUDA or HIP", m_adapter.alias);
    }

    const Stream copier = producer.IsCompute() ? producer : CopyStream();
    const Operation earlier = m_queue->Mark();
    if (copier.IsCompute()) {
        earlier.Wait(copier);
    } else {
        earlier.Wait();
    }

    ExternalCopy(dst, source, copier);
    CountTraffic({.copyBytes = dst.Bytes()});
    m_queue->Follow(copier);
    if (producer.IsHost() && copier.IsCompute()) {
        m_queue->Mark().Wait();
    }
}

Ref<Buffer> Device::Stage(uint64_t source, uint64_t bytes, Stream producer,
                          const std::string& label) {
    if (!SharedDevice()) {
        throw UnsupportedError("{} shares no memory with CUDA or HIP", m_adapter.alias);
    }
    Ref<Buffer> staging =
        CreateBuffer({.bytes = bytes, .usage = BufferUsage::Interop, .label = label});
    if (bytes == 0) {
        return staging;
    }
    const Stream copier = producer.IsCompute() ? producer : CopyStream();
    ExternalCopy(*staging, source, copier);
    CountTraffic({.copyBytes = bytes});
    m_queue->Follow(copier);
    if (producer.IsHost() && copier.IsCompute()) {
        m_queue->Mark().Wait();
    }
    return staging;
}

void Device::ExternalCopy(BufferRange, uint64_t, Stream) {
    throw UnsupportedError("{} doesn't copy in foreign device memory", m_adapter.alias);
}

Stream Device::CopyStream() {
    return Stream::None();
}

void Device::Require(const Resource& resource, const char* what) const {
    if (&resource.GetDevice() != this) {
        throw Error("{} names \"{}\", which is on another device", what, resource.Label());
    }
}

std::vector<MemoryHeap> Device::Heaps() const {
    return {};
}

MemoryStats Device::MemoryUsage() const {
    std::vector<MemoryHeap> heaps = Heaps();
    std::lock_guard lock(m_counters->mutex);
    MemoryStats stats = m_counters->totals;
    stats.heaps = std::move(heaps);
    for (const auto& [tag, use] : m_counters->uses) {
        stats.uses.push_back(use);
    }
    std::sort(stats.uses.begin(), stats.uses.end(), [](const MemoryUse& a, const MemoryUse& b) {
        return a.bytes != b.bytes ? a.bytes > b.bytes : a.label < b.label;
    });
    return stats;
}

TrafficStats Device::Traffic() const {
    const std::lock_guard lock(m_counters->mutex);
    return m_counters->traffic;
}

void Device::CountTraffic(const TrafficStats& traffic) {
    const std::lock_guard lock(m_counters->mutex);
    m_counters->traffic += traffic;
}

DeviceGlobals Device::Globals() const {
    return {.debug = {m_debug->Address()}, .debugCapacity = DebugRecords::kCapacity, .reserved = 0};
}

void Device::CreateGlobals() {
    Ref<Buffer> debug = CreateBuffer(
        {.bytes = DebugRecords::Bytes(), .memory = Memory::Readback, .label = "debug"});
    Adopt(*debug);
    m_debug = std::make_unique<DebugRecords>(std::move(debug));

    m_globals = CreateBuffer(
        {.bytes = sizeof(DeviceGlobals), .memory = Memory::Upload, .label = "globals"});
    Adopt(*m_globals);
    const DeviceGlobals globals = Globals();
    std::memcpy(m_globals->Mapped(), &globals, sizeof(globals));
}

void Device::CloseQueue(Queue& queue) noexcept {
    queue.Close();
}

void Device::LastReleased() noexcept {
    m_programs = nullptr;
    m_queue->Close();
    m_debug.reset();
    m_globals = nullptr;
    m_queue.reset();
    delete this;
}

Resource::Resource(Device& device, std::string label)
    : m_device(&device), m_label(std::move(label)) {
    device.AddRef();
}

Resource::~Resource() {
    if (m_holdsDevice) {
        m_device->Release();
    }
}

void Resource::LastReleased() noexcept {
    m_device->Retire(*this);
}

void Device::Retire(Resource& resource) noexcept {
    const bool held = std::exchange(resource.m_holdsDevice, false);
    m_queue->Retire(resource);
    if (held) {
        Release();
    }
}

void Device::Adopt(Resource& resource) {
    if (std::exchange(resource.m_holdsDevice, false)) {
        resource.m_device->Release();
    }
}

void Device::CountMemory(Buffer& buffer, int sign) {
    const int64_t backed = sign * int64_t(buffer.FullyBacked() ? buffer.m_bytes : 0);
    const int64_t reserved = sign * int64_t(buffer.Virtual() ? buffer.m_bytes : 0);

    std::lock_guard lock(m_counters->mutex);
    MemoryUse& use = m_counters->uses[buffer.Label()];
    use.label = buffer.Label();
    use.buffers = uint32_t(int64_t(use.buffers) + sign);
    use.bytes = uint64_t(int64_t(use.bytes) + backed);
    if (use.buffers == 0 && use.textures == 0) {
        m_counters->uses.erase(buffer.Label());
    }

    MemoryStats& totals = m_counters->totals;
    uint64_t& kind = buffer.m_memory == Memory::Upload     ? totals.uploadBytes
                     : buffer.m_memory == Memory::Readback ? totals.readbackBytes
                     : buffer.m_memory == Memory::Shared   ? totals.sharedBytes
                                                           : totals.deviceBytes;
    kind = uint64_t(int64_t(kind) + backed);
    totals.reservedBytes = uint64_t(int64_t(totals.reservedBytes) + reserved);
    buffer.m_counted = sign > 0;
}

void Device::CountMemory(Texture& texture, int sign) {
    uint64_t bytes = 0;
    for (uint32_t level = 0; level < texture.Desc().levels; ++level) {
        bytes += texture.LevelBytes(level);
    }
    const int64_t change = sign * int64_t(bytes);

    std::lock_guard lock(m_counters->mutex);
    MemoryUse& use = m_counters->uses[texture.Label()];
    use.label = texture.Label();
    use.textures = uint32_t(int64_t(use.textures) + sign);
    use.bytes = uint64_t(int64_t(use.bytes) + change);
    if (use.buffers == 0 && use.textures == 0) {
        m_counters->uses.erase(texture.Label());
    }
    MemoryStats& totals = m_counters->totals;
    totals.deviceBytes = uint64_t(int64_t(totals.deviceBytes) + change);
    totals.textureBytes = uint64_t(int64_t(totals.textureBytes) + change);
    texture.m_counted = sign > 0;
}

}
