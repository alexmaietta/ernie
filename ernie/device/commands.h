// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <type_traits>
#include <variant>
#include <vector>

#include "ernie/device/accel.h"
#include "ernie/device/buffer.h"
#include "ernie/device/program.h"
#include "ernie/device/ref.h"
#include "ernie/device/texture.h"
#include "ernie/export.h"
#include "ernie/util/profiler.h"

namespace ernie::device {

class Queue;
using profiler::ZoneId;

struct CopyCommand {
    Buffer* src;
    uint64_t srcOffset;
    Buffer* dst;
    uint64_t dstOffset;
    uint64_t bytes;
};

struct TextureCopyCommand {
    Buffer* src;
    uint64_t srcOffset;
    Texture* dst;
    uint32_t level;
};

struct FillCommand {
    Buffer* dst;
    uint64_t offset;
    uint64_t bytes;
    uint32_t value;
};

struct UpdateCommand {
    Buffer* dst;
    uint64_t offset;
    uint64_t bytes;
    size_t data;
};

struct DispatchCommand {
    const Kernel* kernel;
    Size3 groups;
    size_t data;
    uint32_t paramBytes;
    Buffer* counts = nullptr;
    uint64_t countsOffset = 0;
};

struct BuildAccelCommand {
    Accel* dst;
    AccelInput input;
    Buffer* scratch;
    uint64_t scratchOffset;
    bool update = false;
};

struct BarrierCommand {};

struct ZoneCommand {
    uint32_t index;
    bool begin;
};

using Command = std::variant<CopyCommand, TextureCopyCommand, FillCommand, UpdateCommand,
                             DispatchCommand, BuildAccelCommand, BarrierCommand, ZoneCommand>;

template <class... Lambdas> struct Overloaded : Lambdas... {
    using Lambdas::operator()...;
};

struct FinishDesc {
    bool reusable = false;
};

class ERNIE_API CommandBuffer {
  public:
    CommandBuffer(CommandBuffer&& other) noexcept;
    CommandBuffer& operator=(CommandBuffer&& other) noexcept;
    ~CommandBuffer();

    bool Empty() const {
        return m_commands.empty();
    }

    bool Reusable() const {
        return m_reusable;
    }

    std::span<const Command> Commands() const {
        return m_commands;
    }

    std::span<const ZoneId> Zones() const {
        return m_zones;
    }

    std::span<const std::byte> Data() const {
        return m_data;
    }

    template <class DataCommand> const std::byte* Data(const DataCommand& command) const {
        return m_data.data() + command.data;
    }

  private:
    friend class CommandEncoder;
    friend class Queue;

    CommandBuffer(Queue& queue, bool own);
    void Release();
    void Close();
    void HoldResources();

    Queue* m_queue;
    Ref<Device> m_device;
    std::vector<Command> m_commands;
    std::vector<std::byte> m_data;
    std::vector<ZoneId> m_zones;
    bool m_open = false;
    bool m_reusable = false;
    std::vector<Ref<Resource>> m_held;
    mutable uint64_t m_submission = 0;

    struct Recorded {
        std::shared_ptr<void> timestamps;
        std::shared_ptr<void> native;
    };

    mutable Recorded m_recorded;
};

class ERNIE_API CommandEncoder {
  public:
    void Copy(BufferRange src, BufferRange dst);

    // Replaces one whole level of dst with tightly packed texels; see Texture::LevelBytes. Every
    // level is undefined until written, so write them all before kernels sample dst.
    void Copy(BufferRange src, Texture& dst, uint32_t level);

    void Fill(BufferRange dst, uint32_t value = 0);

    void Update(BufferRange dst, const void* data);

    template <class Params>
    void Dispatch(Kernel& kernel, const Params& params, Size3 groups,
                  std::optional<BufferRange> counts = std::nullopt) {
        static_assert(std::is_trivially_copyable_v<Params>, "parameters are copied as bytes");
        Dispatch(kernel, &params, sizeof(Params), groups, counts);
    }

    void Dispatch(Kernel& kernel, Size3 groups, std::optional<BufferRange> counts = std::nullopt) {
        Dispatch(kernel, nullptr, 0, groups, counts);
    }

    void BuildAccel(Accel& dst, const AccelInput& input, BufferRange scratch, bool update = false);

    void Dispatch(Kernel& kernel, const void* params, size_t bytes, Size3 groups,
                  std::optional<BufferRange> counts = std::nullopt);

    void Concurrent(bool concurrent);

    void Barrier();

    void BeginZone(ZoneId zone);

    void EndZone();

    CommandBuffer Finish(const FinishDesc& desc = {});

    static constexpr uint64_t kMaxUpdateBytes = 65536;

  private:
    friend class Queue;

    CommandEncoder(Queue& queue, bool own);

    void Append(const CommandBuffer& commands);
    void Require(const Resource& resource, const char* what) const;
    size_t Store(const void* data, size_t bytes, size_t alignment);
    void Record(const Command& command);

    CommandBuffer m_recording;
    std::vector<uint32_t> m_open;
    bool m_concurrent = false;
    bool m_barrier = false;
    bool m_own;
};

class ERNIE_API ZoneScope {
  public:
    ZoneScope(CommandEncoder& encoder, ZoneId zone) : m_encoder(encoder) {
        encoder.BeginZone(zone);
    }

    ~ZoneScope();
    ZoneScope(const ZoneScope&) = delete;
    ZoneScope& operator=(const ZoneScope&) = delete;

  private:
    CommandEncoder& m_encoder;
};

}
