// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/queue.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/format.h>

#include "ernie/device/device.h"
#include "ernie/device/kernels/shared.h"
#include "ernie/device/slang/debug.h"
#include "ernie/util/error.h"
#include "ernie/util/profiler.h"
#include "ernie/util/span.h"

namespace ernie::device {
namespace {

constexpr uint64_t kMinStagingBytes = uint64_t(64) * 1024;
constexpr uint64_t kMaxIdleStagingBytes = 256ull * 1024 * 1024;
constexpr uint64_t kMaxStagingGrowth = 64ull * 1024 * 1024;
constexpr uint64_t kStagingAlignment = 16;

class StagingPool {
  public:
    using Create = std::function<Ref<Buffer>(Memory memory, uint64_t bytes)>;

    explicit StagingPool(Create create) : m_create(std::move(create)) {}

    BufferRange Take(Memory memory, uint64_t bytes) {
        Arena& arena = memory == Memory::Upload ? m_uploads : m_readbacks;
        const uint64_t need = RoundUp(bytes, kStagingAlignment);
        if (arena.buffer == nullptr || arena.cursor + need > arena.buffer->Bytes()) {
            const uint64_t grown = arena.buffer == nullptr
                                       ? 0
                                       : std::min(2 * arena.buffer->Bytes(), kMaxStagingGrowth);
            arena = {Claim(memory, std::max(need, grown)).buffer.Get(), 0};
        }
        Find(arena.buffer).busyUntil = kQueued;

        const uint64_t offset = arena.cursor;
        arena.cursor += need;
        return arena.buffer->Slice(offset, bytes);
    }

    void Submitted(uint64_t value) {
        for (Entry& entry : m_entries) {
            if (entry.busyUntil == kQueued) {
                entry.busyUntil = value;
            }
        }
    }

    std::vector<Ref<Buffer>> Trim(uint64_t completed) {
        m_completed = std::max(m_completed, completed);
        std::vector<Ref<Buffer>> dropped;
        uint64_t idle = 0;
        for (const Entry& entry : m_entries) {
            idle += Idle(entry) ? entry.buffer->Bytes() : 0;
        }

        while (idle > kMaxIdleStagingBytes) {
            auto largest = m_entries.end();
            for (auto it = m_entries.begin(); it != m_entries.end(); ++it) {
                if (Idle(*it) && (largest == m_entries.end() ||
                                  it->buffer->Bytes() > largest->buffer->Bytes())) {
                    largest = it;
                }
            }
            idle -= largest->buffer->Bytes();
            dropped.push_back(std::move(largest->buffer));
            m_entries.erase(largest);
        }
        return dropped;
    }

    std::vector<Ref<Buffer>> Clear() {
        std::vector<Ref<Buffer>> all;
        all.reserve(m_entries.size());
        for (Entry& entry : m_entries) {
            all.push_back(std::move(entry.buffer));
        }
        m_entries.clear();
        m_uploads = {};
        m_readbacks = {};
        return all;
    }

  private:
    static constexpr uint64_t kQueued = ~uint64_t(0);

    struct Entry {
        Ref<Buffer> buffer;
        uint64_t busyUntil = 0;
    };

    struct Arena {
        Buffer* buffer = nullptr;
        uint64_t cursor = 0;
    };

    bool Idle(const Entry& entry) const {
        return entry.buffer->RefCount() == 1 && entry.busyUntil <= m_completed &&
               entry.buffer.Get() != m_uploads.buffer && entry.buffer.Get() != m_readbacks.buffer;
    }

    Entry& Find(const Buffer* buffer) {
        return *std::find_if(m_entries.begin(), m_entries.end(),
                             [buffer](const Entry& entry) { return entry.buffer.Get() == buffer; });
    }

    Entry& Claim(Memory memory, uint64_t need) {
        Entry* best = nullptr;
        for (Entry& entry : m_entries) {
            if (entry.buffer->GetMemory() == memory && entry.buffer->Bytes() >= need &&
                Idle(entry) && (best == nullptr || entry.buffer->Bytes() < best->buffer->Bytes())) {
                best = &entry;
            }
        }
        if (best != nullptr) {
            return *best;
        }
        m_entries.push_back({m_create(memory, std::max(kMinStagingBytes, std::bit_ceil(need)))});
        return m_entries.back();
    }

    Create m_create;
    std::vector<Entry> m_entries;
    Arena m_uploads;
    Arena m_readbacks;
    uint64_t m_completed = 0;
};

BufferRange StageUpload(StagingPool& staging, const void* data, uint64_t bytes) {
    const BufferRange range = staging.Take(Memory::Upload, bytes);
    std::memcpy(static_cast<std::byte*>(range.GetBuffer().Mapped()) + range.Offset(), data, bytes);
    return range;
}

void Free(std::vector<std::shared_ptr<void>>& freed) {
    for (std::shared_ptr<void>& retired : freed) {
        retired.reset();
    }
}

std::string Describe(const Command& command) {
    return std::visit(
        Overloaded{
            [](const CopyCommand& c) {
                return fmt::format("a copy of {} bytes from buffer \"{}\" to buffer \"{}\"",
                                   c.bytes, c.src->Label(), c.dst->Label());
            },
            [](const TextureCopyCommand& c) {
                return fmt::format("a copy from buffer \"{}\" to level {} of texture \"{}\"",
                                   c.src->Label(), c.level, c.dst->Label());
            },
            [](const FillCommand& c) {
                return fmt::format("a fill of buffer \"{}\"", c.dst->Label());
            },
            [](const UpdateCommand& c) {
                return fmt::format("an update of buffer \"{}\"", c.dst->Label());
            },
            [](const DispatchCommand& c) {
                return fmt::format("{}kernel {} of program \"{}\"",
                                   c.counts != nullptr ? "an indirect dispatch of " : "",
                                   c.kernel->Name(), c.kernel->GetProgram().Label());
            },
            [](const BuildAccelCommand& c) {
                return fmt::format("a build of acceleration structure \"{}\"", c.dst->Label());
            },
            [](const BarrierCommand&) { return std::string("a barrier"); },
            [](const ZoneCommand&) { return std::string("a zone"); },
        },
        command);
}

std::vector<uint32_t> Depths(const CommandBuffer& commands) {
    std::vector<uint32_t> depths(commands.Zones().size());
    uint32_t open = 0;
    for (const Command& command : commands.Commands()) {
        if (const auto* zone = std::get_if<ZoneCommand>(&command)) {
            if (zone->begin) {
                depths[zone->index] = open++;
            } else {
                --open;
            }
        }
    }
    return depths;
}

}

struct Queue::State {
    State(Queue& queue, StagingPool::Create create)
        : queued(queue, true), staging(std::move(create)) {}

    std::mutex submitMutex;
    std::atomic<std::thread::id> exclusive{};
    std::atomic<uint64_t> submitted{0};
    std::atomic<int64_t> openRecordings{0};
    std::atomic<int64_t> batches{0};
    std::atomic<bool> lost{false};

    std::mutex mutex;
    uint64_t taken = 0;
    uint64_t completed = 0;
    CommandEncoder queued;
    StagingPool staging;
    std::vector<std::pair<uint64_t, std::shared_ptr<void>>> retired;
    bool closing = false;

    struct Timed {
        uint64_t value;
        uint64_t frame;
        uint64_t submittedNs;
        std::vector<ZoneId> zones;
        std::vector<uint32_t> depths;
        std::shared_ptr<const Timestamps> timestamps;
    };

    mutable std::mutex zoneMutex;
    std::vector<Timed> timed;
    std::optional<uint32_t> track;
};

void Queue::OpenRecording() {
    ++m_state->openRecordings;
}

void Queue::CloseRecording() {
    --m_state->openRecordings;
}

bool Operation::Ready() const {
    if (queue == nullptr || value == 0) {
        return true;
    }
    queue->Reach(value);
    return queue->Completed() >= value;
}

void Operation::Wait() const {
    if (queue != nullptr) {
        queue->Wait(value);
    }
}

void Operation::Wait(Stream stream) const {
    if (stream.IsNone() || queue == nullptr || value == 0) {
        return;
    }
    if (stream.IsHost()) {
        Wait();
        return;
    }
    if (!queue->SharesStreams()) {
        if (stream.IsDefault()) {
            return;
        }
        throw UnsupportedError("{} shares no streams with CUDA or HIP, so none waits for it",
                               queue->GetDevice().Adapter().alias);
    }
    queue->Reach(value);
    queue->StreamWait(stream.Handle(), value);
}

Download::Download() = default;
Download::~Download() = default;
Download::Download(const Download& other) = default;
Download::Download(Download&& other) noexcept = default;
Download& Download::operator=(const Download& other) = default;
Download& Download::operator=(Download&& other) noexcept = default;

Span<const std::byte> Download::Bytes() const {
    if (!m_staging) {
        return {};
    }
    if (!Ready()) {
        throw Error("a download's bytes aren't here yet; wait for it, or check Ready(), first");
    }
    return {static_cast<const std::byte*>(m_staging->Mapped()) + m_offset, m_bytes};
}

Queue::Queue(Device& device, bool runsKernels)
    : m_device(device), m_runsKernels(runsKernels),
      m_state(std::make_unique<State>(*this, [&device](Memory memory, uint64_t bytes) {
          Ref<Buffer> buffer =
              device.CreateBuffer({.bytes = bytes, .memory = memory, .label = "staging"});
          Device::Adopt(*buffer);
          return buffer;
      })) {}

Queue::~Queue() = default;

CommandEncoder Queue::CreateEncoder() {
    return CommandEncoder(*this, false);
}

Operation Queue::Submit(CommandBuffer&& commands) {
    return SubmitCommands(commands);
}

Operation Queue::Submit(const CommandBuffer& commands) {
    if (!commands.m_reusable) {
        throw Error("a command buffer that isn't reusable was submitted to run again; finish it "
                    "with {{.reusable = true}}, or submit it once with std::move");
    }
    return SubmitCommands(commands);
}

Operation Queue::Mark() {
    return Submit(CreateEncoder().Finish());
}

Operation Queue::SubmitCommands(const CommandBuffer& commands) {
    if (commands.m_queue != this) {
        throw Error("a command buffer recorded for another queue was submitted");
    }
    if (commands.m_reusable) {
        if (commands.m_submission > Finished()) {
            Wait(commands.m_submission);
        }
        Collect(Finished());
    }

    if (m_state->batches > 0 && !m_device.m_synchronous) {
        {
            const std::lock_guard lock(m_state->mutex);
            m_state->queued.Append(commands);
            commands.m_submission = m_state->taken + 1;
        }
        m_device.CountTraffic({.inlineBytes = commands.m_data.size()});
        if (!commands.m_reusable) {
            const_cast<CommandBuffer&>(commands).Close();
        }
        return {this, commands.m_submission};
    }

    commands.m_submission = m_device.m_synchronous ? RunEach(commands) : Run(commands);
    if (!commands.m_reusable) {
        const_cast<CommandBuffer&>(commands).Close();
    }

    Collect(Finished());
    return {this, commands.m_submission};
}

void Queue::BeginBatch() {
    ++m_state->batches;
}

Operation Queue::EndBatch() {
    if (--m_state->batches > 0) {
        const std::lock_guard lock(m_state->mutex);
        return {this, m_state->taken + 1};
    }
    return Mark();
}

void Queue::Reach(uint64_t value) {
    if (value > m_state->submitted) {
        Run(CommandBuffer(*this, true));
    }
}

void Queue::Follow(Stream stream) {
    if (!stream.IsCompute()) {
        return;
    }
    if (!SharesStreams()) {
        if (stream.IsDefault()) {
            return;
        }
        throw UnsupportedError("{} shares no streams with CUDA or HIP, so it follows none",
                               m_device.Adapter().alias);
    }
    const std::unique_lock submitLock = LockSubmissions();
    FollowStream(stream.Handle());
}

void Queue::Follow(const Operation& operation) {
    Queue* other = operation.queue;
    if (other == nullptr || other == this || operation.value == 0) {
        return;
    }
    if (&other->m_device == &m_device) {
        other->Reach(operation.value);
        const std::unique_lock submitLock = LockSubmissions();
        FollowQueue(*other, operation.value);
        return;
    }
    if (other->m_device.Adapter().uuid == m_device.Adapter().uuid) {
        if (void* stream = NativeStream(); stream != nullptr && other->SharesStreams()) {
            operation.Wait(Stream::Compute(stream));
            return;
        }
        if (void* stream = other->NativeStream(); stream != nullptr && SharesStreams()) {
            other->Reach(operation.value);
            Follow(Stream::Compute(stream));
            return;
        }
    }
    operation.Wait();
}

std::unique_lock<std::mutex> Queue::LockSubmissions() {
    if (m_state->exclusive.load() == std::this_thread::get_id()) {
        throw Error("{} got work inside Exclusive, which holds its queue; submit it after "
                    "Exclusive returns",
                    m_device.Adapter().alias);
    }
    return std::unique_lock(m_state->submitMutex);
}

void Queue::Exclusive(const std::function<void()>& use) {
    const std::unique_lock submitLock = LockSubmissions();
    m_state->exclusive = std::this_thread::get_id();
    try {
        use();
    } catch (...) {
        m_state->exclusive = std::thread::id();
        throw;
    }
    m_state->exclusive = std::thread::id();
}

Operation Queue::RunNative(const void* native) {
    const uint64_t value = Run(CommandBuffer(*this, true), native);
    if (m_device.m_synchronous) {
        Wait(value);
    }
    Collect(Finished());
    return {this, value};
}

void Queue::FollowQueue(Queue& other, uint64_t value) {
    Operation{&other, value}.Wait();
}

void Queue::ExecuteNative(const CommandBuffer&, uint64_t, const void*) {
    throw UnsupportedError("{} runs no native commands", m_device.Adapter().alias);
}

void Queue::FollowStream(void*) {
    throw UnsupportedError("{} shares no streams with CUDA or HIP", m_device.Adapter().alias);
}

void Queue::StreamWait(void*, uint64_t) {
    throw UnsupportedError("{} shares no streams with CUDA or HIP", m_device.Adapter().alias);
}

uint64_t Queue::Run(const CommandBuffer& commands, const void* native) {
    ERNIE_ZONE("Queue/Run");
    State& state = *m_state;
    const std::unique_lock submitLock = LockSubmissions();
    if (state.lost) {
        throw DeviceLostError("{} was lost; no more work runs on it", m_device.Adapter().alias);
    }

    uint64_t value = 0;
    CommandBuffer queued = [&] {
        const std::lock_guard lock(state.mutex);
        value = ++state.taken;
        state.staging.Submitted(value);
        if (m_runsKernels) {
            state.queued.Fill(m_device.Debug().Submission(), uint32_t(value));
        }
        return state.queued.Finish();
    }();
    m_device.CountTraffic({.inlineBytes = commands.m_data.size(), .submissions = 1});

    if (native == nullptr && !queued.m_zones.empty()) {
        CommandEncoder merged(*this, true);
        merged.Append(queued);
        merged.Append(commands);
        queued = CommandBuffer(*this, true);
        return Launch(queued, merged.Finish(), value, native);
    }
    return Launch(queued, commands, value, native);
}

uint64_t Queue::Launch(const CommandBuffer& queued, const CommandBuffer& commands, uint64_t value,
                       const void* native) {
    State& state = *m_state;
    try {
        if (native != nullptr) {
            ExecuteNative(queued, value, native);
            state.submitted = value;
            return value;
        }
        std::shared_ptr<Timestamps> timestamps = TimestampsFor(commands);
        Execute(queued, commands, timestamps.get(), value);
        if (timestamps) {
            const std::lock_guard lock(state.zoneMutex);
            state.timed.push_back({value, profiler::CurrentFrame(), profiler::Now(),
                                   commands.m_zones, Depths(commands), std::move(timestamps)});
        }
    } catch (...) {
        state.lost = true;
        throw;
    }

    state.submitted = value;
    return value;
}

uint64_t Queue::RunEach(const CommandBuffer& commands) {
    uint64_t value = 0;
    for (const Command& command : commands.m_commands) {
        if (std::holds_alternative<BarrierCommand>(command) ||
            std::holds_alternative<ZoneCommand>(command)) {
            continue;
        }
        const CommandBuffer one = Isolate(commands, command);
        value = Run(one);
        try {
            Wait(value);
        } catch (const DeviceLostError& lost) {
            throw DeviceLostError("{}, running {}", lost.what(), Describe(command));
        }
    }
    return value != 0 ? value : Run(CommandBuffer(*this, true));
}

CommandBuffer Queue::Isolate(const CommandBuffer& commands, const Command& command) {
    CommandBuffer one(*this, true);
    Command& alone = one.m_commands.emplace_back(command);
    const auto move = [&](size_t& data, size_t bytes) {
        one.m_data.assign(commands.m_data.begin() + ptrdiff_t(data),
                          commands.m_data.begin() + ptrdiff_t(data + RoundUp(bytes, 4)));
        data = 0;
    };
    std::visit(Overloaded{
                   [&](UpdateCommand& c) { move(c.data, c.bytes); },
                   [&](DispatchCommand& c) { move(c.data, c.paramBytes); },
                   [](auto&) {},
               },
               alone);
    return one;
}

std::shared_ptr<Queue::Timestamps> Queue::TimestampsFor(const CommandBuffer& commands) {
    if (commands.m_zones.empty() || !m_device.m_profiling || !profiler::Enabled()) {
        return nullptr;
    }

    const auto count = uint32_t(2 * commands.m_zones.size());
    if (!commands.m_reusable) {
        return CreateTimestamps(count);
    }
    std::shared_ptr<void>& kept = commands.m_recorded.timestamps;
    if (!kept) {
        kept = CreateTimestamps(count);
    }
    return std::static_pointer_cast<Timestamps>(kept);
}

void Queue::CollectZones(uint64_t finished) const {
    State& state = *m_state;
    std::vector<State::Timed> read;
    {
        const std::lock_guard lock(state.zoneMutex);
        const auto done = std::ranges::find_if(
            state.timed, [finished](const State::Timed& run) { return run.value > finished; });
        if (done == state.timed.begin()) {
            return;
        }
        if (!state.track) {
            state.track = profiler::AddGpuTrack(m_device.Adapter().alias);
        }
        const uint32_t track = *state.track;

        std::vector<uint64_t> ns;
        for (auto run = state.timed.begin(); run != done; ++run) {
            ns.resize(2 * run->zones.size());
            if (!run->timestamps->Read(ns)) {
                const uint64_t origin = ns[0];
                for (uint64_t& time : ns) {
                    time = time - origin + run->submittedNs;
                }
            }
            for (size_t i = 0; i < run->zones.size(); ++i) {
                profiler::AddZone(run->frame,
                                  {run->zones[i], track, run->depths[i], ns[2 * i], ns[2 * i + 1]});
            }
        }

        read.assign(std::make_move_iterator(state.timed.begin()), std::make_move_iterator(done));
        state.timed.erase(state.timed.begin(), done);
    }
}

void Queue::Write(BufferRange dst, const void* data) {
    m_device.Require(dst.GetBuffer(), "Write");
    State& state = *m_state;
    const std::lock_guard lock(state.mutex);
    constexpr uint64_t kInlineWriteBytes = 4096;
    if (dst.Bytes() <= kInlineWriteBytes && dst.Offset() % 4 == 0 && dst.Bytes() % 4 == 0) {
        state.queued.Update(dst, data);
    } else if (dst.Bytes() > 0) {
        state.queued.Copy(StageUpload(state.staging, data, dst.Bytes()), dst);
    }
    m_device.CountTraffic({.uploadBytes = dst.Bytes()});
}

BufferRange Queue::Upload(const void* data, uint64_t bytes) {
    State& state = *m_state;
    const std::lock_guard lock(state.mutex);
    m_device.CountTraffic({.uploadBytes = bytes});
    return StageUpload(state.staging, data, bytes);
}

void Queue::Write(Texture& dst, uint32_t level, const void* data) {
    m_device.Require(dst, "Write");
    const uint64_t bytes = dst.LevelBytes(level);
    State& state = *m_state;
    const std::lock_guard lock(state.mutex);
    state.queued.Copy(StageUpload(state.staging, data, bytes), dst, level);
    m_device.CountTraffic({.uploadBytes = bytes});
}

Download Queue::ReadAsync(BufferRange src) {
    m_device.Require(src.GetBuffer(), "ReadAsync");
    Download download;
    if (src.Bytes() == 0) {
        return download;
    }

    State& state = *m_state;
    const std::lock_guard lock(state.mutex);
    const BufferRange staging = state.staging.Take(Memory::Readback, src.Bytes());
    state.queued.Copy(src, staging);
    m_device.CountTraffic({.readbackBytes = src.Bytes()});

    static_cast<Operation&>(download) = {this, state.taken + 1};
    download.m_device = Ref<Device>(&m_device);
    download.m_staging = Ref<Buffer>(&staging.GetBuffer());
    download.m_offset = staging.Offset();
    download.m_bytes = src.Bytes();
    return download;
}

void Queue::Read(BufferRange src, void* data) {
    const Download download = ReadAsync(src);
    download.Wait();
    const std::span<const std::byte> bytes = download.Bytes();
    std::memcpy(data, bytes.data(), bytes.size());
}

uint64_t Queue::Submitted() const {
    return m_state->submitted;
}

uint64_t Queue::Completed() const {
    const uint64_t finished = Finished();
    Collect(finished);
    return finished;
}

void Queue::WaitIdle() {
    uint64_t value = 0;
    {
        const std::lock_guard lock(m_state->mutex);
        value = m_state->queued.m_recording.Empty() ? m_state->taken : m_state->taken + 1;
    }
    Wait(value);
}

void Queue::Wait(uint64_t value) {
    if (value == 0) {
        return;
    }
    Reach(value);
    if (Finished() < value) {
        m_device.CountTraffic({.hostWaits = 1});
        WaitFor(value);
    }

    Collect(Finished());
    if (!m_runsKernels) {
        return;
    }
    if (std::optional<std::string> failed = m_device.Debug().TakeFailures(value)) {
        throw Error(*failed);
    }
}

void Queue::Collect(uint64_t finished) const {
    State& state = *m_state;
    std::vector<std::shared_ptr<void>> freed;
    std::vector<Ref<Buffer>> dropped;
    {
        const std::lock_guard lock(state.mutex);
        state.completed = std::max(state.completed, finished);
        const auto done = std::stable_partition(
            state.retired.begin(), state.retired.end(),
            [&](const auto& retired) { return retired.first > state.completed; });
        for (auto it = done; it != state.retired.end(); ++it) {
            freed.push_back(std::move(it->second));
        }
        state.retired.erase(done, state.retired.end());

        if (m_runsKernels) {
            m_device.Debug().Read(state.completed, state.completed >= state.taken);
        }
        dropped = state.staging.Trim(state.completed);
    }

    CollectZones(finished);
    Free(freed);
}

void Queue::Retire(Resource& resource) {
    State& state = *m_state;
    uint64_t value = 0;
    {
        const std::lock_guard lock(state.mutex);
        const bool pending = state.openRecordings > 0 || !state.queued.m_recording.Empty();
        value = pending ? state.taken + 1 : state.taken;
    }

    Retire(
        std::shared_ptr<void>(&resource, [](void* freed) { delete static_cast<Resource*>(freed); }),
        value);
}

void Queue::Retire(std::shared_ptr<void> retired, const Operation& until) {
    Retire(std::move(retired), until.value);
}

void Queue::Retire(std::shared_ptr<void> retired, uint64_t value) {
    State& state = *m_state;
    {
        const std::lock_guard lock(state.mutex);
        if (!state.closing && value > state.completed) {
            state.retired.emplace_back(value, std::move(retired));
            return;
        }
    }
    retired.reset();
}

void Queue::Close() noexcept {
    State& state = *m_state;
    {
        const std::lock_guard lock(state.mutex);
        state.closing = true;
    }

    try {
        if (state.submitted > 0) {
            WaitFor(state.submitted);
            CollectZones(state.submitted);
        }
    } catch (...) {  // NOLINT(bugprone-empty-catch)
    }

    {
        std::vector<State::Timed> timed;
        const std::lock_guard lock(state.zoneMutex);
        timed.swap(state.timed);
    }

    for (;;) {
        std::vector<std::shared_ptr<void>> freed;
        std::vector<Ref<Buffer>> staging;
        {
            const std::lock_guard lock(state.mutex);
            state.queued.Finish();
            staging = state.staging.Clear();
            for (auto& [value, retired] : state.retired) {
                freed.push_back(std::move(retired));
            }
            state.retired.clear();
        }
        if (freed.empty() && staging.empty()) {
            return;
        }
        Free(freed);
        staging.clear();
    }
}

}
