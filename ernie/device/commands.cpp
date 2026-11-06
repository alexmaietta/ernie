// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/commands.h"

#include <algorithm>
#include <cstring>
#include <utility>

#include "ernie/device/device.h"
#include "ernie/device/kernels/shared.h"
#include "ernie/device/queue.h"
#include "ernie/util/error.h"
#include "ernie/util/log.h"

namespace ernie::device {
namespace {

void RequireWords(BufferRange range, const char* what) {
    if (range.Offset() % 4 != 0 || range.Bytes() % 4 != 0) {
        throw Error("{} of buffer \"{}\" has offset {} and size {}; both must be "
                    "multiples of 4",
                    what, range.GetBuffer().Label(), range.Offset(), range.Bytes());
    }
}

constexpr uint32_t kMaxGroups = kMaxGroupsPerAxis;

constexpr bool Covers(Size3 folded, uint32_t count) {
    return folded.x <= kMaxGroups && folded.y <= kMaxGroups && folded.z <= kMaxGroups &&
           uint64_t(folded.x) * folded.y * folded.z >= count;
}

static_assert(FoldGroups(65536) == Size3{32768, 2, 1});
static_assert(Covers(FoldGroups(65536), 65536));
static_assert(Covers(FoldGroups(70001), 70001));
static_assert(Covers(FoldGroups(uint32_t(kMaxGroups) * kMaxGroups),
                     uint32_t(kMaxGroups) * kMaxGroups));
static_assert(FoldGroups(~uint32_t(0)).z == 2 && Covers(FoldGroups(~uint32_t(0)), ~uint32_t(0)));

Size3 Grid(const Kernel& kernel, Size3 groups) {
    if (groups.y > kMaxGroups || groups.z > kMaxGroups) {
        throw Error("Dispatch of kernel {} over {} x {} x {} groups; y and z take at most {}",
                    kernel.Name(), groups.x, groups.y, groups.z, kMaxGroups);
    }
    if (groups.x <= kMaxGroups) {
        return groups;
    }
    if (groups.y != 1 || groups.z != 1) {
        throw Error("Dispatch of kernel {} over {} x {} x {} groups; only x takes more than {}, "
                    "and only when y and z are 1",
                    kernel.Name(), groups.x, groups.y, groups.z, kMaxGroups);
    }
    if (!kernel.Foldable()) {
        throw Error("Dispatch of kernel {} over {} groups, which fold into more axes; it reads "
                    "SV_DispatchThreadID or SV_GroupID, which folding changes: index it with "
                    "LinearThreadIndex()",
                    kernel.Name(), groups.x);
    }
    return FoldGroups(groups.x);
}

void RequireCounts(const Kernel& kernel, BufferRange counts, Size3 groups) {
    if (!Has(counts.GetBuffer().Usage(), BufferUsage::Indirect)) {
        throw Error("Dispatch of kernel {} reads its counts from buffer \"{}\", which lacks "
                    "BufferUsage::Indirect",
                    kernel.Name(), counts.GetBuffer().Label());
    }
    if (counts.Offset() % 4 != 0 || counts.Bytes() < 12) {
        throw Error("Dispatch of kernel {} reads its counts at offset {} of {} bytes of buffer "
                    "\"{}\"; they are 12 bytes at a multiple of 4",
                    kernel.Name(), counts.Offset(), counts.Bytes(), counts.GetBuffer().Label());
    }
    if (groups.x > kMaxGroups || groups.y > kMaxGroups || groups.z > kMaxGroups) {
        throw Error("Dispatch of kernel {} over at most {} x {} x {} groups read; each count "
                    "is at most {}, as they don't fold",
                    kernel.Name(), groups.x, groups.y, groups.z, kMaxGroups);
    }
}

}

CommandBuffer::CommandBuffer(Queue& queue, bool own) : m_queue(&queue), m_open(!own) {
    if (!own) {
        m_device = Ref<Device>(&queue.GetDevice());
        queue.OpenRecording();
    }
}

CommandBuffer::CommandBuffer(CommandBuffer&& other) noexcept
    : m_queue(other.m_queue), m_device(std::move(other.m_device)),
      m_commands(std::move(other.m_commands)), m_data(std::move(other.m_data)),
      m_zones(std::move(other.m_zones)), m_open(std::exchange(other.m_open, false)),
      m_reusable(other.m_reusable), m_held(std::move(other.m_held)),
      m_submission(other.m_submission), m_recorded(std::move(other.m_recorded)) {}

CommandBuffer& CommandBuffer::operator=(CommandBuffer&& other) noexcept {
    if (this != &other) {
        Close();
        Release();
        m_queue = other.m_queue;
        m_device = std::move(other.m_device);
        m_commands = std::move(other.m_commands);
        m_data = std::move(other.m_data);
        m_zones = std::move(other.m_zones);
        m_open = std::exchange(other.m_open, false);
        m_reusable = other.m_reusable;
        m_held = std::move(other.m_held);
        m_submission = other.m_submission;
        m_recorded = std::move(other.m_recorded);
    }
    return *this;
}

CommandBuffer::~CommandBuffer() {
    Close();
    Release();
}

void CommandBuffer::Release() {
    if (m_recorded.native || m_recorded.timestamps) {
        m_queue->Retire(std::make_shared<Recorded>(std::exchange(m_recorded, {})), m_submission);
    }
}

void CommandBuffer::Close() {
    if (std::exchange(m_open, false)) {
        m_queue->CloseRecording();
    }
}

void CommandBuffer::HoldResources() {
    const auto hold = [&](Resource* resource) {
        if (resource != nullptr &&
            std::none_of(m_held.begin(), m_held.end(), [resource](const Ref<Resource>& held) {
                return held.Get() == resource;
            })) {
            m_held.emplace_back(resource);
        }
    };
    for (const Command& command : m_commands) {
        std::visit(Overloaded{
                       [&](const CopyCommand& c) {
                           hold(c.src);
                           hold(c.dst);
                       },
                       [&](const TextureCopyCommand& c) {
                           hold(c.src);
                           hold(c.dst);
                       },
                       [&](const FillCommand& c) { hold(c.dst); },
                       [&](const UpdateCommand& c) { hold(c.dst); },
                       [&](const DispatchCommand& c) {
                           hold(&c.kernel->GetProgram());
                           hold(c.counts);
                       },
                       [&](const BuildAccelCommand& c) {
                           hold(c.dst);
                           hold(c.scratch);
                           ForEachBuffer(c.input,
                                         [&](BufferRange range) { hold(&range.GetBuffer()); });
                       },
                       [&](const BarrierCommand&) {},
                       [&](const ZoneCommand&) {},
                   },
                   command);
    }
}

CommandEncoder::CommandEncoder(Queue& queue, bool own) : m_recording(queue, own), m_own(own) {}

void CommandEncoder::Append(const CommandBuffer& commands) {
    const auto zones = uint32_t(m_recording.m_zones.size());
    m_recording.m_zones.insert(m_recording.m_zones.end(), commands.m_zones.begin(),
                               commands.m_zones.end());
    m_recording.m_commands.push_back(BarrierCommand{});
    for (Command command : commands.m_commands) {
        std::visit(Overloaded{
                       [&](UpdateCommand& c) {
                           c.data = Store(commands.m_data.data() + c.data, c.bytes, 4);
                       },
                       [&](DispatchCommand& c) {
                           c.data = Store(commands.m_data.data() + c.data, c.paramBytes, 8);
                       },
                       [&](ZoneCommand& c) { c.index += zones; },
                       [](auto&) {},
                   },
                   command);
        m_recording.m_commands.push_back(command);
    }
    m_barrier = true;
}

void CommandEncoder::Require(const Resource& resource, const char* what) const {
    m_recording.m_queue->GetDevice().Require(resource, what);
}

size_t CommandEncoder::Store(const void* data, size_t bytes, size_t alignment) {
    std::vector<std::byte>& stored = m_recording.m_data;
    const size_t at = RoundUp(stored.size(), alignment);
    stored.resize(at + RoundUp(bytes, 4));
    if (bytes > 0) {
        std::memcpy(stored.data() + at, data, bytes);
    }
    return at;
}

void CommandEncoder::Record(const Command& command) {
    if (!m_concurrent || m_barrier) {
        m_recording.m_commands.push_back(BarrierCommand{});
        m_barrier = false;
    }
    m_recording.m_commands.push_back(command);
}

void CommandEncoder::Copy(BufferRange src, BufferRange dst) {
    Require(src.GetBuffer(), "Copy");
    Require(dst.GetBuffer(), "Copy");
    if (dst.Bytes() < src.Bytes()) {
        throw Error("Copy of {} bytes into {} bytes of buffer \"{}\"", src.Bytes(), dst.Bytes(),
                    dst.GetBuffer().Label());
    }
    const bool sameBuffer = &src.GetBuffer() == &dst.GetBuffer();
    if (sameBuffer && src.Offset() < dst.Offset() + src.Bytes() &&
        dst.Offset() < src.Offset() + src.Bytes()) {
        throw Error("Copy within buffer \"{}\" overlaps itself", src.GetBuffer().Label());
    }
    if (src.Bytes() > 0) {
        const auto staging = [](const Buffer& buffer) {
            return buffer.GetMemory() == Memory::Upload || buffer.GetMemory() == Memory::Readback;
        };
        if (!staging(src.GetBuffer()) && !staging(dst.GetBuffer())) {
            m_recording.m_queue->GetDevice().CountTraffic({.copyBytes = src.Bytes()});
        }
        Record(CopyCommand{&src.GetBuffer(), src.Offset(), &dst.GetBuffer(), dst.Offset(),
                           src.Bytes()});
    }
}

void CommandEncoder::Copy(BufferRange src, Texture& dst, uint32_t level) {
    Require(src.GetBuffer(), "Copy");
    Require(dst, "Copy");
    if (level >= dst.Desc().levels) {
        throw Error("Copy into level {} of texture \"{}\", which has {}", level, dst.Label(),
                    dst.Desc().levels);
    }
    const uint64_t bytes = dst.LevelBytes(level);
    if (src.Bytes() < bytes) {
        throw Error("Copy of {} bytes into level {} of texture \"{}\", which takes {}", src.Bytes(),
                    level, dst.Label(), bytes);
    }
    if (src.Offset() % 16 != 0) {
        throw Error("Copy into texture \"{}\" from offset {} of buffer \"{}\"; it takes a "
                    "multiple of 16",
                    dst.Label(), src.Offset(), src.GetBuffer().Label());
    }
    Record(TextureCopyCommand{&src.GetBuffer(), src.Offset(), &dst, level});
}

void CommandEncoder::Fill(BufferRange dst, uint32_t value) {
    Require(dst.GetBuffer(), "Fill");
    RequireWords(dst, "Fill");
    if (dst.Bytes() > 0) {
        Record(FillCommand{&dst.GetBuffer(), dst.Offset(), dst.Bytes(), value});
    }
}

void CommandEncoder::Update(BufferRange dst, const void* data) {
    Require(dst.GetBuffer(), "Update");
    RequireWords(dst, "Update");
    if (dst.Bytes() > kMaxUpdateBytes) {
        throw Error("Update of {} bytes of buffer \"{}\"; it writes at most {}; Queue::Write "
                    "stages larger writes",
                    dst.Bytes(), dst.GetBuffer().Label(), kMaxUpdateBytes);
    }
    if (dst.Bytes() > 0) {
        Record(UpdateCommand{&dst.GetBuffer(), dst.Offset(), dst.Bytes(),
                             Store(data, dst.Bytes(), 4)});
    }
}

void CommandEncoder::BuildAccel(Accel& dst, const AccelInput& input, BufferRange scratch,
                                bool update) {
    Require(dst, "BuildAccel");
    Require(scratch.GetBuffer(), "BuildAccel");
    ForEachBuffer(input, [&](BufferRange range) {
        Require(range.GetBuffer(), "BuildAccel");
        if (!Has(range.GetBuffer().Usage(), BufferUsage::AccelBuildInput)) {
            throw Error("BuildAccel of \"{}\" reads buffer \"{}\", which lacks "
                        "BufferUsage::AccelBuildInput",
                        dst.Label(), range.GetBuffer().Label());
        }
    });
    if (GetLevel(input) != dst.Level()) {
        const bool top = dst.Level() == AccelLevel::Top;
        throw Error("BuildAccel of {} \"{}\" of {}", top ? "TLAS" : "BLAS", dst.Label(),
                    top ? "geometry" : "instances");
    }
    if (update && !Refittable(input)) {
        throw Error("BuildAccel updating BLAS \"{}\", which wasn't made refittable", dst.Label());
    }

    const AccelSizes sizes = m_recording.m_queue->GetDevice().AccelBuildSizes(input);
    const uint64_t needed = update ? sizes.updateScratch : sizes.scratch;
    if (!Has(scratch.GetBuffer().Usage(), BufferUsage::AccelStorage) || scratch.Bytes() < needed) {
        throw Error(
            "BuildAccel of \"{}\" with {} bytes of scratch in buffer \"{}\"; it takes {} of "
            "BufferUsage::AccelStorage",
            dst.Label(), scratch.Bytes(), scratch.GetBuffer().Label(), needed);
    }
    if (dst.Storage().Bytes() < sizes.storage) {
        throw Error("BuildAccel of \"{}\", of {} bytes, of an input that takes {}", dst.Label(),
                    dst.Storage().Bytes(), sizes.storage);
    }

    Record(BuildAccelCommand{&dst, input, &scratch.GetBuffer(), scratch.Offset(), update});
}

void CommandEncoder::Dispatch(Kernel& kernel, const void* params, size_t bytes, Size3 groups,
                              std::optional<BufferRange> counts) {
    Require(kernel.GetProgram(), "Dispatch");
    if (bytes != kernel.ParamBytes()) {
        throw Error("Dispatch of kernel {} with {} bytes of parameters; it takes {}", kernel.Name(),
                    bytes, kernel.ParamBytes());
    }
    if (counts) {
        Require(counts->GetBuffer(), "Dispatch");
        RequireCounts(kernel, *counts, groups);
    } else {
        groups = Grid(kernel, groups);
    }

    if (groups.x > 0 && groups.y > 0 && groups.z > 0) {
        Record(DispatchCommand{.kernel = &kernel,
                               .groups = groups,
                               .data = Store(params, bytes, 8),
                               .paramBytes = uint32_t(bytes),
                               .counts = counts ? &counts->GetBuffer() : nullptr,
                               .countsOffset = counts ? counts->Offset() : 0});
    }
}

void CommandEncoder::Concurrent(bool concurrent) {
    m_concurrent = concurrent;
    m_barrier = true;
}

void CommandEncoder::Barrier() {
    m_barrier = true;
}

void CommandEncoder::BeginZone(ZoneId zone) {
    const auto index = uint32_t(m_recording.m_zones.size());
    m_recording.m_zones.push_back(zone);
    m_recording.m_commands.push_back(ZoneCommand{index, true});
    m_open.push_back(index);
}

void CommandEncoder::EndZone() {
    if (m_open.empty()) {
        throw Error("EndZone with no zone open");
    }
    m_recording.m_commands.push_back(ZoneCommand{m_open.back(), false});
    m_open.pop_back();
}

CommandBuffer CommandEncoder::Finish(const FinishDesc& desc) {
    if (!m_open.empty()) {
        throw Error("Finish with zone \"{}\" still open",
                    m_recording.m_zones[m_open.back()].Name());
    }
    m_concurrent = false;
    m_barrier = false;

    CommandBuffer finished = std::exchange(m_recording, CommandBuffer(*m_recording.m_queue, m_own));
    if (desc.reusable) {
        finished.m_reusable = true;
        finished.Close();
        finished.HoldResources();
    }
    return finished;
}

ZoneScope::~ZoneScope() {
    try {
        m_encoder.EndZone();
    } catch (const Error& error) {
        Log(LogLevel::Error, "{}", error.what());
    }
}

}
