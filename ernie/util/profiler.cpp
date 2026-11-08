// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/profiler.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <map>
#include <mutex>
#include <unordered_map>
#include <utility>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include <fmt/format.h>
#include <nvtx3/nvToolsExt.h>

#include "ernie/util/environment.h"
#include "ernie/util/text.h"

namespace ernie::profiler {
namespace {

struct Names {
    std::mutex mutex;
    std::deque<std::string> names;
    std::unordered_map<std::string_view, uint32_t> indices;
};

Names& GetNames() {
    static Names names;
    return names;
}

struct TrackEntry {
    Track track;
    uint32_t osThread = 0;
};

struct State {
    std::mutex mutex;
    Frame current;
    std::deque<Frame> frames;
    size_t history = 300;
    std::vector<TrackEntry> tracks;
};

State& GetState() {
    static State state;
    return state;
}

std::atomic<bool> g_recording{false};
nvtxDomainHandle_t g_domain = nullptr;
thread_local uint32_t t_depth = 0;

uint32_t OsThread() {
#if defined(_WIN32)
    return uint32_t(GetCurrentThreadId());
#elif defined(__linux__)
    return uint32_t(syscall(SYS_gettid));
#else
    return 0;
#endif
}

uint32_t ThreadTrack() {
    thread_local const uint32_t track = [] {
        State& state = GetState();
        const std::lock_guard lock(state.mutex);
        const auto index = uint32_t(state.tracks.size());
        state.tracks.push_back({{fmt::format("Thread {}", index), false}, OsThread()});
        return index;
    }();
    return track;
}

void DropOldFrames(State& state) {
    while (state.frames.size() > state.history) {
        state.frames.pop_front();
    }
}

bool StartNvtx() {
    if (Environment("ERNIE_NVTX") == "0") {
        return false;
    }
    nvtxInitialize(nullptr);
    g_domain = nvtxDomainCreateA("ernie");
    if (g_domain == nullptr) {
        return false;
    }

    State& state = GetState();
    const std::lock_guard lock(state.mutex);
    for (const TrackEntry& entry : state.tracks) {
        if (!entry.track.gpu) {
            nvtxNameOsThreadA(entry.osThread, entry.track.name.c_str());
        }
    }
    return true;
}

bool NvtxActive() {
    static const bool active = StartNvtx();
    return active;
}

void PushRange(ZoneId zone) {
    thread_local std::vector<nvtxStringHandle_t> registered;
    if (zone.Index() >= registered.size()) {
        registered.resize(zone.Index() + 1);
    }
    nvtxStringHandle_t& handle = registered[zone.Index()];
    if (handle == nullptr) {
        handle = nvtxDomainRegisterStringA(g_domain, std::string(zone.Name()).c_str());
    }

    nvtxEventAttributes_t attributes{};
    attributes.version = NVTX_VERSION;
    attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attributes.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
    attributes.message.registered = handle;
    nvtxDomainRangePushEx(g_domain, &attributes);
}

double DurationMs(uint64_t startNs, uint64_t endNs) {
    return double(endNs - startNs) * 1e-6;
}

void Gather(std::vector<ReportNode>& roots, std::vector<const Zone*> zones) {
    std::ranges::sort(zones, [](const Zone* a, const Zone* b) {
        return a->startNs != b->startNs ? a->startNs < b->startNs : a->depth < b->depth;
    });

    std::vector<ReportNode*> path;
    for (const Zone* zone : zones) {
        path.resize(std::min<size_t>(zone->depth, path.size()));
        std::vector<ReportNode>& siblings = path.empty() ? roots : path.back()->children;
        const std::string_view name = zone->id.Name();
        auto node = std::ranges::find(siblings, name, &ReportNode::name);
        if (node == siblings.end()) {
            siblings.emplace_back().name = name;
            node = siblings.end() - 1;
        }

        const double ms = DurationMs(zone->startNs, zone->endNs);
        node->minMs = node->calls == 0 ? ms : std::min(node->minMs, ms);
        node->maxMs = std::max(node->maxMs, ms);
        node->ms += ms;
        node->calls += 1;
        path.push_back(&*node);
    }
}

void Average(std::vector<ReportNode>& nodes, double frames, double frameMs) {
    for (ReportNode& node : nodes) {
        node.ms /= frames;
        node.calls /= frames;
        node.percent = frameMs > 0 ? 100 * node.ms / frameMs : 0;
        Average(node.children, frames, frameMs);
    }
}

std::string TrackName(const std::vector<Track>& tracks, uint32_t index) {
    return index < tracks.size() ? tracks[index].name : fmt::format("Track {}", index);
}

bool IsGpuTrack(const std::vector<Track>& tracks, uint32_t index) {
    return index < tracks.size() && tracks[index].gpu;
}

void AddLines(std::string& text, const std::vector<ReportNode>& nodes, const ReportFilter& filter,
              uint32_t depth) {
    std::vector<const ReportNode*> order;
    order.reserve(nodes.size());
    for (const ReportNode& node : nodes) {
        order.push_back(&node);
    }
    if (filter.byTime) {
        std::ranges::stable_sort(
            order, [](const ReportNode* a, const ReportNode* b) { return a->ms > b->ms; });
    }

    for (const ReportNode* node : order) {
        if (node->ms < filter.minMs) {
            continue;
        }
        text += fmt::format("{:<56} {:12.3f} {:9.2f}%\n",
                            std::string(size_t(2) * (depth + 1), ' ') + node->name, node->ms,
                            node->percent);
        if (depth + 1 < filter.depth) {
            AddLines(text, node->children, filter, depth + 1);
        }
    }
}

std::string Quote(std::string_view text) {
    std::string quoted = "\"";
    for (const char c : text) {
        if (c == '"' || c == '\\') {
            quoted += '\\';
        }
        quoted += static_cast<unsigned char>(c) < 32 ? ' ' : c;
    }
    return quoted + "\"";
}

std::string ZonesJson(const std::vector<ReportNode>& nodes) {
    std::vector<std::string> zones;
    for (const ReportNode& node : nodes) {
        zones.push_back(fmt::format(
            R"({{"name":{},"ms":{:.6g},"calls":{:.6g},"percent":{:.6g},"minMs":{:.6g},"maxMs":{:.6g},"children":{}}})",
            Quote(node.name), node.ms, node.calls, node.percent, node.minMs, node.maxMs,
            ZonesJson(node.children)));
    }
    return "[" + Join(zones, ",") + "]";
}

std::string SpanEvent(std::string_view name, int pid, std::string_view tid, double start,
                      double duration) {
    return fmt::format(R"({{"name":{},"ph":"X","pid":{},"tid":{},"ts":{:.3f},"dur":{:.3f}}})",
                       Quote(name), pid, tid, start, duration);
}

std::string MetaEvent(std::string_view kind, int pid, std::string_view tid, std::string_view name) {
    return fmt::format(R"({{"name":"{}","ph":"M","pid":{}{},"args":{{"name":{}}}}})", kind, pid,
                       tid.empty() ? "" : fmt::format(R"(,"tid":{})", tid), Quote(name));
}

}

ZoneId ZoneId::Named(std::string_view name) {
    Names& names = GetNames();
    const std::lock_guard lock(names.mutex);
    if (const auto found = names.indices.find(name); found != names.indices.end()) {
        return ZoneId(found->second);
    }

    const auto index = uint32_t(names.names.size());
    names.indices.emplace(names.names.emplace_back(name), index);
    return ZoneId(index);
}

std::string_view ZoneId::Name() const {
    Names& names = GetNames();
    const std::lock_guard lock(names.mutex);
    return names.names.at(m_index);
}

void SetEnabled(bool enabled) {
    g_recording.store(enabled, std::memory_order_relaxed);
}

bool Enabled() {
    return g_recording.load(std::memory_order_relaxed);
}

uint64_t Now() {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count());
}

void BeginFrame() {
    State& state = GetState();
    const uint64_t now = Now();
    const std::lock_guard lock(state.mutex);
    if (state.current.startNs != 0) {
        state.current.endNs = now;
        const uint64_t next = state.current.index + 1;
        state.frames.push_back(std::exchange(state.current, Frame{.index = next}));
        DropOldFrames(state);
    }
    state.current.startNs = now;
}

uint64_t CurrentFrame() {
    State& state = GetState();
    const std::lock_guard lock(state.mutex);
    return state.current.index;
}

std::vector<Frame> Frames() {
    State& state = GetState();
    const std::lock_guard lock(state.mutex);
    return {state.frames.begin(), state.frames.end()};
}

void SetHistory(size_t frames) {
    State& state = GetState();
    const std::lock_guard lock(state.mutex);
    state.history = std::max<size_t>(frames, 1);
    DropOldFrames(state);
}

void Clear() {
    State& state = GetState();
    const std::lock_guard lock(state.mutex);
    state.frames.clear();
    state.current.zones.clear();
}

void SetThreadName(std::string_view name) {
    const uint32_t track = ThreadTrack();
    const bool nvtx = NvtxActive();
    State& state = GetState();
    const std::lock_guard lock(state.mutex);
    TrackEntry& entry = state.tracks[track];
    entry.track.name = name;
    if (nvtx) {
        nvtxNameOsThreadA(entry.osThread, entry.track.name.c_str());
    }
}

std::vector<Track> Tracks() {
    State& state = GetState();
    const std::lock_guard lock(state.mutex);
    std::vector<Track> tracks;
    tracks.reserve(state.tracks.size());
    for (const TrackEntry& entry : state.tracks) {
        tracks.push_back(entry.track);
    }
    return tracks;
}

uint32_t AddGpuTrack(std::string_view name) {
    State& state = GetState();
    const std::lock_guard lock(state.mutex);
    state.tracks.push_back({{std::string(name), true}});
    return uint32_t(state.tracks.size() - 1);
}

void AddZone(uint64_t frame, const Zone& zone) {
    State& state = GetState();
    const std::lock_guard lock(state.mutex);
    if (frame == state.current.index) {
        state.current.zones.push_back(zone);
        return;
    }

    const auto kept = std::ranges::find(state.frames, frame, &Frame::index);
    if (kept != state.frames.end()) {
        kept->zones.push_back(zone);
    }
}

ScopedZone::ScopedZone(ZoneId zone) : m_zone(zone) {
    m_nvtx = NvtxActive();
    if (m_nvtx) {
        PushRange(zone);
    }

    m_recording = Enabled();
    if (m_recording) {
        m_startNs = Now();
        ++t_depth;
    }
}

ScopedZone::~ScopedZone() {
    if (m_nvtx) {
        nvtxDomainRangePop(g_domain);
    }
    if (!m_recording) {
        return;
    }

    --t_depth;
    const Zone zone{m_zone, ThreadTrack(), t_depth, m_startNs, Now()};
    State& state = GetState();
    const std::lock_guard lock(state.mutex);
    state.current.zones.push_back(zone);
}

Report Summarize(const std::vector<Frame>& frames) {
    Report report;
    if (frames.empty()) {
        return report;
    }

    std::map<uint32_t, std::vector<ReportNode>> roots;
    std::map<uint32_t, double> outermostMs;
    for (const Frame& frame : frames) {
        report.frameMs += DurationMs(frame.startNs, frame.endNs);
        std::map<uint32_t, std::vector<const Zone*>> byTrack;
        for (const Zone& zone : frame.zones) {
            byTrack[zone.track].push_back(&zone);
            if (zone.depth == 0) {
                outermostMs[zone.track] += DurationMs(zone.startNs, zone.endNs);
            }
        }
        for (auto& [track, zones] : byTrack) {
            Gather(roots[track], std::move(zones));
        }
    }

    const std::vector<Track> tracks = Tracks();
    const auto count = double(frames.size());
    report.frames = frames.size();
    report.frameMs /= count;
    for (auto& [index, zones] : roots) {
        Average(zones, count, report.frameMs);
        report.tracks.push_back({.name = TrackName(tracks, index),
                                 .gpu = IsGpuTrack(tracks, index),
                                 .ms = outermostMs[index] / count,
                                 .zones = std::move(zones)});
    }
    return report;
}

std::string ReportText(const Report& report, const ReportFilter& filter) {
    std::string text = fmt::format("{:<56} {:>12} {:>10}\n", "", "Duration (ms)", "Percent");
    text += fmt::format("{:<56} {:12.3f}   ({} frames)\n", "Frame", report.frameMs, report.frames);
    for (const ReportTrack& track : report.tracks) {
        const std::string title = fmt::format("{} ({})", track.name, track.gpu ? "GPU" : "CPU");
        text += fmt::format("\n{:<56} {:12.3f}\n", title, track.ms);
        AddLines(text, track.zones, filter, 0);
    }
    return text;
}

std::string ReportJson(const Report& report) {
    std::vector<std::string> tracks;
    for (const ReportTrack& track : report.tracks) {
        tracks.push_back(fmt::format(R"({{"name":{},"gpu":{},"ms":{:.6g},"zones":{}}})",
                                     Quote(track.name), track.gpu, track.ms,
                                     ZonesJson(track.zones)));
    }
    return fmt::format(R"({{"frames":{},"frameMs":{:.6g},"tracks":[{}]}})", report.frames,
                       report.frameMs, Join(tracks, ",")) +
           "\n";
}

std::string TraceJson(const std::vector<Frame>& frames) {
    constexpr int kCpu = 1;
    constexpr int kGpu = 2;
    const std::vector<Track> tracks = Tracks();
    const uint64_t origin = frames.empty() ? 0 : frames.front().startNs;
    const auto us = [origin](uint64_t ns) { return double(int64_t(ns - origin)) * 1e-3; };

    std::vector<std::string> events{MetaEvent("process_name", kCpu, "", "CPU"),
                                    MetaEvent("process_name", kGpu, "", "GPU"),
                                    MetaEvent("thread_name", kCpu, "\"frames\"", "Frames")};
    std::map<uint32_t, bool> used;
    for (const Frame& frame : frames) {
        events.push_back(SpanEvent(fmt::format("Frame {}", frame.index), kCpu, "\"frames\"",
                                   us(frame.startNs), double(frame.endNs - frame.startNs) * 1e-3));
        for (const Zone& zone : frame.zones) {
            const bool gpu = IsGpuTrack(tracks, zone.track);
            used[zone.track] = gpu;
            events.push_back(SpanEvent(zone.id.Name(), gpu ? kGpu : kCpu,
                                       std::to_string(zone.track), us(zone.startNs),
                                       double(zone.endNs - zone.startNs) * 1e-3));
        }
    }
    for (const auto& [track, gpu] : used) {
        events.push_back(MetaEvent("thread_name", gpu ? kGpu : kCpu, std::to_string(track),
                                   TrackName(tracks, track)));
    }

    return R"({"traceEvents":[)" + Join(events, ",") + R"(],"displayTimeUnit":"ms"})" + "\n";
}

}
