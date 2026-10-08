#include "ernie/util/profiler.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>

namespace ernie::profiler {

namespace {

std::atomic<bool> enabled{false};

struct State {
    std::mutex mutex;
    Frame current;
    std::deque<Frame> frames;
    std::size_t history = 300;
    std::vector<std::string> threadNames;
};

State& Global() {
    static State state;
    return state;
}

thread_local std::uint32_t depth = 0;

}  // namespace

void SetEnabled(bool on) {
    enabled.store(on, std::memory_order_relaxed);
}

bool Enabled() {
    return enabled.load(std::memory_order_relaxed);
}

std::uint64_t Now() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now().time_since_epoch())
                                          .count());
}

void BeginFrame() {
    State& s = Global();
    const std::uint64_t now = Now();
    std::lock_guard lock(s.mutex);
    if (s.current.start != 0) {
        s.current.end = now;
        s.frames.push_back(std::move(s.current));
        while (s.frames.size() > s.history) {
            s.frames.pop_front();
        }
    }
    const std::uint64_t next = s.frames.empty() ? 0 : s.frames.back().index + 1;
    s.current = Frame{};
    s.current.index = next;
    s.current.start = now;
}

std::uint64_t CurrentFrame() {
    State& s = Global();
    std::lock_guard lock(s.mutex);
    return s.current.index;
}

std::vector<Frame> Frames() {
    State& s = Global();
    std::lock_guard lock(s.mutex);
    return {s.frames.begin(), s.frames.end()};
}

void SetHistory(std::size_t frames) {
    State& s = Global();
    std::lock_guard lock(s.mutex);
    s.history = std::max<std::size_t>(frames, 1);
    while (s.frames.size() > s.history) {
        s.frames.pop_front();
    }
}

std::size_t History() {
    State& s = Global();
    std::lock_guard lock(s.mutex);
    return s.history;
}

void Clear() {
    State& s = Global();
    std::lock_guard lock(s.mutex);
    s.frames.clear();
    s.current.cpu.clear();
    s.current.gpu.clear();
}

std::uint32_t ThreadIndex() {
    thread_local const std::uint32_t index = [] {
        State& s = Global();
        std::lock_guard lock(s.mutex);
        s.threadNames.push_back("Thread " + std::to_string(s.threadNames.size()));
        return static_cast<std::uint32_t>(s.threadNames.size() - 1);
    }();
    return index;
}

void SetThreadName(std::string name) {
    const std::uint32_t index = ThreadIndex();
    State& s = Global();
    std::lock_guard lock(s.mutex);
    s.threadNames[index] = std::move(name);
}

std::vector<std::string> ThreadNames() {
    State& s = Global();
    std::lock_guard lock(s.mutex);
    return s.threadNames;
}

void AddGpuZone(std::uint64_t frame, const Zone& zone) {
    State& s = Global();
    std::lock_guard lock(s.mutex);
    if (frame == s.current.index) {
        s.current.gpu.push_back(zone);
        return;
    }
    for (Frame& kept : s.frames) {
        if (kept.index == frame) {
            kept.gpu.push_back(zone);
            return;
        }
    }
}

ScopedZone::ScopedZone(const char* name) : name_(name) {
    if (Enabled()) {
        active_ = true;
        start_ = Now();
        ++depth;
    }
}

ScopedZone::~ScopedZone() {
    if (!active_) {
        return;
    }
    --depth;
    Zone zone{name_, start_, Now(), ThreadIndex(), depth};
    State& s = Global();
    std::lock_guard lock(s.mutex);
    s.current.cpu.push_back(zone);
}

}  // namespace ernie::profiler
