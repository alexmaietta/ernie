// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/xr/stats.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iterator>
#include <vector>

#include "ernie/util/error.h"

namespace ernie::xr {
namespace {

constexpr size_t kRecentFrames = 600;

TimingSummary Summarize(std::vector<double> values) {
    TimingSummary summary;
    if (values.empty()) {
        return summary;
    }
    double sum = 0;
    for (double value : values) {
        sum += value;
    }
    summary.mean = sum / double(values.size());
    const size_t p95 = std::min(values.size() - 1, values.size() * 95 / 100);
    std::nth_element(values.begin(), values.begin() + std::ptrdiff_t(p95), values.end());
    summary.p95 = values[p95];
    summary.max = *std::max_element(values.begin(), values.end());
    return summary;
}

}

double HostSeconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

FrameStats::FrameStats() : m_watcher([this] { Watch(); }) {}

FrameStats::~FrameStats() {
    {
        std::lock_guard lock(m_mutex);
        m_stop = true;
    }
    m_watchedChanged.notify_all();
    m_watcher.join();
}

void FrameStats::Begin(const Frame& frame, const FrameClock& clock, double wait) {
    Record record;
    record.timing.index = frame.index;
    record.timing.displayTime = frame.displayTime;
    record.timing.wait = wait;
    record.timing.missed = clock.missed;
    record.clock.begun = HostSeconds();
    record.clock.poses = clock.poses;
    record.clock.display = clock.display;
    std::lock_guard lock(m_mutex);
    m_totals.period = clock.period;
    m_totals.estimated = clock.estimated;
    ++m_totals.frames;
    m_totals.missed += clock.missed;
    m_totals.notRendered += frame.shouldRender ? 0 : 1;
    m_recent.push_back(record);
    if (m_recent.size() > kRecentFrames) {
        m_recent.pop_front();
    }
}

void FrameStats::End(const Frame& frame, const device::Operation* work) {
    const double now = HostSeconds();
    std::lock_guard lock(m_mutex);
    Record* record = Find(frame.index);
    if (record == nullptr) {
        return;
    }
    FrameTiming& timing = record->timing;
    record->clock.ended = now;
    timing.cpu = now - record->clock.begun;
    timing.poseToSubmit = now - record->clock.poses;
    timing.motionToPhoton = record->clock.display - record->clock.poses;
    timing.rendered = work != nullptr;
    m_totals.presented += timing.rendered ? 1 : 0;
    if (work != nullptr) {
        m_watched.emplace_back(frame.index, *work);
        m_watchedChanged.notify_all();
    } else {
        timing.gpuDone = true;
    }
}

FrameStats::Record* FrameStats::Find(uint64_t index) {
    for (auto it = m_recent.rbegin(); it != m_recent.rend(); ++it) {
        if (it->timing.index == index) {
            return &*it;
        }
    }
    return nullptr;
}

void FrameStats::Watch() {
    std::unique_lock lock(m_mutex);
    while (true) {
        m_watchedChanged.wait(lock, [this] { return m_stop || !m_watched.empty(); });
        if (m_watched.empty()) {
            return;
        }
        const std::pair<uint64_t, device::Operation> watched = m_watched.front();
        m_watched.pop_front();
        lock.unlock();
        try {
            watched.second.Wait();
        } catch (const std::exception&) {
        }
        const double done = HostSeconds();
        lock.lock();
        Record* record = Find(watched.first);
        if (record == nullptr) {
            continue;
        }
        FrameTiming& timing = record->timing;
        timing.gpu = done - record->clock.ended;
        timing.lateness = std::max(done - record->clock.display, 0.0);
        timing.late = done > record->clock.display;
        timing.motionToPhoton = record->clock.display - record->clock.poses + timing.lateness;
        timing.gpuDone = true;
        m_totals.late += timing.late ? 1 : 0;
    }
}

void FrameStats::SetMetrics(const RuntimeMetrics& metrics) {
    std::lock_guard lock(m_mutex);
    m_metrics = metrics;
}

SessionStats FrameStats::Stats() const {
    std::lock_guard lock(m_mutex);
    SessionStats stats = m_totals;
    stats.runtimeMetrics = m_metrics;
    std::vector<double> wait, cpu, gpu, poseToSubmit, motionToPhoton;
    for (const Record& record : m_recent) {
        const FrameTiming& timing = record.timing;
        stats.recent.push_back(timing);
        wait.push_back(timing.wait);
        if (record.clock.ended == 0) {
            continue;
        }
        cpu.push_back(timing.cpu);
        poseToSubmit.push_back(timing.poseToSubmit);
        if (timing.rendered && timing.gpuDone) {
            gpu.push_back(timing.gpu);
            motionToPhoton.push_back(timing.motionToPhoton);
        }
    }
    stats.wait = Summarize(wait);
    stats.cpu = Summarize(cpu);
    stats.gpu = Summarize(gpu);
    stats.poseToSubmit = Summarize(poseToSubmit);
    stats.motionToPhoton = Summarize(motionToPhoton);
    return stats;
}

void FrameStats::Write(const std::filesystem::path& path) const {
    const SessionStats stats = Stats();
    std::ofstream file(path);
    if (!file) {
        throw Error("could not write {}", path.string());
    }
    const char* names[] = {"index",        "displayTime",    "wait",     "cpu",    "gpu",
                           "poseToSubmit", "motionToPhoton", "lateness", "missed", "rendered",
                           "late"};
    const bool json = path.extension() == ".json";
    if (json) {
        file << "{\"period\": " << stats.period
             << ", \"estimated\": " << (stats.estimated ? "true" : "false") << ", \"frames\": [\n";
    } else {
        for (size_t i = 0; i < std::size(names); ++i) {
            file << (i > 0 ? "," : "") << names[i];
        }
        file << "\n";
    }
    for (size_t row = 0; row < stats.recent.size(); ++row) {
        const FrameTiming& t = stats.recent[row];
        const double values[] = {t.wait,           t.cpu,     t.gpu, t.poseToSubmit,
                                 t.motionToPhoton, t.lateness};
        if (json) {
            file << "  {\"index\": " << t.index << ", \"displayTime\": " << t.displayTime;
            for (size_t i = 0; i < std::size(values); ++i) {
                file << ", \"" << names[i + 2] << "\": " << values[i];
            }
            file << ", \"missed\": " << t.missed
                 << ", \"rendered\": " << (t.rendered ? "true" : "false")
                 << ", \"late\": " << (t.late ? "true" : "false") << "}"
                 << (row + 1 < stats.recent.size() ? ",\n" : "\n");
        } else {
            file << t.index << "," << t.displayTime;
            for (double value : values) {
                file << "," << value;
            }
            file << "," << t.missed << "," << int(t.rendered) << "," << int(t.late) << "\n";
        }
    }
    if (json) {
        file << "]}\n";
    }
}

}
