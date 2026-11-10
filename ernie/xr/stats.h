// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <mutex>
#include <thread>
#include <utility>

#include "ernie/device/operation.h"
#include "ernie/xr/runtime.h"
#include "ernie/xr/session.h"

namespace ernie::xr {

class FrameStats {
  public:
    FrameStats();
    ~FrameStats();
    FrameStats(const FrameStats&) = delete;
    FrameStats& operator=(const FrameStats&) = delete;

    void Begin(const Frame& frame, const FrameClock& clock, double wait);
    void End(const Frame& frame, const device::Operation* work);
    SessionStats Stats() const;
    void SetMetrics(const RuntimeMetrics& metrics);
    void Write(const std::filesystem::path& path) const;

  private:
    struct Clock {
        double begun = 0;
        double poses = 0;
        double display = 0;
        double ended = 0;
    };

    struct Record {
        FrameTiming timing;
        Clock clock;
    };

    Record* Find(uint64_t index);
    void Watch();

    mutable std::mutex m_mutex;
    std::deque<Record> m_recent;
    SessionStats m_totals;
    RuntimeMetrics m_metrics;
    std::deque<std::pair<uint64_t, device::Operation>> m_watched;
    std::condition_variable m_watchedChanged;
    bool m_stop = false;
    std::thread m_watcher;
};

}
