// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "ernie/export.h"

namespace ernie::profiler {

class ERNIE_API ZoneId {
  public:
    static ZoneId Named(std::string_view name);

    std::string_view Name() const;

    uint32_t Index() const {
        return m_index;
    }

    friend bool operator==(ZoneId a, ZoneId b) {
        return a.m_index == b.m_index;
    }

    friend bool operator!=(ZoneId a, ZoneId b) {
        return a.m_index != b.m_index;
    }

  private:
    explicit ZoneId(uint32_t index) : m_index(index) {}

    uint32_t m_index;
};

struct Track {
    std::string name;
    bool gpu = false;
};

struct Zone {
    ZoneId id;
    uint32_t track = 0;
    uint32_t depth = 0;
    uint64_t startNs = 0;
    uint64_t endNs = 0;
};

struct Frame {
    uint64_t index = 0;
    uint64_t startNs = 0;
    uint64_t endNs = 0;
    std::vector<Zone> zones;
};

ERNIE_API void SetEnabled(bool enabled);
ERNIE_API bool Enabled();

ERNIE_API uint64_t Now();

ERNIE_API void BeginFrame();

ERNIE_API uint64_t CurrentFrame();

ERNIE_API std::vector<Frame> Frames();

ERNIE_API void SetHistory(size_t frames);

ERNIE_API void Clear();

ERNIE_API void SetThreadName(std::string_view name);

ERNIE_API std::vector<Track> Tracks();

ERNIE_API uint32_t AddGpuTrack(std::string_view name);

ERNIE_API void AddZone(uint64_t frame, const Zone& zone);

struct ReportNode {
    std::string name;
    double ms = 0;
    double calls = 0;
    double percent = 0;
    double minMs = 0;
    double maxMs = 0;
    std::vector<ReportNode> children;
};

struct ReportTrack {
    std::string name;
    bool gpu = false;
    double ms = 0;
    std::vector<ReportNode> zones;
};

struct Report {
    size_t frames = 0;
    double frameMs = 0;
    std::vector<ReportTrack> tracks;
};

ERNIE_API Report Summarize(const std::vector<Frame>& frames);

struct ReportFilter {
    uint32_t depth = 8;
    double minMs = 0;
    bool byTime = false;
};

ERNIE_API std::string ReportText(const Report& report, const ReportFilter& filter = {});

ERNIE_API std::string ReportJson(const Report& report);

ERNIE_API std::string TraceJson(const std::vector<Frame>& frames);

class ERNIE_API ScopedZone {
  public:
    explicit ScopedZone(ZoneId zone);
    ~ScopedZone();
    ScopedZone(const ScopedZone&) = delete;
    ScopedZone& operator=(const ScopedZone&) = delete;

  private:
    ZoneId m_zone;
    uint64_t m_startNs = 0;
    bool m_recording = false;
    bool m_nvtx = false;
};

}

#define ERNIE_PROFILER_CONCAT_(a, b) a##b
#define ERNIE_PROFILER_CONCAT(a, b) ERNIE_PROFILER_CONCAT_(a, b)

#define ERNIE_ZONE(name)                                                                           \
    static const ::ernie::profiler::ZoneId ERNIE_PROFILER_CONCAT(ernieZoneId, __LINE__) =          \
        ::ernie::profiler::ZoneId::Named(name);                                                    \
    const ::ernie::profiler::ScopedZone ERNIE_PROFILER_CONCAT(ernieZone, __LINE__)(                \
        ERNIE_PROFILER_CONCAT(ernieZoneId, __LINE__))
