// SPDX-License-Identifier: BSD-3-Clause
/// @file profile_report.h
/// @brief Averaged zone trees of profiler frames and their exports.
/// Modelled on Omniverse Kit's profiler window; exports text, JSON
/// and Chrome traces.
///
/// References:
/// - NVIDIA, "Profiling", Omniverse Kit SDK manual (the Profiler
///   window: GPU and CPU zone trees, depths, averaging, exports).
///   https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/guide/profiling.html
/// - Google, "Trace Event Format" (complete events, "ph": "X", and
///   thread-name metadata), read by Perfetto and chrome://tracing.
///   https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU
#pragma once

#include "ernie/util/profiler.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace ernie::profiler {

/// @brief A zone averaged over frames.
/// Zones of the same name under the same parent merge.
struct ReportNode {
    /// The zone's name.
    std::string name;
    /// Mean time per frame, in milliseconds.
    double ms = 0.0;
    /// Mean runs per frame.
    double calls = 0.0;
    /// Share of its section's total, in percent.
    double percent = 0.0;
    /// Nested zones, in the order they first ran.
    std::vector<ReportNode> children;
};

/// @brief GPU and CPU zone trees averaged over frames.
struct Report {
    /// The GPU's zones.
    std::vector<ReportNode> gpu;
    /// The CPU's zones, under a node per thread.
    std::vector<ReportNode> cpu;
    /// Mean GPU time of the outermost zones, in milliseconds.
    double gpuMs = 0.0;
    /// Mean host frame time, in milliseconds.
    double cpuMs = 0.0;
    /// Frames whose GPU zones were averaged.
    std::size_t gpuFrames = 0;
    /// Frames averaged.
    std::size_t cpuFrames = 0;
};

/// @brief Averages frames' zones into a report.
/// @param frames      The frames; GPU zones count only once arrived.
/// @param threadNames CPU thread names by index (`ThreadNames`).
/// @return The report.
Report Summarize(std::span<const Frame> frames, const std::vector<std::string>& threadNames);

/// @brief Which zones of a report to list.
struct ReportFilter {
    /// Deepest nesting shown; 0 lists the outermost alone.
    std::uint32_t depth = 8;
    /// Shortest zone shown, in milliseconds.
    double minMs = 0.0;
    /// Sort siblings longest first instead of by run order.
    bool byTime = false;
};

/// @brief A listed zone and how deep it is shown.
struct ReportRow {
    /// The zone.
    const ReportNode* node = nullptr;
    /// Its nesting depth.
    std::uint32_t depth = 0;
    /// Whether it has listed children.
    bool parent = false;
};

/// @brief Flattens a zone tree into listed rows.
/// @param nodes  The tree.
/// @param filter Which zones to list.
/// @return The rows, parents before their children.
std::vector<ReportRow> Rows(const std::vector<ReportNode>& nodes, const ReportFilter& filter);

/// @brief Formats a report as indented names, times and shares.
/// @param report The report.
/// @param filter Which zones to list.
/// @return The text.
std::string ReportText(const Report& report, const ReportFilter& filter);

/// @brief Formats a report as JSON.
/// The shape is {"gpu": {"ms", "zones": [...]}, "cpu": ...}, each
/// zone {"name", "ms", "calls", "percent", "children"}.
/// @param report The report.
/// @return The JSON text.
std::string ReportJson(const Report& report);

/// @brief Exports frames in the Trace Event Format.
/// CPU threads are threads of process 1, GPU zones process 2; times
/// are microseconds from the first frame. GPU zones are placed from
/// their frame's start since the GPU clock differs from the host's.
/// @param frames      The frames.
/// @param threadNames CPU thread names by index.
/// @return The trace JSON.
std::string TraceJson(std::span<const Frame> frames, const std::vector<std::string>& threadNames);

}  // namespace ernie::profiler
