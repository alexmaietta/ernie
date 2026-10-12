// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/util/profiler.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

/// Summaries of the profiler's frames (ernie/util/profiler.h) as
/// Omniverse Kit's profiler window shows them: each zone nested in
/// the zone that encloses it, with its time averaged over the frames,
/// its share of the frame, and exports as text, JSON and a Chrome
/// trace.
///
/// References:
/// - NVIDIA, "Profiling", Omniverse Kit SDK manual (the Profiler
///   window: GPU and CPU zone trees, depths, averaging, exports).
///   https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/guide/profiling.html
/// - Google, "Trace Event Format" (complete events, "ph": "X", and
///   thread-name metadata), read by Perfetto and chrome://tracing.
///   https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU
namespace ernie::profiler {

/// A zone's averages over the frames summarized: zones of the same
/// name under the same parent merge.
struct ReportNode {
    std::string name;
    double ms = 0.0;       ///< Mean time per frame, in milliseconds.
    double calls = 0.0;    ///< Mean times per frame it ran.
    double percent = 0.0;  ///< Of its section's total.
    std::vector<ReportNode>
        children;  ///< In the order they first ran.
};

/// The GPU's zones and the CPU's, the latter under a node per thread.
struct Report {
    std::vector<ReportNode> gpu;
    std::vector<ReportNode> cpu;
    double gpuMs = 0.0;  ///< Mean GPU time: its outermost zones'.
    double cpuMs = 0.0;  ///< Mean frame time on the host.
    std::size_t gpuFrames = 0;  ///< Frames with GPU zones averaged.
    std::size_t cpuFrames = 0;  ///< Frames averaged.
};

/// Averages `frames`' zones; `threadNames` names the CPU's threads by
/// index (`ThreadNames`). GPU zones average over the frames whose
/// zones have arrived.
Report Summarize(std::span<const Frame> frames,
                 const std::vector<std::string>& threadNames);

/// What to list of a report: zones nested at most `depth` deep (0:
/// the outermost alone) and taking at least `minMs`, in a tree whose
/// siblings come in the order they ran or, `byTime`, longest first.
struct ReportFilter {
    std::uint32_t depth = 8;
    double minMs = 0.0;
    bool byTime = false;
};

/// A listed zone and how deep it is shown.
struct ReportRow {
    const ReportNode* node = nullptr;
    std::uint32_t depth = 0;
    bool parent = false;  ///< Has listed children.
};

/// The rows of `nodes` that `filter` lists, parents before their
/// children.
std::vector<ReportRow> Rows(const std::vector<ReportNode>& nodes,
                            const ReportFilter& filter);

/// The report as lines of indented names, times and shares.
std::string ReportText(const Report& report,
                       const ReportFilter& filter);

/// The report as JSON: {"gpu": {"ms", "zones": [...]}, "cpu": ...},
/// each zone {"name", "ms", "calls", "percent", "children"}.
std::string ReportJson(const Report& report);

/// `frames` in the Trace Event Format: a complete event per zone, the
/// CPU's threads as threads of process 1 and the GPU's zones as
/// process 2, times in microseconds from the first frame. The GPU's
/// own clock is not the host's, so each frame's GPU zones are placed
/// from that frame's start.
std::string TraceJson(std::span<const Frame> frames,
                      const std::vector<std::string>& threadNames);

}  // namespace ernie::profiler
