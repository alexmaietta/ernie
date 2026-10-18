// SPDX-License-Identifier: BSD-3-Clause
/// @file profiler.h
/// @brief Ernie's in-process CPU and GPU zone profiler, its zone macros, and averaged reports.
///
/// Records `ERNIE_ZONE` frame by frame for the Profiler panel, and with -DERNIE_TRACY=ON for
/// Tracy too; `ERNIE_GPU_ZONE` times commands with timestamp queries the device context reads
/// back into `AddGpuZone`, and reaches Tracy too.
/// Off until `SetEnabled`; a zone then costs one atomic load. CPU zones belong to the frame they
/// end in; GPU zones to the frame that recorded their commands. Reports average frames into zone
/// trees, modelled on Omniverse Kit's profiler window, and export text, JSON and Chrome traces. GPU
/// zones are in device/profile.h.
///
///     // The rest of the scope, one per scope; a string literal.
///     ERNIE_ZONE("Sync");
///     // Text attached to the scope's zone (Tracy only).
///     ERNIE_ZONE_TEXT(path.String());
///     // A value plotted over time (Tracy only).
///     ERNIE_PLOT("views", 256);
///     // The end of a frame of a named frame set (Tracy only).
///     ERNIE_FRAME_MARK("Step");
///
/// References:
/// - B. Taudul, "Tracy Profiler". https://github.com/wolfpld/tracy
/// - NVIDIA, "Profiling", Omniverse Kit SDK manual (the Profiler window: GPU and CPU zone trees,
///   depths, averaging, exports).
///   https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/guide/profiling.html
/// - Google, "Trace Event Format" (complete events, "ph": "X", and thread-name metadata), read by
///   Perfetto and chrome://tracing.
///   https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU
#pragma once

#include "ernie/export.h"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ernie::profiler {

/// @brief A timed span on a CPU thread or the GPU.
struct Zone {
    /// A string literal; zones keep the pointer.
    const char* name = "";
    /// Start, in nanoseconds.
    std::uint64_t start = 0;
    /// End, in nanoseconds.
    std::uint64_t end = 0;
    /// The CPU thread's index; unused for GPU zones.
    std::uint32_t thread = 0;
    /// How many zones of the same thread or recording enclose it.
    std::uint32_t depth = 0;
};

/// @brief One frame's zones.
/// CPU times use the steady clock (`Now`). GPU times are converted to it: calibrated where the
/// device reads its clock with the host's, else aligned to when the first GPU zone was recorded.
struct Frame {
    /// The frame's index.
    std::uint64_t index = 0;
    /// Start on the steady clock, in nanoseconds.
    std::uint64_t start = 0;
    /// End on the steady clock, in nanoseconds.
    std::uint64_t end = 0;
    /// CPU zones.
    std::vector<Zone> cpu;
    /// GPU zones.
    std::vector<Zone> gpu;
};

/// Newest frames whose GPU zones may still be arriving.
inline constexpr std::size_t kFramesInFlight = 2;

/// @brief Starts or stops recording both CPU and GPU zones.
/// Stopping keeps the frames recorded so far.
/// @param enabled Whether to record.
ERNIE_API void SetEnabled(bool enabled);
/// @return Whether CPU or GPU zones are recording.
ERNIE_API bool Enabled();

/// @brief Starts or stops recording CPU zones (`ERNIE_ZONE`).
/// @param enabled Whether to record.
ERNIE_API void SetCpuEnabled(bool enabled);
/// @return Whether CPU zones are recording.
ERNIE_API bool CpuEnabled();
/// @brief Starts or stops recording GPU zones (`ERNIE_GPU_ZONE`).
/// @param enabled Whether to record.
ERNIE_API void SetGpuEnabled(bool enabled);
/// @return Whether GPU zones are recording.
ERNIE_API bool GpuEnabled();

/// @return Nanoseconds on the steady clock.
ERNIE_API std::uint64_t Now();

/// @brief Ends the current frame and starts the next.
/// The ended frame is kept even when not recording, so frame times
/// show before any zones.
ERNIE_API void BeginFrame();

/// @return The index of the frame being recorded.
ERNIE_API std::uint64_t CurrentFrame();

/// @return The kept finished frames, oldest first, at most
///         `History`.
ERNIE_API std::vector<Frame> Frames();
/// @brief Sets how many finished frames are kept.
/// @param frames The frame count.
ERNIE_API void SetHistory(std::size_t frames);
/// @return How many finished frames are kept.
ERNIE_API std::size_t History();
/// @brief Drops all kept frames.
ERNIE_API void Clear();

/// @return The calling thread's index, as in `Zone::thread`.
ERNIE_API std::uint32_t ThreadIndex();
/// @brief Names the calling thread.
/// @param name The name; unnamed threads are "Thread <index>".
ERNIE_API void SetThreadName(std::string name);
/// @return Thread names by index.
ERNIE_API std::vector<std::string> ThreadNames();

/// @brief Copies a run-time name to process-lifetime storage, since
/// zones keep their name's pointer.
/// @param name The name.
/// @return The copy; the same pointer for the same name.
ERNIE_API const char* Intern(std::string_view name);

/// @brief Adds a GPU zone to the frame that recorded it.
/// @param frame The frame index; dropped if no longer kept.
/// @param zone  The zone.
ERNIE_API void AddGpuZone(std::uint64_t frame, const Zone& zone);

/// @brief Times the rest of the enclosing scope as a CPU zone.
class ERNIE_API ScopedZone {
  public:
    /// @brief Starts the zone if recording.
    /// @param name A string literal naming the zone.
    explicit ScopedZone(const char* name);
    ~ScopedZone();
    ScopedZone(const ScopedZone&) = delete;
    ScopedZone& operator=(const ScopedZone&) = delete;

  private:
    const char* name_;
    std::uint64_t start_ = 0;
    bool active_ = false;
};

// ----------------------------------------------------------------------------------------------
// Reports

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
ERNIE_API Report Summarize(std::span<const Frame> frames,
                           const std::vector<std::string>& threadNames);

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
ERNIE_API std::vector<ReportRow> Rows(const std::vector<ReportNode>& nodes,
                                      const ReportFilter& filter);

/// @brief Formats a report as indented names, times and shares.
/// @param report The report.
/// @param filter Which zones to list.
/// @return The text.
ERNIE_API std::string ReportText(const Report& report, const ReportFilter& filter);

/// @brief Formats a report as JSON.
/// The shape is {"gpu": {"ms", "zones": [...]}, "cpu": ...}, each
/// zone {"name", "ms", "calls", "percent", "children"}.
/// @param report The report.
/// @return The JSON text.
ERNIE_API std::string ReportJson(const Report& report);

/// @brief Exports frames in the Trace Event Format.
/// CPU threads are threads of process 1, GPU zones process 2; times
/// are microseconds from the first frame. GPU zones are placed from
/// their frame's start since the GPU clock differs from the host's.
/// @param frames      The frames.
/// @param threadNames CPU thread names by index.
/// @return The trace JSON.
ERNIE_API std::string TraceJson(std::span<const Frame> frames,
                                const std::vector<std::string>& threadNames);

}  // namespace ernie::profiler

#define ERNIE_PROFILE_CONCAT_(a, b) a##b
#define ERNIE_PROFILE_CONCAT(a, b) ERNIE_PROFILE_CONCAT_(a, b)
#define ERNIE_OWN_ZONE(name)                                                                       \
    const ::ernie::profiler::ScopedZone ERNIE_PROFILE_CONCAT(ernieZone, __LINE__)(name)

#if ERNIE_TRACY
#include <tracy/Tracy.hpp>

#define ERNIE_ZONE(name)                                                                           \
    ZoneScopedN(name);                                                                             \
    ERNIE_OWN_ZONE(name)
#define ERNIE_ZONE_TEXT(text)                                                                      \
    do {                                                                                           \
        const auto& ernieZoneText = (text);                                                        \
        ZoneText(ernieZoneText.data(), ernieZoneText.size());                                      \
    } while (false)
#define ERNIE_PLOT(name, value) TracyPlot(name, static_cast<double>(value))
#define ERNIE_FRAME_MARK(name) FrameMarkNamed(name)
#else
#define ERNIE_ZONE(name) ERNIE_OWN_ZONE(name)
#define ERNIE_ZONE_TEXT(text)
#define ERNIE_PLOT(name, value)
#define ERNIE_FRAME_MARK(name)
#endif
