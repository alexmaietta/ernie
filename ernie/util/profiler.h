// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <string>
#include <vector>

/// Ernie's in-process profiler: the zones of `ERNIE_ZONE` and
/// `ERNIE_GPU_ZONE` (ernie/util/profile.h,
/// ernie/device/gpu_profile.h) recorded frame by frame for a profiler
/// panel, alongside Tracy's (which streams them to its own viewer in
/// a profiling build). Recording is off until `SetEnabled`; while off
/// a zone costs one atomic load.
///
/// A frame runs from one `BeginFrame` to the next: CPU zones belong
/// to the frame in which they end, GPU zones to the frame in which
/// their commands were recorded (they arrive later, once the device
/// has run them). The latest frames are kept, oldest first
/// (`Frames`).
namespace ernie::profiler {

/// A timed span: a CPU zone of thread `thread`, or a GPU zone
/// (`thread` unused) on the device's own clock. Times are
/// nanoseconds; `depth` is how many zones of the same thread (or
/// command recording) enclose it.
struct Zone {
    const char* name =
        "";  ///< A string literal: zones keep the pointer.
    std::uint64_t start = 0;
    std::uint64_t end = 0;
    std::uint32_t thread = 0;
    std::uint32_t depth = 0;
};

/// One frame's zones. CPU times are on the steady clock (`Now`); GPU
/// times on the device's timestamp clock, comparable only with each
/// other.
struct Frame {
    std::uint64_t index = 0;
    std::uint64_t start = 0;
    std::uint64_t end = 0;
    std::vector<Zone> cpu;
    std::vector<Zone> gpu;
};

/// Starts or stops recording; stopping keeps the frames recorded so
/// far.
void SetEnabled(bool enabled);
bool Enabled();

/// Nanoseconds on the steady clock.
std::uint64_t Now();

/// Ends the current frame and starts the next; the ended frame joins
/// `Frames` (whether recording or not, so frame times show even
/// before any zones).
void BeginFrame();

/// The index of the frame being recorded.
std::uint64_t CurrentFrame();

/// The finished frames kept, oldest first, at most `History` of them.
std::vector<Frame> Frames();
void SetHistory(std::size_t frames);
std::size_t History();
void Clear();

/// The calling thread's index in `Zone::thread`, and the names
/// threads gave themselves
/// (`SetThreadName`, else "Thread <index>"), by index.
std::uint32_t ThreadIndex();
void SetThreadName(std::string name);
std::vector<std::string> ThreadNames();

/// Adds a GPU zone recorded during frame `frame`; dropped if that
/// frame is no longer kept.
void AddGpuZone(std::uint64_t frame, const Zone& zone);

/// Times the rest of the enclosing scope as a CPU zone named `name`
/// while recording.
class ScopedZone {
  public:
    explicit ScopedZone(const char* name);
    ~ScopedZone();
    ScopedZone(const ScopedZone&) = delete;
    ScopedZone& operator=(const ScopedZone&) = delete;

  private:
    const char* name_;
    std::uint64_t start_ = 0;
    bool active_ = false;
};

}  // namespace ernie::profiler
