// SPDX-License-Identifier: BSD-3-Clause
/// @file profiler.h
/// @brief Ernie's in-process CPU and GPU zone profiler.
/// Records `ERNIE_ZONE` and `ERNIE_GPU_ZONE` frame by frame for the
/// Profiler panel. Off until `SetEnabled`; a zone then costs one
/// atomic load. CPU zones belong to the frame they end in; GPU zones
/// to the frame that recorded their commands.
#pragma once

#include <cstdint>
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
/// CPU times use the steady clock (`Now`); GPU times use the
/// device's timestamp clock and compare only with each other.
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
void SetEnabled(bool enabled);
/// @return Whether CPU or GPU zones are recording.
bool Enabled();

/// @brief Starts or stops recording CPU zones (`ERNIE_ZONE`).
/// @param enabled Whether to record.
void SetCpuEnabled(bool enabled);
/// @return Whether CPU zones are recording.
bool CpuEnabled();
/// @brief Starts or stops recording GPU zones (`ERNIE_GPU_ZONE`).
/// @param enabled Whether to record.
void SetGpuEnabled(bool enabled);
/// @return Whether GPU zones are recording.
bool GpuEnabled();

/// @return Nanoseconds on the steady clock.
std::uint64_t Now();

/// @brief Ends the current frame and starts the next.
/// The ended frame is kept even when not recording, so frame times
/// show before any zones.
void BeginFrame();

/// @return The index of the frame being recorded.
std::uint64_t CurrentFrame();

/// @return The kept finished frames, oldest first, at most
///         `History`.
std::vector<Frame> Frames();
/// @brief Sets how many finished frames are kept.
/// @param frames The frame count.
void SetHistory(std::size_t frames);
/// @return How many finished frames are kept.
std::size_t History();
/// @brief Drops all kept frames.
void Clear();

/// @return The calling thread's index, as in `Zone::thread`.
std::uint32_t ThreadIndex();
/// @brief Names the calling thread.
/// @param name The name; unnamed threads are "Thread <index>".
void SetThreadName(std::string name);
/// @return Thread names by index.
std::vector<std::string> ThreadNames();

/// @brief Copies a run-time name to process-lifetime storage, since
/// zones keep their name's pointer.
/// @param name The name.
/// @return The copy; the same pointer for the same name.
const char* Intern(std::string_view name);

/// @brief Adds a GPU zone to the frame that recorded it.
/// @param frame The frame index; dropped if no longer kept.
/// @param zone  The zone.
void AddGpuZone(std::uint64_t frame, const Zone& zone);

/// @brief Times the rest of the enclosing scope as a CPU zone.
class ScopedZone {
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

}  // namespace ernie::profiler
