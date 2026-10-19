// SPDX-License-Identifier: BSD-3-Clause
/// @file capture.h
/// @brief Naming captures and encoding image sequences into video with ffmpeg.
#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace ernie::app {

/// @brief What a capture's file name may name (`ExpandCaptureName`).
struct CaptureNameFields {
    /// The scene file's stem.
    std::string scene = "ernie";
    /// The camera looked through.
    std::string camera = "Perspective";
    std::chrono::system_clock::time_point when = std::chrono::system_clock::now();
    /// The frame of a sequence.
    std::int64_t frame = 0;
};

/// @return A capture name pattern filled in: `{scene}`, `{camera}`, `{date}` (YYYY-MM-DD),
/// `{time}` (local HH-MM-SS) and `{frame}` (at least four digits). Characters a file name cannot
/// hold become `_`.
std::string ExpandCaptureName(std::string_view pattern, const CaptureNameFields& fields);

/// @return `path` when no file is there, else with `_1`, `_2`, ... before its extension.
std::filesystem::path UniquePath(const std::filesystem::path& path);

/// @return Where captures go by default: Pictures/Ernie in the home folder, else `captures` in
/// the working directory.
std::filesystem::path DefaultCaptureFolder();

/// @brief The video a sequence is encoded into.
enum class VideoFormat : std::uint8_t {
    /// Frames only.
    None,
    /// H.264 in MP4, 4:2:0, for sharing.
    Mp4,
    /// ProRes 422 HQ in QuickTime, for editing.
    ProRes,
};

/// @return The format's extension, ".mp4" or ".mov"; "" for none.
const char* VideoExtension(VideoFormat format);

/// @return The ffmpeg command encoding frames numbered from 0 in `framePattern`, such as
/// `shot_%04d.png`, at `fps` into `video`: the program and its arguments.
std::vector<std::string> EncodeCommand(const std::filesystem::path& ffmpeg,
                                       const std::filesystem::path& framePattern, double fps,
                                       const std::filesystem::path& video, VideoFormat format);

}  // namespace ernie::app
