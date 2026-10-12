// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace ernie::app {

/// What a capture's file name may name (`ExpandCaptureName`).
struct CaptureNameFields {
    std::string scene = "ernie";  ///< The scene file's stem.
    std::string camera;           ///< The active camera's name.
    std::chrono::system_clock::time_point when =
        std::chrono::system_clock::now();
    std::int64_t frame = 0;  ///< The frame of a sequence.
};

/// `pattern` with its fields filled in: `{scene}`, `{camera}`,
/// `{date}` (YYYY-MM-DD), `{time}` (HH-MM-SS, local) and `{frame}`
/// (four digits at least). Characters a file name cannot hold become
/// `_`.
std::string ExpandCaptureName(std::string_view pattern,
                              const CaptureNameFields& fields);

/// `path` if nothing is there, else with `_1`, `_2`, ... before its
/// extension: a capture never overwrites another.
std::filesystem::path UniquePath(const std::filesystem::path& path);

/// Where captures go unless chosen: the user's Pictures/Ernie, else
/// `captures` in the working directory.
std::filesystem::path DefaultCaptureFolder();

/// The video a sequence encodes into, with ffmpeg.
enum class VideoFormat {
    None,    ///< Frames only.
    Mp4,     ///< H.264 in MP4, 4:2:0, for sharing.
    ProRes,  ///< ProRes 422 HQ in QuickTime, for editing.
};

/// The file extension of `format`'s videos (".mp4", ".mov"), or ""
/// for `None`.
const char* VideoExtension(VideoFormat format);

/// ffmpeg's executable on the PATH, or empty when it is not there.
std::filesystem::path FindFfmpeg();

/// The command line that encodes the frames `framePattern` (an ffmpeg
/// image pattern such as `shot_%04d.png`, numbered from 0) at `fps`
/// into `video` as `format`, with `ffmpeg`.
std::vector<std::string>
EncodeCommand(const std::filesystem::path& ffmpeg,
              const std::filesystem::path& framePattern, double fps,
              const std::filesystem::path& video, VideoFormat format);

/// Runs `command` (the program, then its arguments) and waits for
/// it, without a shell; its output is discarded. Returns its exit
/// status, or -1 when it could not start.
int RunProcess(const std::vector<std::string>& command);

}  // namespace ernie::app
