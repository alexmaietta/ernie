// SPDX-License-Identifier: BSD-3-Clause
/// @file capture.cpp
/// @brief Capture names, unique paths, the default folder and ffmpeg's encoding commands.
#include "ernie/app/capture.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <system_error>

namespace ernie::app {
namespace {

/// @return Whether a file name cannot hold the character on some system.
bool Forbidden(char c) {
    return c == '/' || c == '\\' || c == ':' || c == '*' || c == '?' || c == '"' || c == '<' ||
           c == '>' || c == '|' || static_cast<unsigned char>(c) < 32;
}

std::tm LocalTime(std::chrono::system_clock::time_point when) {
    const std::time_t seconds = std::chrono::system_clock::to_time_t(when);
    std::tm local{};
#if defined(_WIN32)
    localtime_s(&local, &seconds);
#else
    localtime_r(&seconds, &local);
#endif
    return local;
}

}  // namespace

std::string ExpandCaptureName(std::string_view pattern, const CaptureNameFields& fields) {
    const std::tm local = LocalTime(fields.when);
    std::array<char, 16> date{};
    std::array<char, 16> time{};
    std::array<char, 32> frame{};
    std::strftime(date.data(), date.size(), "%Y-%m-%d", &local);
    std::strftime(time.data(), time.size(), "%H-%M-%S", &local);
    std::snprintf(frame.data(), frame.size(), "%04lld", static_cast<long long>(fields.frame));
    std::string name;
    for (std::size_t i = 0; i < pattern.size(); ++i) {
        const std::size_t close = pattern.find('}', i);
        if (pattern[i] == '{' && close != std::string_view::npos) {
            const std::string_view key = pattern.substr(i + 1, close - i - 1);
            name += key == "scene"    ? fields.scene
                    : key == "camera" ? fields.camera
                    : key == "date"   ? std::string(date.data())
                    : key == "time"   ? std::string(time.data())
                    : key == "frame"  ? std::string(frame.data())
                                      : std::string(pattern.substr(i, close - i + 1));
            i = close;
            continue;
        }
        name += pattern[i];
    }
    for (char& c : name) {
        if (Forbidden(c)) {
            c = '_';
        }
    }
    return name.empty() ? std::string("capture") : name;
}

std::filesystem::path UniquePath(const std::filesystem::path& path) {
    std::error_code error;
    if (!std::filesystem::exists(path, error)) {
        return path;
    }
    const std::filesystem::path stem = path.parent_path() / path.stem();
    const std::string extension = path.extension().string();
    for (int n = 1;; ++n) {
        std::filesystem::path next = stem;
        next += "_" + std::to_string(n) + extension;
        if (!std::filesystem::exists(next, error)) {
            return next;
        }
    }
}

std::filesystem::path DefaultCaptureFolder() {
#if defined(_WIN32)
    const char* home = std::getenv("USERPROFILE");
#else
    const char* home = std::getenv("HOME");
#endif
    std::error_code error;
    if (home != nullptr && *home != '\0') {
        const std::filesystem::path pictures = std::filesystem::path(home) / "Pictures";
        if (std::filesystem::is_directory(pictures, error)) {
            return pictures / "Ernie";
        }
    }
    return std::filesystem::current_path(error) / "captures";
}

const char* VideoExtension(VideoFormat format) {
    switch (format) {
    case VideoFormat::Mp4:
        return ".mp4";
    case VideoFormat::ProRes:
        return ".mov";
    case VideoFormat::None:
        break;
    }
    return "";
}

std::vector<std::string> EncodeCommand(const std::filesystem::path& ffmpeg,
                                       const std::filesystem::path& framePattern, double fps,
                                       const std::filesystem::path& video, VideoFormat format) {
    std::array<char, 32> rate{};
    std::snprintf(rate.data(), rate.size(), "%g", fps);
    std::vector<std::string> command{ffmpeg.string(), "-hide_banner", "-loglevel",
                                     "error",         "-y",           "-framerate",
                                     rate.data(),     "-i",           framePattern.string()};
    if (format == VideoFormat::ProRes) {
        // Profile 3: 422 HQ.
        command.insert(command.end(),
                       {"-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"});
    } else {
        // Even sizes, which 4:2:0 needs.
        command.insert(command.end(), {"-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                                       "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"});
    }
    command.push_back(video.string());
    return command;
}

}  // namespace ernie::app
