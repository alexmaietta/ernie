// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/viewer/shell/capture.h"

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <system_error>
#include <utility>

#include <fmt/format.h>

#include "ernie/io/image.h"
#include "ernie/util/log.h"

namespace ernie::app {

namespace fs = std::filesystem;

namespace {

constexpr size_t kRecentKept = 32;

const char* Extension(ImageFormat format) {
    switch (format) {
    case ImageFormat::Jpeg:
        return ".jpg";
    case ImageFormat::Exr:
        return ".exr";
    default:
        return ".png";
    }
}

std::string Quoted(const fs::path& path) {
    return "\"" + path.string() + "\"";
}

}

std::string ExpandPattern(std::string_view pattern, std::string_view scene, std::string_view camera,
                          int64_t frame) {
    const std::time_t now = std::time(nullptr);
    std::tm local{};
#if defined(_WIN32)
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    char date[16];
    char time[16];
    std::strftime(date, sizeof(date), "%Y-%m-%d", &local);
    std::strftime(time, sizeof(time), "%H-%M-%S", &local);
    std::string name;
    for (size_t at = 0; at < pattern.size();) {
        const size_t close = pattern[at] == '{' ? pattern.find('}', at) : std::string_view::npos;
        if (close == std::string_view::npos) {
            name += pattern[at++];
            continue;
        }
        const std::string_view token = pattern.substr(at + 1, close - at - 1);
        if (token == "scene") {
            name += scene;
        } else if (token == "camera") {
            name += camera;
        } else if (token == "date") {
            name += date;
        } else if (token == "time") {
            name += time;
        } else if (token == "frame") {
            name += fmt::format("{:04}", frame);
        } else {
            name += pattern.substr(at, close - at + 1);
        }
        at = close + 1;
    }
    std::ranges::replace_if(
        name,
        [](char c) {
            return c == '/' || c == '\\' || c == ':' || c == '*' || c == '?' || c == '"' ||
                   c == '<' || c == '>' || c == '|';
        },
        '_');
    return name.empty() ? std::string("capture") : name;
}

fs::path FreePath(const fs::path& path) {
    std::error_code error;
    if (!fs::exists(path, error)) {
        return path;
    }
    for (int copy = 1;; ++copy) {
        fs::path candidate = path.parent_path() / fmt::format("{}_{}{}", path.stem().string(), copy,
                                                              path.extension().string());
        if (!fs::exists(candidate, error)) {
            return candidate;
        }
    }
}

fs::path FindFfmpeg() {
    const char* found = std::getenv("PATH");
    if (found == nullptr) {
        return {};
    }
#if defined(_WIN32)
    constexpr char kSeparator = ';';
    constexpr const char* kName = "ffmpeg.exe";
#else
    constexpr char kSeparator = ':';
    constexpr const char* kName = "ffmpeg";
#endif
    const std::string_view paths(found);
    for (size_t start = 0; start <= paths.size();) {
        size_t end = paths.find(kSeparator, start);
        if (end == std::string_view::npos) {
            end = paths.size();
        }
        std::error_code error;
        const fs::path candidate = fs::path(paths.substr(start, end - start)) / kName;
        if (end > start && fs::is_regular_file(candidate, error)) {
            return candidate;
        }
        start = end + 1;
    }
    return {};
}

Capture::Capture() {
    std::error_code error;
    const char* home = std::getenv(
#if defined(_WIN32)
        "USERPROFILE"
#else
        "HOME"
#endif
    );
    const fs::path pictures = home != nullptr ? fs::path(home) / "Pictures" : fs::path();
    settings.folder = !pictures.empty() && fs::is_directory(pictures, error)
                          ? pictures / "Ernie"
                          : fs::current_path(error) / "captures";
}

Capture::~Capture() {
    if (m_encoder.joinable()) {
        m_encoder.join();
    }
}

void Capture::Screenshot() {
    m_screenshot = true;
}

void Capture::Record() {
    if (m_recording || m_encoding) {
        return;
    }
    m_recording = true;
    m_recorded = 0;
    m_sequence.clear();
}

void Capture::Stop() {
    if (m_recording) {
        m_recording = false;
        m_finishing = m_recorded > 0;
    }
}

void Capture::Update(const device::Tensor* shown, uint64_t frame, std::string_view scene,
                     std::string_view camera) {
    for (auto job = m_jobs.begin(); job != m_jobs.end();) {
        if (!job->read.Ready()) {
            ++job;
            continue;
        }
        Write(*job);
        job = m_jobs.erase(job);
    }
    if (m_finishing && m_jobs.empty()) {
        m_finishing = false;
        Encode();
    }
    if (shown == nullptr || shown->Empty()) {
        return;
    }
    const auto start = [&](fs::path path) {
        m_jobs.push_back({shown->ToHostAsync(), shown->Type(),
                          std::vector<int64_t>(shown->Shape().begin(), shown->Shape().end()),
                          std::move(path)});
    };
    const char* extension = Extension(settings.format);
    if (m_screenshot) {
        m_screenshot = false;
        start(FreePath(settings.folder /
                       (ExpandPattern(settings.pattern, scene, camera, 0) + extension)));
    }
    if (!m_recording || frame == m_lastFrame) {
        return;
    }
    m_lastFrame = frame;
    if (m_sequence.empty()) {
        m_sequence = FreePath(settings.folder / ExpandPattern(settings.pattern, scene, camera, 0));
    }
    start(m_sequence / fmt::format("frame_{:04}{}", m_recorded, extension));
    if (++m_recorded >= settings.frames) {
        Stop();
    }
}

void Capture::Write(const Job& job) {
    try {
        std::error_code error;
        fs::create_directories(job.path.parent_path(), error);
        const Span<const std::byte> bytes = job.read.Bytes();
        const Pixels pixels(bytes, job.type, job.shape.size() > 0 ? job.shape[0] : 1,
                            job.shape.size() > 1 ? job.shape[1] : 1,
                            job.shape.size() > 2 ? job.shape[2] : 1);
        if (job.path.extension() == ".exr") {
            WriteExr(job.path, pixels);
        } else if (job.path.extension() == ".jpg") {
            WriteJpeg(job.path, pixels);
        } else {
            WritePng(job.path, pixels);
        }
        if (job.path.parent_path() != m_sequence || m_sequence.empty()) {
            Log(LogLevel::Info, "saved {}", job.path.string());
            m_recent.insert(m_recent.begin(), job.path);
            m_recent.resize(std::min(m_recent.size(), kRecentKept));
        }
    } catch (const std::exception& error) {
        Log(LogLevel::Error, "couldn't write {}: {}", job.path.string(), error.what());
    }
}

void Capture::Encode() {
    const fs::path ffmpeg = FindFfmpeg();
    if (settings.video == VideoFormat::None || settings.format == ImageFormat::Exr ||
        ffmpeg.empty()) {
        Log(LogLevel::Info, "saved {} frames in {}", m_recorded, m_sequence.string());
        m_recent.insert(m_recent.begin(), m_sequence);
        return;
    }
    if (m_encoder.joinable()) {
        m_encoder.join();
    }
    const bool mp4 = settings.video == VideoFormat::Mp4;
    const fs::path video = FreePath(fs::path(m_sequence).concat(mp4 ? ".mp4" : ".mov"));
    const std::string codec = mp4 ? "-c:v libx264 -pix_fmt yuv420p -crf 18"
                                  : "-c:v prores_ks -profile:v 3 -pix_fmt yuv422p10le";
#if defined(_WIN32)
    constexpr const char* kQuiet = " >NUL 2>&1";
#else
    constexpr const char* kQuiet = " >/dev/null 2>&1";
#endif
    const std::string command =
        fmt::format("{} -y -framerate {} -i {} {} {}{}", Quoted(ffmpeg), settings.fps,
                    Quoted(m_sequence / (std::string("frame_%04d") + Extension(settings.format))),
                    codec, Quoted(video), kQuiet);
    m_recent.insert(m_recent.begin(), video);
    m_encoding = true;
    m_encoder =
        std::thread([this, command, video, frames = m_sequence, keep = settings.keepFrames] {
            const int status = std::system(
#if defined(_WIN32)
                ("\"" + command + "\"").c_str()
#else
                command.c_str()
#endif
            );
            if (status == 0) {
                Log(LogLevel::Info, "saved {}", video.string());
                if (!keep) {
                    std::error_code error;
                    fs::remove_all(frames, error);
                }
            } else {
                Log(LogLevel::Error, "ffmpeg couldn't encode {}; the frames are in {}",
                    video.string(), frames.string());
            }
            m_encoding = false;
        });
}

}
