// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/capture.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <system_error>

#ifdef _WIN32
#include <process.h>
#else
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;
#endif

namespace ernie::app {

namespace {

/// Characters file names cannot hold on some system.
bool Forbidden(char c) {
    return c == '/' || c == '\\' || c == ':' || c == '*' ||
           c == '?' || c == '"' || c == '<' || c == '>' || c == '|' ||
           static_cast<unsigned char>(c) < 32;
}

std::tm LocalTime(std::chrono::system_clock::time_point when) {
    const std::time_t seconds =
        std::chrono::system_clock::to_time_t(when);
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &seconds);
#else
    localtime_r(&seconds, &local);
#endif
    return local;
}

}  // namespace

std::string ExpandCaptureName(std::string_view pattern,
                              const CaptureNameFields& fields) {
    const std::tm local = LocalTime(fields.when);
    char date[16];
    char time[16];
    char frame[32];
    std::strftime(date, sizeof(date), "%Y-%m-%d", &local);
    std::strftime(time, sizeof(time), "%H-%M-%S", &local);
    std::snprintf(frame, sizeof(frame), "%04lld",
                  static_cast<long long>(fields.frame));
    std::string name;
    for (std::size_t i = 0; i < pattern.size(); ++i) {
        const std::size_t close = pattern.find('}', i);
        if (pattern[i] == '{' && close != std::string_view::npos) {
            const std::string_view key =
                pattern.substr(i + 1, close - i - 1);
            const std::string value =
                key == "scene"    ? fields.scene
                : key == "camera" ? fields.camera
                : key == "date"   ? date
                : key == "time"   ? time
                : key == "frame"
                    ? frame
                    : std::string(pattern.substr(i, close - i + 1));
            name += value;
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
    const std::filesystem::path stem =
        path.parent_path() / path.stem();
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
#ifdef _WIN32
    const char* home = std::getenv("USERPROFILE");
#else
    const char* home = std::getenv("HOME");
#endif
    std::error_code error;
    if (home != nullptr && *home != '\0') {
        const std::filesystem::path pictures =
            std::filesystem::path(home) / "Pictures";
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

std::filesystem::path FindFfmpeg() {
    const char* path = std::getenv("PATH");
    if (path == nullptr) {
        return {};
    }
#ifdef _WIN32
    constexpr char kSeparator = ';';
    constexpr const char* kName = "ffmpeg.exe";
#else
    constexpr char kSeparator = ':';
    constexpr const char* kName = "ffmpeg";
#endif
    const std::string dirs(path);
    std::size_t begin = 0;
    while (begin <= dirs.size()) {
        std::size_t end = dirs.find(kSeparator, begin);
        if (end == std::string::npos) {
            end = dirs.size();
        }
        if (end > begin) {
            const std::filesystem::path candidate =
                std::filesystem::path(
                    dirs.substr(begin, end - begin)) /
                kName;
            std::error_code error;
            if (std::filesystem::is_regular_file(candidate, error)) {
                return candidate;
            }
        }
        begin = end + 1;
    }
    return {};
}

std::vector<std::string>
EncodeCommand(const std::filesystem::path& ffmpeg,
              const std::filesystem::path& framePattern, double fps,
              const std::filesystem::path& video,
              VideoFormat format) {
    char rate[32];
    std::snprintf(rate, sizeof(rate), "%g", fps);
    std::vector<std::string> command = {ffmpeg.string(),
                                        "-hide_banner",
                                        "-loglevel",
                                        "error",
                                        "-y",
                                        "-framerate",
                                        rate,
                                        "-i",
                                        framePattern.string()};
    if (format == VideoFormat::ProRes) {
        // Profile 3: 422 HQ.
        command.insert(command.end(),
                       {"-c:v", "prores_ks", "-profile:v", "3",
                        "-pix_fmt", "yuv422p10le"});
    } else {
        // Even sizes, which 4:2:0 needs.
        command.insert(command.end(),
                       {"-c:v", "libx264", "-crf", "18", "-pix_fmt",
                        "yuv420p", "-vf",
                        "pad=ceil(iw/2)*2:ceil(ih/2)*2"});
    }
    command.push_back(video.string());
    return command;
}

int RunProcess(const std::vector<std::string>& command) {
    if (command.empty()) {
        return -1;
    }
#ifdef _WIN32
    // The C runtime joins the arguments with spaces: quote each.
    std::vector<std::string> quoted;
    for (const std::string& argument : command) {
        std::string q = "\"";
        for (const char c : argument) {
            if (c == '"') {
                q += '\\';
            }
            q += c;
        }
        quoted.push_back(q + "\"");
    }
    std::vector<const char*> argv;
    for (const std::string& argument : quoted) {
        argv.push_back(argument.c_str());
    }
    argv.push_back(nullptr);
    const intptr_t status =
        _spawnv(_P_WAIT, command.front().c_str(), argv.data());
    return status < 0 ? -1 : static_cast<int>(status);
#else
    std::vector<char*> argv;
    for (const std::string& argument : command) {
        argv.push_back(const_cast<char*>(argument.c_str()));
    }
    argv.push_back(nullptr);
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO,
                                     "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_addopen(&actions, STDERR_FILENO,
                                     "/dev/null", O_WRONLY, 0);
    pid_t pid = 0;
    const int spawned = posix_spawn(&pid, argv.front(), &actions,
                                    nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    if (spawned != 0) {
        return -1;
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        return -1;
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}

}  // namespace ernie::app
