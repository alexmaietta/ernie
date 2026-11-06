// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/cli.h"

#include <charconv>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <optional>
#include <string_view>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "ernie/ernie.h"

namespace ernie::app {
namespace {

constexpr const char* kHelp = R"(Usage: ernie [options]

Opens the viewer on the default scene, if this Ernie has one.

Options
  -h, --help                Show this help
  -V, --version             Show the build's version and commit
  -v, --verbose             Log debug messages too
  -q, --quiet               Log only warnings and errors
  --devices                 List the adapters and what each supports
  --device ALIAS            Render on an adapter from --devices, such as vulkan:0 or cuda:0
  --validation              Turn on the backend's validation
  --traversal MODE          Ray traversal, auto, hardware or software
  --sampling MODE           Texture sampling, auto, hardware or software
  --size WxH                Window size, such as 1600x900
  --vsync                   Wait for the display's refresh when presenting
  --aov NAME                The AOV shown, such as LdrColor, Normal or DistanceToCamera
  --layout FILE             A viewer layout to open with, saved from Window > Save Layout
)";

uint32_t ParseNumber(std::string_view text, std::string_view flag) {
    uint32_t number = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), number);
    if (error != std::errc() || end != text.data() + text.size() || number == 0) {
        throw Error("{} takes a number above 0, not \"{}\"", flag, text);
    }
    return number;
}

template <class Mode> Mode ParseMode(std::string_view text, std::string_view flag) {
    if (text == "auto") {
        return Mode::Auto;
    }
    if (text == "hardware") {
        return Mode::Hardware;
    }
    if (text == "software") {
        return Mode::Software;
    }
    throw Error("{} takes auto, hardware or software, not \"{}\"", flag, text);
}

void PrintDevices() {
    for (const device::AdapterInfo& adapter : device::EnumerateAdapters()) {
        std::printf(
            "%-10s %s, %.1f GiB%s%s%s\n", adapter.alias.c_str(), adapter.name.c_str(),
            double(adapter.memoryBytes) / double(1ull << 30),
            adapter.features.presentation ? ", presents" : "",
            adapter.unsupported.empty() ? "" : ", unsupported: ", adapter.unsupported.c_str());
    }
}

RunViewer FindViewer() {
#if defined(_WIN32)
    HMODULE self = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCWSTR>(&FindViewer), &self);
    wchar_t path[MAX_PATH] = {};
    GetModuleFileNameW(self, path, MAX_PATH);
    const std::filesystem::path library =
        std::filesystem::path(path).parent_path() / "ernie_viewer.dll";
    const HMODULE viewer =
        LoadLibraryExW(library.c_str(), nullptr,
                       LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    return viewer ? reinterpret_cast<RunViewer>(GetProcAddress(viewer, kRunViewer)) : nullptr;
#else
#if defined(__APPLE__)
    const char* const names[] = {"@rpath/libernie_viewer." ERNIE_SOVERSION ".dylib",
                                 "@rpath/libernie_viewer.dylib"};
#else
    const char* const names[] = {"libernie_viewer.so." ERNIE_SOVERSION, "libernie_viewer.so"};
#endif
    for (const char* name : names) {
        if (void* viewer = dlopen(name, RTLD_NOW | RTLD_LOCAL)) {
            return reinterpret_cast<RunViewer>(dlsym(viewer, kRunViewer));
        }
    }
    return nullptr;
#endif
}

}

Command ParseCommand(const std::vector<std::string>& args) {
    Command command;
    for (size_t at = 0; at < args.size(); ++at) {
        std::string_view flag = args[at];
        std::optional<std::string_view> attached;
        if (const size_t equals = flag.find('='); flag.starts_with("--") && equals != flag.npos) {
            attached = flag.substr(equals + 1);
            flag = flag.substr(0, equals);
        }
        bool taken = false;
        const auto value = [&]() -> std::string_view {
            taken = true;
            if (attached) {
                return *attached;
            }
            if (++at == args.size()) {
                throw Error("{} needs a value", flag);
            }
            return args[at];
        };
        if (flag == "-h" || flag == "--help") {
            command.kind = Command::Kind::Help;
        } else if (flag == "-v" || flag == "--verbose" || flag == "-q" || flag == "--quiet") {
            const LogLevel level =
                flag == "-v" || flag == "--verbose" ? LogLevel::Debug : LogLevel::Warning;
            if (command.logLevel && *command.logLevel != level) {
                throw Error("--verbose and --quiet can't be used together");
            }
            command.logLevel = level;
        } else if (flag == "-V" || flag == "--version") {
            command.kind = Command::Kind::Version;
        } else if (flag == "--devices") {
            command.kind = Command::Kind::Devices;
        } else if (flag == "--device") {
            command.device.adapter = value();
        } else if (flag == "--validation") {
            command.device.validation = true;
        } else if (flag == "--traversal") {
            command.device.traversal = ParseMode<device::Traversal>(value(), flag);
        } else if (flag == "--sampling") {
            command.device.sampling = ParseMode<device::Sampling>(value(), flag);
        } else if (flag == "--size") {
            const std::string_view size = value();
            const size_t x = size.find('x');
            if (x == size.npos) {
                throw Error("--size takes WxH, such as 1600x900, not \"{}\"", size);
            }
            command.width = ParseNumber(size.substr(0, x), flag);
            command.height = ParseNumber(size.substr(x + 1), flag);
        } else if (flag == "--vsync") {
            command.vsync = true;
        } else if (flag == "--aov") {
            command.aov = value();
        } else if (flag == "--layout") {
            command.layout = value();
        } else {
            throw Error("Unknown option \"{}\"; --help lists them", args[at]);
        }
        if (attached && !taken) {
            throw Error("{} takes no value", flag);
        }
    }
    return command;
}

int RunCommand(const Command& command) {
    if (command.logLevel) {
        SetLogLevel(*command.logLevel);
    }
    switch (command.kind) {
    case Command::Kind::Help:
        std::fputs(kHelp, stdout);
        return 0;
    case Command::Kind::Version: {
        const BuildInfo& build = GetBuildInfo();
        std::printf("ernie %s (%.*s%s, %.*s)\n", build.version.ToString().c_str(),
                    int(build.commit.size()), build.commit.data(), build.modified ? "+" : "",
                    int(build.commitDate.size()), build.commitDate.data());
        return 0;
    }
    case Command::Kind::Devices:
        PrintDevices();
        return 0;
    case Command::Kind::Viewer:
        break;
    }
    const RunViewer run = FindViewer();
    if (run == nullptr) {
        throw Error(
            "This Ernie has no viewer; install the viewer with Ernie's ernie-viewer package "
            "or the ernie-renderer wheel. --help lists what runs without it");
    }
    std::string error;
    const int status = run(&command, &error);
    if (!error.empty()) {
        throw Error("{}", error);
    }
    return status;
}

int RunCli(const std::vector<std::string>& args) {
    Command command;
    try {
        command = ParseCommand(args);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ernie: %s\n", error.what());
        return 2;
    }
    try {
        return RunCommand(command);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ernie: %s\n", error.what());
        return 1;
    }
}

}
