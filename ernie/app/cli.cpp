// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/cli.h"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <optional>
#include <string_view>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "ernie/app/render.h"
#include "ernie/ernie.h"

namespace ernie::app {
namespace {

constexpr const char* kHelp = R"(Usage: ernie [options]

Opens the viewer on the default scene, if this Ernie has one, or renders it without a window.

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

Scene
  --worlds N                Clone the default scene into N worlds, each with its own camera
  --camera PATH             Render from this camera instead of each world's /World/Camera
  --product PATH            Render this render product as authored instead of Ernie's own
  --preset NAME             Shade with real-time, the default, or minimal
  --size WxH                The image or window size, such as 1600x900
  --aovs A,B                Also render these AOVs, such as DistanceToCamera,Normal

Without a window
  --headless                Render without opening the viewer
  --frames N                Steps to render without a window, 1 by default, or to time, 100
  --output DIR              Write the last frame's AOVs to DIR, which implies --headless
  --benchmark               Time steps one at a time and three in flight
  --warmup N                Steps a benchmark takes before timing, 10 by default
  --json                    Print a benchmark's timings as one JSON line
  --profile                 Print the profiler's report of the timed steps

Viewer
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
    bool headless = false;
    bool benchmark = false;
    std::string benchmarkOnly;
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
        } else if (flag == "--worlds") {
            command.worlds = ParseNumber(value(), flag);
        } else if (flag == "--camera") {
            command.camera = value();
        } else if (flag == "--product") {
            command.product = value();
        } else if (flag == "--preset") {
            command.preset = value();
            if (command.preset != "real-time" && command.preset != "minimal") {
                throw Error("--preset takes real-time or minimal, not \"{}\"", command.preset);
            }
        } else if (flag == "--aovs") {
            std::string_view list = value();
            while (!list.empty()) {
                const size_t comma = std::min(list.find(','), list.size());
                if (comma > 0) {
                    command.aovs.emplace_back(list.substr(0, comma));
                }
                list.remove_prefix(std::min(comma + 1, list.size()));
            }
        } else if (flag == "--headless") {
            headless = true;
        } else if (flag == "--frames") {
            command.frames = ParseNumber(value(), flag);
            headless = true;
        } else if (flag == "--output") {
            command.output = value();
            headless = true;
        } else if (flag == "--benchmark") {
            benchmark = true;
        } else if (flag == "--warmup") {
            command.warmup = ParseNumber(value(), flag);
            benchmarkOnly = benchmarkOnly.empty() ? std::string(flag) : benchmarkOnly;
        } else if (flag == "--json") {
            command.json = true;
            benchmarkOnly = benchmarkOnly.empty() ? std::string(flag) : benchmarkOnly;
        } else if (flag == "--profile") {
            command.profile = true;
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
    if (!benchmarkOnly.empty() && !benchmark) {
        throw Error("{} goes with --benchmark", benchmarkOnly);
    }
    if (benchmark && !command.output.empty()) {
        throw Error("--benchmark writes no images, so it takes no --output");
    }
    if (command.kind == Command::Kind::Viewer && (headless || benchmark)) {
        command.kind = benchmark ? Command::Kind::Benchmark : Command::Kind::Render;
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
        std::printf("ernie %.*s (%.*s%s, %.*s)\n", int(build.fullVersion.size()),
                    build.fullVersion.data(), int(build.commit.size()), build.commit.data(),
                    build.modified ? "+" : "", int(build.commitDate.size()),
                    build.commitDate.data());
        return 0;
    }
    case Command::Kind::Devices:
        PrintDevices();
        return 0;
    case Command::Kind::Viewer:
    case Command::Kind::Render:
    case Command::Kind::Benchmark:
        break;
    }

    const bool viewer = command.kind == Command::Kind::Viewer;
    RunViewer run = nullptr;
    device::DeviceDesc desc = command.device;
    if (viewer) {
        run = FindViewer();
        if (run == nullptr) {
            throw Error("This Ernie has no viewer; install the viewer with Ernie's ernie-viewer "
                        "package or the ernie-renderer wheel, or render with --headless");
        }
        if (desc.adapter.empty()) {
            desc.backend = device::Backend::Vulkan;
        }
        for (const device::AdapterInfo& adapter : device::EnumerateAdapters()) {
            if (adapter.alias == desc.adapter) {
                desc.backend = adapter.backend;
            }
        }
        desc.presentation = desc.backend == device::Backend::Vulkan;
    }
    const Ref<device::Device> device = device::Device::Create(desc);
    const Ref<Scene> scene = DefaultScene(*device, command.worlds, !viewer);
    const std::string product = ChooseProduct(*scene, command);
    if (command.kind == Command::Kind::Render) {
        return RenderFrames(scene, product, command);
    }
    if (command.kind == Command::Kind::Benchmark) {
        return Benchmark(scene, product, command);
    }
    std::string error;
    const int status = run(&command, scene.Get(), &product, &error);
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
    int status = 1;
    try {
        status = RunCommand(command);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ernie: %s\n", error.what());
    }
    if (!device::CurrentBuilds().Done()) {
        std::fflush(nullptr);
        std::_Exit(status);
    }
    return status;
}

}
