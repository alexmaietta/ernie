// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/cli.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <exception>
#include <filesystem>
#include <optional>
#include <string_view>

#include <fmt/format.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

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

constexpr double kStep = 1.0 / 60;

std::string WorldPath(uint32_t world) {
    return world == 0 ? std::string("/World") : fmt::format("/World_{}", world);
}

void PrintProfile() {
    const profiler::Report report = profiler::Summarize(profiler::Frames());
    std::fputs(profiler::ReportText(report).c_str(), stderr);
}

Ref<Scene> DefaultScene(device::Device& device, uint32_t worlds, bool waitForKernels) {
    const Ref<Scene> scene = Scene::Create(device, {.waitForKernels = waitForKernels});
    if (worlds > 1) {
        scene->Define<World>("/World");
    } else {
        scene->Define<Xform>("/World");
    }
    scene->Define<Scope>("/World/Looks");
    const std::pair<const char*, Float3> looks[] = {{"Red", {0.8f, 0.3f, 0.2f}},
                                                    {"Blue", {0.2f, 0.5f, 0.8f}},
                                                    {"Green", {0.3f, 0.7f, 0.3f}},
                                                    {"Yellow", {0.9f, 0.7f, 0.2f}},
                                                    {"Purple", {0.6f, 0.4f, 0.8f}}};
    for (const auto& [name, color] : looks) {
        const std::string look = fmt::format("/World/Looks/{}", name);
        const Prim material = scene->Define<Material>(look);
        const Prim surface = scene->Define<Shader>(look + "/Surface", {.id = "UsdPreviewSurface"});
        surface.Set(UsdPreviewSurface::kDiffuseColor, color);
        material.Connect("surface", surface, "surface");
    }
    scene->Define<Plane>("/World/Ground", {.checkerboard = true,
                                           .checkerColorA = {0.6f, 0.6f, 0.6f},
                                           .checkerColorB = {0.3f, 0.3f, 0.3f},
                                           .width = 0,
                                           .length = 0});
    scene->Define<Cube>("/World/Cube",
                        {.materialBinding = {"/World/Looks/Red"}, .translate = {-6, 0, 1}});
    scene->Define<Sphere>("/World/Sphere",
                          {.materialBinding = {"/World/Looks/Blue"}, .translate = {-3, 0, 1}});
    scene->Define<Cylinder>("/World/Cylinder",
                            {.materialBinding = {"/World/Looks/Green"}, .translate = {0, 0, 1}});
    scene->Define<Cone>("/World/Cone",
                        {.materialBinding = {"/World/Looks/Yellow"}, .translate = {3, 0, 1}});
    scene->Define<Capsule>("/World/Capsule",
                           {.materialBinding = {"/World/Looks/Purple"}, .translate = {6, 0, 1}});
    scene->Define<DistantLight>("/World/Sun", {.orient = FromAzimuthElevation(146, 44)});

    const Double3 eye{0, -15.6, 3.4};
    scene->Define<Camera>("/World/Camera", {.translate = eye,
                                            .orient = LookAt(eye, {0, 0, 1}),
                                            .focalLength = 29,
                                            .horizontalAperture = 24.0f * 16 / 9,
                                            .verticalAperture = 24});

    scene->Clone(*scene->Find("/World"), worlds - 1);
    return scene;
}

std::string ChooseProduct(Scene& scene, const Command& command) {
    if (!command.product.empty()) {
        const std::optional<Prim> product = scene.Find(command.product);
        if (!product || !product->IsA(SchemaId::RenderProduct)) {
            throw Error("The scene has no render product {}", command.product);
        }
        return command.product;
    }
    const bool viewer = command.kind == Command::Kind::Viewer;
    if (viewer && command.camera.empty() && command.preset.empty() && command.aovs.empty()) {
        return {};
    }

    std::vector<std::string> cameras;
    if (!command.camera.empty()) {
        const std::optional<Prim> camera = scene.Find(command.camera);
        if (!camera || !camera->IsA(SchemaId::Camera)) {
            throw Error("The scene has no camera {}", command.camera);
        }
        cameras.push_back(command.camera);
    } else {
        for (uint32_t world = 0; world < command.worlds; ++world) {
            cameras.push_back(WorldPath(world) + "/Camera");
        }
    }

    scene.Define<Scope>("/Render");
    std::vector<std::string> vars;
    std::vector<std::string> names = {"LdrColor"};
    names.insert(names.end(), command.aovs.begin(), command.aovs.end());
    for (const std::string& name : names) {
        const std::string path = "/Render/" + name;
        if (std::ranges::find(vars, path) == vars.end()) {
            if (!FindRenderVarSource(name)) {
                throw Error("\"{}\" is no AOV; try LdrColor, HdrColor, DistanceToCamera or Normal",
                            name);
            }
            scene.Define<RenderVar>(path, {.sourceName = name});
            vars.push_back(path);
        }
    }
    const std::string product = "/Render/Product";
    scene.Define<RenderProduct>(
        product,
        {.renderMode = command.preset == "minimal" ? RenderMode::Minimal : RenderMode::RealTime,
         .camera = cameras,
         .resolution = {command.width, command.height},
         .orderedVars = vars});
    return product;
}

int RenderFrames(const Ref<Scene>& scene, const std::string& product, const Command& command) {
    Renderer renderer(scene);
    profiler::SetEnabled(command.profile);
    const uint32_t frames = std::max(command.frames, 1u);
    RenderOutput output;
    for (uint32_t frame = 0; frame < frames; ++frame) {
        profiler::BeginFrame();
        output = renderer.Step(product, kStep);
    }
    output.Wait();
    if (!command.output.empty()) {
        for (const std::filesystem::path& written :
             WriteFrame(command.output, output[product].frames.back())) {
            Log(LogLevel::Info, "wrote {}", written.string());
        }
    }
    if (command.profile) {
        PrintProfile();
    }
    return 0;
}

int Benchmark(const Ref<Scene>& scene, const std::string& product, const Command& command) {
    using Clock = std::chrono::steady_clock;
    Renderer renderer(scene);
    const uint32_t steps = command.frames > 0 ? command.frames : 100;
    int64_t pixels = 0;
    for (uint32_t step = 0; step < std::max(command.warmup, 1u); ++step) {
        const RenderOutput output = renderer.Step(product, kStep);
        output.Wait();
        const ProductOutput& rendered = output[product];
        if (!rendered.frames.empty() && !rendered.frames.front().renderVars.empty()) {
            const Span<const int64_t> shape =
                rendered.frames.front().renderVars.begin()->second.tensor.Shape();
            pixels = shape.size() >= 3 ? shape[0] * shape[1] * shape[2] : 0;
        }
    }

    profiler::SetEnabled(command.profile);
    const Clock::time_point serialStart = Clock::now();
    for (uint32_t step = 0; step < steps; ++step) {
        profiler::BeginFrame();
        renderer.Step(product, kStep).Wait();
    }
    const double latencyMs =
        std::chrono::duration<double, std::milli>(Clock::now() - serialStart).count() / steps;

    std::deque<RenderOutput> inFlight;
    const Clock::time_point pipelinedStart = Clock::now();
    for (uint32_t step = 0; step < steps; ++step) {
        profiler::BeginFrame();
        inFlight.push_back(renderer.Step(product, kStep));
        if (inFlight.size() == 3) {
            inFlight.front().Wait();
            inFlight.pop_front();
        }
    }
    for (const RenderOutput& output : inFlight) {
        output.Wait();
    }
    const double seconds = std::chrono::duration<double>(Clock::now() - pipelinedStart).count();
    const double fps = steps / seconds;
    const double megapixels = double(pixels) * fps / 1e6;

    if (command.json) {
        std::printf("{\"product\": \"%s\", \"worlds\": %u, \"pixels\": %lld, \"warmup\": %u, "
                    "\"steps\": %u, \"latency_ms\": %.4f, \"fps\": %.2f, "
                    "\"megapixels_per_second\": %.2f}\n",
                    product.c_str(), command.worlds, static_cast<long long>(pixels), command.warmup,
                    steps, latencyMs, fps, megapixels);
    } else {
        std::printf("%s, %lld pixels a step\n", product.c_str(), static_cast<long long>(pixels));
        std::printf("  one step at a time  %.3f ms\n", latencyMs);
        std::printf("  three in flight     %.1f steps a second, %.1f megapixels a second\n", fps,
                    megapixels);
    }
    std::fflush(stdout);
    if (command.profile) {
        PrintProfile();
    }
    return 0;
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
