// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/viewer/cli/cli.h"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <optional>
#include <string_view>

#include "ernie/ernie.h"
#include "ernie/viewer/viewer.h"
#include "ernie/viewer/viewport/orbit_camera.h"

namespace ernie::app {
namespace {

constexpr const char* kHelp = R"(Usage: ernie [options]

Opens the viewer on the default scene.

Options
  -h, --help                Show this help
  --version                 Show the build's version and commit
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

Ref<Scene> DefaultScene(device::Device& device) {
    const Ref<Scene> scene = Scene::Create(device, {.waitForKernels = false});
    scene->Define<Xform>("/World");
    scene->Define<Scope>("/World/Looks");
    scene->Define<Material>(
        "/World/Looks/Red",
        {.id = "UsdPreviewSurface", .inputs = {{"diffuseColor", Float3{0.8f, 0.3f, 0.2f}}}});
    scene->Define<Material>(
        "/World/Looks/Blue",
        {.id = "UsdPreviewSurface", .inputs = {{"diffuseColor", Float3{0.2f, 0.5f, 0.8f}}}});
    scene->Define<Material>(
        "/World/Looks/Green",
        {.id = "UsdPreviewSurface", .inputs = {{"diffuseColor", Float3{0.3f, 0.7f, 0.3f}}}});
    scene->Define<Material>(
        "/World/Looks/Yellow",
        {.id = "UsdPreviewSurface", .inputs = {{"diffuseColor", Float3{0.9f, 0.7f, 0.2f}}}});
    scene->Define<Material>(
        "/World/Looks/Purple",
        {.id = "UsdPreviewSurface", .inputs = {{"diffuseColor", Float3{0.6f, 0.4f, 0.8f}}}});
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
    const double halfFov = std::numbers::pi / 8;
    const OrbitCamera view{
        .target = {0, 0, 1}, .yaw = 0.6, .pitch = 0.45, .distance = 1.1 * 5.5 / std::sin(halfFov)};
    scene->Define<Camera>("/World/Camera", {.translate = view.Eye(),
                                            .orient = view.Orient(),
                                            .focalLength = float(12 / std::tan(halfFov)),
                                            .horizontalAperture = 24.0f * 16 / 9,
                                            .verticalAperture = 24});
    return scene;
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
        } else if (flag == "--version") {
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
    device::DeviceDesc desc = command.device;
    if (desc.adapter.empty()) {
        desc.backend = device::Backend::Vulkan;
    }
    for (const device::AdapterInfo& adapter : device::EnumerateAdapters()) {
        if (adapter.alias == desc.adapter) {
            desc.backend = adapter.backend;
        }
    }
    desc.presentation = desc.backend == device::Backend::Vulkan;
    const Ref<device::Device> device = device::Device::Create(desc);
    Viewer viewer(DefaultScene(*device), {.aov = command.aov,
                                          .width = command.width,
                                          .height = command.height,
                                          .vsync = command.vsync,
                                          .layout = command.layout});
    viewer.Run();
    if (!device::CurrentBuilds().Done()) {
        std::fflush(nullptr);
        std::_Exit(0);
    }
    return 0;
}

}
