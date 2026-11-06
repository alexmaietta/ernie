// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "ernie/device/device.h"
#include "ernie/util/log.h"

namespace ernie {
class Scene;
}

namespace ernie::app {

struct Command {
    enum class Kind : uint8_t { Viewer, Render, Benchmark, Help, Version, Devices };
    Kind kind = Kind::Viewer;
    std::optional<LogLevel> logLevel;
    device::DeviceDesc device;
    uint32_t width = 1280;
    uint32_t height = 720;
    bool sized = false;
    bool vsync = false;
    std::string aov = "LdrColor";
    std::filesystem::path layout;
    uint32_t frames = 0;
    std::filesystem::path output;
    std::vector<std::string> aovs;
    std::string camera;
    std::string product;
    std::string preset;
    uint32_t worlds = 1;
    uint32_t warmup = 10;
    bool json = false;
    bool profile = false;
};

using RunViewer = int (*)(const Command* command, Scene* scene, const std::string* product,
                          std::string* error);

inline constexpr const char* kRunViewer = "ErnieRunViewer";

Command ParseCommand(const std::vector<std::string>& args);
int RunCommand(const Command& command);
int RunCli(const std::vector<std::string>& args);

}
