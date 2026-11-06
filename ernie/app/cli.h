// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "ernie/device/device.h"
#include "ernie/util/log.h"

namespace ernie::app {

struct Command {
    enum class Kind : uint8_t { Viewer, Help, Version, Devices };
    Kind kind = Kind::Viewer;
    std::optional<LogLevel> logLevel;
    device::DeviceDesc device;
    uint32_t width = 1280;
    uint32_t height = 720;
    bool vsync = false;
    std::string aov = "LdrColor";
    std::filesystem::path layout;
};

using RunViewer = int (*)(const Command* command, std::string* error);

inline constexpr const char* kRunViewer = "ErnieRunViewer";

Command ParseCommand(const std::vector<std::string>& args);
int RunCommand(const Command& command);
int RunCli(const std::vector<std::string>& args);

}
