// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "ernie/device/device.h"
#include "ernie/viewer_export.h"

namespace ernie::app {

struct Command {
    enum class Kind : uint8_t { Viewer, Help, Version, Devices };

    Kind kind = Kind::Viewer;
    device::DeviceDesc device;
    uint32_t width = 1280;
    uint32_t height = 720;
    bool vsync = false;
    std::string aov = "LdrColor";
    std::filesystem::path layout;
};

ERNIE_VIEWER_API Command ParseCommand(const std::vector<std::string>& args);

ERNIE_VIEWER_API int RunCommand(const Command& command);

}
