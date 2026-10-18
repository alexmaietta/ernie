// SPDX-License-Identifier: BSD-3-Clause
/// @file main.cpp
/// @brief The ernie program.
#include "ernie/device/vulkan_devices.h"
#include "ernie/util/version.h"

#include <cstdio>
#include <exception>
#include <string_view>

namespace {

void PrintUsage() {
    std::puts("Usage: ernie [--version] [--devices] [--help]");
}

int PrintDevices() {
    for (const ernie::VulkanDeviceInfo& device : ernie::ListVulkanDevices()) {
        std::printf("%s (%s, Vulkan %u.%u%s)\n", device.name.c_str(), ernie::ToString(device.type),
                    device.apiMajor, device.apiMinor, device.rayQuery ? ", ray queries" : "");
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string_view option = argc > 1 ? argv[1] : "--help";
    try {
        if (option == "--version") {
            std::printf("ernie %.*s\n", static_cast<int>(ernie::Version().size()),
                        ernie::Version().data());
            return 0;
        }
        if (option == "--devices") {
            return PrintDevices();
        }
        PrintUsage();
        return option == "--help" ? 0 : 2;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ernie: %s\n", error.what());
        return 1;
    }
}
