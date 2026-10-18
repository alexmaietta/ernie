// SPDX-License-Identifier: BSD-3-Clause
/// @file main.cpp
/// @brief The ernie program.
#include "ernie/device/vulkan_devices.h"
#include "ernie/scene/reference.h"
#include "ernie/util/file.h"
#include "ernie/util/version.h"

#include <cstdio>
#include <exception>
#include <filesystem>
#include <string_view>
#include <vector>

namespace {

namespace fs = std::filesystem;

void PrintUsage() {
    std::puts("Usage: ernie [--version] [--devices] [--help]\n"
              "       ernie schema [--json-schema FILE] [--markdown DIR] [--usda DIR]\n"
              "       ernie validate SCENE.json...");
}

int PrintDevices() {
    for (const ernie::VulkanDeviceInfo& device : ernie::ListVulkanDevices()) {
        std::printf("%s (%s, Vulkan %u.%u%s)\n", device.name.c_str(), ernie::ToString(device.type),
                    device.apiMajor, device.apiMinor, device.rayQuery ? ", ray queries" : "");
    }
    return 0;
}

void Write(const fs::path& path, std::string_view text) {
    if (path.has_parent_path()) {
        fs::create_directories(path.parent_path());
    }
    ernie::WriteTextFile(path, text);
}

/// Writes the views of the schema registry that `arguments` ask for.
int Schema(const std::vector<std::string_view>& arguments) {
    if (arguments.empty() || arguments.size() % 2 != 0) {
        PrintUsage();
        return 2;
    }
    for (std::size_t i = 0; i < arguments.size(); i += 2) {
        const fs::path out(arguments[i + 1]);
        if (arguments[i] == "--json-schema") {
            Write(out, ernie::SceneJsonSchema());
        } else if (arguments[i] == "--markdown") {
            for (const ernie::SchemaFamily& family : ernie::SchemaFamilies()) {
                Write(out / (std::string(family.name) + ".md"), ernie::ReferencePage(family));
            }
        } else if (arguments[i] == "--usda") {
            const ernie::UsdSchemaPlugin plugin = ernie::ErnieUsdSchemaPlugin();
            Write(out / "generatedSchema.usda", plugin.generatedSchema);
            Write(out / "plugInfo.json", plugin.plugInfo);
        } else {
            PrintUsage();
            return 2;
        }
    }
    return 0;
}

/// Prints every problem of the scene files; fails if any has one.
int Validate(const std::vector<std::string_view>& files) {
    if (files.empty()) {
        PrintUsage();
        return 2;
    }
    int status = 0;
    for (const std::string_view file : files) {
        for (const ernie::SceneProblem& problem :
             ernie::ValidateScene(ernie::ReadTextFile(fs::path(file)))) {
            std::printf("%.*s: %s\n", static_cast<int>(file.size()), file.data(),
                        problem.Message().c_str());
            status = 1;
        }
    }
    return status;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string_view option = argc > 1 ? argv[1] : "--help";
    const std::vector<std::string_view> rest(argv + std::min(argc, 2), argv + argc);
    try {
        if (option == "--version") {
            std::printf("ernie %.*s\n", static_cast<int>(ernie::Version().size()),
                        ernie::Version().data());
            return 0;
        }
        if (option == "--devices") {
            return PrintDevices();
        }
        if (option == "schema") {
            return Schema(rest);
        }
        if (option == "validate") {
            return Validate(rest);
        }
        PrintUsage();
        return option == "--help" ? 0 : 2;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ernie: %s\n", error.what());
        return 1;
    }
}
