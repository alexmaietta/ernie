// SPDX-License-Identifier: BSD-3-Clause
/// @file main.cpp
/// @brief The ernie program: the viewer, and the schema and scene tools.
#include "ernie/app/view.h"
#include "ernie/device/device.h"
#include "ernie/scene/reference.h"
#include "ernie/util/file.h"
#include "ernie/util/version.h"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace fs = std::filesystem;

void PrintUsage() {
    std::printf("Usage: %.*s\n"
                "       ernie --version | --devices | --help\n"
                "       ernie schema [--json-schema FILE] [--markdown DIR] [--usda DIR]\n"
                "       ernie validate SCENE.json...\n",
                static_cast<int>(ernie::app::ViewUsage().size()), ernie::app::ViewUsage().data());
}

int PrintDevices() {
    for (const ernie::VulkanDeviceInfo& device : ernie::ListVulkanDevices()) {
        std::printf("%s (%s, Vulkan %u.%u%s)\n", device.name.c_str(), ernie::ToString(device.type),
                    device.apiMajor, device.apiMinor,
                    device.capabilities.rayQuery ? ", ray queries" : "");
    }
    return 0;
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
            ernie::WriteTextFile(out, ernie::SceneJsonSchema());
        } else if (arguments[i] == "--markdown") {
            for (const ernie::SchemaFamily& family : ernie::SchemaFamilies()) {
                ernie::WriteTextFile(out / (std::string(family.name) + ".md"),
                                     ernie::ReferencePage(family));
            }
        } else if (arguments[i] == "--usda") {
            const ernie::UsdSchemaPlugin plugin = ernie::ErnieUsdSchemaPlugin();
            ernie::WriteTextFile(out / "generatedSchema.usda", plugin.generatedSchema);
            ernie::WriteTextFile(out / "plugInfo.json", plugin.plugInfo);
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

/// Opens the viewer; only `ernie`, `ernie view` and a scene file or viewer options get here.
int View(const std::vector<std::string_view>& arguments) {
    ernie::app::ViewCommand command;
    try {
        command = ernie::app::ParseViewCommand(arguments);
    } catch (const std::invalid_argument& error) {
        std::fprintf(stderr, "ernie: %s\n", error.what());
        PrintUsage();
        return 2;
    }
    return ernie::app::RunView(command);
}

}  // namespace

int main(int argc, char** argv) {
    const std::vector<std::string_view> arguments(argv + std::min(argc, 1), argv + argc);
    const std::string_view command = arguments.empty() ? "" : arguments.front();
    const std::vector<std::string_view> rest(arguments.begin() + (arguments.empty() ? 0 : 1),
                                             arguments.end());
    try {
        if (command == "--version") {
            std::printf("ernie %.*s\n", static_cast<int>(ernie::Version().size()),
                        ernie::Version().data());
            return 0;
        }
        if (command == "--help") {
            PrintUsage();
            return 0;
        }
        if (command == "--devices") {
            return PrintDevices();
        }
        if (command == "schema") {
            return Schema(rest);
        }
        if (command == "validate") {
            return Validate(rest);
        }
        return View(command == "view" ? rest : arguments);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ernie: %s\n", error.what());
        return 1;
    }
}
