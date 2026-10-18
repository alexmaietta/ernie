// SPDX-License-Identifier: BSD-3-Clause
/// @file view.cpp
/// @brief Parsing the viewer's command line and running it on a scene.
#include "ernie/app/view.h"

#include "ernie/scene/json.h"
#include "ernie/scene/scene.h"

#include <charconv>
#include <stdexcept>
#include <string>

namespace ernie::app {
namespace {

template <typename Number> Number ParseNumber(std::string_view text, std::string_view option) {
    Number value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc() || end != text.data() + text.size()) {
        throw std::invalid_argument(std::string(option) + " takes a number, not " +
                                    std::string(text));
    }
    return value;
}

glm::uvec2 ParseSize(std::string_view text) {
    const std::size_t x = text.find('x');
    if (x == std::string_view::npos) {
        throw std::invalid_argument("--size takes WIDTHxHEIGHT, not " + std::string(text));
    }
    const glm::uvec2 size(ParseNumber<std::uint32_t>(text.substr(0, x), "--size"),
                          ParseNumber<std::uint32_t>(text.substr(x + 1), "--size"));
    if (size.x == 0 || size.y == 0) {
        throw std::invalid_argument("--size needs a positive width and height");
    }
    return size;
}

}  // namespace

std::string_view ViewUsage() {
    return "ernie [view] [SCENE.json] [--offscreen] [--frames N] [--size WxH] [--device NAME]\n"
           "             [--validation] [--no-vsync] [--profile]";
}

ViewCommand ParseViewCommand(std::span<const std::string_view> arguments) {
    ViewCommand command;
    ViewerOptions& viewer = command.viewer;
    for (std::size_t i = 0; i < arguments.size(); ++i) {
        const std::string_view argument = arguments[i];
        const auto value = [&]() {
            if (i + 1 >= arguments.size()) {
                throw std::invalid_argument(std::string(argument) + " needs a value");
            }
            return arguments[++i];
        };
        if (argument == "--offscreen") {
            viewer.offscreen = true;
        } else if (argument == "--frames") {
            viewer.frames = ParseNumber<std::uint64_t>(value(), argument);
        } else if (argument == "--size") {
            viewer.size = ParseSize(value());
        } else if (argument == "--device") {
            viewer.device = value();
        } else if (argument == "--validation") {
            viewer.validation = true;
        } else if (argument == "--no-vsync") {
            viewer.vsync = false;
        } else if (argument == "--profile") {
            viewer.shell.profile = true;
        } else if (argument.starts_with("-")) {
            throw std::invalid_argument("Unknown option " + std::string(argument));
        } else if (command.scene) {
            throw std::invalid_argument("The viewer opens one scene");
        } else {
            command.scene = std::filesystem::path(argument);
        }
    }
    if (command.scene) {
        viewer.title = "Ernie - " + command.scene->filename().string();
    }
    return command;
}

int RunView(const ViewCommand& command) {
    Scene scene;
    if (command.scene) {
        LoadSceneFile(scene, *command.scene);
    }
    scene.Update();
    Viewer viewer(scene, command.viewer);
    if (command.scene) {
        viewer.GetShell().Log().Info("Opened " + command.scene->string());
    }
    viewer.Run();
    return 0;
}

}  // namespace ernie::app
