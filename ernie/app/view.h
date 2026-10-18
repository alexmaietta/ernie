// SPDX-License-Identifier: BSD-3-Clause
/// @file view.h
/// @brief The command line that opens the viewer: `ernie [view] [SCENE.json] [options]`.
#pragma once

#include "ernie/app/viewer.h"

#include <filesystem>
#include <optional>
#include <span>
#include <string_view>

namespace ernie::app {

/// @brief What the viewer's command line asks for.
struct ViewCommand {
    /// The scene file to open; none for an empty scene.
    std::optional<std::filesystem::path> scene;
    ViewerOptions viewer;
};

/// @return The options the viewer's command line usage lists.
std::string_view ViewUsage();

/// @brief Reads the viewer's arguments: an optional scene file and options.
/// @throws std::invalid_argument for an unknown option, a missing value or a second scene.
ViewCommand ParseViewCommand(std::span<const std::string_view> arguments);

/// @brief Opens the viewer on the command's scene and runs it until closed.
/// @return The process's exit status.
int RunView(const ViewCommand& command);

}  // namespace ernie::app
