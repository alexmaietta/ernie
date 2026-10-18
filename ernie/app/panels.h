// SPDX-License-Identifier: BSD-3-Clause
/// @file panels.h
/// @brief The viewer's built-in panels, each given what it shows.
#pragma once

#include "ernie/app/panel.h"
#include "ernie/scene/path.h"

#include <memory>
#include <string_view>
#include <unordered_set>

namespace ernie {
class DeviceContext;
class Scene;
}  // namespace ernie

namespace ernie::app {

class MessageLog;
class Selection;
struct Theme;

/// @return The Log panel over `log`, filtered by severity and text.
std::unique_ptr<Panel> MakeLogPanel(MessageLog& log, const Theme& theme);

/// @return The Stage panel: the scene's prims as a tree, picking the selection.
std::unique_ptr<Panel> MakeStagePanel(const Scene& scene, Selection& selection);

/// @return The prims whose path contains `filter`, ignoring case, and their ancestors: what the
/// Stage panel lists for it. Empty for an empty filter, which lists everything.
std::unordered_set<Path> StageMatches(const Scene& scene, std::string_view filter);

/// @return The Profiler panel over Ernie's zone profiler, saving exports to the working directory
/// and saying so in `log`.
/// @param alwaysRecord Whether zones record while the panel is hidden too.
std::unique_ptr<Panel> MakeProfilerPanel(MessageLog& log, bool alwaysRecord);

/// @return The Devices panel: the device in use, or none, and every Vulkan device installed.
std::unique_ptr<Panel> MakeDevicePanel(const DeviceContext* device);

}  // namespace ernie::app
