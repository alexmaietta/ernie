// SPDX-License-Identifier: BSD-3-Clause
/// @file hud.h
/// @brief The heads-up display at the viewport's top left.
#pragma once

#include "ernie/app/orbit_camera.h"

#include <imgui.h>

#include <string>
#include <vector>

namespace ernie::app {

/// @brief Frame rate, viewport size and camera, refreshed a few times a second so they stay
/// readable, drawn behind every window.
class Hud {
  public:
    /// @param origin, size The viewport's rectangle on the screen.
    void Draw(ImVec2 origin, ImVec2 size, const OrbitCamera& camera);

  private:
    void Refresh(ImVec2 size, const OrbitCamera& camera);

    std::vector<std::string> lines_;
    double refreshed_ = -1.0;
};

}  // namespace ernie::app
