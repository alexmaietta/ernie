// SPDX-License-Identifier: BSD-3-Clause
/// @file navigation.h
/// @brief Moving the viewer's camera with the mouse over the viewport.
#pragma once

#include "ernie/app/orbit_camera.h"

namespace ernie::app {

/// @brief Mouse controls of an orbit camera, as Ernie 1's viewer has them: a left drag orbits, a
/// right drag pans along the ground, a middle drag pans across the view and the wheel zooms. A
/// drag that starts over the viewport keeps control until its button is released.
class Navigation {
  public:
    /// @brief Applies this frame's mouse input from ImGui.
    /// @param hovered Whether the mouse is over the viewport and no window.
    /// @param height  The viewport's height in pixels, so pans follow the cursor.
    /// @return Whether the camera moved.
    bool Update(OrbitCamera& camera, bool hovered, float height);

  private:
    bool dragging_ = false;
};

}  // namespace ernie::app
