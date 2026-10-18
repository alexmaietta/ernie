// SPDX-License-Identifier: BSD-3-Clause
/// @file imgui_session.h
/// @brief A Dear ImGui context set up as the viewer uses it.
#pragma once

#include "ernie/app/theme.h"

struct ImGuiContext;

namespace ernie::app {

/// @brief Makes and destroys an ImGui context with docking and keyboard navigation, the theme
/// applied and no settings file, current while it lives. One at a time.
class ImGuiSession {
  public:
    explicit ImGuiSession(const Theme& theme, float contentScale = 1.0f);
    ~ImGuiSession();
    ImGuiSession(const ImGuiSession&) = delete;
    ImGuiSession& operator=(const ImGuiSession&) = delete;

  private:
    ImGuiContext* context_;
};

}  // namespace ernie::app
