// SPDX-License-Identifier: BSD-3-Clause
/// @file ui_frame.h
/// @brief Drawing the shell's ImGui frames without a window or a device, for the app tests.
#pragma once

#include "ernie/app/imgui_session.h"
#include "ernie/app/shell.h"

#include <imgui.h>
#include <imgui_internal.h>

namespace ernie::test {

/// @brief An ImGui context that draws frames of a fixed size and takes simulated input.
class UiFrames {
  public:
    explicit UiFrames(ImVec2 size = {1280.0f, 720.0f}) : session_(app::Theme()) {
        ImGuiIO& io = ImGui::GetIO();
        io.DisplaySize = size;
        // Textures ImGui asks for are left unmade, as no renderer draws these frames.
        io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
    }

    /// @brief Draws one frame of the shell.
    void Draw(app::Shell& shell) const {
        ImGui::GetIO().DeltaTime = 1.0f / 60.0f;
        ImGui::NewFrame();
        shell.Draw();
        ImGui::Render();
    }

    /// @return Whether the window titled `title` was drawn last frame.
    static bool Drawn(const char* title) {
        const ImGuiWindow* window = ImGui::FindWindowByName(title);
        return window != nullptr && window->Active;
    }

  private:
    app::ImGuiSession session_;
};

}  // namespace ernie::test
