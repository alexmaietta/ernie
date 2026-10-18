// SPDX-License-Identifier: BSD-3-Clause
/// @file imgui_session.cpp
/// @brief Creating the ImGui context.
#include "ernie/app/imgui_session.h"

#include <imgui.h>

namespace ernie::app {

ImGuiSession::ImGuiSession(const Theme& theme, float contentScale) {
    IMGUI_CHECKVERSION();
    context_ = ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    // Layouts are the viewer's to save; nothing is written behind its back.
    io.IniFilename = nullptr;
    io.LogFilename = nullptr;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable | ImGuiConfigFlags_NavEnableKeyboard;
    io.Fonts->AddFontDefaultVector();
    ApplyTheme(theme, contentScale, ImGui::GetStyle());
}

ImGuiSession::~ImGuiSession() {
    ImGui::DestroyContext(context_);
}

}  // namespace ernie::app
