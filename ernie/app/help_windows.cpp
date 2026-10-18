// SPDX-License-Identifier: BSD-3-Clause
/// @file help_windows.cpp
/// @brief The controls table and the About window.
#include "ernie/app/help_windows.h"

#include "ernie/app/widgets.h"
#include "ernie/util/version.h"

#include <imgui.h>

#include <array>
#include <string>

namespace ernie::app {
namespace {

constexpr std::array kControls{
    Control{"Left drag", "Orbit the camera around its target"},
    Control{"Right drag", "Pan along the ground"},
    Control{"Middle drag", "Pan across the view"},
    Control{"Wheel", "Zoom towards the target"},
    Control{"F", "Frame the selection, or the whole scene"},
    Control{"Click in Stage", "Select a prim; Ctrl adds or removes it"},
    Control{"Ctrl+Q", "Quit"},
};

constexpr ImGuiWindowFlags kHelpFlags =
    ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse;

void CentreNextWindow() {
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing,
                            ImVec2(0.5f, 0.5f));
}

}  // namespace

std::span<const Control> Controls() {
    return kControls;
}

void DrawControlsWindow(bool& open) {
    if (!open) {
        return;
    }
    CentreNextWindow();
    if (ImGui::Begin("Keyboard and Mouse", &open, kHelpFlags) &&
        ImGui::BeginTable("##controls", 2, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV)) {
        for (const Control& control : kControls) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(control.input);
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(control.action);
        }
        ImGui::EndTable();
    }
    ImGui::End();
}

void DrawAboutWindow(bool& open, std::string_view device) {
    if (!open) {
        return;
    }
    CentreNextWindow();
    if (ImGui::Begin("About Ernie", &open, kHelpFlags)) {
        ImGui::TextUnformatted(
            "Ernie: Efficient Renderer for Natural and Interplanetary Environments");
        LabelValue("Version", Version());
        LabelValue("Device", device.empty() ? std::string_view("none") : device);
        LabelValue("Dear ImGui", IMGUI_VERSION);
        ImGui::Separator();
        ImGui::TextUnformatted("BSD-3-Clause licence. https://github.com/alexmaietta/ernie");
    }
    ImGui::End();
}

}  // namespace ernie::app
