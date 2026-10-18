// SPDX-License-Identifier: BSD-3-Clause
/// @file shell.cpp
/// @brief Drawing the viewer's UI and acting on its menus and shortcuts.
#include "ernie/app/shell.h"

#include "ernie/app/help_windows.h"
#include "ernie/app/panels.h"
#include "ernie/device/context.h"
#include "ernie/scene/scene.h"
#include "ernie/util/profiler.h"

#include <imgui.h>

#include <algorithm>

namespace ernie::app {

Shell::Shell(const Scene& scene, const DeviceContext* device, const ShellOptions& options)
    : scene_(scene), device_(device), theme_(options.theme), layout_(options.docks) {
    // Stage above Devices on the right, Log and Profiler at the bottom.
    panels_.push_back(MakeStagePanel(scene, selection_));
    panels_.push_back(MakeDevicePanel(device));
    panels_.push_back(MakeLogPanel(log_, theme_));
    panels_.push_back(MakeProfilerPanel(log_, options.profile));
    if (options.profile) {
        profiler::SetEnabled(true);
    }
}

Panel* Shell::FindPanel(std::string_view title) const {
    const auto found = std::ranges::find(panels_, title, &Panel::Title);
    return found != panels_.end() ? found->get() : nullptr;
}

void Shell::Draw() {
    ERNIE_ZONE("Viewer UI");
    menu_.ClearRequests();
    DrawMenuBar(panels_, !selection_.Empty(), menu_);
    layout_.Draw(panels_);
    DrawPanels();
    DrawControlsWindow(menu_.controlsShown);
    DrawAboutWindow(menu_.aboutShown, device_ != nullptr ? device_->Info().name : "");
    if (menu_.demoShown) {
        ImGui::ShowDemoWindow(&menu_.demoShown);
    }
    HandleShortcuts();
    Navigate();
    Apply(menu_);
}

void Shell::DrawPanels() {
    for (const std::unique_ptr<Panel>& panel : panels_) {
        if (!panel->Visible()) {
            panel->Undrawn();
            continue;
        }
        layout_.DockNext(*panel);
        bool keep = true;
        if (ImGui::Begin(panel->Title().c_str(), &keep)) {
            panel->Draw();
        } else {
            panel->Undrawn();
        }
        ImGui::End();
        panel->SetVisible(keep);
    }
}

void Shell::HandleShortcuts() {
    const ImGuiIO& io = ImGui::GetIO();
    if (io.WantTextInput) {
        return;
    }
    if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Q, false)) {
        menu_.quit = true;
    }
    if (!io.KeyCtrl && !io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_F, false)) {
        menu_.frame = true;
    }
}

void Shell::Navigate() {
    const auto [origin, size] = layout_.Viewport();
    const ImVec2 mouse = ImGui::GetIO().MousePos;
    // Over the viewport and no window: the dock space lets the mouse through its central node.
    const bool hovered = !ImGui::GetIO().WantCaptureMouse && mouse.x >= origin.x &&
                         mouse.y >= origin.y && mouse.x < origin.x + size.x &&
                         mouse.y < origin.y + size.y;
    navigation_.Update(camera_, hovered, size.y);
    hud_.Draw(origin, size, camera_);
}

void Shell::Apply(MenuChoices& choices) {
    if (choices.quit) {
        open_ = false;
    }
    if (choices.resetLayout) {
        layout_.Reset();
    }
    if (choices.clearSelection) {
        selection_.Clear();
    }
    if (choices.frame) {
        Frame();
    }
}

void Shell::Frame() {
    Box bounds;
    const std::vector<Path> framed =
        selection_.Empty() ? scene_.Children(Path::Root()) : selection_.Paths();
    for (const Path& path : framed) {
        if (scene_.Has(path)) {
            bounds.Extend(scene_.WorldBounds(path));
        }
    }
    camera_.Frame(bounds);
}

}  // namespace ernie::app
