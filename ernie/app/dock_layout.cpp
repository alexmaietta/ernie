// SPDX-License-Identifier: BSD-3-Clause
/// @file dock_layout.cpp
/// @brief Building the default layout with ImGui's dock builder.
#include "ernie/app/dock_layout.h"

#include <imgui_internal.h>

#include <algorithm>

namespace ernie::app {
namespace {

constexpr const char* kDockSpace = "ErnieDockSpace";

float Share(float fraction) {
    return std::clamp(fraction, 0.02f, 0.98f);
}

}  // namespace

ImGuiID DockLayout::DockSpace() const {
    // Hashed alone, not on the ID stack, so it is the same outside a frame.
    return ImHashStr(kDockSpace);
}

void DockLayout::Draw(std::span<const std::unique_ptr<Panel>> panels) {
    const ImGuiID dockSpace = DockSpace();
    if (!built_) {
        Build(dockSpace, panels);
        built_ = true;
    }
    // The viewport shows through the central node.
    ImGui::DockSpaceOverViewport(dockSpace, ImGui::GetMainViewport(),
                                 ImGuiDockNodeFlags_PassthruCentralNode);
}

void DockLayout::DockNext(const Panel& panel) const {
    if (const auto node = nodes_.find(panel.DefaultDock()); node != nodes_.end()) {
        ImGui::SetNextWindowDockID(node->second, ImGuiCond_FirstUseEver);
    }
}

std::pair<ImVec2, ImVec2> DockLayout::Viewport() const {
    if (const ImGuiDockNode* centre = ImGui::DockBuilderGetCentralNode(DockSpace())) {
        return {centre->Pos, centre->Size};
    }
    const ImGuiViewport* main = ImGui::GetMainViewport();
    return {main->WorkPos, main->WorkSize};
}

void DockLayout::Build(ImGuiID dockSpace, std::span<const std::unique_ptr<Panel>> panels) {
    ImGui::DockBuilderRemoveNode(dockSpace);
    ImGui::DockBuilderAddNode(dockSpace, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dockSpace, ImGui::GetMainViewport()->WorkSize);
    nodes_.clear();
    // Hidden panels take no room; they dock where they ask once shown.
    const auto wants = [&](Dock dock) {
        return std::ranges::any_of(panels, [dock](const std::unique_ptr<Panel>& panel) {
            return panel->Visible() && panel->DefaultDock() == dock;
        });
    };
    const auto split = [](ImGuiID& rest, ImGuiDir direction, float share) {
        return ImGui::DockBuilderSplitNode(rest, direction, Share(share), nullptr, &rest);
    };
    ImGuiID centre = dockSpace;
    if (wants(Dock::Left)) {
        nodes_[Dock::Left] = split(centre, ImGuiDir_Left, sizes_.left);
    }
    // The right column: `Right` above `RightLower`, either taking all of it alone.
    const bool upper = wants(Dock::Right);
    const bool lower = wants(Dock::RightLower);
    if (upper || lower) {
        ImGuiID column = split(centre, ImGuiDir_Right, sizes_.right);
        if (upper && lower) {
            nodes_[Dock::RightLower] = split(column, ImGuiDir_Down, sizes_.rightLower);
        }
        nodes_[upper ? Dock::Right : Dock::RightLower] = column;
    }
    if (wants(Dock::Bottom)) {
        nodes_[Dock::Bottom] = split(centre, ImGuiDir_Down, sizes_.bottom);
    }
    for (const std::unique_ptr<Panel>& panel : panels) {
        const auto node = nodes_.find(panel->DefaultDock());
        if (panel->Visible() && node != nodes_.end()) {
            ImGui::DockBuilderDockWindow(panel->Title().c_str(), node->second);
        }
    }
    ImGui::DockBuilderFinish(dockSpace);
}

}  // namespace ernie::app
