// SPDX-License-Identifier: BSD-3-Clause
/// @file menu_bar.cpp
/// @brief The File, Edit, Window and Help menus.
#include "ernie/app/menu_bar.h"

#include "ernie/app/menu.h"

#include <imgui.h>

namespace ernie::app {
namespace {

void DrawFileMenu(MenuChoices& choices) {
    if (menu::Action("Quit", "Ctrl+Q")) {
        choices.quit = true;
    }
}

void DrawEditMenu(bool selected, MenuChoices& choices) {
    if (menu::Action(selected ? "Frame Selection" : "Frame Scene", "F")) {
        choices.frame = true;
    }
    if (menu::Action("Clear Selection", nullptr, selected)) {
        choices.clearSelection = true;
    }
}

void DrawWindowMenu(std::span<const std::unique_ptr<Panel>> panels, MenuChoices& choices) {
    for (const std::unique_ptr<Panel>& panel : panels) {
        bool visible = panel->Visible();
        if (menu::Check(panel->Title().c_str(), &visible)) {
            panel->SetVisible(visible);
        }
    }
    ImGui::Separator();
    if (menu::Action("Reset Layout")) {
        choices.resetLayout = true;
    }
    menu::Check("ImGui Demo", &choices.demoShown);
}

void DrawHelpMenu(MenuChoices& choices) {
    menu::Check("Keyboard and Mouse", &choices.controlsShown);
    menu::Check("About Ernie", &choices.aboutShown);
}

}  // namespace

void DrawMenuBar(std::span<const std::unique_ptr<Panel>> panels, bool selected,
                 MenuChoices& choices) {
    if (!ImGui::BeginMainMenuBar()) {
        return;
    }
    const auto submenu = [](const char* name, auto&& draw) {
        menu::NextMenuWide();
        if (ImGui::BeginMenu(name)) {
            draw();
            ImGui::EndMenu();
        }
    };
    submenu("File", [&] { DrawFileMenu(choices); });
    submenu("Edit", [&] { DrawEditMenu(selected, choices); });
    submenu("Window", [&] { DrawWindowMenu(panels, choices); });
    submenu("Help", [&] { DrawHelpMenu(choices); });
    ImGui::EndMainMenuBar();
}

}  // namespace ernie::app
