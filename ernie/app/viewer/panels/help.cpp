// SPDX-License-Identifier: BSD-3-Clause
#include <filesystem>
#include <string>
#include <utility>

#include <GLFW/glfw3.h>
#include <imgui.h>

#include "ernie/app/viewer/panels/panels.h"
#include "ernie/util/version.h"

namespace ernie::app {
namespace {

constexpr ImGuiWindowFlags kHelpWindow =
    ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_AlwaysAutoResize;

constexpr std::pair<const char*, const char*> kControls[] = {
    {"Left drag", "Orbit around the target"},
    {"Right drag", "Pan along the ground"},
    {"Shift+left drag", "Pan along the ground"},
    {"Middle drag", "Pan across the screen, up and down in z"},
    {"Wheel", "Zoom towards the target"},
    {"Click", "Select what is under the cursor"},
    {"Ctrl+click", "Add to or remove from the selection"},
    {"Ctrl+Z", "Undo"},
    {"Ctrl+Shift+Z, Ctrl+Y", "Redo"},
    {"Ctrl+D", "Duplicate the selection"},
    {"Del", "Delete the selection"},
    {"F", "Frame the selection, or everything when nothing is selected"},
    {"Shift+F", "Frame the whole scene"},
    {"F2", "Rename the selected prim in the Stage panel"},
    {"F10", "Save a screenshot of the active viewport"},
    {"Drag in the Stage panel", "Move a prim under another"},
    {"Tab", "Show or hide the interface"},
    {"Ctrl+Q", "Close the viewer"},
};

}

void DrawControls(bool& open, const Theme& theme) {
    if (!open) {
        return;
    }
    if (ImGui::Begin("Keyboard and Mouse", &open, kHelpWindow) &&
        ImGui::BeginTable("##controls", 2, ImGuiTableFlags_RowBg)) {
        for (const auto& [keys, action] : kControls) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextColored(theme.accent, "%s", keys);
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(action);
        }
        ImGui::EndTable();
    }
    ImGui::End();
}

void DrawAbout(bool& open, device::Device& device) {
    if (!open) {
        return;
    }
    if (!ImGui::Begin("About Ernie", &open, kHelpWindow)) {
        ImGui::End();
        return;
    }
    const BuildInfo& build = GetBuildInfo();
    const std::string version(build.fullVersion);
    ImGui::Text("Ernie %s", version.c_str());
    ImGui::TextDisabled("A physically based, spectral renderer for planetary robotics and sensing");
    std::string report = "Ernie " + version + "\n";
    const auto section = [&](const char* title) {
        ImGui::SeparatorText(title);
        report += std::string("\n") + title + "\n";
        return ImGui::BeginTable(title, 2, ImGuiTableFlags_SizingFixedFit);
    };
    const auto row = [&](const char* label, const std::string& value) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextDisabled("%s", label);
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(value.c_str());
        report += std::string("  ") + label + ": " + value + "\n";
    };
    if (section("Links")) {
        constexpr std::pair<const char*, const char*> kLinks[] = {
            {"Documentation", "https://alexmaietta.github.io/ernie/"},
            {"Source code", "https://github.com/alexmaietta/ernie"},
            {"Report a bug", "https://github.com/alexmaietta/ernie/issues/new?labels=bug"},
            {"Request a feature",
             "https://github.com/alexmaietta/ernie/issues/new?labels=enhancement"},
            {"Known issues", "https://github.com/alexmaietta/ernie/issues"},
        };
        for (const auto& [label, url] : kLinks) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextLinkOpenURL(label, url);
            ImGui::SetItemTooltip("%s", url);
            report += std::string("  ") + label + ": " + url + "\n";
        }
        ImGui::EndTable();
    }
    if (section("Build")) {
        row("Version", version);
        row("Commit", std::string(build.commit) + (build.modified ? " (with changes)" : ""));
        row("Commit date", std::string(build.commitDate));
        row("Built", std::string(build.buildDate));
        ImGui::EndTable();
    }
    if (section("Device")) {
        row("GPU", device.Adapter().name);
        row("Backend", std::string(device::BackendName(device.Adapter().backend)));
        row("Driver", device.Adapter().driverVersion.ToString());
        ImGui::EndTable();
    }
    if (section("Folders")) {
        std::error_code error;
        row("Cache", device.CacheDirectory().string());
        row("Working directory", std::filesystem::current_path(error).string());
        ImGui::EndTable();
    }
    if (section("Libraries")) {
        row("Dear ImGui", ImGui::GetVersion());
        row("GLFW", glfwGetVersionString());
        ImGui::EndTable();
    }
    if (section("License")) {
        row("Ernie", "BSD-3-Clause");
        ImGui::EndTable();
    }
    ImGui::Spacing();
    if (ImGui::Button("Copy to Clipboard")) {
        ImGui::SetClipboardText(report.c_str());
    }
    ImGui::SetItemTooltip("All of it as text, for a bug report");
    ImGui::End();
}

}
