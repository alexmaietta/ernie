// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/builtin_panels.h"
#include "ernie/app/prim_schemas.h"
#include "ernie/app/property_fields.h"
#include "ernie/app/session.h"
#include "ernie/app/viewer.h"
#include "ernie/scene/tokens.h"

#include <imgui.h>

#include <exception>
#include <optional>
#include <string>
#include <vector>

namespace ernie::app {

namespace {

const char* LightingName(Lighting lighting) {
    switch (lighting) {
    case Lighting::Stage:
        return "Stage lights";
    case Lighting::Camera:
        return "Camera light";
    case Lighting::Default:
        return "Default sun";
    }
    return "";
}

class SettingsPanel final : public Panel {
  public:
    SettingsPanel() : Panel("Settings", Dock::RightLower) {}

    void Draw(Viewer& viewer) override {
        if (!ImGui::BeginTabBar("##settings")) {
            return;
        }
        if (ImGui::BeginTabItem("Render")) {
            DrawRender(viewer);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Viewer")) {
            DrawViewer(viewer);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

  private:
    /// The viewer's own settings, which change only the viewport.
    static void DrawViewer(Viewer& viewer) {
        if (!ImGui::BeginTable("##viewer", 2, ImGuiTableFlags_SizingFixedFit)) {
            return;
        }
        ImGui::TableSetupColumn("name", ImGuiTableColumnFlags_WidthFixed,
                                viewer.GetTheme().propertyLabelWidth * ImGui::GetFontSize() /
                                    viewer.GetTheme().fontSize);
        ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch);
        const auto row = [](const char* label) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted(label);
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(-1.0f);
        };
        row("Renderer");
        if (ImGui::BeginCombo("##renderer", viewer.Renderer().c_str())) {
            for (const std::string& name : Viewer::Renderers()) {
                if (ImGui::Selectable(name.c_str(), name == viewer.Renderer()) &&
                    name != viewer.Renderer()) {
                    try {
                        viewer.SetRenderer(name);
                    } catch (const std::exception& error) {
                        viewer.Log(LogLevel::Error, error.what());
                    }
                }
            }
            ImGui::EndCombo();
        }
        row("Shows");
        if (ImGui::BeginCombo("##var", viewer.ViewVar().c_str())) {
            for (const std::string& name : viewer.Delegate().ViewVars()) {
                if (ImGui::Selectable(name.c_str(), name == viewer.ViewVar())) {
                    viewer.SetViewVar(name);
                }
            }
            ImGui::EndCombo();
        }
        row("Lighting");
        if (ImGui::BeginCombo("##lighting", LightingName(viewer.GetLighting()))) {
            for (const Lighting mode : {Lighting::Stage, Lighting::Camera, Lighting::Default}) {
                if (ImGui::Selectable(LightingName(mode), mode == viewer.GetLighting())) {
                    viewer.SetLighting(mode);
                }
            }
            ImGui::EndCombo();
        }
        row("Wireframe");
        const std::optional<Wireframe> wireframe = viewer.GetWireframe();
        const char* wireNames[] = {"Off", "Overlay", "Hidden line"};
        const char* wireLabel =
            wireframe ? wireNames[static_cast<std::size_t>(*wireframe)] : "From settings";
        if (ImGui::BeginCombo("##wireframe", wireLabel)) {
            if (ImGui::Selectable("From settings", !wireframe)) {
                viewer.SetWireframe(std::nullopt);
            }
            for (std::uint32_t mode = 0; mode < 3; ++mode) {
                if (ImGui::Selectable(wireNames[mode], wireframe && static_cast<std::uint32_t>(
                                                                        *wireframe) == mode)) {
                    viewer.SetWireframe(static_cast<Wireframe>(mode));
                }
            }
            ImGui::EndCombo();
        }
        row("VSync");
        bool vsync = viewer.GetVsync();
        if (ImGui::Checkbox("##vsync", &vsync)) {
            viewer.SetVsync(vsync);
        }
        row("Auto exposure");
        bool automatic = viewer.AutoExposing();
        if (ImGui::Checkbox("##autoExposure", &automatic)) {
            viewer.SetExposure(automatic ? Viewer::Exposure::Auto : Viewer::Exposure::Camera);
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Key the scene's metered luminance to middle grey; off, the camera's "
                              "own exposure applies");
        }
        row("Exposure comp.");
        float stops = static_cast<float>(viewer.ExposureCompensation());
        if (ImGui::SliderFloat("##compensation", &stops, -6.0f, 6.0f, "%+.1f EV")) {
            viewer.SetExposureCompensation(stops);
        }
        row("Gizmo");
        const char* gizmoNames[] = {"None", "Translate", "Rotate", "Scale"};
        int gizmo = static_cast<int>(viewer.GetGizmo());
        if (ImGui::Combo("##gizmo", &gizmo, gizmoNames, 4)) {
            viewer.SetGizmo(static_cast<Viewer::Gizmo>(gizmo));
        }
        row("Gizmo axes");
        bool local = viewer.GizmoLocal();
        if (ImGui::Checkbox("Local##gizmoLocal", &local)) {
            viewer.SetGizmoLocal(local);
        }
        row("Theme");
        if (ImGui::Button("Dark")) {
            viewer.SetTheme(DarkTheme());
        }
        ImGui::SameLine();
        if (ImGui::Button("Light")) {
            viewer.SetTheme(LightTheme());
        }
        ImGui::EndTable();
    }

    void DrawRender(Viewer& viewer) {
        Scene& scene = viewer.GetScene();
        std::vector<Path> choices;
        for (const Path& path : scene.Prims(tokens::kRenderSettings)) {
            if (!IsSessionPath(path)) {
                choices.push_back(path);
            }
        }
        const Path& settings = viewer.Settings();
        if (choices.empty() || settings.IsEmpty()) {
            ImGui::TextWrapped("The scene has no RenderSettings: the viewport uses Ernie's "
                               "fallbacks.");
            if (ImGui::Button("Create /Render/Settings")) {
                const Path path("/Render/Settings");
                if (!scene.Has(path)) {
                    scene.Define(path, tokens::kRenderSettings);
                }
                viewer.SetSettings(path);
            }
            return;
        }
        ImGui::SetNextItemWidth(-1.0f);
        if (ImGui::BeginCombo("##settings", settings.String().c_str())) {
            for (const Path& path : choices) {
                if (ImGui::Selectable(path.String().c_str(), path == settings)) {
                    viewer.SetSettings(path);
                }
            }
            ImGui::EndCombo();
        }
        const std::string error =
            DrawPropertySections(viewer, settings, schema_, PropertyLabelWidth(viewer));
        if (!error.empty()) {
            ImGui::TextColored(ImVec4(0.89f, 0.34f, 0.32f, 1.0f), "%s", error.c_str());
        }
    }

    std::vector<PropertySection> schema_ = RenderSettingsSchema();
};

}  // namespace

std::unique_ptr<Panel> MakeSettingsPanel() {
    return std::make_unique<SettingsPanel>();
}

}  // namespace ernie::app
