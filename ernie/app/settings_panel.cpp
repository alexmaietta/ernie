#include "ernie/app/builtin_panels.h"
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

using Kind = PropertyField::Kind;

/// The settings the rasterizer reads, with its fallbacks.
std::vector<PropertyField> ImageFields() {
    using namespace tokens;
    return {
        {kColorMode, "Terrain colour", Kind::Token, std::string(kShadedColormap), 0, 0,
         {kHillshade, kColormap, kShadedColormap}},
        {kAmbient, "Ambient", Kind::Float, 0.15, 0.0, 1.0, {}, "Fraction of sunlight on unlit sides"},
        {kClearColor, "Background", Kind::Color, glm::vec3(0.0f)},
        {kWireframe, "Wireframe", Kind::Token, std::string(kOff), 0, 0,
         {kOff, kOverlay, kHiddenLine}, "The viewport's toolbar may override it"},
        {kOverlays, "Overlays", Kind::Text, std::string(), 0, 0, {},
         "\"*\" or names; the viewport shows every overlay unless set"},
    };
}

std::vector<PropertyField> CullingFields() {
    using namespace tokens;
    return {
        {kCulling, "Culling", Kind::Token, std::string(kGpu), 0, 0, {kOff, kCpu, kGpu}},
        {kOcclusion, "Occlusion", Kind::Token, std::string(kOff), 0, 0,
         {kOff, kOccluders, kTwoPass}},
        {kLodMeshError, "Mesh LOD error", Kind::Float, 1.0, 0.0, 16.0, {},
         "Pixels of error a coarser mesh level may add"},
        {kLodEnable, "Terrain LOD", Kind::Bool, true},
        {kLodDetail, "Terrain detail", Kind::Float, 4.0, 1.0, 16.0},
        {kLodMorphStart, "Morph start", Kind::Float, 0.67, 0.0, 1.0},
        {kLodFreeze, "Freeze LOD", Kind::Bool, false},
        {kLodShowLevels, "Show LOD levels", Kind::Bool, false},
    };
}

std::vector<PropertyField> ShadowFields() {
    using namespace tokens;
    return {
        {kShadowFit, "Fit", Kind::Token, std::string(kScene), 0, 0, {kScene, kVisible},
         "visible: sample distribution shadow maps (the viewport's default)"},
        {kShadowCascades, "Cascades", Kind::Int, 3, 1, 4},
        {kShadowResolution, "Resolution", Kind::Int, 2048, 256, 8192},
        {kShadowSoftness, "Softness", Kind::Float, 1.5, 0.0, 8.0, {}, "Filter radius in texels"},
        {kShadowNormalOffset, "Normal offset", Kind::Float, 1.5, 0.0, 8.0},
        {kShadowSplitLambda, "Split lambda", Kind::Float, 0.8, 0.0, 1.0},
        {kShadowShowCascades, "Show cascades", Kind::Bool, false},
        {kShadowWorlds, "Worlds", Kind::Token, std::string(kSeparate), 0, 0,
         {kSeparate, kShared, kMerged}, "Batches of views of several worlds"},
        {kShadowWorldFit, "World fit", Kind::Token, std::string(kScene), 0, 0,
         {kScene, kWorldFit}},
        {kShadowWorldResolution, "World resolution", Kind::Int, 0, 0, 8192, {}, "0 for automatic"},
        {kShadowWorldMargin, "World margin", Kind::Float, 1.0, 0.0, 100.0},
    };
}

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
    SettingsPanel() : Panel("Settings", Dock::Right) {}

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
            for (const std::string& name : viewer.Delegate().Info().viewVars) {
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
                if (ImGui::Selectable(wireNames[mode],
                                      wireframe && static_cast<std::uint32_t>(*wireframe) == mode)) {
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
        const float labelWidth = viewer.GetTheme().propertyLabelWidth * ImGui::GetFontSize() /
                                 viewer.GetTheme().fontSize;
        std::string error;
        const auto section = [&](const char* title, const std::vector<PropertyField>& fields) {
            if (ImGui::CollapsingHeader(title, ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::PushID(title);
                const std::string failure = DrawPropertyFields(viewer, settings, fields, labelWidth);
                if (error.empty()) {
                    error = failure;
                }
                ImGui::PopID();
            }
        };
        section("Image", image_);
        section("Culling and LOD", culling_);
        section("Shadows", shadows_);
        if (!error.empty()) {
            ImGui::TextColored(ImVec4(0.89f, 0.34f, 0.32f, 1.0f), "%s", error.c_str());
        }
    }

    std::vector<PropertyField> image_ = ImageFields();
    std::vector<PropertyField> culling_ = CullingFields();
    std::vector<PropertyField> shadows_ = ShadowFields();
};

}  // namespace

std::unique_ptr<Panel> MakeSettingsPanel() {
    return std::make_unique<SettingsPanel>();
}

}  // namespace ernie::app
