// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/viewer/overlays/toolbar.h"

#include <imgui_stdlib.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ernie/app/viewer/widgets/widgets.h"

namespace ernie::app {
namespace {

constexpr double kRefreshSeconds = 0.5;
constexpr float kOrthographicUnits = 0.1f;

enum LensField : size_t { Focal, Height, Exposure, Iso, Shutter, ExposureFStop, DofFStop, Focus };

constexpr decltype(Camera::kFocalLength) kLensFields[] = {
    Camera::kFocalLength,  Camera::kVerticalAperture, Camera::kExposure, Camera::kExposureIso,
    Camera::kExposureTime, Camera::kExposureFStop,    Camera::kFStop,    Camera::kFocusDistance};

constexpr std::pair<MinimalMode, const char*> kMinimalModes[] = {
    {MinimalMode::Full, "Full"},
    {MinimalMode::TextureDiffuse, "Texture Diffuse"},
    {MinimalMode::DisplayColor, "Display Color"},
    {MinimalMode::Data, "Data"},
};

const char* MinimalName(MinimalMode mode) {
    for (const auto& [value, name] : kMinimalModes) {
        if (value == mode) {
            return name;
        }
    }
    return "Minimal";
}

constexpr std::tuple<LightingMode, const char*, const char*> kLightings[] = {
    {LightingMode::Stage, "Stage Lights", "The scene's own lights"},
    {LightingMode::Camera, "Camera Light",
     "Light this viewport with one light along the view, instead of the scene's lights"},
    {LightingMode::Default, "Default Light",
     "Light this viewport with a fixed white sun from above, instead of the scene's lights"},
};

const char* LightingName(LightingMode lighting) {
    for (const auto& [value, name, tip] : kLightings) {
        if (value == lighting) {
            return name;
        }
    }
    return "Lighting";
}

std::string Leaf(const std::string& path) {
    return path.substr(path.rfind('/') + 1);
}

void CollectCameras(const Prim& prim, std::vector<std::string>& cameras) {
    if (prim.IsA(SchemaId::Camera)) {
        cameras.push_back(prim.Path());
    }
    for (const Prim& child : prim.Children()) {
        CollectCameras(child, cameras);
    }
}

}

Toolbar::Toolbar(Ref<Scene> scene, std::string product, Viewport& viewport, const Theme& theme,
                 Editor* editor)
    : m_scene(std::move(scene)), m_product(std::move(product)), m_viewport(viewport),
      m_theme(theme), m_editor(editor) {}

void Toolbar::Refresh() {
    if (m_scene->StructureVersion() != m_structure) {
        m_structure = m_scene->StructureVersion();
        m_cameras.clear();
        const View roots = m_scene->Select("/*");
        for (const uint32_t root : roots.Prims()) {
            const Prim prim = m_scene->ByIndex(root);
            if (!prim.IsSession()) {
                CollectCameras(prim, m_cameras);
            }
        }
    }
    if (!m_reads.empty()) {
        if (!std::ranges::all_of(m_reads,
                                 [](const device::Download& read) { return read.Ready(); })) {
            return;
        }
        uint32_t words[2] = {uint32_t(m_mode), uint32_t(m_minimal)};
        for (size_t read = 0; read < 2; ++read) {
            if (m_reads[read].Bytes().size() >= sizeof(uint32_t)) {
                std::memcpy(&words[read], m_reads[read].Bytes().data(), sizeof(uint32_t));
            }
        }
        m_mode = RenderMode(words[0]);
        m_minimal = MinimalMode(words[1]);
        m_reads.clear();
    }
    if (ImGui::GetTime() - m_requested >= kRefreshSeconds) {
        const View product = m_scene->Select(m_product);
        m_reads.push_back(product.Get(RenderProduct::kRenderMode.Column()).ReadHostAsync());
        m_reads.push_back(product.Get(RenderProduct::kMode.Column()).ReadHostAsync());
        m_requested = ImGui::GetTime();
    }
}

bool Toolbar::BeginPill(const char* id, const std::string& label, const char* tip) {
    constexpr ImGuiComboFlags kPill = ImGuiComboFlags_WidthFitPreview |
                                      ImGuiComboFlags_NoArrowButton | ImGuiComboFlags_HeightLargest;
    ImGui::SetNextWindowSizeConstraints({14.0f * ImGui::GetFontSize(), 0}, {FLT_MAX, FLT_MAX});
    const bool open = ImGui::BeginCombo(id, label.c_str(), kPill);
    ImGui::SetItemTooltip("%s", tip);
    return open;
}

void Toolbar::Draw(ImVec2 min) {
    if (!m_scene || m_product.empty()) {
        return;
    }
    Refresh();
    const float scale = ImGui::GetStyle().FontScaleDpi;
    ImGui::PushStyleColor(ImGuiCol_FrameBg, m_theme.overlay);
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, m_theme.fieldHovered);
    ImGui::PushStyleColor(ImGuiCol_PopupBg, WithAlpha(m_theme.overlay, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_Border, m_theme.overlayBorder);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, scale);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6.0f * scale);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {9.0f * scale, 4.0f * scale});
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {5.0f * scale, 4.0f * scale});
    const float pad = 8.0f * scale;
    ImGui::SetCursorScreenPos({min.x + pad, min.y + pad});

    if (BeginPill("##settings", icons::kSettings, "Viewport settings")) {
        DrawSettingsMenu();
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    if (BeginPill("##hud", icons::kEye, "What the heads-up display shows")) {
        DrawHudMenu();
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    const std::string renderer = m_mode == RenderMode::RealTime
                                     ? std::string("Real-Time")
                                     : std::string("Minimal: ") + MinimalName(m_minimal);
    if (BeginPill("##renderer", std::string(icons::kMonitor) + "  " + renderer, "Rendering mode")) {
        DrawRendererMenu();
        ImGui::EndCombo();
    }
    if (m_viewport.ShowsWorlds()) {
        ImGui::SameLine();
        if (BeginPill("##lighting",
                      std::string(icons::kLightbulb) + "  " + LightingName(m_viewport.Lighting()),
                      "Which lights shade this viewport")) {
            DrawLightingMenu();
            ImGui::EndCombo();
        }
    }
    if (m_viewport.ShowsWorlds() && !m_viewport.WorldPrims().empty()) {
        ImGui::SameLine();
        constexpr const char* kPicks[] = {"All Worlds", "First", "Random", "Listed", "Selected"};
        const WorldDisplay& worlds = m_viewport.Worlds();
        std::string label = kPicks[size_t(worlds.pick)];
        if ((worlds.pick == WorldPick::First || worlds.pick == WorldPick::Random) &&
            worlds.count > 0) {
            label += " " + std::to_string(worlds.count);
        }
        if (BeginPill("##worlds", std::string(icons::kGlobe) + "  " + label,
                      "Which worlds the viewport shows, and how they're laid out")) {
            DrawWorldsMenu();
            ImGui::EndCombo();
        }
    }
    if (!m_viewport.Aov().empty()) {
        ImGui::SameLine();
        if (BeginPill("##aov", std::string(icons::kImage) + "  " + m_viewport.Aov(),
                      "The AOV shown, and how it's colored")) {
            DrawAovMenu();
            ImGui::EndCombo();
        }
    }
    ImGui::SameLine();
    const std::string& camera = m_viewport.CameraPath();
    if (BeginPill("##camera",
                  std::string(icons::kCamera) + "  " +
                      (camera.empty() ? "No camera" : Leaf(camera)),
                  "Camera to look through, and its lens")) {
        DrawCameraMenu();
        ImGui::EndCombo();
    }
    if (!camera.empty()) {
        DrawCameraBar();
    }

    ImGui::PopStyleVar(4);
    ImGui::PopStyleColor(4);
}

void Toolbar::DrawSettingsMenu() {
    if (ImGui::Selectable("Frame Selection")) {
        m_viewport.Frame(true);
    }
    ImGui::SetItemTooltip("Fit the selected prims in view, or everything when nothing is "
                          "selected (F)");
    if (ImGui::Selectable("Frame All")) {
        m_viewport.Frame(false);
    }
    ImGui::SetItemTooltip("Fit the whole scene in view (Shift+F)");
    ImGui::Separator();
    ImGui::PushItemFlag(ImGuiItemFlags_AutoClosePopups, false);
    const float field = 10 * ImGui::GetFontSize();
    if (ImGui::BeginMenu("Render Resolution")) {
        int percent = int(std::round(m_scale * 100));
        ImGui::SetNextItemWidth(field);
        if (ImGui::SliderInt("##resolution", &percent, 25, 100, "%d%% of the viewport")) {
            m_scale = float(percent) / 100;
        }
        ImGui::SetItemTooltip("Render fewer pixels and scale them up, for speed on slower GPUs");
        ImGui::EndMenu();
    }
    ViewportGuides& guides = m_viewport.guides;
    if (ImGui::BeginMenu("Selection")) {
        ImGui::Checkbox("Outline", &guides.outline);
        ImGui::SetItemTooltip("Trace the selected prims' edges in the image");
        ImGui::ColorEdit3("Outline Color", &guides.outlineColor.x, ImGuiColorEditFlags_NoInputs);
        ImGui::Checkbox("Bounds", &guides.bounds);
        ImGui::SetItemTooltip("Box each selected prim's world bounds");
        ImGui::Checkbox("Axes", &guides.frames);
        ImGui::SetItemTooltip("Each selected prim's x, y and z axes at its origin");
        ImGui::EndMenu();
    }
    if (Gizmo* gizmo = m_viewport.GetGizmo(); gizmo && ImGui::BeginMenu("Gizmo")) {
        constexpr std::pair<GizmoMode, const char*> kModes[] = {{GizmoMode::None, "None (Q)"},
                                                                {GizmoMode::Translate, "Move (W)"},
                                                                {GizmoMode::Rotate, "Rotate (E)"},
                                                                {GizmoMode::Scale, "Scale (R)"}};
        for (const auto& [mode, name] : kModes) {
            if (ImGui::RadioButton(name, gizmo->mode == mode)) {
                gizmo->mode = mode;
            }
        }
        ImGui::Separator();
        ImGui::Checkbox("Local Axes", &gizmo->local);
        ImGui::SetItemTooltip("Move and rotate along the prim's own axes rather than the world's");
        ImGui::Checkbox("Snap", &gizmo->snap);
        ImGui::SetItemTooltip("Move, turn and scale in steps; hold Ctrl while dragging to switch");
        ImGui::BeginDisabled(!gizmo->snap);
        ImGui::SetNextItemWidth(6 * ImGui::GetFontSize());
        ImGui::DragFloat("Move Step", &gizmo->moveStep, 0.01f, 0.001f, 100, "%.3f m");
        ImGui::SetNextItemWidth(6 * ImGui::GetFontSize());
        ImGui::DragFloat("Turn Step", &gizmo->turnStep, 0.5f, 0.5f, 90, "%.1f deg");
        ImGui::SetNextItemWidth(6 * ImGui::GetFontSize());
        ImGui::DragFloat("Scale Step", &gizmo->scaleStep, 0.01f, 0.01f, 10, "%.2f");
        ImGui::EndDisabled();
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Grid")) {
        GridGuide& grid = guides.grid;
        ImGui::Checkbox("Show Grid", &grid.shown);
        ImGui::SetItemTooltip("A ground grid around the orbit target");
        ImGui::BeginDisabled(!grid.shown);
        ImGui::SeparatorText("Spacing");
        if (ImGui::RadioButton("Follow Zoom", !grid.fixed)) {
            grid.fixed = false;
        }
        ImGui::SetItemTooltip("Pick the line spacing from the camera's distance, fading between "
                              "levels");
        if (ImGui::RadioButton("Fixed", grid.fixed)) {
            grid.fixed = true;
        }
        ImGui::SetItemTooltip("Keep the minor lines a set distance apart");
        ImGui::BeginDisabled(!grid.fixed);
        ImGui::SetNextItemWidth(field);
        ImGui::DragFloat("Minor Spacing", &grid.spacing, 0.01f, 0.001f, 10000, "%.3f m",
                         ImGuiSliderFlags_Logarithmic);
        ImGui::EndDisabled();
        int divisions = int(grid.divisions);
        ImGui::SetNextItemWidth(field);
        if (ImGui::SliderInt("Divisions", &divisions, 2, 20)) {
            grid.divisions = uint32_t(divisions);
        }
        ImGui::SetItemTooltip("Minor lines to each major line");
        ImGui::SetNextItemWidth(field);
        ImGui::SliderFloat("Extent", &grid.reach, 10, 200, "%.0f minor lines");
        ImGui::SetItemTooltip("How far the grid reaches from its centre before it fades out");
        ImGui::SetNextItemWidth(field);
        ImGui::DragFloat("Height", &grid.height, 0.01f, -1e6f, 1e6f, "%.3f m");
        ImGui::SetItemTooltip("The grid's z");
        ImGui::SeparatorText("Brightness");
        ImGui::SetNextItemWidth(field);
        ImGui::SliderFloat("Minor Lines", &grid.minorBrightness, 0, 1, "%.2f");
        ImGui::SetNextItemWidth(field);
        ImGui::SliderFloat("Major Lines", &grid.majorBrightness, 0, 1, "%.2f");
        ImGui::EndDisabled();
        ImGui::EndMenu();
    }
    ImGui::PopItemFlag();
}

void Toolbar::DrawRendererMenu() {
    const View product = m_scene->Select(m_product);
    const auto set = [&](RenderMode mode, std::optional<MinimalMode> minimal) {
        product.Get(RenderProduct::kRenderMode).Set(mode);
        m_mode = mode;
        if (minimal) {
            product.Get(RenderProduct::kMode).Set(*minimal);
            m_minimal = *minimal;
        }
        m_requested = ImGui::GetTime();
    };
    if (ImGui::Selectable("Real-Time", m_mode == RenderMode::RealTime)) {
        set(RenderMode::RealTime, std::nullopt);
    }
    ImGui::SetItemTooltip("Path-traced lighting with shadows and reflections");
    ImGui::SeparatorText("Minimal");
    for (const auto& [mode, name] : kMinimalModes) {
        if (ImGui::Selectable(name, m_mode == RenderMode::Minimal && m_minimal == mode)) {
            set(RenderMode::Minimal, mode);
        }
    }
}

void Toolbar::DrawCameraMenu() {
    const auto choice = [&](const std::string& camera, const char* tip) {
        ImGui::PushID(camera.c_str());
        ImGui::SetNextItemAllowOverlap();
        if (ImGui::Selectable(Leaf(camera).c_str(), camera == m_viewport.CameraPath())) {
            m_viewport.LookThrough(camera);
        }
        ImGui::SetItemTooltip("%s", tip);
        if (RowButton(icons::kSettings, "Select the camera and show its properties")) {
            m_viewport.Inspect(camera);
        }
        ImGui::PopID();
    };
    if (!m_cameras.empty()) {
        ImGui::SeparatorText("Cameras");
        for (const std::string& camera : m_cameras) {
            choice(camera, camera.c_str());
        }
    }
    if (!m_viewport.Views().empty()) {
        ImGui::SeparatorText("Views");
        for (const std::string& view : m_viewport.Views()) {
            choice(view, view == m_viewport.Views().front() ? "The viewer's own perspective camera"
                                                            : "An orthographic view");
        }
    }
}

void Toolbar::RefreshLens(const std::string& camera) {
    if (camera != m_lensCamera) {
        m_lensCamera = camera;
        m_lensValid = false;
        m_lensReads.clear();
        m_lensBefore.reset();
        m_lensRequested = -1;
    }
    if (!m_lensReads.empty()) {
        if (!std::ranges::all_of(m_lensReads,
                                 [](const device::Download& read) { return read.Ready(); })) {
            return;
        }
        bool complete = m_lensReads.back().Bytes().size() >= sizeof(uint32_t);
        for (size_t field = 0; field < m_lens.size() && complete; ++field) {
            complete = m_lensReads[field].Bytes().size() >= sizeof(float);
        }
        if (complete && !m_lensBefore) {
            for (size_t field = 0; field < m_lens.size(); ++field) {
                std::memcpy(&m_lens[field], m_lensReads[field].Bytes().data(), sizeof(float));
            }
            uint32_t projection = 0;
            std::memcpy(&projection, m_lensReads.back().Bytes().data(), sizeof(projection));
            m_orthographic = projection == uint32_t(Projection::Orthographic);
            m_lensValid = true;
        }
        m_lensReads.clear();
    }
    if (m_lensRequested < 0 || ImGui::GetTime() - m_lensRequested >= kRefreshSeconds) {
        const View view = m_scene->Select(camera);
        for (const auto& field : kLensFields) {
            m_lensReads.push_back(view.Get(field.Column()).ReadHostAsync());
        }
        m_lensReads.push_back(view.Get(Camera::kProjection.Column()).ReadHostAsync());
        m_lensRequested = ImGui::GetTime();
    }
}

void Toolbar::DrawLensField(size_t field, const char* label, const char* tip, float scale,
                            float speed, float low, float high, const char* format) {
    ImGui::SameLine();
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(label);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(4.5f * ImGui::GetFontSize());
    float shown = m_lens[field] * scale;
    const std::string id = std::string("##lens") + label;
    const bool changed = ImGui::DragFloat(id.c_str(), &shown, speed, low, high, format,
                                          ImGuiSliderFlags_AlwaysClamp);
    ImGui::SetItemTooltip("%s", tip);
    if (ImGui::IsItemActivated()) {
        m_lensBefore = m_lens[field];
    }
    if (changed) {
        m_lens[field] = shown / scale;
        m_scene->Select(m_lensCamera).Get(kLensFields[field]).Set(m_lens[field]);
    }
    if (!ImGui::IsItemDeactivated() || !m_lensBefore) {
        return;
    }
    const std::string_view path = m_lensCamera;
    const bool session = path.starts_with(kSessionRoot) && path.size() > kSessionRoot.size() &&
                         path[kSessionRoot.size()] == '/';
    if (ImGui::IsItemDeactivatedAfterEdit() && m_editor && !session &&
        *m_lensBefore != m_lens[field]) {
        const auto bytes = [](float value) {
            std::vector<std::byte> cells(sizeof(float));
            std::memcpy(cells.data(), &value, sizeof(float));
            return cells;
        };
        m_editor->SetCells(
            std::string("Set ") + label,
            [scene = m_scene, camera = m_lensCamera] { return scene->Select(camera); },
            kLensFields[field].Column(), bytes(*m_lensBefore), bytes(m_lens[field]));
    }
    m_lensBefore.reset();
    m_lensRequested = -1;
}

void Toolbar::DrawCameraBar() {
    ImGui::SameLine();
    if (ImGui::Button(m_lensOpen ? "<<##lens" : ">>##lens")) {
        m_lensOpen = !m_lensOpen;
    }
    ImGui::SetItemTooltip(m_lensOpen ? "Hide the camera's lens and exposure"
                                     : "Show the camera's lens and exposure");
    if (!m_lensOpen) {
        return;
    }
    RefreshLens(m_viewport.CameraPath());
    if (!m_lensValid) {
        return;
    }
    if (m_orthographic) {
        DrawLensField(Height, "Height", "How tall the orthographic view is", kOrthographicUnits,
                      0.05f, 0.001f, 1e6f, "%.2f m");
    } else {
        DrawLensField(Focal, "Focal", "Focal length", 1, 0.2f, 1, 2000, "%.1f mm");
    }
    DrawLensField(Exposure, "EV", "Exposure compensation in stops", 1, 0.02f, -20, 20, "%+.1f");
    DrawLensField(Iso, "ISO", "The sensor's sensitivity", 1, 1, 1, 409600, "%.0f");
    DrawLensField(Shutter, "Time", "How long the sensor gathers light", 1, 0.001f, 1e-6f, 60,
                  "%.4g s");
    DrawLensField(ExposureFStop, "f/", "The f-number the exposure is computed with", 1, 0.05f, 0.5f,
                  64, "%.1f");
    ImGui::BeginDisabled(m_orthographic);
    DrawLensField(DofFStop, "DoF f/",
                  "The lens's f-number for depth of field, 0 keeps all in focus", 1, 0.05f, 0, 64,
                  m_lens[DofFStop] > 0 ? "%.1f" : "off");
    DrawLensField(Focus, "Focus", "Distance to the plane in sharpest focus", 1, 0.01f, 0, 1e5f,
                  "%.2f m");
    ImGui::EndDisabled();
}

void Toolbar::DrawAovMenu() {
    std::string_view group;
    for (const AovInfo* info : m_viewport.Aovs()) {
        if (group != info->group) {
            group = info->group;
            ImGui::SeparatorText(info->group);
        }
        const std::string name(info->name);
        if (ImGui::Selectable(name.c_str(), name == m_viewport.Aov())) {
            m_viewport.SetAov(name);
        }
    }
    const AovInfo* shown = FindAov(m_viewport.Aov());
    if (shown == nullptr || (shown->kind != AovKind::Scalar && shown->kind != AovKind::Hdr)) {
        return;
    }
    AovDisplay& display = m_viewport.Display();
    ImGui::SeparatorText("Display");
    ImGui::SetNextItemWidth(-1);
    if (shown->kind == AovKind::Hdr) {
        ImGui::SliderFloat("##exposure", &display.exposure, -10, 10, "%+.1f stops of exposure");
        return;
    }
    ColorizeOptions& colors = display.colors;
    constexpr std::pair<Colormap, const char*> kColormaps[] = {{Colormap::Grayscale, "Grayscale"},
                                                               {Colormap::Turbo, "Turbo"},
                                                               {Colormap::Viridis, "Viridis"}};
    for (const auto& [colormap, name] : kColormaps) {
        if (ImGui::RadioButton(name, colors.colormap == colormap)) {
            colors.colormap = colormap;
        }
        ImGui::SameLine();
    }
    ImGui::NewLine();
    ImGui::BeginDisabled(colors.centered);
    ImGui::Checkbox("Logarithmic", &colors.logarithmic);
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(colors.logarithmic);
    ImGui::Checkbox("Centered", &colors.centered);
    ImGui::EndDisabled();
    ImGui::SetItemTooltip("Center the range on 0, for signed values such as velocity");
    ImGui::SameLine();
    ImGui::Checkbox("Reversed", &colors.reversed);
    bool hold = colors.minimum.has_value();
    if (ImGui::Checkbox("Hold Range", &hold)) {
        const std::optional<std::array<float, 2>>& range = m_viewport.AovRange();
        colors.minimum = hold ? std::optional(range ? (*range)[0] : 0.0f) : std::nullopt;
        colors.maximum = hold ? std::optional(range ? (*range)[1] : 1.0f) : std::nullopt;
    }
    ImGui::SetItemTooltip("Keep the current range instead of fitting each frame's values");
    if (hold) {
        float low = *colors.minimum;
        float high = *colors.maximum;
        ImGui::SetNextItemWidth(-1);
        const float speed = std::max(std::abs(high - low), 1e-3f) * 0.005f;
        if (ImGui::DragFloatRange2("##range", &low, &high, speed, 0, 0, "%.4g", "%.4g")) {
            colors.minimum = low;
            colors.maximum = std::max(high, low);
        }
    }
}

void Toolbar::DrawLightingMenu() {
    for (const auto& [lighting, name, tip] : kLightings) {
        if (ImGui::Selectable(name, m_viewport.Lighting() == lighting)) {
            m_viewport.SetLighting(lighting);
        }
        ImGui::SetItemTooltip("%s", tip);
    }
}

void Toolbar::DrawHudMenu() {
    ImGui::PushItemFlag(ImGuiItemFlags_AutoClosePopups, false);
    if (ImGui::BeginMenu("Heads-Up Display")) {
        ImGui::MenuItem("Show", nullptr, &m_viewport.hudShown);
        ImGui::Separator();
        ImGui::BeginDisabled(!m_viewport.hudShown);
        for (size_t at = 0; at < m_viewport.hudItems.size(); ++at) {
            bool shown = m_viewport.hudItems[at];
            if (ImGui::MenuItem(kHudItemNames[at], nullptr, &shown)) {
                m_viewport.hudItems[at] = shown;
            }
        }
        ImGui::EndDisabled();
        ImGui::EndMenu();
    }
    ViewportGuides& guides = m_viewport.guides;
    ImGui::SeparatorText("Show By Type");
    ImGui::MenuItem("Cameras", nullptr, &guides.cameras);
    ImGui::SetItemTooltip("Each scene camera's frustum, but the one looked through");
    ImGui::MenuItem("Lights", nullptr, &guides.lights);
    ImGui::SetItemTooltip("Each light's shape and direction");
    ImGui::SeparatorText("Guides");
    ImGui::MenuItem("Selection Outline", nullptr, &guides.outline);
    ImGui::MenuItem("Selection Bounds", nullptr, &guides.bounds);
    ImGui::MenuItem("Selection Axes", nullptr, &guides.frames);
    ImGui::SetItemTooltip("Each selected prim's x, y and z axes at its origin");
    ImGui::MenuItem("All Bounds", nullptr, &guides.allBounds);
    ImGui::SetItemTooltip("Every geometry prim's bounding box, a color a prim");
    ImGui::MenuItem("Grid", nullptr, &guides.grid.shown);
    ImGui::MenuItem("Axis Indicator", nullptr, &guides.axes);
    ImGui::SetItemTooltip("The world's axes as the camera sees them, in the bottom left corner");
    ImGui::SeparatorText("Overlays");
    bool overlays = m_viewport.SceneOverlaysShown();
    if (ImGui::MenuItem("Scene Overlays", nullptr, &overlays)) {
        m_viewport.ShowSceneOverlays(overlays);
    }
    ImGui::SetItemTooltip("Draw the renderer's overlays from the scene or a simulation over the "
                          "image, beside the guides above");
    ImGui::BeginDisabled(!overlays);
    ImGui::Indent();
    for (const std::string& name : m_viewport.SceneOverlayNames()) {
        bool shown = !m_viewport.hiddenOverlays.contains(name);
        if (ImGui::MenuItem(name.c_str(), nullptr, &shown)) {
            if (shown) {
                m_viewport.hiddenOverlays.erase(name);
            } else {
                m_viewport.hiddenOverlays.insert(name);
            }
        }
    }
    ImGui::Unindent();
    ImGui::EndDisabled();
    ImGui::PopItemFlag();
}

void Toolbar::DrawWorldsMenu() {
    WorldDisplay display = m_viewport.Worlds();
    bool changed = false;
    ImGui::PushItemFlag(ImGuiItemFlags_AutoClosePopups, false);
    ImGui::SeparatorText("Show");
    constexpr std::tuple<WorldPick, const char*, const char*> kPicks[] = {
        {WorldPick::All, "All", "Every world"},
        {WorldPick::First, "First", "The first worlds, up to the count"},
        {WorldPick::Random, "Random", "Worlds picked at random, up to the count"},
        {WorldPick::Listed, "Listed", "The worlds checked below"},
        {WorldPick::Selected, "Selected", "The worlds of the selected prims"}};
    for (const auto& [pick, name, tip] : kPicks) {
        if (ImGui::RadioButton(name, display.pick == pick)) {
            display.pick = pick;
            changed = true;
        }
        ImGui::SetItemTooltip("%s", tip);
        ImGui::SameLine();
    }
    ImGui::NewLine();
    if (display.pick == WorldPick::First || display.pick == WorldPick::Random) {
        int count = int(display.count);
        ImGui::SetNextItemWidth(8 * ImGui::GetFontSize());
        if (ImGui::InputInt("Count", &count, 1, 16)) {
            display.count = uint32_t(std::max(count, 0));
            changed = true;
        }
        ImGui::SetItemTooltip("Most worlds shown, 0 for all of them");
        if (display.pick == WorldPick::Random) {
            ImGui::SameLine();
            if (ImGui::Button("Reshuffle")) {
                ++display.seed;
                changed = true;
            }
        }
    }
    if (display.pick == WorldPick::Listed) {
        ImGui::SetNextItemWidth(-1);
        ImGui::InputTextWithHint("##worldsearch", "Search worlds", &m_worldSearch);
        if (ImGui::BeginChild("##worldlist", {0, 12 * ImGui::GetTextLineHeightWithSpacing()},
                              ImGuiChildFlags_Borders)) {
            const std::string search = Lowercase(m_worldSearch);
            for (const auto& [path, world] : m_viewport.WorldPrims()) {
                if (!search.empty() && Lowercase(path).find(search) == std::string::npos) {
                    continue;
                }
                const auto found = std::ranges::find(display.worlds, path);
                bool shown = found != display.worlds.end();
                if (ImGui::Checkbox(path.c_str(), &shown)) {
                    if (shown) {
                        display.worlds.push_back(path);
                    } else {
                        display.worlds.erase(found);
                    }
                    changed = true;
                }
            }
        }
        ImGui::EndChild();
    }
    changed |= ImGui::Checkbox("Global Prims", &display.global);
    ImGui::SetItemTooltip("Show prims outside every world, such as a shared ground, once");
    ImGui::SeparatorText("Layout");
    if (ImGui::RadioButton("Grid", display.layout == WorldLayout::Grid)) {
        display.layout = WorldLayout::Grid;
        changed = true;
    }
    ImGui::SetItemTooltip("Spread the shown worlds on a grid, for the viewer only");
    ImGui::SameLine();
    if (ImGui::RadioButton("Overlap", display.layout == WorldLayout::Overlap)) {
        display.layout = WorldLayout::Overlap;
        changed = true;
    }
    ImGui::SetItemTooltip("Show the worlds where they are, on top of each other");
    if (display.layout == WorldLayout::Grid) {
        bool automatic = !display.spacing;
        if (ImGui::Checkbox("Auto Spacing", &automatic)) {
            display.spacing = automatic ? std::nullopt : std::optional(Double3{10, 10, 0});
            changed = true;
        }
        ImGui::SetItemTooltip("Space worlds by their largest size, half again apart");
        if (display.spacing) {
            float spacing[2] = {float(display.spacing->x), float(display.spacing->y)};
            ImGui::SetNextItemWidth(-1);
            if (ImGui::DragFloat2("##spacing", spacing, 0.1f, 0, 1e5f, "%.1f m")) {
                display.spacing = Double3{spacing[0], spacing[1], 0};
                changed = true;
            }
        }
    }
    ImGui::PopItemFlag();
    if (changed) {
        m_viewport.SetWorlds(std::move(display));
    }
}

}
