// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/builtin_panels.h"
#include "ernie/app/prim_schemas.h"
#include "ernie/app/property_fields.h"
#include "ernie/app/sensor_chain.h"
#include "ernie/app/session.h"
#include "ernie/app/viewer.h"
#include "ernie/scene/camera.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/sensor_filters.h"
#include "ernie/scene/tokens.h"
#include "ernie/session/resolve.h"

#include <imgui.h>
#include <implot.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ernie::app {

namespace {

class CameraPanel final : public Panel {
  public:
    CameraPanel() : Panel("Camera", Dock::RightLower) {}

    void Draw(Viewer& viewer) override {
        Scene& scene = viewer.GetScene();
        const Path& camera = viewer.ActiveCamera();
        ImGui::TextUnformatted(camera.String().c_str());
        if (IsSessionPath(camera)) {
            ImGui::SameLine();
            ImGui::TextDisabled("(the viewer's)");
        }
        if (ImGui::Button("Create Camera from View")) {
            const Path created = viewer.CameraFromView(
                FreePath(scene, "/World/Camera"));
            viewer.SetActiveCamera(created);
            viewer.SetSelection({created});
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("A scene camera with the view's pose "
                              "and lens, looked through");
        }
        if (SectionHeader(viewer, "Viewport Exposure")) {
            DrawExposure(viewer, scene, camera);
        }
        const std::string error = DrawPropertySections(
            viewer, camera, schema_, PropertyLabelWidth(viewer));
        if (SectionHeader(viewer, "Quantum Efficiency")) {
            DrawEfficiency(scene, camera);
        }
        if (SectionHeader(viewer, "Filter Chain")) {
            DrawChain(viewer, scene, camera);
        }
        if (!error.empty()) {
            ImGui::TextColored(ImVec4(0.89f, 0.34f, 0.32f, 1.0f),
                               "%s", error.c_str());
        }
    }

  private:
    /// The viewport's exposure mode and compensation, and the meter:
    /// the scene's metered EV100 and the camera's own.
    static void DrawExposure(Viewer& viewer, const Scene& scene,
                             const Path& camera) {
        const char* modes[] = {
            "Default (auto for the viewer's camera)", "Auto",
            "Camera"};
        int mode = static_cast<int>(viewer.GetExposure());
        ImGui::SetNextItemWidth(-1.0f);
        if (ImGui::Combo("##exposure", &mode, modes, 3)) {
            viewer.SetExposure(static_cast<Viewer::Exposure>(mode));
        }
        float compensation =
            static_cast<float>(viewer.ExposureCompensation());
        ImGui::SetNextItemWidth(-1.0f);
        if (ImGui::SliderFloat("##compensation", &compensation, -6.0f,
                               6.0f, "Compensation %+.1f EV")) {
            viewer.SetExposureCompensation(compensation);
        }
        if (const std::optional<double> ev = viewer.MeteredEv()) {
            ImGui::Text("Scene EV100 %.1f%s", *ev,
                        viewer.AutoExposing() ? ", exposing for it"
                                              : "");
        } else {
            ImGui::TextDisabled("Scene EV100: not metered yet");
        }
        try {
            // The exposure value the camera's settings make, at ISO
            // 100 (EV100).
            const CameraParams settings =
                CameraParamsOf(scene, camera, 64, 64);
            const double ev =
                std::log2(settings.fStop * settings.fStop /
                          settings.exposureTime) -
                std::log2(settings.iso / 100.0) -
                settings.exposureStops;
            ImGui::TextDisabled(
                "Camera EV100 %.1f (f/%.1f, %.4g s, ISO %.0f)", ev,
                settings.fStop, settings.exposureTime, settings.iso);
        } catch (const std::exception&) {
        }
    }

    /// The sensor chain the viewport runs through `camera`: its
    /// product's (`ViewChainProduct`) as rows to bypass, reorder,
    /// remove and edit, with filters to add; else the default chain,
    /// with a button making it the camera's own. Edits are the
    /// product's `ernie:filters` and its SensorFilter prims.
    void DrawChain(Viewer& viewer, Scene& scene, const Path& camera) {
        const SensorFilterRegistry& registry =
            viewer.Delegate().SensorFilters();
        Path product;
        try {
            product =
                ViewChainProduct(scene, viewer.Settings(), camera);
        } catch (const std::exception& error) {
            ImGui::TextDisabled("%s", error.what());
            return;
        }
        if (product.IsEmpty()) {
            std::string chain;
            for (const SensorFilter filter : DefaultFilterChain()) {
                chain += (chain.empty() ? "" : "  >  ") +
                         std::string(NameOf(filter));
            }
            ImGui::TextWrapped("%s", chain.c_str());
            if (IsSessionPath(camera)) {
                ImGui::TextDisabled("The default chain; create a "
                                    "camera from the view to give it "
                                    "its own.");
            } else if (ImGui::Button("Customize")) {
                CustomizeChain(scene, camera);
            } else if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("A render product looking through "
                                  "this camera with a copy of the "
                                  "default chain, to edit");
            }
            return;
        }
        ImGui::TextDisabled("%s", product.String().c_str());
        const std::vector<Path> filters =
            RenderProduct(scene, product).GetFilters();
        // One edit a frame, after the rows are drawn.
        std::function<void()> edit;
        const float labelWidth = PropertyLabelWidth(viewer);
        for (std::size_t i = 0; i < filters.size(); ++i) {
            const Path& filter = filters[i];
            ImGui::PushID(static_cast<int>(i));
            if (!scene.Has(filter) ||
                scene.TypeOf(filter) != tokens::kSensorFilter) {
                ImGui::TextDisabled("%s: not a SensorFilter",
                                    filter.String().c_str());
                ImGui::PopID();
                continue;
            }
            SensorFilterPrim prim(scene, filter);
            bool enabled = prim.GetFilterEnabled();
            if (ImGui::Checkbox("##enabled", &enabled)) {
                prim.SetFilterEnabled(enabled);
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Unchecked, the chain runs "
                                  "without it");
            }
            ImGui::SameLine();
            const std::string type = prim.GetFilterType();
            const std::string label =
                type + "##" + std::string(filter.Name());
            const float buttons =
                3.0f * ImGui::GetFrameHeight() +
                2.0f * ImGui::GetStyle().ItemSpacing.x;
            const bool open = ImGui::TreeNodeEx(
                label.c_str(), ImGuiTreeNodeFlags_AllowOverlap);
            ImGui::SameLine();
            ImGui::TextDisabled("%s",
                                ChainFlowOf(type, registry).c_str());
            ImGui::SameLine(ImGui::GetContentRegionMax().x - buttons);
            if (ImGui::ArrowButton("##up", ImGuiDir_Up)) {
                edit = [&scene, product, i] {
                    MoveChainFilter(scene, product, i, -1);
                };
            }
            ImGui::SameLine();
            if (ImGui::ArrowButton("##down", ImGuiDir_Down)) {
                edit = [&scene, product, i] {
                    MoveChainFilter(scene, product, i, 1);
                };
            }
            ImGui::SameLine();
            if (ImGui::Button(
                    "x", ImVec2(ImGui::GetFrameHeight(), 0.0f))) {
                edit = [&scene, product, i] {
                    RemoveChainFilter(scene, product, i);
                };
            }
            if (open) {
                std::vector<PropertySection> sections =
                    SchemaOf(scene, filter, viewer.Brdfs(), registry);
                // Its type and switch are the row's.
                std::erase_if(sections, [](const PropertySection& s) {
                    return s.title == "Prim";
                });
                for (PropertySection& section : sections) {
                    std::erase_if(
                        section.fields,
                        [](const PropertyField& field) {
                            return field.attribute ==
                                       tokens::kFilterType ||
                                   field.attribute ==
                                       tokens::kFilterEnabled;
                        });
                }
                std::erase_if(sections, [](const PropertySection& s) {
                    return s.fields.empty();
                });
                const std::string error = DrawPropertySections(
                    viewer, filter, sections, labelWidth);
                if (!error.empty()) {
                    ImGui::TextColored(kError, "%s", error.c_str());
                }
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
        if (ImGui::Button("Add Filter")) {
            ImGui::OpenPopup("##add");
        }
        if (ImGui::BeginPopup("##add")) {
            if (std::optional<std::string> type = AddMenu(registry)) {
                edit = [&scene, &registry, product, type] {
                    AddChainFilter(scene, product, *type, registry);
                };
            }
            ImGui::EndPopup();
        }
        if (edit) {
            edit();
        }
        // What the chain makes of it, as rendering would.
        try {
            CheckFilterChain(
                SensorFilterStepsOf(
                    scene,
                    RenderProduct(scene, product).GetFilters()),
                registry);
        } catch (const std::exception& error) {
            ImGui::TextColored(kError, "%s", error.what());
        }
    }

    static constexpr ImVec4 kError{0.89f, 0.34f, 0.32f, 1.0f};

    /// The add menu: the built-in filters, then the registered ones
    /// by kind. Returns the type chosen.
    static std::optional<std::string>
    AddMenu(const SensorFilterRegistry& registry) {
        std::optional<std::string> chosen;
        ImGui::TextDisabled("Built-in");
        for (const SensorFilter filter : kSensorFilters) {
            if (filter == SensorFilter::Lens) {
                continue;  // always first
            }
            if (ImGui::Selectable(
                    std::string(NameOf(filter)).c_str())) {
                chosen = std::string(NameOf(filter));
            }
        }
        const std::pair<FilterKind, const char*> kinds[] = {
            {FilterKind::Spatial, "Spatial"},
            {FilterKind::Band, "Band"},
            {FilterKind::Integrator, "Integrator"},
            {FilterKind::Pixel, "Pixel"}};
        for (const auto& [kind, title] : kinds) {
            bool header = false;
            for (const SensorFilterModel& model : registry.Models()) {
                if (model.kind != kind) {
                    continue;
                }
                if (!header) {
                    ImGui::Separator();
                    ImGui::TextDisabled("%s", title);
                    header = true;
                }
                if (ImGui::Selectable(model.id.c_str())) {
                    chosen = model.id;
                }
                if (ImGui::IsItemHovered() &&
                    !model.description.empty()) {
                    ImGui::SetTooltip("%s",
                                      model.description.c_str());
                }
            }
        }
        return chosen;
    }

    /// `base`, or it with the first number after it that names no
    /// prim.
    static Path FreePath(const Scene& scene,
                         const std::string& base) {
        Path path(base);
        for (int n = 1; scene.Has(path); ++n) {
            path = Path(base + std::to_string(n));
        }
        return path;
    }

    /// Each channel's quantum efficiency over wavelength, and as
    /// binned into the camera's bands.
    static void DrawEfficiency(const Scene& scene, const Path& path) {
        CameraParams camera;
        try {
            camera = CameraParamsOf(scene, path, 64, 64);
        } catch (const std::exception& error) {
            ImGui::TextDisabled("%s", error.what());
            return;
        }
        const SpectralBands& bands = camera.bands;
        if (!ImPlot::BeginPlot("##qe", ImVec2(-1.0f, 180.0f),
                               ImPlotFlags_NoMenus)) {
            return;
        }
        ImPlot::SetupAxes("nm", "QE", ImPlotAxisFlags_AutoFit,
                          ImPlotAxisFlags_AutoFit);
        const double low = bands.minNm - 20.0;
        const double high = bands.maxNm + 20.0;
        for (const SensorChannel& channel : camera.sensor.channels) {
            std::vector<double> wavelengths;
            std::vector<double> values;
            for (double nm = low; nm <= high; nm += 2.0) {
                wavelengths.push_back(nm);
                values.push_back(channel.quantumEfficiency(nm));
            }
            ImPlot::PlotLine(channel.name.c_str(), wavelengths.data(),
                             values.data(),
                             static_cast<int>(values.size()));
            // What the renderer uses: the curve's mean over each
            // band.
            const std::vector<float> binned =
                channel.quantumEfficiency.Bin(bands);
            std::vector<double> edges;
            std::vector<double> steps;
            for (int band = 0; band < bands.count; ++band) {
                edges.push_back(bands.Lower(band));
                steps.push_back(
                    binned[static_cast<std::size_t>(band)]);
            }
            edges.push_back(bands.maxNm);
            steps.push_back(steps.empty() ? 0.0 : steps.back());
            ImPlotSpec spec;
            spec.FillAlpha = 0.25f;
            spec.Flags = ImPlotStairsFlags_Shaded;
            ImPlot::PlotStairs((channel.name + " bands").c_str(),
                               edges.data(), steps.data(),
                               static_cast<int>(steps.size()), spec);
        }
        ImPlot::EndPlot();
        ImGui::TextDisabled(
            "%d bands of %.1f nm from %.0f to %.0f nm", bands.count,
            bands.Width(), bands.minNm, bands.maxNm);
    }

    std::vector<PropertySection> schema_ = CameraSchema();
};

}  // namespace

std::unique_ptr<Panel> MakeCameraPanel() {
    return std::make_unique<CameraPanel>();
}

}  // namespace ernie::app
