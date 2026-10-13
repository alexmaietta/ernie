// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/camera_sections.h"

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
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ernie::app {

namespace {

constexpr ImVec4 kError{0.89f, 0.34f, 0.32f, 1.0f};

/// What the viewport shows of a chain's stage, per viewer: the filter
/// its chain ends after (`RenderDelegate::SetViewStage`) and the var
/// it showed before.
struct Stage {
    Path filter;
    std::string previousVar;
};

std::map<const Viewer*, Stage>& Stages() {
    static std::map<const Viewer*, Stage> stages;
    return stages;
}

/// Each camera's product whose chain the Filter Chain section edits,
/// as chosen there.
std::map<Path, Path>& ChosenProducts() {
    static std::map<Path, Path> chosen;
    return chosen;
}

/// Ends `viewer`'s stage view, its var back as it was.
void ClearStage(Viewer& viewer) {
    const auto it = Stages().find(&viewer);
    if (it == Stages().end()) {
        return;
    }
    viewer.Delegate().SetViewStage({});
    viewer.SetViewVar(it->second.previousVar);
    Stages().erase(it);
}

/// The camera's exposure value at ISO 100, and the viewport's meter
/// when it looks through it.
void DrawExposure(Viewer& viewer, const Scene& scene,
                  const Path& camera) {
    try {
        const CameraParams settings =
            CameraParamsOf(scene, camera, 64, 64);
        const double ev = std::log2(settings.fStop * settings.fStop /
                                    settings.exposureTime) -
                          std::log2(settings.iso / 100.0) -
                          settings.exposureStops;
        ImGui::Text("Camera EV100 %.1f (f/%.1f, %.4g s, ISO %.0f)",
                    ev, settings.fStop, settings.exposureTime,
                    settings.iso);
    } catch (const std::exception& error) {
        ImGui::TextDisabled("%s", error.what());
    }
    if (viewer.ActiveCamera() != camera) {
        return;
    }
    if (const std::optional<double> ev = viewer.MeteredEv()) {
        ImGui::TextDisabled(
            "Scene EV100 %.1f%s", *ev,
            viewer.AutoExposing() ? ", exposing for it" : "");
    }
}

/// Each channel's quantum efficiency over wavelength, and as binned
/// into the camera's bands.
void DrawEfficiency(const Scene& scene, const Path& path) {
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
        // What the renderer uses: the curve's mean over each band.
        const std::vector<float> binned =
            channel.quantumEfficiency.Bin(bands);
        std::vector<double> edges;
        std::vector<double> steps;
        for (int band = 0; band < bands.count; ++band) {
            edges.push_back(bands.Lower(band));
            steps.push_back(binned[static_cast<std::size_t>(band)]);
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
    ImGui::TextDisabled("%d bands of %.1f nm from %.0f to %.0f nm",
                        bands.count, bands.Width(), bands.minNm,
                        bands.maxNm);
}

/// `sections` without the Prim section, the fields `hidden` and the
/// sections left empty.
void Without(std::vector<PropertySection>& sections,
             std::initializer_list<std::string_view> hidden) {
    std::erase_if(sections, [](const PropertySection& section) {
        return section.title == "Prim";
    });
    for (PropertySection& section : sections) {
        std::erase_if(
            section.fields, [&](const PropertyField& field) {
                return std::ranges::find(hidden, field.attribute) !=
                       hidden.end();
            });
    }
    std::erase_if(sections, [](const PropertySection& section) {
        return section.fields.empty();
    });
}

/// The products looking through the camera: each one's settings and
/// the render vars it holds, and a new one.
void DrawProducts(Viewer& viewer, Scene& scene, const Path& camera,
                  const std::vector<Path>& products,
                  float labelWidth) {
    std::function<void()> edit;
    if (products.empty()) {
        ImGui::TextDisabled("None: the viewport runs the default "
                            "chain through it.");
    }
    for (const Path& product : products) {
        ImGui::PushID(product.String().c_str());
        if (ImGui::TreeNode(product.String().c_str())) {
            std::vector<PropertySection> sections =
                SchemaOf(scene, product, viewer.Brdfs(),
                         viewer.Delegate().SensorFilters());
            Without(sections,
                    {tokens::kCameraRel, tokens::kOrderedVars,
                     tokens::kFilters});
            const std::string error = DrawPropertySections(
                viewer, product, sections, labelWidth);
            if (!error.empty()) {
                ImGui::TextColored(kError, "%s", error.c_str());
            }
            ImGui::TextDisabled("Render vars");
            for (const std::string_view name : kRenderVarNames) {
                bool held = HoldsVar(scene, product, name);
                const std::string label(name);
                if (ImGui::Checkbox(label.c_str(), &held)) {
                    edit = [&scene, product, name, held] {
                        SetProductVar(scene, product, name, held);
                    };
                }
            }
            ImGui::TreePop();
        }
        ImGui::PopID();
    }
    if (ImGui::Button("New Product")) {
        edit = [&scene, camera] {
            ChosenProducts()[camera] = CustomizeChain(scene, camera);
        };
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("A render product looking through this "
                          "camera, with LdrColor and a copy of the "
                          "default chain");
    }
    if (edit) {
        edit();
    }
}

/// The add menu: the built-in filters, then the registered ones by
/// kind. Returns the type chosen.
std::optional<std::string>
AddMenu(const SensorFilterRegistry& registry) {
    std::optional<std::string> chosen;
    ImGui::TextDisabled("Built-in");
    for (const SensorFilter filter : kSensorFilters) {
        if (filter == SensorFilter::Lens) {
            continue;  // always first
        }
        if (ImGui::Selectable(std::string(NameOf(filter)).c_str())) {
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
                ImGui::SetTooltip("%s", model.description.c_str());
            }
        }
    }
    return chosen;
}

/// The chain of one of the camera's products (the viewport's unless
/// chosen) as rows to bypass, reorder, remove, edit and view the
/// image after; with filters to add. Without products, the default
/// chain and a button customizing it.
void DrawChain(Viewer& viewer, Scene& scene, const Path& camera,
               const std::vector<Path>& products, float labelWidth) {
    const SensorFilterRegistry& registry =
        viewer.Delegate().SensorFilters();
    Path viewed;
    try {
        viewed = ViewChainProduct(scene, viewer.Settings(), camera);
    } catch (const std::exception& error) {
        ImGui::TextDisabled("%s", error.what());
        return;
    }
    if (products.empty()) {
        std::string chain;
        for (const SensorFilter filter : DefaultFilterChain()) {
            chain += (chain.empty() ? "" : "  >  ") +
                     std::string(NameOf(filter));
        }
        ImGui::TextWrapped("%s", chain.c_str());
        if (ImGui::Button("Customize")) {
            ChosenProducts()[camera] = CustomizeChain(scene, camera);
        } else if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("A render product looking through this "
                              "camera with a copy of the default "
                              "chain, to edit");
        }
        return;
    }
    Path& chosen = ChosenProducts()[camera];
    if (std::ranges::find(products, chosen) == products.end()) {
        chosen = viewed.IsEmpty() ? products.front() : viewed;
    }
    if (products.size() > 1) {
        ImGui::SetNextItemWidth(-1.0f);
        if (ImGui::BeginCombo("##product", chosen.String().c_str())) {
            for (const Path& product : products) {
                std::string label = product.String();
                if (product == viewed) {
                    label += "  (the viewport's)";
                }
                if (ImGui::Selectable(label.c_str(),
                                      product == chosen)) {
                    chosen = product;
                }
            }
            ImGui::EndCombo();
        }
    }
    const Path product = chosen;
    const bool viewport = product == viewed;
    const std::vector<Path> filters =
        RenderProduct(scene, product).GetFilters();
    // The stage shown, ended when its filter left the chain.
    Stage* stage = nullptr;
    if (const auto it = Stages().find(&viewer);
        it != Stages().end()) {
        if (std::ranges::find(filters, it->second.filter) ==
                filters.end() &&
            !scene.Has(it->second.filter)) {
            ClearStage(viewer);
        } else {
            stage = &it->second;
        }
    }
    // One edit a frame, after the rows are drawn.
    std::function<void()> edit;
    const float buttons = 4.0f * ImGui::GetFrameHeight() +
                          3.0f * ImGui::GetStyle().ItemSpacing.x;
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
            ImGui::SetTooltip("Unchecked, the chain runs without it");
        }
        ImGui::SameLine();
        const std::string type = prim.GetFilterType();
        const std::string label =
            type + "##" + std::string(filter.Name());
        const bool open = ImGui::TreeNodeEx(
            label.c_str(), ImGuiTreeNodeFlags_AllowOverlap);
        ImGui::SameLine();
        ImGui::TextDisabled("%s",
                            ChainFlowOf(type, registry).c_str());
        ImGui::SameLine(ImGui::GetContentRegionMax().x - buttons);
        // The image right after it, in the viewport.
        const bool shown =
            stage != nullptr && stage->filter == filter;
        ImGui::BeginDisabled(!viewport);
        if (shown) {
            ImGui::PushStyleColor(
                ImGuiCol_Button,
                ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
        }
        if (ImGui::Button("o",
                          ImVec2(ImGui::GetFrameHeight(), 0.0f))) {
            edit = [&viewer, camera, filter, type, shown, &registry] {
                if (shown) {
                    ClearStage(viewer);
                    return;
                }
                Stage& next = Stages()[&viewer];
                if (next.filter.IsEmpty()) {
                    next.previousVar = viewer.ViewVar();
                }
                next.filter = filter;
                if (viewer.ActiveCamera() != camera) {
                    viewer.SetActiveCamera(camera);
                }
                viewer.Delegate().SetViewStage(filter);
                viewer.SetViewVar(
                    std::string(StageVarOf(type, registry)));
            };
        }
        if (shown) {
            ImGui::PopStyleColor();
        }
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(
                ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip(
                viewport
                    ? (shown ? "Showing the image after it; click "
                               "for the whole chain's"
                             : "Show the image right after it in "
                               "the viewport")
                    : "The viewport runs another product's chain");
        }
        ImGui::SameLine();
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
        if (ImGui::Button("x",
                          ImVec2(ImGui::GetFrameHeight(), 0.0f))) {
            edit = [&scene, product, i] {
                RemoveChainFilter(scene, product, i);
            };
        }
        if (open) {
            std::vector<PropertySection> sections =
                SchemaOf(scene, filter, viewer.Brdfs(), registry);
            // Its type and switch are the row's.
            Without(sections,
                    {tokens::kFilterType, tokens::kFilterEnabled});
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
                scene, RenderProduct(scene, product).GetFilters()),
            registry);
    } catch (const std::exception& error) {
        ImGui::TextColored(kError, "%s", error.what());
    }
}

}  // namespace

void DrawCameraSections(Viewer& viewer, const Path& camera,
                        float labelWidth, std::string_view filter) {
    Scene& scene = viewer.GetScene();
    const auto shows = [&](const char* title) {
        return MatchesFilter(title, title, filter) &&
               SectionHeader(viewer, title);
    };
    if (shows("Exposure")) {
        DrawExposure(viewer, scene, camera);
    }
    if (shows("Quantum Efficiency")) {
        DrawEfficiency(scene, camera);
    }
    const std::vector<Path> products = ProductsOf(scene, camera);
    if (shows("Render Products")) {
        DrawProducts(viewer, scene, camera, products, labelWidth);
    }
    if (shows("Filter Chain")) {
        DrawChain(viewer, scene, camera, ProductsOf(scene, camera),
                  labelWidth);
    }
}

}  // namespace ernie::app
