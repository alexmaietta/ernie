// SPDX-License-Identifier: BSD-3-Clause
/// @file ui.cpp
/// @brief The viewer's UI, in the sections of ui.h: the panels, the theme and widgets, the orbit
/// camera, the shell and its menus, layout, mouse controls and heads-up display, and the
/// presenters: an offscreen image, and GLFW's windows, routed through volk's Vulkan loader, with
/// their swapchains and their input through ImGui's GLFW backend.
#include "ernie/app/ui.h"

#include "ernie/app/create.h"
#include "ernie/app/icons.h"
#include "ernie/device/context.h"
#include "ernie/scene/json.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/schema/camera.h"
#include "ernie/scene/schema/geom.h"
#include "ernie/scene/schema/render.h"
#include "ernie/scene/tokens.h"
#include "ernie/util/file.h"
#include "ernie/util/on_failure.h"
#include "ernie/util/process.h"
#include "ernie/util/profiler.h"
#include "ernie/util/text.h"
#include "ernie/util/version.h"

#include <glm/geometric.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <imgui_internal.h>
#include <imgui_stdlib.h>

#if ERNIE_VIEWER_WINDOW
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <imgui_impl_glfw.h>
#endif

#include <array>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <variant>

namespace ernie::app {

// Panels

void MessageLog::Add(Severity severity, std::string text) {
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start_;
    messages_.push_back({elapsed.count(), severity, std::move(text)});
    while (messages_.size() > limit_) {
        messages_.pop_front();
    }
}

std::unordered_set<Path> StageMatches(const Scene& scene, std::string_view filter) {
    std::unordered_set<Path> matches;
    if (filter.empty()) {
        return matches;
    }
    const std::string wanted = ToLower(filter);
    for (const Path& path : scene.Prims()) {
        if (ToLower(path.String()).find(wanted) == std::string::npos) {
            continue;
        }
        for (Path shown = path; !shown.IsEmpty() && !shown.IsRoot(); shown = shown.Parent()) {
            if (!matches.insert(shown).second) {
                break;
            }
        }
    }
    return matches;
}

namespace {

class LogPanel final : public Panel {
  public:
    LogPanel(MessageLog& log, const Theme& theme)
        : Panel("Log", Dock::Bottom), log_(log), theme_(theme) {}

    void Draw() override {
        ImGui::Checkbox("Info", &shown_[0]);
        ImGui::SameLine();
        ImGui::Checkbox("Warnings", &shown_[1]);
        ImGui::SameLine();
        ImGui::Checkbox("Errors", &shown_[2]);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(14.0f * ImGui::GetFontSize());
        ImGui::InputTextWithHint("##filter", "Filter", filter_.data(), filter_.size());
        ImGui::SameLine();
        if (ImGui::Button("Clear")) {
            log_.Clear();
        }
        ImGui::SameLine();
        const bool copy = ImGui::Button("Copy");
        std::string copied;
        if (ImGui::BeginChild("##messages", ImVec2(0.0f, 0.0f), ImGuiChildFlags_Borders)) {
            const std::string_view filter(filter_.data());
            for (const Message& message : log_.Messages()) {
                if (!shown_[static_cast<std::size_t>(message.severity)] ||
                    (!filter.empty() && message.text.find(filter) == std::string::npos)) {
                    continue;
                }
                ImGui::TextDisabled("%8.2f", message.time);
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, Colour(message.severity));
                ImGui::TextWrapped("%s", message.text.c_str());
                ImGui::PopStyleColor();
                if (copy) {
                    copied += message.text + "\n";
                }
            }
            // Follows new messages while scrolled to the end.
            if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 1.0f) {
                ImGui::SetScrollHereY(1.0f);
            }
        }
        ImGui::EndChild();
        if (copy) {
            ImGui::SetClipboardText(copied.c_str());
        }
    }

  private:
    ImVec4 Colour(Severity severity) const {
        switch (severity) {
        case Severity::Warning:
            return ToImVec4(theme_.warning);
        case Severity::Error:
            return ToImVec4(theme_.error);
        case Severity::Info:
            break;
        }
        return ImGui::GetStyleColorVec4(ImGuiCol_Text);
    }

    MessageLog& log_;
    const Theme& theme_;
    std::array<bool, 3> shown_{true, true, true};
    std::array<char, 128> filter_{};
};

// Prim type icons in the manner of Kit's stage icons, drawn as vectors so they are crisp at any
// size with no image assets.

ImU32 Rgb(int r, int g, int b, int a = 255) {
    return IM_COL32(r, g, b, a);
}

/// A point of an icon's square: (u, v) in [0, 1] from its top-left corner, or a length in it.
struct IconFrame {
    ImVec2 corner;
    float size;

    ImVec2 operator()(float u, float v) const {
        return {corner.x + u * size, corner.y + v * size};
    }
    float operator()(float length) const {
        return length * size;
    }
};

constexpr ImU32 kLamp = IM_COL32(226, 220, 132, 255);

void DrawTripod(ImDrawList* draw, const IconFrame& at) {
    const float width = std::max(1.2f, at(0.09f));
    const ImVec2 centre = at(0.5f, 0.58f);
    draw->AddLine(centre, at(0.5f, 0.1f), Rgb(110, 200, 90), width);
    draw->AddLine(centre, at(0.1f, 0.86f), Rgb(225, 80, 70), width);
    draw->AddLine(centre, at(0.9f, 0.86f), Rgb(80, 140, 235), width);
}

void DrawFolder(ImDrawList* draw, const IconFrame& at) {
    draw->AddRectFilled(at(0.08f, 0.2f), at(0.45f, 0.38f), Rgb(74, 85, 88), at(0.06f));
    draw->AddRectFilled(at(0.08f, 0.3f), at(0.92f, 0.82f), Rgb(96, 108, 112), at(0.08f));
}

/// A cube in three-quarter view, its faces lit from above-left.
void DrawCube(ImDrawList* draw, const IconFrame& at, float scale = 1.0f, ImVec2 offset = {}) {
    const auto p = [&](float u, float v) {
        return at(offset.x + 0.5f + (u - 0.5f) * scale, offset.y + 0.5f + (v - 0.5f) * scale);
    };
    const std::array top{p(0.5f, 0.1f), p(0.9f, 0.3f), p(0.5f, 0.5f), p(0.1f, 0.3f)};
    const std::array left{p(0.1f, 0.3f), p(0.5f, 0.5f), p(0.5f, 0.92f), p(0.1f, 0.72f)};
    const std::array right{p(0.5f, 0.5f), p(0.9f, 0.3f), p(0.9f, 0.72f), p(0.5f, 0.92f)};
    draw->AddConvexPolyFilled(top.data(), 4, Rgb(150, 160, 164));
    draw->AddConvexPolyFilled(left.data(), 4, Rgb(104, 114, 118));
    draw->AddConvexPolyFilled(right.data(), 4, Rgb(74, 85, 88));
}

/// A shaded ball: a base colour and a highlight up and to the left.
void DrawBall(ImDrawList* draw, const IconFrame& at, ImU32 base, ImU32 middle, ImU32 highlight) {
    draw->AddCircleFilled(at(0.5f, 0.52f), at(0.4f), base);
    draw->AddCircleFilled(at(0.44f, 0.44f), at(0.28f), middle);
    draw->AddCircleFilled(at(0.38f, 0.36f), at(0.11f), highlight);
}

void DrawCylinder(ImDrawList* draw, const IconFrame& at, bool cone) {
    const ImU32 side = Rgb(96, 108, 112);
    if (cone) {
        draw->AddTriangleFilled(at(0.5f, 0.08f), at(0.88f, 0.8f), at(0.12f, 0.8f), side);
        draw->AddEllipseFilled(at(0.5f, 0.8f), ImVec2(at(0.38f), at(0.12f)), Rgb(74, 85, 88));
        return;
    }
    draw->AddRectFilled(at(0.15f, 0.26f), at(0.85f, 0.8f), side);
    draw->AddEllipseFilled(at(0.5f, 0.8f), ImVec2(at(0.35f), at(0.12f)), side);
    draw->AddEllipseFilled(at(0.5f, 0.26f), ImVec2(at(0.35f), at(0.12f)), Rgb(150, 160, 164));
}

void DrawSun(ImDrawList* draw, const IconFrame& at) {
    draw->AddCircleFilled(at(0.5f, 0.5f), at(0.2f), kLamp);
    for (int k = 0; k < 8; ++k) {
        const float a = static_cast<float>(k) * std::numbers::pi_v<float> / 4.0f;
        draw->AddLine(at(0.5f + 0.3f * std::cos(a), 0.5f + 0.3f * std::sin(a)),
                      at(0.5f + 0.44f * std::cos(a), 0.5f + 0.44f * std::sin(a)), kLamp,
                      std::max(1.0f, at(0.07f)));
    }
}

void DrawBulb(ImDrawList* draw, const IconFrame& at) {
    draw->AddCircleFilled(at(0.5f, 0.4f), at(0.28f), kLamp);
    draw->AddRectFilled(at(0.38f, 0.64f), at(0.62f, 0.9f), Rgb(150, 160, 164), at(0.04f));
}

void DrawLightPanel(ImDrawList* draw, const IconFrame& at, bool disk) {
    if (disk) {
        draw->AddEllipseFilled(at(0.5f, 0.32f), ImVec2(at(0.38f), at(0.16f)), kLamp);
    } else {
        const std::array quad{at(0.22f, 0.16f), at(0.92f, 0.16f), at(0.78f, 0.46f),
                              at(0.08f, 0.46f)};
        draw->AddConvexPolyFilled(quad.data(), 4, kLamp);
    }
    for (const float u : {0.3f, 0.5f, 0.7f}) {
        draw->AddLine(at(u, 0.56f), at(u, 0.9f), kLamp, std::max(1.0f, at(0.07f)));
    }
}

void DrawDome(ImDrawList* draw, const IconFrame& at) {
    draw->PathArcTo(at(0.5f, 0.72f), at(0.38f), std::numbers::pi_v<float>,
                    2.0f * std::numbers::pi_v<float>, 16);
    draw->PathFillConvex(kLamp);
    draw->AddLine(at(0.08f, 0.76f), at(0.92f, 0.76f), kLamp, std::max(1.0f, at(0.08f)));
}

/// A five-pointed star with a smaller one beside it.
void DrawStars(ImDrawList* draw, const IconFrame& at) {
    const auto star = [&](float u, float v, float outer) {
        std::array<ImVec2, 10> points{};
        for (int k = 0; k < 10; ++k) {
            const float a = -0.5f * std::numbers::pi_v<float> +
                            static_cast<float>(k) * std::numbers::pi_v<float> / 5.0f;
            const float r = k % 2 == 0 ? outer : 0.42f * outer;
            points[static_cast<std::size_t>(k)] = at(u + r * std::cos(a), v + r * std::sin(a));
        }
        // Not convex: a fan of triangles from the centre.
        for (std::size_t k = 0; k < points.size(); ++k) {
            draw->AddTriangleFilled(at(u, v), points[k], points[(k + 1) % points.size()], kLamp);
        }
    };
    star(0.42f, 0.56f, 0.38f);
    star(0.8f, 0.2f, 0.16f);
}

void DrawCameraIcon(ImDrawList* draw, const IconFrame& at) {
    const ImU32 body = Rgb(150, 160, 164);
    draw->AddRectFilled(at(0.08f, 0.32f), at(0.66f, 0.78f), body, at(0.08f));
    draw->AddTriangleFilled(at(0.66f, 0.55f), at(0.94f, 0.32f), at(0.94f, 0.78f), body);
    draw->AddCircleFilled(at(0.26f, 0.22f), at(0.1f), body);
    draw->AddCircleFilled(at(0.48f, 0.2f), at(0.12f), body);
}

void DrawPlaneIcon(ImDrawList* draw, const IconFrame& at) {
    const std::array quad{at(0.3f, 0.3f), at(0.95f, 0.3f), at(0.7f, 0.78f), at(0.05f, 0.78f)};
    draw->AddConvexPolyFilled(quad.data(), 4, Rgb(104, 114, 118));
    draw->AddLine(at(0.5f, 0.3f), at(0.38f, 0.78f), Rgb(150, 160, 164), 1.0f);
    draw->AddLine(at(0.18f, 0.54f), at(0.83f, 0.54f), Rgb(150, 160, 164), 1.0f);
}

void DrawMountain(ImDrawList* draw, const IconFrame& at) {
    draw->AddTriangleFilled(at(0.36f, 0.22f), at(0.72f, 0.84f), at(0.02f, 0.84f),
                            Rgb(120, 140, 96));
    draw->AddTriangleFilled(at(0.66f, 0.38f), at(0.98f, 0.84f), at(0.36f, 0.84f),
                            Rgb(150, 126, 92));
    draw->AddTriangleFilled(at(0.36f, 0.22f), at(0.45f, 0.38f), at(0.27f, 0.38f),
                            Rgb(230, 232, 234));
}

void DrawFilm(ImDrawList* draw, const IconFrame& at) {
    const ImU32 colour = Rgb(127, 166, 194);
    draw->AddRect(at(0.1f, 0.18f), at(0.9f, 0.82f), colour, at(0.06f), 0,
                  std::max(1.2f, at(0.08f)));
    for (const float u : {0.24f, 0.42f, 0.6f, 0.78f}) {
        draw->AddRectFilled(at(u - 0.05f, 0.25f), at(u + 0.05f, 0.32f), colour);
        draw->AddRectFilled(at(u - 0.05f, 0.68f), at(u + 0.05f, 0.75f), colour);
    }
}

void DrawUntyped(ImDrawList* draw, const IconFrame& at) {
    draw->AddCircle(at(0.5f, 0.5f), at(0.24f), Rgb(110, 120, 124), 0, std::max(1.2f, at(0.08f)));
}

/// @brief Draws a prim type's icon in the square at `corner`, `size` pixels a side.
void DrawPrimIcon(ImDrawList* draw, const ImVec2& corner, float size, std::string_view type) {
    const IconFrame at{corner, size};
    const auto is = [&](std::initializer_list<std::string_view> types) {
        return std::ranges::find(types, type) != types.end();
    };
    if (type == "Xform") {
        DrawTripod(draw, at);
    } else if (type.empty() || type == "Scope") {
        // Untyped prims group others, as folders do.
        DrawFolder(draw, at);
    } else if (is({"Sphere", "Capsule"})) {
        DrawBall(draw, at, Rgb(74, 85, 88), Rgb(104, 114, 118), Rgb(170, 178, 182));
    } else if (is({"Cylinder", "Cone"})) {
        DrawCylinder(draw, at, type == "Cone");
    } else if (is({"Mesh", "Cube", "Points", "OpenVDBAsset", "NanoVDBField", "DenseField"})) {
        DrawCube(draw, at);
    } else if (is({"Plane", "GroundPlane"})) {
        DrawPlaneIcon(draw, at);
    } else if (type == "PointInstancer") {
        DrawCube(draw, at, 0.55f, {-0.2f, 0.18f});
        DrawCube(draw, at, 0.55f, {0.2f, 0.18f});
        DrawCube(draw, at, 0.55f, {0.0f, -0.18f});
    } else if (is({"Material", "Shader", "NodeGraph"})) {
        DrawBall(draw, at, Rgb(110, 92, 140), Rgb(150, 128, 178), Rgb(206, 198, 214));
    } else if (type == "DistantLight") {
        DrawSun(draw, at);
    } else if (is({"SphereLight", "CylinderLight", "LaserLight"})) {
        DrawBulb(draw, at);
    } else if (is({"RectLight", "DiskLight", "PortalLight"})) {
        DrawLightPanel(draw, at, type == "DiskLight");
    } else if (is({"DomeLight", "Atmosphere"})) {
        DrawDome(draw, at);
    } else if (type == "StarField") {
        DrawStars(draw, at);
    } else if (type == "Camera") {
        DrawCameraIcon(draw, at);
    } else if (type == "Terrain") {
        DrawMountain(draw, at);
    } else if (is({"RenderSettings", "RenderProduct", "RenderVar", "RenderPass"})) {
        DrawFilm(draw, at);
    } else {
        DrawUntyped(draw, at);
    }
}

class StagePanel final : public Panel {
  public:
    StagePanel(const Scene& scene, Selection& selection)
        : Panel("Stage", Dock::Right), scene_(scene), selection_(selection) {}

    void Draw() override {
        ImGui::SetNextItemWidth(-1.0f);
        if (ImGui::InputTextWithHint("##search", "Search", search_.data(), search_.size()) ||
            version_ != scene_.Version()) {
            matches_ = StageMatches(scene_, search_.data());
            version_ = scene_.Version();
        }
        constexpr ImGuiTableFlags kFlags = ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY |
                                           ImGuiTableFlags_Resizable |
                                           ImGuiTableFlags_BordersInnerV;
        if (!ImGui::BeginTable("##prims", 2, kFlags)) {
            return;
        }
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed,
                                8.0f * ImGui::GetFontSize());
        ImGui::TableHeadersRow();
        DrawChildren(Path::Root());
        ImGui::EndTable();
    }

  private:
    bool Searching() const {
        return search_[0] != '\0';
    }

    void DrawChildren(const Path& parent) {
        for (const Path& child : scene_.Children(parent)) {
            if (!Searching() || matches_.contains(child)) {
                DrawPrim(child);
            }
        }
    }

    void DrawPrim(const Path& path) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        const bool leaf = scene_.Children(path).empty();
        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanAllColumns |
                                   ImGuiTreeNodeFlags_OpenOnArrow |
                                   ImGuiTreeNodeFlags_OpenOnDoubleClick;
        if (leaf) {
            flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
        }
        if (selection_.Contains(path)) {
            flags |= ImGuiTreeNodeFlags_Selected;
        }
        // Searching opens the matches' ancestors.
        if (Searching()) {
            ImGui::SetNextItemOpen(true);
        }
        // Room before the name for the type's icon, drawn once the node has its place.
        const std::string_view type = scene_.TypeName(scene_.Require(path));
        const float icon = ImGui::GetTextLineHeight();
        const std::string name(path.Name());
        const bool open =
            ImGui::TreeNodeEx(path.String().c_str(), flags, "%*s%s",
                              static_cast<int>(std::ceil(1.4f * icon / ImGui::CalcTextSize(" ").x)),
                              "", name.c_str());
        const ImVec2 row = ImGui::GetItemRectMin();
        DrawPrimIcon(ImGui::GetWindowDrawList(),
                     ImVec2(row.x + ImGui::GetTreeNodeToLabelSpacing(),
                            row.y + 0.5f * (ImGui::GetItemRectSize().y - icon)),
                     icon, type);
        if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
            if (ImGui::GetIO().KeyCtrl) {
                selection_.Toggle(path);
            } else {
                selection_.Set(path);
            }
        }
        TableTexts({type});
        if (open && !leaf) {
            DrawChildren(path);
            ImGui::TreePop();
        }
    }

    const Scene& scene_;
    Selection& selection_;
    std::array<char, 128> search_{};
    std::unordered_set<Path> matches_;
    std::uint64_t version_ = 0;
};

/// @return The world box around the prims of `paths` the scene has.
Box WorldBounds(const Scene& scene, const std::vector<Path>& paths) {
    Box bounds;
    for (const Path& path : paths) {
        if (scene.Has(path)) {
            bounds.Extend(scene.WorldBounds(path));
        }
    }
    return bounds;
}

std::string VectorText(const glm::dvec3& vector) {
    std::array<char, 96> text{};
    std::snprintf(text.data(), text.size(), "%.4g, %.4g, %.4g", vector.x, vector.y, vector.z);
    return text.data();
}

std::string BoxText(const Box& box) {
    return box.Empty() ? "empty" : VectorText(box.min) + " to " + VectorText(box.max) + " m";
}

std::string_view TypeLabel(std::string_view type) {
    return type.empty() ? "untyped" : type;
}

std::string ValueText(const Value& value) {
    const auto numbers = [](const auto& components) {
        return Join(components, ", ", [](auto n) { return ToText(static_cast<double>(n)); });
    };
    const auto vector = [&]<glm::length_t N, typename T>(const glm::vec<N, T>& v) {
        return "(" + numbers(std::span(glm::value_ptr(v), static_cast<std::size_t>(N))) + ")";
    };
    return std::visit(
        [&]<typename T>(const T& data) -> std::string {
            if constexpr (std::is_same_v<T, std::monostate>) {
                return "none";
            } else if constexpr (std::is_same_v<T, bool>) {
                return data ? "true" : "false";
            } else if constexpr (std::is_arithmetic_v<T>) {
                return ToText(static_cast<double>(data));
            } else if constexpr (std::is_same_v<T, glm::dmat4>) {
                return Join(std::array{data[0], data[1], data[2], data[3]}, " ", vector);
            } else if constexpr (std::is_same_v<T, glm::dquat>) {
                return vector(glm::dvec4(data.w, data.x, data.y, data.z));
            } else if constexpr (std::is_same_v<T, std::string>) {
                return data;
            } else if constexpr (std::is_same_v<T, Path>) {
                return data.String();
            } else if constexpr (std::is_same_v<T, std::vector<float>>) {
                return "[" + numbers(data) + "]";
            } else if constexpr (std::is_same_v<T, std::vector<Path>>) {
                return "[" + Join(data, ", ", &Path::String) + "]";
            } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
                return "[" + Join(data, ", ") + "]";
            } else {
                return vector(data);
            }
        },
        value.Raw());
}

std::string AttributeText(const Scene& scene, PrimId prim, const std::string& attribute) {
    const std::size_t samples = scene.Samples(prim, attribute).size();
    const std::string sampled = samples > 0 ? ", " + std::to_string(samples) + " samples" : "";
    if (const Tensor* array = scene.Read(prim, attribute)) {
        return "array of " + Join(array->Shape(), " x ", [](auto n) { return std::to_string(n); }) +
               ", " + FormatBytes(array->Bytes()) + (array->OnHost() ? "" : " on a device") +
               sampled;
    }
    const std::optional<Value> value = scene.Get(prim, attribute);
    return (value ? ValueText(*value) : "none") + sampled;
}

/// @brief After Omniverse Kit's Property window, read-only: the first selected prim's path, type,
/// API schemas and bounds, then its authored attributes.
/// @return The edited value when the editor of a declared attribute changed it this frame.
/// Numbers drag, or slide within the declaration's range; colours pick; tokens with options
/// choose; texts and relationships commit on Enter; arrays and matrices show only.
std::optional<Value> EditValue(const AttributeDef& def, const Value& value) {
    const auto drag = [&](ImGuiDataType type, void* data, int components) {
        const bool ranged = def.meta.min < def.meta.max;
        return ranged ? ImGui::SliderScalarN("##v", type, data, components, &def.meta.min,
                                             &def.meta.max)
                      : ImGui::DragScalarN("##v", type, data, components, 0.01f);
    };
    switch (def.type) {
    case ColumnType::Bool: {
        bool on = value.Get<bool>() != nullptr && *value.Get<bool>();
        return ImGui::Checkbox("##v", &on) ? std::optional<Value>(on) : std::nullopt;
    }
    case ColumnType::Int: {
        double number = value.Get<int>() != nullptr ? *value.Get<int>() : 0.0;
        return drag(ImGuiDataType_Double, &number, 1)
                   ? std::optional<Value>(static_cast<int>(std::lround(number)))
                   : std::nullopt;
    }
    case ColumnType::Double: {
        double number = value.Get<double>() != nullptr ? *value.Get<double>() : 0.0;
        return drag(ImGuiDataType_Double, &number, 1) ? std::optional<Value>(number) : std::nullopt;
    }
    case ColumnType::Vec2:
    case ColumnType::Vec3:
    case ColumnType::Vec4: {
        glm::dvec4 v(0.0);
        const int n = def.type == ColumnType::Vec2 ? 2 : def.type == ColumnType::Vec3 ? 3 : 4;
        if (const auto* v2 = value.Get<glm::dvec2>()) {
            v = glm::dvec4(*v2, 0.0, 0.0);
        } else if (const auto* v3 = value.Get<glm::dvec3>()) {
            v = glm::dvec4(*v3, 0.0);
        } else if (const auto* v4 = value.Get<glm::dvec4>()) {
            v = *v4;
        }
        bool changed = false;
        if (def.meta.color && n >= 3) {
            glm::vec4 colour(v);
            changed =
                n == 3
                    ? ImGui::ColorEdit3("##v", glm::value_ptr(colour), ImGuiColorEditFlags_Float)
                    : ImGui::ColorEdit4("##v", glm::value_ptr(colour), ImGuiColorEditFlags_Float);
            v = glm::dvec4(colour);
        } else {
            changed = drag(ImGuiDataType_Double, glm::value_ptr(v), n);
        }
        if (!changed) {
            return std::nullopt;
        }
        return n == 2 ? Value(glm::dvec2(v)) : n == 3 ? Value(glm::dvec3(v)) : Value(v);
    }
    case ColumnType::TokenValue: {
        const std::string current =
            value.Get<std::string>() != nullptr ? *value.Get<std::string>() : std::string();
        if (!def.meta.options.empty()) {
            std::optional<Value> chosen;
            if (ImGui::BeginCombo("##v", current.c_str())) {
                for (const std::string_view option : def.meta.options) {
                    if (ImGui::Selectable(std::string(option).c_str(), option == current)) {
                        chosen = std::string(option);
                    }
                }
                ImGui::EndCombo();
            }
            return chosen;
        }
        std::string text = current;
        return ImGui::InputText("##v", &text, ImGuiInputTextFlags_EnterReturnsTrue)
                   ? std::optional<Value>(text)
                   : std::nullopt;
    }
    case ColumnType::Relationship: {
        std::string text = value.Get<Path>() != nullptr ? value.Get<Path>()->String() : "";
        return ImGui::InputTextWithHint("##v", "No target", &text,
                                        ImGuiInputTextFlags_EnterReturnsTrue)
                   ? std::optional<Value>(text.empty() ? Path() : Path(text))
                   : std::nullopt;
    }
    default:
        return std::nullopt;
    }
}

/// After Ernie 1's Properties panel: the first selected prim's schemas, a section each, their
/// attributes grouped and edited through the undo history, authored ones in the accent colour.
class PropertiesPanel final : public Panel {
  public:
    PropertiesPanel(Scene& scene, const Selection& selection, EditHistory& history, MessageLog& log)
        : Panel("Properties", Dock::RightLower), scene_(scene), selection_(selection),
          history_(history), log_(log) {}

    void Draw() override {
        if (selection_.Empty() || !scene_.Has(selection_.Paths().front())) {
            ImGui::TextDisabled("Select a prim in the Stage or the viewport.");
            return;
        }
        const Path& path = selection_.Paths().front();
        const PrimId prim = scene_.Require(path);
        ImGui::SetNextItemWidth(-1.0f);
        ImGui::InputTextWithHint("##search", "Search", search_.data(), search_.size());
        LabelValue("Path", path.String());
        LabelValue("Type", TypeLabel(scene_.TypeName(prim)));
        LabelValue("World bounds", BoxText(scene_.WorldBounds(path)));
        if (const std::size_t selected = selection_.Paths().size(); selected > 1) {
            ImGui::TextDisabled("The first of %zu selected.", selected);
        }
        if (const SchemaDef* schema = scene_.Schema(prim)) {
            DrawSchema(path, prim, schema->name, schema->AllAttributes(), {});
            for (const std::string_view api : schema->AllBuiltinApis()) {
                DrawApi(path, prim, api);
            }
        }
        for (const std::string& api : scene_.AppliedApis(prim)) {
            DrawApi(path, prim, api);
        }
        DrawUndeclared(path, prim);
    }

  private:
    void DrawApi(const Path& path, PrimId prim, std::string_view api) {
        const ApiName name = ApiName::Parse(api);
        DrawSchema(path, prim, api, name.schema->AllAttributes(), name);
    }

    /// Draws a schema's attributes in its section, a table a group.
    void DrawSchema(const Path& path, PrimId prim, std::string_view title,
                    const std::vector<const AttributeDef*>& attributes, const ApiName& api) {
        const std::string wanted = ToLower(search_.data());
        std::map<std::string_view, std::vector<std::pair<const AttributeDef*, std::string>>> groups;
        for (const AttributeDef* def : attributes) {
            std::string name =
                api.schema != nullptr ? api.Attribute(def->name) : std::string(def->name);
            if (ToLower(name).find(wanted) != std::string::npos) {
                groups[def->meta.group.empty() ? "Attributes" : def->meta.group].emplace_back(
                    def, std::move(name));
            }
        }
        if (groups.empty() ||
            !ImGui::CollapsingHeader(std::string(title).c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
            return;
        }
        for (const auto& [group, defs] : groups) {
            ImGui::SeparatorText(std::string(group).c_str());
            if (!ImGui::BeginTable(("##" + std::string(title) + std::string(group)).c_str(), 2,
                                   ImGuiTableFlags_Resizable | ImGuiTableFlags_BordersInnerV)) {
                continue;
            }
            ImGui::TableSetupColumn("Attribute", ImGuiTableColumnFlags_WidthFixed,
                                    11.0f * ImGui::GetFontSize());
            ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
            for (const auto& [def, name] : defs) {
                DrawAttribute(path, prim, *def, name);
            }
            ImGui::EndTable();
        }
    }

    void DrawAttribute(const Path& path, PrimId prim, const AttributeDef& def,
                       const std::string& name) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        const bool authored = scene_.IsAuthored(prim, name);
        const std::string_view label = def.meta.label.empty() ? def.name : def.meta.label;
        if (authored) {
            ImGui::TextColored(ImGui::GetStyleColorVec4(ImGuiCol_TextLink), "%.*s",
                               static_cast<int>(label.size()), label.data());
        } else {
            ImGui::Text("%.*s", static_cast<int>(label.size()), label.data());
        }
        if (ImGui::BeginItemTooltip()) {
            ImGui::Text("%s", name.c_str());
            ImGui::TextDisabled("%.*s", static_cast<int>(def.meta.doc.size()), def.meta.doc.data());
            ImGui::EndTooltip();
        }
        ImGui::PushID(name.c_str());
        if (ImGui::BeginPopupContextItem("##reset")) {
            if (ImGui::MenuItem("Reset to Fallback", nullptr, false, authored)) {
                Attempt([&] { history_.Clear(scene_, path, name); });
            }
            ImGui::EndPopup();
        }
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-FLT_MIN);
        const std::optional<Value> value = scene_.GetOrFallback(prim, name);
        const std::optional<Value> edited = value ? EditValue(def, *value) : std::nullopt;
        if (edited) {
            // A drag's frames merge into one step, closed once the editor lets go.
            Attempt([&] { history_.Set(scene_, path, name, *edited, ImGui::IsItemActive()); });
        } else if (!value || !EditableType(def.type)) {
            ImGui::TextDisabled("%s", AttributeText(scene_, prim, name).c_str());
        }
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            history_.Close();
        }
        ImGui::PopID();
    }

    /// Lists authored attributes no schema declares, as transform ops are, read-only.
    void DrawUndeclared(const Path& path, PrimId prim) {
        std::vector<std::string> undeclared;
        for (const std::string& name : scene_.Attributes(path)) {
            if (scene_.Declaration(prim, name) == nullptr &&
                !name.starts_with(tokens::kResolvedPrefix)) {
                undeclared.push_back(name);
            }
        }
        if (undeclared.empty() ||
            !ImGui::CollapsingHeader("Other", ImGuiTreeNodeFlags_DefaultOpen)) {
            return;
        }
        if (ImGui::BeginTable("##other", 2,
                              ImGuiTableFlags_Resizable | ImGuiTableFlags_BordersInnerV)) {
            for (const std::string& name : undeclared) {
                ImGui::TableNextRow();
                TableTexts({name, AttributeText(scene_, prim, name)});
            }
            ImGui::EndTable();
        }
    }

    static bool EditableType(ColumnType type) {
        return type <= ColumnType::Vec4 || type == ColumnType::TokenValue ||
               type == ColumnType::Relationship;
    }

    template <typename Edit> void Attempt(Edit&& edit) {
        try {
            edit();
        } catch (const std::exception& error) {
            log_.Error(error.what());
        }
    }

    Scene& scene_;
    const Selection& selection_;
    EditHistory& history_;
    MessageLog& log_;
    std::array<char, 128> search_{};
};

/// @brief After Ernie 1's Info panel: the scene's scale and the device's use, in sections side by
/// side, as the bottom dock is wide and short.
class InfoPanel final : public Panel {
  public:
    InfoPanel(const Scene& scene, const DeviceContext* device)
        : Panel("Info", Dock::Bottom), scene_(scene), device_(device) {}

    void Draw() override {
        if (counted_ != scene_.Version()) {
            Count();
        }
        if (!ImGui::BeginTable("##info", 3, ImGuiTableFlags_SizingStretchSame)) {
            return;
        }
        ImGui::TableNextColumn();
        ImGui::SeparatorText("Scene");
        LabelValue("Prims", std::to_string(prims_));
        LabelValue("Bounds", BoxText(bounds_));
        LabelValue("Size", bounds_.Empty() ? "none" : VectorText(bounds_.max - bounds_.min) + " m");
        LabelValue("Time code", ToText(scene_.Time()));
        ImGui::TableNextColumn();
        ImGui::SeparatorText("Prims by type");
        for (const auto& [type, count] : types_) {
            LabelValue(TypeLabel(type), std::to_string(count));
        }
        ImGui::TableNextColumn();
        ImGui::SeparatorText("Device");
        DrawDevice();
        ImGui::EndTable();
    }

  private:
    void Count() {
        prims_ = 0;
        types_.clear();
        for (const Path& path : scene_.Prims()) {
            if (!path.IsRoot()) {
                ++prims_;
                ++types_[std::string(scene_.TypeName(scene_.Require(path)))];
            }
        }
        bounds_ = WorldBounds(scene_, scene_.Children(Path::Root()));
        counted_ = scene_.Version();
    }

    void DrawDevice() const {
        const float fps = ImGui::GetIO().Framerate;
        std::array<char, 48> frame{};
        std::snprintf(frame.data(), frame.size(), "%.2f ms, %.0f fps",
                      fps > 0.0f ? 1000.0 / fps : 0.0, static_cast<double>(fps));
        LabelValue("Frame", frame.data());
        if (device_ == nullptr) {
            ImGui::TextDisabled("No device in use.");
            return;
        }
        const Transfers moved = device_->Transfers();
        LabelValue("Uploaded", FormatBytes(moved.uploaded));
        LabelValue("Read back", FormatBytes(moved.readBack));
        LabelValue("Copied", FormatBytes(moved.copied));
        const VkPhysicalDeviceMemoryProperties* memory = nullptr;
        vmaGetMemoryProperties(device_->Allocator(), &memory);
        std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> budgets{};
        vmaGetHeapBudgets(device_->Allocator(), budgets.data());
        for (std::uint32_t heap = 0; heap < memory->memoryHeapCount; ++heap) {
            const VkMemoryHeapFlags flags = memory->memoryHeaps[heap].flags;
            const bool local = (flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
            LabelValue("Heap " + std::to_string(heap) + (local ? ", device" : ", host"),
                       FormatBytes(budgets[heap].usage) + " of " +
                           FormatBytes(budgets[heap].budget));
        }
    }

    const Scene& scene_;
    const DeviceContext* device_;
    std::optional<std::uint64_t> counted_;
    std::size_t prims_ = 0;
    std::map<std::string, std::size_t> types_;
    Box bounds_;
};

std::string ApiVersion(const VulkanDeviceInfo& device) {
    return std::to_string(device.apiMajor) + "." + std::to_string(device.apiMinor);
}

/// @return The capabilities a device has, as a list of names.
std::string CapabilityNames(const DeviceCapabilities& capabilities) {
    const std::pair<bool, const char*> all[] = {
        {capabilities.rayQuery, "ray queries"},
        {capabilities.externalMemory, "shared memory"},
        {capabilities.externalTimeline, "shared timelines"},
        {capabilities.shaderInt64, "64-bit integers"},
        {capabilities.shaderFloat16, "16-bit floats"},
        {capabilities.memoryBudget, "memory budget"},
        {capabilities.deviceFault, "fault reports"},
        {capabilities.calibratedTimestamps, "GPU zones"},
        {capabilities.present, "presenting"},
        {capabilities.sparseResidency, "sparse residency"},
    };
    std::vector<std::string_view> names;
    for (const auto& [has, name] : all) {
        if (has) {
            names.emplace_back(name);
        }
    }
    return names.empty() ? "none" : Join(names, ", ");
}

class DevicePanel final : public Panel {
  public:
    explicit DevicePanel(const DeviceContext* device)
        : Panel("Devices", Dock::Bottom), device_(device) {}

    void Draw() override {
        if (device_ != nullptr) {
            const VulkanDeviceInfo& info = device_->Info();
            ImGui::SeparatorText("In use");
            LabelValue("Name", info.name);
            LabelValue("Type", ToString(info.type));
            LabelValue("Vulkan", ApiVersion(info));
            LabelValue("UUID", info.uuid);
            LabelValue("Capabilities", CapabilityNames(info.capabilities));
        } else {
            ImGui::TextDisabled("No device in use.");
        }
        // Listed when first opened: listing makes a Vulkan instance.
        if (!ImGui::CollapsingHeader("Installed")) {
            return;
        }
        if (ImGui::Button("Refresh") || !installed_) {
            List();
        }
        if (!error_.empty()) {
            ImGui::TextWrapped("%s", error_.c_str());
        }
        DrawInstalled();
    }

  private:
    void List() {
        try {
            installed_ = ListVulkanDevices();
            error_.clear();
        } catch (const DeviceError& error) {
            installed_ = std::vector<VulkanDeviceInfo>();
            error_ = error.what();
        }
    }

    void DrawInstalled() const {
        constexpr ImGuiTableFlags kFlags =
            ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_Resizable;
        if (!ImGui::BeginTable("##devices", 4, kFlags)) {
            return;
        }
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Type");
        ImGui::TableSetupColumn("Vulkan");
        ImGui::TableSetupColumn("Supported");
        ImGui::TableHeadersRow();
        for (const VulkanDeviceInfo& device : *installed_) {
            ImGui::TableNextRow();
            TableTexts({device.name});
            ImGui::SetItemTooltip("%s", device.uuid.c_str());
            TableTexts(
                {ToString(device.type), ApiVersion(device), device.supported ? "yes" : "no"});
        }
        ImGui::EndTable();
    }

    const DeviceContext* device_;
    std::optional<std::vector<VulkanDeviceInfo>> installed_;
    std::string error_;
};

using profiler::Frame;
using profiler::ReportFilter;
using profiler::ReportNode;
using profiler::ReportRow;

double FrameMs(const Frame& frame) {
    return static_cast<double>(frame.end - frame.start) * 1e-6;
}

/// @brief After Omniverse Kit's Profiler window: CPU and GPU zones averaged into trees, a bar per
/// frame, and exports as text, JSON and a Chrome trace.
class ProfilerPanel final : public Panel {
  public:
    ProfilerPanel(MessageLog& log, bool alwaysRecord)
        : Panel("Profiler", Dock::Bottom), log_(log), alwaysRecord_(alwaysRecord) {}

    void Draw() override {
        profiler::SetCpuEnabled(cpu_ || alwaysRecord_);
        profiler::SetGpuEnabled(gpu_ || alwaysRecord_);
        if (!paused_) {
            // Not the newest, whose GPU zones may still be arriving.
            frames_ = profiler::Frames();
            frames_.resize(frames_.size() - std::min(frames_.size(), profiler::kFramesInFlight));
        }
        DrawControls();
        if (ImGui::BeginTabBar("##profiler")) {
            if (ImGui::BeginTabItem("Zones")) {
                DrawZones();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Frames")) {
                DrawFrameBars();
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
    }

    /// Recording costs time, so it stops while nobody looks.
    void Undrawn() override {
        if (!alwaysRecord_) {
            profiler::SetEnabled(false);
        }
    }

  private:
    void DrawControls() {
        const float number = 5.0f * ImGui::GetFontSize();
        ImGui::Checkbox("CPU", &cpu_);
        ImGui::SetItemTooltip("Record the CPU's zones (ERNIE_ZONE).");
        ImGui::SameLine();
        ImGui::Checkbox("GPU", &gpu_);
        ImGui::SetItemTooltip("Record the GPU's zones (ERNIE_GPU_ZONE).");
        ImGui::SameLine();
        ImGui::Checkbox("Pause", &paused_);
        ImGui::SameLine();
        if (ImGui::Button("Clear")) {
            profiler::Clear();
            frames_.clear();
            selected_.reset();
        }
        if (!frames_.empty()) {
            ImGui::SameLine();
            ImGui::TextDisabled("%zu frames, %.1f fps", frames_.size(),
                                1000.0 / std::max(MeanMs(), 1e-3));
        }

        ImGui::Checkbox("Sort by time", &byTime_);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(number);
        ImGui::InputDouble("Min ms", &minMs_, 0.0, 0.0, "%.3f");
        minMs_ = std::max(minMs_, 0.0);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(number);
        ImGui::InputInt("Depth", &depth_, 0, 0);
        depth_ = std::clamp(depth_, 0, 32);
        ImGui::SameLine();
        ImGui::Checkbox("Average over (s)", &average_);
        ImGui::SetItemTooltip("Average the zones over the frames of the last seconds; else show "
                              "the newest frame, or the one clicked in Frames.");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(number);
        ImGui::BeginDisabled(!average_);
        ImGui::InputDouble("##averageSeconds", &averageSeconds_, 0.0, 0.0, "%.2f");
        averageSeconds_ = std::clamp(averageSeconds_, 0.05, 10.0);
        ImGui::EndDisabled();

        if (ImGui::Button("Copy Text")) {
            ImGui::SetClipboardText(profiler::ReportText(report_, Filter()).c_str());
        }
        ImGui::SameLine();
        if (ImGui::Button("Save JSON")) {
            Save("ernie_profile.json", profiler::ReportJson(report_));
        }
        ImGui::SameLine();
        if (ImGui::Button("Save Trace")) {
            Save("ernie_trace.json", profiler::TraceJson(frames_, profiler::ThreadNames()));
        }
        ImGui::SetItemTooltip("The frames kept, as a Chrome trace for Perfetto or "
                              "chrome://tracing.");
    }

    void Save(const std::filesystem::path& file, const std::string& text) {
        const std::filesystem::path path = std::filesystem::absolute(file);
        try {
            WriteTextFile(path, text);
            log_.Info("Saved " + path.string());
        } catch (const std::exception& error) {
            log_.Error(error.what());
        }
    }

    ReportFilter Filter() const {
        return {static_cast<std::uint32_t>(depth_), minMs_, byTime_};
    }

    double MeanMs() const {
        double total = 0.0;
        for (const Frame& frame : frames_) {
            total += FrameMs(frame);
        }
        return frames_.empty() ? 0.0 : total / static_cast<double>(frames_.size());
    }

    /// @return The frame inspected: the one clicked, else the newest with GPU zones, as the
    /// device runs a frame's commands a frame or two after the host records them.
    const Frame* Inspected() {
        if (selected_) {
            const auto found = std::ranges::find(frames_, *selected_, &Frame::index);
            if (found != frames_.end()) {
                return &*found;
            }
            selected_.reset();
        }
        const auto withGpu = std::ranges::find_if(frames_.rbegin(), frames_.rend(),
                                                  [](const Frame& f) { return !f.gpu.empty(); });
        if (withGpu != frames_.rend()) {
            return &*withGpu;
        }
        return frames_.empty() ? nullptr : &frames_.back();
    }

    /// @return The frames the Zones tab averages: those of the last `averageSeconds_`, else the
    /// one inspected.
    std::vector<Frame> Averaged() {
        if (!average_) {
            const Frame* frame = Inspected();
            return frame != nullptr ? std::vector<Frame>{*frame} : std::vector<Frame>{};
        }
        std::vector<Frame> frames;
        if (frames_.empty()) {
            return frames;
        }
        const auto window = static_cast<std::uint64_t>(averageSeconds_ * 1e9);
        for (const Frame& frame : frames_) {
            if (frame.end + window >= frames_.back().end) {
                frames.push_back(frame);
            }
        }
        return frames;
    }

    void DrawZones() {
        report_ = profiler::Summarize(Averaged(), profiler::ThreadNames());
        constexpr ImGuiTableFlags kFlags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV |
                                           ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable;
        if (!ImGui::BeginTable("##zones", 4, kFlags)) {
            return;
        }
        const float em = ImGui::GetFontSize();
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Duration (ms)", ImGuiTableColumnFlags_WidthFixed, 7.0f * em);
        ImGui::TableSetupColumn("Calls", ImGuiTableColumnFlags_WidthFixed, 3.5f * em);
        ImGui::TableSetupColumn("Percentage", ImGuiTableColumnFlags_WidthFixed, 8.0f * em);
        ImGui::TableHeadersRow();
        DrawSection("GPU", report_.gpu, report_.gpuMs, report_.gpuFrames, Filter());
        // The CPU's zones sit under a node per thread.
        ReportFilter cpu = Filter();
        ++cpu.depth;
        DrawSection("CPU", report_.cpu, report_.cpuMs, report_.cpuFrames, cpu);
        ImGui::EndTable();
    }

    /// One section of the Zones table: its total heading its nested zones.
    void DrawSection(const char* title, const std::vector<ReportNode>& nodes, double totalMs,
                     std::size_t frames, const ReportFilter& filter) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextLink));
        const bool shown = ImGui::TreeNodeEx(title, ImGuiTreeNodeFlags_SpanAllColumns |
                                                        ImGuiTreeNodeFlags_DefaultOpen);
        ImGui::TableNextColumn();
        ImGui::Text("%.3f", totalMs);
        ImGui::TableNextColumn();
        ImGui::TableNextColumn();
        if (frames == 0) {
            ImGui::TextUnformatted("(no frames)");
        } else {
            ImGui::Text("%zu frame%s", frames, frames == 1 ? "" : "s");
        }
        ImGui::PopStyleColor();
        if (!shown) {
            return;
        }
        DrawRows(profiler::Rows(nodes, filter));
        ImGui::TreePop();
    }

    static void DrawRows(const std::vector<ReportRow>& rows) {
        constexpr std::uint32_t kNone = std::numeric_limits<std::uint32_t>::max();
        std::uint32_t open = 0;        // tree levels pushed
        std::uint32_t closed = kNone;  // the depth of a closed parent
        for (const ReportRow& row : rows) {
            if (row.depth > closed) {
                continue;
            }
            closed = kNone;
            for (; open > row.depth; --open) {
                ImGui::TreePop();
            }
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGuiTreeNodeFlags flags =
                ImGuiTreeNodeFlags_SpanAllColumns | ImGuiTreeNodeFlags_DefaultOpen;
            if (!row.parent) {
                flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
            }
            const bool expanded = ImGui::TreeNodeEx(row.node->name.c_str(), flags);
            if (row.parent) {
                if (expanded) {
                    ++open;
                } else {
                    closed = row.depth;
                }
            }
            ImGui::TableNextColumn();
            ImGui::Text("%.3f", row.node->ms);
            ImGui::TableNextColumn();
            ImGui::Text("%.3g", row.node->calls);
            ImGui::TableNextColumn();
            std::array<char, 16> share{};
            std::snprintf(share.data(), share.size(), "%.2f%%", row.node->percent);
            ImGui::ProgressBar(static_cast<float>(row.node->percent / 100.0),
                               ImVec2(-1.0f, ImGui::GetTextLineHeight()), share.data());
        }
        for (; open > 0; --open) {
            ImGui::TreePop();
        }
    }

    /// A bar per frame, scaled to the frames' 95th percentile so one slow frame does not flatten
    /// the rest; clicking one inspects it.
    void DrawFrameBars() {
        const float height = 4.5f * ImGui::GetFontSize();
        const ImVec2 origin = ImGui::GetCursorScreenPos();
        const float width = std::max(ImGui::GetContentRegionAvail().x, 1.0f);
        ImGui::InvisibleButton("##frames", ImVec2(width, height));
        const bool hovered = ImGui::IsItemHovered();
        ImDrawList* draw = ImGui::GetWindowDrawList();
        draw->AddRectFilled(origin, ImVec2(origin.x + width, origin.y + height),
                            ImGui::GetColorU32(ImGuiCol_FrameBg));
        if (frames_.empty()) {
            ImGui::TextDisabled("No frames yet.");
            return;
        }
        std::vector<double> times;
        for (const Frame& frame : frames_) {
            times.push_back(FrameMs(frame));
        }
        std::ranges::sort(times);
        const double longest = std::max(1000.0 / 30.0 * 1.1, 1.5 * times[times.size() * 95 / 100]);
        const auto barHeight = [&](double ms) {
            return static_cast<float>(std::min(ms / longest, 1.0)) * height;
        };
        // Guides at 60 and 30 frames per second.
        for (const double guide : {1000.0 / 60.0, 1000.0 / 30.0}) {
            const float y = origin.y + height - barHeight(guide);
            draw->AddLine(ImVec2(origin.x, y), ImVec2(origin.x + width, y),
                          ImGui::GetColorU32(ImGuiCol_TextDisabled, 0.5f));
        }
        const std::size_t slots = std::max(profiler::History(), frames_.size());
        const float bar = width / static_cast<float>(slots);
        const float first = origin.x + width - bar * static_cast<float>(frames_.size());
        const Frame* inspected = Inspected();
        const ImVec2 mouse = ImGui::GetIO().MousePos;
        for (std::size_t i = 0; i < frames_.size(); ++i) {
            const Frame& frame = frames_[i];
            const double ms = FrameMs(frame);
            const float x = first + bar * static_cast<float>(i);
            const bool under = hovered && mouse.x >= x && mouse.x < x + bar;
            const ImGuiCol colour = inspected != nullptr && frame.index == inspected->index
                                        ? ImGuiCol_PlotHistogramHovered
                                        : (under ? ImGuiCol_ButtonHovered : ImGuiCol_PlotHistogram);
            draw->AddRectFilled(ImVec2(x, origin.y + height - barHeight(ms)),
                                ImVec2(x + std::max(bar - 1.0f, 1.0f), origin.y + height),
                                ImGui::GetColorU32(colour));
            if (under) {
                ImGui::SetTooltip("Frame %llu: %.2f ms",
                                  static_cast<unsigned long long>(frame.index), ms);
                if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                    selected_ = frame.index;
                    paused_ = true;
                }
            }
        }
        if (inspected != nullptr) {
            ImGui::Text("Frame %llu: %.2f ms, %zu CPU zones, %zu GPU zones",
                        static_cast<unsigned long long>(inspected->index), FrameMs(*inspected),
                        inspected->cpu.size(), inspected->gpu.size());
        }
    }

    MessageLog& log_;
    bool alwaysRecord_;
    bool cpu_ = true;
    bool gpu_ = true;
    bool paused_ = false;
    bool byTime_ = false;
    double minMs_ = 0.0;
    int depth_ = 8;
    bool average_ = true;
    double averageSeconds_ = 1.0;
    std::vector<Frame> frames_;
    std::optional<std::uint64_t> selected_;
    profiler::Report report_;
};

/// @brief A help window, centred when it opens, drawn while `open`; its close button clears it.
template <typename Contents> void HelpWindow(const char* title, bool& open, Contents&& contents) {
    if (!open) {
        return;
    }
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing,
                            ImVec2(0.5f, 0.5f));
    constexpr ImGuiWindowFlags kFlags = ImGuiWindowFlags_NoDocking |
                                        ImGuiWindowFlags_AlwaysAutoResize |
                                        ImGuiWindowFlags_NoCollapse;
    if (ImGui::Begin(title, &open, kFlags)) {
        contents();
    }
    ImGui::End();
}

}  // namespace

std::unique_ptr<Panel> MakeLogPanel(MessageLog& log, const Theme& theme) {
    return std::make_unique<LogPanel>(log, theme);
}

std::unique_ptr<Panel> MakeStagePanel(const Scene& scene, Selection& selection) {
    return std::make_unique<StagePanel>(scene, selection);
}

std::unique_ptr<Panel> MakePropertiesPanel(Scene& scene, const Selection& selection,
                                           EditHistory& history, MessageLog& log) {
    return std::make_unique<PropertiesPanel>(scene, selection, history, log);
}

std::unique_ptr<Panel> MakeInfoPanel(const Scene& scene, const DeviceContext* device) {
    return std::make_unique<InfoPanel>(scene, device);
}

std::unique_ptr<Panel> MakeDevicePanel(const DeviceContext* device) {
    return std::make_unique<DevicePanel>(device);
}

std::unique_ptr<Panel> MakeProfilerPanel(MessageLog& log, bool alwaysRecord) {
    return std::make_unique<ProfilerPanel>(log, alwaysRecord);
}

void DrawControlsWindow(bool& open) {
    constexpr std::pair<const char*, const char*> kControls[] = {
        {"Left drag", "Orbit the camera around its target"},
        {"Right drag", "Pan along the ground"},
        {"Middle drag", "Pan across the view"},
        {"Wheel", "Zoom towards the target"},
        {"F", "Frame the selection, or the whole scene"},
        {"Click", "Select what the viewport shows there, or nothing; Ctrl adds or removes it"},
        {"Click in Stage", "Select a prim; Ctrl adds or removes it"},
        {"H", "Hide or show the toolbar and heads-up display"},
        {"Ctrl+Z, Ctrl+Shift+Z", "Undo, redo"},
        {"Ctrl+D, Del", "Duplicate, delete the selection"},
        {"Ctrl+O, Ctrl+Shift+S", "Open a scene, save it as"},
        {"Ctrl+Q", "Quit"},
    };
    HelpWindow("Keyboard and Mouse", open, [&] {
        if (ImGui::BeginTable("##controls", 2,
                              ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV)) {
            for (const auto& [input, action] : kControls) {
                ImGui::TableNextRow();
                TableTexts({input, action});
            }
            ImGui::EndTable();
        }
    });
}

void DrawAboutWindow(bool& open, std::string_view device) {
    HelpWindow("About Ernie", open, [&] {
        ImGui::TextUnformatted(
            "Ernie: Efficient Renderer for Natural and Interplanetary Environments");
        LabelValue("Version", Version());
        LabelValue("Device", device.empty() ? std::string_view("none") : device);
        LabelValue("Dear ImGui", IMGUI_VERSION);
        ImGui::Separator();
        ImGui::TextUnformatted("BSD-3-Clause licence. https://github.com/alexmaietta/ernie");
    });
}

// Theme and widgets

namespace {

ImVec2 ToImVec2(const glm::vec2& size) {
    return {size.x, size.y};
}

glm::vec4 WithAlpha(const glm::vec4& colour, float alpha) {
    return {colour.r, colour.g, colour.b, alpha};
}

void SetMetrics(const Theme& theme, ImGuiStyle& style) {
    style.WindowPadding = {8.0f, 6.0f};
    style.FramePadding = ToImVec2(theme.framePadding);
    style.ItemSpacing = ToImVec2(theme.itemSpacing);
    style.ItemInnerSpacing = {4.0f, 3.0f};
    style.CellPadding = {4.0f, 2.0f};
    style.IndentSpacing = 14.0f;
    style.ScrollbarSize = 10.0f;
    style.GrabMinSize = 8.0f;
    style.WindowRounding = 0.0f;
    style.ChildRounding = 0.0f;
    style.FrameRounding = theme.rounding;
    style.PopupRounding = theme.rounding;
    style.ScrollbarRounding = theme.rounding;
    style.GrabRounding = theme.rounding;
    style.TabRounding = theme.rounding;
    style.WindowBorderSize = 1.0f;
    style.ChildBorderSize = 1.0f;
    style.PopupBorderSize = 1.0f;
    style.FrameBorderSize = theme.frameBorder;
    // Framed tabs, without a line over the shown one.
    style.TabBorderSize = theme.frameBorder;
    style.TabBarOverlineSize = 0.0f;
    style.DockingSeparatorSize = 1.0f;
    style.WindowMenuButtonPosition = ImGuiDir_None;
    style.WindowTitleAlign = {0.0f, 0.5f};
    style.SeparatorTextBorderSize = 1.0f;
}

void SetColours(const Theme& theme, ImGuiStyle& style) {
    const auto set = [&](ImGuiCol index, const glm::vec4& colour) {
        style.Colors[index] = ToImVec4(colour);
    };
    const glm::vec4 accentDim = WithAlpha(theme.accent, 0.45f);
    const glm::vec4 accentFaint = WithAlpha(theme.accent, 0.25f);
    const glm::vec4 clear(0.0f);
    set(ImGuiCol_Text, theme.text);
    set(ImGuiCol_TextDisabled, theme.textMuted);
    set(ImGuiCol_WindowBg, theme.panel);
    set(ImGuiCol_ChildBg, clear);
    set(ImGuiCol_PopupBg, theme.popup);
    set(ImGuiCol_Border, theme.border);
    set(ImGuiCol_BorderShadow, clear);
    set(ImGuiCol_FrameBg, theme.field);
    set(ImGuiCol_FrameBgHovered, theme.fieldHovered);
    set(ImGuiCol_FrameBgActive, theme.fieldHovered);
    for (const ImGuiCol title : {ImGuiCol_TitleBg, ImGuiCol_TitleBgActive,
                                 ImGuiCol_TitleBgCollapsed, ImGuiCol_MenuBarBg}) {
        set(title, theme.background);
    }
    set(ImGuiCol_ScrollbarBg, clear);
    set(ImGuiCol_ScrollbarGrab, theme.fieldHovered);
    set(ImGuiCol_ScrollbarGrabHovered, theme.textMuted);
    set(ImGuiCol_ScrollbarGrabActive, theme.accent);
    // A white box with a black check; menus mark their items in the accent instead.
    set(ImGuiCol_CheckMark, theme.checkMark);
    set(ImGuiCol_CheckboxSelectedBg, theme.checkBox);
    set(ImGuiCol_SliderGrab, theme.buttonLit);
    set(ImGuiCol_SliderGrabActive, theme.text);
    set(ImGuiCol_Button, theme.button);
    set(ImGuiCol_ButtonHovered, theme.fieldHovered);
    set(ImGuiCol_ButtonActive, theme.buttonLit);
    // Selected and hovered rows of trees, lists and menus.
    set(ImGuiCol_Header, theme.selection);
    set(ImGuiCol_HeaderHovered, WithAlpha(theme.selection, 0.85f));
    set(ImGuiCol_HeaderActive, theme.selection);
    set(ImGuiCol_Separator, theme.separator);
    set(ImGuiCol_SeparatorHovered, accentDim);
    set(ImGuiCol_SeparatorActive, theme.accent);
    set(ImGuiCol_ResizeGrip, clear);
    set(ImGuiCol_ResizeGripHovered, accentDim);
    set(ImGuiCol_ResizeGripActive, theme.accent);
    set(ImGuiCol_InputTextCursor, theme.text);
    // Tabs on the darker bar, the shown one in the panel's colour.
    set(ImGuiCol_Tab, theme.background);
    set(ImGuiCol_TabHovered, theme.fieldHovered);
    set(ImGuiCol_TabSelected, theme.panel);
    set(ImGuiCol_TabSelectedOverline, theme.accent);
    set(ImGuiCol_TabDimmed, theme.background);
    set(ImGuiCol_TabDimmedSelected, theme.panel);
    set(ImGuiCol_TabDimmedSelectedOverline, WithAlpha(theme.accent, 0.0f));
    set(ImGuiCol_DockingPreview, accentDim);
    set(ImGuiCol_DockingEmptyBg, theme.background);
    set(ImGuiCol_PlotLines, theme.accent);
    set(ImGuiCol_PlotLinesHovered, theme.text);
    set(ImGuiCol_PlotHistogram, theme.accent);
    set(ImGuiCol_PlotHistogramHovered, theme.text);
    set(ImGuiCol_TableHeaderBg, theme.header);
    set(ImGuiCol_TableBorderStrong, theme.separator);
    set(ImGuiCol_TableBorderLight, theme.separator);
    set(ImGuiCol_TableRowBg, clear);
    set(ImGuiCol_TableRowBgAlt, WithAlpha(theme.text, 0.03f));
    set(ImGuiCol_TextLink, theme.accent);
    set(ImGuiCol_TextSelectedBg, accentFaint);
    set(ImGuiCol_TreeLines, theme.separator);
    set(ImGuiCol_DragDropTarget, theme.accent);
    set(ImGuiCol_DragDropTargetBg, accentFaint);
    set(ImGuiCol_UnsavedMarker, theme.warning);
    set(ImGuiCol_NavCursor, theme.accent);
    set(ImGuiCol_NavWindowingHighlight, theme.accent);
    set(ImGuiCol_NavWindowingDimBg, WithAlpha(theme.background, 0.6f));
    set(ImGuiCol_ModalWindowDimBg, WithAlpha(theme.background, 0.6f));
}

}  // namespace

void ApplyTheme(const Theme& theme, float contentScale) {
    ImGuiStyle& style = ImGui::GetStyle();
    style = ImGuiStyle();
    SetMetrics(theme, style);
    SetColours(theme, style);
    style.ScaleAllSizes(contentScale);
    style.FontSizeBase = theme.fontSize;
    style.FontScaleDpi = contentScale;
}

void LabelValue(std::string_view label, std::string_view value) {
    ImGui::TextUnformatted(label.data(), label.data() + label.size());
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(value.data(), value.data() + value.size());
    ImGui::PopTextWrapPos();
    ImGui::PopStyleColor();
}

void TableTexts(std::initializer_list<std::string_view> texts) {
    for (const std::string_view text : texts) {
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(text.data(), text.data() + text.size());
    }
}

std::string FormatBytes(std::uint64_t bytes) {
    constexpr std::array kUnits{"B", "KiB", "MiB", "GiB", "TiB"};
    auto value = static_cast<double>(bytes);
    std::size_t unit = 0;
    while (value >= 1024.0 && unit + 1 < kUnits.size()) {
        value /= 1024.0;
        ++unit;
    }
    std::array<char, 32> text{};
    std::snprintf(text.data(), text.size(), unit == 0 ? "%.0f %s" : "%.1f %s", value, kUnits[unit]);
    return text.data();
}

// Orbit camera

namespace {

/// @return Metres per pixel at the target's depth in a view `height` pixels tall.
double MetresPerPixel(const OrbitCamera& camera, double height) {
    return 2.0 * camera.distance * std::tan(0.5 * camera.verticalFov) / std::max(height, 1.0);
}

}  // namespace

glm::dvec3 OrbitCamera::Eye() const {
    const double horizontal = std::cos(pitch) * distance;
    return target + glm::dvec3(std::sin(yaw) * horizontal, -std::cos(yaw) * horizontal,
                               std::sin(pitch) * distance);
}

glm::dmat4 OrbitCamera::Transform() const {
    return glm::inverse(glm::lookAtRH(Eye(), target, glm::dvec3(0.0, 0.0, 1.0)));
}

void OrbitCamera::LookFrom(const glm::dmat4& cameraToWorld) {
    // The camera looks down its -z: Eye's direction to the target, inverted.
    const glm::dvec3 forward = glm::normalize(-glm::dvec3(cameraToWorld[2]));
    pitch = std::clamp(std::asin(std::clamp(-forward.z, -1.0, 1.0)), kMinPitch, kMaxPitch);
    yaw = std::atan2(-forward.x, forward.y);
    target = glm::dvec3(cameraToWorld[3]) + distance * forward;
}

void OrbitCamera::Orbit(double deltaYaw, double deltaPitch) {
    yaw += deltaYaw;
    pitch = std::clamp(pitch + deltaPitch, kMinPitch, kMaxPitch);
}

void OrbitCamera::Pan(double dx, double dy, double height) {
    const glm::dvec3 right(std::cos(yaw), std::sin(yaw), 0.0);
    const glm::dvec3 forward(-std::sin(yaw), std::cos(yaw), 0.0);
    target += (-dx * right + dy * forward) * MetresPerPixel(*this, height);
}

void OrbitCamera::PanVertical(double dx, double dy, double height) {
    const glm::dvec3 right(std::cos(yaw), std::sin(yaw), 0.0);
    target += (-dx * right + dy * glm::dvec3(0.0, 0.0, 1.0)) * MetresPerPixel(*this, height);
}

void OrbitCamera::Zoom(double steps) {
    distance = std::clamp(distance * std::pow(0.9, steps), kMinDistance, kMaxDistance);
}

void OrbitCamera::Frame(const Box& box) {
    if (box.Empty()) {
        return;
    }
    target = box.Center();
    const double radius = std::max(0.5 * glm::length(box.max - box.min), kMinDistance);
    distance = std::clamp(1.1 * radius / std::sin(0.5 * verticalFov), kMinDistance, kMaxDistance);
}

// Shell

namespace {

/// Hashed alone, not on the ID stack, so it is the same outside a frame.
const ImGuiID kDockSpace = ImHashStr("ErnieDockSpace");

/// Radians the camera turns per pixel dragged.
constexpr double kOrbitRate = 0.005;

/// Seconds between refreshes of the heads-up display.
constexpr double kHudRefresh = 0.25;

constexpr std::array kMouseButtons{ImGuiMouseButton_Left, ImGuiMouseButton_Right,
                                   ImGuiMouseButton_Middle};

// Menu items as Omniverse Kit draws them: a left gutter holds a check on toggles in the accent
// colour (`ImGuiCol_TextLink`), shortcuts sit muted at the right, and menus open at least 17 lines
// wide.

/// Spaces before every label, where the markers go.
constexpr const char* kGutter = "    ";

/// @return Whether the plain item was clicked.
bool MenuAction(const char* label, const char* shortcut = nullptr, bool enabled = true) {
    return ImGui::MenuItem((std::string(kGutter) + label).c_str(), shortcut, false, enabled);
}

/// @brief A toggle, checked in the gutter when on; a click flips `on`.
void MenuCheck(const char* label, bool& on) {
    if (MenuAction(label)) {
        on = !on;
    }
    if (on) {
        const ImVec2 low = ImGui::GetItemRectMin();
        const ImVec2 high = ImGui::GetItemRectMax();
        const float size = 0.7f * ImGui::GetTextLineHeight();
        const ImVec2 centre(low.x + 0.5f * ImGui::CalcTextSize(kGutter).x, 0.5f * (low.y + high.y));
        ImGui::RenderCheckMark(ImGui::GetWindowDrawList(),
                               ImVec2(centre.x - 0.5f * size, centre.y - 0.5f * size),
                               ImGui::GetColorU32(ImGuiCol_TextLink), size);
    }
}

/// @brief A section heading inside a menu, muted, before its items.
void MenuSection(const char* label) {
    ImGui::TextDisabled("%s", label);
}

template <typename Items> void Menu(const char* name, Items&& items) {
    // A menu that stays closed leaves the constraint unused.
    ImGui::SetNextWindowSizeConstraints(ImVec2(17.0f * ImGui::GetFontSize(), 0.0f),
                                        ImVec2(FLT_MAX, FLT_MAX));
    if (ImGui::BeginMenu(name)) {
        items();
        ImGui::EndMenu();
    }
}

/// @brief Opens `target`, a URL or a folder, in the desktop's own application, in the background.
/// @return Whether that could start.
bool OpenExternally(const std::string& target) {
#if defined(_WIN32)
    const std::string command = "start \"\" \"" + target + "\"";
#elif defined(__APPLE__)
    const std::string command = "open '" + target + "' &";
#else
    const std::string command = "xdg-open '" + target + "' >/dev/null 2>&1 &";
#endif
    return std::system(command.c_str()) == 0;
}

constexpr const char* kDocumentation = "https://alexmaietta.github.io/ernie/";
constexpr const char* kRepository = "https://github.com/alexmaietta/ernie";

/// @return Whether a prim of `type` groups others: new prims go under a selected one.
bool IsGroup(std::string_view type) {
    return type == Xform::kName || type == Scope::kName;
}

/// @return A camera the scene's render settings name, for new render prims; empty for none.
Path SettingsCamera(const Scene& scene) {
    for (const Path& settings : scene.Prims(RenderSettings::kName)) {
        const Path camera = RenderSettings(scene, settings).GetCamera();
        if (!camera.IsEmpty()) {
            return camera;
        }
    }
    return {};
}

}  // namespace

Shell::Shell(Scene& scene, const DeviceContext* device, const ShellOptions& options)
    : scene_(scene), device_(device), theme_(options.theme), sizes_(options.docks),
      display_(options.display) {
    // Stage and Demos above Properties on the right; Log, Capture, Info, Devices and Profiler
    // tabbed at the bottom.
    panels_.push_back(MakeStagePanel(scene, selection_));
    panels_.push_back(MakeDemosPanel(*this));
    panels_.push_back(MakePropertiesPanel(scene, selection_, history_, log_));
    panels_.push_back(MakeLogPanel(log_, theme_));
    panels_.push_back(MakeCapturePanel(*this));
    panels_.push_back(MakeInfoPanel(scene, device));
    panels_.push_back(MakeDevicePanel(device));
    panels_.push_back(MakeProfilerPanel(log_, options.profile));
    capture_.folder = DefaultCaptureFolder();
    if (options.profile) {
        profiler::SetEnabled(true);
    }
}

Panel* Shell::FindPanel(std::string_view title) const {
    const auto found = std::ranges::find(panels_, title, &Panel::Title);
    return found != panels_.end() ? found->get() : nullptr;
}

std::pair<ImVec2, ImVec2> Shell::Viewport() const {
    ImVec2 origin;
    ImVec2 size;
    if (const ImGuiDockNode* centre = ImGui::DockBuilderGetCentralNode(kDockSpace)) {
        origin = centre->Pos;
        size = centre->Size;
    } else {
        const ImGuiViewport* main = ImGui::GetMainViewport();
        origin = main->WorkPos;
        size = main->WorkSize;
    }
    size.y = std::max(size.y - TimelineHeight(), 1.0f);
    return {origin, size};
}

float Shell::TimelineHeight() const {
    return display_.timeline && !overlaysHidden_
               ? ImGui::GetFrameHeight() + 2.0f * ImGui::GetStyle().WindowPadding.y
               : 0.0f;
}

void Shell::Draw() {
    ERNIE_ZONE("Draw the UI");
    choices_ = {};
    {
        ERNIE_ZONE("Animate");
        timeline_.Advance(scene_, ImGui::GetIO().DeltaTime);
        if (demo_ && !demoPaused) {
            Attempt("Stepping the demo", [&] { demo_->Advance(); });
        }
    }
    {
        ERNIE_ZONE("Draw the menus");
        DrawMenuBar();
    }
    {
        ERNIE_ZONE("Lay out the docks");
        DrawDockSpace();
    }
    DrawPanels();
    {
        ERNIE_ZONE("Draw the windows");
        DrawControlsWindow(windows_.controls);
        DrawAboutWindow(windows_.about, device_ != nullptr ? device_->Info().name : "");
        DrawFileWindow();
    }
    TakeShortcuts();
    {
        ERNIE_ZONE("Move the camera");
        PullActiveCamera();
        Navigate();
    }
    if (!overlaysHidden_) {
        ERNIE_ZONE("Draw over the viewport");
        DrawToolbar();
        DrawTimeline();
        DrawHud();
    }
    Apply();
}

void Shell::DrawDockSpace() {
    if (!layoutBuilt_) {
        BuildLayout();
        layoutBuilt_ = true;
    }
    // The viewport shows through the central node.
    ImGui::DockSpaceOverViewport(kDockSpace, ImGui::GetMainViewport(),
                                 ImGuiDockNodeFlags_PassthruCentralNode);
}

void Shell::BuildLayout() {
    ImGui::DockBuilderRemoveNode(kDockSpace);
    ImGui::DockBuilderAddNode(kDockSpace, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(kDockSpace, ImGui::GetMainViewport()->WorkSize);
    docks_.clear();
    // Hidden panels take no room; they dock where they ask once shown.
    const auto wants = [&](Dock dock) {
        return std::ranges::any_of(panels_, [dock](const std::unique_ptr<Panel>& panel) {
            return panel->Visible() && panel->DefaultDock() == dock;
        });
    };
    const auto split = [](ImGuiID& rest, ImGuiDir direction, float share) {
        return ImGui::DockBuilderSplitNode(rest, direction, std::clamp(share, 0.02f, 0.98f),
                                           nullptr, &rest);
    };
    ImGuiID centre = kDockSpace;
    if (wants(Dock::Left)) {
        docks_[Dock::Left] = split(centre, ImGuiDir_Left, sizes_.left);
    }
    // The right column: `Right` above `RightLower`, either taking all of it alone.
    const bool upper = wants(Dock::Right);
    const bool lower = wants(Dock::RightLower);
    if (upper || lower) {
        ImGuiID column = split(centre, ImGuiDir_Right, sizes_.right);
        if (upper && lower) {
            docks_[Dock::RightLower] = split(column, ImGuiDir_Down, sizes_.rightLower);
        }
        docks_[upper ? Dock::Right : Dock::RightLower] = column;
    }
    if (wants(Dock::Bottom)) {
        docks_[Dock::Bottom] = split(centre, ImGuiDir_Down, sizes_.bottom);
    }
    for (const std::unique_ptr<Panel>& panel : panels_) {
        const auto node = docks_.find(panel->DefaultDock());
        if (panel->Visible() && node != docks_.end()) {
            ImGui::DockBuilderDockWindow(panel->Title().c_str(), node->second);
        }
    }
    ImGui::DockBuilderFinish(kDockSpace);
    // A dock opens on the last window docked into it; the bottom one opens on the Log instead.
    const Panel* log = FindPanel("Log");
    selectLog_ = log != nullptr && log->Visible() && log->DefaultDock() == Dock::Bottom &&
                 docks_.contains(Dock::Bottom);
}

void Shell::SelectLogTab() {
    // A dock shows the focused window's tab, and ImGui focuses each window as it is made: once
    // every panel has been, the Log takes the focus.
    if (selectLog_ && ImGui::FindWindowByName("Log") != nullptr) {
        ImGui::SetWindowFocus("Log");
        selectLog_ = false;
    }
}

void Shell::DrawPanels() {
    ERNIE_ZONE("Draw the panels");
    for (const std::unique_ptr<Panel>& panel : panels_) {
        ERNIE_ZONE("Draw a panel");
        ERNIE_ZONE_TEXT(panel->Title());
        if (!panel->Visible()) {
            panel->Undrawn();
            continue;
        }
        if (const auto node = docks_.find(panel->DefaultDock()); node != docks_.end()) {
            ImGui::SetNextWindowDockID(node->second, ImGuiCond_FirstUseEver);
        }
        bool keep = true;
        if (ImGui::Begin(panel->Title().c_str(), &keep)) {
            panel->Draw();
        } else {
            panel->Undrawn();
        }
        ImGui::End();
        panel->SetVisible(keep);
    }
    SelectLogTab();
}

void Shell::Navigate() {
    const ImGuiIO& io = ImGui::GetIO();
    const auto [origin, size] = Viewport();
    // Over the viewport and no window: the dock space lets the mouse through its central node.
    const bool hovered = !io.WantCaptureMouse && io.MousePos.x >= origin.x &&
                         io.MousePos.y >= origin.y && io.MousePos.x < origin.x + size.x &&
                         io.MousePos.y < origin.y + size.y;
    const auto any = [](auto&& test) { return std::ranges::any_of(kMouseButtons, test); };
    // A left click that does not move picks what is under it.
    if (dragging_ && ImGui::IsMouseReleased(ImGuiMouseButton_Left) &&
        !ImGui::IsMouseDragPastThreshold(ImGuiMouseButton_Left)) {
        const glm::vec2 at(io.MousePos.x - origin.x, io.MousePos.y - origin.y);
        pick_ = Pick{.at = at / glm::vec2(size.x, size.y), .toggle = io.KeyCtrl};
    }
    if (!any([](ImGuiMouseButton button) { return ImGui::IsMouseDown(button); })) {
        dragging_ = false;
    } else if (hovered &&
               any([](ImGuiMouseButton button) { return ImGui::IsMouseClicked(button); })) {
        dragging_ = true;
    }
    const double dx = io.MouseDelta.x;
    const double dy = io.MouseDelta.y;
    if (dragging_ && (dx != 0.0 || dy != 0.0)) {
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            camera_.Orbit(-dx * kOrbitRate, dy * kOrbitRate);
        } else if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
            camera_.Pan(dx, dy, size.y);
        } else {
            camera_.PanVertical(dx, dy, size.y);
        }
    }
    if (hovered && io.MouseWheel != 0.0f) {
        camera_.Zoom(io.MouseWheel);
    }
}

bool Shell::BeginPills(const char* id, ImVec2 at, ImVec2 pivot) {
    ImGui::SetNextWindowPos(at, ImGuiCond_Always, pivot);
    constexpr ImGuiWindowFlags kBar =
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground |
        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDocking |
        ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoNav;
    if (!ImGui::Begin(id, nullptr, kBar)) {
        ImGui::End();
        return false;
    }
    // Kit's pills: the panels' grey, with a thin darker edge.
    ImGui::PushStyleColor(ImGuiCol_FrameBg, ToImVec4(theme_.overlay));
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ToImVec4(theme_.fieldHovered));
    ImGui::PushStyleColor(ImGuiCol_Button, ToImVec4(theme_.overlay));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ToImVec4(theme_.fieldHovered));
    ImGui::PushStyleColor(ImGuiCol_PopupBg, ToImVec4(glm::vec4(glm::vec3(theme_.overlay), 1.0f)));
    ImGui::PushStyleColor(ImGuiCol_Border, ToImVec4(theme_.overlayBorder));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, theme_.pillRounding);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(9.0f, 4.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(5.0f, 4.0f));
    return true;
}

void Shell::EndPills() {
    ImGui::PopStyleVar(4);
    ImGui::PopStyleColor(6);
    ImGui::End();
}

void Shell::Pill(const char* id, const std::string& label, const char* tip,
                 void (Shell::*items)()) {
    constexpr ImGuiComboFlags kPill = ImGuiComboFlags_WidthFitPreview |
                                      ImGuiComboFlags_NoArrowButton | ImGuiComboFlags_HeightLargest;
    ImGui::SetNextWindowSizeConstraints(ImVec2(16.0f * ImGui::GetFontSize(), 0.0f),
                                        ImVec2(FLT_MAX, FLT_MAX));
    if (ImGui::BeginCombo(id, label.c_str(), kPill)) {
        (this->*items)();
        ImGui::EndCombo();
    }
    ImGui::SetItemTooltip("%s", tip);
}

void Shell::DrawToolbar() {
    if (!display_.toolbar) {
        return;
    }
    const auto [origin, size] = Viewport();
    const float pad = 8.0f;
    if (BeginPills("##viewport toolbar", ImVec2(origin.x + pad, origin.y + pad), {0.0f, 0.0f})) {
        // The settings pill, drawn with Kit's three bars where its spaces leave room.
        const ImVec2 corner = ImGui::GetCursorScreenPos();
        Pill("##settings", "   ", "Viewport settings", &Shell::DrawSettingsMenu);
        const float height = ImGui::GetFrameHeight();
        const float half = 0.22f * height;
        const ImVec2 centre(corner.x + 0.5f * (ImGui::CalcTextSize("   ").x +
                                               2.0f * ImGui::GetStyle().FramePadding.x),
                            corner.y + 0.5f * height);
        for (const float dy : {-0.6f, 0.0f, 0.6f}) {
            ImGui::GetWindowDrawList()->AddLine(ImVec2(centre.x - half, centre.y + dy * half),
                                                ImVec2(centre.x + half, centre.y + dy * half),
                                                ImGui::GetColorU32(ImGuiCol_Text), 1.5f);
        }
        ImGui::SameLine();
        Pill("##show", icons::kEye, "What the viewport shows", &Shell::DrawShowMenu);
        ImGui::SameLine();
        const std::string camera =
            activeCamera_.IsEmpty() ? "Perspective" : std::string(activeCamera_.Name());
        Pill("##camera", std::string(icons::kCamera) + "  " + camera, "Camera and views",
             &Shell::DrawCameraMenu);
        EndPills();
    }
    if (BeginPills("##viewport toolbar right", ImVec2(origin.x + size.x - pad, origin.y + pad),
                   {1.0f, 0.0f})) {
        if (ecoPaused_) {
            ImGui::AlignTextToFramePadding();
            ImGui::TextColored(ImVec4(0.30f, 0.78f, 0.35f, 1.0f), "%s  Eco", icons::kLeaf);
            ImGui::SetItemTooltip("Eco mode: rendering paused until something changes");
            ImGui::SameLine();
        }
        if (ImGui::Button((std::string(icons::kDownload) + "  Capture##screenshot").c_str())) {
            RequestScreenshot();
        }
        ImGui::SetItemTooltip("Save a screenshot of the viewport (F10)");
        ImGui::SameLine();
        Pill("##lighting", std::string(icons::kSun) + "  " + LightingName(display_.lighting),
             "Lighting", &Shell::DrawLightingMenu);
        EndPills();
    }
}

void Shell::DrawLightingMenu() {
    // Kit's order. Minimal lights from the eye; scene lights render with the lighting to come.
    for (const Lighting lighting : {Lighting::Camera, Lighting::Stage, Lighting::Default}) {
        const bool available = lighting == Lighting::Camera;
        ImGui::BeginDisabled(!available);
        bool chosen = display_.lighting == lighting;
        MenuCheck(LightingName(lighting), chosen);
        ImGui::EndDisabled();
        if (!available) {
            ImGui::SetItemTooltip("Scene lights render once the Realtime tier lights scenes");
        } else if (chosen) {
            display_.lighting = lighting;
        }
    }
}

void Shell::DrawSettingsMenu() {
    MenuSection("Navigation");
    float fov = static_cast<float>(glm::degrees(camera_.verticalFov));
    ImGui::SetNextItemWidth(10.0f * ImGui::GetFontSize());
    if (ImGui::SliderFloat("Field of view", &fov, 10.0f, 120.0f, "%.0f°")) {
        camera_.verticalFov = glm::radians(static_cast<double>(fov));
    }
    ImGui::Separator();
    MenuSection("Eco Mode");
    ImGui::PushItemFlag(ImGuiItemFlags_AutoClosePopups, false);
    MenuCheck("Pause Rendering When Still", display_.eco);
    ImGui::PopItemFlag();
    ImGui::BeginDisabled(!display_.eco);
    ImGui::SetNextItemWidth(10.0f * ImGui::GetFontSize());
    ImGui::SliderInt("Frames without change", &display_.ecoFrames, 1, 10'000, "%d",
                     ImGuiSliderFlags_Logarithmic);
    ImGui::EndDisabled();
    ImGui::Separator();
    if (MenuAction("Reset Camera")) {
        const double fovKept = camera_.verticalFov;
        camera_ = OrbitCamera{.verticalFov = fovKept};
        Frame();
    }
}

void Shell::DrawShowMenu() {
    // Toggles keep the menu open, so several switch at once.
    ImGui::PushItemFlag(ImGuiItemFlags_AutoClosePopups, false);
    if (ImGui::BeginMenu("    Heads Up Display")) {
        MenuCheck("Show", display_.hud);
        ImGui::Separator();
        ImGui::BeginDisabled(!display_.hud);
        MenuCheck("FPS", display_.hudFps);
        MenuCheck("Device Memory", display_.hudDeviceMemory);
        MenuCheck("Host Memory", display_.hudHostMemory);
        MenuCheck("Resolution", display_.hudResolution);
        MenuCheck("Camera", display_.hudCamera);
        ImGui::EndDisabled();
        ImGui::EndMenu();
    }
    ImGui::Separator();
    MenuCheck("Toolbar", display_.toolbar);
    MenuCheck("Timeline", display_.timeline);
    ImGui::PopItemFlag();
}

void Shell::DrawCameraMenu() {
    const std::vector<Path> cameras = scene_.Prims(ernie::Camera::kName);
    if (!cameras.empty()) {
        MenuSection("Cameras");
        for (const Path& camera : cameras) {
            bool active = camera == activeCamera_;
            MenuCheck(std::string(camera.Name()).c_str(), active);
            ImGui::SetItemTooltip("%s", camera.String().c_str());
            if (active && camera != activeCamera_) {
                Attempt("Looking through a camera", [&] { LookThrough(camera); });
            }
        }
        ImGui::Separator();
    }
    MenuSection("Views");
    // Kit's views: the orbit from the side, or square on from the top, front or right.
    struct View {
        const char* name;
        double yaw;
        double pitch;
    };
    constexpr std::array<View, 4> kViews{{{"Perspective", 0.6, 0.45},
                                          {"Top", 0.0, OrbitCamera::kMaxPitch},
                                          {"Front", 0.0, 0.0},
                                          {"Right", 1.5707963267948966, 0.0}}};
    for (const View& view : kViews) {
        if (MenuAction(view.name)) {
            // The viewer's own camera, so a scene camera stays where it was.
            activeCamera_ = Path();
            camera_.yaw = view.yaw;
            camera_.pitch = view.pitch;
        }
    }
    ImGui::Separator();
    choices_.frame |= MenuAction(selection_.Empty() ? "Frame Scene" : "Frame Selection", "F");
    if (MenuAction("Create from View")) {
        choices_.create = ernie::Camera::kName;
    }
}

void Shell::RefreshHud() {
    const double now = ImGui::GetTime();
    if (hudRefreshed_ >= 0.0 && now - hudRefreshed_ < kHudRefresh) {
        return;
    }
    hudRefreshed_ = now;
    hudLines_.clear();
    std::array<char, 192> line{};
    if (display_.hudFps) {
        const double fps = std::max(static_cast<double>(ImGui::GetIO().Framerate), 1e-3);
        std::snprintf(line.data(), line.size(), "FPS: %.1f, Frame time: %.2f ms", fps,
                      1000.0 / fps);
        hudLines_.emplace_back(line.data());
    }
    if (display_.hudDeviceMemory && device_ != nullptr) {
        // The device-local heaps: what the process uses where the driver reports budgets.
        const VkPhysicalDeviceMemoryProperties* memory = nullptr;
        vmaGetMemoryProperties(device_->Allocator(), &memory);
        std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> budgets{};
        vmaGetHeapBudgets(device_->Allocator(), budgets.data());
        std::uint64_t used = 0;
        std::uint64_t total = 0;
        for (std::uint32_t heap = 0; heap < memory->memoryHeapCount; ++heap) {
            if ((memory->memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
                used += budgets[heap].usage;
                total += memory->memoryHeaps[heap].size;
            }
        }
        hudLines_.push_back("Device Memory: " + FormatBytes(used) + " / " + FormatBytes(total));
    }
    if (display_.hudHostMemory) {
        const std::optional<std::uint64_t> resident = ReadResidentBytes();
        hudLines_.push_back("Host Memory: " + (resident ? FormatBytes(*resident) : "unknown"));
    }
    if (display_.hudResolution) {
        const auto [origin, size] = Viewport();
        const ImVec2 scale = ImGui::GetIO().DisplayFramebufferScale;
        std::snprintf(line.data(), line.size(), "Resolution: %.0f x %.0f",
                      static_cast<double>(size.x * scale.x), static_cast<double>(size.y * scale.y));
        hudLines_.emplace_back(line.data());
    }
    if (display_.hudCamera) {
        const glm::dvec3 eye = camera_.Eye();
        std::snprintf(line.data(), line.size(), "Eye: %.3g, %.3g, %.3g m", eye.x, eye.y, eye.z);
        hudLines_.emplace_back(line.data());
        const glm::dvec3& target = camera_.target;
        std::snprintf(line.data(), line.size(), "Target: %.3g, %.3g, %.3g m", target.x, target.y,
                      target.z);
        hudLines_.emplace_back(line.data());
    }
}

void Shell::DrawTimeline() {
    if (!display_.timeline) {
        return;
    }
    // Under the viewport's image, in the panels' grey, so the docks around resize freely.
    const auto [origin, size] = Viewport();
    ImGui::SetNextWindowPos(ImVec2(origin.x, origin.y + size.y));
    ImGui::SetNextWindowSize(ImVec2(size.x, TimelineHeight()));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ToImVec4(theme_.panel));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    constexpr ImGuiWindowFlags kStrip =
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav |
        ImGuiWindowFlags_NoScrollWithMouse;
    if (ImGui::Begin("##timeline", nullptr, kStrip)) {
        const float unit = ImGui::GetFontSize();
        const float field = 4.0f * unit;
        const float spacing = ImGui::GetStyle().ItemSpacing.x;
        // An icon button: the widest icon and the frame's padding.
        float icon = 0.0f;
        for (const char* glyph : {icons::kSkipBack, icons::kStepBack, icons::kPlay, icons::kSquare,
                                  icons::kRepeat, icons::kSettings}) {
            icon = std::max(icon, ImGui::CalcTextSize(glyph).x);
        }
        const float button = icon + 2.0f * ImGui::GetStyle().FramePadding.x;
        const auto timeField = [&](const char* id, double value, const char* tip) {
            ImGui::SetNextItemWidth(field);
            ImGui::InputDouble(id, &value, 0.0, 0.0, "%g");
            ImGui::SetItemTooltip("%s", tip);
            return ImGui::IsItemDeactivatedAfterEdit() ? std::optional(value) : std::nullopt;
        };
        if (const std::optional<double> start =
                timeField("##start", timeline_.Start(), "Start time code")) {
            timeline_.SetRange(*start, timeline_.End());
        }
        ImGui::SameLine();
        const float right = 2.0f * field + 8.0f * button + 10.0f * spacing;
        DrawRuler(std::max(ImGui::GetContentRegionAvail().x - right, 4.0f * unit));
        ImGui::SameLine();
        if (const std::optional<double> end =
                timeField("##end", timeline_.End(), "End time code")) {
            timeline_.SetRange(timeline_.Start(), *end);
        }
        ImGui::SameLine();
        double now = scene_.Time();
        ImGui::SetNextItemWidth(field);
        if (ImGui::InputDouble("##now", &now, 0.0, 0.0, "%g")) {
            timeline_.Seek(scene_, now);
        }
        ImGui::SetItemTooltip("Current time code");
        ImGui::SameLine();
        DrawTransport();
    }
    ImGui::End();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();
}

void Shell::DrawRuler(float width) {
    const double start = timeline_.Start();
    const double end = timeline_.End();
    const float height = ImGui::GetFrameHeight();
    const ImVec2 low = ImGui::GetCursorScreenPos();
    const ImVec2 high(low.x + width, low.y + height);
    ImGui::InvisibleButton("##ruler", ImVec2(width, height));
    const double perCode = static_cast<double>(width) / std::max(end - start, 1e-9);
    if (ImGui::IsItemActive()) {
        timeline_.Pause();
        const double at = start + static_cast<double>(ImGui::GetIO().MousePos.x - low.x) / perCode;
        timeline_.Seek(scene_, std::clamp(start + std::round(at - start), start, end));
    }
    ImDrawList* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(low, high, ImGui::GetColorU32(ToImVec4(theme_.field)),
                        ImGui::GetStyle().FrameRounding);
    // Labelled ticks at least five characters apart, at 1, 2 or 5 times a power of ten, and
    // unlabelled fifths between them where they fit.
    const double least = (ImGui::CalcTextSize("00000").x + 8.0f) / perCode;
    const double decade = std::pow(10.0, std::floor(std::log10(least)));
    double major = 10.0 * decade;
    for (const double factor : {1.0, 2.0, 5.0}) {
        if (factor * decade >= least) {
            major = factor * decade;
            break;
        }
    }
    const double minor = major / 5.0;
    const bool minors = minor * perCode >= 5.0;
    const ImU32 tick = ImGui::GetColorU32(ToImVec4(theme_.textMuted));
    for (double t = std::ceil(start / minor) * minor; t <= end; t += minor) {
        const float x = low.x + static_cast<float>((t - start) * perCode);
        const bool labelled = std::fabs(t / major - std::round(t / major)) < 1e-6;
        if (!labelled && !minors) {
            continue;
        }
        draw->AddLine(ImVec2(x, high.y - (labelled ? 0.45f : 0.2f) * height), ImVec2(x, high.y),
                      tick);
        if (labelled) {
            std::array<char, 32> label{};
            std::snprintf(label.data(), label.size(), "%g", t);
            draw->AddText(ImVec2(x + 2.0f, low.y), tick, label.data());
        }
    }
    // The current time code's line, with a notch at its top.
    const double now = scene_.Time();
    if (now >= start && now <= end) {
        const float x = low.x + static_cast<float>((now - start) * perCode);
        const ImU32 accent = ImGui::GetColorU32(ToImVec4(theme_.accent));
        draw->AddLine(ImVec2(x, low.y), ImVec2(x, high.y), accent, 2.0f);
        const float half = 0.2f * height;
        draw->AddTriangleFilled(ImVec2(x - half, low.y), ImVec2(x + half, low.y),
                                ImVec2(x, low.y + half), accent);
    }
    if (ImGui::IsItemHovered() && !ImGui::IsItemActive()) {
        ImGui::SetTooltip("Click or drag to scrub");
    }
}

void Shell::DrawTransport() {
    const auto button = [](const char* glyph, const char* id, const char* tip) {
        const bool clicked = ImGui::Button((std::string(glyph) + id).c_str());
        ImGui::SetItemTooltip("%s", tip);
        ImGui::SameLine();
        return clicked;
    };
    if (button(icons::kSkipBack, "##first", "First frame")) {
        timeline_.Pause();
        timeline_.Seek(scene_, timeline_.Start());
    }
    if (button(icons::kStepBack, "##back", "Previous frame")) {
        timeline_.StepFrames(scene_, -1);
    }
    if (timeline_.Playing()) {
        if (button(icons::kPause, "##play", "Pause (Space)")) {
            timeline_.Pause();
        }
    } else if (button(icons::kPlay, "##play", "Play (Space)")) {
        timeline_.Play();
    }
    if (button(icons::kSquare, "##stop", "Stop: back to the start")) {
        timeline_.Stop(scene_);
    }
    if (button(icons::kStepForward, "##forward", "Next frame")) {
        timeline_.StepFrames(scene_, 1);
    }
    if (button(icons::kSkipForward, "##last", "Last frame")) {
        timeline_.Pause();
        timeline_.Seek(scene_, timeline_.End());
    }
    // Loop lit while on.
    const bool looping = timeline_.looping;
    if (looping) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
    }
    const bool loop = ImGui::Button((std::string(icons::kRepeat) + "##loop").c_str());
    if (looping) {
        ImGui::PopStyleColor();
    }
    ImGui::SetItemTooltip(looping ? "Looping: on" : "Looping: off");
    if (loop) {
        timeline_.looping = !looping;
    }
    ImGui::SameLine();
    if (ImGui::Button((std::string(icons::kSettings) + "##settings").c_str())) {
        ImGui::OpenPopup("##timeline settings");
    }
    ImGui::SetItemTooltip("Timeline settings");
    ImGui::SetNextWindowSizeConstraints(ImVec2(16.0f * ImGui::GetFontSize(), 0.0f),
                                        ImVec2(FLT_MAX, FLT_MAX));
    if (ImGui::BeginPopup("##timeline settings")) {
        DrawTimelineSettings();
        ImGui::EndPopup();
    }
}

void Shell::DrawTimelineSettings() {
    // Time codes and frames a second alike, so a time code is a frame.
    MenuSection("Frame Rate");
    for (const double fps : {24.0, 25.0, 30.0, 48.0, 50.0, 60.0, 120.0}) {
        std::array<char, 32> label{};
        std::snprintf(label.data(), label.size(), "%g fps", fps);
        bool chosen = timeline_.FramesPerSecond() == fps;
        MenuCheck(label.data(), chosen);
        if (chosen && timeline_.FramesPerSecond() != fps) {
            timeline_.SetFramesPerSecond(fps);
        }
    }
    ImGui::Separator();
    MenuSection("Playback");
    MenuCheck("Play Every Frame", timeline_.everyFrame);
    ImGui::SetItemTooltip("A frame each step however long it takes, rather than real time");
    MenuCheck("Loop", timeline_.looping);
}

void Shell::DrawHud() {
    if (!display_.hud) {
        return;
    }
    RefreshHud();
    if (hudLines_.empty()) {
        return;
    }
    const auto [origin, size] = Viewport();
    const float pad = 6.0f;
    const float lineHeight = ImGui::GetTextLineHeight();
    float width = 0.0f;
    for (const std::string& text : hudLines_) {
        width = std::max(width, ImGui::CalcTextSize(text.c_str()).x);
    }
    // Below the lighting pill, as Kit's.
    const float below = display_.toolbar ? ImGui::GetFrameHeight() + 14.0f : 8.0f;
    const ImVec2 topRight(origin.x + size.x - 8.0f, origin.y + below);
    const ImVec2 corner(topRight.x - width - 2.0f * pad, topRight.y);
    // Over the viewport's image, which the background list holds, and under every window.
    ImDrawList* draw = ImGui::GetForegroundDrawList(ImGui::GetMainViewport());
    draw->PushClipRect(origin, ImVec2(origin.x + size.x, origin.y + size.y));
    ImVec4 box = ToImVec4(theme_.overlay);
    box.w *= 0.85f;
    draw->AddRectFilled(
        corner,
        ImVec2(topRight.x,
               corner.y + lineHeight * static_cast<float>(hudLines_.size()) + 2.0f * pad),
        ImGui::GetColorU32(box), 4.0f);
    // Right-aligned, as Kit's: the label muted, its value bright.
    float y = corner.y + pad;
    for (const std::string& text : hudLines_) {
        const std::size_t colon = text.find(": ");
        const std::string label = text.substr(0, colon == std::string::npos ? 0 : colon + 2);
        const float x = topRight.x - pad - ImGui::CalcTextSize(text.c_str()).x;
        draw->AddText(ImVec2(x, y), ImGui::GetColorU32(ToImVec4(theme_.textMuted)), label.c_str());
        draw->AddText(ImVec2(x + ImGui::CalcTextSize(label.c_str()).x, y),
                      ImGui::GetColorU32(ToImVec4(theme_.text)), text.c_str() + label.size());
        y += lineHeight;
    }
    draw->PopClipRect();
}

void Shell::Frame() {
    const std::vector<Path> framed =
        selection_.Empty() ? scene_.Children(Path::Root()) : selection_.Paths();
    camera_.Frame(WorldBounds(scene_, framed));
}

template <typename Edit> void Shell::Attempt(const char* what, Edit&& edit) {
    try {
        edit();
    } catch (const std::exception& error) {
        log_.Error(std::string(what) + ": " + error.what());
    }
}

void Shell::DrawMenuBar() {
    if (!ImGui::BeginMainMenuBar()) {
        return;
    }
    const bool selected = !selection_.Empty();
    Menu("File", [&] {
        windows_.open |= MenuAction("Open Scene...", "Ctrl+O");
        windows_.save |= MenuAction("Save Scene As...", "Ctrl+Shift+S");
        ImGui::Separator();
        choices_.quit |= MenuAction("Quit", "Ctrl+Q");
    });
    Menu("Edit", [&] {
        const std::string undo = history_.CanUndo() ? "Undo " + history_.UndoLabel() : "Undo";
        const std::string redo = history_.CanRedo() ? "Redo " + history_.RedoLabel() : "Redo";
        choices_.undo |= MenuAction(undo.c_str(), "Ctrl+Z", history_.CanUndo());
        choices_.redo |= MenuAction(redo.c_str(), "Ctrl+Shift+Z", history_.CanRedo());
        ImGui::Separator();
        choices_.duplicate |= MenuAction("Duplicate", "Ctrl+D", selected);
        choices_.remove |= MenuAction("Delete", "Del", selected);
        ImGui::Separator();
        choices_.frame |= MenuAction(selected ? "Frame Selection" : "Frame Scene", "F");
        choices_.clearSelection |= MenuAction("Clear Selection", nullptr, selected);
    });
    Menu("Create", [&] { DrawCreateMenu(); });
    Menu("Window", [&] {
        for (const std::unique_ptr<Panel>& panel : panels_) {
            bool visible = panel->Visible();
            MenuCheck(panel->Title().c_str(), visible);
            panel->SetVisible(visible);
        }
        ImGui::Separator();
        choices_.resetLayout |= MenuAction("Reset Layout");
    });
    Menu("Help", [&] { DrawHelpMenu(); });
    ImGui::EndMainMenuBar();
}

void Shell::DrawCreateMenu() {
    std::string_view group;
    for (const CreatableType& creatable : CreatableTypes()) {
        if (group != creatable.group) {
            if (!group.empty()) {
                ImGui::Separator();
            }
            group = creatable.group;
            MenuSection(creatable.group);
        }
        if (MenuAction(creatable.label)) {
            choices_.create = creatable.type;
        }
    }
}

void Shell::DrawHelpMenu() {
    const std::string docs = kDocumentation;
    const std::array<std::pair<const char*, std::string>, 5> links{{
        {"Documentation", docs},
        {"Viewer Guide", docs + "getting-started/viewer/"},
        {"Demos", docs + "getting-started/demos/"},
        {"Python API", docs + "api/python/scene/"},
        {"C++ API", docs + "api/cpp/"},
    }};
    const auto open = [&](const std::string& url) {
        if (!OpenExternally(url)) {
            log_.Error("Could not open " + url);
        }
    };
    for (const auto& [label, url] : links) {
        if (MenuAction(label)) {
            open(url);
        }
    }
    ImGui::Separator();
    if (MenuAction("Report an Issue")) {
        open(std::string(kRepository) + "/issues");
    }
    if (MenuAction("Source Code")) {
        open(kRepository);
    }
    ImGui::Separator();
    MenuCheck("Keyboard and Mouse", windows_.controls);
    MenuCheck("About Ernie", windows_.about);
}

void Shell::DrawFileWindow() {
    if (!windows_.open && !windows_.save) {
        return;
    }
    const char* title = windows_.open ? "Open Scene" : "Save Scene As";
    bool shown = true;
    ImGui::SetNextWindowSize(ImVec2(32.0f * ImGui::GetFontSize(), 0.0f), ImGuiCond_Appearing);
    if (ImGui::Begin(title, &shown, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoDocking)) {
        ImGui::TextUnformatted("Scene file (.json):");
        ImGui::SetNextItemWidth(-1.0f);
        if (ImGui::IsWindowAppearing()) {
            ImGui::SetKeyboardFocusHere();
        }
        const bool entered =
            ImGui::InputText("##path", &filePath_, ImGuiInputTextFlags_EnterReturnsTrue);
        if (ImGui::Button(windows_.open ? "Open" : "Save") || entered) {
            if (windows_.open) {
                Attempt("Opening a scene", [&] { OpenScene(filePath_); });
            } else {
                Attempt("Saving the scene", [&] { SaveScene(filePath_); });
            }
            shown = false;
        }
        ImGui::SameLine();
        shown &= !ImGui::Button("Cancel");
    }
    ImGui::End();
    if (!shown) {
        windows_.open = windows_.save = false;
    }
}

void Shell::TakeShortcuts() {
    const ImGuiIO& io = ImGui::GetIO();
    if (io.WantTextInput) {
        return;
    }
    const auto pressed = [](ImGuiKey key) { return ImGui::IsKeyPressed(key, false); };
    const bool plain = !io.KeyCtrl && !io.KeyAlt;
    choices_.quit |= io.KeyCtrl && pressed(ImGuiKey_Q);
    choices_.frame |= plain && pressed(ImGuiKey_F);
    choices_.undo |= io.KeyCtrl && !io.KeyShift && pressed(ImGuiKey_Z);
    choices_.redo |= io.KeyCtrl && ((io.KeyShift && pressed(ImGuiKey_Z)) || pressed(ImGuiKey_Y));
    choices_.duplicate |= io.KeyCtrl && pressed(ImGuiKey_D);
    choices_.remove |= plain && pressed(ImGuiKey_Delete);
    overlaysHidden_ ^= plain && pressed(ImGuiKey_H);
    screenshot_ |= pressed(ImGuiKey_F10);
    if (plain && pressed(ImGuiKey_Space)) {
        timeline_.Playing() ? timeline_.Pause() : timeline_.Play();
    }
    windows_.open |= io.KeyCtrl && pressed(ImGuiKey_O);
    windows_.save |= io.KeyCtrl && io.KeyShift && pressed(ImGuiKey_S);
}

void Shell::Apply() {
    open_ = open_ && !choices_.quit;
    layoutBuilt_ = layoutBuilt_ && !choices_.resetLayout;
    if (choices_.clearSelection) {
        selection_.Clear();
    }
    if (choices_.undo) {
        Attempt("Undoing", [&] { history_.Undo(scene_); });
    }
    if (choices_.redo) {
        Attempt("Redoing", [&] { history_.Redo(scene_); });
    }
    if (choices_.duplicate) {
        Attempt("Duplicating", [&] { DuplicateSelection(); });
    }
    if (choices_.remove) {
        Attempt("Deleting", [&] { DeleteSelection(); });
    }
    if (!choices_.create.empty()) {
        Attempt("Creating", [&] { Create(choices_.create); });
    }
    // Undone prims leave the selection.
    std::vector<Path> kept;
    for (const Path& path : selection_.Paths()) {
        if (scene_.Find(path)) {
            kept.push_back(path);
        }
    }
    if (kept.size() != selection_.Paths().size()) {
        selection_.Clear();
        for (const Path& path : kept) {
            selection_.Toggle(path);
        }
    }
    if (choices_.frame) {
        Frame();
    }
    PushActiveCamera();
}

void Shell::LookThrough(const Path& camera) {
    if (camera.IsEmpty()) {
        activeCamera_ = Path();
        return;
    }
    const std::optional<PrimId> prim = scene_.Find(camera);
    if (!prim || scene_.TypeName(*prim) != ernie::Camera::kName) {
        throw std::invalid_argument("'" + camera.String() + "' is no Camera");
    }
    const ernie::Camera lens(scene_, camera);
    camera_.verticalFov = 2.0 * std::atan(0.5 * lens.GetVerticalAperture() / lens.GetFocalLength());
    activeCamera_ = camera;
    activeWorld_ = scene_.WorldTransform(*prim);
    camera_.LookFrom(activeWorld_);
}

void Shell::PullActiveCamera() {
    if (activeCamera_.IsEmpty()) {
        return;
    }
    const std::optional<PrimId> prim = scene_.Find(activeCamera_);
    if (!prim) {
        // Removed: back to the viewer's own camera, where it was.
        activeCamera_ = Path();
        return;
    }
    if (const glm::dmat4 world = scene_.WorldTransform(*prim); world != activeWorld_) {
        activeWorld_ = world;
        camera_.LookFrom(world);
    }
}

void Shell::PushActiveCamera() {
    if (activeCamera_.IsEmpty() || !scene_.Has(activeCamera_)) {
        return;
    }
    const glm::dmat4 world = camera_.Transform();
    if (world == activeWorld_) {
        return;
    }
    const Path parent = activeCamera_.Parent();
    const std::optional<PrimId> above = parent.IsRoot() ? std::nullopt : scene_.Find(parent);
    const glm::dmat4 parentWorld = above ? scene_.WorldTransform(*above) : glm::dmat4(1.0);
    Attempt("Moving the camera", [&] {
        scene_.Set(activeCamera_, tokens::kTransform, glm::inverse(parentWorld) * world);
        activeWorld_ = world;
    });
}

Path Shell::Create(std::string_view type) {
    const CreatableType* creatable = FindCreatable(type);
    if (creatable == nullptr) {
        throw std::invalid_argument("The viewer cannot create a '" + std::string(type) + "'");
    }
    Path parent(creatable->parent);
    const std::optional<PrimId> selected =
        selection_.Empty() ? std::nullopt : scene_.Find(selection_.Paths().front());
    if (selected && IsGroup(scene_.TypeName(*selected))) {
        parent = selection_.Paths().front();
    }
    const Path path = FreeChildPath(scene_, parent, creatable->name);
    const std::optional<PrimId> parentPrim = scene_.Find(parent);
    // Authored in a scratch scene, then copied in as one step with any missing parents.
    Scene scratch;
    const CreateContext context{.target = camera_.target,
                                .scale = std::max(0.05 * camera_.distance, 1e-3),
                                .parentWorld = parentPrim ? scene_.WorldTransform(*parentPrim)
                                                          : glm::dmat4(1.0),
                                .view = camera_.Transform(),
                                .verticalFov = camera_.verticalFov,
                                .camera = SettingsCamera(scene_)};
    AuthorDefaultPrim(scratch, path, type, context);
    // Missing ancestors become Xforms, outermost first, in the same step.
    std::vector<Path> missing;
    for (Path at = parent; !at.IsRoot() && !scene_.Find(at); at = at.Parent()) {
        missing.insert(missing.begin(), at);
    }
    history_.BeginGroup();
    for (const Path& at : missing) {
        history_.Define(scene_, at, Xform::kName);
    }
    history_.Add(scene_, scratch, path);
    history_.EndGroup();
    selection_.Set(path);
    log_.Info("Created " + path.String());
    return path;
}

void Shell::DuplicateSelection() {
    std::vector<Path> copies;
    history_.BeginGroup();
    for (const Path& path : selection_.Paths()) {
        if (!scene_.Find(path)) {
            continue;
        }
        const Path copy = FreeChildPath(scene_, path.Parent(), path.Name());
        history_.Copy(scene_, path, copy);
        copies.push_back(copy);
    }
    history_.EndGroup();
    selection_.Clear();
    for (const Path& copy : copies) {
        selection_.Toggle(copy);
    }
}

void Shell::DeleteSelection() {
    history_.BeginGroup();
    for (const Path& path : selection_.Paths()) {
        history_.Remove(scene_, path);
    }
    history_.EndGroup();
    selection_.Clear();
}

void Shell::OpenScene(const std::filesystem::path& path) {
    // Read once aside, so a bad file leaves the scene as it was.
    Scene checked;
    LoadSceneFile(checked, path);
    for (const Path& root : scene_.Children(Path::Root())) {
        scene_.Remove(root);
    }
    LoadSceneFile(scene_, path);
    history_.Reset();
    selection_.Clear();
    log_.Info("Opened " + path.string());
    Frame();
}

std::optional<Shell::Pick> Shell::TakePick() {
    return std::exchange(pick_, std::nullopt);
}

void Shell::Picked(const std::optional<Path>& path, bool toggle) {
    if (path && toggle) {
        selection_.Toggle(*path);
    } else if (path) {
        selection_.Set(*path);
    } else if (!toggle) {
        selection_.Clear();
    }
}

void Shell::SaveScene(const std::filesystem::path& path) {
    SaveSceneFile(scene_, path);
    log_.Info("Saved " + path.string());
}

void Shell::LoadDemo(const DemoInfo& demo) {
    demo_.reset();
    for (const Path& root : scene_.Children(Path::Root())) {
        scene_.Remove(root);
    }
    history_.Reset();
    selection_.Clear();
    activeCamera_ = Path();
    demoInfo_ = demo;
    demo_ = demo.create(scene_);
    scene_.Update();
    log_.Info("Running " + demo.Label());
    Frame();
}

void Shell::StepDemo() {
    if (demo_) {
        Attempt("Stepping the demo", [&] { demo_->Advance(); });
    }
}

void Shell::ResetDemo() {
    if (demo_) {
        const DemoInfo demo = demoInfo_;
        Attempt("Resetting the demo", [&] { LoadDemo(demo); });
    }
}

void Shell::AddRecentCapture(const std::filesystem::path& path) {
    recentCaptures_.push_back(path);
    // The newest dozen.
    if (recentCaptures_.size() > 12) {
        recentCaptures_.erase(recentCaptures_.begin());
    }
    log_.Info("Captured " + path.string());
}

CaptureNameFields Shell::CaptureFields() const {
    return {.scene = demo_ ? demoInfo_.Id() : std::string("ernie"),
            .camera = activeCamera_.IsEmpty() ? "Perspective" : std::string(activeCamera_.Name())};
}

namespace {

/// After Ernie 1's Demos panel: a header a category, a row a demo, then the running demo's
/// controls, description and options.
class DemosPanel final : public Panel {
  public:
    explicit DemosPanel(Shell& shell)
        : Panel("Demos", Dock::Right), shell_(shell), demos_(RegisteredDemos()) {}

    void Draw() override {
        const DemoInfo* running = shell_.CurrentDemoInfo();
        std::string_view category;
        bool open = false;
        for (const DemoInfo& demo : demos_) {
            if (demo.category != category) {
                category = demo.category;
                if (running != nullptr && running->category == category) {
                    ImGui::SetNextItemOpen(true, ImGuiCond_Appearing);
                }
                open = ImGui::CollapsingHeader(demo.category.c_str());
            }
            if (!open) {
                continue;
            }
            ImGui::Indent();
            const bool current = running != nullptr && running->Id() == demo.Id();
            if (ImGui::Selectable(demo.name.c_str(), current)) {
                Load(demo);
            }
            ImGui::SetItemTooltip("%s", demo.description.c_str());
            ImGui::Unindent();
        }
        Demo* demo = shell_.CurrentDemo();
        ImGui::SeparatorText(running != nullptr ? running->Label().c_str() : "No demo");
        if (demo == nullptr) {
            ImGui::TextDisabled(demos_.empty() ? "No demos are registered." : "Pick a demo above.");
            return;
        }
        ImGui::Checkbox("Paused", &shell_.demoPaused);
        ImGui::SameLine();
        if (ImGui::Button("Step")) {
            shell_.StepDemo();
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset")) {
            shell_.ResetDemo();
        }
        ImGui::SameLine();
        ImGui::TextDisabled("frame %llu, %.2f s", static_cast<unsigned long long>(demo->Frame()),
                            demo->Time());
        ImGui::TextWrapped("%s", demo->description.c_str());
        ImGui::SeparatorText("Options");
        demo->Gui();
    }

  private:
    void Load(const DemoInfo& demo) {
        try {
            shell_.LoadDemo(demo);
        } catch (const std::exception& error) {
            shell_.Log().Error("Loading " + demo.Label() + ": " + error.what());
        }
    }

    Shell& shell_;
    std::vector<DemoInfo> demos_;
};

/// After Ernie 1's Capture panel: where captures go and how they are named, the viewport's
/// screenshot, sequences of the timeline's frames encoded into video, and the recent captures.
class CapturePanel final : public Panel {
  public:
    explicit CapturePanel(Shell& shell) : Panel("Capture", Dock::Bottom), shell_(shell) {}

    void Draw() override {
        CaptureSettings& settings = shell_.Capture();
        if (ImGui::BeginTable("##capture", 2, ImGuiTableFlags_SizingFixedFit)) {
            ImGui::TableSetupColumn("name", ImGuiTableColumnFlags_WidthFixed,
                                    8.0f * ImGui::GetFontSize());
            ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch);
            DrawOutput(settings);
            Row("Screenshot");
            if (ImGui::Button((std::string(icons::kDownload) + "  Capture (F10)").c_str())) {
                shell_.RequestScreenshot();
            }
            ImGui::SameLine();
            ImGui::TextDisabled("the viewport's image as rendered");
            DrawSequence(settings);
            ImGui::EndTable();
        }
        DrawRecent();
    }

  private:
    static void Row(const char* label) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(label);
        ImGui::TableSetColumnIndex(1);
    }

    static void DrawOutput(CaptureSettings& settings) {
        Row("Folder");
        std::string folder = settings.folder.string();
        ImGui::SetNextItemWidth(-1.0f);
        if (ImGui::InputText("##folder", &folder)) {
            settings.folder = folder;
        }
        Row("Name");
        ImGui::SetNextItemWidth(-1.0f);
        ImGui::InputText("##name", &settings.name);
        ImGui::SetItemTooltip("{scene}, {camera}, {date}, {time} and {frame} fill in; a capture "
                              "never overwrites another");
    }

    void DrawSequence(CaptureSettings& settings) {
        const app::Timeline& timeline = shell_.Timeline();
        if (!rangeSet_) {
            start_ = timeline.Start();
            end_ = timeline.End();
        }
        Row("Frames");
        const float field = 5.0f * ImGui::GetFontSize();
        ImGui::SetNextItemWidth(field);
        rangeSet_ |= ImGui::InputDouble("##from", &start_, 0.0, 0.0, "%g");
        ImGui::SameLine();
        ImGui::TextUnformatted("to");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(field);
        rangeSet_ |= ImGui::InputDouble("##to", &end_, 0.0, 0.0, "%g");
        ImGui::SameLine();
        if (ImGui::Button("Timeline Range")) {
            rangeSet_ = false;
        }
        ImGui::SameLine();
        ImGui::TextUnformatted("every");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(field);
        ImGui::InputInt("##every", &every_);
        every_ = std::max(every_, 1);
        ImGui::SameLine();
        ImGui::TextDisabled("at %g fps", timeline.FramesPerSecond() / every_);

        Row("Video");
        static const bool ffmpeg = !FindProgram("ffmpeg").empty();
        constexpr std::array kFormats{"Frames only", "MP4 (H.264)", "ProRes 422 HQ (.mov)"};
        int format = static_cast<int>(settings.video);
        ImGui::BeginDisabled(!ffmpeg);
        ImGui::SetNextItemWidth(14.0f * ImGui::GetFontSize());
        if (ImGui::Combo("##video", &format, kFormats.data(), static_cast<int>(kFormats.size()))) {
            settings.video = static_cast<VideoFormat>(format);
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(settings.video == VideoFormat::None);
        ImGui::Checkbox("Keep frames", &settings.keepFrames);
        ImGui::EndDisabled();
        ImGui::EndDisabled();
        if (!ffmpeg) {
            ImGui::SameLine();
            ImGui::TextDisabled("ffmpeg is not on the PATH: frames only");
        }

        Row("");
        if (const auto progress = shell_.Recording()) {
            if (ImGui::Button((std::string(icons::kSquare) + "  Stop").c_str())) {
                shell_.stopRecording = true;
            }
            ImGui::SameLine();
            const auto [done, total] = *progress;
            const std::string label = std::to_string(done) + " / " + std::to_string(total);
            ImGui::ProgressBar(total > 0 ? static_cast<float>(done) / static_cast<float>(total)
                                         : 0.0f,
                               ImVec2(-1.0f, 0.0f), label.c_str());
            return;
        }
        ImGui::BeginDisabled(shell_.Encoding());
        if (ImGui::Button((std::string(icons::kVideo) + "  Record").c_str())) {
            shell_.RequestSequence({.start = start_,
                                    .end = end_,
                                    .every = every_,
                                    .video = ffmpeg ? settings.video : VideoFormat::None,
                                    .keepFrames = settings.keepFrames});
        }
        ImGui::EndDisabled();
        if (shell_.Encoding()) {
            ImGui::SameLine();
            ImGui::TextDisabled("Encoding the last video...");
        }
    }

    void DrawRecent() {
        const std::vector<std::filesystem::path>& recent = shell_.RecentCaptures();
        if (recent.empty()) {
            return;
        }
        ImGui::SeparatorText("Recent");
        // Newest first; a click copies the path.
        for (auto it = recent.rbegin(); it != recent.rend(); ++it) {
            const std::string path = it->string();
            if (ImGui::Selectable(path.c_str())) {
                ImGui::SetClipboardText(path.c_str());
            }
            ImGui::SetItemTooltip("Click to copy the path");
        }
    }

    Shell& shell_;
    double start_ = 0.0;
    double end_ = 0.0;
    /// Whether the range was chosen apart from the timeline's.
    bool rangeSet_ = false;
    int every_ = 1;
};

}  // namespace

std::unique_ptr<Panel> MakeDemosPanel(Shell& shell) {
    return std::make_unique<DemosPanel>(shell);
}

std::unique_ptr<Panel> MakeCapturePanel(Shell& shell) {
    return std::make_unique<CapturePanel>(shell);
}

// Presenters

OffscreenPresenter::OffscreenPresenter(DeviceContext& context, glm::uvec2 extent)
    : image_(context,
             {.format = ImageFormat::Rgba8Unorm,
              .extent = {extent, 1},
              .usage = ImageUsage::Attachment | ImageUsage::Sampled},
             "Viewer frame"),
      drawn_(image_.Ready()) {}

void OffscreenPresenter::BeginUiFrame(double delta) {
    ImGuiIO& io = ImGui::GetIO();
    const glm::uvec3 extent = image_.Description().extent;
    io.DisplaySize = {static_cast<float>(extent.x), static_cast<float>(extent.y)};
    io.DeltaTime = static_cast<float>(std::max(delta, 1e-4));
}

#if ERNIE_VIEWER_WINDOW
namespace {

int glfwUsers = 0;

/// @return Whether GLFW is initialised, initialising it for one more user.
bool AcquireGlfw() {
    if (glfwUsers == 0) {
        glfwSetErrorCallback([](int code, const char* message) {
            log::Detail("GLFW error " + std::to_string(code) + ": " + message);
        });
        // GLFW finds Vulkan through the loader volk opened, as the device layer does.
        if (volkInitialize() != VK_SUCCESS) {
            return false;
        }
        glfwInitVulkanLoader(vkGetInstanceProcAddr);
        if (glfwInit() == GLFW_FALSE) {
            return false;
        }
    }
    ++glfwUsers;
    return true;
}

void ReleaseGlfw() {
    if (--glfwUsers == 0) {
        glfwTerminate();
    }
}

template <typename T, typename Query> std::vector<T> Enumerate(Query&& query) {
    std::uint32_t count = 0;
    query(&count, nullptr);
    std::vector<T> items(count);
    query(&count, items.data());
    return items;
}

VkSurfaceKHR CreateSurface(VkInstance instance, GLFWwindow* window) {
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    CheckVk(glfwCreateWindowSurface(instance, window, nullptr, &surface),
            "glfwCreateWindowSurface");
    return surface;
}

/// @return An 8-bit UNORM format: the UI's colours are sRGB-encoded already.
VkSurfaceFormatKHR ChooseFormat(VkPhysicalDevice physical, VkSurfaceKHR surface) {
    const std::vector<VkSurfaceFormatKHR> formats =
        Enumerate<VkSurfaceFormatKHR>([&](std::uint32_t* count, VkSurfaceFormatKHR* items) {
            vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, count, items);
        });
    if (formats.empty()) {
        throw DeviceError(DeviceErrorKind::Unsupported, "The window's surface has no formats");
    }
    for (const VkFormat wanted : {VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM}) {
        const auto found = std::ranges::find_if(formats, [&](const VkSurfaceFormatKHR& format) {
            return format.format == wanted &&
                   format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
        });
        if (found != formats.end()) {
            return *found;
        }
    }
    return formats.front();
}

VkPresentModeKHR ChoosePresentMode(VkPhysicalDevice physical, VkSurfaceKHR surface, bool vsync) {
    if (vsync) {
        return VK_PRESENT_MODE_FIFO_KHR;
    }
    const std::vector<VkPresentModeKHR> modes =
        Enumerate<VkPresentModeKHR>([&](std::uint32_t* count, VkPresentModeKHR* items) {
            vkGetPhysicalDeviceSurfacePresentModesKHR(physical, surface, count, items);
        });
    for (const VkPresentModeKHR wanted :
         {VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR}) {
        if (std::ranges::find(modes, wanted) != modes.end()) {
            return wanted;
        }
    }
    // The one mode every surface supports.
    return VK_PRESENT_MODE_FIFO_KHR;
}

/// @return Whether the swapchain still matches the surface after an acquire or present.
bool Matches(VkResult result) {
    return result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR;
}

}  // namespace

bool Window::Available() {
    if (!AcquireGlfw()) {
        return false;
    }
    const bool supported = glfwVulkanSupported() == GLFW_TRUE;
    ReleaseGlfw();
    return supported;
}

std::vector<std::string> Window::InstanceExtensions() {
    if (!AcquireGlfw()) {
        return {};
    }
    std::uint32_t count = 0;
    const char** names = glfwGetRequiredInstanceExtensions(&count);
    std::vector<std::string> extensions(names, names + (names != nullptr ? count : 0));
    ReleaseGlfw();
    return extensions;
}

Window::Window(const std::string& title, glm::uvec2 size) {
    if (!AcquireGlfw()) {
        throw std::runtime_error("No display to open a window on; --offscreen draws without one");
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window_ = glfwCreateWindow(static_cast<int>(size.x), static_cast<int>(size.y), title.c_str(),
                               nullptr, nullptr);
    if (window_ == nullptr) {
        ReleaseGlfw();
        throw std::runtime_error("Could not open a window");
    }
}

Window::~Window() {
    glfwDestroyWindow(window_);
    ReleaseGlfw();
}

glm::uvec2 Window::FramebufferSize() const {
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    return {static_cast<std::uint32_t>(std::max(width, 0)),
            static_cast<std::uint32_t>(std::max(height, 0))};
}

float Window::ContentScale() const {
    float x = 1.0f;
    float y = 1.0f;
    glfwGetWindowContentScale(window_, &x, &y);
    return std::max(x, 1.0f);
}

WindowPresenter::WindowPresenter(DeviceContext& context, std::unique_ptr<Window> window, bool vsync)
    : context_(context), window_(std::move(window)),
      surface_(CreateSurface(context.Instance(), window_->Handle())), vsync_(vsync) {
    const OnFailure destroySurface(
        [&] { vkDestroySurfaceKHR(context_.Instance(), surface_, nullptr); });
    if (!context.Capabilities().present) {
        throw DeviceError(DeviceErrorKind::Unsupported,
                          "The device cannot present to a window; --offscreen draws without one");
    }
    VkBool32 presents = VK_FALSE;
    CheckVk(vkGetPhysicalDeviceSurfaceSupportKHR(context.Physical(), context.Main().Family(),
                                                 surface_, &presents),
            "vkGetPhysicalDeviceSurfaceSupportKHR");
    if (presents == VK_FALSE) {
        throw DeviceError(DeviceErrorKind::Unsupported,
                          "The device's main queue cannot present to the window");
    }
    format_ = ChooseFormat(context.Physical(), surface_);
    const OnFailure destroyHandles([&] {
        DestroyFrames();
        vkDestroySwapchainKHR(context_.Device(), swapchain_, nullptr);
    });
    CreateSwapchain(window_->FramebufferSize());
    ImGui_ImplGlfw_InitForVulkan(window_->Handle(), true);
}

WindowPresenter::~WindowPresenter() {
    ImGui_ImplGlfw_Shutdown();
    DestroyFrames();
    vkDestroySwapchainKHR(context_.Device(), swapchain_, nullptr);
    vkDestroySurfaceKHR(context_.Instance(), surface_, nullptr);
}

void WindowPresenter::CreateSwapchain(glm::uvec2 extent) {
    ERNIE_ZONE("Create the swapchain");
    VkSurfaceCapabilitiesKHR capabilities{};
    CheckVk(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(context_.Physical(), surface_, &capabilities),
            "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
    if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max()) {
        extent_ = capabilities.currentExtent;
    } else {
        extent_ = {std::clamp(extent.x, capabilities.minImageExtent.width,
                              capabilities.maxImageExtent.width),
                   std::clamp(extent.y, capabilities.minImageExtent.height,
                              capabilities.maxImageExtent.height)};
    }
    std::uint32_t count = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0) {
        count = std::min(count, capabilities.maxImageCount);
    }
    const VkSwapchainKHR old = swapchain_;
    const VkSwapchainCreateInfoKHR info{
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = surface_,
        .minImageCount = count,
        .imageFormat = format_.format,
        .imageColorSpace = format_.colorSpace,
        .imageExtent = extent_,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .preTransform = capabilities.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = ChoosePresentMode(context_.Physical(), surface_, vsync_),
        .clipped = VK_TRUE,
        .oldSwapchain = old};
    CheckVk(vkCreateSwapchainKHR(context_.Device(), &info, nullptr, &swapchain_),
            "vkCreateSwapchainKHR");
    if (old != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(context_.Device(), old, nullptr);
    }
    images_ = Enumerate<VkImage>([&](std::uint32_t* n, VkImage* items) {
        vkGetSwapchainImagesKHR(context_.Device(), swapchain_, n, items);
    });
    for (const VkImage image : images_) {
        const VkImageViewCreateInfo view{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = format_.format,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        CheckVk(vkCreateImageView(context_.Device(), &view, nullptr, &views_.emplace_back()),
                "vkCreateImageView");
    }
    const VkSemaphoreCreateInfo semaphore{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    const auto create = [&](std::vector<VkSemaphore>& semaphores, std::size_t wanted) {
        while (semaphores.size() < wanted) {
            CheckVk(vkCreateSemaphore(context_.Device(), &semaphore, nullptr,
                                      &semaphores.emplace_back()),
                    "vkCreateSemaphore");
        }
    };
    create(acquiredSemaphores_, images_.size() + 1);
    create(drawnSemaphores_, images_.size());
    nextAcquired_ = 0;
}

void WindowPresenter::DestroyFrames() {
    // The queue's idle once its presents have waited on their semaphores too.
    Quietly([this] { context_.Main().Borrow([](VkQueue queue) { vkQueueWaitIdle(queue); }); });
    for (const VkImageView view : views_) {
        vkDestroyImageView(context_.Device(), view, nullptr);
    }
    for (const VkSemaphore semaphore : acquiredSemaphores_) {
        vkDestroySemaphore(context_.Device(), semaphore, nullptr);
    }
    for (const VkSemaphore semaphore : drawnSemaphores_) {
        vkDestroySemaphore(context_.Device(), semaphore, nullptr);
    }
    views_.clear();
    acquiredSemaphores_.clear();
    drawnSemaphores_.clear();
}

void WindowPresenter::PollEvents() {
    glfwPollEvents();
}

bool WindowPresenter::CloseRequested() const {
    return glfwWindowShouldClose(window_->Handle()) == GLFW_TRUE;
}

void WindowPresenter::BeginUiFrame(double /*delta*/) {
    // The backend measures time steps and sizes itself.
    ImGui_ImplGlfw_NewFrame();
}

std::optional<FrameTarget> WindowPresenter::Acquire() {
    const glm::uvec2 size = window_->FramebufferSize();
    if (size.x == 0 || size.y == 0) {
        return std::nullopt;
    }
    const glm::uvec2 extent(extent_.width, extent_.height);
    if (stale_ || size != extent) {
        ERNIE_ZONE("Resize the swapchain");
        DestroyFrames();
        CreateSwapchain(size);
        stale_ = false;
    }
    // Blocks only while every image is queued for the display, as vsync paces frames; the
    // device waits for the image itself.
    ERNIE_ZONE("Wait for a swapchain image");
    const VkSemaphore acquired = acquiredSemaphores_[nextAcquired_];
    std::uint32_t index = 0;
    const VkResult result = vkAcquireNextImageKHR(context_.Device(), swapchain_,
                                                  std::numeric_limits<std::uint64_t>::max(),
                                                  acquired, VK_NULL_HANDLE, &index);
    if (!Matches(result)) {
        if (result != VK_ERROR_OUT_OF_DATE_KHR) {
            CheckVk(result, "vkAcquireNextImageKHR");
        }
        stale_ = true;
        return std::nullopt;
    }
    nextAcquired_ = (nextAcquired_ + 1) % acquiredSemaphores_.size();
    acquired_ = index;
    return FrameTarget{images_[index],
                       views_[index],
                       glm::uvec2(extent_.width, extent_.height),
                       VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                       {acquired, drawnSemaphores_[index]}};
}

void WindowPresenter::Present(Ticket /*drawn*/) {
    if (!acquired_) {
        return;
    }
    // The device waits for the frame's drawing; the host goes on to the next.
    const VkPresentInfoKHR info{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                                .waitSemaphoreCount = 1,
                                .pWaitSemaphores = &drawnSemaphores_[*acquired_],
                                .swapchainCount = 1,
                                .pSwapchains = &swapchain_,
                                .pImageIndices = &*acquired_};
    const VkResult result = context_.Main().Present(info);
    if (!Matches(result) && result != VK_ERROR_OUT_OF_DATE_KHR) {
        CheckVk(result, "vkQueuePresentKHR");
    }
    stale_ = result != VK_SUCCESS;
    acquired_.reset();
}
#endif

}  // namespace ernie::app
