// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/builtin_panels.h"
#include "ernie/app/create.h"
#include "ernie/app/prim_icons.h"
#include "ernie/app/session.h"
#include "ernie/app/viewer.h"
#include "ernie/scene/tokens.h"

#include <imgui.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace ernie::app {

namespace {

bool ContainsIgnoringCase(std::string_view text, std::string_view part) {
    return std::search(text.begin(), text.end(), part.begin(), part.end(), [](char a, char b) {
               return std::tolower(static_cast<unsigned char>(a)) ==
                      std::tolower(static_cast<unsigned char>(b));
           }) != text.end();
}

/// The kinds of prim the filter button shows or hides.
enum Kind { kGeometry, kLights, kCameras, kMaterials, kRender, kOther, kKinds };
constexpr const char* kKindNames[kKinds] = {"Geometry",  "Lights", "Cameras",
                                            "Materials", "Render", "Other"};

Kind KindOf(std::string_view type) {
    using namespace tokens;
    if (type == kMesh || type == kCube || type == kSphere || type == kCylinder || type == kCone ||
        type == kCapsule || type == kPlane || type == kPointInstancer || type == kGroundPlane ||
        type == kTerrain || type == kVolume) {
        return kGeometry;
    }
    if (type == kDistantLight || type == kSphereLight || type == kRectLight || type == kDiskLight ||
        type == kCylinderLight || type == kDomeLight || type == kStarField) {
        return kLights;
    }
    if (type == kCamera) {
        return kCameras;
    }
    if (type == kMaterial) {
        return kMaterials;
    }
    if (type == kRenderSettings || type == kRenderProduct || type == kRenderVar ||
        type == kSensorFilter) {
        return kRender;
    }
    return kOther;
}

/// Whether prims of `type` can be shown and hidden (an eye in the visibility column).
bool Imageable(std::string_view type) {
    const Kind kind = KindOf(type);
    return kind == kGeometry || kind == kLights || kind == kCameras || type == tokens::kXform ||
           type == tokens::kScope || type.empty();
}

ImU32 Colour(const glm::vec4& c, float alpha = 1.0f) {
    return ImGui::GetColorU32(ImVec4(c.r, c.g, c.b, c.a * alpha));
}

/// A magnifying glass in the square of `size` at `corner`.
void Magnifier(ImDrawList* draw, ImVec2 corner, float size, ImU32 colour) {
    const ImVec2 centre(corner.x + 0.42f * size, corner.y + 0.42f * size);
    draw->AddCircle(centre, 0.27f * size, colour, 0, 1.4f);
    draw->AddLine(ImVec2(corner.x + 0.62f * size, corner.y + 0.62f * size),
                  ImVec2(corner.x + 0.88f * size, corner.y + 0.88f * size), colour, 1.6f);
}

/// A funnel in the square of `size` at `corner`.
void Funnel(ImDrawList* draw, ImVec2 corner, float size, ImU32 colour) {
    const ImVec2 points[] = {{corner.x + 0.15f * size, corner.y + 0.22f * size},
                             {corner.x + 0.85f * size, corner.y + 0.22f * size},
                             {corner.x + 0.56f * size, corner.y + 0.55f * size},
                             {corner.x + 0.44f * size, corner.y + 0.55f * size}};
    draw->AddConvexPolyFilled(points, 4, colour);
    draw->AddRectFilled(ImVec2(corner.x + 0.44f * size, corner.y + 0.5f * size),
                        ImVec2(corner.x + 0.56f * size, corner.y + 0.82f * size), colour);
}

/// Three bars in the square of `size` at `corner`.
void Hamburger(ImDrawList* draw, ImVec2 corner, float size, ImU32 colour) {
    for (const float v : {0.28f, 0.5f, 0.72f}) {
        draw->AddLine(ImVec2(corner.x + 0.2f * size, corner.y + v * size),
                      ImVec2(corner.x + 0.8f * size, corner.y + v * size), colour, 1.6f);
    }
}

/// An eye in the square of `size` centred at `centre`: open, or struck through when not `shown`.
void Eye(ImDrawList* draw, ImVec2 centre, float size, bool shown, ImU32 colour) {
    const float w = 0.36f * size;
    const float h = 0.22f * size;
    const float thickness = std::max(1.0f, 0.07f * size);
    draw->AddBezierQuadratic(ImVec2(centre.x - w, centre.y), ImVec2(centre.x, centre.y - 2 * h),
                             ImVec2(centre.x + w, centre.y), colour, thickness);
    draw->AddBezierQuadratic(ImVec2(centre.x - w, centre.y), ImVec2(centre.x, centre.y + 2 * h),
                             ImVec2(centre.x + w, centre.y), colour, thickness);
    if (shown) {
        draw->AddCircleFilled(centre, 0.12f * size, colour);
    } else {
        draw->AddLine(ImVec2(centre.x - w, centre.y + w), ImVec2(centre.x + w, centre.y - w),
                      colour, thickness);
    }
}

/// The Stage panel, after Omniverse Kit's stage window: a search field with filter and options
/// buttons, then the prim tree in Name, visibility and Type columns, with boxed +/- toggles,
/// coloured type icons, alternate row shading and whole-row selection.
class StagePanel final : public Panel {
  public:
    StagePanel() : Panel("Stage", Dock::Right) {
        shownKinds_.fill(true);
    }

    void Draw(Viewer& viewer) override {
        Scene& scene = viewer.GetScene();
        theme_ = viewer.GetTheme();
        DrawSearchRow();
        // Reveal what was selected elsewhere, such as by a click in the viewport.
        const std::vector<Path>& selection = viewer.Selection();
        reveal_ = selection != shown_;
        shown_ = selection;
        // F2 renames the selected prim, as in Kit.
        if (selection.size() == 1 && renaming_.IsEmpty() &&
            ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) &&
            !ImGui::GetIO().WantTextInput && ImGui::IsKeyPressed(ImGuiKey_F2, false) &&
            !IsSessionPath(selection.front())) {
            StartRename(selection.front());
        }
        if (reveal_) {
            for (const Path& selected : selection) {
                for (Path at = selected.Parent(); !at.IsEmpty() && !at.IsRoot(); at = at.Parent()) {
                    expanded_.insert(at);
                }
            }
        }
        constexpr ImGuiTableFlags kFlags = ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY |
                                           ImGuiTableFlags_BordersInnerV |
                                           ImGuiTableFlags_Resizable;
        ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(4.0f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_TableRowBgAlt, ImVec4(1.0f, 1.0f, 1.0f, 0.02f));
        if (ImGui::BeginTable("##stage", 3, kFlags)) {
            const float eye = ImGui::GetFrameHeight();
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn(
                "##visible", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoResize,
                eye);
            ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed,
                                    7.0f * ImGui::GetFontSize());
            DrawHeaderRow();
            const std::string_view search(search_.data());
            if (search.empty()) {
                for (const Path& child : scene.Children(Path::Root())) {
                    DrawNode(viewer, scene, child, 0);
                }
            } else {
                for (const Path& path : scene.Prims()) {
                    if (Shown(scene, path) && ContainsIgnoringCase(path.String(), search)) {
                        DrawRow(viewer, scene, path, path.String(), 0, false);
                    }
                }
            }
            ImGui::EndTable();
        }
        ImGui::PopStyleColor();
        ImGui::PopStyleVar();
    }

  private:
    /// The search field with a magnifier, and the filter and options buttons at its right.
    void DrawSearchRow() {
        const float button = ImGui::GetFrameHeight();
        const float spacing = ImGui::GetStyle().ItemSpacing.x;
        ImGui::SetNextItemWidth(-2.0f * (button + spacing));
        const ImVec2 field = ImGui::GetCursorScreenPos();
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,
                            ImVec2(button + 2.0f, ImGui::GetStyle().FramePadding.y));
        ImGui::InputTextWithHint("##search", "Search", search_.data(), search_.size());
        ImGui::PopStyleVar();
        ImDrawList* draw = ImGui::GetWindowDrawList();
        const float glyph = 0.8f * button;
        Magnifier(draw, ImVec2(field.x + 0.25f * button, field.y + 0.1f * button), glyph,
                  Colour(theme_.textMuted));

        const bool filtering =
            std::any_of(shownKinds_.begin(), shownKinds_.end(), [](bool on) { return !on; });
        ImGui::SameLine();
        ImVec2 corner = ImGui::GetCursorScreenPos();
        if (ImGui::InvisibleButton("##filter", ImVec2(button, button))) {
            ImGui::OpenPopup("##filters");
        }
        Funnel(draw, ImVec2(corner.x + 0.1f * button, corner.y + 0.1f * button), glyph,
               Colour(filtering ? theme_.accent
                                : (ImGui::IsItemHovered() ? theme_.text : theme_.textMuted)));
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Show prims by kind");
        }
        if (ImGui::BeginPopup("##filters")) {
            for (int kind = 0; kind < kKinds; ++kind) {
                ImGui::Checkbox(kKindNames[kind], &shownKinds_[static_cast<std::size_t>(kind)]);
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Show All")) {
                shownKinds_.fill(true);
            }
            ImGui::EndPopup();
        }
        ImGui::SameLine();
        corner = ImGui::GetCursorScreenPos();
        if (ImGui::InvisibleButton("##options", ImVec2(button, button))) {
            ImGui::OpenPopup("##stageOptions");
        }
        Hamburger(draw, ImVec2(corner.x + 0.1f * button, corner.y + 0.1f * button), glyph,
                  Colour(ImGui::IsItemHovered() ? theme_.text : theme_.textMuted));
        if (ImGui::BeginPopup("##stageOptions")) {
            ImGui::MenuItem("Show Session Prims", nullptr, &showSession_);
            if (ImGui::MenuItem("Collapse All")) {
                expanded_.clear();
            }
            ImGui::EndPopup();
        }
    }

    /// The header row, with an eye over the visibility column.
    void DrawHeaderRow() {
        ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
        ImGui::TableSetColumnIndex(0);
        ImGui::TableHeader("Name");
        ImGui::TableSetColumnIndex(1);
        const ImVec2 corner = ImGui::GetCursorScreenPos();
        ImGui::TableHeader("##eye");
        const float size = ImGui::GetFontSize();
        Eye(ImGui::GetWindowDrawList(),
            ImVec2(corner.x + 0.5f * ImGui::GetFrameHeight() - 4.0f, corner.y + 0.5f * size), size,
            true, Colour(theme_.textMuted));
        ImGui::TableSetColumnIndex(2);
        ImGui::TableHeader("Type");
    }

    bool Shown(const Scene& scene, const Path& path) const {
        return (showSession_ || !IsSessionPath(path)) &&
               shownKinds_[static_cast<std::size_t>(KindOf(scene.TypeOf(path)))];
    }

    void DrawNode(Viewer& viewer, Scene& scene, const Path& path, int depth) {
        if (!showSession_ && IsSessionPath(path)) {
            return;
        }
        const std::vector<Path> children = scene.Children(path);
        // New prims at the top start open, as Kit opens the default prim.
        if (seen_.insert(path).second && depth == 0) {
            expanded_.insert(path);
        }
        const bool open = expanded_.contains(path);
        if (Shown(scene, path)) {
            DrawRow(viewer, scene, path, std::string(path.Name()), depth, !children.empty());
        }
        if (open) {
            for (const Path& child : children) {
                DrawNode(viewer, scene, child, depth + 1);
            }
        }
    }

    /// One prim's row: selection across its width, its toggle, icon and name, its eye and type.
    void DrawRow(Viewer& viewer, Scene& scene, const Path& path, const std::string& label,
                 int depth, bool parent) {
        const float height = ImGui::GetFrameHeight();
        ImGui::TableNextRow(ImGuiTableRowFlags_None, height);
        ImGui::TableSetColumnIndex(0);
        ImGui::PushID(path.String().c_str());
        const std::vector<Path>& selection = viewer.Selection();
        const bool selected =
            std::find(selection.begin(), selection.end(), path) != selection.end();
        const ImVec2 rowStart = ImGui::GetCursorScreenPos();
        if (ImGui::Selectable("##row", selected,
                              ImGuiSelectableFlags_SpanAllColumns |
                                  ImGuiSelectableFlags_AllowOverlap |
                                  ImGuiSelectableFlags_AllowDoubleClick,
                              ImVec2(0.0f, height))) {
            if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                const Path frame[] = {path};
                viewer.Frame(frame);
            } else {
                Select(viewer, path, ImGui::GetIO().KeyCtrl);
            }
        }
        if (selected && reveal_) {
            ImGui::SetScrollHereY();
        }
        DrawContextMenu(viewer, scene, path);

        // The boxed +/- toggle, the icon and the name, indented by depth.
        const std::string& type = scene.TypeOf(path);
        const bool visible = scene.IsVisible(path);
        ImDrawList* draw = ImGui::GetWindowDrawList();
        const float indent = std::round(0.9f * ImGui::GetFontSize());
        const float box = std::round(0.62f * ImGui::GetFontSize());
        float x = rowStart.x + static_cast<float>(depth) * indent;
        const float middle = rowStart.y + 0.5f * height;
        if (parent) {
            const ImVec2 low(x, middle - 0.5f * box);
            const ImVec2 high(low.x + box, low.y + box);
            ImGui::SetCursorScreenPos(ImVec2(x - 2.0f, rowStart.y));
            if (ImGui::InvisibleButton("##toggle", ImVec2(box + 4.0f, height))) {
                if (!expanded_.erase(path)) {
                    expanded_.insert(path);
                }
            }
            const ImU32 line = Colour(ImGui::IsItemHovered() ? theme_.text : theme_.textMuted);
            draw->AddRect(low, high, line, 1.0f);
            const float c = 0.5f * box;
            draw->AddLine(ImVec2(low.x + 2.0f, low.y + c), ImVec2(high.x - 2.0f, low.y + c), line);
            if (!expanded_.contains(path)) {
                draw->AddLine(ImVec2(low.x + c, low.y + 2.0f), ImVec2(low.x + c, high.y - 2.0f),
                              line);
            }
        }
        x += box + 5.0f;
        const float icon = std::round(1.05f * ImGui::GetFontSize());
        DrawPrimIcon(draw, ImVec2(x, middle - 0.5f * icon), icon, type);
        x += icon + 5.0f;
        if (renaming_ == path) {
            DrawRenameField(viewer, path, ImVec2(x - 4.0f, rowStart.y),
                            ImGui::GetContentRegionMax().x + ImGui::GetWindowPos().x - x);
        } else {
            draw->AddText(ImVec2(x, middle - 0.5f * ImGui::GetFontSize()),
                          Colour(visible ? theme_.text : theme_.textMuted), label.c_str());
        }

        // The eye, for prims that can be hidden.
        ImGui::TableSetColumnIndex(1);
        if (Imageable(type)) {
            const bool authoredInvisible =
                scene.Get<std::string>(path, tokens::kVisibility, "") == tokens::kInvisible;
            const ImVec2 corner = ImGui::GetCursorScreenPos();
            if (ImGui::InvisibleButton("##visibility", ImVec2(height, height))) {
                viewer.Edit(
                    path, tokens::kVisibility,
                    std::string(authoredInvisible ? tokens::kInherited : tokens::kInvisible));
            }
            const bool hovered = ImGui::IsItemHovered();
            // Kit's eyes are small and grey, lighting up under the mouse.
            Eye(draw, ImVec2(corner.x + 0.5f * height, corner.y + 0.5f * height), 0.8f * height,
                !authoredInvisible,
                Colour(hovered ? theme_.text : theme_.textMuted, visible ? 0.9f : 0.45f));
            if (hovered) {
                ImGui::SetTooltip(authoredInvisible ? "Hidden; click to show"
                                  : visible ? "Visible; click to hide"
                                            : "Hidden by an ancestor; click to hide it too");
            }
        }
        ImGui::TableSetColumnIndex(2);
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("%s", type.c_str());
        ImGui::PopID();
    }

    /// Kit's prim context menu: creating under it, renaming, duplicating and deleting, its
    /// visibility, selecting and framing, and copying its path. Right-clicking an unselected
    /// prim selects it first.
    void DrawContextMenu(Viewer& viewer, Scene& scene, const Path& path) {
        if (ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
            const std::vector<Path>& selection = viewer.Selection();
            if (std::find(selection.begin(), selection.end(), path) == selection.end()) {
                viewer.SetSelection({path});
            }
        }
        if (!ImGui::BeginPopupContextItem("##context")) {
            return;
        }
        const bool editable = !IsSessionPath(path) && !path.IsRoot();
        if (ImGui::BeginMenu("Create", editable)) {
            // Under this prim when it groups others (an Xform or Scope), as the Create menu.
            const char* group = nullptr;
            for (const CreatableType& creatable : CreatableTypes()) {
                if (group == nullptr || std::string_view(group) != creatable.group) {
                    group = creatable.group;
                    ImGui::SeparatorText(group);
                }
                if (ImGui::MenuItem(creatable.label)) {
                    viewer.SetSelection({path});
                    try {
                        viewer.Create(creatable.type);
                    } catch (const std::invalid_argument& error) {
                        viewer.Log(LogLevel::Error, error.what());
                    }
                }
            }
            ImGui::EndMenu();
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Rename", "F2", false, editable)) {
            StartRename(path);
        }
        if (ImGui::MenuItem("Duplicate", "Ctrl+D", false, editable)) {
            viewer.Duplicate(std::vector<Path>(viewer.Selection()));
        }
        if (ImGui::MenuItem("Delete", "Del", false, editable)) {
            viewer.Delete(std::vector<Path>(viewer.Selection()));
        }
        ImGui::Separator();
        const bool hidden =
            scene.Get<std::string>(path, tokens::kVisibility, "") == tokens::kInvisible;
        if (ImGui::MenuItem(hidden ? "Show" : "Hide", nullptr, false,
                            editable && Imageable(scene.TypeOf(path)))) {
            viewer.Edit(path, tokens::kVisibility,
                        std::string(hidden ? tokens::kInherited : tokens::kInvisible));
        }
        if (ImGui::MenuItem("Select Children", nullptr, false, !scene.Children(path).empty())) {
            viewer.SetSelection(scene.Children(path));
        }
        if (ImGui::MenuItem("Expand All Below")) {
            ExpandBelow(scene, path);
        }
        if (ImGui::MenuItem("Frame", "F")) {
            const Path frame[] = {path};
            viewer.Frame(frame);
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Copy Path")) {
            ImGui::SetClipboardText(path.String().c_str());
        }
        ImGui::EndPopup();
    }

    void ExpandBelow(const Scene& scene, const Path& path) {
        expanded_.insert(path);
        for (const Path& child : scene.Children(path)) {
            ExpandBelow(scene, child);
        }
    }

    void StartRename(const Path& path) {
        renaming_ = path;
        renameFocus_ = true;
        const std::string name(path.Name());
        std::fill(renameBuffer_.begin(), renameBuffer_.end(), '\0');
        std::copy_n(name.begin(), std::min(name.size(), renameBuffer_.size() - 1),
                    renameBuffer_.begin());
    }

    /// The name field of the prim being renamed, at `corner` and `width` wide: Enter renames,
    /// Escape or clicking away cancels.
    void DrawRenameField(Viewer& viewer, const Path& path, ImVec2 corner, float width) {
        ImGui::SetCursorScreenPos(corner);
        ImGui::SetNextItemWidth(width);
        if (renameFocus_) {
            ImGui::SetKeyboardFocusHere();
            renameFocus_ = false;
        }
        const bool entered = ImGui::InputText(
            "##rename", renameBuffer_.data(), renameBuffer_.size(),
            ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll);
        if (entered) {
            try {
                viewer.Rename(path, renameBuffer_.data());
            } catch (const std::invalid_argument& error) {
                viewer.Log(LogLevel::Error, error.what());
            }
            renaming_ = Path();
        } else if (ImGui::IsItemDeactivated() || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            renaming_ = Path();
        }
    }

    static void Select(Viewer& viewer, const Path& path, bool toggle) {
        if (!toggle) {
            viewer.SetSelection({path});
            return;
        }
        std::vector<Path> paths = viewer.Selection();
        const auto found = std::find(paths.begin(), paths.end(), path);
        if (found != paths.end()) {
            paths.erase(found);
        } else {
            paths.push_back(path);
        }
        viewer.SetSelection(std::move(paths));
    }

    Theme theme_;
    std::array<char, 128> search_{};
    std::array<bool, kKinds> shownKinds_{};
    bool showSession_ = false;
    std::set<Path> expanded_;  ///< Prims whose children show.
    std::set<Path> seen_;      ///< Prims drawn before, to open new top-level ones once.
    std::vector<Path> shown_;  ///< The selection last drawn.
    bool reveal_ = false;      ///< Whether the selection changed since, to open and scroll to.
    Path renaming_;            ///< The prim whose name is being edited, if any.
    bool renameFocus_ = false;
    std::array<char, 256> renameBuffer_{};
};

}  // namespace

std::unique_ptr<Panel> MakeStagePanel() {
    return std::make_unique<StagePanel>();
}

}  // namespace ernie::app
