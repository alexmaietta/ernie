#include "ernie/app/builtin_panels.h"
#include "ernie/app/icons.h"
#include "ernie/app/session.h"
#include "ernie/app/viewer.h"
#include "ernie/scene/tokens.h"

#include <imgui.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <string>
#include <string_view>
#include <vector>

namespace ernie::app {

namespace {

/// The icon shown for a prim of `type`.
const char* IconOf(std::string_view type) {
    if (type == tokens::kCamera) {
        return icons::kCamera;
    }
    if (type == tokens::kDistantLight) {
        return icons::kSun;
    }
    if (type == tokens::kSphereLight || type == tokens::kRectLight ||
        type == tokens::kDiskLight || type == tokens::kDomeLight) {
        return icons::kLightbulb;
    }
    if (type == tokens::kRenderSettings || type == tokens::kRenderProduct ||
        type == tokens::kRenderVar) {
        return icons::kImage;
    }
    if (type == tokens::kSensorFilter) {
        return icons::kAperture;
    }
    if (type == tokens::kMaterial) {
        return icons::kSlidersHorizontal;
    }
    if (type.empty() || type == tokens::kXform || type == tokens::kScope) {
        return icons::kLayers;
    }
    return icons::kBox;
}

bool ContainsIgnoringCase(std::string_view text, std::string_view part) {
    return std::search(text.begin(), text.end(), part.begin(), part.end(), [](char a, char b) {
               return std::tolower(static_cast<unsigned char>(a)) ==
                      std::tolower(static_cast<unsigned char>(b));
           }) != text.end();
}

class StagePanel final : public Panel {
  public:
    StagePanel() : Panel("Stage", Dock::Left) {}

    void Draw(Viewer& viewer) override {
        Scene& scene = viewer.GetScene();
        ImGui::SetNextItemWidth(-ImGui::CalcTextSize("Session").x -
                                ImGui::GetFrameHeight() * 1.5f);
        ImGui::InputTextWithHint("##filter", "Filter", filter_.data(), filter_.size());
        ImGui::SameLine();
        ImGui::Checkbox("Session", &showSession_);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Show the prims the viewer defines for itself (%s)",
                              Viewer::SessionRoot().String().c_str());
        }
        // Reveal what was selected elsewhere, such as by a click in the viewport.
        const std::vector<Path>& selection = viewer.Selection();
        reveal_ = selection != shown_;
        shown_ = selection;
        if (ImGui::BeginChild("##tree")) {
            const std::string_view filter(filter_.data());
            if (filter.empty()) {
                for (const Path& child : scene.Children(Path::Root())) {
                    DrawNode(viewer, scene, child);
                }
            } else {
                for (const Path& path : scene.Prims()) {
                    if (Shown(path) && ContainsIgnoringCase(path.String(), filter)) {
                        DrawRow(viewer, scene, path, path.String(), true);
                    }
                }
            }
        }
        ImGui::EndChild();
    }

  private:
    bool Shown(const Path& path) const {
        return showSession_ || !IsSessionPath(path);
    }

    void DrawNode(Viewer& viewer, Scene& scene, const Path& path) {
        if (!Shown(path)) {
            return;
        }
        std::vector<Path> children = scene.Children(path);
        const bool leaf = children.empty();
        if (reveal_ && !leaf) {
            for (const Path& selected : viewer.Selection()) {
                if (selected != path && selected.HasPrefix(path)) {
                    ImGui::SetNextItemOpen(true);
                }
            }
        }
        if (DrawRow(viewer, scene, path, std::string(path.Name()), leaf)) {
            for (const Path& child : children) {
                DrawNode(viewer, scene, child);
            }
            ImGui::TreePop();
        }
    }

    /// One prim's row; returns whether it is open with its children to draw (then `TreePop`).
    bool DrawRow(Viewer& viewer, Scene& scene, const Path& path, const std::string& label,
                 bool leaf) {
        const std::vector<Path>& selection = viewer.Selection();
        const bool selected = std::find(selection.begin(), selection.end(), path) != selection.end();
        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow |
                                   ImGuiTreeNodeFlags_OpenOnDoubleClick |
                                   ImGuiTreeNodeFlags_SpanAvailWidth;
        if (selected) {
            flags |= ImGuiTreeNodeFlags_Selected;
        }
        if (leaf) {
            flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
        }
        const std::string& type = scene.TypeOf(path);
        const bool visible = scene.IsVisible(path);
        if (!visible) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
        }
        const std::string text = std::string(IconOf(type)) + "  " + label;
        const bool open = ImGui::TreeNodeEx(path.String().c_str(), flags, "%s", text.c_str());
        if (!visible) {
            ImGui::PopStyleColor();
        }
        if (selected && reveal_) {
            ImGui::SetScrollHereY();
        }
        if (ImGui::IsItemClicked(ImGuiMouseButton_Left) && !ImGui::IsItemToggledOpen()) {
            Select(viewer, path, ImGui::GetIO().KeyCtrl);
        }
        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && leaf) {
            const Path frame[] = {path};
            viewer.Frame(frame);
        }
        if (ImGui::BeginPopupContextItem()) {
            if (ImGui::MenuItem("Frame", "F")) {
                const Path frame[] = {path};
                viewer.Frame(frame);
            }
            if (ImGui::MenuItem("Copy Path")) {
                ImGui::SetClipboardText(path.String().c_str());
            }
            ImGui::EndPopup();
        }
        // The type, muted, and a visibility toggle at the right.
        const float right = ImGui::GetWindowContentRegionMax().x;
        const float eye = ImGui::CalcTextSize(icons::kEye).x + 2.0f * ImGui::GetStyle().FramePadding.x;
        if (!type.empty()) {
            const float typeWidth = ImGui::CalcTextSize(type.c_str()).x;
            ImGui::SameLine(std::max(ImGui::GetCursorPosX(),
                                     right - eye - typeWidth - ImGui::GetStyle().ItemSpacing.x));
            ImGui::TextDisabled("%s", type.c_str());
        }
        ImGui::SameLine(right - eye);
        ImGui::PushID(path.String().c_str());
        const bool authoredInvisible =
            scene.Get<std::string>(path, tokens::kVisibility, "") == tokens::kInvisible;
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
        if (authoredInvisible) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
        }
        if (ImGui::SmallButton(icons::kEye)) {
            viewer.Edit(path, tokens::kVisibility,
                        std::string(authoredInvisible ? tokens::kInherited : tokens::kInvisible));
        }
        if (authoredInvisible) {
            ImGui::PopStyleColor();
        }
        ImGui::PopStyleColor();
        ImGui::PopID();
        return open && !leaf;
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

    std::array<char, 128> filter_{};
    bool showSession_ = false;
    std::vector<Path> shown_;  ///< The selection last drawn.
    bool reveal_ = false;      ///< Whether the selection changed since, to open and scroll to.
};

}  // namespace

std::unique_ptr<Panel> MakeStagePanel() {
    return std::make_unique<StagePanel>();
}

}  // namespace ernie::app
