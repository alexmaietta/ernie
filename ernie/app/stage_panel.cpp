// SPDX-License-Identifier: BSD-3-Clause
/// @file stage_panel.cpp
/// @brief The Stage panel: the scene's outliner.
#include "ernie/app/panels.h"
#include "ernie/app/selection.h"
#include "ernie/scene/scene.h"
#include "ernie/util/text.h"

#include <imgui.h>

#include <array>
#include <string>

namespace ernie::app {

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
    void DrawChildren(const Path& parent) {
        for (const Path& child : scene_.Children(parent)) {
            if (search_[0] == '\0' || matches_.contains(child)) {
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
        if (search_[0] != '\0') {
            ImGui::SetNextItemOpen(true);
        }
        const std::string name(path.Name());
        const bool open = ImGui::TreeNodeEx(path.String().c_str(), flags, "%s", name.c_str());
        if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
            if (ImGui::GetIO().KeyCtrl) {
                selection_.Toggle(path);
            } else {
                selection_.Set(path);
            }
        }
        ImGui::TableNextColumn();
        const std::string type(scene_.TypeName(scene_.Require(path)));
        ImGui::TextDisabled("%s", type.c_str());
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

}  // namespace

std::unique_ptr<Panel> MakeStagePanel(const Scene& scene, Selection& selection) {
    return std::make_unique<StagePanel>(scene, selection);
}

}  // namespace ernie::app
