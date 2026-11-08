// SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <imgui.h>
#include <imgui_stdlib.h>

#include "ernie/app/viewer/panels/panels.h"
#include "ernie/app/viewer/viewport/viewport.h"
#include "ernie/app/viewer/widgets/widgets.h"
#include "ernie/scene/schema/schema.h"

namespace ernie::app {
namespace {

enum class Shown { Any, Visible, Hidden };

bool Glob(std::string_view pattern, std::string_view text) {
    size_t p = 0;
    size_t t = 0;
    size_t star = std::string_view::npos;
    size_t resume = 0;
    while (t < text.size()) {
        if (p < pattern.size() && (pattern[p] == '?' || pattern[p] == text[t])) {
            ++p;
            ++t;
        } else if (p < pattern.size() && pattern[p] == '*') {
            star = p++;
            resume = t;
        } else if (star != std::string_view::npos) {
            p = star + 1;
            t = ++resume;
        } else {
            return false;
        }
    }
    while (p < pattern.size() && pattern[p] == '*') {
        ++p;
    }
    return p == pattern.size();
}

void DrawEye(ImDrawList& draw, ImVec2 center, float size, bool shown, ImU32 color) {
    const float w = 0.36f * size;
    const float h = 0.22f * size;
    const float thickness = std::max(1.0f, 0.07f * size);
    draw.AddBezierQuadratic({center.x - w, center.y}, {center.x, center.y - 2 * h},
                            {center.x + w, center.y}, color, thickness);
    draw.AddBezierQuadratic({center.x - w, center.y}, {center.x, center.y + 2 * h},
                            {center.x + w, center.y}, color, thickness);
    if (shown) {
        draw.AddCircleFilled(center, 0.12f * size, color);
    } else {
        draw.AddLine({center.x - w, center.y + w}, {center.x + w, center.y - w}, color, thickness);
    }
}

class Stage {
  public:
    Stage(Ref<Scene> scene, Selection& selection, Editor& editor, const Theme& theme)
        : m_scene(std::move(scene)), m_selection(selection), m_editor(editor), m_theme(theme) {}

    void Draw() {
        if (!m_scene) {
            ImGui::TextDisabled("No scene");
            return;
        }
        Refresh();
        LandVisibility();
        if (!m_visibility && ImGui::GetTime() - m_visibilityRead >= 0.5) {
            RequestVisibility();
        }
        if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) && m_renaming.empty() &&
            m_selection.paths.size() == 1 && ImGui::IsKeyPressed(ImGuiKey_F2)) {
            StartRename(m_selection.paths.front());
        }
        Reveal();
        DrawSearch();
        constexpr ImGuiTableFlags kTable = ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY |
                                           ImGuiTableFlags_BordersInnerV |
                                           ImGuiTableFlags_Resizable;
        ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, {4.0f, 1.0f});
        if (ImGui::BeginTable("##stage", 3, kTable)) {
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn(
                "##visible", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoResize,
                ImGui::GetFrameHeight());
            ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed,
                                    7.0f * ImGui::GetFontSize());
            DrawHeaderRow();
            if (!Filtering()) {
                for (const uint32_t root : m_roots) {
                    DrawNode(root, 0, false);
                }
                if (m_session && SessionShown()) {
                    DrawNode(*m_session, 0, false);
                }
            } else {
                const bool session = SessionShown();
                for (uint32_t prim = 0; prim < m_prims.size(); ++prim) {
                    if ((session || !InSession(m_prims[prim].path)) && Matches(m_prims[prim])) {
                        DrawRow(prim, m_prims[prim].path, 0, false);
                    }
                }
            }
            ImGui::EndTable();
        }
        ImGui::PopStyleVar();
    }

  private:
    struct Entry {
        std::string path;
        std::string name;
        std::string type;
        PrimGroup group = PrimGroup::Other;
        std::vector<uint32_t> children;
        bool imageable = false;
        bool invisible = false;
    };

    void Refresh() {
        const uint64_t version = m_scene->StructureVersion();
        if (version == m_version) {
            return;
        }
        m_version = version;
        m_prims.clear();
        m_roots.clear();
        m_session.reset();
        const View roots = m_scene->Select("/*");
        for (const uint32_t root : roots.Prims()) {
            const Prim prim = m_scene->ByIndex(root);
            if (prim.IsSession()) {
                m_session = Add(prim);
            } else {
                m_roots.push_back(Add(prim));
            }
        }
        std::vector<std::string> imageable;
        for (const Entry& entry : m_prims) {
            if (entry.imageable) {
                imageable.push_back(entry.path);
            }
        }
        m_imageable = imageable.empty() ? View() : m_scene->Select(imageable, SchemaId::Imageable);
        RequestVisibility();
    }

    void RequestVisibility() {
        m_visibility.reset();
        m_visibilityRead = ImGui::GetTime();
        if (m_imageable.PrimCount() > 0) {
            m_visibility = m_imageable.Get(Imageable::kVisibility.Column()).ReadHostAsync();
        }
    }

    void LandVisibility() {
        if (!m_visibility || !m_visibility->Ready()) {
            return;
        }
        const Span<const std::byte> bytes = m_visibility->Bytes();
        size_t at = 0;
        for (Entry& entry : m_prims) {
            if (entry.imageable && (at + 1) * sizeof(uint32_t) <= bytes.size()) {
                uint32_t value = 0;
                std::memcpy(&value, bytes.data() + at * sizeof(uint32_t), sizeof(value));
                entry.invisible = value == uint32_t(Visibility::Invisible);
                ++at;
            }
        }
        m_visibility.reset();
    }

    uint32_t Add(const Prim& prim) {
        const auto index = uint32_t(m_prims.size());
        m_prims.push_back({prim.Path(),
                           std::string(prim.Name()),
                           std::string(Info(prim.Schema()).name),
                           Classify(prim.Schema()),
                           {},
                           prim.IsA(SchemaId::Imageable),
                           false});
        for (const Prim& child : prim.Children()) {
            const uint32_t added = Add(child);
            m_prims[index].children.push_back(added);
        }
        return index;
    }

    void DrawSearch() {
        ImGui::SetNextItemWidth(-ImGui::GetFrameHeight() - ImGui::GetStyle().ItemSpacing.x);
        ImGui::InputTextWithHint("##search", "Search, * and ? match any", &m_search);
        ImGui::SameLine();
        const bool filtered = Filtered();
        if (filtered) {
            ImGui::PushStyleColor(ImGuiCol_Button, m_theme.buttonLit);
        }
        const bool open = ImGui::Button(icons::kSettings);
        if (filtered) {
            ImGui::PopStyleColor();
        }
        ImGui::SetItemTooltip(filtered ? "Filters are on" : "Filters and options");
        if (open) {
            ImGui::OpenPopup("##stage options");
        }
        if (ImGui::BeginPopup("##stage options")) {
            if (ImGui::Selectable("Collapse All")) {
                m_expanded.clear();
            }
            ImGui::SeparatorText("Options");
            ImGui::MenuItem("Show Session Prims", nullptr, &m_showSession);
            ImGui::SetItemTooltip("Show the viewer's own cameras, product and render var under %s",
                                  std::string(kSessionRoot).c_str());
            DrawFilters();
            ImGui::EndPopup();
        }
    }

    void DrawFilters() {
        ImGui::SeparatorText("Filters");
        ImGui::PushItemFlag(ImGuiItemFlags_AutoClosePopups, false);
        if (ImGui::MenuItem("Show All", nullptr, false, Filtered())) {
            m_types.clear();
            m_groups.clear();
            m_shown = Shown::Any;
        }
        std::set<PrimGroup> groups;
        std::set<std::string> types;
        for (const Entry& entry : m_prims) {
            groups.insert(entry.group);
            types.insert(entry.type);
        }
        for (const PrimGroup group : groups) {
            const bool on = m_groups.contains(group);
            if (ImGui::MenuItem(kPrimGroupNames[size_t(group)], nullptr, on)) {
                if (on) {
                    m_groups.erase(group);
                } else {
                    m_groups.insert(group);
                }
            }
        }
        if (ImGui::BeginMenu(m_types.empty() ? "Types" : "Types (on)")) {
            for (const std::string& type : types) {
                const bool on = m_types.contains(type);
                if (ImGui::MenuItem(type.c_str(), nullptr, on)) {
                    if (on) {
                        m_types.erase(type);
                    } else {
                        m_types.insert(type);
                    }
                }
            }
            ImGui::EndMenu();
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Visible Only", nullptr, m_shown == Shown::Visible)) {
            m_shown = m_shown == Shown::Visible ? Shown::Any : Shown::Visible;
        }
        if (ImGui::MenuItem("Hidden Only", nullptr, m_shown == Shown::Hidden)) {
            m_shown = m_shown == Shown::Hidden ? Shown::Any : Shown::Hidden;
        }
        ImGui::PopItemFlag();
    }

    bool Filtered() const {
        return !m_types.empty() || !m_groups.empty() || m_shown != Shown::Any;
    }

    bool Filtering() const {
        return !m_search.empty() || Filtered();
    }

    bool Matches(const Entry& entry) const {
        if ((!m_types.empty() && !m_types.contains(entry.type)) ||
            (!m_groups.empty() && !m_groups.contains(entry.group)) ||
            (m_shown == Shown::Visible && (!entry.imageable || entry.invisible)) ||
            (m_shown == Shown::Hidden && !entry.invisible)) {
            return false;
        }
        const std::string search = Lowercase(m_search);
        if (search.find_first_of("*?") != std::string::npos) {
            return Glob(search, Lowercase(entry.path)) || Glob(search, Lowercase(entry.name));
        }
        return Lowercase(entry.path).find(search) != std::string::npos;
    }

    void DrawHeaderRow() {
        ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
        ImGui::TableSetColumnIndex(0);
        ImGui::TableHeader("Name");
        ImGui::TableSetColumnIndex(1);
        const ImVec2 corner = ImGui::GetCursorScreenPos();
        ImGui::TableHeader("##eye");
        const float size = ImGui::GetFontSize();
        DrawEye(*ImGui::GetWindowDrawList(),
                {corner.x + 0.5f * ImGui::GetFrameHeight() - 4.0f, corner.y + 0.5f * size}, size,
                true, ImGui::GetColorU32(m_theme.textMuted));
        ImGui::TableSetColumnIndex(2);
        ImGui::TableHeader("Type");
    }

    void DrawNode(uint32_t prim, int depth, bool hiddenAbove) {
        const Entry& entry = m_prims[prim];
        if (m_seen.insert(entry.path).second && depth == 0) {
            m_expanded.insert(entry.path);
        }
        DrawRow(prim, entry.name, depth, hiddenAbove);
        if (m_expanded.contains(entry.path)) {
            for (const uint32_t child : entry.children) {
                DrawNode(child, depth + 1, hiddenAbove || entry.invisible);
            }
        }
    }

    void DrawRow(uint32_t prim, const std::string& label, int depth, bool hiddenAbove) {
        const Entry& entry = m_prims[prim];
        const float height = ImGui::GetFrameHeight();
        ImGui::TableNextRow(ImGuiTableRowFlags_None, height);
        ImGui::TableSetColumnIndex(0);
        ImGui::PushID(entry.path.c_str());
        const ImVec2 start = ImGui::GetCursorScreenPos();
        if (ImGui::Selectable("##row", m_selection.Contains(entry.path),
                              ImGuiSelectableFlags_SpanAllColumns |
                                  ImGuiSelectableFlags_AllowOverlap,
                              {0, height})) {
            if (ImGui::GetIO().KeyCtrl) {
                m_selection.Toggle(entry.path);
            } else {
                m_selection.Set(entry.path);
            }
            m_revealed = m_selection.paths;
        }
        if (entry.path == m_scrollTo) {
            ImGui::SetScrollHereY(0.5f);
            m_scrollTo.clear();
        }
        if (ImGui::BeginDragDropSource()) {
            ImGui::SetDragDropPayload("ERNIE_PRIM", entry.path.c_str(), entry.path.size() + 1);
            ImGui::TextUnformatted(entry.name.c_str());
            ImGui::EndDragDropSource();
        }
        if (ImGui::BeginDragDropTarget()) {
            if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("ERNIE_PRIM")) {
                m_editor.Reparent(static_cast<const char*>(payload->Data), entry.path);
            }
            ImGui::EndDragDropTarget();
        }
        if (ImGui::BeginPopupContextItem("##prim")) {
            DrawContextMenu(prim);
            ImGui::EndPopup();
        }

        ImDrawList& draw = *ImGui::GetWindowDrawList();
        const float font = ImGui::GetFontSize();
        const float box = std::round(0.62f * font);
        float x = start.x + float(depth) * std::round(0.9f * font);
        const float middle = start.y + 0.5f * height;
        if (!entry.children.empty() && !Filtering()) {
            const ImVec2 low(x, middle - 0.5f * box);
            const ImVec2 high(low.x + box, low.y + box);
            ImGui::SetCursorScreenPos({x - 2.0f, start.y});
            if (ImGui::InvisibleButton("##toggle", {box + 4.0f, height})) {
                if (m_expanded.erase(entry.path) == 0) {
                    m_expanded.insert(entry.path);
                }
            }
            const ImU32 line =
                ImGui::GetColorU32(ImGui::IsItemHovered() ? m_theme.text : m_theme.textMuted);
            const float c = 0.5f * box;
            draw.AddRect(low, high, line, 1.0f);
            draw.AddLine({low.x + 2, low.y + c}, {high.x - 2, low.y + c}, line);
            if (!m_expanded.contains(entry.path)) {
                draw.AddLine({low.x + c, low.y + 2}, {low.x + c, high.y - 2}, line);
            }
        }
        x += box + 5.0f;
        const float icon = std::round(1.05f * font);
        DrawPrimIcon(draw, {x, middle - 0.5f * icon}, icon, entry.type);
        x += icon + 5.0f;
        const bool shown = !entry.invisible && !hiddenAbove;
        if (m_renaming == entry.path) {
            ImGui::SetCursorScreenPos({x - ImGui::GetStyle().FramePadding.x, start.y});
            ImGui::SetNextItemWidth(std::max(ImGui::GetContentRegionAvail().x, 6.0f * font));
            if (m_focusRename) {
                ImGui::SetKeyboardFocusHere();
                m_focusRename = false;
            }
            const bool done = ImGui::InputText("##rename", &m_name,
                                               ImGuiInputTextFlags_EnterReturnsTrue |
                                                   ImGuiInputTextFlags_AutoSelectAll);
            if (done) {
                m_editor.Rename(m_renaming, m_name);
            }
            if (done || ImGui::IsItemDeactivated() || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
                m_renaming.clear();
            }
        } else {
            draw.AddText({x, middle - 0.5f * font},
                         ImGui::GetColorU32(shown ? m_theme.text : m_theme.textMuted),
                         label.c_str());
        }

        ImGui::TableSetColumnIndex(1);
        if (entry.imageable) {
            DrawVisibility(prim, hiddenAbove);
        }
        ImGui::TableSetColumnIndex(2);
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("%s", entry.type.c_str());
        ImGui::PopID();
    }

    void DrawVisibility(uint32_t prim, bool hiddenAbove) {
        const Entry& entry = m_prims[prim];
        const float height = ImGui::GetFrameHeight();
        const ImVec2 corner = ImGui::GetCursorScreenPos();
        if (ImGui::InvisibleButton("##visibility", {height, height})) {
            ToggleVisibility(prim);
        }
        const bool hovered = ImGui::IsItemHovered();
        ImVec4 color = hovered ? m_theme.text : m_theme.textMuted;
        color.w = entry.invisible || hiddenAbove ? 0.45f : 0.9f;
        DrawEye(*ImGui::GetWindowDrawList(), {corner.x + 0.5f * height, corner.y + 0.5f * height},
                0.8f * height, !entry.invisible, ImGui::GetColorU32(color));
        if (hovered) {
            ImGui::SetTooltip(entry.invisible ? "Hidden; click to show"
                              : hiddenAbove   ? "Hidden by an ancestor; click to hide it too"
                                              : "Visible; click to hide");
        }
    }

    void DrawContextMenu(uint32_t prim) {
        const Entry& entry = m_prims[prim];
        if (!m_selection.Contains(entry.path)) {
            m_selection.Set(entry.path);
        }
        if (entry.imageable && ImGui::MenuItem(entry.invisible ? "Show" : "Hide")) {
            ToggleVisibility(prim);
        }
        if (ImGui::MenuItem("Expand All Below")) {
            Expand(prim);
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Rename", "F2", false, m_selection.paths.size() == 1)) {
            StartRename(entry.path);
        }
        if (ImGui::MenuItem("Duplicate", "Ctrl+D")) {
            m_editor.Duplicate();
        }
        if (ImGui::MenuItem("Move to Top Level", nullptr, false, entry.path.rfind('/') != 0)) {
            m_editor.Reparent(entry.path, "");
        }
        if (ImGui::MenuItem("Delete", "Del")) {
            m_editor.Delete();
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Copy Path")) {
            ImGui::SetClipboardText(entry.path.c_str());
        }
    }

    void ToggleVisibility(uint32_t prim) {
        Entry& entry = m_prims[prim];
        m_editor.SetVisible(entry.path, entry.invisible);
        entry.invisible = !entry.invisible;
        RequestVisibility();
    }

    void StartRename(const std::string& path) {
        m_renaming = path;
        m_name = path.substr(path.rfind('/') + 1);
        m_focusRename = true;
    }

    static bool InSession(std::string_view path) {
        return path.starts_with(kSessionRoot) &&
               (path.size() == kSessionRoot.size() || path[kSessionRoot.size()] == '/');
    }

    bool SessionShown() const {
        return m_showSession || std::ranges::any_of(m_selection.paths, [](const std::string& path) {
                   return InSession(path);
               });
    }

    void Reveal() {
        if (m_selection.paths == m_revealed) {
            return;
        }
        m_revealed = m_selection.paths;
        for (const std::string& path : m_revealed) {
            for (size_t slash = path.find('/', 1); slash != std::string::npos;
                 slash = path.find('/', slash + 1)) {
                m_expanded.insert(path.substr(0, slash));
            }
        }
        m_scrollTo = m_revealed.empty() ? std::string() : m_revealed.front();
    }

    void Expand(uint32_t prim) {
        m_expanded.insert(m_prims[prim].path);
        for (const uint32_t child : m_prims[prim].children) {
            Expand(child);
        }
    }

    Ref<Scene> m_scene;
    Selection& m_selection;
    Editor& m_editor;
    const Theme& m_theme;
    std::vector<Entry> m_prims;
    std::vector<uint32_t> m_roots;
    std::optional<uint32_t> m_session;
    bool m_showSession = false;
    std::vector<std::string> m_revealed;
    std::string m_scrollTo;
    uint64_t m_version = ~uint64_t(0);
    View m_imageable;
    std::optional<device::Download> m_visibility;
    double m_visibilityRead = -1;
    std::string m_renaming;
    std::string m_name;
    bool m_focusRename = false;
    std::set<std::string> m_expanded;
    std::set<std::string> m_seen;
    std::string m_search;
    std::set<std::string> m_types;
    std::set<PrimGroup> m_groups;
    Shown m_shown = Shown::Any;
};

}

Panel CreateStagePanel(Ref<Scene> scene, Selection& selection, Editor& editor, const Theme& theme) {
    auto stage = std::make_shared<Stage>(std::move(scene), selection, editor, theme);
    return {"Stage", Dock::Right, [stage] { stage->Draw(); }};
}

}
