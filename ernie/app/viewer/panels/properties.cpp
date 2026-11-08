// SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_stdlib.h>

#include "ernie/app/viewer/panels/panels.h"
#include "ernie/app/viewer/theme/theme.h"
#include "ernie/app/viewer/widgets/asset_path.h"
#include "ernie/app/viewer/widgets/widgets.h"
#include "ernie/scene/rotation.h"
#include "ernie/scene/schema/schema.h"
#include "ernie/util/log.h"
#include "ernie/util/span.h"

namespace ernie::app {
namespace {

using Clock = std::chrono::steady_clock;

constexpr auto kRefreshEvery = std::chrono::milliseconds(250);
constexpr size_t kShownItems = 16;
constexpr std::string_view kTransform = "Transform";
constexpr size_t kMostPickedPrims = 200;
constexpr const char* kAxisNames[] = {"X", "Y", "Z"};

const char* UnitSuffix(Unit unit) {
    switch (unit) {
    case Unit::Meters:
        return " m";
    case Unit::Millimeters:
        return " mm";
    case Unit::Degrees:
        return "\xc2\xb0";
    case Unit::Radians:
        return " rad";
    case Unit::Kelvin:
        return " K";
    case Unit::Seconds:
        return " s";
    case Unit::Nits:
        return " nt";
    case Unit::Lux:
        return " lx";
    case Unit::Watts:
        return " W";
    case Unit::Hertz:
        return " Hz";
    default:
        return "";
    }
}

template <class T> T Cell(const std::byte* bytes) {
    T value;
    std::memcpy(&value, bytes, sizeof(T));
    return value;
}

std::string Floats(const float* values, size_t count, Unit unit) {
    std::string text;
    for (size_t i = 0; i < count; ++i) {
        text += fmt::format("{}{:.4g}", i > 0 ? ", " : "", values[i]);
    }
    return text + UnitSuffix(unit);
}

std::string Doubles(const double* values, size_t count, Unit unit) {
    std::string text;
    for (size_t i = 0; i < count; ++i) {
        text += fmt::format("{}{:.6g}", i > 0 ? ", " : "", values[i]);
    }
    return text + UnitSuffix(unit);
}

std::string Format(const ColumnInfo& info, const char* precision) {
    return std::string(precision) + UnitSuffix(info.unit);
}

float Speed(const ColumnInfo& info, float value) {
    if (info.maximum > info.minimum) {
        return (info.maximum - info.minimum) / 500.0f;
    }
    return std::max(std::abs(value) * 0.01f, 0.001f);
}

struct Fallback {
    std::vector<std::byte> cell;
    std::optional<std::string> text;
    std::optional<std::vector<std::string>> targets;
};

class Fallbacks {
  public:
    const Fallback* Find(const ColumnRef& ref) {
        if (m_written.insert(ref.schema).second) {
            WriteFallbacks(ref.schema, *this);
        }
        const auto found = m_values.find({ref.schema, ref.index});
        return found == m_values.end() ? nullptr : &found->second;
    }

    template <class T> void Set(SchemaId owner, uint32_t column, const T& value) {
        Fallback& fallback = m_values[{owner, column}];
        if constexpr (std::is_same_v<T, std::vector<std::string>>) {
            fallback.targets = value;
        } else if constexpr (std::is_same_v<T, bool>) {
            const uint32_t word = value ? 1 : 0;
            fallback.cell.resize(sizeof(word));
            std::memcpy(fallback.cell.data(), &word, sizeof(word));
        } else if constexpr (std::is_convertible_v<const T&, std::string_view>) {
            fallback.text = std::string(std::string_view(value));
        } else if constexpr (requires {
                                 value.data();
                                 value.size();
                             }) {
            m_values.erase({owner, column});
        } else {
            static_assert(std::is_trivially_copyable_v<T>, "fallbacks are copied as bytes");
            fallback.cell.resize(sizeof(T));
            std::memcpy(fallback.cell.data(), &value, sizeof(T));
        }
    }

    template <class F> void Instance(SchemaId, std::string_view, const F&) {}

  private:
    std::set<SchemaId> m_written;
    std::map<std::pair<SchemaId, uint32_t>, Fallback> m_values;
};

bool Near(const std::byte* a, const std::byte* b, ColumnType type) {
    const auto close = [](double x, double y) {
        return std::abs(x - y) <= 1e-6 * std::max({1.0, std::abs(x), std::abs(y)});
    };
    const auto all = [&]<class T>(size_t count) {
        for (size_t at = 0; at < count; ++at) {
            if (!close(double(Cell<T>(a + at * sizeof(T))), double(Cell<T>(b + at * sizeof(T))))) {
                return false;
            }
        }
        return true;
    };
    switch (type) {
    case ColumnType::Float:
        return all.template operator()<float>(1);
    case ColumnType::Float2:
        return all.template operator()<float>(2);
    case ColumnType::Float3:
        return all.template operator()<float>(3);
    case ColumnType::Float4:
        return all.template operator()<float>(4);
    case ColumnType::Double3:
        return all.template operator()<double>(3);
    case ColumnType::Quatd:
        return all.template operator()<double>(4);
    default:
        return std::memcmp(a, b, ElementBytes(type)) == 0;
    }
}

bool DragAxes(ImGuiDataType type, void* values, float speed, const char* format) {
    const ImGuiStyle& style = ImGui::GetStyle();
    const float height = ImGui::GetFrameHeight();
    const float spacing = style.ItemInnerSpacing.x;
    const float tag = ImGui::CalcTextSize("X").x + 2 * style.FramePadding.x;
    const float each = (ImGui::GetContentRegionAvail().x - 2 * spacing) / 3;
    const size_t stride = type == ImGuiDataType_Double ? sizeof(double) : sizeof(float);
    bool changed = false;
    for (int axis = 0; axis < 3; ++axis) {
        if (axis > 0) {
            ImGui::SameLine(0, spacing);
        }
        ImGui::PushID(axis);
        const ImVec2 corner = ImGui::GetCursorScreenPos();
        ImDrawList& list = *ImGui::GetWindowDrawList();
        list.AddRectFilled(corner, {corner.x + tag, corner.y + height},
                           ImGui::GetColorU32(ImGui::ColorConvertU32ToFloat4(kAxisColors[axis])),
                           style.FrameRounding, ImDrawFlags_RoundCornersLeft);
        const ImVec2 letter = ImGui::CalcTextSize(kAxisNames[axis]);
        list.AddText({corner.x + 0.5f * (tag - letter.x), corner.y + 0.5f * (height - letter.y)},
                     ImGui::GetColorU32(ImVec4{1, 1, 1, 1}), kAxisNames[axis]);
        ImGui::Dummy({tag, height});
        ImGui::SameLine(0, 0);
        ImGui::SetNextItemWidth(std::max(each - tag, 1.0f));
        changed |=
            ImGui::DragScalar("##axis", type, static_cast<std::byte*>(values) + axis * stride,
                              speed, nullptr, nullptr, format);
        ImGui::PopID();
    }
    return changed;
}

class Properties {
  public:
    Properties(Ref<Scene> scene, Selection& selection, Editor& editor, FileDialog& dialog,
               const Theme& theme)
        : m_scene(std::move(scene)), m_selection(selection), m_editor(editor), m_dialog(dialog),
          m_theme(theme) {}

    void Draw() {
        if (!m_scene) {
            ImGui::TextDisabled("No scene");
            return;
        }
        Refresh();
        if (m_prims.empty()) {
            ImGui::TextDisabled("Select a prim to see its properties");
            return;
        }
        DrawHeader();
        ImGui::SetNextItemWidth(-1);
        ImGui::InputTextWithHint("##search", "Search properties", &m_search);
        if (ImGui::BeginChild("##fields")) {
            DrawSections();
        }
        ImGui::EndChild();
        if (m_live && !ImGui::IsAnyItemActive()) {
            Commit();
        }
    }

  private:
    struct Row {
        ColumnRef column;
        std::string instance;
        View view;
        std::string section;
        std::string label;
        std::vector<std::byte> cells;
    };

    struct Live {
        size_t row = 0;
        std::vector<std::byte> before;
        std::vector<std::byte> after;
        std::optional<Double3> euler;
        std::string label;
    };

    void Refresh() {
        const uint64_t structure = m_scene->StructureVersion();
        if (m_selection.paths != m_paths || structure != m_structure) {
            Commit();
            m_paths = m_selection.paths;
            m_structure = structure;
            Rebuild();
        }
        if (!m_reads.empty()) {
            if (!std::ranges::all_of(m_reads,
                                     [](const device::Download& read) { return read.Ready(); })) {
                return;
            }
            for (size_t row = 0; row < m_rows.size(); ++row) {
                if (!m_live || m_live->row != row) {
                    const Span<const std::byte> bytes = m_reads[row].Bytes();
                    m_rows[row].cells.assign(bytes.begin(), bytes.end());
                }
            }
            m_reads.clear();
        }
        if (!m_rows.empty() && Clock::now() - m_requested >= kRefreshEvery) {
            for (const Row& row : m_rows) {
                m_reads.push_back(row.view.Get(row.column).ReadHostAsync());
            }
            m_requested = Clock::now();
        }
    }

    void Rebuild() {
        m_prims.clear();
        m_alive.clear();
        m_rows.clear();
        m_reads.clear();
        m_items.clear();
        m_requested = {};
        for (const std::string& path : m_paths) {
            if (const std::optional<Prim> prim = m_scene->Find(path)) {
                m_prims.push_back(*prim);
                m_alive.push_back(path);
            }
        }
        if (m_prims.empty()) {
            return;
        }
        m_schema = m_prims.front().Schema();
        while (m_schema != SchemaId::None && !std::ranges::all_of(m_prims, [&](const Prim& prim) {
                   return prim.IsA(m_schema);
               })) {
            m_schema = Info(m_schema).base;
        }
        const View shared = m_scene->Select(Span<const std::string>(m_alive), m_schema);
        const auto add = [&](SchemaId owner, const View& view, std::string_view fallback,
                             const std::string& instance) {
            const Span<const ColumnInfo> columns = Columns(owner);
            for (uint32_t index = 0; index < columns.size(); ++index) {
                const ColumnInfo& info = columns[index];
                if (Has(info.flags, ColumnFlags::Internal)) {
                    continue;
                }
                const std::string_view label =
                    info.displayName.empty() ? info.field : info.displayName;
                const std::string_view section = info.group.empty() ? fallback : info.group;
                m_rows.push_back({{owner, &info, index},
                                  instance,
                                  view,
                                  instance.empty() ? std::string(section)
                                                   : fmt::format("{}:{}", section, instance),
                                  std::string(label),
                                  {}});
            }
        };
        for (SchemaId level = m_schema; level != SchemaId::None; level = Info(level).base) {
            add(level, shared, Info(level).name, {});
        }
        for (const SchemaId api : kApiSchemas) {
            if (!std::ranges::all_of(m_prims, [&](const Prim& prim) { return prim.HasApi(api); })) {
                continue;
            }
            if (!IsMultipleApi(api)) {
                add(api, shared, Info(api).name, {});
            } else if (m_prims.size() == 1) {
                for (const std::string& instance : m_prims.front().Instances(api)) {
                    add(api, m_scene->Select(m_prims.front(), api, instance), Info(api).name,
                        instance);
                }
            }
        }
        const auto rank = [](const Row& row) { return row.section == kTransform ? 0 : 1; };
        std::ranges::stable_sort(m_rows, {}, rank);
    }

    Editor::Target Target(const Row& row) const {
        if (!row.instance.empty()) {
            return [scene = m_scene, path = m_alive.front(), api = row.column.schema,
                    instance = row.instance] { return scene->Select(path, api, instance); };
        }
        return [scene = m_scene, paths = m_alive, schema = m_schema] {
            return scene->Select(Span<const std::string>(paths), schema);
        };
    }

    void Change(size_t index, const void* value) {
        Row& row = m_rows[index];
        const uint32_t cell = ElementBytes(row.column.column->type);
        if (m_live && m_live->row != index) {
            Commit();
        }
        if (!m_live) {
            m_live = Live{index, row.cells, {}, {}, {}};
        }
        const auto* bytes = static_cast<const std::byte*>(value);
        m_live->after.assign(bytes, bytes + cell);
        try {
            row.view.Get(row.column).Fill(m_live->after);
        } catch (const std::exception& error) {
            Log(LogLevel::Warning, "Setting {}: {}", row.label, error.what());
            m_live.reset();
            return;
        }
        for (size_t at = 0; at + cell <= row.cells.size(); at += cell) {
            std::memcpy(row.cells.data() + at, bytes, cell);
        }
        m_reads.clear();
        m_requested = Clock::now();
    }

    void Commit() {
        if (!m_live || m_live->row >= m_rows.size()) {
            m_live.reset();
            return;
        }
        const Row& row = m_rows[m_live->row];
        m_editor.SetCells(m_live->label.empty() ? "Set " + row.label : std::move(m_live->label),
                          Target(row), row.column, std::move(m_live->before),
                          std::move(m_live->after));
        m_live.reset();
    }

    Row* Find(const ColumnInfo* info, const std::string& instance) {
        const auto found = std::ranges::find_if(m_rows, [&](const Row& row) {
            return row.column.column == info && row.instance == instance;
        });
        return found == m_rows.end() ? nullptr : &*found;
    }

    bool Enabled(const Row& row) {
        const std::string_view by = row.column.column->enabledBy;
        if (by.empty()) {
            return true;
        }
        const auto found = std::ranges::find_if(m_rows, [&](const Row& other) {
            return other.column.schema == row.column.schema && other.instance == row.instance &&
                   other.column.column->field == by;
        });
        return found == m_rows.end() || found->cells.size() < 4 ||
               Cell<uint32_t>(found->cells.data()) != 0;
    }

    void DrawHeader() {
        const Prim& prim = m_prims.front();
        const std::string_view type = Info(m_schema).name;
        const float size = ImGui::GetFrameHeight();
        const ImVec2 corner = ImGui::GetCursorScreenPos();
        DrawPrimIcon(*ImGui::GetWindowDrawList(), corner, size, type);
        ImGui::Dummy({size, size});
        ImGui::SameLine();
        ImGui::AlignTextToFramePadding();
        if (m_prims.size() == 1) {
            ImGui::TextUnformatted(std::string(prim.Name()).c_str());
            ImGui::SameLine();
            ImGui::TextDisabled("%.*s", int(type.size()), type.data());
            const std::string path = prim.Path();
            ImGui::TextDisabled("%s", path.c_str());
            if (ImGui::IsItemClicked()) {
                ImGui::SetClipboardText(path.c_str());
            }
            ImGui::SetItemTooltip("Click to copy the path");
        } else {
            ImGui::Text("%zu prims", m_prims.size());
            ImGui::SameLine();
            ImGui::TextDisabled("%.*s fields they share", int(type.size()), type.data());
        }
    }

    void DrawSections() {
        const std::string search = Lowercase(m_search);
        std::vector<std::string> sections;
        for (const Row& row : m_rows) {
            if (std::ranges::find(sections, row.section) == sections.end()) {
                sections.push_back(row.section);
            }
        }
        constexpr ImGuiTableFlags kTable = ImGuiTableFlags_SizingStretchProp |
                                           ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV;
        for (const std::string& section : sections) {
            const auto matches = [&](const Row& row) {
                return row.section == section &&
                       (search.empty() || Lowercase(row.label).find(search) != std::string::npos ||
                        Lowercase(row.column.column->field).find(search) != std::string::npos);
            };
            if (std::ranges::none_of(m_rows, matches)) {
                continue;
            }
            if (!ImGui::CollapsingHeader(section.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
                continue;
            }
            if (!ImGui::BeginTable(section.c_str(), 3, kTable)) {
                continue;
            }
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch, 0.4f);
            ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch, 0.6f);
            ImGui::TableSetupColumn("Reset", ImGuiTableColumnFlags_WidthFixed,
                                    ImGui::GetFrameHeight());
            for (size_t index = 0; index < m_rows.size(); ++index) {
                if (matches(m_rows[index])) {
                    DrawRow(index);
                }
            }
            ImGui::EndTable();
        }
    }

    void DrawRow(size_t index) {
        const Row& row = m_rows[index];
        const ColumnInfo& info = *row.column.column;
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(row.label.c_str());
        if (ImGui::BeginItemTooltip()) {
            ImGui::TextUnformatted(std::string(info.field).c_str());
            if (!info.doc.empty()) {
                ImGui::PushTextWrapPos(ImGui::GetFontSize() * 24);
                ImGui::TextDisabled("%.*s", int(info.doc.size()), info.doc.data());
                ImGui::PopTextWrapPos();
            }
            ImGui::EndTooltip();
        }
        ImGui::TableNextColumn();
        ImGui::PushID(&info);
        ImGui::PushID(row.instance.c_str());
        const uint32_t cell = ElementBytes(info.type);
        if (row.cells.size() < cell) {
            ImGui::TextDisabled("...");
        } else {
            const bool mixed = !SameAcrossPrims(row, cell);
            const bool enabled = Enabled(row);
            ImGui::BeginDisabled(!enabled);
            ImGui::PushItemFlag(ImGuiItemFlags_MixedValue, mixed);
            DrawEditor(index, row.cells.data(), mixed);
            ImGui::PopItemFlag();
            ImGui::EndDisabled();
        }
        ImGui::TableNextColumn();
        if (row.cells.size() >= cell) {
            DrawReset(index);
        }
        ImGui::PopID();
        ImGui::PopID();
    }

    std::optional<bool> Authored(const Row& row) {
        const Fallback* fallback = m_fallbacks.Find(row.column);
        if (!fallback) {
            return std::nullopt;
        }
        const ColumnInfo& info = *row.column.column;
        const uint32_t cell = ElementBytes(info.type);
        for (size_t at = 0; at + cell <= row.cells.size(); at += cell) {
            const std::byte* bytes = row.cells.data() + at;
            if (fallback->text) {
                if (m_scene->Text(Token{Cell<uint32_t>(bytes)}) != *fallback->text) {
                    return true;
                }
            } else if (fallback->targets) {
                if (m_prims.size() != 1 || !fallback->targets->empty()) {
                    return std::nullopt;
                }
                if (Cell<ArrayRef>(bytes).count != 0) {
                    return true;
                }
            } else if (fallback->cell.size() != cell) {
                return std::nullopt;
            } else if (!Near(bytes, fallback->cell.data(), info.type)) {
                return true;
            }
        }
        return false;
    }

    void DrawReset(size_t index) {
        const Row& row = m_rows[index];
        const std::optional<bool> authored = Authored(row);
        if (!authored || !Enabled(row)) {
            return;
        }
        const float size = ImGui::GetFrameHeight();
        const ImVec2 corner = ImGui::GetCursorScreenPos();
        const bool clicked = ImGui::InvisibleButton("##reset", {size, size}) && *authored;
        const ImVec2 centre{corner.x + 0.5f * size, corner.y + 0.5f * size};
        ImDrawList& list = *ImGui::GetWindowDrawList();
        if (*authored) {
            const ImVec4 color = ImGui::IsItemHovered() ? m_theme.selection : m_theme.accent;
            list.AddRectFilled({centre.x - 0.2f * size, centre.y - 0.2f * size},
                               {centre.x + 0.2f * size, centre.y + 0.2f * size},
                               ImGui::GetColorU32(color), 0.05f * size);
            ImGui::SetItemTooltip("Reset to the default value");
        } else {
            list.AddRect({centre.x - 0.16f * size, centre.y - 0.16f * size},
                         {centre.x + 0.16f * size, centre.y + 0.16f * size},
                         ImGui::GetColorU32(m_theme.textMuted), 0.05f * size);
            ImGui::SetItemTooltip("The default value");
        }
        if (clicked) {
            Reset(index);
        }
    }

    void Reset(size_t index) {
        const Fallback* fallback = m_fallbacks.Find(m_rows[index].column);
        if (!fallback) {
            return;
        }
        Commit();
        Row& row = m_rows[index];
        if (fallback->targets) {
            const std::vector<std::string> targets = row.view.Get(row.column).GetTargets(0);
            m_editor.SetTargets("Reset " + row.label, Target(row), row.column, {targets},
                                *fallback->targets);
            m_items.clear();
            return;
        }
        if (fallback->text) {
            const Token token = m_scene->Token(*fallback->text);
            Change(index, &token);
        } else {
            Change(index, fallback->cell.data());
        }
        if (m_live) {
            m_live->label = "Reset " + row.label;
        }
        Commit();
        m_texts.clear();
    }

    static bool SameAcrossPrims(const Row& row, uint32_t cell) {
        for (size_t at = cell; at + cell <= row.cells.size(); at += cell) {
            if (std::memcmp(row.cells.data(), row.cells.data() + at, cell) != 0) {
                return false;
            }
        }
        return true;
    }

    void DrawEditor(size_t index, const std::byte* bytes, bool mixed) {
        const ColumnInfo& info = *m_rows[index].column.column;
        const bool ranged = info.maximum > info.minimum;
        ImGui::SetNextItemWidth(-1);
        switch (info.type) {
        case ColumnType::Bool: {
            bool value = Cell<uint32_t>(bytes) != 0;
            if (ImGui::Checkbox("##value", &value)) {
                const uint32_t word = value ? 1 : 0;
                Change(index, &word);
            }
            return;
        }
        case ColumnType::Int: {
            int32_t value = Cell<int32_t>(bytes);
            const int low = ranged ? int(info.minimum) : 0;
            const int high = ranged ? int(info.maximum) : 0;
            if (ImGui::DragInt("##value", &value, 0.2f, low, high, Format(info, "%d").c_str())) {
                Change(index, &value);
            }
            return;
        }
        case ColumnType::Uint: {
            uint32_t value = Cell<uint32_t>(bytes);
            const uint32_t low = ranged ? uint32_t(info.minimum) : 0;
            const uint32_t high = ranged ? uint32_t(info.maximum) : UINT32_MAX;
            if (ImGui::DragScalar("##value", ImGuiDataType_U32, &value, 0.2f, &low, &high,
                                  Format(info, "%u").c_str())) {
                Change(index, &value);
            }
            return;
        }
        case ColumnType::Uint64: {
            uint64_t value = Cell<uint64_t>(bytes);
            if (ImGui::InputScalar("##value", ImGuiDataType_U64, &value)) {
                Change(index, &value);
            }
            return;
        }
        case ColumnType::Uint2: {
            Uint2 value = Cell<Uint2>(bytes);
            const uint32_t low = 0;
            if (ImGui::DragScalarN("##value", ImGuiDataType_U32, &value.x, 2, 0.2f, &low)) {
                Change(index, &value);
            }
            return;
        }
        case ColumnType::Float: {
            float value = Cell<float>(bytes);
            if (ImGui::DragFloat("##value", &value, Speed(info, value), ranged ? info.minimum : 0,
                                 ranged ? info.maximum : 0, Format(info, "%.4g").c_str())) {
                Change(index, &value);
            }
            return;
        }
        case ColumnType::Float2: {
            Float2 value = Cell<Float2>(bytes);
            if (ImGui::DragFloat2("##value", &value.x, Speed(info, value.x),
                                  ranged ? info.minimum : 0, ranged ? info.maximum : 0,
                                  Format(info, "%.4g").c_str())) {
                Change(index, &value);
            }
            return;
        }
        case ColumnType::Float3: {
            Float3 value = Cell<Float3>(bytes);
            const bool changed =
                info.role == ColumnRole::Color
                    ? ImGui::ColorEdit3("##value", &value.x,
                                        ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR)
                : ranged
                    ? ImGui::DragFloat3("##value", &value.x, Speed(info, value.x), info.minimum,
                                        info.maximum, Format(info, "%.4g").c_str())
                    : DragAxes(ImGuiDataType_Float, &value.x,
                               Speed(info, std::max({std::abs(value.x), std::abs(value.y),
                                                     std::abs(value.z)})),
                               Format(info, "%.4g").c_str());
            if (changed) {
                Change(index, &value);
            }
            return;
        }
        case ColumnType::Float4: {
            Float4 value = Cell<Float4>(bytes);
            const bool changed =
                info.role == ColumnRole::Color
                    ? ImGui::ColorEdit4("##value", &value.x,
                                        ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR)
                    : ImGui::DragFloat4("##value", &value.x, Speed(info, value.x),
                                        ranged ? info.minimum : 0, ranged ? info.maximum : 0,
                                        Format(info, "%.4g").c_str());
            if (changed) {
                Change(index, &value);
            }
            return;
        }
        case ColumnType::Double3: {
            Double3 value = Cell<Double3>(bytes);
            const float speed = Speed(
                info, float(std::max({std::abs(value.x), std::abs(value.y), std::abs(value.z)})));
            if (DragAxes(ImGuiDataType_Double, &value.x, speed, Format(info, "%.6g").c_str())) {
                Change(index, &value);
            }
            return;
        }
        case ColumnType::Quatd: {
            const bool live = m_live && m_live->row == index && m_live->euler;
            Double3 euler = live ? *m_live->euler : ToEulerDegrees(Cell<Quatd>(bytes));
            if (DragAxes(ImGuiDataType_Double, &euler.x, 0.5f, "%.2f\xc2\xb0")) {
                const Quatd value = FromEulerDegrees(euler);
                Change(index, &value);
                if (m_live) {
                    m_live->euler = euler;
                }
            }
            ImGui::SetItemTooltip("Turns about x, y and z in degrees, applied in that order; "
                                  "stored as an xyzw quaternion");
            return;
        }
        case ColumnType::Enum: {
            const uint32_t value = Cell<uint32_t>(bytes);
            const std::string preview = mixed ? "Mixed"
                                        : value < info.tokens.size()
                                            ? std::string(info.tokens[value])
                                            : std::to_string(value);
            if (ImGui::BeginCombo("##value", preview.c_str())) {
                for (uint32_t option = 0; option < info.tokens.size(); ++option) {
                    const std::string label(info.tokens[option]);
                    if (ImGui::Selectable(label.c_str(), option == value && !mixed)) {
                        Change(index, &option);
                    }
                }
                ImGui::EndCombo();
            }
            return;
        }
        case ColumnType::Token:
        case ColumnType::Asset:
            DrawText(index, bytes);
            return;
        case ColumnType::PrimArray:
            DrawTargets(index, Cell<ArrayRef>(bytes).count, mixed);
            return;
        default:
            DrawReadOnly(m_rows[index], bytes);
            return;
        }
    }

    void DrawText(size_t index, const std::byte* bytes) {
        const Row& row = m_rows[index];
        const ColumnInfo* info = row.column.column;
        std::string& text = m_texts[fmt::format("{}:{}", info->field, row.instance)];
        if (!ImGui::IsAnyItemActive()) {
            text = std::string(m_scene->Text(Token{Cell<uint32_t>(bytes)}));
        }
        if (info->type == ColumnType::Token) {
            ImGui::InputTextWithHint("##value", "none", &text);
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                const Token token = m_scene->Token(text);
                Change(index, &token);
                Commit();
            }
            return;
        }
        const auto chosen = [this, info,
                             instance = row.instance](const std::filesystem::path& file) {
            try {
                const Token token = m_scene->Token(AddAsset(*m_scene, file));
                if (const Row* found = Find(info, instance)) {
                    Change(size_t(found - m_rows.data()), &token);
                    Commit();
                }
            } catch (const std::exception& error) {
                Log(LogLevel::Warning, "Loading {}: {}", file.string(), error.what());
            }
        };
        if (DrawAssetPath("##value", text, m_dialog, AssetKind::Image, chosen)) {
            const Token token = m_scene->Token(text);
            Change(index, &token);
            Commit();
        }
    }

    void DrawTargets(size_t index, uint32_t count, bool mixed) {
        const Row& row = m_rows[index];
        if (m_prims.size() != 1 || mixed) {
            DrawReadOnly(row, row.cells.data());
            return;
        }
        const std::string key = fmt::format("{}:{}", row.column.column->field, row.instance);
        auto found = m_items.find(key);
        if (found == m_items.end() || found->second.size() != count) {
            found = m_items.insert_or_assign(key, row.view.Get(row.column).GetTargets(0)).first;
        }
        const std::vector<std::string> targets = found->second;
        std::optional<std::vector<std::string>> after;
        for (size_t at = 0; at < targets.size(); ++at) {
            ImGui::PushID(int(at));
            if (ImGui::SmallButton(icons::kX)) {
                after = targets;
                after->erase(after->begin() + std::ptrdiff_t(at));
            }
            ImGui::SetItemTooltip("Remove the target");
            ImGui::SameLine();
            ImGui::TextUnformatted(targets[at].c_str());
            if (ImGui::IsItemClicked()) {
                m_selection.paths = {targets[at]};
            }
            ImGui::SetItemTooltip("Click to select it");
            ImGui::PopID();
        }
        const float button = ImGui::GetFrameHeight();
        ImGui::SetNextItemWidth(std::max(
            ImGui::GetContentRegionAvail().x - button - ImGui::GetStyle().ItemSpacing.x, 1.0f));
        std::string& adding = m_texts[key];
        if (ImGui::InputTextWithHint("##add", "Add a path, or drop prims here", &adding,
                                     ImGuiInputTextFlags_EnterReturnsTrue) &&
            !adding.empty()) {
            after = targets;
            after->push_back(std::exchange(adding, {}));
        }
        if (ImGui::BeginDragDropTarget()) {
            if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("ERNIE_PRIM")) {
                after = targets;
                after->push_back(static_cast<const char*>(payload->Data));
            }
            ImGui::EndDragDropTarget();
        }
        ImGui::SameLine();
        if (ImGui::Button(icons::kListTree, {button, button})) {
            m_pickSearch.clear();
            ImGui::OpenPopup("##pick");
        }
        ImGui::SetItemTooltip("Choose a prim from the stage");
        if (ImGui::BeginPopup("##pick")) {
            if (std::optional<std::string> picked = DrawPicker(targets)) {
                after = targets;
                after->push_back(std::move(*picked));
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
        if (after) {
            Commit();
            m_editor.SetTargets("Set " + row.label, Target(row), row.column, {targets},
                                std::move(*after));
            m_items.erase(key);
        }
    }

    std::optional<std::string> DrawPicker(const std::vector<std::string>& targets) {
        if (ImGui::IsWindowAppearing()) {
            ImGui::SetKeyboardFocusHere();
        }
        ImGui::SetNextItemWidth(22 * ImGui::GetFontSize());
        ImGui::InputTextWithHint("##find", "Search prims", &m_pickSearch);
        std::vector<Prim> roots;
        const View tops = m_scene->Select("/*");
        for (const uint32_t top : tops.Prims()) {
            const Prim prim = m_scene->ByIndex(top);
            if (!prim.IsSession()) {
                roots.push_back(prim);
            }
        }
        std::optional<std::string> picked;
        if (ImGui::BeginChild("##prims", {22 * ImGui::GetFontSize(), 18 * ImGui::GetFontSize()},
                              ImGuiChildFlags_Borders)) {
            const std::string search = Lowercase(m_pickSearch);
            if (search.empty()) {
                for (const Prim& root : roots) {
                    DrawPickerNode(root, targets, picked);
                }
            } else {
                size_t listed = 0;
                std::vector<Prim> stack(roots.rbegin(), roots.rend());
                while (!stack.empty() && listed < kMostPickedPrims) {
                    const Prim prim = stack.back();
                    stack.pop_back();
                    const std::string path = prim.Path();
                    if (Lowercase(path).find(search) != std::string::npos) {
                        ++listed;
                        DrawPickerRow(prim, path, targets, picked, true);
                    }
                    const std::vector<Prim> children = prim.Children();
                    stack.insert(stack.end(), children.rbegin(), children.rend());
                }
                if (listed == kMostPickedPrims) {
                    ImGui::TextDisabled("More match; search for more of the path");
                }
            }
        }
        ImGui::EndChild();
        return picked;
    }

    void DrawPickerNode(const Prim& prim, const std::vector<std::string>& targets,
                        std::optional<std::string>& picked) {
        const std::string path = prim.Path();
        if (DrawPickerRow(prim, path, targets, picked, false)) {
            for (const Prim& child : prim.Children()) {
                DrawPickerNode(child, targets, picked);
            }
            ImGui::TreePop();
        }
    }

    bool DrawPickerRow(const Prim& prim, const std::string& path,
                       const std::vector<std::string>& targets, std::optional<std::string>& picked,
                       bool flat) {
        const bool leaf = flat || prim.Children().empty();
        const bool taken = std::ranges::find(targets, path) != targets.end();
        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow |
                                   ImGuiTreeNodeFlags_OpenOnDoubleClick |
                                   ImGuiTreeNodeFlags_SpanAvailWidth;
        if (leaf) {
            flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
        }
        const std::string_view type = Info(prim.Schema()).name;
        ImGui::BeginDisabled(taken);
        const bool open = ImGui::TreeNodeEx(path.c_str(), flags, "%s",
                                            flat ? path.c_str() : std::string(prim.Name()).c_str());
        if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen() && !taken) {
            picked = path;
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::TextDisabled("%.*s", int(type.size()), type.data());
        return open && !leaf;
    }

    void DrawReadOnly(const Row& row, const std::byte* bytes) {
        const ColumnInfo& info = *row.column.column;
        switch (info.type) {
        case ColumnType::Bool:
            Text(Cell<uint32_t>(bytes) != 0 ? "true" : "false");
            return;
        case ColumnType::Int:
            Text(fmt::format("{}{}", Cell<int32_t>(bytes), UnitSuffix(info.unit)));
            return;
        case ColumnType::Uint:
            Text(fmt::format("{}{}", Cell<uint32_t>(bytes), UnitSuffix(info.unit)));
            return;
        case ColumnType::Uint64:
            Text(fmt::format("{}", Cell<uint64_t>(bytes)));
            return;
        case ColumnType::Uint2: {
            const Uint2 value = Cell<Uint2>(bytes);
            Text(fmt::format("{}, {}", value.x, value.y));
            return;
        }
        case ColumnType::Float:
            Text(Floats(reinterpret_cast<const float*>(bytes), 1, info.unit));
            return;
        case ColumnType::Float2:
            Text(Floats(reinterpret_cast<const float*>(bytes), 2, info.unit));
            return;
        case ColumnType::Float3:
            Text(Floats(reinterpret_cast<const float*>(bytes), 3, info.unit));
            return;
        case ColumnType::Float4:
            Text(Floats(reinterpret_cast<const float*>(bytes), 4, info.unit));
            return;
        case ColumnType::Double3:
            Text(Doubles(reinterpret_cast<const double*>(bytes), 3, info.unit));
            return;
        case ColumnType::Quatd:
            Text(Doubles(reinterpret_cast<const double*>(bytes), 4, Unit::None));
            ImGui::SetItemTooltip("xyzw");
            return;
        case ColumnType::WorldTransform: {
            const float* words = reinterpret_cast<const float*>(bytes);
            const double translation[] = {double(words[9]) + words[12],
                                          double(words[10]) + words[13],
                                          double(words[11]) + words[14]};
            Text(Doubles(translation, 3, Unit::None));
            ImGui::SetItemTooltip("World position");
            return;
        }
        case ColumnType::Enum: {
            const uint32_t value = Cell<uint32_t>(bytes);
            Text(value < info.tokens.size() ? std::string(info.tokens[value])
                                            : std::to_string(value));
            return;
        }
        case ColumnType::Token:
        case ColumnType::Asset: {
            const std::string_view text = m_scene->Text(Token{Cell<uint32_t>(bytes)});
            if (text.empty()) {
                ImGui::TextDisabled("none");
            } else {
                Text(std::string(text));
            }
            return;
        }
        default:
            DrawArray(row, Cell<ArrayRef>(bytes).count);
            return;
        }
    }

    void DrawArray(const Row& row, uint32_t count) {
        const ColumnInfo& info = *row.column.column;
        const char* noun = info.type == ColumnType::PrimArray ? "target" : "item";
        if (count == 0) {
            ImGui::TextDisabled("no %ss", noun);
            return;
        }
        const std::string key = fmt::format("{}:{}", info.field, row.instance);
        const auto found = m_items.find(key);
        if (found == m_items.end()) {
            ImGui::Text("%u %s%s", count, noun, count == 1 ? "" : "s");
            ImGui::SameLine();
            if (ImGui::SmallButton("Show")) {
                m_items[key] = ReadItems(row, count);
            }
            return;
        }
        for (const std::string& line : found->second) {
            Text(line);
        }
        if (count > found->second.size()) {
            ImGui::TextDisabled("and %zu more", count - found->second.size());
        }
    }

    std::vector<std::string> ReadItems(const Row& row, uint32_t count) const {
        const ColumnInfo& info = *row.column.column;
        const UntypedAttribute attribute = row.view.Get(row.column);
        std::vector<std::string> lines;
        if (info.type == ColumnType::PrimArray) {
            lines = attribute.GetTargets(0);
        } else if (info.type == ColumnType::TokenArray) {
            lines = attribute.GetTokens(0);
        } else if (info.type == ColumnType::IntArray) {
            const std::vector<std::byte> bytes = attribute.GetArray(0);
            for (size_t at = 0; at + 4 <= bytes.size() && lines.size() < kShownItems; at += 4) {
                lines.push_back(std::to_string(Cell<int32_t>(bytes.data() + at)));
            }
        } else if (info.type == ColumnType::FloatArray || info.type == ColumnType::Float2Array ||
                   info.type == ColumnType::Float3Array || info.type == ColumnType::Float4Array) {
            const size_t width = info.type == ColumnType::FloatArray    ? 1
                                 : info.type == ColumnType::Float2Array ? 2
                                 : info.type == ColumnType::Float3Array ? 3
                                                                        : 4;
            const std::vector<std::byte> bytes = attribute.GetArray(0);
            for (size_t at = 0; at + width * 4 <= bytes.size() && lines.size() < kShownItems;
                 at += width * 4) {
                lines.push_back(
                    Floats(reinterpret_cast<const float*>(bytes.data() + at), width, info.unit));
            }
        } else {
            lines.push_back(fmt::format("{} items", count));
        }
        if (lines.size() > kShownItems) {
            lines.resize(kShownItems);
        }
        return lines;
    }

    static void Text(const std::string& text) {
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(text.c_str());
        if (ImGui::IsItemClicked()) {
            ImGui::SetClipboardText(text.c_str());
        }
    }

    Ref<Scene> m_scene;
    Selection& m_selection;
    Editor& m_editor;
    FileDialog& m_dialog;
    const Theme& m_theme;
    std::vector<std::string> m_paths;
    std::vector<std::string> m_alive;
    uint64_t m_structure = ~uint64_t(0);
    std::vector<Prim> m_prims;
    SchemaId m_schema = SchemaId::None;
    std::vector<Row> m_rows;
    std::vector<device::Download> m_reads;
    Clock::time_point m_requested;
    std::map<std::string, std::vector<std::string>> m_items;
    std::map<std::string, std::string> m_texts;
    std::optional<Live> m_live;
    std::string m_search;
    std::string m_pickSearch;
    Fallbacks m_fallbacks;
};

}

Panel CreatePropertiesPanel(Ref<Scene> scene, Selection& selection, Editor& editor,
                            FileDialog& dialog, const Theme& theme) {
    auto properties =
        std::make_shared<Properties>(std::move(scene), selection, editor, dialog, theme);
    return {"Properties", Dock::RightBottom, [properties] { properties->Draw(); }};
}

}
