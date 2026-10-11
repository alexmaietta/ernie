// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/property_fields.h"

#include "ernie/app/session.h"
#include "ernie/app/viewer.h"
#include "ernie/scene/schema.h"

#include <glm/gtc/quaternion.hpp>
#include <imgui.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>

namespace ernie::app {

namespace {

double NumberOf(const Value& value) {
    if (const auto* d = value.Get<double>()) {
        return *d;
    }
    if (const auto* f = value.Get<float>()) {
        return *f;
    }
    if (const auto* i = value.Get<int>()) {
        return *i;
    }
    if (const auto* b = value.Get<bool>()) {
        return *b ? 1.0 : 0.0;
    }
    return 0.0;
}

std::string TextOf(const Value& value) {
    const auto* text = value.Get<std::string>();
    return text != nullptr ? *text : std::string();
}

glm::dvec3 Vec3Of(const Value& value) {
    if (const auto* v = value.Get<glm::vec3>()) {
        return glm::dvec3(*v);
    }
    if (const auto* d = value.Get<glm::dvec3>()) {
        return *d;
    }
    return glm::dvec3(0.0);
}

glm::dvec4 Vec4Of(const Value& value) {
    if (const auto* v = value.Get<glm::vec4>()) {
        return glm::dvec4(*v);
    }
    if (const auto* d = value.Get<glm::dvec4>()) {
        return *d;
    }
    return glm::dvec4(0.0);
}

glm::dvec2 Vec2Of(const Value& value) {
    if (const auto* v = value.Get<glm::vec2>()) {
        return glm::dvec2(*v);
    }
    if (const auto* d = value.Get<glm::dvec2>()) {
        return *d;
    }
    if (const auto* i = value.Get<glm::ivec2>()) {
        return glm::dvec2(*i);
    }
    return glm::dvec2(0.0);
}

Path TargetOf(const Value& value) {
    if (const auto* path = value.Get<Path>()) {
        return *path;
    }
    if (const auto* paths = value.Get<std::vector<Path>>(); paths != nullptr && !paths->empty()) {
        return paths->front();
    }
    return {};
}

/// The value a field shows: the authored one, read the way Ernie reads it, or the fallback.
Value Shown(const Scene& scene, const Path& path, const PropertyField& field) {
    using Kind = PropertyField::Kind;
    const Value* authored = scene.Get(path, field.attribute);
    if (authored == nullptr) {
        return field.fallback;
    }
    switch (field.kind) {
    case Kind::Bool:
        return GetBool(scene, path, field.attribute, NumberOf(field.fallback) != 0.0);
    case Kind::Int:
        return static_cast<int>(GetNumber(scene, path, field.attribute, NumberOf(field.fallback)));
    case Kind::Float:
        return GetNumber(scene, path, field.attribute, NumberOf(field.fallback));
    case Kind::Token:
    case Kind::Text:
        return GetToken(scene, path, field.attribute, TextOf(field.fallback));
    case Kind::Color:
        return GetVec3(scene, path, field.attribute, Vec3Of(field.fallback));
    case Kind::Color4:
        return authored->Get<glm::vec4>() != nullptr || authored->Get<glm::dvec4>() != nullptr
                   ? Value(Vec4Of(*authored))
                   : field.fallback;
    case Kind::Vec2:
    case Kind::Int2:
        return Vec2Of(*authored);
    case Kind::Vec3:
        return GetVec3(scene, path, field.attribute, Vec3Of(field.fallback));
    case Kind::Vec4:
        return authored->Get<glm::vec4>() != nullptr || authored->Get<glm::dvec4>() != nullptr
                   ? Value(Vec4Of(*authored))
                   : field.fallback;
    case Kind::Quat:
        return authored->Get<glm::dquat>() != nullptr ? *authored : field.fallback;
    case Kind::Matrix:
        return authored->Get<glm::dmat4>() != nullptr ? *authored : field.fallback;
    case Kind::Target:
        return TargetOf(*authored);
    case Kind::Curve:
        if (authored->Get<std::vector<float>>() != nullptr) {
            return *authored;
        }
        // A written array when it is a flat CPU float32 one; edits write a float array value.
        if (const Tensor* tensor = scene.Read(path, field.attribute);
            tensor != nullptr && tensor->Location().kind == DeviceKind::Cpu &&
            tensor->Rank() == 1 && tensor->Type() == DType::Float32) {
            const Tensor packed = tensor->Contiguous();
            const auto count = static_cast<std::size_t>(packed.Elements());
            return std::vector<float>(packed.As<float>(), packed.As<float>() + count);
        }
        return field.fallback;
    }
    return field.fallback;
}

/// A curve editor for `values`, `height` pixels tall over the available width: the samples as a
/// line over [0, 1], set to the pointer's height where it drags across them. The value range is
/// `min` to `max` when they differ, else 0 to the larger of 1 and the curve's peak, kept while a
/// drag lasts so the curve does not rescale under the pointer. Returns whether `values` changed.
bool CurveEditor(std::vector<float>& values, double min, double max, float height) {
    const float width = std::max(ImGui::GetContentRegionAvail().x, 40.0f);
    const ImVec2 low = ImGui::GetCursorScreenPos();
    const ImVec2 high(low.x + width, low.y + height);
    ImGui::InvisibleButton("##curve", ImVec2(width, height));
    const bool active = ImGui::IsItemActive();
    ImGuiStorage& storage = *ImGui::GetStateStorage();
    const ImGuiID rangeId = ImGui::GetItemID();
    float lo = static_cast<float>(min);
    float hi = static_cast<float>(max);
    if (!(max > min)) {
        lo =
            std::min(0.0f, values.empty() ? 0.0f : *std::min_element(values.begin(), values.end()));
        hi =
            std::max(1.0f, values.empty() ? 1.0f : *std::max_element(values.begin(), values.end()));
        if (active && ImGui::IsItemActivated()) {
            storage.SetFloat(rangeId, hi);
        } else if (active) {
            hi = storage.GetFloat(rangeId, hi);
        }
    }
    const auto toScreen = [&](std::size_t i, float v) {
        const float t = values.size() > 1 ? float(i) / float(values.size() - 1) : 0.5f;
        return ImVec2(low.x + t * width, high.y - (v - lo) / (hi - lo) * height);
    };
    ImDrawList* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(low, high, ImGui::GetColorU32(ImGuiCol_FrameBg), 2.0f);
    // Quarter grid lines, then the curve.
    for (int q = 1; q < 4; ++q) {
        const float x = low.x + width * float(q) / 4.0f;
        const float y = low.y + height * float(q) / 4.0f;
        const ImU32 grid = ImGui::GetColorU32(ImGuiCol_Border, 0.5f);
        draw->AddLine(ImVec2(x, low.y), ImVec2(x, high.y), grid);
        draw->AddLine(ImVec2(low.x, y), ImVec2(high.x, y), grid);
    }
    std::vector<ImVec2> points;
    for (std::size_t i = 0; i < values.size(); ++i) {
        points.push_back(toScreen(i, values[i]));
    }
    draw->AddPolyline(points.data(), static_cast<int>(points.size()),
                      ImGui::GetColorU32(ImGuiCol_PlotLines), ImDrawFlags_None, 1.5f);
    if (ImGui::IsItemHovered() && !active && !values.empty()) {
        const float t = std::clamp((ImGui::GetIO().MousePos.x - low.x) / width, 0.0f, 1.0f);
        const auto i = static_cast<std::size_t>(std::lround(t * float(values.size() - 1)));
        ImGui::SetTooltip("[%zu] %.4g", i, static_cast<double>(values[i]));
    }
    if (!active || values.empty()) {
        return false;
    }
    // Every sample between the pointer's last and current place, so a fast drag leaves no gaps.
    const ImGuiIO& io = ImGui::GetIO();
    const ImVec2 to = io.MousePos;
    const ImVec2 from =
        ImGui::IsItemActivated() ? to : ImVec2(to.x - io.MouseDelta.x, to.y - io.MouseDelta.y);
    const auto indexOf = [&](float x) {
        const float t = std::clamp((x - low.x) / width, 0.0f, 1.0f);
        return static_cast<std::ptrdiff_t>(std::lround(t * float(values.size() - 1)));
    };
    const auto valueAt = [&](float y) {
        return std::clamp(lo + (high.y - y) / height * (hi - lo), lo, hi);
    };
    const std::ptrdiff_t a = indexOf(from.x);
    const std::ptrdiff_t b = indexOf(to.x);
    bool changed = false;
    for (std::ptrdiff_t i = std::min(a, b); i <= std::max(a, b); ++i) {
        const float s = a == b ? 1.0f : float(i - a) / float(b - a);
        const float v = valueAt(from.y + s * (to.y - from.y));
        changed |= values[static_cast<std::size_t>(i)] != v;
        values[static_cast<std::size_t>(i)] = v;
    }
    return changed;
}

/// "array (100, 3) float32", "array (100, 3) float32 on the GPU": what a field shows for an array.
std::string ArrayText(const Tensor& array) {
    std::string text = "array (";
    for (std::size_t i = 0; i < array.Rank(); ++i) {
        text += (i > 0 ? ", " : "") + std::to_string(array.Shape()[i]);
    }
    text += array.Rank() == 1 ? ",) " : ") ";
    switch (array.Type()) {
    case DType::UInt8:
        text += "uint8";
        break;
    case DType::UInt16:
        text += "uint16";
        break;
    case DType::UInt32:
        text += "uint32";
        break;
    case DType::Int32:
        text += "int32";
        break;
    case DType::Int64:
        text += "int64";
        break;
    case DType::Float16:
        text += "float16";
        break;
    case DType::Float32:
        text += "float32";
        break;
    case DType::Float64:
        text += "float64";
        break;
    }
    return array.Location().kind == DeviceKind::Cpu ? text : text + " on the GPU";
}

/// The field's widget for `shown`; the new value when edited.
std::optional<Value> Widget(Viewer& viewer, const PropertyField& field, const Value& shown) {
    using Kind = PropertyField::Kind;
    const bool limited = field.max > field.min;
    switch (field.kind) {
    case Kind::Bool: {
        bool value = NumberOf(shown) != 0.0;
        if (ImGui::Checkbox("##v", &value)) {
            return value;
        }
        break;
    }
    case Kind::Int: {
        int value = static_cast<int>(NumberOf(shown));
        if (ImGui::DragInt("##v", &value, 0.2f, limited ? static_cast<int>(field.min) : 0,
                           limited ? static_cast<int>(field.max) : 0)) {
            return value;
        }
        break;
    }
    case Kind::Float: {
        double value = NumberOf(shown);
        const float speed = static_cast<float>(limited ? (field.max - field.min) / 500.0
                                                       : std::max(std::abs(value), 1.0) / 200.0);
        if (ImGui::DragScalar("##v", ImGuiDataType_Double, &value, speed,
                              limited ? &field.min : nullptr, limited ? &field.max : nullptr,
                              "%.4g")) {
            return value;
        }
        break;
    }
    case Kind::Token: {
        const std::string current = TextOf(shown);
        std::optional<Value> chosen;
        if (ImGui::BeginCombo("##v", current.c_str())) {
            for (const std::string& option : field.options) {
                if (ImGui::Selectable(option.c_str(), option == current)) {
                    chosen = option;
                }
            }
            ImGui::EndCombo();
        }
        return chosen;
    }
    case Kind::Text: {
        std::array<char, 256> buffer{};
        const std::string current = TextOf(shown);
        std::strncpy(buffer.data(), current.c_str(), buffer.size() - 1);
        if (ImGui::InputText("##v", buffer.data(), buffer.size(),
                             ImGuiInputTextFlags_EnterReturnsTrue) &&
            current != buffer.data()) {
            return std::string(buffer.data());
        }
        break;
    }
    case Kind::Color: {
        glm::vec3 value(Vec3Of(shown));
        if (ImGui::ColorEdit3("##v", &value[0], ImGuiColorEditFlags_Float)) {
            return value;
        }
        break;
    }
    case Kind::Color4: {
        glm::vec4 value(Vec4Of(shown));
        if (ImGui::ColorEdit4("##v", &value[0], ImGuiColorEditFlags_Float)) {
            return glm::dvec4(value);
        }
        break;
    }
    case Kind::Vec2: {
        glm::dvec2 value = Vec2Of(shown);
        if (ImGui::DragScalarN("##v", ImGuiDataType_Double, &value[0], 2,
                               static_cast<float>(std::max(std::abs(value.y), 1.0) / 200.0))) {
            return value;
        }
        break;
    }
    case Kind::Vec3: {
        glm::dvec3 value = Vec3Of(shown);
        const double largest = std::max({std::abs(value.x), std::abs(value.y), std::abs(value.z)});
        if (ImGui::DragScalarN("##v", ImGuiDataType_Double, &value[0], 3,
                               static_cast<float>(std::max(largest, 1.0) / 200.0), nullptr,
                               nullptr, "%.4g")) {
            return value;
        }
        break;
    }
    case Kind::Vec4: {
        glm::dvec4 value = Vec4Of(shown);
        const double largest = std::max({std::abs(value.x), std::abs(value.y), std::abs(value.z),
                                         std::abs(value.w)});
        if (ImGui::DragScalarN("##v", ImGuiDataType_Double, &value[0], 4,
                               static_cast<float>(std::max(largest, 1.0) / 200.0), nullptr,
                               nullptr, "%.4g")) {
            return value;
        }
        break;
    }
    case Kind::Int2: {
        glm::ivec2 value(Vec2Of(shown));
        if (ImGui::DragInt2("##v", &value[0], 1.0f, 1, 16384)) {
            return value;
        }
        break;
    }
    case Kind::Quat: {
        // As Euler angles, glm's convention both ways (glm::eulerAngles and the dquat of a vector
        // of angles), so an edit round-trips.
        const auto* q = shown.Get<glm::dquat>();
        glm::dvec3 degrees =
            glm::degrees(glm::eulerAngles(q != nullptr ? *q : glm::dquat(1, 0, 0, 0)));
        if (ImGui::DragScalarN("##v", ImGuiDataType_Double, &degrees[0], 3, 0.5f, nullptr, nullptr,
                               "%.2f°")) {
            return glm::normalize(glm::dquat(glm::radians(degrees)));
        }
        break;
    }
    case Kind::Matrix: {
        const auto* shownMatrix = shown.Get<glm::dmat4>();
        glm::dmat4 matrix = shownMatrix != nullptr ? *shownMatrix : glm::dmat4(1.0);
        bool changed = false;
        for (int row = 0; row < 4; ++row) {
            ImGui::PushID(row);
            ImGui::SetNextItemWidth(-1.0f);
            std::array<double, 4> values{matrix[0][row], matrix[1][row], matrix[2][row],
                                         matrix[3][row]};
            if (ImGui::DragScalarN("##row", ImGuiDataType_Double, values.data(), 4, 0.01f,
                                   nullptr, nullptr, "%.4g")) {
                for (int column = 0; column < 4; ++column) {
                    matrix[column][row] = values[static_cast<std::size_t>(column)];
                }
                changed = true;
            }
            ImGui::PopID();
        }
        if (changed) {
            return matrix;
        }
        break;
    }
    case Kind::Target: {
        // The prims of the wanted type, or none to clear it.
        const Path current = TargetOf(shown);
        std::optional<Value> chosen;
        const std::string label = current.IsEmpty() ? "(none)" : current.String();
        if (ImGui::BeginCombo("##v", label.c_str())) {
            if (ImGui::Selectable("(none)", current.IsEmpty())) {
                chosen = Value();
            }
            const Scene& scene = viewer.GetScene();
            for (const Path& prim :
                 scene.Prims(field.options.empty() ? std::string_view() : field.options.front())) {
                if (!IsSessionPath(prim) &&
                    ImGui::Selectable(prim.String().c_str(), prim == current)) {
                    chosen = prim;
                }
            }
            ImGui::EndCombo();
        }
        return chosen;
    }
    case Kind::Curve: {
        const auto* shownValues = shown.Get<std::vector<float>>();
        std::vector<float> values = shownValues != nullptr ? *shownValues : std::vector<float>{};
        if (CurveEditor(values, field.min, field.max, 5.0f * ImGui::GetFrameHeight())) {
            return values;
        }
        break;
    }
    }
    return std::nullopt;
}

}  // namespace

float PropertyLabelWidth(const Viewer& viewer) {
    return viewer.GetTheme().propertyLabelWidth * ImGui::GetFontSize() / viewer.GetTheme().fontSize;
}

bool SectionHeader(const Viewer& viewer, const char* title) {
    const Theme& theme = viewer.GetTheme();
    const auto colour = [](const glm::vec4& c) { return ImVec4(c.r, c.g, c.b, c.a); };
    ImGui::PushStyleColor(ImGuiCol_Header, colour(theme.header));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, colour(theme.fieldHovered));
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, colour(theme.header));
    const bool open = ImGui::CollapsingHeader(title, ImGuiTreeNodeFlags_DefaultOpen);
    ImGui::PopStyleColor(3);
    return open;
}

bool MatchesFilter(std::string_view label, std::string_view attribute, std::string_view filter) {
    if (filter.empty()) {
        return true;
    }
    const auto contains = [&](std::string_view text) {
        return std::search(text.begin(), text.end(), filter.begin(), filter.end(),
                           [](char a, char b) {
                               return std::tolower(static_cast<unsigned char>(a)) ==
                                      std::tolower(static_cast<unsigned char>(b));
                           }) != text.end();
    };
    return contains(label) || contains(attribute);
}

bool ResetSquare(const Viewer& viewer, bool authored) {
    const float size = std::round(0.55f * ImGui::GetFontSize());
    const float row = ImGui::GetFrameHeight();
    const ImVec2 corner = ImGui::GetCursorScreenPos();
    const bool clicked = ImGui::InvisibleButton("##reset", ImVec2(size + 4.0f, row));
    const ImVec2 low(corner.x + 2.0f, corner.y + 0.5f * (row - size));
    const ImVec2 high(low.x + size, low.y + size);
    ImDrawList* draw = ImGui::GetWindowDrawList();
    const Theme& theme = viewer.GetTheme();
    if (authored) {
        const glm::vec4& accent = theme.accent;
        draw->AddRectFilled(low, high,
                            ImGui::GetColorU32(ImVec4(accent.r, accent.g, accent.b,
                                                      ImGui::IsItemHovered() ? 1.0f : 0.85f)),
                            1.0f);
    } else {
        draw->AddRect(low, high, ImGui::GetColorU32(ImGuiCol_TextDisabled, 0.45f), 1.0f);
    }
    if (authored && ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Reset to fallback");
    }
    return clicked && authored;
}

bool BeginPropertyTable(const char* id, float labelWidth) {
    if (!ImGui::BeginTable(id, 3, ImGuiTableFlags_SizingFixedFit)) {
        return false;
    }
    ImGui::TableSetupColumn("name", ImGuiTableColumnFlags_WidthFixed, labelWidth);
    ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("reset", ImGuiTableColumnFlags_WidthFixed,
                            std::round(0.55f * ImGui::GetFontSize()) + 4.0f);
    return true;
}

std::string DrawPropertyFields(Viewer& viewer, const Path& path,
                               std::span<const PropertyField> fields, float labelWidth,
                               std::string_view filter) {
    Scene& scene = viewer.GetScene();
    std::string error;
    if (!BeginPropertyTable("##fields", labelWidth)) {
        return error;
    }
    for (const PropertyField& field : fields) {
        if (!MatchesFilter(field.label, field.attribute, filter)) {
            continue;
        }
        ImGui::PushID(field.attribute.c_str());
        // An array written in place of the value (per-vertex colours) is shown, not edited, so
        // an edit cannot replace it unseen; a curve edits a flat one itself.
        const Tensor* array = scene.Get(path, field.attribute) == nullptr &&
                                      field.kind != PropertyField::Kind::Curve
                                  ? scene.Read(path, field.attribute)
                                  : nullptr;
        const bool authored = scene.Get(path, field.attribute) != nullptr || array != nullptr;
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::AlignTextToFramePadding();
        if (authored) {
            ImGui::TextUnformatted(field.label.c_str());
        } else {
            ImGui::TextDisabled("%s", field.label.c_str());
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s%s%s%s", field.attribute.c_str(), authored ? "" : " (fallback)",
                              field.tooltip.empty() ? "" : "\n", field.tooltip.c_str());
        }
        if (authored && ImGui::BeginPopupContextItem("##reset")) {
            if (ImGui::MenuItem("Reset to Fallback")) {
                viewer.ClearAttribute(path, field.attribute);
            }
            ImGui::EndPopup();
        }
        ImGui::TableSetColumnIndex(2);
        if (ResetSquare(viewer, authored)) {
            viewer.ClearAttribute(path, field.attribute);
        }
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1.0f);
        if (array != nullptr) {
            ImGui::AlignTextToFramePadding();
            ImGui::TextDisabled("%s", ArrayText(*array).c_str());
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Written as an array, one value per element; reset it to edit "
                                  "one value for all");
            }
        } else if (const std::optional<Value> edited =
                       Widget(viewer, field, Shown(scene, path, field))) {
            try {
                if (edited->Empty()) {
                    viewer.ClearAttribute(path, field.attribute);
                } else {
                    viewer.Edit(path, field.attribute, *edited, ImGui::IsItemActive());
                }
            } catch (const std::exception& failure) {
                if (error.empty()) {
                    error = failure.what();
                }
            }
        }
        ImGui::PopID();
    }
    ImGui::EndTable();
    return error;
}

std::string DrawPropertySections(Viewer& viewer, const Path& path,
                                 std::span<const PropertySection> sections, float labelWidth,
                                 std::string_view filter) {
    std::string error;
    for (const PropertySection& section : sections) {
        const bool matches =
            std::any_of(section.fields.begin(), section.fields.end(), [&](const auto& field) {
                return MatchesFilter(field.label, field.attribute, filter);
            });
        if (!matches) {
            continue;
        }
        if (SectionHeader(viewer, section.title.c_str())) {
            ImGui::PushID(section.title.c_str());
            const std::string failure =
                DrawPropertyFields(viewer, path, section.fields, labelWidth, filter);
            if (error.empty()) {
                error = failure;
            }
            ImGui::PopID();
        }
    }
    return error;
}

}  // namespace ernie::app
