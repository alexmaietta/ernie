#include "ernie/app/property_fields.h"

#include "ernie/app/session.h"
#include "ernie/app/viewer.h"
#include "ernie/scene/schema.h"

#include <imgui.h>

#include <algorithm>
#include <array>
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
    case Kind::Vec2:
    case Kind::Int2:
        return Vec2Of(*authored);
    case Kind::Target:
        return TargetOf(*authored);
    }
    return field.fallback;
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
        const float speed = static_cast<float>(
            limited ? (field.max - field.min) / 500.0 : std::max(std::abs(value), 1.0) / 200.0);
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
    case Kind::Vec2: {
        glm::dvec2 value = Vec2Of(shown);
        if (ImGui::DragScalarN("##v", ImGuiDataType_Double, &value[0], 2,
                               static_cast<float>(std::max(std::abs(value.y), 1.0) / 200.0))) {
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
    }
    return std::nullopt;
}

}  // namespace

float PropertyLabelWidth(const Viewer& viewer) {
    return viewer.GetTheme().propertyLabelWidth * ImGui::GetFontSize() / viewer.GetTheme().fontSize;
}

std::string DrawPropertyFields(Viewer& viewer, const Path& path,
                               std::span<const PropertyField> fields, float labelWidth) {
    Scene& scene = viewer.GetScene();
    std::string error;
    if (!ImGui::BeginTable("##fields", 2, ImGuiTableFlags_SizingFixedFit)) {
        return error;
    }
    ImGui::TableSetupColumn("name", ImGuiTableColumnFlags_WidthFixed, labelWidth);
    ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch);
    for (const PropertyField& field : fields) {
        ImGui::PushID(field.attribute.c_str());
        const bool authored = scene.Get(path, field.attribute) != nullptr;
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::AlignTextToFramePadding();
        if (authored) {
            ImGui::TextUnformatted(field.label.c_str());
        } else {
            ImGui::TextDisabled("%s", field.label.c_str());
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s%s%s%s", field.attribute.c_str(),
                              authored ? "" : " (fallback)", field.tooltip.empty() ? "" : "\n",
                              field.tooltip.c_str());
        }
        if (authored && ImGui::BeginPopupContextItem("##reset")) {
            if (ImGui::MenuItem("Reset to Fallback")) {
                viewer.ClearAttribute(path, field.attribute);
            }
            ImGui::EndPopup();
        }
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1.0f);
        if (const std::optional<Value> edited = Widget(viewer, field, Shown(scene, path, field))) {
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
                                 std::span<const PropertySection> sections, float labelWidth) {
    std::string error;
    for (const PropertySection& section : sections) {
        if (ImGui::CollapsingHeader(section.title.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::PushID(section.title.c_str());
            const std::string failure = DrawPropertyFields(viewer, path, section.fields, labelWidth);
            if (error.empty()) {
                error = failure;
            }
            ImGui::PopID();
        }
    }
    return error;
}

}  // namespace ernie::app
