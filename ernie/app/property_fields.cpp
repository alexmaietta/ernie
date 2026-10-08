#include "ernie/app/property_fields.h"

#include "ernie/app/viewer.h"
#include "ernie/scene/schema.h"

#include <imgui.h>

#include <array>
#include <cmath>
#include <cstring>
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
    return glm::dvec2(0.0);
}

/// The value a field shows: the authored one, read the way Ernie reads it, or the fallback.
Value Shown(const Scene& scene, const Path& path, const PropertyField& field, bool authored) {
    using Kind = PropertyField::Kind;
    if (!authored) {
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
        return GetVec2(scene, path, field.attribute, Vec2Of(field.fallback));
    }
    return field.fallback;
}

}  // namespace

std::string DrawPropertyFields(Viewer& viewer, const Path& path,
                               std::span<const PropertyField> fields, float labelWidth) {
    using Kind = PropertyField::Kind;
    Scene& scene = viewer.GetScene();
    std::string error;
    if (!ImGui::BeginTable("##fields", 2, ImGuiTableFlags_SizingFixedFit)) {
        return error;
    }
    ImGui::TableSetupColumn("name", ImGuiTableColumnFlags_WidthFixed, labelWidth);
    ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch);
    for (const PropertyField& field : fields) {
        ImGui::PushID(field.attribute.data(), field.attribute.data() + field.attribute.size());
        const bool authored = scene.Get(path, field.attribute) != nullptr;
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::AlignTextToFramePadding();
        if (authored) {
            ImGui::TextUnformatted(field.label);
        } else {
            ImGui::TextDisabled("%s", field.label);
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s%s%s", std::string(field.attribute).c_str(),
                              field.tooltip != nullptr ? "\n" : "",
                              field.tooltip != nullptr ? field.tooltip : "");
        }
        if (authored && ImGui::BeginPopupContextItem("##reset")) {
            if (ImGui::MenuItem("Reset to Fallback")) {
                viewer.ClearAttribute(path, field.attribute);
            }
            ImGui::EndPopup();
        }
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1.0f);
        const Value shown = Shown(scene, path, field, authored);
        std::optional<Value> edited;
        const bool limited = field.max > field.min;
        switch (field.kind) {
        case Kind::Bool: {
            bool value = NumberOf(shown) != 0.0;
            if (ImGui::Checkbox("##v", &value)) {
                edited = value;
            }
            break;
        }
        case Kind::Int: {
            int value = static_cast<int>(NumberOf(shown));
            if (ImGui::DragInt("##v", &value, 0.2f, limited ? static_cast<int>(field.min) : 0,
                               limited ? static_cast<int>(field.max) : 0)) {
                edited = value;
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
                edited = value;
            }
            break;
        }
        case Kind::Token: {
            const std::string current = TextOf(shown);
            if (ImGui::BeginCombo("##v", current.c_str())) {
                for (const std::string_view option : field.options) {
                    if (ImGui::Selectable(std::string(option).c_str(), option == current)) {
                        edited = std::string(option);
                    }
                }
                ImGui::EndCombo();
            }
            break;
        }
        case Kind::Text: {
            std::array<char, 256> buffer{};
            const std::string current = TextOf(shown);
            std::strncpy(buffer.data(), current.c_str(), buffer.size() - 1);
            if (ImGui::InputText("##v", buffer.data(), buffer.size(),
                                 ImGuiInputTextFlags_EnterReturnsTrue) &&
                current != buffer.data()) {
                edited = std::string(buffer.data());
            }
            break;
        }
        case Kind::Color: {
            glm::vec3 value(Vec3Of(shown));
            if (ImGui::ColorEdit3("##v", &value[0], ImGuiColorEditFlags_Float)) {
                edited = value;
            }
            break;
        }
        case Kind::Vec2: {
            glm::dvec2 value = Vec2Of(shown);
            if (ImGui::DragScalarN("##v", ImGuiDataType_Double, &value[0], 2,
                                   static_cast<float>(std::max(std::abs(value.y), 1.0) / 200.0))) {
                edited = value;
            }
            break;
        }
        }
        if (edited) {
            try {
                viewer.Edit(path, field.attribute, *edited, ImGui::IsItemActive());
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

}  // namespace ernie::app
