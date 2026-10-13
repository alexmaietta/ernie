// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/builtin_panels.h"
#include "ernie/app/camera_sections.h"
#include "ernie/app/icons.h"
#include "ernie/app/prim_icons.h"
#include "ernie/app/prim_schemas.h"
#include "ernie/app/property_fields.h"
#include "ernie/app/viewer.h"
#include "ernie/scene/tokens.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <imgui.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

namespace ernie::app {

namespace {

/// Three drag fields for x, y and z in the item width, each marked on
/// its left edge in its axis's colour (red, green, blue), as Kit's
/// transform fields are.
bool DragXyz(double* values, float speed) {
    constexpr ImU32 kAxisColours[] = {IM_COL32(222, 74, 64, 255),
                                      IM_COL32(110, 190, 70, 255),
                                      IM_COL32(70, 130, 230, 255)};
    const float spacing = ImGui::GetStyle().ItemInnerSpacing.x;
    const float width = std::max(
        (ImGui::CalcItemWidth() - 2.0f * spacing) / 3.0f, 1.0f);
    const float mark =
        std::max(2.0f, std::round(0.15f * ImGui::GetFontSize()));
    bool changed = false;
    for (int k = 0; k < 3; ++k) {
        if (k > 0) {
            ImGui::SameLine(0.0f, spacing);
        }
        ImGui::PushID(k);
        const ImVec2 corner = ImGui::GetCursorScreenPos();
        ImGui::SetNextItemWidth(width);
        changed |= ImGui::DragScalar("##axis", ImGuiDataType_Double,
                                     &values[k], speed, nullptr,
                                     nullptr, "%.3f");
        ImGui::GetWindowDrawList()->AddRectFilled(
            corner,
            ImVec2(corner.x + mark,
                   corner.y + ImGui::GetFrameHeight()),
            kAxisColours[k], ImGui::GetStyle().FrameRounding,
            ImDrawFlags_RoundCornersLeft);
        ImGui::PopID();
    }
    return changed;
}

const char* DTypeName(DType type) {
    switch (type) {
    case DType::Float32:
        return "float32";
    case DType::Float64:
        return "float64";
    case DType::Int32:
        return "int32";
    case DType::Int64:
        return "int64";
    case DType::UInt8:
        return "uint8";
    case DType::UInt16:
        return "uint16";
    case DType::UInt32:
        return "uint32";
    default:
        return "?";
    }
}

/// Whether an attribute named `name` holds a colour, to edit with a
/// colour picker.
bool IsColor(std::string_view name) {
    std::string lower(name);
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return lower.find("color") != std::string::npos ||
           lower.find("albedo") != std::string::npos ||
           lower.find("tint") != std::string::npos;
}

/// A local transform as translation, rotation (XYZ Euler angles in
/// degrees) and scale.
struct Trs {
    glm::dvec3 translate{0.0};
    glm::dvec3 rotate{0.0};
    glm::dvec3 scale{1.0};
};

/// Translation, rotation and scale of an affine `matrix` without
/// shear; a mirrored one takes the mirror into its x scale.
Trs Decompose(const glm::dmat4& matrix) {
    Trs trs;
    trs.translate = glm::dvec3(matrix[3]);
    glm::dmat3 rotation(matrix);
    for (int axis = 0; axis < 3; ++axis) {
        trs.scale[axis] = glm::length(rotation[axis]);
        if (trs.scale[axis] > 0.0) {
            rotation[axis] /= trs.scale[axis];
        }
    }
    if (glm::determinant(rotation) < 0.0) {
        trs.scale.x = -trs.scale.x;
        rotation[0] = -rotation[0];
    }
    trs.rotate =
        glm::degrees(glm::eulerAngles(glm::quat_cast(rotation)));
    return trs;
}

glm::dmat4 Compose(const Trs& trs) {
    const glm::dquat orientation(glm::radians(trs.rotate));
    return glm::translate(glm::dmat4(1.0), trs.translate) *
           glm::mat4_cast(orientation) *
           glm::scale(glm::dmat4(1.0), trs.scale);
}

class PropertiesPanel final : public Panel {
  public:
    PropertiesPanel() : Panel("Properties", Dock::RightLower) {}

    void Draw(Viewer& viewer) override {
        const std::vector<Path>& selection = viewer.Selection();
        Scene& scene = viewer.GetScene();
        if (selection.empty() || !scene.Has(selection.front())) {
            ImGui::TextDisabled(
                "Select a prim in the Stage or the viewport.");
            return;
        }
        const Path& path = selection.front();
        // As Kit's property window: a search field, then the prim's
        // path and type.
        ImGui::SetNextItemWidth(-1.0f);
        ImGui::InputTextWithHint("##search", "Search", search_,
                                 sizeof(search_));
        const std::string_view filter(search_);
        const std::string& type = scene.TypeOf(path);
        PrimIcon(type, ImGui::GetFrameHeight());
        ImGui::SameLine();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(path.String().c_str());
        ImGui::SameLine();
        ImGui::TextDisabled(
            "%s%s", type.empty() ? "untyped" : type.c_str(),
            !selected              ? "  (the viewport's camera)"
            : selection.size() > 1 ? "  (first of the selection)"
                                   : "");
        if (!error_.empty()) {
            ImGui::TextColored(ImVec4(0.89f, 0.34f, 0.32f, 1.0f),
                               "%s", error_.c_str());
        }
        labelWidth_ = PropertyLabelWidth(viewer);
        if (!path.IsRoot() && HasTransform(type) &&
            MatchesFilter("Translate Rotate Scale",
                          tokens::kTransform, filter) &&
            SectionHeader(viewer, "Transform")) {
            DrawTransform(viewer, scene, path);
        }
        // What Ernie reads on this type, set or not, then whatever
        // else the prim holds.
        const std::vector<PropertySection> sections =
            SchemaOf(scene, path, viewer.Brdfs(),
                     viewer.Delegate().SensorFilters());
        // A camera's exposure joins its meter in its own sections.
        const bool camera = type == tokens::kCamera;
        std::vector<PropertySection> drawn = sections;
        if (camera) {
            std::erase_if(drawn, [](const PropertySection& section) {
                return section.title == "Exposure";
            });
        }
        const std::string failure = DrawPropertySections(
            viewer, path, drawn, labelWidth_, filter);
        if (!failure.empty()) {
            error_ = failure;
        }
        if (camera) {
            DrawCameraSections(viewer, path, sections, labelWidth_,
                               filter);
        }
        std::set<std::string, std::less<>> known{
            std::string(tokens::kTransform)};
        for (const PropertySection& section : sections) {
            for (const PropertyField& field : section.fields) {
                known.insert(field.attribute);
            }
        }
        std::vector<std::string> others;
        for (const std::string& attribute : scene.Attributes(path)) {
            if (!known.contains(attribute) &&
                MatchesFilter(attribute, attribute, filter)) {
                others.push_back(attribute);
            }
        }
        if (!others.empty() &&
            SectionHeader(viewer, "Other attributes")) {
            DrawAttributes(viewer, scene, path, others);
        }
    }

  private:
    /// Starts a property row labelled `label`, with a context menu to
    /// reset the attribute.
    void Row(Viewer& viewer, const Path& path, const char* label,
             std::string_view attribute, bool authored) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::AlignTextToFramePadding();
        if (authored) {
            ImGui::TextUnformatted(label);
        } else {
            ImGui::TextDisabled("%s", label);
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", std::string(attribute).c_str());
        }
        if (authored && ImGui::BeginPopupContextItem(label)) {
            if (ImGui::MenuItem("Reset to Fallback")) {
                viewer.ClearAttribute(path, attribute);
            }
            ImGui::EndPopup();
        }
        ImGui::TableSetColumnIndex(2);
        if (ResetSquare(viewer, authored)) {
            viewer.ClearAttribute(path, attribute);
        }
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1.0f);
    }

    bool BeginTable(const char* id) {
        return BeginPropertyTable(id, labelWidth_);
    }

    /// Applies `value` to `attribute`, keeping a failure to show.
    void Apply(Viewer& viewer, const Path& path,
               std::string_view attribute, Value value) {
        try {
            viewer.Edit(path, attribute, std::move(value),
                        ImGui::IsItemActive());
            error_.clear();
        } catch (const std::exception& error) {
            error_ = error.what();
        }
    }

    void DrawTransform(Viewer& viewer, Scene& scene,
                       const Path& path) {
        if (!BeginTable("##transform")) {
            return;
        }
        // A prim with transform ops resets to the bare
        // xformOp:transform by clearing its order.
        const bool ops =
            scene.Get(path, tokens::kXformOpOrder) != nullptr;
        const std::string_view attribute =
            ops ? tokens::kXformOpOrder : tokens::kTransform;
        const bool authored = scene.Get(path, attribute) != nullptr;
        Trs trs = Decompose(scene.LocalTransform(path));
        const char* labels[] = {"Translate", "Rotate", "Scale"};
        glm::dvec3* fields[] = {&trs.translate, &trs.rotate,
                                &trs.scale};
        const double speeds[] = {0.01, 0.5, 0.01};
        for (int i = 0; i < 3; ++i) {
            ImGui::PushID(i);
            Row(viewer, path, labels[i], attribute, authored);
            if (DragXyz(&(*fields[i])[0],
                        static_cast<float>(speeds[i]))) {
                try {
                    viewer.EditLocalTransform(path, Compose(trs),
                                              ImGui::IsItemActive());
                    error_.clear();
                } catch (const std::exception& error) {
                    error_ = error.what();
                }
            }
            ImGui::PopID();
        }
        ImGui::EndTable();
    }

    void DrawAttributes(Viewer& viewer, Scene& scene,
                        const Path& path,
                        const std::vector<std::string>& attributes) {
        if (!BeginTable("##attributes")) {
            return;
        }
        for (const std::string& attribute : attributes) {
            ImGui::PushID(attribute.c_str());
            Row(viewer, path, attribute.c_str(), attribute, true);
            if (const Value* value = scene.Get(path, attribute)) {
                DrawValue(viewer, path, attribute, *value);
            } else if (const Tensor* tensor =
                           scene.Read(path, attribute)) {
                std::string shape;
                for (const std::int64_t extent : tensor->Shape()) {
                    shape += (shape.empty() ? "" : " x ") +
                             std::to_string(extent);
                }
                const DeviceKind kind = tensor->Location().kind;
                ImGui::TextDisabled(
                    "%s [%s] %s", DTypeName(tensor->Type()),
                    shape.c_str(),
                    kind == DeviceKind::Cuda   ? "cuda"
                    : kind == DeviceKind::Rocm ? "rocm"
                                               : "cpu");
            }
            ImGui::PopID();
        }
        ImGui::EndTable();
    }

    void DrawValue(Viewer& viewer, const Path& path,
                   const std::string& attribute, const Value& value) {
        std::visit(
            [&](const auto& v) {
                using T = std::decay_t<decltype(v)>;
                T edit = v;
                bool changed = false;
                if constexpr (std::is_same_v<T, std::monostate>) {
                    ImGui::TextDisabled("(empty)");
                } else if constexpr (std::is_same_v<T, bool>) {
                    changed = ImGui::Checkbox("##v", &edit);
                } else if constexpr (std::is_same_v<T, int>) {
                    changed = ImGui::DragInt("##v", &edit);
                } else if constexpr (std::is_same_v<T, float>) {
                    changed =
                        ImGui::DragFloat("##v", &edit, Speed(edit),
                                         0.0f, 0.0f, "%.4g");
                } else if constexpr (std::is_same_v<T, double>) {
                    changed = ImGui::DragScalar(
                        "##v", ImGuiDataType_Double, &edit,
                        Speed(static_cast<float>(edit)), nullptr,
                        nullptr, "%.6g");
                } else if constexpr (std::is_same_v<T, glm::ivec2>) {
                    changed = ImGui::DragInt2("##v", &edit[0]);
                } else if constexpr (std::is_same_v<T, glm::vec2>) {
                    changed =
                        ImGui::DragFloat2("##v", &edit[0], 0.01f);
                } else if constexpr (std::is_same_v<T, glm::vec3>) {
                    changed = IsColor(attribute)
                                  ? ImGui::ColorEdit3(
                                        "##v", &edit[0],
                                        ImGuiColorEditFlags_Float)
                                  : ImGui::DragFloat3("##v", &edit[0],
                                                      0.01f);
                } else if constexpr (std::is_same_v<T, glm::vec4>) {
                    changed = IsColor(attribute)
                                  ? ImGui::ColorEdit4(
                                        "##v", &edit[0],
                                        ImGuiColorEditFlags_Float)
                                  : ImGui::DragFloat4("##v", &edit[0],
                                                      0.01f);
                } else if constexpr (std::is_same_v<T, glm::dvec2>) {
                    changed = ImGui::DragScalarN("##v",
                                                 ImGuiDataType_Double,
                                                 &edit[0], 2, 0.01f);
                } else if constexpr (std::is_same_v<T, glm::dvec3>) {
                    changed = ImGui::DragScalarN("##v",
                                                 ImGuiDataType_Double,
                                                 &edit[0], 3, 0.01f);
                } else if constexpr (std::is_same_v<T, glm::dvec4>) {
                    changed = ImGui::DragScalarN("##v",
                                                 ImGuiDataType_Double,
                                                 &edit[0], 4, 0.01f);
                } else if constexpr (std::is_same_v<T, glm::dmat4>) {
                    for (int row = 0; row < 4; ++row) {
                        ImGui::PushID(row);
                        glm::dvec4 values(edit[0][row], edit[1][row],
                                          edit[2][row], edit[3][row]);
                        ImGui::SetNextItemWidth(-1.0f);
                        if (ImGui::DragScalarN(
                                "##row", ImGuiDataType_Double,
                                &values[0], 4, 0.01f)) {
                            for (int col = 0; col < 4; ++col) {
                                edit[col][row] = values[col];
                            }
                            changed = true;
                        }
                        ImGui::PopID();
                    }
                } else if constexpr (std::is_same_v<T, glm::dquat>) {
                    // w, x, y, z as USD writes a quaternion;
                    // renormalised after a drag.
                    glm::dvec4 wxyz(v.w, v.x, v.y, v.z);
                    if (ImGui::DragScalarN("##v",
                                           ImGuiDataType_Double,
                                           &wxyz[0], 4, 0.005f)) {
                        const double length = glm::length(wxyz);
                        if (length > 0.0) {
                            wxyz /= length;
                            edit = glm::dquat(wxyz[0], wxyz[1],
                                              wxyz[2], wxyz[3]);
                            changed = true;
                        }
                    }
                } else if constexpr (std::is_same_v<
                                         T,
                                         std::vector<std::string>>) {
                    if (v.empty()) {
                        ImGui::TextDisabled("(none)");
                    }
                    for (const std::string& token : v) {
                        ImGui::TextUnformatted(token.c_str());
                    }
                } else if constexpr (std::is_same_v<T, std::string>) {
                    if (const auto text =
                            TextField(path, attribute, v)) {
                        edit = *text;
                        changed = true;
                    }
                } else if constexpr (std::is_same_v<T, Path>) {
                    if (const auto text =
                            TextField(path, attribute, v.String())) {
                        try {
                            edit = Path(*text);
                            changed = true;
                        } catch (const std::exception& error) {
                            error_ = error.what();
                        }
                    }
                } else if constexpr (std::is_same_v<
                                         T, std::vector<float>>) {
                    ImGui::TextDisabled("%zu numbers", v.size());
                } else if constexpr (std::is_same_v<
                                         T, std::vector<Path>>) {
                    if (v.empty()) {
                        ImGui::TextDisabled("(none)");
                    }
                    for (const Path& target : v) {
                        ImGui::TextUnformatted(
                            target.String().c_str());
                    }
                }
                if constexpr (!std::is_same_v<T, std::monostate>) {
                    if (changed) {
                        Apply(viewer, path, attribute, Value(edit));
                    }
                }
            },
            value.Raw());
    }

    static float Speed(float value) {
        return std::max(std::abs(value) * 0.005f, 0.001f);
    }

    /// A text field for `attribute`, committed on Enter or when it
    /// loses focus; the new text then, else none.
    std::optional<std::string> TextField(const Path& path,
                                         const std::string& attribute,
                                         const std::string& current) {
        const std::string key = path.String() + "\n" + attribute;
        std::string& buffer = buffers_[key];
        if (!ImGui::IsItemActive() && editing_ != key) {
            buffer = current;
        }
        buffer.resize(std::max<std::size_t>(buffer.size(), 256),
                      '\0');
        const bool entered =
            ImGui::InputText("##v", buffer.data(), buffer.size(),
                             ImGuiInputTextFlags_EnterReturnsTrue);
        editing_ = ImGui::IsItemActive()
                       ? key
                       : (editing_ == key ? std::string() : editing_);
        if (entered || ImGui::IsItemDeactivatedAfterEdit()) {
            // The text up to the NUL padding of the edit buffer.
            std::string text = buffer.substr(0, buffer.find('\0'));
            buffer = text;
            if (text != current) {
                return text;
            }
        }
        return std::nullopt;
    }

    float labelWidth_ = 120.0f;
    char search_[128] = {};
    std::string error_;
    std::map<std::string, std::string>
        buffers_;  ///< Text being typed, by prim and attribute.
    std::string editing_;
};

}  // namespace

std::unique_ptr<Panel> MakePropertiesPanel() {
    return std::make_unique<PropertiesPanel>();
}

}  // namespace ernie::app
