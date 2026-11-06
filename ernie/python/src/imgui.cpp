// SPDX-License-Identifier: BSD-3-Clause
#include <array>
#include <cfloat>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_stdlib.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace ernie::app {
namespace {

using Float2 = std::array<float, 2>;
using Float3 = std::array<float, 3>;
using Float4 = std::array<float, 4>;

void RequireFrame() {
    if (ImGui::GetCurrentContext() == nullptr || !GImGui->WithinFrameScope) {
        throw nb::value_error(
            "ernie.viewer.imgui draws only in a viewer panel or overlay, while a frame is drawn");
    }
}

ImVec2 Vec2(const std::optional<Float2>& value) {
    return value ? ImVec2((*value)[0], (*value)[1]) : ImVec2(0, 0);
}

template <class Return, class... Arguments> auto Widget(Return (*widget)(Arguments...)) {
    return [widget](Arguments... arguments) -> Return {
        RequireFrame();
        return widget(arguments...);
    };
}

}

void BindImGui(nb::module_& parent) {
    nb::module_ m = parent.def_submodule(
        "imgui", "Dear ImGui's common widgets, for viewer panels and overlays.");

    m.def("begin", Widget(+[](const std::string& name) { return ImGui::Begin(name.c_str()); }),
          nb::arg("name"), "Starts a window; call end() whatever it returns.");
    m.def("end", Widget(+[] { ImGui::End(); }));
    m.def("text", Widget(+[](const std::string& text) { ImGui::TextUnformatted(text.c_str()); }),
          nb::arg("text"));
    m.def("text_colored", Widget(+[](Float4 color, const std::string& text) {
              ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(color[0], color[1], color[2], color[3]));
              ImGui::TextUnformatted(text.c_str());
              ImGui::PopStyleColor();
          }),
          nb::arg("color"), nb::arg("text"));
    m.def("text_wrapped",
          Widget(+[](const std::string& text) { ImGui::TextWrapped("%s", text.c_str()); }),
          nb::arg("text"));
    m.def("text_disabled",
          Widget(+[](const std::string& text) { ImGui::TextDisabled("%s", text.c_str()); }),
          nb::arg("text"));
    m.def("label_text", Widget(+[](const std::string& label, const std::string& text) {
              ImGui::LabelText(label.c_str(), "%s", text.c_str());
          }),
          nb::arg("label"), nb::arg("text"));
    m.def("separator", Widget(+[] { ImGui::Separator(); }));
    m.def("separator_text",
          Widget(+[](const std::string& label) { ImGui::SeparatorText(label.c_str()); }),
          nb::arg("label"));
    m.def("same_line",
          Widget(+[](float offset, float spacing) { ImGui::SameLine(offset, spacing); }),
          nb::arg("offset") = 0.0f, nb::arg("spacing") = -1.0f);
    m.def("spacing", Widget(+[] { ImGui::Spacing(); }));
    m.def("new_line", Widget(+[] { ImGui::NewLine(); }));
    m.def("indent", Widget(+[](float width) { ImGui::Indent(width); }), nb::arg("width") = 0.0f);
    m.def("unindent", Widget(+[](float width) { ImGui::Unindent(width); }),
          nb::arg("width") = 0.0f);

    m.def("button", Widget(+[](const std::string& label, std::optional<Float2> size) {
              return ImGui::Button(label.c_str(), Vec2(size));
          }),
          nb::arg("label"), nb::arg("size") = nb::none(), "True when clicked.");
    m.def("small_button",
          Widget(+[](const std::string& label) { return ImGui::SmallButton(label.c_str()); }),
          nb::arg("label"));
    m.def("checkbox", Widget(+[](const std::string& label, bool value) {
              const bool changed = ImGui::Checkbox(label.c_str(), &value);
              return std::make_pair(changed, value);
          }),
          nb::arg("label"), nb::arg("value"));
    m.def("radio_button", Widget(+[](const std::string& label, bool active) {
              return ImGui::RadioButton(label.c_str(), active);
          }),
          nb::arg("label"), nb::arg("active"));
    m.def("slider_float",
          Widget(+[](const std::string& label, float value, float low, float high,
                     const std::string& format) {
              const bool changed =
                  ImGui::SliderFloat(label.c_str(), &value, low, high, format.c_str());
              return std::make_pair(changed, value);
          }),
          nb::arg("label"), nb::arg("value"), nb::arg("low"), nb::arg("high"),
          nb::arg("format") = "%.3f");
    m.def("slider_int", Widget(+[](const std::string& label, int value, int low, int high) {
              const bool changed = ImGui::SliderInt(label.c_str(), &value, low, high);
              return std::make_pair(changed, value);
          }),
          nb::arg("label"), nb::arg("value"), nb::arg("low"), nb::arg("high"));
    m.def("drag_float",
          Widget(+[](const std::string& label, float value, float speed, float low, float high) {
              const bool changed = ImGui::DragFloat(label.c_str(), &value, speed, low, high);
              return std::make_pair(changed, value);
          }),
          nb::arg("label"), nb::arg("value"), nb::arg("speed") = 1.0f, nb::arg("low") = 0.0f,
          nb::arg("high") = 0.0f);
    m.def("drag_float3", Widget(+[](const std::string& label, Float3 value, float speed) {
              const bool changed = ImGui::DragFloat3(label.c_str(), value.data(), speed);
              return std::make_pair(changed, value);
          }),
          nb::arg("label"), nb::arg("value"), nb::arg("speed") = 1.0f);
    m.def("input_float", Widget(+[](const std::string& label, float value) {
              const bool changed = ImGui::InputFloat(label.c_str(), &value);
              return std::make_pair(changed, value);
          }),
          nb::arg("label"), nb::arg("value"));
    m.def("input_int", Widget(+[](const std::string& label, int value) {
              const bool changed = ImGui::InputInt(label.c_str(), &value);
              return std::make_pair(changed, value);
          }),
          nb::arg("label"), nb::arg("value"));
    m.def("input_text", Widget(+[](const std::string& label, std::string value) {
              const bool changed = ImGui::InputText(label.c_str(), &value);
              return std::make_pair(changed, value);
          }),
          nb::arg("label"), nb::arg("value"));
    m.def("combo",
          Widget(+[](const std::string& label, int current, const std::vector<std::string>& items) {
              bool changed = false;
              const char* preview =
                  current >= 0 && current < int(items.size()) ? items[current].c_str() : "";
              if (ImGui::BeginCombo(label.c_str(), preview)) {
                  for (int i = 0; i < int(items.size()); ++i) {
                      if (ImGui::Selectable(items[i].c_str(), i == current)) {
                          current = i;
                          changed = true;
                      }
                  }
                  ImGui::EndCombo();
              }
              return std::make_pair(changed, current);
          }),
          nb::arg("label"), nb::arg("current"), nb::arg("items"));
    m.def("color_edit3", Widget(+[](const std::string& label, Float3 color) {
              const bool changed = ImGui::ColorEdit3(label.c_str(), color.data());
              return std::make_pair(changed, color);
          }),
          nb::arg("label"), nb::arg("color"));
    m.def("color_edit4", Widget(+[](const std::string& label, Float4 color) {
              const bool changed = ImGui::ColorEdit4(label.c_str(), color.data());
              return std::make_pair(changed, color);
          }),
          nb::arg("label"), nb::arg("color"));
    m.def("progress_bar", Widget(+[](float fraction, const std::string& overlay) {
              ImGui::ProgressBar(fraction, ImVec2(-1, 0),
                                 overlay.empty() ? nullptr : overlay.c_str());
          }),
          nb::arg("fraction"), nb::arg("overlay") = "");
    m.def("plot_lines",
          Widget(+[](const std::string& label, const std::vector<float>& values,
                     std::optional<Float2> size) {
              ImGui::PlotLines(label.c_str(), values.data(), int(values.size()), 0, nullptr,
                               FLT_MAX, FLT_MAX, Vec2(size));
          }),
          nb::arg("label"), nb::arg("values"), nb::arg("size") = nb::none());
    m.def("selectable", Widget(+[](const std::string& label, bool selected) {
              return ImGui::Selectable(label.c_str(), selected);
          }),
          nb::arg("label"), nb::arg("selected") = false, "True when clicked.");

    m.def("collapsing_header",
          Widget(+[](const std::string& label) { return ImGui::CollapsingHeader(label.c_str()); }),
          nb::arg("label"));
    m.def("tree_node",
          Widget(+[](const std::string& label) { return ImGui::TreeNode(label.c_str()); }),
          nb::arg("label"), "True when open; call tree_pop() then.");
    m.def("tree_pop", Widget(+[] { ImGui::TreePop(); }));
    m.def("begin_child",
          Widget(+[](const std::string& name, std::optional<Float2> size, bool border) {
              return ImGui::BeginChild(name.c_str(), Vec2(size),
                                       border ? ImGuiChildFlags_Borders : ImGuiChildFlags_None);
          }),
          nb::arg("name"), nb::arg("size") = nb::none(), nb::arg("border") = false);
    m.def("end_child", Widget(+[] { ImGui::EndChild(); }));
    m.def("push_id", Widget(+[](const std::string& id) { ImGui::PushID(id.c_str()); }),
          nb::arg("id"));
    m.def("pop_id", Widget(+[] { ImGui::PopID(); }));
    m.def("is_item_hovered", Widget(+[] { return ImGui::IsItemHovered(); }));
    m.def("set_tooltip",
          Widget(+[](const std::string& text) { ImGui::SetTooltip("%s", text.c_str()); }),
          nb::arg("text"));
    m.def("framerate", Widget(+[] { return ImGui::GetIO().Framerate; }),
          "Frames a second, averaged over the last frames.");

    m.def("add_line", Widget(+[](Float2 from, Float2 to, Float4 color, float thickness) {
              ImGui::GetWindowDrawList()->AddLine(
                  {from[0], from[1]}, {to[0], to[1]},
                  ImGui::GetColorU32(ImVec4(color[0], color[1], color[2], color[3])), thickness);
          }),
          nb::arg("start"), nb::arg("end"), nb::arg("color"), nb::arg("thickness") = 1.0f,
          "Draws a line in screen pixels, as an overlay's rectangle gives them.");
    m.def(
        "add_rect", Widget(+[](Float2 min, Float2 max, Float4 color, float thickness, bool filled) {
            ImDrawList* list = ImGui::GetWindowDrawList();
            const ImU32 packed = ImGui::GetColorU32(ImVec4(color[0], color[1], color[2], color[3]));
            if (filled) {
                list->AddRectFilled({min[0], min[1]}, {max[0], max[1]}, packed);
            } else {
                list->AddRect({min[0], min[1]}, {max[0], max[1]}, packed, 0.0f, thickness);
            }
        }),
        nb::arg("min"), nb::arg("max"), nb::arg("color"), nb::arg("thickness") = 1.0f,
        nb::arg("filled") = false);
    m.def("add_circle",
          Widget(+[](Float2 center, float radius, Float4 color, float thickness, bool filled) {
              ImDrawList* list = ImGui::GetWindowDrawList();
              const ImU32 packed =
                  ImGui::GetColorU32(ImVec4(color[0], color[1], color[2], color[3]));
              if (filled) {
                  list->AddCircleFilled({center[0], center[1]}, radius, packed);
              } else {
                  list->AddCircle({center[0], center[1]}, radius, packed, 0, thickness);
              }
          }),
          nb::arg("center"), nb::arg("radius"), nb::arg("color"), nb::arg("thickness") = 1.0f,
          nb::arg("filled") = false);
    m.def("add_text", Widget(+[](Float2 position, Float4 color, const std::string& text) {
              ImGui::GetWindowDrawList()->AddText(
                  {position[0], position[1]},
                  ImGui::GetColorU32(ImVec4(color[0], color[1], color[2], color[3])), text.c_str());
          }),
          nb::arg("position"), nb::arg("color"), nb::arg("text"));
}

}
