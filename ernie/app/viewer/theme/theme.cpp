// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/viewer/theme/theme.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <string_view>

namespace ernie::app {
namespace {

constexpr float kPi = 3.14159265f;
const ImU32 kLamp = IM_COL32(226, 220, 132, 255);

void SetMetrics(const Theme& theme, ImGuiStyle& style) {
    style.WindowPadding = {8.0f, 6.0f};
    style.FramePadding = theme.framePadding;
    style.ItemSpacing = theme.itemSpacing;
    style.ItemInnerSpacing = {4.0f, 3.0f};
    style.CellPadding = {4.0f, 2.0f};
    style.IndentSpacing = 14.0f;
    style.ScrollbarSize = 10.0f;
    style.GrabMinSize = 8.0f;
    style.WindowRounding = 0.0f;
    style.ChildRounding = 0.0f;
    style.FrameRounding = theme.rounding;
    style.PopupRounding = theme.rounding;
    style.ScrollbarRounding = theme.rounding;
    style.GrabRounding = theme.rounding;
    style.TabRounding = theme.rounding;
    style.WindowBorderSize = 1.0f;
    style.ChildBorderSize = 1.0f;
    style.PopupBorderSize = 1.0f;
    style.FrameBorderSize = theme.frameBorder;
    style.TabBorderSize = theme.frameBorder;
    style.TabBarOverlineSize = 0.0f;
    style.DockingSeparatorSize = 1.0f;
    style.WindowMenuButtonPosition = ImGuiDir_None;
    style.WindowTitleAlign = {0.0f, 0.5f};
    style.SeparatorTextBorderSize = 1.0f;
}

void SetColors(const Theme& theme, ImGuiStyle& style) {
    auto set = [&](ImGuiCol index, const ImVec4& color) { style.Colors[index] = color; };
    ImVec4 accentDim = WithAlpha(theme.accent, 0.45f);
    ImVec4 accentFaint = WithAlpha(theme.accent, 0.25f);
    ImVec4 clear(0.0f, 0.0f, 0.0f, 0.0f);
    set(ImGuiCol_Text, theme.text);
    set(ImGuiCol_TextDisabled, theme.textMuted);
    set(ImGuiCol_WindowBg, theme.panel);
    set(ImGuiCol_ChildBg, clear);
    set(ImGuiCol_PopupBg, WithAlpha(theme.overlay, 1.0f));
    set(ImGuiCol_Border, theme.border);
    set(ImGuiCol_BorderShadow, clear);
    set(ImGuiCol_FrameBg, theme.field);
    set(ImGuiCol_FrameBgHovered, theme.fieldHovered);
    set(ImGuiCol_FrameBgActive, theme.fieldHovered);
    for (ImGuiCol title : {ImGuiCol_TitleBg, ImGuiCol_TitleBgActive, ImGuiCol_TitleBgCollapsed,
                           ImGuiCol_MenuBarBg}) {
        set(title, theme.background);
    }
    set(ImGuiCol_ScrollbarBg, clear);
    set(ImGuiCol_ScrollbarGrab, theme.fieldHovered);
    set(ImGuiCol_ScrollbarGrabHovered, theme.textMuted);
    set(ImGuiCol_ScrollbarGrabActive, theme.accent);
    set(ImGuiCol_CheckMark, theme.checkMark);
    set(ImGuiCol_CheckboxSelectedBg, theme.checkBox);
    set(ImGuiCol_SliderGrab, theme.buttonLit);
    set(ImGuiCol_SliderGrabActive, theme.text);
    set(ImGuiCol_Button, theme.button);
    set(ImGuiCol_ButtonHovered, theme.fieldHovered);
    set(ImGuiCol_ButtonActive, theme.buttonLit);
    set(ImGuiCol_Header, theme.selection);
    set(ImGuiCol_HeaderHovered, WithAlpha(theme.selection, 0.85f));
    set(ImGuiCol_HeaderActive, theme.selection);
    set(ImGuiCol_Separator, theme.separator);
    set(ImGuiCol_SeparatorHovered, accentDim);
    set(ImGuiCol_SeparatorActive, theme.accent);
    set(ImGuiCol_ResizeGrip, clear);
    set(ImGuiCol_ResizeGripHovered, accentDim);
    set(ImGuiCol_ResizeGripActive, theme.accent);
    set(ImGuiCol_InputTextCursor, theme.text);
    set(ImGuiCol_Tab, theme.background);
    set(ImGuiCol_TabHovered, theme.fieldHovered);
    set(ImGuiCol_TabSelected, theme.panel);
    set(ImGuiCol_TabSelectedOverline, theme.accent);
    set(ImGuiCol_TabDimmed, theme.background);
    set(ImGuiCol_TabDimmedSelected, theme.panel);
    set(ImGuiCol_TabDimmedSelectedOverline, WithAlpha(theme.accent, 0.0f));
    set(ImGuiCol_DockingPreview, accentDim);
    set(ImGuiCol_DockingEmptyBg, theme.background);
    set(ImGuiCol_PlotLines, theme.accent);
    set(ImGuiCol_PlotLinesHovered, theme.text);
    set(ImGuiCol_PlotHistogram, theme.accent);
    set(ImGuiCol_PlotHistogramHovered, theme.text);
    set(ImGuiCol_TableHeaderBg, theme.header);
    set(ImGuiCol_TableBorderStrong, theme.separator);
    set(ImGuiCol_TableBorderLight, theme.separator);
    set(ImGuiCol_TableRowBg, clear);
    set(ImGuiCol_TableRowBgAlt, WithAlpha(theme.text, 0.03f));
    set(ImGuiCol_TextLink, theme.accent);
    set(ImGuiCol_TextSelectedBg, accentFaint);
    set(ImGuiCol_TreeLines, theme.separator);
    set(ImGuiCol_DragDropTarget, theme.accent);
    set(ImGuiCol_DragDropTargetBg, accentFaint);
    set(ImGuiCol_UnsavedMarker, theme.warning);
    set(ImGuiCol_NavCursor, theme.accent);
    set(ImGuiCol_NavWindowingHighlight, theme.accent);
    set(ImGuiCol_NavWindowingDimBg, WithAlpha(theme.background, 0.6f));
    set(ImGuiCol_ModalWindowDimBg, WithAlpha(theme.background, 0.6f));
}

ImU32 Rgb(int r, int g, int b) {
    return IM_COL32(r, g, b, 255);
}

struct Frame {
    ImVec2 corner;
    float size;

    ImVec2 operator()(float u, float v) const {
        return {corner.x + u * size, corner.y + v * size};
    }

    float operator()(float length) const {
        return length * size;
    }
};

void Tripod(ImDrawList* draw, const Frame& at) {
    float width = std::max(1.2f, at(0.09f));
    ImVec2 center = at(0.5f, 0.58f);
    draw->AddLine(center, at(0.5f, 0.1f), Rgb(110, 200, 90), width);
    draw->AddLine(center, at(0.1f, 0.86f), Rgb(225, 80, 70), width);
    draw->AddLine(center, at(0.9f, 0.86f), Rgb(80, 140, 235), width);
}

void Folder(ImDrawList* draw, const Frame& at) {
    draw->AddRectFilled(at(0.08f, 0.2f), at(0.45f, 0.38f), Rgb(74, 85, 88), at(0.06f));
    draw->AddRectFilled(at(0.08f, 0.3f), at(0.92f, 0.82f), Rgb(96, 108, 112), at(0.08f));
}

void Cube(ImDrawList* draw, const Frame& at, ImU32 topColor = Rgb(150, 160, 164)) {
    const ImVec2 top[] = {at(0.5f, 0.1f), at(0.9f, 0.3f), at(0.5f, 0.5f), at(0.1f, 0.3f)};
    const ImVec2 left[] = {at(0.1f, 0.3f), at(0.5f, 0.5f), at(0.5f, 0.92f), at(0.1f, 0.72f)};
    const ImVec2 right[] = {at(0.5f, 0.5f), at(0.9f, 0.3f), at(0.9f, 0.72f), at(0.5f, 0.92f)};
    draw->AddConvexPolyFilled(top, 4, topColor);
    draw->AddConvexPolyFilled(left, 4, Rgb(104, 114, 118));
    draw->AddConvexPolyFilled(right, 4, Rgb(74, 85, 88));
}

void Ball(ImDrawList* draw, const Frame& at, ImU32 base, ImU32 middle, ImU32 highlight) {
    draw->AddCircleFilled(at(0.5f, 0.52f), at(0.4f), base);
    draw->AddCircleFilled(at(0.44f, 0.44f), at(0.28f), middle);
    draw->AddCircleFilled(at(0.38f, 0.36f), at(0.11f), highlight);
}

void Cylinder(ImDrawList* draw, const Frame& at, bool cone) {
    ImU32 side = Rgb(96, 108, 112);
    if (cone) {
        draw->AddTriangleFilled(at(0.5f, 0.08f), at(0.88f, 0.8f), at(0.12f, 0.8f), side);
        draw->AddEllipseFilled(at(0.5f, 0.8f), ImVec2(at(0.38f), at(0.12f)), Rgb(74, 85, 88));
        return;
    }
    draw->AddRectFilled(at(0.15f, 0.26f), at(0.85f, 0.8f), side);
    draw->AddEllipseFilled(at(0.5f, 0.8f), ImVec2(at(0.35f), at(0.12f)), side);
    draw->AddEllipseFilled(at(0.5f, 0.26f), ImVec2(at(0.35f), at(0.12f)), Rgb(150, 160, 164));
}

void Plane(ImDrawList* draw, const Frame& at, bool checkered) {
    const ImVec2 quad[] = {at(0.3f, 0.3f), at(0.95f, 0.3f), at(0.7f, 0.78f), at(0.05f, 0.78f)};
    draw->AddConvexPolyFilled(quad, 4, Rgb(104, 114, 118));
    if (checkered) {
        const ImVec2 a[] = {at(0.3f, 0.3f), at(0.625f, 0.3f), at(0.5f, 0.54f), at(0.175f, 0.54f)};
        const ImVec2 b[] = {at(0.5f, 0.54f), at(0.825f, 0.54f), at(0.7f, 0.78f), at(0.375f, 0.78f)};
        draw->AddConvexPolyFilled(a, 4, Rgb(74, 85, 88));
        draw->AddConvexPolyFilled(b, 4, Rgb(74, 85, 88));
        return;
    }
    draw->AddLine(at(0.5f, 0.3f), at(0.38f, 0.78f), Rgb(150, 160, 164), 1.0f);
    draw->AddLine(at(0.18f, 0.54f), at(0.83f, 0.54f), Rgb(150, 160, 164), 1.0f);
}

void Sun(ImDrawList* draw, const Frame& at) {
    draw->AddCircleFilled(at(0.5f, 0.5f), at(0.2f), kLamp);
    for (int k = 0; k < 8; ++k) {
        float a = static_cast<float>(k) * kPi / 4.0f;
        draw->AddLine(at(0.5f + 0.3f * std::cos(a), 0.5f + 0.3f * std::sin(a)),
                      at(0.5f + 0.44f * std::cos(a), 0.5f + 0.44f * std::sin(a)), kLamp,
                      std::max(1.0f, at(0.07f)));
    }
}

void Bulb(ImDrawList* draw, const Frame& at) {
    draw->AddCircleFilled(at(0.5f, 0.4f), at(0.28f), kLamp);
    draw->AddRectFilled(at(0.38f, 0.64f), at(0.62f, 0.9f), Rgb(150, 160, 164), at(0.04f));
}

void Emitter(ImDrawList* draw, const Frame& at, bool disk) {
    if (disk) {
        draw->AddEllipseFilled(at(0.5f, 0.32f), ImVec2(at(0.38f), at(0.16f)), kLamp);
    } else {
        const ImVec2 quad[] = {at(0.22f, 0.16f), at(0.92f, 0.16f), at(0.78f, 0.46f),
                               at(0.08f, 0.46f)};
        draw->AddConvexPolyFilled(quad, 4, kLamp);
    }
    for (const float u : {0.3f, 0.5f, 0.7f}) {
        draw->AddLine(at(u, 0.56f), at(u, 0.9f), kLamp, std::max(1.0f, at(0.07f)));
    }
}

void Tube(ImDrawList* draw, const Frame& at) {
    draw->AddRectFilled(at(0.08f, 0.38f), at(0.92f, 0.62f), kLamp, at(0.12f));
    draw->AddRectFilled(at(0.02f, 0.42f), at(0.1f, 0.58f), Rgb(150, 160, 164));
    draw->AddRectFilled(at(0.9f, 0.42f), at(0.98f, 0.58f), Rgb(150, 160, 164));
}

void Dome(ImDrawList* draw, const Frame& at) {
    draw->PathArcTo(at(0.5f, 0.72f), at(0.38f), kPi, 2.0f * kPi, 16);
    draw->PathFillConvex(kLamp);
    draw->AddLine(at(0.08f, 0.76f), at(0.92f, 0.76f), kLamp, std::max(1.0f, at(0.08f)));
}

void Globe(ImDrawList* draw, const Frame& at) {
    const ImU32 color = Rgb(127, 166, 194);
    const float width = std::max(1.0f, at(0.07f));
    draw->AddCircle(at(0.5f, 0.5f), at(0.4f), color, 0, width);
    draw->AddEllipse(at(0.5f, 0.5f), ImVec2(at(0.16f), at(0.4f)), color, 0, 0, width);
    draw->AddLine(at(0.1f, 0.5f), at(0.9f, 0.5f), color, width);
}

void Backdrop(ImDrawList* draw, const Frame& at, bool gradient) {
    const ImU32 top = gradient ? Rgb(127, 166, 194) : Rgb(104, 114, 118);
    const ImU32 bottom = gradient ? Rgb(206, 198, 214) : Rgb(104, 114, 118);
    draw->AddRectFilledMultiColor(at(0.1f, 0.14f), at(0.9f, 0.86f), top, top, bottom, bottom);
    draw->AddRect(at(0.1f, 0.14f), at(0.9f, 0.86f), Rgb(74, 85, 88), 0.0f,
                  std::max(1.0f, at(0.06f)));
}

void CameraIcon(ImDrawList* draw, const Frame& at) {
    ImU32 body = Rgb(150, 160, 164);
    draw->AddRectFilled(at(0.08f, 0.32f), at(0.66f, 0.78f), body, at(0.08f));
    draw->AddTriangleFilled(at(0.66f, 0.55f), at(0.94f, 0.32f), at(0.94f, 0.78f), body);
    draw->AddCircleFilled(at(0.26f, 0.22f), at(0.1f), body);
    draw->AddCircleFilled(at(0.48f, 0.2f), at(0.12f), body);
}

void Film(ImDrawList* draw, const Frame& at) {
    ImU32 color = Rgb(127, 166, 194);
    draw->AddRect(at(0.1f, 0.18f), at(0.9f, 0.82f), color, at(0.06f), std::max(1.2f, at(0.08f)));
    for (float u : {0.24f, 0.42f, 0.6f, 0.78f}) {
        draw->AddRectFilled(at(u - 0.05f, 0.25f), at(u + 0.05f, 0.32f), color);
        draw->AddRectFilled(at(u - 0.05f, 0.68f), at(u + 0.05f, 0.75f), color);
    }
}

void Aperture(ImDrawList* draw, const Frame& at) {
    ImU32 color = Rgb(127, 166, 194);
    draw->AddCircle(at(0.5f, 0.5f), at(0.38f), color, 0, std::max(1.2f, at(0.08f)));
    for (int k = 0; k < 6; ++k) {
        float a = static_cast<float>(k) * kPi / 3.0f;
        draw->AddLine(at(0.5f + 0.15f * std::cos(a), 0.5f + 0.15f * std::sin(a)),
                      at(0.5f + 0.38f * std::cos(a + 0.6f), 0.5f + 0.38f * std::sin(a + 0.6f)),
                      color, 1.0f);
    }
}

void Untyped(ImDrawList* draw, const Frame& at) {
    draw->AddCircle(at(0.5f, 0.5f), at(0.24f), Rgb(110, 120, 124), 0, std::max(1.2f, at(0.08f)));
}

bool EndsWith(std::string_view text, std::string_view suffix) {
    return text.size() >= suffix.size() &&
           text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}

void ApplyTheme(const Theme& theme, float scale) {
    ImGuiStyle& style = ImGui::GetStyle();
    style = ImGuiStyle();
    SetMetrics(theme, style);
    SetColors(theme, style);
    style.ScaleAllSizes(scale);
    style.FontSizeBase = theme.fontSize;
    style.FontScaleDpi = scale;
}

void DrawPrimIcon(ImDrawList& list, ImVec2 corner, float size, std::string_view type) {
    ImDrawList* draw = &list;
    Frame at{corner, size};
    if (type == "Xform") {
        Tripod(draw, at);
    } else if (type == "Scope" || type.empty()) {
        Folder(draw, at);
    } else if (type == "Sphere" || type == "Capsule") {
        Ball(draw, at, Rgb(74, 85, 88), Rgb(104, 114, 118), Rgb(170, 178, 182));
    } else if (type == "Cylinder" || type == "Cone") {
        Cylinder(draw, at, type == "Cone");
    } else if (type == "Mesh" || type == "Cube") {
        Cube(draw, at);
    } else if (type == "Plane" || type == "GroundPlane") {
        Plane(draw, at, type == "GroundPlane");
    } else if (type == "GeomSubset") {
        Cube(draw, at, Rgb(150, 128, 178));
    } else if (type == "Material" || EndsWith(type, "Surface")) {
        Ball(draw, at, Rgb(110, 92, 140), Rgb(150, 128, 178), Rgb(206, 198, 214));
    } else if (type == "DistantLight") {
        Sun(draw, at);
    } else if (type == "RectLight" || type == "DiskLight") {
        Emitter(draw, at, type == "DiskLight");
    } else if (type == "CylinderLight") {
        Tube(draw, at);
    } else if (type == "DomeLight") {
        Dome(draw, at);
    } else if (type == "World") {
        Globe(draw, at);
    } else if (type == "ColorBackground" || type == "GradientBackground") {
        Backdrop(draw, at, type == "GradientBackground");
    } else if (EndsWith(type, "Light")) {
        Bulb(draw, at);
    } else if (type == "Camera") {
        CameraIcon(draw, at);
    } else if (type == "RenderSettings" || type == "RenderProduct" || type == "RenderVar") {
        Film(draw, at);
    } else if (EndsWith(type, "Filter")) {
        Aperture(draw, at);
    } else {
        Untyped(draw, at);
    }
}

}
