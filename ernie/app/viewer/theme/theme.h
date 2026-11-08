// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <string_view>

#include <imgui.h>

namespace ernie::app {

struct Theme {
    ImVec4 background{0.788f, 0.800f, 0.773f, 1.0f};
    ImVec4 panel{0.831f, 0.843f, 0.816f, 1.0f};
    ImVec4 header{0.776f, 0.788f, 0.761f, 1.0f};
    ImVec4 field{1.0f, 1.0f, 1.0f, 1.0f};
    ImVec4 fieldHovered{0.957f, 0.965f, 0.945f, 1.0f};
    ImVec4 button{0.863f, 0.875f, 0.847f, 1.0f};
    ImVec4 buttonLit{0.698f, 0.761f, 0.835f, 1.0f};
    ImVec4 separator{0.667f, 0.678f, 0.651f, 1.0f};
    ImVec4 border{0.553f, 0.565f, 0.537f, 1.0f};
    ImVec4 text{0.0f, 0.0f, 0.0f, 1.0f};
    ImVec4 textMuted{0.333f, 0.345f, 0.325f, 1.0f};
    ImVec4 selection{0.627f, 0.741f, 0.863f, 1.0f};
    ImVec4 accent{0.122f, 0.290f, 0.541f, 1.0f};
    ImVec4 outline{1.0f, 0.62f, 0.16f, 1.0f};
    ImVec4 warning{0.910f, 0.698f, 0.235f, 1.0f};
    ImVec4 error{0.886f, 0.341f, 0.318f, 1.0f};
    ImVec4 overlay{0.831f, 0.843f, 0.816f, 0.92f};
    ImVec4 overlayBorder{0.553f, 0.565f, 0.537f, 1.0f};
    ImVec4 checkBox{1.0f, 1.0f, 1.0f, 1.0f};
    ImVec4 checkMark{0.0f, 0.0f, 0.0f, 1.0f};
    float fontSize = 14.0f;
    float iconSize = 15.0f;
    float rounding = 0.0f;
    float frameBorder = 1.0f;
    ImVec2 framePadding{6.0f, 3.0f};
    ImVec2 itemSpacing{6.0f, 4.0f};
    float propertyLabelWidth = 130.0f;
};

void ApplyTheme(const Theme& theme, float scale);

void DrawPrimIcon(ImDrawList& draw, ImVec2 corner, float size, std::string_view type);

inline ImVec4 WithAlpha(const ImVec4& color, float alpha) {
    return {color.x, color.y, color.z, alpha};
}

inline constexpr ImU32 kAxisColors[3] = {IM_COL32(222, 74, 64, 255), IM_COL32(110, 190, 70, 255),
                                         IM_COL32(70, 130, 230, 255)};

}

namespace ernie::app::icons {

inline constexpr const char* kCamera = "\xee\x81\xa4";
inline constexpr const char* kEye = "\xee\x82\xba";
inline constexpr const char* kSun = "\xee\x85\xb8";
inline constexpr const char* kLightbulb = "\xee\x87\x82";
inline constexpr const char* kLayers = "\xee\x94\xa9";
inline constexpr const char* kSettings = "\xee\x85\x94";
inline constexpr const char* kBox = "\xee\x81\xa1";
inline constexpr const char* kFocus = "\xee\x8a\x9e";
inline constexpr const char* kMonitor = "\xee\x84\x9d";
inline constexpr const char* kChartLine = "\xee\x8a\xa5";
inline constexpr const char* kSlidersHorizontal = "\xee\x8a\x9a";
inline constexpr const char* kListTree = "\xee\x90\x88";
inline constexpr const char* kAperture = "\xee\x81\x80";
inline constexpr const char* kImage = "\xee\x83\xb6";
inline constexpr const char* kRotateCcw = "\xee\x85\x88";
inline constexpr const char* kX = "\xee\x86\xb2";
inline constexpr const char* kFlashlight = "\xee\x83\x93";
inline constexpr const char* kSave = "\xee\x85\x8d";
inline constexpr const char* kFolderOpen = "\xee\x89\x87";
inline constexpr const char* kMousePointer = "\xee\x87\x83";
inline constexpr const char* kMove = "\xee\x84\xa1";
inline constexpr const char* kRotateCw = "\xee\x85\x89";
inline constexpr const char* kScaling = "\xee\x8b\xac";
inline constexpr const char* kGlobe = "\xee\x83\xa8";
inline constexpr const char* kFolder = "\xee\x83\x97";
inline constexpr const char* kFile = "\xee\x83\x80";
inline constexpr const char* kArrowUp = "\xee\x81\x8a";
inline constexpr const char* kHome = "\xee\x83\xb5";
inline constexpr const char* kHardDrive = "\xee\x83\xad";
inline constexpr const char* kHistory = "\xee\x87\xb5";
inline constexpr const char* kPlay = "\xee\x84\xbc";
inline constexpr const char* kPause = "\xee\x84\xae";
inline constexpr const char* kSquare = "\xee\x85\xa7";
inline constexpr const char* kSkipBack = "\xee\x85\x9f";
inline constexpr const char* kSkipForward = "\xee\x85\xa0";
inline constexpr const char* kStepBack = "\xee\x8f\xa9";
inline constexpr const char* kStepForward = "\xee\x8f\xaa";
inline constexpr const char* kRepeat = "\xee\x85\x86";
inline constexpr const char* kVideo = "\xee\x86\xa5";
inline constexpr const char* kDownload = "\xee\x82\xb2";

inline constexpr ImWchar kFirst = 0xE000;
inline constexpr ImWchar kLast = 0xF8FF;

}
