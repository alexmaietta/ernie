// SPDX-License-Identifier: BSD-3-Clause
#pragma once

/// Lucide icon glyphs (lucide-static 1.52.0, private-use codepoints)
/// as UTF-8 strings, for text drawn with the viewer's font, which has
/// the icons merged in: `ImGui::Button(icons::kSave)`, or
/// `std::string(icons::kCamera) + " Camera"` as a label.
namespace ernie::app::icons {

inline constexpr const char* kCamera = "\xee\x81\xa4";     ///< U+E064
inline constexpr const char* kEye = "\xee\x82\xba";        ///< U+E0BA
inline constexpr const char* kSun = "\xee\x85\xb8";        ///< U+E178
inline constexpr const char* kLightbulb = "\xee\x87\x82";  ///< U+E1C2
inline constexpr const char* kLayers = "\xee\x94\xa9";     ///< U+E529
inline constexpr const char* kSettings = "\xee\x85\x94";   ///< U+E154
inline constexpr const char* kBox = "\xee\x81\xa1";        ///< U+E061
inline constexpr const char* kFocus = "\xee\x8a\x9e";      ///< U+E29E
inline constexpr const char* kMonitor = "\xee\x84\x9d";    ///< U+E11D
inline constexpr const char* kChartLine = "\xee\x8a\xa5";  ///< U+E2A5
inline constexpr const char* kSlidersHorizontal =
    "\xee\x8a\x9a";                                        ///< U+E29A
inline constexpr const char* kListTree = "\xee\x90\x88";   ///< U+E408
inline constexpr const char* kAperture = "\xee\x81\x80";   ///< U+E040
inline constexpr const char* kImage = "\xee\x83\xb6";      ///< U+E0F6
inline constexpr const char* kRotateCcw = "\xee\x85\x88";  ///< U+E148
inline constexpr const char* kX = "\xee\x86\xb2";          ///< U+E1B2
inline constexpr const char* kFlashlight =
    "\xee\x83\x93";                                   ///< U+E0D3
inline constexpr const char* kSave = "\xee\x85\x8d";  ///< U+E14D
inline constexpr const char* kFolderOpen =
    "\xee\x89\x87";  ///< U+E247
inline constexpr const char* kMousePointer =
    "\xee\x87\x83";  ///< U+E1C3 mouse-pointer-2
inline constexpr const char* kMove = "\xee\x84\xa1";       ///< U+E121
inline constexpr const char* kRotateCw = "\xee\x85\x89";   ///< U+E149
inline constexpr const char* kScaling = "\xee\x8b\xac";    ///< U+E2EC
inline constexpr const char* kGlobe = "\xee\x83\xa8";      ///< U+E0E8
inline constexpr const char* kFolder = "\xee\x83\x97";     ///< U+E0D7
inline constexpr const char* kFile = "\xee\x83\x80";       ///< U+E0C0
inline constexpr const char* kArrowUp = "\xee\x81\x8a";    ///< U+E04A
inline constexpr const char* kHome = "\xee\x83\xb5";       ///< U+E0F5
inline constexpr const char* kHardDrive = "\xee\x83\xad";  ///< U+E0ED
inline constexpr const char* kHistory = "\xee\x87\xb5";    ///< U+E1F5
inline constexpr const char* kPlay = "\xee\x84\xbc";       ///< U+E13C
inline constexpr const char* kPause = "\xee\x84\xae";      ///< U+E12E
inline constexpr const char* kSquare = "\xee\x85\xa7";     ///< U+E167
inline constexpr const char* kSkipBack = "\xee\x85\x9f";   ///< U+E15F
inline constexpr const char* kSkipForward =
    "\xee\x85\xa0";                                       ///< U+E160
inline constexpr const char* kStepBack = "\xee\x8f\xa9";  ///< U+E3E9
inline constexpr const char* kStepForward =
    "\xee\x8f\xaa";                                     ///< U+E3EA
inline constexpr const char* kRepeat = "\xee\x85\x86";  ///< U+E146
inline constexpr const char* kVideo = "\xee\x86\xa5";   ///< U+E1A5
inline constexpr const char* kDownload =
    "\xee\x82\xb2";  ///< U+E0B2, saving (captures)

/// First and last codepoints of the icon font, for loading its
/// glyphs.
inline constexpr unsigned kFirst = 0xE000;
inline constexpr unsigned kLast = 0xF8FF;

}  // namespace ernie::app::icons
