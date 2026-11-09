// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <string_view>

namespace ernie::pipeline {

enum class SignalType : uint8_t {
    ElectronRate,
    Electrons,
    Digital,
    LinearColor,
    DisplayColor,
    Radiance,
    Returns,
    PointCloud,
    RadarCube,
    CaptureStat,
};

inline constexpr std::string_view kSignalNames[] = {
    "ElectronRate", "Electrons", "Digital",    "LinearColor", "DisplayColor",
    "Radiance",     "Returns",   "PointCloud", "RadarCube",   "CaptureStat",
};

inline constexpr std::string_view kSignalTypes[] = {
    "ElectronRate", "Electrons", "Digital",     "LinearColor", "DisplayColor",
    "Radiance",     "Returns",   "CloudSignal", "RadarCube",   "CaptureStat",
};

inline constexpr uint32_t kReturnsChannels = 8;

inline constexpr std::string_view kSignalPortNames[] = {
    "electronRate", "electrons", "digital", "color", "display",
    "radiance",     "returns",   "cloud",   "cube",  "stat",
};

enum class ElementType : uint8_t {
    F32,
    F16,
    U16,
    U8,
};

inline constexpr uint32_t kElementBytes[] = {4, 2, 2, 1};

inline constexpr uint32_t kStatWords = 16;
inline constexpr uint32_t kReduceGroups = 64;
inline constexpr uint32_t kStatStorageWords = (1 + kReduceGroups) * kStatWords;

struct SignalLayout {
    uint32_t channels = 1;
    uint32_t pattern = 0;
    bool mosaic = false;
    ElementType element = ElementType::F32;
    bool perCapture = false;

    bool operator==(const SignalLayout&) const = default;
};

}
