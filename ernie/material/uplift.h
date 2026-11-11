// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "ernie/util/disk_cache.h"

namespace ernie {

inline constexpr uint32_t kSigmoidResolution = 32;
inline constexpr float kSigmoidFirstNm = 360;
inline constexpr float kSigmoidLastNm = 830;

struct SigmoidTable {
    std::vector<float> scales;
    std::vector<float> coefficients;
};

SigmoidTable CachedSigmoids(const DiskCache& cache);

const SigmoidTable& Sigmoids();

std::array<float, 3> FitSigmoid(std::array<float, 3> rgb, std::array<float, 3> start);

float SigmoidValue(const std::array<float, 3>& coefficients, float nanometers);

std::array<float, 3> SigmoidRgb(const std::array<float, 3>& coefficients);

inline constexpr float kLinearUpliftFirstNm = 300;
inline constexpr float kLinearUpliftStepNm = 5;
inline constexpr uint32_t kLinearUpliftSamples = 181;

std::vector<std::array<float, 3>> LinearUplift();

}
