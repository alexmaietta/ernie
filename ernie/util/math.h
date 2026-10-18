// SPDX-License-Identifier: BSD-3-Clause
/// @file math.h
/// @brief Small scalar functions shared across Ernie's numerics.
#pragma once

#include <glm/glm.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <span>
#include <utility>

namespace ernie {

/// @brief The fractional part of a number.
/// @param x The number.
/// @return x minus its floor, in [0, 1).
inline double Fract(double x) {
    return x - std::floor(x);
}

/// @brief Floored modulo, which wraps negative numbers into [0, m).
/// @param x The number.
/// @param m The modulus.
/// @return x modulo m, with the sign of m.
inline double Mod(double x, double m) {
    return x - m * std::floor(x / m);
}

/// @brief Hermite interpolation from 0 to 1 between two edges.
/// @param edge0 Where the result starts rising from 0.
/// @param edge1 Where the result reaches 1.
/// @param x     The value.
/// @return 0 below edge0, 1 above edge1, and a smooth cubic between.
inline double SmoothStep(double edge0, double edge1, double x) {
    const double t = std::clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

/// @brief Trilinear interpolation between the eight corners of a cell.
/// @param corners The corner values, indexed x + 2y + 4z.
/// @param t       The position within the cell, each component in [0, 1].
/// @return The interpolated value.
inline double Trilinear(const std::array<double, 8>& corners, const glm::dvec3& t) {
    const double z0 = std::lerp(std::lerp(corners[0], corners[1], t.x),
                                std::lerp(corners[2], corners[3], t.x), t.y);
    const double z1 = std::lerp(std::lerp(corners[4], corners[5], t.x),
                                std::lerp(corners[6], corners[7], t.x), t.y);
    return std::lerp(z0, z1, t.z);
}

/// @brief Cosine spacing: sample positions in [0, 1] that crowd towards both ends.
/// @param i     The sample.
/// @param count How many samples, at least 2.
/// @return (1 - cos(pi i / (count - 1))) / 2.
inline double CosineSpacing(std::size_t i, std::size_t count) {
    const double u = static_cast<double>(i) / static_cast<double>(count - 1);
    return 0.5 * (1.0 - std::cos(std::numbers::pi * u));
}

/// @brief The interval of a table holding a position, and the fraction across it.
/// @param xs The sample positions, increasing, at least two.
/// @param x  The position.
/// @return The interval's first index and the fraction across it, clamped to [0, 1]; 0 across an
///         interval of no width.
inline std::pair<std::size_t, double> Bracket(std::span<const double> xs, double x) {
    const std::ptrdiff_t upper = std::upper_bound(xs.begin(), xs.end(), x) - xs.begin();
    const auto i = static_cast<std::size_t>(
        std::clamp<std::ptrdiff_t>(upper - 1, 0, static_cast<std::ptrdiff_t>(xs.size()) - 2));
    const double width = xs[i + 1] - xs[i];
    return {i, width > 0.0 ? std::clamp((x - xs[i]) / width, 0.0, 1.0) : 0.0};
}

/// @brief Piecewise-linear interpolation of a table, constant beyond its ends.
/// @param xs     The sample positions, increasing.
/// @param values The value at each position.
/// @param x      Where to interpolate.
/// @return The interpolated value, or a default value for an empty table.
template <typename T>
T Interpolate(std::span<const double> xs, std::span<const T> values, double x) {
    if (xs.empty()) {
        return T{};
    }
    if (x <= xs.front()) {
        return values.front();
    }
    if (x >= xs.back()) {
        return values.back();
    }
    const auto [i, t] = Bracket(xs, x);
    return values[i] + (values[i + 1] - values[i]) * t;
}

}  // namespace ernie
