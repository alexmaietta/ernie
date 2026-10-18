// SPDX-License-Identifier: BSD-3-Clause
/// @file bounds.h
/// @brief Axis-aligned boxes in double precision.
#pragma once

#include <glm/glm.hpp>

#include <cmath>
#include <limits>

namespace ernie {

/// @brief An axis-aligned box; the default one is empty and grows to hold what it is given.
struct Box {
    glm::dvec3 min{std::numeric_limits<double>::infinity()};
    glm::dvec3 max{-std::numeric_limits<double>::infinity()};

    static Box Around(const glm::dvec3& center, const glm::dvec3& halfSize) {
        return {center - halfSize, center + halfSize};
    }

    bool Empty() const {
        return !(min.x <= max.x && min.y <= max.y && min.z <= max.z);
    }
    glm::dvec3 Center() const {
        return 0.5 * (min + max);
    }
    glm::dvec3 Size() const {
        return Empty() ? glm::dvec3(0.0) : max - min;
    }

    Box& Extend(const glm::dvec3& point) {
        min = glm::min(min, point);
        max = glm::max(max, point);
        return *this;
    }
    Box& Extend(const Box& other) {
        if (!other.Empty()) {
            min = glm::min(min, other.min);
            max = glm::max(max, other.max);
        }
        return *this;
    }
    Box Grown(double margin) const {
        return Empty() ? *this : Box{min - margin, max + margin};
    }

    /// @return The box holding this one after `transform` (Arvo's method, exact for affine maps).
    Box Transformed(const glm::dmat4& transform) const {
        if (Empty()) {
            return *this;
        }
        const glm::dvec3 translation(transform[3]);
        Box out{translation, translation};
        for (int column = 0; column < 3; ++column) {
            for (int row = 0; row < 3; ++row) {
                const double a = transform[column][row] * min[column];
                const double b = transform[column][row] * max[column];
                out.min[row] += std::min(a, b);
                out.max[row] += std::max(a, b);
            }
        }
        return out;
    }

    bool operator==(const Box&) const = default;
};

}  // namespace ernie
