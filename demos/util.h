// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Small helpers the C++ demos share, as ernie.demo's helpers do the
// Python demos'.

#include "ernie/util/tensor.h"

#include <glm/vec3.hpp>

#include <span>
#include <stdexcept>
#include <string>

namespace ernie::demos {

/// uint8 (rows, cols, channels) images of one height and channel
/// count next to each other, with black gaps of `gap` columns.
Tensor SideBySide(std::span<const Tensor> images, int gap = 4);

/// An sRGB-encoded colour in [0, 1] to linear.
glm::dvec3 SrgbToLinear(glm::dvec3 srgb);

/// The mean colour of a uint8 (rows, cols, 3 or 4) image, decoded
/// from sRGB to linear.
glm::dvec3 MeanLinearRgb(const Tensor& image);

/// The standard deviation of a uint8 image's colour channels (alpha
/// left out), for checking an image is not flat.
double ColourDeviation(const Tensor& image);

/// Throws std::runtime_error with `message` unless `condition` holds:
/// a demo's checks.
inline void Check(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace ernie::demos
