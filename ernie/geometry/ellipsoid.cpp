// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/geometry/ellipsoid.h"

#include <cmath>

namespace ernie {

// The foot x = p / (1 + lambda / a^2) (component-wise) lies on the
// ellipsoid where f(lambda) = sum x_i^2 / a_i^2 - 1 is 0. Starting
// from the radial point's lambda, f falls monotonically, so Newton's
// method converges in a few steps.
glm::dvec3 GeodeticSurfacePoint(const glm::dvec3& point,
                                const glm::dvec3& radii) {
    const glm::dvec3 inverse2 = 1.0 / (radii * radii);
    const glm::dvec3 squares = point * point * inverse2;
    const double norm2 = squares.x + squares.y + squares.z;
    if (!(norm2 > 0.0)) {
        return point;
    }
    const double ratio = std::sqrt(1.0 / norm2);
    const glm::dvec3 radial = point * ratio;
    if (norm2 < 0.01) {
        return radial;
    }
    const glm::dvec3 gradient = radial * inverse2 * 2.0;
    double lambda = (1.0 - ratio) * glm::length(point) /
                    (0.5 * glm::length(gradient));
    glm::dvec3 multiplier(1.0);
    for (int step = 0; step < 32; ++step) {
        multiplier = 1.0 / (1.0 + lambda * inverse2);
        const glm::dvec3 m2 = multiplier * multiplier;
        const double f = glm::dot(squares, m2) - 1.0;
        if (std::abs(f) < 1e-12) {
            break;
        }
        const double derivative =
            -2.0 * glm::dot(squares * m2 * multiplier, inverse2);
        lambda -= f / derivative;
    }
    return point * multiplier;
}

glm::dvec3 GeodeticNormal(const glm::dvec3& surface,
                          const glm::dvec3& radii) {
    return glm::normalize(surface / (radii * radii));
}

}  // namespace ernie
