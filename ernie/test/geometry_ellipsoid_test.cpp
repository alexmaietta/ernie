// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/geometry/ellipsoid.h"

#include <glm/glm.hpp>
#include <gtest/gtest.h>

#include <cmath>

namespace {

// A point at a height along the normal of a triaxial ellipsoid's
// surface point projects back to that point, at any height down to
// near the centre.
TEST(Ellipsoid, GeodeticSurfacePointIsTheFootOfTheNormal) {
    const glm::dvec3 radii(6378137.0, 6370000.0, 6356752.3);
    for (const glm::dvec3& direction :
         {glm::dvec3(1.0, 0.0, 0.0), glm::dvec3(0.3, -0.4, 0.8),
          glm::dvec3(-0.1, 0.9, -0.3), glm::dvec3(0.0, 0.0, -1.0)}) {
        const glm::dvec3 d = glm::normalize(direction);
        const glm::dvec3 surface =
            d / std::sqrt(glm::dot(d / radii, d / radii));
        const glm::dvec3 normal =
            ernie::GeodeticNormal(surface, radii);
        EXPECT_NEAR(glm::length(normal), 1.0, 1e-12);
        for (const double height :
             {0.0, 2.0, 8848.0, 400e3, -100e3}) {
            const glm::dvec3 foot = ernie::GeodeticSurfacePoint(
                surface + height * normal, radii);
            // Newton stops within a part in 10^12 of the radius.
            EXPECT_LT(glm::distance(foot, surface), 1e-5)
                << "height " << height;
        }
    }
    EXPECT_EQ(ernie::GeodeticSurfacePoint(glm::dvec3(0.0), radii),
              glm::dvec3(0.0));
}

}  // namespace
