// SPDX-License-Identifier: BSD-3-Clause
/// @file main.cpp
/// @brief Calls an installed libernie through its public API, glm included.
#include <ernie/io/raster.h>
#include <ernie/util/version.h>

#include <glm/glm.hpp>

#include <cmath>
#include <cstdio>

int main() {
    const glm::dvec3 north = ernie::BodyDirection(90.0, 0.0);
    if (std::abs(north.z - 1.0) > 1e-12) {
        std::fprintf(stderr, "BodyDirection(90, 0) is not +z\n");
        return 1;
    }
    std::printf("Ernie %s\n", std::string(ernie::Version()).c_str());
    return 0;
}
