// SPDX-License-Identifier: BSD-3-Clause
/// @file common.h
/// @brief Helpers shared by the tests: temporary and fixture paths, small files and box checks.
#pragma once

#include "ernie/geometry/bounds.h"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>

namespace ernie::test {

/// @return `name` in the test run's temporary directory.
inline std::filesystem::path TempPath(const std::string& name) {
    return std::filesystem::path(testing::TempDir()) / name;
}

/// @return `relative` under tests/fixtures.
inline std::filesystem::path FixturePath(const std::filesystem::path& relative) {
    return std::filesystem::path(ERNIE_TEST_FIXTURES) / relative;
}

/// @brief Writes bytes to a file, replacing it.
/// @return The path.
inline std::filesystem::path WriteFile(const std::filesystem::path& path, std::string_view bytes) {
    std::ofstream(path, std::ios::binary)
        .write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    return path;
}

/// @return Success when each of the box's corners is within `tolerance` of the expected ones;
/// use as `EXPECT_TRUE(BoxNear(...)) << why`.
inline testing::AssertionResult BoxNear(const Box& box, const glm::dvec3& min,
                                        const glm::dvec3& max, double tolerance = 1e-12) {
    for (int i = 0; i < 3; ++i) {
        if (std::abs(box.min[i] - min[i]) > tolerance ||
            std::abs(box.max[i] - max[i]) > tolerance) {
            std::ostringstream text;
            text << "box (" << box.min.x << ", " << box.min.y << ", " << box.min.z << ") to ("
                 << box.max.x << ", " << box.max.y << ", " << box.max.z << ")";
            return testing::AssertionFailure() << text.str();
        }
    }
    return testing::AssertionSuccess();
}

}  // namespace ernie::test
