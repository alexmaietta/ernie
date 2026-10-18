// SPDX-License-Identifier: BSD-3-Clause
/// @file common.h
/// @brief Helpers shared by the tests: temporary and fixture paths, small files and tensors, and
/// checks of boxes, text and changes.
#pragma once

#include "ernie/geometry/bounds.h"
#include "ernie/scene/storage.h"
#include "ernie/util/file.h"
#include "ernie/util/tensor.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

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
    WriteTextFile(path, bytes);
    return path;
}

/// @return A float32 CPU tensor of `values` in `shape`.
inline Tensor Floats(const std::vector<float>& values, std::vector<std::int64_t> shape) {
    return Tensor::Copy<float>(values, std::move(shape));
}

/// @return A flat int32 CPU tensor of `values`.
inline Tensor Ints(const std::vector<std::int32_t>& values) {
    return Tensor::Copy<std::int32_t>(values, {static_cast<std::int64_t>(values.size())});
}

inline bool Contains(std::string_view text, std::string_view part) {
    return text.find(part) != std::string_view::npos;
}

/// @return Whether `changes` report the value of `attribute` changed at the bucket's row.
inline bool ValueChanged(const ChangeSet& changes, BucketId bucket, AttributeId attribute,
                         std::uint32_t row) {
    for (const ChangeSet::Values& values : changes.values) {
        if (values.bucket == bucket && values.attribute == attribute) {
            for (const RowRange range : values.rows) {
                if (range.begin <= row && row < range.end) {
                    return true;
                }
            }
        }
    }
    return false;
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
