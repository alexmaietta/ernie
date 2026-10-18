// SPDX-License-Identifier: BSD-3-Clause
/// @file triangulation_test.cpp
/// @brief Tests of fan triangulation and its checks of USD topology.
#include "ernie/geometry/triangulation.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace ernie {
namespace {

TEST(TriangulateFaces, FansPolygonsAndKeepsTriangles) {
    const std::vector<std::int64_t> counts{3, 4, 5};
    const std::vector<std::int64_t> indices{0, 1, 2, 0, 2, 3, 4, 1, 2, 3, 4, 5};
    EXPECT_EQ(TriangulateFaces(counts, indices, 6),
              (std::vector<std::uint32_t>{0, 1, 2, 0, 2, 3, 0, 3, 4, 1, 2, 3, 1, 3, 4, 1, 4, 5}));
    EXPECT_TRUE(TriangulateFaces({}, {}, 0).empty());
}

TEST(TriangulateFaces, RejectsMalformedTopology) {
    const std::vector<std::int64_t> three{3};
    const std::vector<std::int64_t> triangle{0, 1, 2};
    EXPECT_THROW(TriangulateFaces(std::vector<std::int64_t>{2}, std::vector<std::int64_t>{0, 1}, 3),
                 std::invalid_argument);
    EXPECT_THROW(TriangulateFaces(std::vector<std::int64_t>{4}, triangle, 3),
                 std::invalid_argument);
    EXPECT_THROW(TriangulateFaces(three, std::vector<std::int64_t>{0, 1, 2, 3}, 4),
                 std::invalid_argument);
    EXPECT_THROW(TriangulateFaces(three, triangle, 2), std::invalid_argument);
    EXPECT_THROW(TriangulateFaces(three, std::vector<std::int64_t>{0, -1, 2}, 3),
                 std::invalid_argument);
}

}  // namespace
}  // namespace ernie
