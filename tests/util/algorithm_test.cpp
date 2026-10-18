// SPDX-License-Identifier: BSD-3-Clause
/// @file algorithm_test.cpp
/// @brief Tests of the range helpers.
#include "ernie/util/algorithm.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace ernie {
namespace {

TEST(Algorithm, MapTransformsEachElementInOrder) {
    const std::vector<int> numbers{1, 2, 3};
    EXPECT_EQ(Map(numbers, [](int n) { return std::to_string(n * 2); }),
              (std::vector<std::string>{"2", "4", "6"}));
    EXPECT_TRUE(Map(std::vector<int>{}, [](int n) { return n; }).empty());
}

}  // namespace
}  // namespace ernie
