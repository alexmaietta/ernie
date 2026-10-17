// SPDX-License-Identifier: BSD-3-Clause
/// @file version_test.cpp
/// @brief Tests of Ernie's version string.
#include "ernie/core/version.h"

#include <gtest/gtest.h>

#include <regex>
#include <string>

TEST(Version, IsSemanticWithOptionalDevSuffix) {
    const std::string version(ernie::Version());
    EXPECT_TRUE(std::regex_match(version, std::regex(R"(\d+\.\d+\.\d+(\.dev\d+)?)"))) << version;
}
