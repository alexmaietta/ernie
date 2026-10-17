// SPDX-License-Identifier: BSD-3-Clause
/// @file version_test.cpp
/// @brief Tests of Ernie's version and build information.
#include "ernie/core/version.h"

#include <gtest/gtest.h>

#include <regex>
#include <string>

namespace ernie {
namespace {

TEST(Version, IsSemanticWithOptionalDevSuffix) {
    const std::string version(Version());
    EXPECT_TRUE(std::regex_match(version, std::regex(R"(\d+\.\d+\.\d+(\.dev\d+)?)"))) << version;
}

TEST(Version, BuildInformationIsFilledIn) {
    EXPECT_FALSE(BuildDate().empty());
    EXPECT_FALSE(BuildType().empty());
    EXPECT_NE(Compiler(), "unknown");
}

}  // namespace
}  // namespace ernie
