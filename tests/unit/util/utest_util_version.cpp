// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/version.h"

#include <gtest/gtest.h>

namespace ernie {
namespace {

TEST(VersionTest, ComparesMajorThenMinorThenPatch) {
    const Version a{6, 2};
    const Version b{6, 10};
    const Version c{7, 2, 1};
    EXPECT_TRUE(a < b);
    EXPECT_TRUE(b < c);
    EXPECT_FALSE(c < a);
    EXPECT_TRUE(c > a);
    EXPECT_TRUE(a <= a);
    EXPECT_FALSE(c <= a);
    EXPECT_TRUE(c >= c);
    EXPECT_FALSE(a >= c);
    EXPECT_TRUE(a == (Version{6, 2, 0}));
    EXPECT_TRUE(a != b);
    EXPECT_FALSE(a != a);
}

TEST(VersionTest, PrintsThePatchOnlyWhenSet) {
    EXPECT_EQ((Version{12, 0}).ToString(), "12.0");
    EXPECT_EQ((Version{7, 2, 53211}).ToString(), "7.2.53211");
}

TEST(VersionTest, BuildInfoMatchesTheProject) {
    const BuildInfo& build = GetBuildInfo();
    EXPECT_EQ(build.version, (Version{ERNIE_TEST_VERSION_MAJOR, ERNIE_TEST_VERSION_MINOR,
                                      ERNIE_TEST_VERSION_PATCH}));
    EXPECT_EQ(build.fullVersion, ERNIE_TEST_VERSION_FULL);
    EXPECT_FALSE(build.commit.empty());
    EXPECT_EQ(build.buildDate.size(), 20u);
    EXPECT_EQ(&GetBuildInfo(), &build);
}

}
}
