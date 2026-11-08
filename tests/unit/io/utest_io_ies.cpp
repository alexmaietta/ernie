// SPDX-License-Identifier: BSD-3-Clause
#include <string>

#include <gtest/gtest.h>

#include "ernie/io/ies.h"
#include "ernie/util/error.h"

namespace ernie {
namespace {

float Sample(const Image& image, double around, double polar) {
    return image.Pixel(uint32_t(around), uint32_t(polar)).r;
}

TEST(IesTest, SymmetricProfilesResampleByPolarAngleRelativeToTheirPeak) {
    const Image image = ReadIes("IESNA:LM-63-2002\n[TEST] cone\nTILT=NONE\n"
                                "1 1000 2 3 1 1 2 0 0 0\n1 1 100\n"
                                "0 90 180\n0\n100 50 0\n");
    ASSERT_EQ(image.width, kIesColumns);
    ASSERT_EQ(image.height, kIesRows);
    EXPECT_FLOAT_EQ(Sample(image, 0, 0), 1);
    EXPECT_FLOAT_EQ(Sample(image, 200, 45), 0.75f);
    EXPECT_FLOAT_EQ(Sample(image, 90, 90), 0.5f);
    EXPECT_FLOAT_EQ(Sample(image, 360, 180), 0);
}

TEST(IesTest, HorizontalSymmetryFoldsTheAngleAround) {
    const std::string planes = "1 1000 1 2 2 1 2 0 0 0\n1 1 100\n0 180\n0, 90\n100 100\n0 0\n";
    const Image quadrant = ReadIes("TILT=NONE\n" + planes);
    EXPECT_FLOAT_EQ(Sample(quadrant, 0, 30), 1);
    EXPECT_FLOAT_EQ(Sample(quadrant, 45, 30), 0.5f);
    EXPECT_FLOAT_EQ(Sample(quadrant, 90, 30), 0);
    EXPECT_FLOAT_EQ(Sample(quadrant, 180, 30), 1);
    EXPECT_FLOAT_EQ(Sample(quadrant, 270, 30), 0);

    const Image full = ReadIes("TILT=NONE\n1 1000 1 2 3 1 2 0 0 0\n1 1 100\n0 180\n0 90 270\n"
                               "100 100\n0 0\n50 50\n");
    EXPECT_FLOAT_EQ(Sample(full, 270, 10), 0.5f);
    EXPECT_FLOAT_EQ(Sample(full, 315, 10), 0.75f);
}

TEST(IesTest, TiltedProfilesSkipTheirTiltTable) {
    const Image image = ReadIes("TILT=INCLUDE\n1\n3\n0 45 90\n1 0.9 0.8\n"
                                "1 1000 1 2 1 1 2 0 0 0\n1 1 100\n0 180\n0\n40 20\n");
    EXPECT_FLOAT_EQ(Sample(image, 0, 0), 1);
    EXPECT_FLOAT_EQ(Sample(image, 0, 90), 0.75f);
}

TEST(IesTest, UnsupportedOrTruncatedProfilesThrow) {
    EXPECT_THROW(ReadIes("no tilt here"), Error);
    EXPECT_THROW(ReadIes("TILT=NONE\n1 1000 1 2 1 2 2 0 0 0\n1 1 100\n0 180\n0\n1 0\n"), Error);
    EXPECT_THROW(ReadIes("TILT=NONE\n1 1000 1 2 1 1 2 0 0 0\n1 1 100\n0 180\n0\n1\n"), Error);
    EXPECT_TRUE(IsIes("lamp.IES"));
    EXPECT_FALSE(IsIes("lamp.png"));
}

}
}
