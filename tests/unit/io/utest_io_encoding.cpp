// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/io/kernels/encoding.h"

#include <cmath>
#include <limits>

#include <gtest/gtest.h>

namespace ernie {
namespace {

constexpr float kInfinity = std::numeric_limits<float>::infinity();
constexpr float kNan = std::numeric_limits<float>::quiet_NaN();

struct HalfCase {
    float value;
    uint half;
};

constexpr HalfCase kExactHalves[] = {
    {0.0f, 0x0000},         {-0.0f, 0x8000},     {1.0f, 0x3C00},
    {-2.0f, 0xC000},        {65504.0f, 0x7BFF},  {0x1p-24f, 0x0001},
    {0x1.ff8p-15f, 0x03FF}, {kInfinity, 0x7C00}, {-kInfinity, 0xFC00},
};

constexpr HalfCase kRoundedHalves[] = {
    {0x1.002p+0f, 0x3C00}, {0x1.006p+0f, 0x3C02}, {65520.0f, 0x7C00},   {1e6f, 0x7C00},
    {kNan, 0x7E00},        {0x1p-40f, 0x0000},    {-0x1p-40f, 0x8000},  {0x1p-25f, 0x0000},
    {0x1.8p-24f, 0x0002},  {0x1.4p-24f, 0x0001},  {0x1.cp-24f, 0x0002},
};

TEST(EncodingTest, BitCastsBetweenFloatAndUint) {
    EXPECT_EQ(asuint(1.0f), 0x3F800000u);
    EXPECT_EQ(asfloat(0x40000000u), 2.0f);
}

TEST(EncodingTest, DecodesHalvesOfEveryKind) {
    for (const HalfCase& expected : kExactHalves) {
        EXPECT_EQ(asuint(HalfToFloat(expected.half)), asuint(expected.value)) << expected.half;
    }
    EXPECT_TRUE(std::isnan(HalfToFloat(0x7E00)));
}

TEST(EncodingTest, EncodesExactHalves) {
    for (const HalfCase& expected : kExactHalves) {
        EXPECT_EQ(FloatToHalf(expected.value), expected.half) << expected.value;
    }
}

TEST(EncodingTest, RoundsToNearestEvenAndSaturates) {
    for (const HalfCase& expected : kRoundedHalves) {
        EXPECT_EQ(FloatToHalf(expected.value), expected.half) << expected.value;
    }
}

TEST(EncodingTest, RoundTripsEveryFiniteHalf) {
    for (uint bits = 0; bits < 0x10000; ++bits) {
        if ((bits & 0x7C00) != 0x7C00) {
            ASSERT_EQ(FloatToHalf(HalfToFloat(bits)), bits);
        }
    }
}

TEST(EncodingTest, ConvertsSrgbBothWays) {
    EXPECT_FLOAT_EQ(DecodeSrgb(0.02f), 0.02f / 12.92f);
    EXPECT_FLOAT_EQ(DecodeSrgb(1.0f), 1.0f);
    EXPECT_FLOAT_EQ(EncodeSrgb(0.001f), 0.01292f);
    EXPECT_FLOAT_EQ(EncodeSrgb(1.0f), 1.0f);
    EXPECT_EQ(EncodeSrgb(-1.0f), 0.0f);
    EXPECT_FLOAT_EQ(EncodeSrgb(2.0f), 1.0f);
    EXPECT_NEAR(EncodeSrgb(DecodeSrgb(0.5f)), 0.5f, 1e-6f);
}

TEST(EncodingTest, EncodesUnorm8WithClamping) {
    EXPECT_EQ(EncodeUnorm8(-1.0f), 0u);
    EXPECT_EQ(EncodeUnorm8(0.5f), 128u);
    EXPECT_EQ(EncodeUnorm8(1.0f), 255u);
    EXPECT_EQ(EncodeUnorm8(9.0f), 255u);
}

}
}
