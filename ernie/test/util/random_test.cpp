// SPDX-License-Identifier: BSD-3-Clause
/// @file random_test.cpp
/// @brief Tests of the deterministic random streams.
#include "ernie/util/random.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <set>

namespace ernie {
namespace {

static_assert(HashStream("") == 2166136261u, "FNV-1a offset basis");
static_assert(HashStream("a") == 0xe40c292cu, "FNV-1a of a");
static_assert(SampleBits({1, 2, 3, 4}, 5) == SampleBits({1, 2, 3, 4}, 5), "usable at compile time");

TEST(Random, Pcg3dMixesEveryInputWord) {
    const glm::uvec3 base = Pcg3d(glm::uvec3(0u));
    for (int word = 0; word < 3; ++word) {
        glm::uvec3 input(0u);
        input[word] = 1u;
        const glm::uvec3 h = Pcg3d(input);
        for (int out = 0; out < 3; ++out) {
            EXPECT_NE(h[out], base[out]) << "input " << word << " output " << out;
        }
    }
}

TEST(Random, StreamsOfProductPathsDiffer) {
    EXPECT_NE(HashStream("/Render/Left"), HashStream("/Render/Right"));
    EXPECT_EQ(HashStream("/Render/Left"), HashStream("/Render/Left"));
}

TEST(Random, FrameSeedsAreDistinctAcrossSeedsStreamsAndFrames) {
    std::set<std::uint32_t> seeds;
    int count = 0;
    for (std::uint32_t seed = 0; seed < 8; ++seed) {
        for (std::uint32_t stream = 0; stream < 8; ++stream) {
            for (std::uint64_t frame = 0; frame < 64; ++frame) {
                seeds.insert(FrameSeed(seed, stream, frame));
                ++count;
            }
        }
    }
    EXPECT_EQ(seeds.size(), static_cast<std::size_t>(count));
}

TEST(Random, FrameSeedUsesTheHighWordOfTheFrame) {
    EXPECT_NE(FrameSeed(0, 0, 1), FrameSeed(0, 0, (std::uint64_t{1} << 32u) + 1u));
}

TEST(Random, SampleBitsDependOnEveryPartOfTheKey) {
    const SampleKey key{11, 3, 1000, 7};
    const std::uint32_t base = SampleBits(key, 2);
    EXPECT_NE(SampleBits({12, 3, 1000, 7}, 2), base);
    EXPECT_NE(SampleBits({11, 4, 1000, 7}, 2), base);
    EXPECT_NE(SampleBits({11, 3, 1001, 7}, 2), base);
    EXPECT_NE(SampleBits({11, 3, 1000, 8}, 2), base);
    EXPECT_NE(SampleBits(key, 3), base);
}

// A chi-square test over 64 bins: 65,536 uniforms per dimension stay within the 99.9th percentile
// of chi-square with 63 degrees of freedom (103.4).
TEST(Random, SampleUniformsAreUniformInEveryDimension) {
    constexpr int kBins = 64;
    constexpr int kSamples = 65536;
    for (std::uint32_t dimension = 0; dimension < 8; ++dimension) {
        std::array<int, kBins> counts{};
        for (std::uint32_t pixel = 0; pixel < kSamples; ++pixel) {
            const float u = SampleUniform({5, 0, pixel, 0}, dimension);
            ASSERT_GE(u, 0.0f);
            ASSERT_LT(u, 1.0f);
            ++counts[static_cast<std::size_t>(u * kBins)];
        }
        const double expected = static_cast<double>(kSamples) / kBins;
        double chiSquare = 0.0;
        for (const int count : counts) {
            chiSquare += (count - expected) * (count - expected) / expected;
        }
        EXPECT_LT(chiSquare, 103.4) << "dimension " << dimension;
    }
}

// Neighbouring pixels and successive dimensions must not be correlated: |r| < 0.02 over 65,536
// pairs is about 5 standard errors.
TEST(Random, NeighboursAndDimensionsAreUncorrelated) {
    const auto correlation = [](auto first, auto second) {
        double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
        constexpr int kPairs = 65536;
        for (std::uint32_t i = 0; i < kPairs; ++i) {
            const double x = first(i);
            const double y = second(i);
            sx += x;
            sy += y;
            sxx += x * x;
            syy += y * y;
            sxy += x * y;
        }
        const double n = kPairs;
        return (sxy - sx * sy / n) / std::sqrt((sxx - sx * sx / n) * (syy - sy * sy / n));
    };
    EXPECT_LT(
        std::abs(correlation([](std::uint32_t i) { return SampleUniform({9, 0, i, 0}, 0); },
                             [](std::uint32_t i) { return SampleUniform({9, 0, i + 1, 0}, 0); })),
        0.02);
    EXPECT_LT(std::abs(correlation([](std::uint32_t i) { return SampleUniform({9, 0, i, 0}, 0); },
                                   [](std::uint32_t i) { return SampleUniform({9, 0, i, 0}, 1); })),
              0.02);
    EXPECT_LT(std::abs(correlation([](std::uint32_t i) { return SampleUniform({9, 0, i, 0}, 0); },
                                   [](std::uint32_t i) { return SampleUniform({9, 0, i, 1}, 0); })),
              0.02);
}

// Fixed values, so a change to the hash is noticed and the shaders' copy can be checked against it.
TEST(Random, SampleBitsAreStable) {
    EXPECT_EQ(SampleBits({0, 0, 0, 0}, 0), 3995832722u);
    EXPECT_EQ(SampleBits({1, 2, 3, 4}, 5), 2812305927u);
}

TEST(Random, SplitMix64MatchesTheReferenceAndDrawsUniforms) {
    SplitMix64 reference(0);
    EXPECT_EQ(reference.NextBits(), 0xe220a8397b1dcdafull);  // Vigna's splitmix64.c from seed 0.
    SplitMix64 random(42);
    double sum = 0.0;
    constexpr int kDraws = 100000;
    for (int i = 0; i < kDraws; ++i) {
        const double u = random.NextUniform();
        ASSERT_GE(u, 0.0);
        ASSERT_LT(u, 1.0);
        sum += u;
    }
    // The mean of 10^5 uniforms is 0.5 within about 5 standard errors (0.0009 each).
    EXPECT_NEAR(sum / kDraws, 0.5, 0.005);
}

}  // namespace
}  // namespace ernie
