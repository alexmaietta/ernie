// SPDX-License-Identifier: BSD-3-Clause
/// @file disk_cache_test.cpp
/// @brief Tests of cache folders and cache keys.
#include "ernie/util/disk_cache.h"

#include "tests/common.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

namespace ernie {
namespace {

void SetVariable(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

TEST(DiskCache, KeysAreHexAndDependOnEveryPiece) {
    CacheKey a;
    a.Add("shader");
    a.Add("v1");
    CacheKey b;
    b.Add("shader");
    b.Add("v2");
    EXPECT_EQ(a.Hex().size(), 32u);
    EXPECT_EQ(a.Hex().find_first_not_of("0123456789abcdef"), std::string::npos);
    EXPECT_NE(a.Hex(), b.Hex());
}

TEST(DiskCache, PiecesAreLengthPrefixed) {
    CacheKey split;
    split.Add("ab");
    split.Add("c");
    CacheKey joined;
    joined.Add("a");
    joined.Add("bc");
    EXPECT_NE(split.Hex(), joined.Hex());
}

TEST(DiskCache, TheVariableNamesOrDisablesTheFolder) {
    SetVariable("ERNIE_TEST_CACHE", "/tmp/ernie-test-cache");
    EXPECT_EQ(CacheDirectory("ERNIE_TEST_CACHE", "test"),
              std::filesystem::path("/tmp/ernie-test-cache"));
    SetVariable("ERNIE_TEST_CACHE", "0");
    EXPECT_FALSE(CacheDirectory("ERNIE_TEST_CACHE", "test").has_value());
}

TEST(DiskCache, CachedFloatsRoundTripAndRejectTheWrongSize) {
    const std::filesystem::path path = test::TempPath("ernie_cache") / "floats.bin";
    std::filesystem::remove(path);
    EXPECT_FALSE(ReadCachedFloats(path, 3).has_value());
    const std::vector<float> values{1.5f, -2.0f, 3.25f};
    WriteCachedFloats(path, values);
    EXPECT_EQ(ReadCachedFloats(path, 3), values);
    EXPECT_FALSE(ReadCachedFloats(path, 2).has_value());
    EXPECT_FALSE(ReadCachedFloats(path, 4).has_value());
}

}  // namespace
}  // namespace ernie
