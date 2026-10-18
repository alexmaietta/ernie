// SPDX-License-Identifier: BSD-3-Clause
/// @file text_test.cpp
/// @brief Tests of the string helpers.
#include "ernie/util/text.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace ernie {
namespace {

TEST(Text, TrimRemovesSurroundingWhitespaceOnly) {
    EXPECT_EQ(Trim("  a b \t\n"), "a b");
    EXPECT_EQ(Trim("ab"), "ab");
    EXPECT_EQ(Trim("   "), "");
    EXPECT_EQ(Trim(""), "");
}

TEST(Text, CaseConversionChangesLettersOnly) {
    EXPECT_EQ(ToUpper("Lola_ldem.img 16"), "LOLA_LDEM.IMG 16");
    EXPECT_EQ(ToLower("DTM.LBL"), "dtm.lbl");
}

TEST(Text, IdentifiersAreAsciiLettersDigitsAndUnderscores) {
    static_assert(IsIdentifier("Wheel_FL2") && IsIdentifier("_x"));
    EXPECT_FALSE(IsIdentifier(""));
    EXPECT_FALSE(IsIdentifier("2nd"));
    EXPECT_FALSE(IsIdentifier("a-b"));
    EXPECT_FALSE(IsIdentifier("inputs:color")) << "a namespaced name is several identifiers";
    EXPECT_FALSE(IsIdentifier("\xc3\xa9")) << "ASCII only";
}

TEST(Text, NumberReaderReadsBlankAndCommaSeparatedNumbers) {
    NumberReader numbers("1, 2.5\n-3e2 ,4\t7 1.5", "Test file");
    EXPECT_EQ(numbers.Next(), 1.0);
    EXPECT_EQ(numbers.Values(2), (std::vector<double>{2.5, -300.0}));
    numbers.Skip(1);
    EXPECT_EQ(numbers.Count(), 7u);
    EXPECT_THROW(numbers.Count(), std::invalid_argument) << "1.5 is not a count";
    try {
        numbers.Next();
        FAIL() << "read past the end";
    } catch (const std::invalid_argument& error) {
        EXPECT_STREQ(error.what(), "Test file ends early");
    }
}

}  // namespace
}  // namespace ernie
