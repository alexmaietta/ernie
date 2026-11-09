// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/text.h"

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace ernie {
namespace {

using Parts = std::vector<std::string_view>;
using Halves = std::pair<std::string_view, std::string_view>;

TEST(TextTest, LowercasesOnlyAsciiLetters) {
    EXPECT_EQ(Lowercase("NVIDIA GeForce RTX 5070"), "nvidia geforce rtx 5070");
    EXPECT_EQ(Lowercase("caf\xc3\x89"), "caf\xc3\x89");
    EXPECT_EQ(Lowercase(""), "");
}

TEST(TextTest, JoinsWithASeparator) {
    EXPECT_EQ(Join({}), "");
    EXPECT_EQ(Join({"a"}), "a");
    EXPECT_EQ(Join({"a", "b", "c"}), "a, b, c");
    EXPECT_EQ(Join({"a", "b"}, " | "), "a | b");
}

TEST(TextTest, SplitsAtEverySeparator) {
    EXPECT_EQ(Split("", '/'), (Parts{""}));
    EXPECT_EQ(Split("abc", '/'), (Parts{"abc"}));
    EXPECT_EQ(Split("a/b/c", '/'), (Parts{"a", "b", "c"}));
    EXPECT_EQ(Split("/a//b/", '/'), (Parts{"", "a", "", "b", ""}));
}

TEST(TextTest, SplitsAtTheLastSeparator) {
    EXPECT_EQ(SplitLast("abc", '/'), (Halves{"", "abc"}));
    EXPECT_EQ(SplitLast("/a/b/c", '/'), (Halves{"/a/b", "c"}));
    EXPECT_EQ(SplitLast("/a", '/'), (Halves{"", "a"}));
    EXPECT_EQ(SplitLast("a/", '/'), (Halves{"a", ""}));
}

TEST(TextTest, IdentifiersStartWithALetterOrUnderscore) {
    EXPECT_TRUE(IsIdentifier("Robot"));
    EXPECT_TRUE(IsIdentifier("robot"));
    EXPECT_TRUE(IsIdentifier("_private2"));
    EXPECT_TRUE(IsIdentifier("Z9_z"));
    EXPECT_FALSE(IsIdentifier(""));
    EXPECT_FALSE(IsIdentifier("2fast"));
    EXPECT_FALSE(IsIdentifier("b-c"));
    EXPECT_FALSE(IsIdentifier("a b"));
    EXPECT_FALSE(IsIdentifier("caf\xc3\xa9"));
    EXPECT_FALSE(IsIdentifier("{"));
    EXPECT_FALSE(IsIdentifier("a{"));
    EXPECT_FALSE(IsIdentifier("a:"));
}

TEST(TextTest, GlobsMatchLiteralsQuestionMarksAndStars) {
    EXPECT_TRUE(Glob("", ""));
    EXPECT_FALSE(Glob("", "a"));
    EXPECT_FALSE(Glob("a", ""));
    EXPECT_TRUE(Glob("base_link", "base_link"));
    EXPECT_FALSE(Glob("base_link", "base_lint"));
    EXPECT_TRUE(Glob("?_link", "a_link"));
    EXPECT_FALSE(Glob("?_link", "ab_link"));
    EXPECT_TRUE(Glob("*", ""));
    EXPECT_TRUE(Glob("**", "abc"));
    EXPECT_TRUE(Glob("*_link", "base_link"));
    EXPECT_TRUE(Glob("a*b*c", "aXXbYc"));
    EXPECT_TRUE(Glob("a*c", "abcbc"));
    EXPECT_FALSE(Glob("a*b", "ac"));
    EXPECT_FALSE(Glob("a*", "ba"));
}

TEST(TextTest, GlobsStayFastOnPathologicalPatterns) {
    const std::string text(200, 'a');
    EXPECT_FALSE(Glob("a*a*a*a*a*a*a*a*a*a*b", text));
}

TEST(TextTest, DigestsAreStableHexFnv1a) {
    EXPECT_EQ(Digest(""), "cbf29ce484222325");
    EXPECT_EQ(Digest("a"), "af63dc4c8601ec8c");
    EXPECT_EQ(Digest(std::string_view("\xff", 1)), "af64724c8602eb6e");
    EXPECT_NE(Digest("kernel a"), Digest("kernel b"));
}

TEST(TextTest, ModulePathsAndErrorOriginsMapGeneratedFilesToTheirSource) {
    EXPECT_EQ(ModulePath("ernie.material.nodes.n12"), "ernie/material/nodes/n12.slang");
    const std::pair<std::string, std::string> origins[] = {
        {"ernie/material/nodes/n12.slang", "/home/user/wet.slang"}};
    EXPECT_EQ(MapOrigins("./ernie/material/nodes/n12.slang(3): error, "
                         "ernie/material/nodes/n12.slang(4): note",
                         origins),
              "/home/user/wet.slang(3): error, /home/user/wet.slang(4): note");
}

}
}
