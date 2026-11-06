// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/catalog/tokens.h"

#include <cstdint>
#include <format>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <gtest/gtest.h>

namespace ernie {
namespace {

TEST(TokensTest, TextsAreInternedOnce) {
    Tokens tokens;
    EXPECT_EQ(tokens.Size(), 1u);
    EXPECT_EQ(tokens.Text({}), "");
    const Token sphere = tokens.Intern("Sphere");
    EXPECT_NE(sphere.id, 0u);
    EXPECT_EQ(tokens.Intern("Sphere"), sphere);
    EXPECT_EQ(tokens.Find("Sphere"), sphere);
    EXPECT_EQ(tokens.Find("Cube"), Token{});
    EXPECT_EQ(tokens.Text(sphere), "Sphere");
    EXPECT_THROW(tokens.Text({99}), std::runtime_error);
}

TEST(TokensTest, TextsStayValidAsTokensGrowAndMove) {
    Tokens tokens;
    const Token first = tokens.Intern("first");
    const std::string_view text = tokens.Text(first);
    for (uint32_t i = 0; i < 10000; ++i) {
        tokens.Intern(std::format("name{}", i));
    }
    EXPECT_EQ(text, "first");
    Tokens moved = std::move(tokens);
    EXPECT_EQ(moved.Text(first), "first");
    EXPECT_EQ(moved.Find("name9999").id, 10001u);
    EXPECT_EQ(moved.Text(first).data(), text.data());
}

}
}
