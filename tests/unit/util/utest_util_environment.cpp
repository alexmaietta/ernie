// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/environment.h"

#include <optional>

#include <gtest/gtest.h>

#include "ernie/util/error.h"
#include "support/support.h"

namespace ernie {
namespace {

enum class Fruit { Apple, Pear, Plum };

constexpr Choice<Fruit> kFruits[] = {
    {"apple", Fruit::Apple}, {"pear", Fruit::Pear}, {"plum", Fruit::Plum}};

constexpr Choice<Fruit> kOneFruit[] = {{"apple", Fruit::Apple}};

constexpr const char* kName = "ERNIE_TEST_FRUIT";

TEST(EnvironmentTest, ReadsVariablesAndTreatsMissingAsEmpty) {
    const test::ScopedEnvironment unset(kName, std::nullopt);
    EXPECT_EQ(Environment(kName), "");
    const test::ScopedEnvironment set(kName, "Pear");
    EXPECT_EQ(Environment(kName), "Pear");
}

TEST(EnvironmentTest, ChoicesAreEmptyWhenUnsetAndIgnoreCase) {
    const test::ScopedEnvironment unset(kName, std::nullopt);
    EXPECT_EQ(EnvironmentChoice(kName, kFruits), std::nullopt);
    const test::ScopedEnvironment pear(kName, "PEAR");
    EXPECT_EQ(EnvironmentChoice(kName, kFruits), Fruit::Pear);
}

TEST(EnvironmentTest, UnknownChoicesListTheAlternatives) {
    const test::ScopedEnvironment kiwi(kName, "Kiwi");
    try {
        EnvironmentChoice(kName, kFruits);
        FAIL();
    } catch (const Error& error) {
        EXPECT_STREQ(error.what(), "ERNIE_TEST_FRUIT is \"kiwi\"; it may be apple, pear or plum");
    }
    try {
        EnvironmentChoice(kName, kOneFruit);
        FAIL();
    } catch (const Error& error) {
        EXPECT_STREQ(error.what(), "ERNIE_TEST_FRUIT is \"kiwi\"; it may be apple");
    }
}

}
}
