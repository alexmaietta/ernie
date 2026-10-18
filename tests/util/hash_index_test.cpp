// SPDX-License-Identifier: BSD-3-Clause
/// @file hash_index_test.cpp
/// @brief Tests of the open-addressed id index.
#include "ernie/util/hash_index.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace ernie {
namespace {

TEST(HashIndex, FindsIdsThroughTheOwnersKeysAcrossRehashes) {
    std::vector<std::string> keys;
    HashIndex index;
    const auto hash = [](const std::string& key) { return std::hash<std::string>()(key); };
    for (std::uint32_t i = 0; i < 1000; ++i) {
        const std::string key = "k" + std::to_string(i);
        const auto next = static_cast<std::uint32_t>(keys.size());
        ASSERT_EQ(index.Insert(hash(key), next, [&](std::uint32_t id) { return keys[id] == key; }),
                  next);
        keys.push_back(key);
    }
    EXPECT_EQ(index.Size(), 1000u);
    for (std::uint32_t i = 0; i < 1000; ++i) {
        const std::string& key = keys[i];
        EXPECT_EQ(index.Find(hash(key), [&](std::uint32_t id) { return keys[id] == key; }), i);
    }
    const std::string again = "k7";
    EXPECT_EQ(index.Insert(hash(again), 1000, [&](std::uint32_t id) { return keys[id] == again; }),
              7u);
    EXPECT_EQ(index.Size(), 1000u);
    EXPECT_FALSE(
        index.Find(hash("absent"), [&](std::uint32_t id) { return keys[id] == "absent"; }));
}

TEST(HashIndex, CollidingHashesAreToldApartByTheOwner) {
    HashIndex index;
    index.Reserve(4);
    EXPECT_EQ(index.Insert(42, 0, [](std::uint32_t) { return false; }), 0u);
    EXPECT_EQ(index.Insert(42, 1, [](std::uint32_t id) { return id == 1; }), 1u);
    EXPECT_EQ(index.Find(42, [](std::uint32_t id) { return id == 1; }), 1u);
    EXPECT_EQ(index.Find(42, [](std::uint32_t id) { return id == 0; }), 0u);
    EXPECT_FALSE(HashIndex().Find(42, [](std::uint32_t) { return true; }));
}

}  // namespace
}  // namespace ernie
