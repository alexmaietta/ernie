// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/catalog/paths.h"

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <gtest/gtest.h>

namespace ernie {
namespace {

std::vector<uint32_t> Children(const Paths& paths, uint32_t prim) {
    return paths.Children(prim);
}

TEST(PathsTest, PrimsAreFoundAtTheirPaths) {
    Tokens tokens;
    Paths paths(tokens);
    paths.Insert("/World", 0);
    paths.Insert("/World/Robot", 1);
    paths.Insert("/World/Robot/Base", 2);
    paths.Insert("/Sun", 3);
    EXPECT_EQ(paths.Size(), 4u);
    EXPECT_EQ(paths.Find("/World"), 0u);
    EXPECT_EQ(paths.Find("/World/Robot/Base"), 2u);
    EXPECT_EQ(paths.Find("/Sun"), 3u);
    EXPECT_EQ(paths.Find("/World/Robo"), Paths::kNone);
    EXPECT_EQ(paths.Find("/World/Robot/Base/Arm"), Paths::kNone);
    EXPECT_EQ(paths.Find("/Unknown"), Paths::kNone);
    EXPECT_EQ(paths.Find("World"), Paths::kNone);
    EXPECT_EQ(paths.Find("/"), Paths::kNone);
    EXPECT_EQ(paths.Find("/World/"), Paths::kNone);
    EXPECT_EQ(paths.Find(""), Paths::kNone);
    EXPECT_EQ(paths.FindParent("/World/Robot/Arm"), 1u);
    EXPECT_EQ(paths.FindParent("/Moon"), Paths::kRoot);
}

TEST(PathsTest, PrimsKnowTheirPathParentNameAndDepth) {
    Tokens tokens;
    Paths paths(tokens);
    paths.Insert("/envs", 7);
    paths.Insert("/envs/env_0", 3);
    paths.Insert("/envs/env_0/Robot", 12);
    EXPECT_EQ(paths.Path(12), "/envs/env_0/Robot");
    EXPECT_EQ(paths.Path(7), "/envs");
    EXPECT_EQ(paths.Name(3), "env_0");
    EXPECT_EQ(paths.Parent(12), 3u);
    EXPECT_EQ(paths.Parent(7), Paths::kRoot);
    EXPECT_EQ(paths.Depth(7), 0u);
    EXPECT_EQ(paths.Depth(12), 2u);
    EXPECT_TRUE(paths.Contains(3));
    EXPECT_FALSE(paths.Contains(4));
    EXPECT_FALSE(paths.Contains(100));
    EXPECT_THROW(paths.Path(4), std::runtime_error);
}

TEST(PathsTest, ChildrenKeepTheirInsertionOrder) {
    Tokens tokens;
    Paths paths(tokens);
    paths.Insert("/B", 0);
    paths.Insert("/A", 1);
    paths.Insert("/B/Z", 2);
    paths.Insert("/B/Y", 3);
    paths.Insert("/B/X", 4);
    EXPECT_EQ(Children(paths, Paths::kRoot), (std::vector<uint32_t>{0, 1}));
    EXPECT_EQ(Children(paths, 0), (std::vector<uint32_t>{2, 3, 4}));
    EXPECT_TRUE(Children(paths, 1).empty());
}

TEST(PathsTest, SubtreesAreListedParentsFirst) {
    Tokens tokens;
    Paths paths(tokens);
    paths.Insert("/A", 0);
    paths.Insert("/A/B", 1);
    paths.Insert("/A/B/C", 2);
    paths.Insert("/A/D", 3);
    paths.Insert("/E", 4);
    EXPECT_EQ(paths.Subtree(0), (std::vector<uint32_t>{0, 1, 2, 3}));
    EXPECT_EQ(paths.Subtree(1), (std::vector<uint32_t>{1, 2}));
    EXPECT_EQ(paths.Subtree(Paths::kRoot), (std::vector<uint32_t>{0, 1, 2, 3, 4}));
    EXPECT_THROW(paths.Subtree(9), std::runtime_error);
}

TEST(PathsTest, LeavesAreErasedAndTheirIndicesReused) {
    Tokens tokens;
    Paths paths(tokens);
    paths.Insert("/A", 0);
    paths.Insert("/A/B", 1);
    paths.Insert("/A/C", 2);
    EXPECT_THROW(paths.Erase(0), std::runtime_error);
    paths.Erase(1);
    EXPECT_EQ(paths.Size(), 2u);
    EXPECT_FALSE(paths.Contains(1));
    EXPECT_EQ(paths.Find("/A/B"), Paths::kNone);
    EXPECT_EQ(Children(paths, 0), (std::vector<uint32_t>{2}));

    paths.Insert("/A/B", 1);
    EXPECT_EQ(paths.Find("/A/B"), 1u);
    paths.Insert("/A/Other", 5);
    for (const uint32_t prim : {1u, 2u, 5u, 0u}) {
        paths.Erase(prim);
    }
    EXPECT_EQ(paths.Size(), 0u);
    EXPECT_TRUE(Children(paths, Paths::kRoot).empty());
    paths.Insert("/A", 2);
    EXPECT_EQ(paths.Find("/A"), 2u);
}

TEST(PathsTest, BadPathsAndIndicesAreRejected) {
    Tokens tokens;
    Paths paths(tokens);
    paths.Insert("/A", 0);
    EXPECT_THROW(paths.Insert("A", 1), std::runtime_error);
    EXPECT_THROW(paths.Insert("/", 1), std::runtime_error);
    EXPECT_THROW(paths.Insert("/A/", 1), std::runtime_error);
    EXPECT_THROW(paths.Insert("/A//B", 1), std::runtime_error);
    EXPECT_THROW(paths.Insert("/A/1B", 1), std::runtime_error);
    EXPECT_THROW(paths.Insert("/A/B-C", 1), std::runtime_error);
    EXPECT_THROW(paths.Insert("/Missing/B", 1), std::runtime_error);
    EXPECT_THROW(paths.Insert("/A", 1), std::runtime_error);
    EXPECT_THROW(paths.Insert("/B", 0), std::runtime_error);
    EXPECT_THROW(paths.Insert("/B", Paths::kNone), std::runtime_error);
    EXPECT_EQ(paths.Size(), 1u);
}

TEST(PathsTest, ManyPrimsAreFoundByPath) {
    Tokens tokens;
    Paths paths(tokens);
    paths.Insert("/envs", 0);
    uint32_t prim = 1;
    for (uint32_t env = 0; env < 1000; ++env) {
        const std::string root = fmt::format("/envs/env_{}", env);
        paths.Insert(root, prim++);
        for (uint32_t link = 0; link < 9; ++link) {
            paths.Insert(fmt::format("{}/link_{}", root, link), prim++);
        }
    }
    EXPECT_EQ(paths.Size(), 10001u);
    EXPECT_EQ(paths.Find("/envs/env_731/link_4"), 1u + 731 * 10 + 1 + 4);
    EXPECT_EQ(paths.Path(1u + 731 * 10 + 1 + 4), "/envs/env_731/link_4");
    EXPECT_EQ(paths.Children(0).size(), 1000u);
}

TEST(PathsTest, ClonesAreFoundWithoutANodePerPrim) {
    Tokens tokens;
    Paths paths(tokens);
    paths.Insert("/envs", 0);
    paths.Insert("/envs/env_0", 1);
    paths.Insert("/envs/env_0/Robot", 2);
    paths.Insert("/envs/env_0/Robot/Arm", 3);
    paths.Insert("/envs/env_0/Lamp", 4);
    std::vector<std::string> targets;
    for (uint32_t i = 1; i <= 1000; ++i) {
        targets.push_back(fmt::format("/envs/env_{}", i));
    }
    paths.InsertClones(paths.Shape(1), targets, 100);
    EXPECT_EQ(paths.Size(), 5u + 4000u);
    EXPECT_EQ(paths.Explicit(), 5u + 1000u);

    const uint32_t arm = 100 + 731 * 4 + 2;
    EXPECT_EQ(paths.Find("/envs/env_732/Robot/Arm"), arm);
    EXPECT_EQ(paths.Path(arm), "/envs/env_732/Robot/Arm");
    EXPECT_EQ(paths.Name(arm), "Arm");
    EXPECT_EQ(paths.Parent(arm), arm - 1);
    EXPECT_EQ(paths.Depth(arm), 3u);
    EXPECT_EQ(Children(paths, 100 + 731 * 4), (std::vector<uint32_t>{arm - 1, arm + 1}));
    EXPECT_EQ(paths.Subtree(100 + 731 * 4),
              (std::vector<uint32_t>{arm - 2, arm - 1, arm, arm + 1}));
    EXPECT_EQ(paths.Find("/envs/env_732/Robot/Leg"), Paths::kNone);
    EXPECT_TRUE(paths.Contains(arm));
    EXPECT_EQ(Children(paths, 0).size(), 1001u);

    paths.Insert("/envs/env_732/Robot/Leg", 5000);
    EXPECT_EQ(paths.Explicit(), 5u + 1000u + 3u + 1u);
    EXPECT_EQ(paths.Find("/envs/env_732/Robot/Arm"), arm);
    EXPECT_EQ(Children(paths, arm - 1), (std::vector<uint32_t>{arm, 5000}));
    EXPECT_THROW(paths.Insert("/envs/env_733/Robot/Arm", 5001), std::runtime_error);

    paths.Erase(102);
    EXPECT_EQ(paths.Find("/envs/env_1/Robot/Arm"), Paths::kNone);
    EXPECT_EQ(paths.Find("/envs/env_1/Lamp"), 103u);
    EXPECT_EQ(paths.Size(), 5u + 4000u);
    paths.Insert("/envs/env_1/Robot/Arm2", 102);
    EXPECT_EQ(paths.Path(102), "/envs/env_1/Robot/Arm2");
}
}
}
