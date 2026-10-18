// SPDX-License-Identifier: BSD-3-Clause
/// @file storage_budget_test.cpp
/// @brief The prim store's time budgets at 100,000 prims.
#include "ernie/scene/storage.h"

#include "tests/common.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

namespace ernie {
namespace {

using test::BestOf3;
using test::Budget;

constexpr std::uint32_t kPrims = 100'000;

/// 100,000 instanced robots, one per environment, with transforms.
struct Robots {
    std::unique_ptr<PrimStore> store;
    AttributeId transform = 0;
    BucketId bucket = 0;
    std::vector<PrimId> prims;

    void Reset() {
        store = std::make_unique<PrimStore>();
        transform = store->Tokens().Intern("xformOp:transform");
        bucket = store->AddBucket(
            {store->Tokens().Intern("Xform"), {}},
            std::vector<ColumnLayout>{ColumnLayout::Of<glm::dmat4>(transform, glm::dmat4(1.0))});
        prims.clear();
    }

    void Create() {
        PathTable& paths = store->Paths();
        const PathId world = paths.Child(kRootPath, "World");
        const Token robot = store->Tokens().Intern("robot");
        std::vector<PathId> robots(kPrims);
        paths.Reserve(2 * kPrims);
        for (std::uint32_t i = 0; i < kPrims; ++i) {
            robots[i] = paths.Child(paths.Child(world, "env_" + std::to_string(i)), robot);
        }
        prims = store->DefineMany(robots, bucket);
        const std::span<glm::dmat4> transforms = store->Write<glm::dmat4>(bucket, transform);
        for (std::uint32_t i = 0; i < kPrims; ++i) {
            transforms[i] = glm::dmat4(1.0);
            transforms[i][3] = glm::dvec4(i, 0.0, 0.0, 1.0);
        }
    }
};

TEST(PrimStoreBudget, Creates100000PrimsWithTransformsIn50Ms) {
    Robots robots;
    const double ms = BestOf3([&] { robots.Reset(); }, [&] { robots.Create(); });
    EXPECT_LT(ms, Budget(50.0)) << ms << " ms";
    EXPECT_EQ(robots.store->PrimCount(), kPrims);
}

TEST(PrimStoreBudget, FirstSyncOf100000PrimsIn50Ms) {
    Robots robots;
    ChangeSet changes;
    const double ms = BestOf3(
        [&] {
            robots.Reset();
            robots.Create();
        },
        [&] { changes = robots.store->ChangesSince(0); });
    EXPECT_LT(ms, Budget(50.0)) << ms << " ms";
    EXPECT_EQ(changes.rows.size(), 1u);
}

TEST(PrimStoreBudget, OneEditAmong100000IsFoundIn1Ms) {
    Robots robots;
    robots.Reset();
    robots.Create();
    std::uint64_t cursor = 0;
    ChangeSet changes;
    const double ms =
        BestOf3([&] { cursor = robots.store->Version(); },
                [&] {
                    robots.store->Set(robots.prims[54'321], robots.transform, glm::dmat4(2.0));
                    changes = robots.store->ChangesSince(cursor);
                });
    EXPECT_LT(ms, Budget(1.0)) << ms << " ms";
    ASSERT_EQ(changes.values.size(), 1u);
    EXPECT_EQ(changes.values[0].rows.size(), 1u);
}

TEST(PrimStoreBudget, EditingAll100000In10Ms) {
    Robots robots;
    robots.Reset();
    robots.Create();
    std::uint64_t cursor = 0;
    ChangeSet changes;
    const double ms =
        BestOf3([&] { cursor = robots.store->Version(); },
                [&] {
                    const std::span<glm::dmat4> transforms =
                        robots.store->Write<glm::dmat4>(robots.bucket, robots.transform);
                    for (glm::dmat4& transform : transforms) {
                        transform[3].y += 1.0;
                    }
                    changes = robots.store->ChangesSince(cursor);
                });
    EXPECT_LT(ms, Budget(10.0)) << ms << " ms";
    ASSERT_EQ(changes.values.size(), 1u);
    EXPECT_EQ(changes.values[0].rows, (std::vector<RowRange>{{0, kPrims}}));
}

TEST(PrimStoreBudget, AddingAndRemoving1000PerStepIn5Ms) {
    Robots robots;
    std::uint64_t cursor = 0;
    ChangeSet changes;
    const double ms = BestOf3(
        [&] {
            robots.Reset();
            robots.Create();
            cursor = robots.store->Version();
        },
        [&] {
            for (std::uint32_t i = 0; i < 1000; ++i) {
                robots.store->Remove(robots.prims[i * 97]);
            }
            PathTable& paths = robots.store->Paths();
            const PathId spawned = paths.Child(kRootPath, "Spawned");
            std::vector<PathId> added(1000);
            for (std::uint32_t i = 0; i < 1000; ++i) {
                added[i] = paths.Child(spawned, "p" + std::to_string(i));
            }
            robots.store->DefineMany(added, robots.bucket);
            changes = robots.store->ChangesSince(cursor);
        });
    EXPECT_LT(ms, Budget(5.0)) << ms << " ms";
    EXPECT_EQ(robots.store->PrimCount(), kPrims);
    EXPECT_EQ(changes.rows.size(), 2001u)
        << "1000 removed, 999 moved, one added; the last removed row moves nothing";
}

}  // namespace
}  // namespace ernie
