// SPDX-License-Identifier: BSD-3-Clause
/// @file storage.cpp
/// @brief The prim store at 100,000 prims: create, first sync, one edit, edit all, add and remove.
#include "ernie/scene/storage.h"
#include "benchmarks/bench.h"

#include <memory>
#include <string>
#include <vector>

namespace ernie::bench {
namespace {

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
            std::vector<ColumnLayout>{ColumnLayout::Make<glm::dmat4>(transform, glm::dmat4(1.0))});
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

    void Fresh() {
        Reset();
        Create();
    }
};

const bool kCreate = Register("scene.storage.create_100k", [] {
    Robots robots;
    return Timing{Fastest([&] { robots.Reset(); }, [&] { robots.Create(); }), 50.0};
});

const bool kFirstSync = Register("scene.storage.first_sync_100k", [] {
    Robots robots;
    return Timing{Fastest([&] { robots.Fresh(); }, [&] { robots.store->ChangesSince(0); }), 50.0};
});

const bool kOneEdit = Register("scene.storage.one_edit_of_100k", [] {
    Robots robots;
    robots.Fresh();
    std::uint64_t cursor = 0;
    const double ms =
        Fastest([&] { cursor = robots.store->Version(); },
                [&] {
                    robots.store->Set(robots.prims[54'321], robots.transform, glm::dmat4(2.0));
                    robots.store->ChangesSince(cursor);
                });
    return Timing{ms, 1.0};
});

const bool kEditAll = Register("scene.storage.edit_all_100k", [] {
    Robots robots;
    robots.Fresh();
    std::uint64_t cursor = 0;
    const double ms = Fastest([&] { cursor = robots.store->Version(); },
                              [&] {
                                  for (glm::dmat4& transform : robots.store->Write<glm::dmat4>(
                                           robots.bucket, robots.transform)) {
                                      transform[3].y += 1.0;
                                  }
                                  robots.store->ChangesSince(cursor);
                              });
    return Timing{ms, 10.0};
});

const bool kAddRemove = Register("scene.storage.add_remove_1000_of_100k", [] {
    Robots robots;
    std::uint64_t cursor = 0;
    const double ms = Fastest(
        [&] {
            robots.Fresh();
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
            robots.store->ChangesSince(cursor);
        });
    return Timing{ms, 5.0};
});

}  // namespace
}  // namespace ernie::bench
