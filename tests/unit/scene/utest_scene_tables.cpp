// SPDX-License-Identifier: BSD-3-Clause
#include <cstdint>
#include <format>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/device/device.h"
#include "ernie/scene/access/view.h"
#include "ernie/scene/catalog/catalog.h"
#include "ernie/scene/kernels/schema/columns.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/scene_impl.h"
#include "ernie/scene/store/store.h"
#include "ernie/scene/systems/hierarchy.h"
#include "ernie/scene/timeline/changes.h"
#include "ernie/scene/timeline/table_log.h"
#include "ernie/scene/timeline/timeline.h"
#include "support/devices.h"

namespace ernie {
namespace {

TEST(TableLogTest, CollectsTheWorldsWrittenSinceASequence) {
    TableLog log;
    EXPECT_EQ(log.Version(TableId::Transforms), 0u);
    const uint32_t first[] = {1, 2};
    log.Wrote(TableId::Transforms, first);
    const uint64_t seen = log.Sequence();
    const uint32_t other[] = {7};
    log.Wrote(TableId::Imageables, other);
    const uint32_t second[] = {2, 3};
    log.Wrote(TableId::Transforms, second);
    EXPECT_EQ(log.Version(TableId::Transforms), 3u);
    EXPECT_EQ(log.Version(TableId::Imageables), 2u);

    std::vector<uint32_t> all;
    log.Collect(TableId::Transforms, 0, all);
    EXPECT_EQ(all, (std::vector<uint32_t>{1, 2, 3}));
    std::vector<uint32_t> later;
    log.Collect(TableId::Transforms, seen, later);
    EXPECT_EQ(later, (std::vector<uint32_t>{2, 3}));

    log.Trim(seen);
    std::vector<uint32_t> trimmed;
    log.Collect(TableId::Transforms, 0, trimmed);
    EXPECT_EQ(trimmed, (std::vector<uint32_t>{2, 3}));
    EXPECT_EQ(log.Version(TableId::Transforms), 3u);
}

class Probe : public System {
  public:
    std::string_view Name() const override {
        return "probe";
    }

    std::span<const ColumnRef> Inputs() const override {
        return {};
    }

    std::span<const ColumnRef> Outputs() const override {
        return {};
    }

    std::span<const TableId> TableInputs() const override {
        static constexpr TableId kTables[] = {TableId::Transforms};
        return kTables;
    }

    bool Structural() const override {
        return false;
    }

    device::Operation Run(const ChangeSet& changes, Stream) override {
        ++runs;
        worlds = changes.worlds;
        wrote = changes.Wrote(TableId::Transforms);
        transforms = changes.tables->transforms.address;
        return {};
    }

    uint32_t runs = 0;
    std::vector<uint32_t> worlds;
    bool wrote = false;
    uint64_t transforms = 0;
};

struct Fixture {
    Store store;
    Catalog catalog;
    Timeline timeline;
    Hierarchy hierarchy;
    Probe probe;
    Ordinal ordinal = 0;

    explicit Fixture(device::Device& device)
        : store(device), catalog(store), timeline(catalog), hierarchy(catalog) {
        timeline.Add(hierarchy);
        timeline.Add(probe);
    }

    void Publish() {
        timeline.Publish(++ordinal).Wait();
    }

    View One(uint32_t prim, SchemaId schema) {
        return View(catalog, schema, std::vector<uint32_t>{prim});
    }
};

TEST(TableTrackingTest, ReadersRunOnlyWhenTheirTablesAreWritten) {
    test::ForEachDevice([](device::Device& device) {
        Fixture f(device);
        std::vector<uint32_t> balls;
        for (uint32_t i = 0; i < 2; ++i) {
            f.catalog.Define(SchemaId::World, std::vector<std::string>{std::format("/Env{}", i)});
            balls.push_back(f.catalog
                                .Define(SchemaId::Sphere,
                                        std::vector<std::string>{std::format("/Env{}/Ball", i)})
                                .firstPrim);
        }
        f.Publish();
        EXPECT_EQ(f.probe.runs, 1u);
        EXPECT_TRUE(f.probe.wrote);
        EXPECT_NE(f.probe.transforms, 0u);

        f.One(balls[1], SchemaId::Xformable)
            .Get<Double3>(kXformableTranslate)
            .Set(Double3{0, 1, 0});
        f.Publish();
        EXPECT_EQ(f.probe.runs, 2u);
        EXPECT_TRUE(f.probe.wrote);
        EXPECT_EQ(f.probe.worlds, (std::vector<uint32_t>{1}));

        f.One(balls[0], SchemaId::Imageable)
            .Get<Visibility>(kImageableVisibility)
            .Set(Visibility::Invisible);
        f.Publish();
        EXPECT_EQ(f.probe.runs, 2u);

        f.Publish();
        EXPECT_EQ(f.probe.runs, 2u);
    });
}

TEST(TableTrackingTest, ReadersMustBeAddedAfterTheirWriters) {
    test::ForEachDevice([](device::Device& device) {
        Store store(device);
        Catalog catalog(store);
        Timeline timeline(catalog);
        Hierarchy hierarchy(catalog);
        Probe probe;
        timeline.Add(probe);
        EXPECT_THROW(timeline.Add(hierarchy), Error);
    });
}

TEST(TableTrackingTest, GrowingPastTheTableRebuildsItAndHeldSnapshotsKeepTheOldOne) {
    test::ForEachDevice([](device::Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        scene->Define<Sphere>("/First", {.translate = {1, 2, 3}});
        scene->Publish().Wait();
        const Snapshot held = scene->Acquire(Stream::Host());
        const uint64_t before = PinnedTables::From(held).tables.transforms.address;

        std::vector<std::string> paths;
        for (uint32_t i = 0; i < 5000; ++i) {
            paths.push_back(std::format("/Many{}", i));
        }
        scene->Define<Sphere>(Span<const std::string>(paths), {.translate = {0, 0, 7}});
        scene->Publish().Wait();
        const Snapshot after = scene->Acquire(Stream::Host());
        EXPECT_NE(PinnedTables::From(after).tables.transforms.address, before);
        bool kept = false;
        for (const Ref<const RefCounted>& item : PinnedTables::From(held).held) {
            const auto* buffer = dynamic_cast<const device::Buffer*>(item.Get());
            kept = kept || (buffer != nullptr && buffer->Address() == before);
        }
        EXPECT_TRUE(kept);

        const std::vector<WorldTransform> transforms =
            scene->WorldTransforms(std::vector<std::string>{"/First", "/Many0", "/Many4999"});
        EXPECT_FLOAT_EQ(transforms[0].high.z + transforms[0].low.z, 3.0f);
        EXPECT_FLOAT_EQ(transforms[1].high.z + transforms[1].low.z, 7.0f);
        EXPECT_FLOAT_EQ(transforms[2].high.z + transforms[2].low.z, 7.0f);
        scene->Release(held);
        scene->Release(after);
    });
}

TEST(TableTrackingTest, MissingPrimsReadAsZeroTransforms) {
    test::ForEachDevice([](device::Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        scene->Define<Scope>("/Group");
        scene->Define<Sphere>("/Ball", {.translate = {0, 4, 0}});
        scene->Publish().Wait();
        const std::vector<WorldTransform> transforms =
            scene->WorldTransforms(std::vector<std::string>{"/Nothing", "/Group", "/Ball"});
        ASSERT_EQ(transforms.size(), 3u);
        EXPECT_EQ(transforms[0].high.y, 0.0f);
        EXPECT_EQ(transforms[1].high.y, 0.0f);
        EXPECT_FLOAT_EQ(transforms[2].high.y + transforms[2].low.y, 4.0f);
        EXPECT_TRUE(scene->WorldTransforms(std::vector<std::string>{}).empty());
    });
}

}
}
