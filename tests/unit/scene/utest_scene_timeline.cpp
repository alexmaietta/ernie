// SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include "ernie/device/device.h"
#include "ernie/device/queue.h"
#include "ernie/scene/builders/table_builder.h"
#include "ernie/scene/columns/column_store.h"
#include "ernie/scene/kernels/schema/columns.h"
#include "ernie/scene/publish/publisher.h"
#include "ernie/scene/structure/structure.h"
#include "ernie/scene/views/attribute.h"
#include "ernie/scene/views/view.h"
#include "support/devices.h"

namespace ernie {
namespace {

std::vector<std::string> Paths(std::string_view prefix, uint32_t count) {
    std::vector<std::string> paths;
    for (uint32_t i = 0; i < count; ++i) {
        paths.push_back(fmt::format("{}{}", prefix, i));
    }
    return paths;
}

std::vector<uint32_t> Rows(Catalog& catalog, std::span<const uint32_t> prims, SchemaId table) {
    std::vector<uint32_t> rows;
    for (const uint32_t prim : prims) {
        rows.push_back(catalog.GetPrims().Row(prim, table));
    }
    std::ranges::sort(rows);
    return rows;
}

class Counter : public System {
  public:
    explicit Counter(std::vector<ColumnRef> inputs, bool structural = true)
        : m_inputs(std::move(inputs)), m_structural(structural) {}

    std::string_view Name() const override {
        return "counter";
    }

    std::span<const ColumnRef> Inputs() const override {
        return m_inputs;
    }

    bool Structural() const override {
        return m_structural;
    }

    void Run(const ChangeSet& changes, TableLog&) override {
        ++runs;
        last = changes;
    }

    uint32_t runs = 0;
    ChangeSet last;

  private:
    std::vector<ColumnRef> m_inputs;
    bool m_structural;
};

template <class Test> void ForEachTimeline(Test test) {
    test::ForEachDevice([&](device::Device& device) {
        Store store(device);
        Catalog catalog(store);
        Timeline timeline(catalog);
        test(device, catalog, timeline);
    });
}

TEST(TimelineTest, PublishingNothingIssuesNothing) {
    ForEachTimeline([](device::Device& device, Catalog& catalog, Timeline& timeline) {
        catalog.Define(SchemaId::Sphere, Paths("/Ball", 3));
        EXPECT_TRUE(timeline.Pending());
        EXPECT_NE(timeline.Publish(1).value, 0u);
        EXPECT_FALSE(timeline.Pending());
        const uint64_t submitted = device.GetQueue().Submitted();
        const uint64_t uploaded = device.Traffic().uploadBytes;
        const device::Operation idle = timeline.Publish(2);
        EXPECT_EQ(idle.value, 0u);
        idle.Wait();
        EXPECT_EQ(device.GetQueue().Submitted(), submitted);
        EXPECT_EQ(device.Traffic().uploadBytes, uploaded);
        EXPECT_EQ(timeline.Published(), 2u);
    });
}

TEST(TimelineTest, OrdinalsOnlyMoveForward) {
    ForEachTimeline([](device::Device&, Catalog&, Timeline& timeline) {
        timeline.Publish(5);
        EXPECT_THROW(timeline.Publish(5), Error);
        EXPECT_THROW(timeline.Publish(3), Error);
        EXPECT_NO_THROW(timeline.Publish(100));
        EXPECT_THROW(timeline.Changes(101), Error);
    });
}

TEST(TimelineTest, ChangesHoldExactlyTheRowsWrittenBetweenOrdinals) {
    ForEachTimeline([](device::Device&, Catalog& catalog, Timeline& timeline) {
        const PrimBlock block = catalog.Define(SchemaId::Sphere, Paths("/Ball", 6));
        const View view(catalog, SchemaId::Sphere, block);
        timeline.Publish(10);
        const std::vector<uint32_t> picked = {block.Prim(1, 0), block.Prim(4, 0)};
        View(catalog, SchemaId::Sphere, picked)
            .Get<Double3>(kXformableTranslate)
            .Set(Double3{1, 2, 3});
        timeline.Publish(20);
        const std::vector<ColumnRef> columns = {kXformableTranslate};

        const ChangeSet recent = timeline.Changes(10, columns);
        EXPECT_EQ(recent.since, 10u);
        EXPECT_EQ(recent.until, 20u);
        ASSERT_NE(recent.Rows(kXformableTranslate), nullptr);
        EXPECT_EQ(recent.Rows(kXformableTranslate)->ToHost(),
                  Rows(catalog, picked, SchemaId::Xformable));
        EXPECT_FALSE(recent.structure);

        const ChangeSet all = timeline.Changes(0, columns);
        EXPECT_EQ(all.Rows(kXformableTranslate)->ToHost().size(), 6u);
        EXPECT_EQ(all.added.size(), 6u);
        EXPECT_EQ(all.added.front().ordinal, 10u);

        const ChangeSet none = timeline.Changes(20, columns);
        EXPECT_TRUE(none.Rows(kXformableTranslate)->ToHost().empty());
        EXPECT_TRUE(none.worlds.empty());
    });
}

TEST(TimelineTest, ChangesNameOnlyTheWorldsThatChanged) {
    ForEachTimeline([](device::Device&, Catalog& catalog, Timeline& timeline) {
        for (uint32_t world = 0; world < 3; ++world) {
            catalog.Define(SchemaId::World, std::vector<std::string>{fmt::format("/Env{}", world)});
        }
        const PrimBlock balls = catalog.Define(SchemaId::Sphere, Paths("/Env1/Ball", 2));
        timeline.Publish(1);
        View(catalog, SchemaId::Sphere, balls).Get<float>(kSphereRadius).Set(2.0f);
        timeline.Publish(2);
        const ChangeSet changes = timeline.Changes(1);
        EXPECT_EQ(changes.worlds, std::vector<uint32_t>{1});
        EXPECT_TRUE(changes.World(1));
        EXPECT_FALSE(changes.World(0));
        catalog.Define(SchemaId::Sphere, std::vector<std::string>{"/Sun"});
        timeline.Publish(3);
        const ChangeSet global = timeline.Changes(2);
        EXPECT_EQ(global.worlds, std::vector<uint32_t>{kGlobalWorld});
        EXPECT_TRUE(global.World(0));
    });
}

TEST(TimelineTest, StructuralChangesCarryTheirOrdinal) {
    ForEachTimeline([](device::Device&, Catalog& catalog, Timeline& timeline) {
        const uint32_t a =
            catalog.Define(SchemaId::Sphere, std::vector<std::string>{"/A"}).firstPrim;
        timeline.Publish(7);
        catalog.Remove(a);
        catalog.Define(SchemaId::Sphere, std::vector<std::string>{"/B"});
        timeline.Publish(9);
        const ChangeSet changes = timeline.Changes(7);
        ASSERT_EQ(changes.removed.size(), 1u);
        EXPECT_EQ(changes.removed.front().prim, a);
        EXPECT_EQ(changes.removed.front().ordinal, 9u);
        ASSERT_EQ(changes.added.size(), 1u);
        EXPECT_TRUE(changes.structure);
    });
}

TEST(TimelineTest, SystemsRunOnlyWhenTheirInputsChange) {
    ForEachTimeline([](device::Device&, Catalog& catalog, Timeline& timeline) {
        Counter transforms({kXformableTranslate});
        Counter radii({kSphereRadius}, false);
        timeline.Add(transforms);
        timeline.Add(radii);
        const PrimBlock block = catalog.Define(SchemaId::Sphere, Paths("/Ball", 4));
        const View view(catalog, SchemaId::Sphere, block);
        timeline.Publish(1);
        EXPECT_EQ(transforms.runs, 1u);
        EXPECT_EQ(radii.runs, 1u);

        timeline.Publish(2);
        EXPECT_EQ(transforms.runs, 1u);

        view.Get<float>(kSphereRadius).Set(3.0f);
        timeline.Publish(3);
        EXPECT_EQ(transforms.runs, 1u);
        EXPECT_EQ(radii.runs, 2u);

        const std::vector<uint32_t> one = {block.Prim(2, 0)};
        View(catalog, SchemaId::Sphere, one).Get<Double3>(kXformableTranslate).Set(Double3{});
        timeline.Publish(4);
        EXPECT_EQ(transforms.runs, 2u);
        EXPECT_EQ(transforms.last.since, 1u);
        EXPECT_EQ(transforms.last.until, 4u);
        EXPECT_EQ(transforms.last.Rows(kXformableTranslate)->ToHost(),
                  Rows(catalog, one, SchemaId::Xformable));

        catalog.Define(SchemaId::Cube, std::vector<std::string>{"/More"});
        timeline.Publish(5);
        EXPECT_EQ(transforms.runs, 3u);
        EXPECT_EQ(radii.runs, 2u);
        EXPECT_EQ(transforms.last.added.size(), 1u);
    });
}

TEST(TimelineTest, SnapshotsAreCountedAndOrdered) {
    ForEachTimeline([](device::Device&, Catalog& catalog, Timeline& timeline) {
        EXPECT_THROW(timeline.Acquire(), Error);
        catalog.Define(SchemaId::Sphere, std::vector<std::string>{"/A"});
        timeline.Publish(1);
        const Snapshot snapshot = timeline.Acquire(Stream::Host());
        EXPECT_EQ(snapshot.ordinal, 1u);
        EXPECT_NE(snapshot.tables, 0u);
        EXPECT_TRUE(snapshot.ready.Ready());
        EXPECT_EQ(timeline.Outstanding(), 1u);
        timeline.Release(snapshot);
        EXPECT_EQ(timeline.Outstanding(), 0u);
        EXPECT_THROW(timeline.Release(snapshot), Error);
    });
}

}
}
