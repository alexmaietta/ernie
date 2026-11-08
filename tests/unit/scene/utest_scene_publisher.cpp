// SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <cstdint>
#include <format>
#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/device/device.h"
#include "ernie/device/queue.h"
#include "ernie/scene/columns/column_store.h"
#include "ernie/scene/publish/publisher.h"
#include "ernie/scene/schema/columns.h"
#include "ernie/scene/structure/structure.h"
#include "ernie/scene/tables/table.h"
#include "ernie/scene/views/attribute.h"
#include "ernie/scene/views/view.h"
#include "support/devices.h"

namespace ernie {
namespace {

std::vector<std::string> Paths(std::string_view prefix, uint32_t count) {
    std::vector<std::string> paths;
    for (uint32_t i = 0; i < count; ++i) {
        paths.push_back(std::format("{}{}", prefix, i));
    }
    return paths;
}

std::vector<uint32_t> Rows(Structure& structure, std::span<const uint32_t> prims, SchemaId table) {
    std::vector<uint32_t> rows;
    for (const uint32_t prim : prims) {
        rows.push_back(structure.GetPrims().Row(prim, table));
    }
    std::ranges::sort(rows);
    return rows;
}

class Counter : public Table {
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

    void Run(const ChangeSet& changes, TableVersions&) override {
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
        ColumnStore store(device);
        Structure structure(store);
        Publisher publisher(structure);
        test(device, structure, publisher);
    });
}

TEST(PublisherTest, PublishingNothingIssuesNothing) {
    ForEachTimeline([](device::Device& device, Structure& structure, Publisher& publisher) {
        structure.Define(SchemaId::Sphere, Paths("/Ball", 3));
        EXPECT_TRUE(publisher.Pending());
        EXPECT_NE(publisher.Publish(1).value, 0u);
        EXPECT_FALSE(publisher.Pending());
        const uint64_t submitted = device.GetQueue().Submitted();
        const uint64_t uploaded = device.Traffic().uploadBytes;
        const device::Operation idle = publisher.Publish(2);
        EXPECT_EQ(idle.value, 0u);
        idle.Wait();
        EXPECT_EQ(device.GetQueue().Submitted(), submitted);
        EXPECT_EQ(device.Traffic().uploadBytes, uploaded);
        EXPECT_EQ(publisher.Published(), 2u);
    });
}

TEST(PublisherTest, OrdinalsOnlyMoveForward) {
    ForEachTimeline([](device::Device&, Structure&, Publisher& publisher) {
        publisher.Publish(5);
        EXPECT_THROW(publisher.Publish(5), Error);
        EXPECT_THROW(publisher.Publish(3), Error);
        EXPECT_NO_THROW(publisher.Publish(100));
        EXPECT_THROW(publisher.Changes(101), Error);
    });
}

TEST(PublisherTest, ChangesHoldExactlyTheRowsWrittenBetweenOrdinals) {
    ForEachTimeline([](device::Device&, Structure& structure, Publisher& publisher) {
        const PrimBlock block = structure.Define(SchemaId::Sphere, Paths("/Ball", 6));
        const View view(structure, SchemaId::Sphere, block);
        publisher.Publish(10);
        const std::vector<uint32_t> picked = {block.Prim(1, 0), block.Prim(4, 0)};
        View(structure, SchemaId::Sphere, picked)
            .Get<Double3>(kXformableTranslate)
            .Set(Double3{1, 2, 3});
        publisher.Publish(20);
        const std::vector<ColumnRef> columns = {kXformableTranslate};

        const ChangeSet recent = publisher.Changes(10, columns);
        EXPECT_EQ(recent.since, 10u);
        EXPECT_EQ(recent.until, 20u);
        ASSERT_NE(recent.Rows(kXformableTranslate), nullptr);
        EXPECT_EQ(recent.Rows(kXformableTranslate)->ToHost(),
                  Rows(structure, picked, SchemaId::Xformable));
        EXPECT_FALSE(recent.structure);

        const ChangeSet all = publisher.Changes(0, columns);
        EXPECT_EQ(all.Rows(kXformableTranslate)->ToHost().size(), 6u);
        EXPECT_EQ(all.added.size(), 6u);
        EXPECT_EQ(all.added.front().ordinal, 10u);

        const ChangeSet none = publisher.Changes(20, columns);
        EXPECT_TRUE(none.Rows(kXformableTranslate)->ToHost().empty());
        EXPECT_TRUE(none.worlds.empty());
    });
}

TEST(PublisherTest, ChangesNameOnlyTheWorldsThatChanged) {
    ForEachTimeline([](device::Device&, Structure& structure, Publisher& publisher) {
        for (uint32_t world = 0; world < 3; ++world) {
            structure.Define(SchemaId::World,
                             std::vector<std::string>{std::format("/Env{}", world)});
        }
        const PrimBlock balls = structure.Define(SchemaId::Sphere, Paths("/Env1/Ball", 2));
        publisher.Publish(1);
        View(structure, SchemaId::Sphere, balls).Get<float>(kSphereRadius).Set(2.0f);
        publisher.Publish(2);
        const ChangeSet changes = publisher.Changes(1);
        EXPECT_EQ(changes.worlds, std::vector<uint32_t>{1});
        EXPECT_TRUE(changes.World(1));
        EXPECT_FALSE(changes.World(0));
        structure.Define(SchemaId::Sphere, std::vector<std::string>{"/Sun"});
        publisher.Publish(3);
        const ChangeSet global = publisher.Changes(2);
        EXPECT_EQ(global.worlds, std::vector<uint32_t>{kGlobalWorld});
        EXPECT_TRUE(global.World(0));
    });
}

TEST(PublisherTest, StructuralChangesCarryTheirOrdinal) {
    ForEachTimeline([](device::Device&, Structure& structure, Publisher& publisher) {
        const uint32_t a =
            structure.Define(SchemaId::Sphere, std::vector<std::string>{"/A"}).firstPrim;
        publisher.Publish(7);
        structure.Remove(a);
        structure.Define(SchemaId::Sphere, std::vector<std::string>{"/B"});
        publisher.Publish(9);
        const ChangeSet changes = publisher.Changes(7);
        ASSERT_EQ(changes.removed.size(), 1u);
        EXPECT_EQ(changes.removed.front().prim, a);
        EXPECT_EQ(changes.removed.front().ordinal, 9u);
        ASSERT_EQ(changes.added.size(), 1u);
        EXPECT_TRUE(changes.structure);
    });
}

TEST(PublisherTest, SystemsRunOnlyWhenTheirInputsChange) {
    ForEachTimeline([](device::Device&, Structure& structure, Publisher& publisher) {
        Counter transforms({kXformableTranslate});
        Counter radii({kSphereRadius}, false);
        publisher.Add(transforms);
        publisher.Add(radii);
        const PrimBlock block = structure.Define(SchemaId::Sphere, Paths("/Ball", 4));
        const View view(structure, SchemaId::Sphere, block);
        publisher.Publish(1);
        EXPECT_EQ(transforms.runs, 1u);
        EXPECT_EQ(radii.runs, 1u);

        publisher.Publish(2);
        EXPECT_EQ(transforms.runs, 1u);

        view.Get<float>(kSphereRadius).Set(3.0f);
        publisher.Publish(3);
        EXPECT_EQ(transforms.runs, 1u);
        EXPECT_EQ(radii.runs, 2u);

        const std::vector<uint32_t> one = {block.Prim(2, 0)};
        View(structure, SchemaId::Sphere, one).Get<Double3>(kXformableTranslate).Set(Double3{});
        publisher.Publish(4);
        EXPECT_EQ(transforms.runs, 2u);
        EXPECT_EQ(transforms.last.since, 1u);
        EXPECT_EQ(transforms.last.until, 4u);
        EXPECT_EQ(transforms.last.Rows(kXformableTranslate)->ToHost(),
                  Rows(structure, one, SchemaId::Xformable));

        structure.Define(SchemaId::Cube, std::vector<std::string>{"/More"});
        publisher.Publish(5);
        EXPECT_EQ(transforms.runs, 3u);
        EXPECT_EQ(radii.runs, 2u);
        EXPECT_EQ(transforms.last.added.size(), 1u);
    });
}

TEST(PublisherTest, SnapshotsAreCountedAndOrdered) {
    ForEachTimeline([](device::Device&, Structure& structure, Publisher& publisher) {
        EXPECT_THROW(publisher.Acquire(), Error);
        structure.Define(SchemaId::Sphere, std::vector<std::string>{"/A"});
        publisher.Publish(1);
        const Snapshot snapshot = publisher.Acquire(Stream::Host());
        EXPECT_EQ(snapshot.ordinal, 1u);
        EXPECT_NE(snapshot.tables, 0u);
        EXPECT_TRUE(snapshot.ready.Ready());
        EXPECT_EQ(publisher.Outstanding(), 1u);
        publisher.Release(snapshot);
        EXPECT_EQ(publisher.Outstanding(), 0u);
        EXPECT_THROW(publisher.Release(snapshot), Error);
    });
}

}
}
