// SPDX-License-Identifier: BSD-3-Clause
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/device/device.h"
#include "ernie/device/queue.h"
#include "ernie/scene/builders/semantics.h"
#include "ernie/scene/columns/column_store.h"
#include "ernie/scene/kernels/schema/columns.h"
#include "ernie/scene/publish/publisher.h"
#include "ernie/scene/structure/packing.h"
#include "ernie/scene/structure/structure.h"
#include "ernie/scene/views/attribute.h"
#include "ernie/scene/views/view.h"
#include "support/devices.h"

namespace ernie {
namespace {

struct Scene {
    device::Device& device;
    ColumnStore& store;
    Structure& structure;
    Publisher& publisher;
    SemanticBuilder& semantics;
    Token taxonomy;
    Ordinal ordinal = 0;

    void Publish() {
        publisher.Publish(++ordinal).Wait();
    }

    uint32_t Define(SchemaId schema, const std::string& path) {
        return structure.Define(schema, std::vector<std::string>{path}).firstPrim;
    }

    uint32_t Labeled(const std::string& path, const std::string& label) {
        return structure
            .Define<Xform>(std::vector<std::string>{path},
                           {.semanticsLabels = {{"class", {label}}}})
            .firstPrim;
    }

    void Relabel(uint32_t prim, const std::string& label) {
        const uint32_t id = structure.GetTokens().Intern(label).id;
        ArrayHeap& arrays = store.Arrays(ColumnType::TokenArray);
        const ArrayRef array = arrays.Allocate(1);
        arrays.Write(array, std::as_bytes(std::span(&id, 1)));
        store.SetArray(kSemanticsLabelsAPISemanticsLabels,
                       structure.GetPrims().InstanceRow(prim, SchemaId::SemanticsLabelsAPI,
                                                        structure.GetTokens().Intern("class").id),
                       array);
    }
};

template <class Test> void ForEachScene(Test test) {
    test::ForEachDevice([&](device::Device& device) {
        ColumnStore store(device);
        Structure structure(store);
        Publisher publisher(structure);
        SemanticBuilder semantics(structure);
        publisher.Add(semantics);
        const Token taxonomy = structure.GetTokens().Intern("class");
        semantics.Require(std::vector<Token>{taxonomy});
        Scene scene{device, store, structure, publisher, semantics, taxonomy};
        test(scene);
    });
}

std::vector<uint32_t> Ids(const StableId& id) {
    return {id.words[0], id.words[1], id.words[2], id.words[3]};
}

TEST(SemanticsPackingTest, IdMapsPackRecordsThenTextThenCount) {
    const std::vector<std::pair<StableId, std::string>> entries = {
        {StableId{{1, 2, 3, 4}}, "/A"}, {StableId{{5, 0, 0, 0}}, "class: rock;"}};
    const std::vector<std::byte> bytes = PackIdMap(entries);
    ASSERT_EQ(bytes.size(), 2 * 24 + 2 + 12 + 4);
    uint32_t words[6];
    std::memcpy(words, bytes.data() + 24, sizeof(words));
    EXPECT_EQ(words[0], 5u);
    EXPECT_EQ(words[4], 12u);
    EXPECT_EQ(words[5], 50u);
    uint32_t count;
    std::memcpy(&count, bytes.data() + bytes.size() - 4, 4);
    EXPECT_EQ(count, 2u);
    EXPECT_EQ(Ids(HashPath("/A")), Ids(HashPath("/A")));
    EXPECT_NE(Ids(HashPath("/A")), Ids(HashPath("/B")));
}

TEST(SemanticsTest, LabelsResolveDownTheTreeByTaxonomy) {
    ForEachScene([](Scene& s) {
        const uint32_t robot = s.Labeled("/Robot", "robot");
        const uint32_t arm = s.Define(SchemaId::Sphere, "/Robot/Arm");
        const uint32_t hand = s.Labeled("/Robot/Hand", "hand");
        const uint32_t free = s.Define(SchemaId::Sphere, "/Free");
        s.Publish();
        EXPECT_GE(s.semantics.SemanticId(s.taxonomy, robot), 2u);
        EXPECT_EQ(s.semantics.SemanticId(s.taxonomy, arm),
                  s.semantics.SemanticId(s.taxonomy, robot));
        EXPECT_EQ(s.semantics.LabeledPrim(s.taxonomy, arm), robot);
        EXPECT_NE(s.semantics.SemanticId(s.taxonomy, hand),
                  s.semantics.SemanticId(s.taxonomy, robot));
        EXPECT_EQ(s.semantics.LabeledPrim(s.taxonomy, hand), hand);
        EXPECT_EQ(s.semantics.SemanticId(s.taxonomy, free), 1u);
        EXPECT_EQ(s.semantics.LabeledPrim(s.taxonomy, free), kNoPrim);
        EXPECT_GE(s.semantics.Data().stride, s.structure.GetPrims().Size());
        EXPECT_EQ(s.semantics.Data().taxonomyCount, 1u);
    });
}

TEST(SemanticsTest, ARequestedTaxonomyResolvesOnTheNextPublish) {
    ForEachScene([](Scene& s) {
        const uint32_t robot = s.Labeled("/Robot", "robot");
        s.Publish();
        const Token material = s.structure.GetTokens().Intern("material");
        s.semantics.Require(std::vector<Token>{s.taxonomy, material});
        EXPECT_TRUE(s.publisher.Pending());
        s.Publish();
        EXPECT_EQ(s.semantics.Data().taxonomyCount, 2u);
        EXPECT_EQ(s.semantics.SemanticId(material, robot), 1u);
        EXPECT_GE(s.semantics.SemanticId(s.taxonomy, robot), 2u);
    });
}

TEST(SemanticsTest, AddingAPrimResolvesOnlyIt) {
    ForEachScene([](Scene& s) {
        const uint32_t robot = s.Labeled("/Robot", "robot");
        s.Define(SchemaId::Sphere, "/Robot/Arm");
        s.Publish();
        const uint32_t leg = s.Define(SchemaId::Sphere, "/Robot/Leg");
        s.Publish();
        EXPECT_EQ(s.semantics.Stats().resolved, 1u);
        EXPECT_FALSE(s.semantics.Stats().reallocated);
        EXPECT_EQ(s.semantics.LabeledPrim(s.taxonomy, leg), robot);
    });
}

TEST(SemanticsTest, RelabelingResolvesItsSubtreeOnly) {
    ForEachScene([](Scene& s) {
        const uint32_t robot = s.Labeled("/Robot", "robot");
        const uint32_t arm = s.Define(SchemaId::Sphere, "/Robot/Arm");
        s.Define(SchemaId::Sphere, "/Robot/Arm/Finger");
        const uint32_t other = s.Labeled("/Other", "other");
        s.Publish();
        const uint32_t before = s.semantics.SemanticId(s.taxonomy, arm);
        const uint32_t untouched = s.semantics.SemanticId(s.taxonomy, other);
        s.Relabel(robot, "droid");
        s.Publish();
        EXPECT_EQ(s.semantics.Stats().resolved, 3u);
        EXPECT_NE(s.semantics.SemanticId(s.taxonomy, arm), before);
        EXPECT_EQ(s.semantics.SemanticId(s.taxonomy, other), untouched);
    });
}

TEST(SemanticsTest, RemovedPrimsLoseTheirEntries) {
    ForEachScene([](Scene& s) {
        const uint32_t ball = s.Labeled("/Ball", "ball");
        s.Publish();
        s.structure.Remove(ball);
        s.Publish();
        EXPECT_EQ(s.semantics.SemanticId(s.taxonomy, ball), 0u);
        EXPECT_EQ(s.semantics.LabeledPrim(s.taxonomy, ball), kNoPrim);
    });
}

TEST(SemanticsTest, StableIdsAreTheirPathsHash) {
    ForEachScene([](Scene& s) {
        const uint32_t ball = s.Define(SchemaId::Sphere, "/Ball");
        s.Publish();
        StableId uploaded{};
        s.device.GetQueue().Read(
            s.semantics.StableIds()->Slice(ball * sizeof(StableId), sizeof(StableId)), &uploaded);
        const StableId expected = HashPath("/Ball");
        for (uint32_t word = 0; word < 4; ++word) {
            EXPECT_EQ(uploaded.words[word], expected.words[word]);
        }
        EXPECT_NE(HashPath("/Ball").words[0], HashPath("/Ball2").words[0]);
        EXPECT_FALSE(s.semantics.StableIdEntries().empty());
    });
}

TEST(SemanticsTest, TransformsAloneResolveNothing) {
    ForEachScene([](Scene& s) {
        const uint32_t ball = s.Labeled("/Ball", "ball");
        s.Publish();
        const uint32_t runs = s.semantics.Runs();
        View(s.structure, SchemaId::Xformable, std::vector<uint32_t>{ball})
            .Get<Double3>(kXformableTranslate)
            .Set(Double3{1, 0, 0});
        s.Publish();
        EXPECT_EQ(s.semantics.Runs(), runs);
    });
}

}
}
