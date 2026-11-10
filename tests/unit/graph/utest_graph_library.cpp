// SPDX-License-Identifier: BSD-3-Clause
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/graph/library.h"

namespace ernie::graph {
namespace {

struct Chain {
    uint32_t noiseKind = 2;
    uint32_t input = 0;
    uint32_t shot = 0;
};

ShapeKey ChainShape(const Chain& chain) {
    Graph graph;
    const uint32_t output = graph.Add(0);
    const uint32_t noise = graph.Add(chain.noiseKind);
    const uint32_t signal = graph.Add(3);
    graph.Connect(output, 0, noise, 0);
    graph.Connect(noise, chain.input, signal, 0);
    const Ordering ordering = graph.Order();
    return GraphShape(graph, ordering.order, [&](uint32_t node, ShapeWriter& writer) {
        if (node == noise) {
            writer.Add(chain.shot);
        }
    });
}

TEST(ShapeTest, KeysFollowKindsConnectionsAndUniforms) {
    const ShapeKey key = ChainShape({});
    EXPECT_EQ(ChainShape({}), key);
    EXPECT_NE(ChainShape({.noiseKind = 5}), key);
    EXPECT_NE(ChainShape({.input = 1}), key);
    EXPECT_NE(ChainShape({.shot = 1}), key);
    EXPECT_EQ(ChainShape({}).Hash(), key.Hash());
}

TEST(DefinitionLibraryTest, InternsEachShapeOnceWithGrowingIds) {
    DefinitionLibrary<std::string> library(4);
    uint32_t made = 0;
    const auto make = [&](uint32_t id) {
        ++made;
        return "definition " + std::to_string(id);
    };
    const uint32_t first = library.Intern(ChainShape({}), make);
    EXPECT_EQ(first, 4u);
    EXPECT_EQ(library.Intern(ChainShape({}), make), first);
    EXPECT_EQ(made, 1u);

    const auto before = library.Snapshot();
    EXPECT_EQ(library.Snapshot(), before);
    const uint32_t second = library.Intern(ChainShape({.shot = 1}), make);
    EXPECT_EQ(second, 5u);
    const auto after = library.Snapshot();
    EXPECT_NE(after, before);
    EXPECT_GT(after->Version(), before->Version());
    EXPECT_EQ(before->Entries().size(), 1u);
    EXPECT_EQ(before->Find(second), nullptr);
    ASSERT_NE(after->Find(second), nullptr);
    EXPECT_EQ(*after->Find(second), "definition 5");
    EXPECT_EQ(*after->Find(first), "definition 4");
}

TEST(PublishedTest, UpdatesPublishNewSnapshotsAndFailedEditsKeepTheOldOne) {
    Published<std::vector<int>> published({1});
    const std::shared_ptr<const std::vector<int>> before = published.Get();
    published.Update([](std::vector<int>& values) { values.push_back(2); });
    EXPECT_EQ(*before, (std::vector<int>{1}));
    EXPECT_EQ(*published.Get(), (std::vector<int>{1, 2}));
    EXPECT_EQ(published.Update([](std::vector<int>& values) { return values.size(); }), 2u);
    EXPECT_THROW(published.Update([](std::vector<int>& values) {
        values.clear();
        throw std::runtime_error("rejected");
    }),
                 std::runtime_error);
    EXPECT_EQ(*published.Get(), (std::vector<int>{1, 2}));
}

}
}
