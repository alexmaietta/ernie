// SPDX-License-Identifier: BSD-3-Clause
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/graph/library.h"

namespace ernie::graph {
namespace {

struct Chain {
    uint32_t firstPrim = 0;
    uint32_t noiseKind = 2;
    std::string input = "in";
    uint32_t shot = 0;
};

ShapeKey ChainShape(const Chain& chain) {
    Graph graph;
    const uint32_t output = graph.Add(chain.firstPrim, 0);
    const uint32_t noise = graph.Add(chain.firstPrim + 1, chain.noiseKind);
    const uint32_t signal = graph.Add(chain.firstPrim + 2, 3);
    graph.Connect(output, "in", noise, "out");
    graph.Connect(noise, chain.input, signal, "rate");
    const Ordering ordering = graph.Order();
    return GraphShape(graph, ordering.order, [&](uint32_t node, ShapeWriter& writer) {
        if (node == noise) {
            writer.Add(chain.shot);
        }
    });
}

TEST(ShapeTest, KeysFollowKindsConnectionsAndUniformsButNotPrims) {
    const ShapeKey key = ChainShape({});
    EXPECT_EQ(ChainShape({.firstPrim = 40}), key);
    EXPECT_NE(ChainShape({.noiseKind = 5}), key);
    EXPECT_NE(ChainShape({.input = "signal"}), key);
    EXPECT_NE(ChainShape({.shot = 1}), key);
    EXPECT_EQ(ChainShape({.firstPrim = 40}).Hash(), key.Hash());
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
    EXPECT_EQ(library.Intern(ChainShape({.firstPrim = 9}), make), first);
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

}
}
