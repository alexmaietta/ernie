// SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/graph/graph.h"
#include "ernie/util/error.h"

namespace ernie::graph {
namespace {

uint32_t Position(const std::vector<uint32_t>& order, uint32_t node) {
    return uint32_t(std::ranges::find(order, node) - order.begin());
}

TEST(GraphTest, OrdersEverySourceBeforeTheNodesReadingIt) {
    Graph graph;
    const uint32_t output = graph.Add(0);
    const uint32_t adc = graph.Add(1);
    const uint32_t noise = graph.Add(2);
    const uint32_t signal = graph.Add(3);
    const uint32_t loose = graph.Add(4);
    graph.Connect(output, 0, adc, 0);
    graph.Connect(adc, 0, noise, 0);
    graph.Connect(adc, 1, signal, 0);
    graph.Connect(noise, 0, signal, 0);
    const Ordering ordering = graph.Order();
    ASSERT_TRUE(ordering.cycle.empty());
    ASSERT_EQ(ordering.order.size(), 5u);
    EXPECT_LT(Position(ordering.order, signal), Position(ordering.order, noise));
    EXPECT_LT(Position(ordering.order, noise), Position(ordering.order, adc));
    EXPECT_LT(Position(ordering.order, adc), Position(ordering.order, output));
    EXPECT_NE(Position(ordering.order, loose), 5u);
}

TEST(GraphTest, ReportsTheNodesOfACycle) {
    Graph graph;
    const uint32_t a = graph.Add(0);
    const uint32_t b = graph.Add(0);
    const uint32_t c = graph.Add(0);
    const uint32_t d = graph.Add(0);
    graph.Connect(a, 0, b, 0);
    graph.Connect(b, 0, c, 0);
    graph.Connect(c, 0, b, 0);
    graph.Connect(c, 1, d, 0);
    const Ordering ordering = graph.Order();
    EXPECT_TRUE(ordering.order.empty());
    std::vector<uint32_t> cycle = ordering.cycle;
    std::ranges::sort(cycle);
    EXPECT_EQ(cycle, (std::vector<uint32_t>{b, c}));
}

TEST(GraphTest, AnInputTakesOneConnection) {
    Graph graph;
    const uint32_t a = graph.Add(0);
    const uint32_t b = graph.Add(0);
    graph.Connect(a, 0, b, 0);
    EXPECT_THROW(graph.Connect(a, 0, b, 1), Error);
    EXPECT_THROW(graph.Connect(a, 1, 7, 0), Error);
}

}
}
