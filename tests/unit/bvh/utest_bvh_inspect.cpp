// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/bvh/inspect.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/bvh/reference.h"
#include "support/geometry.h"

namespace ernie::bvh {
namespace {

using Bytes = std::vector<std::byte>;

template <class T> T Get(const Bytes& bytes, uint64_t at) {
    T value;
    std::memcpy(&value, bytes.data() + at, sizeof(T));
    return value;
}

template <class T> void Put(Bytes& bytes, uint64_t at, const T& value) {
    std::memcpy(bytes.data() + at, &value, sizeof(T));
}

template <class Change> Bytes Changed(Bytes bytes, Change change) {
    BvhHeader header = test::Header(bytes);
    BvhNode root = Get<BvhNode>(bytes, header.nodes);
    BvhTriangle first = Get<BvhTriangle>(bytes, header.leaves);
    change(header, root, first);
    Put(bytes, 0, header);
    Put(bytes, header.nodes, root);
    Put(bytes, header.leaves, first);
    return bytes;
}

void ExpectFault(const Bytes& bytes, std::string_view fault) {
    const Inspection found = Inspect(bytes);
    for (const std::string& error : found.errors) {
        if (error.find(fault) != std::string::npos) {
            return;
        }
    }
    ADD_FAILURE() << "no fault mentions \"" << fault << "\" among " << found.errors.size();
}

uint32_t EmptySlot(const BvhNode& node) {
    for (uint32_t slot = 0; slot < kBvhWidth; ++slot) {
        if (BvhByte(const_cast<uint*>(node.meta), slot) == 0) {
            return slot;
        }
    }
    return kBvhWidth;
}

TEST(InspectTest, FindsFaultsInTheHeader) {
    const Bytes good = BuildReference(test::Soup(100, 1));
    ASSERT_TRUE(Inspect(good).errors.empty());

    ExpectFault(Bytes(good.begin(), good.begin() + 10), "fewer than its header");
    ExpectFault(Changed(good, [](BvhHeader& h, auto&, auto&) { h.version = 99; }), "version");
    ExpectFault(Changed(good, [](BvhHeader& h, auto&, auto&) { h.kind = 9; }), "kind 9");
    ExpectFault(Changed(good, [](BvhHeader& h, auto&, auto&) { h.nodeCount = 1u << 30; }),
                "run past");
    ExpectFault(Changed(good, [](BvhHeader& h, auto&, auto&) { h.leaves = h.nodes; }), "overlap");
    ExpectFault(Changed(good, [](BvhHeader& h, auto&, auto&) { h.nodeCount = 0; }), "no nodes");
    ExpectFault(Changed(good, [](BvhHeader& h, auto&, auto&) { h.leafCount -= 1; }),
                "not one of its");
    Bytes padded = good;
    padded.resize(good.size() + sizeof(BvhTriangle));
    ExpectFault(Changed(padded, [](BvhHeader& h, auto&, auto&) { h.leafCount += 1; }),
                "leaf records, but its leaves hold");

    const test::Instances instances = test::Scatter(10, 2);
    const Bytes placed = BuildReference(instances.records, instances.bounds);
    const std::vector<BvhBox> boxes = instances.WorldBoxes();
    const Inspection found = Inspect(placed, std::span(boxes).first(4));
    ASSERT_FALSE(found.errors.empty());
    EXPECT_NE(found.errors.front().find("but 4 boxes were given"), std::string::npos);
}

TEST(InspectTest, FindsFaultsInNodes) {
    const Bytes good = BuildReference(test::Soup(20, 1));
    ExpectFault(Changed(good, [](auto&, BvhNode& root, auto&) { root.firstChild = 1u << 20; }),
                "past its");
    ExpectFault(Changed(good,
                        [](auto&, BvhNode& root, auto&) {
                            const uint32_t imask = root.exponents >> 24;
                            root.exponents &= ~(uint(imask & (0u - imask)) << 24);
                        }),
                "never reached");
    ExpectFault(Changed(good,
                        [](auto&, BvhNode& root, auto&) {
                            const uint32_t imask = root.exponents >> 24;
                            root.exponents &= ~(uint(imask & (0u - imask)) << 24);
                        }),
                "internal child's meta byte");
    const Bytes garbled =
        Changed(good, [](auto&, BvhNode& root, auto&) { root.meta[0] = 0xA5A5A5A5; });
    ExpectFault(garbled, "neither empty, internal nor a leaf's");
    ExpectFault(garbled, "is in its imask, but its meta byte");

    const Bytes small = BuildReference(test::Soup(5, 2));
    const BvhNode root = Get<BvhNode>(small, test::Header(small).nodes);
    ASSERT_LT(EmptySlot(root), kBvhWidth);
    ExpectFault(Changed(small, [](auto&, BvhNode& node,
                                  auto&) { node.exponents |= (1u << EmptySlot(node)) << 24; }),
                "empty but in its imask");
}

TEST(InspectTest, FindsFaultsInLeafRecords) {
    const Bytes good = BuildReference(test::Soup(100, 1));
    ExpectFault(Changed(good, [](auto&, auto&, BvhTriangle& first) { first.index = 100000; }),
                "past its 100 primitives");
    ExpectFault(Changed(good, [](auto&, auto&, BvhTriangle& first) { first.index ^= 1; }),
                "in 2 leaf records");
    ExpectFault(Changed(good, [](auto&, auto&, BvhTriangle& first) { first.a[0] = 50; }),
                "lies outside");

    Bytes broken = good;
    const BvhHeader header = test::Header(broken);
    for (uint32_t i = 0; i < header.leafCount; ++i) {
        const uint64_t at = header.leaves + uint64_t(i) * sizeof(BvhTriangle);
        BvhTriangle triangle = Get<BvhTriangle>(broken, at);
        triangle.index = 0;
        Put(broken, at, triangle);
    }
    const Inspection found = Inspect(broken);
    ASSERT_EQ(found.errors.size(), 33u);
    EXPECT_NE(found.errors.back().find("more faults"), std::string::npos);
}

}
}
