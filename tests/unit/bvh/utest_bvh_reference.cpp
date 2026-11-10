// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/bvh/reference.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/bvh/inspect.h"
#include "ernie/util/error.h"
#include "support/geometry.h"

namespace ernie::bvh {
namespace {

void ExpectWellFormed(const std::vector<std::byte>& bytes, uint32_t primitives, uint32_t kind,
                      std::span<const BvhBox> boxes = {}) {
    const Inspection found = Inspect(bytes, boxes);
    ASSERT_TRUE(found.errors.empty()) << found.errors.front();
    EXPECT_EQ(found.records, primitives);
    EXPECT_EQ(test::Header(bytes).primitives, primitives);
    EXPECT_EQ(test::Header(bytes).kind, kind);
    if (primitives > kBvhLeafSize) {
        EXPECT_GT(found.childrenPerNode, 2);
        EXPECT_LE(found.recordsPerLeaf, kBvhLeafSize);
    }
}

TEST(ReferenceTest, BuildsWellFormedBvhsOfTriangles) {
    for (const std::vector<Triangle>& mesh :
         {test::Soup(0, 1), test::Soup(1, 2), test::Soup(500, 3), test::Grid(8),
          test::Points(40)}) {
        SCOPED_TRACE(mesh.size());
        ExpectWellFormed(BuildReference(mesh), uint32_t(mesh.size()), kBvhTriangles);
    }
}

TEST(ReferenceTest, CollapsesIntoNearlyFullNodes) {
    for (const std::vector<Triangle>& mesh : {test::Grid(32), test::Points(5000)}) {
        SCOPED_TRACE(mesh.size());
        const Inspection found = Inspect(BuildReference(mesh));
        ASSERT_TRUE(found.errors.empty()) << found.errors.front();
        EXPECT_GT(found.childrenPerNode, 7.5);
        EXPECT_LT(found.nodes, mesh.size() / 12);
        EXPECT_LT(found.nodeCost, found.sahCost);
        EXPECT_DOUBLE_EQ(found.nodeCost + found.recordCost, found.sahCost);
    }
}

TEST(ReferenceTest, BuildsWellFormedBvhsOfBoxes) {
    const std::vector<BvhBox> boxes = test::Boxes(300, 4);
    ExpectWellFormed(BuildReference(boxes), 300, kBvhBoxes);
}

TEST(ReferenceTest, BuildsWellFormedBvhsOfInstances) {
    const test::Instances instances = test::Scatter(200, 5);
    const std::vector<std::byte> bytes = BuildReference(instances.records, instances.bounds);
    ExpectWellFormed(bytes, 200, kBvhInstances, instances.WorldBoxes());
    EXPECT_THROW(BuildReference(instances.records, std::span(instances.bounds).first(3)), Error);
}

}
}
