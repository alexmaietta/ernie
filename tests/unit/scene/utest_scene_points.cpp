// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/tables/points.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/bvh/builder.h"
#include "ernie/device/device.h"
#include "ernie/device/queue.h"
#include "ernie/scene/columns/column_store.h"
#include "ernie/scene/publish/publisher.h"
#include "ernie/scene/schema/columns.h"
#include "ernie/scene/structure/structure.h"
#include "support/devices.h"

namespace ernie {
namespace {

constexpr uint32_t kMaxRefits = 2;

struct Scene {
    ColumnStore& store;
    Structure& structure;
    Publisher& publisher;
    PointsTable& points;
    Ordinal ordinal = 0;

    void Publish() {
        publisher.Publish(++ordinal).Wait();
    }

    void Move(uint32_t prim, const std::vector<Float3>& moved) {
        const uint32_t row = structure.GetPrims().Row(prim, SchemaId::PointBased);
        const ArrayRef array = store.Array(kPointBasedPoints, row);
        store.Arrays(ColumnType::Float3Array).Write(array, std::as_bytes(std::span(moved)));
        store.SetArray(kPointBasedPoints, row, array);
    }

    std::vector<Float3> Bounds(uint32_t prim) {
        const PointsRecord& record =
            points.Record(structure.GetPrims().Row(prim, SchemaId::Points));
        return Items<Float3>(store.Arrays(ColumnType::Float3Array).Read(record.bounds));
    }
};

template <class Test> void ForEachScene(Test test) {
    test::ForEachDevice([&](device::Device& device) {
        if (device.Adapter().backend == device::Backend::Hip) {
            return;
        }
        ColumnStore store(device);
        Structure structure(store);
        Publisher publisher(structure);
        const std::unique_ptr<bvh::Builder> builder = bvh::Builder::Create(device);
        PointsTable points(structure, *builder, kMaxRefits);
        publisher.Add(points);
        Scene scene{store, structure, publisher, points};
        test(scene);
    });
}

void ExpectBounds(const std::vector<Float3>& bounds, Float3 lower, Float3 upper) {
    ASSERT_EQ(bounds.size(), 2u);
    EXPECT_FLOAT_EQ(bounds[0].x, lower.x);
    EXPECT_FLOAT_EQ(bounds[0].y, lower.y);
    EXPECT_FLOAT_EQ(bounds[0].z, lower.z);
    EXPECT_FLOAT_EQ(bounds[1].x, upper.x);
    EXPECT_FLOAT_EQ(bounds[1].y, upper.y);
    EXPECT_FLOAT_EQ(bounds[1].z, upper.z);
}

TEST(PointsTest, MovedPointsRefitUntilTheLimitAndThenRebuild) {
    ForEachScene([](Scene& s) {
        const uint32_t prim = s.structure
                                  .Define<Points>(std::vector<std::string>{"/Cloud"},
                                                  {.points = {{0, 0, 0}, {4, 0, 0}}, .widths = {2}})
                                  .firstPrim;
        s.Publish();
        EXPECT_EQ(s.points.Stats().built, 1u);
        EXPECT_EQ(s.points.Stats().refit, 0u);
        ExpectBounds(s.Bounds(prim), {-1, -1, -1}, {5, 1, 1});
        const uint64_t blas = s.points.Record(0).blas;
        for (uint32_t step = 1; step <= kMaxRefits + 1; ++step) {
            const float x = float(step);
            s.Move(prim, {{x, 0, 0}, {x + 4, 0, 0}});
            s.Publish();
            const bool rebuilt = step > kMaxRefits;
            EXPECT_EQ(s.points.Stats().built, rebuilt ? 1u : 0u) << step;
            EXPECT_EQ(s.points.Stats().refit, rebuilt ? 0u : 1u) << step;
            ExpectBounds(s.Bounds(prim), {x - 1, -1, -1}, {x + 5, 1, 1});
        }
        EXPECT_EQ(s.points.Record(0).blas, blas);
    });
}

TEST(PointsTest, ACountChangeRebuildsANewBlas) {
    ForEachScene([](Scene& s) {
        const std::vector<std::string> paths = {"/Cloud"};
        const uint32_t prim =
            s.structure.Define<Points>(paths, {.points = {{0, 0, 0}, {4, 0, 0}}}).firstPrim;
        s.Publish();
        s.Move(prim, {{1, 0, 0}, {5, 0, 0}});
        s.Publish();
        ASSERT_EQ(s.points.Stats().refit, 1u);
        const uint32_t row = s.structure.GetPrims().Row(prim, SchemaId::PointBased);
        const std::vector<Float3> grown = {{0, 0, 0}, {4, 0, 0}, {0, 8, 0}};
        ArrayPool& float3s = s.store.Arrays(ColumnType::Float3Array);
        const ArrayRef array = float3s.Allocate(3);
        float3s.Write(array, std::as_bytes(std::span(grown)));
        s.store.SetArray(kPointBasedPoints, row, array);
        s.Publish();
        EXPECT_EQ(s.points.Stats().built, 1u);
        EXPECT_EQ(s.points.Stats().refit, 0u);
        EXPECT_EQ(s.points.Record(0).pointCount, 3u);
        ExpectBounds(s.Bounds(prim), {-0.005f, -0.005f, -0.005f}, {4.005f, 8.005f, 0.005f});
    });
}

TEST(PointsTest, PointsWrittenOnTheDeviceMoveTheirBounds) {
    ForEachScene([](Scene& s) {
        const uint32_t prim = s.structure
                                  .Define<Points>(std::vector<std::string>{"/Cloud"},
                                                  {.points = {{0, 0, 0}, {4, 0, 0}}, .widths = {2}})
                                  .firstPrim;
        s.Publish();
        const uint32_t row = s.structure.GetPrims().Row(prim, SchemaId::PointBased);
        const ArrayRef array = s.store.Array(kPointBasedPoints, row);
        const std::vector<Float3> moved = {{0, 10, 0}, {0, 14, 3}};
        device::Device& device = s.store.GetDevice();
        const Ref<device::Buffer> source =
            device.CreateBuffer({.label = "simulated points"}, std::span<const Float3>(moved));
        device::Queue& queue = device.GetQueue();
        device::CommandEncoder encoder = queue.CreateEncoder();
        encoder.Copy(source->Slice(0), s.store.Arrays(ColumnType::Float3Array).Range(array));
        queue.Submit(encoder.Finish());
        s.store.SetArray(kPointBasedPoints, row, array);
        s.Publish();
        EXPECT_EQ(s.points.Stats().refit, 1u);
        ExpectBounds(s.Bounds(prim), {-1, 9, -1}, {1, 15, 4});
    });
}

std::vector<Float3> Line(uint32_t count, float x) {
    std::vector<Float3> points;
    for (uint32_t i = 0; i < count; ++i) {
        points.push_back({x, float(i), 0});
    }
    return points;
}

TEST(PointsTest, RecountsWithinCapacityKeepTheBlasAndBoundOnlyTheLiving) {
    ForEachScene([](Scene& s) {
        const uint32_t a =
            s.structure.Define<Points>(std::vector<std::string>{"/A"}, {.points = Line(8, 0)})
                .firstPrim;
        const uint32_t b =
            s.structure.Define<Points>(std::vector<std::string>{"/B"}, {.points = Line(5, 20)})
                .firstPrim;
        s.Publish();
        const auto row = [&](uint32_t prim) {
            return s.structure.GetPrims().Row(prim, SchemaId::Points);
        };
        const auto blas = [&](uint32_t prim) { return s.points.Record(row(prim)).blas; };
        const auto recount = [&](uint32_t prim, const std::vector<Float3>& points) {
            ArrayPool& float3s = s.store.Arrays(ColumnType::Float3Array);
            const ArrayRef array = float3s.Allocate(uint32_t(points.size()));
            if (!points.empty()) {
                float3s.Write(array, std::as_bytes(std::span(points)));
            }
            s.store.SetArray(kPointBasedPoints,
                             s.structure.GetPrims().Row(prim, SchemaId::PointBased), array);
        };
        const float r = kDefaultPointWidth / 2;
        const uint64_t first = blas(a);
        const uint64_t other = blas(b);

        recount(a, Line(6, 0));
        recount(b, Line(3, 20));
        s.Publish();
        EXPECT_EQ(blas(a), first);
        EXPECT_EQ(blas(b), other);
        EXPECT_EQ(s.points.Stats().built, 2u);
        EXPECT_EQ(s.points.Stats().points, 9u);
        EXPECT_EQ(s.points.Stats().capacity, 13u);
        EXPECT_EQ(s.points.Record(row(a)).pointCount, 6u);
        ExpectBounds(s.Bounds(a), {-r, -r, -r}, {r, 5 + r, r});
        ExpectBounds(s.Bounds(b), {20 - r, -r, -r}, {20 + r, 2 + r, r});

        s.Move(a, Line(6, 1));
        s.Publish();
        EXPECT_EQ(s.points.Stats().refit, 1u);
        EXPECT_EQ(s.points.Stats().built, 0u);
        ExpectBounds(s.Bounds(a), {1 - r, -r, -r}, {1 + r, 5 + r, r});

        recount(a, Line(8, 0));
        s.Publish();
        EXPECT_EQ(blas(a), first);
        EXPECT_EQ(s.points.Stats().built, 1u);
        ExpectBounds(s.Bounds(a), {-r, -r, -r}, {r, 7 + r, r});

        recount(a, Line(9, 0));
        s.Publish();
        EXPECT_NE(blas(a), first);
        const uint64_t grown = blas(a);
        EXPECT_EQ(s.points.Stats().capacity, 12u);
        ExpectBounds(s.Bounds(a), {-r, -r, -r}, {r, 8 + r, r});

        recount(a, Line(12, 0));
        s.Publish();
        EXPECT_EQ(blas(a), grown);
        EXPECT_EQ(s.points.Stats().capacity, 12u);
        ExpectBounds(s.Bounds(a), {-r, -r, -r}, {r, 11 + r, r});

        recount(a, Line(3, 0));
        s.Publish();
        EXPECT_EQ(blas(a), grown);
        EXPECT_EQ(s.points.Stats().capacity, 12u);
        ExpectBounds(s.Bounds(a), {-r, -r, -r}, {r, 2 + r, r});

        recount(a, Line(2, 0));
        s.Publish();
        EXPECT_NE(blas(a), grown);
        EXPECT_EQ(s.points.Stats().capacity, 2u);
        ExpectBounds(s.Bounds(a), {-r, -r, -r}, {r, 1 + r, r});

        recount(a, {});
        s.Publish();
        EXPECT_EQ(blas(a), 0u);
        EXPECT_EQ(s.points.Record(row(a)).pointCount, 0u);

        recount(a, Line(4, 0));
        s.Publish();
        EXPECT_NE(blas(a), 0u);
        ExpectBounds(s.Bounds(a), {-r, -r, -r}, {r, 3 + r, r});
        ExpectBounds(s.Bounds(b), {20 - r, -r, -r}, {20 + r, 2 + r, r});
    });
}

}
}
