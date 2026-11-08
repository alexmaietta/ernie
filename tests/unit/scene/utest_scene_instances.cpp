// SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include "ernie/bvh/builder.h"
#include "ernie/bvh/kernels/bvh.h"
#include "ernie/device/device.h"
#include "ernie/device/queue.h"
#include "ernie/scene/columns/column_store.h"
#include "ernie/scene/kernels/shapes.h"
#include "ernie/scene/publish/publisher.h"
#include "ernie/scene/schema/columns.h"
#include "ernie/scene/structure/structure.h"
#include "ernie/scene/tables/instances.h"
#include "ernie/scene/tables/meshes.h"
#include "ernie/scene/tables/transforms.h"
#include "ernie/scene/views/attribute.h"
#include "ernie/scene/views/view.h"
#include "support/devices.h"

namespace ernie {
namespace {

const std::vector<Float3> kQuad = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};

struct Scene {
    device::Device& device;
    ColumnStore& store;
    Structure& structure;
    Publisher& publisher;
    InstanceTable& instances;
    Ordinal ordinal = 0;

    uint32_t Define(SchemaId schema, const std::string& path) {
        return structure.Define(schema, std::vector<std::string>{path}).firstPrim;
    }

    template <class T> uint32_t Define(const std::string& path, const typename T::Desc& desc) {
        return structure.Define<T>(std::vector<std::string>{path}, desc).firstPrim;
    }

    void Publish() {
        publisher.Publish(++ordinal).Wait();
    }

    void Move(uint32_t prim, Double3 to) {
        View(structure, SchemaId::Xformable, std::vector<uint32_t>{prim})
            .Get<Double3>(kXformableTranslate)
            .Set(to);
    }

    InstanceRecord Record(uint32_t prim, uint32_t tlas) {
        const std::span<const uint32_t> first = instances.FirstInstances();
        const std::span<const uint32_t> prims = instances.InstancePrims();
        const auto begin = prims.begin() + first[tlas];
        const auto found = std::find(begin, prims.begin() + first[tlas + 1], prim);
        EXPECT_NE(found, prims.begin() + first[tlas + 1]) << "prim " << prim << " in " << tlas;
        InstanceRecord record{};
        device.GetQueue().Read(
            instances.Records()->Slice(uint64_t(found - prims.begin()) * sizeof(InstanceRecord),
                                       sizeof(InstanceRecord)),
            &record);
        return record;
    }
};

template <class Test> void ForEachScene(Test test) {
    test::ForEachDevice([&](device::Device& device) {
        ColumnStore store(device);
        Structure structure(store);
        Publisher publisher(structure);
        const std::unique_ptr<bvh::Builder> builder = bvh::Builder::Create(device);
        TransformTable hierarchy(structure);
        MeshTable geometry(structure, *builder);
        InstanceTable instances(structure, *builder, 2);
        publisher.Add(hierarchy);
        publisher.Add(geometry);
        publisher.Add(instances);
        Scene scene{device, store, structure, publisher, instances};
        test(scene);
    });
}

std::vector<uint32_t> Worlds(Scene& s, uint32_t count) {
    std::vector<uint32_t> balls;
    for (uint32_t world = 0; world < count; ++world) {
        s.Define<World>(fmt::format("/Env{}", world), {.translate = {10.0 * (world + 1), 0, 0}});
        balls.push_back(
            s.Define<Sphere>(fmt::format("/Env{}/Ball", world), {.translate = {1, 0, 0}}));
    }
    return balls;
}

TEST(InstancesTest, EachWorldHoldsItsGprimsAndTheGlobalOnes) {
    ForEachScene([](Scene& s) {
        const std::vector<uint32_t> balls = Worlds(s, 3);
        const uint32_t sun = s.Define<Sphere>("/Sun", {.translate = {5, 0, 0}});
        s.Define(SchemaId::Sphere, "/Moon");
        s.Publish();
        const InstanceStats& stats = s.instances.Stats();
        EXPECT_EQ(stats.tlases, 4u);
        EXPECT_EQ(stats.instances, 11u);
        EXPECT_EQ(stats.built, 4u);
        EXPECT_TRUE(stats.relaid);
        const std::span<const uint32_t> counts = s.instances.InstanceCounts();
        EXPECT_EQ(counts[0], 3u);
        EXPECT_EQ(counts[3], 2u);

        const InstanceRecord ball = s.Record(balls[1], 1);
        EXPECT_NE(ball.blas, 0u);
        EXPECT_EQ(ball.custom >> 24, kGeometryMask);
        EXPECT_FLOAT_EQ(ball.transform[3], 1.0f);
        EXPECT_FLOAT_EQ(s.Record(sun, 1).transform[3], 5.0f - 20.0f);
        EXPECT_FLOAT_EQ(s.Record(sun, 3).transform[3], 5.0f);
    });
}

TEST(InstancesTest, MovingAPrimWritesAndRefitsOnlyItsWorld) {
    ForEachScene([](Scene& s) {
        const std::vector<uint32_t> balls = Worlds(s, 3);
        s.Define(SchemaId::Sphere, "/Sun");
        s.Publish();
        s.Move(balls[1], {2, 0, 0});
        s.Publish();
        const InstanceStats& stats = s.instances.Stats();
        EXPECT_FALSE(stats.relaid);
        EXPECT_EQ(stats.written, 2u);
        EXPECT_EQ(stats.refit, 1u);
        EXPECT_EQ(stats.built, 0u);
        EXPECT_FLOAT_EQ(s.Record(balls[1], 1).transform[3], 2.0f);
    });
}

TEST(InstancesTest, RefitsGiveWayToABuildAfterTheirLimit) {
    ForEachScene([](Scene& s) {
        const std::vector<uint32_t> balls = Worlds(s, 2);
        s.Publish();
        for (uint32_t step = 1; step <= 3; ++step) {
            s.Move(balls[0], {double(step), 0, 0});
            s.Publish();
            EXPECT_EQ(s.instances.Stats().refit, step <= 2 ? 1u : 0u) << step;
            EXPECT_EQ(s.instances.Stats().built, step <= 2 ? 0u : 1u) << step;
        }
    });
}

TEST(InstancesTest, HidingAPrimRebuildsWithItInactive) {
    ForEachScene([](Scene& s) {
        const std::vector<uint32_t> balls = Worlds(s, 2);
        s.Publish();
        View(s.structure, SchemaId::Imageable, std::vector<uint32_t>{balls[0]})
            .Get<Visibility>(kImageableVisibility)
            .Set(Visibility::Invisible);
        s.Publish();
        EXPECT_EQ(s.instances.Stats().built, 1u);
        EXPECT_EQ(s.instances.Stats().refit, 0u);
        const InstanceRecord hidden = s.Record(balls[0], 0);
        EXPECT_EQ(hidden.blas, 0u);
        EXPECT_EQ(hidden.custom >> 24, 0u);
    });
}

TEST(InstancesTest, GlobalChangesReachEveryTlas) {
    ForEachScene([](Scene& s) {
        Worlds(s, 3);
        const uint32_t sun = s.Define(SchemaId::Sphere, "/Sun");
        s.Publish();
        s.Move(sun, {0, 0, 9});
        s.Publish();
        EXPECT_EQ(s.instances.Stats().refit, 4u);
        EXPECT_FLOAT_EQ(s.Record(sun, 2).transform[11], 9.0f);
    });
}

TEST(InstancesTest, ANewBlasMarksOnlyItsWorld) {
    ForEachScene([](Scene& s) {
        s.Define(SchemaId::World, "/Env0");
        const uint32_t quad = s.Define<Mesh>(
            "/Env0/Quad",
            {.points = kQuad, .faceVertexIndices = {0, 1, 2, 3}, .faceVertexCounts = {4}});
        s.structure.Clone(s.structure.Find("/Env0"), std::vector<std::string>{"/Env1", "/Env2"});
        s.Publish();
        const uint32_t clone = s.structure.Find("/Env2/Quad");
        EXPECT_EQ(s.Record(clone, 2).blas, s.Record(quad, 0).blas);

        ArrayHeap& float3s = s.store.Arrays(ColumnType::Float3Array);
        const ArrayRef own = float3s.Allocate(4);
        float3s.Write(own, std::as_bytes(std::span(kQuad)));
        s.store.SetArray(kPointBasedPoints, s.structure.GetPrims().Row(clone, SchemaId::PointBased),
                         own);
        s.Publish();
        EXPECT_EQ(s.instances.Stats().written, 1u);
        EXPECT_EQ(s.instances.Stats().built, 1u);
        EXPECT_NE(s.Record(clone, 2).blas, s.Record(quad, 0).blas);
    });
}

TEST(InstancesTest, NothingChangedSubmitsNothing) {
    ForEachScene([](Scene& s) {
        Worlds(s, 2);
        s.Publish();
        const uint64_t submitted = s.device.GetQueue().Submitted();
        s.Publish();
        EXPECT_EQ(s.device.GetQueue().Submitted(), submitted);
    });
}

TEST(InstancesTest, AreaLightsAreEmittersAndPointLightsAreNot) {
    ForEachScene([](Scene& s) {
        const uint32_t lamp = s.Define<SphereLight>("/Lamp", {});
        const uint32_t hidden = s.Define<SphereLight>("/Hidden", {.visibleInPrimaryRay = false});
        const uint32_t point = s.Define<SphereLight>("/Point", {.treatAsPoint = true});
        const uint32_t sun = s.Define<DistantLight>("/Sun", {});
        s.Publish();
        EXPECT_EQ(s.instances.Stats().instances, 3u);
        EXPECT_EQ(s.Record(lamp, 0).custom >> 24, kEmitterMask | kCameraEmitterMask);
        EXPECT_EQ(s.Record(hidden, 0).custom >> 24, kEmitterMask);
        EXPECT_EQ(s.Record(point, 0).blas, 0u);
        (void)sun;
    });
}

}
}
