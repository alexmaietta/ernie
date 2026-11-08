// SPDX-License-Identifier: BSD-3-Clause
#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/device/device.h"
#include "ernie/device/queue.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/scene_impl.h"
#include "ernie/scene/tables/table.h"
#include "ernie/util/error.h"
#include "support/devices.h"

namespace ernie {
namespace {

template <class Test> void ForEachScene(Test test) {
    test::ForEachDevice([&](device::Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        test(*scene, device);
    });
}

RenderTables ReadHeader(device::Device& device, const Snapshot& snapshot) {
    RenderTables tables{};
    device.GetQueue().Read(PinnedTables::From(snapshot).buffer->Slice(0, sizeof(tables)), &tables);
    return tables;
}

const RefCounted* Held(const Snapshot& snapshot, uint64_t address) {
    for (const Ref<const RefCounted>& held : PinnedTables::From(snapshot).held) {
        const auto* buffer = dynamic_cast<const device::Buffer*>(held.Get());
        if (buffer != nullptr && buffer->Address() == address) {
            return buffer;
        }
    }
    return nullptr;
}

TEST(SnapshotTest, TheHeaderOnTheDeviceMatchesTheTablesItPins) {
    ForEachScene([](Scene& scene, device::Device& device) {
        scene.Define<Sphere>("/Ball");
        scene.Define<DistantLight>("/Sun");
        scene.Publish().Wait();
        const Snapshot snapshot = scene.Acquire(Stream::Host());
        const PinnedTables& pinned = PinnedTables::From(snapshot);
        const RenderTables header = ReadHeader(device, snapshot);
        EXPECT_EQ(std::memcmp(&header, &pinned.Header<RenderTables>(), sizeof(header)), 0);
        SceneImpl& impl = SceneImpl::Get(scene);
        EXPECT_EQ(header.instances.address, impl.GetInstances().Layout()->Address());
        EXPECT_EQ(header.lights.address, impl.GetLights().Layout()->Address());
        EXPECT_EQ(header.origins.address, impl.GetInstances().Origins()->Address());
        EXPECT_NE(Held(snapshot, header.instances.address), nullptr);
        EXPECT_NE(Held(snapshot, header.lights.address), nullptr);
        EXPECT_NE(Held(snapshot, header.origins.address), nullptr);
        scene.Release(snapshot);
    });
}

TEST(SnapshotTest, TheHeaderCarriesThePrimTableAsPublished) {
    ForEachScene([](Scene& scene, device::Device& device) {
        scene.Define<Sphere>("/Ball");
        scene.Publish().Wait();
        const Snapshot first = scene.Acquire(Stream::Host());
        const RenderTables before = ReadHeader(device, first);
        const AuthoredData data = SceneImpl::Get(scene).GetColumnStore().Data();
        EXPECT_EQ(before.prims.count, data.primCount);
        EXPECT_EQ(before.prims.schema.pages.address, data.prims.schema.pages.address);
        EXPECT_EQ(before.prims.rows.pages.address, data.rows.pages.address);
        scene.Release(first);

        scene.Define<Sphere>("/Other");
        scene.Publish().Wait();
        const Snapshot second = scene.Acquire(Stream::Host());
        EXPECT_EQ(ReadHeader(device, second).prims.count, before.prims.count + 1);
        scene.Release(second);
    });
}

TEST(SnapshotTest, OnlyRelayoutsReplaceTheHeader) {
    ForEachScene([](Scene& scene, device::Device& device) {
        const Prim ball = scene.Define<Sphere>("/Ball");
        scene.Publish().Wait();
        const Snapshot first = scene.Acquire(Stream::Host());
        const RenderTables before = ReadHeader(device, first);

        ball.Set(Sphere::kRadius, 2.0f);
        scene.Publish().Wait();
        const Snapshot moved = scene.Acquire(Stream::Host());
        EXPECT_EQ(moved.tables, first.tables);

        scene.Define<World>("/Env0");
        scene.Define<Sphere>("/Env0/Ball");
        scene.Publish().Wait();
        const Snapshot relaid = scene.Acquire(Stream::Host());
        EXPECT_NE(relaid.tables, first.tables);
        EXPECT_EQ(PinnedTables::From(relaid).Header<RenderTables>().worldCount, 1u);
        const RenderTables kept = ReadHeader(device, first);
        EXPECT_EQ(std::memcmp(&kept, &before, sizeof(kept)), 0);
        for (const Snapshot* snapshot : {&first, &moved, &relaid}) {
            scene.Release(*snapshot);
        }
    });
}

TEST(SnapshotTest, HeldSnapshotsKeepRelaidTablesAlive) {
    ForEachScene([](Scene& scene, device::Device& device) {
        scene.Define<Sphere>("/Ball");
        scene.Publish().Wait();
        const Snapshot held = scene.Acquire(Stream::Host());
        const uint64_t layout = PinnedTables::From(held).Header<RenderTables>().instances.address;
        const RefCounted* buffer = Held(held, layout);
        ASSERT_NE(buffer, nullptr);

        scene.Define<World>("/Env0");
        scene.Publish().Wait();
        ASSERT_NE(SceneImpl::Get(scene).GetInstances().Layout()->Address(), layout);
        EXPECT_EQ(buffer->RefCount(), 1u);
        EXPECT_EQ(ReadHeader(device, held).instances.address, layout);
        scene.Release(held);
    });
}

TEST(SnapshotTest, HeldSnapshotsKeepRemovedMeshesTraceable) {
    ForEachScene([](Scene& scene, device::Device&) {
        const Prim quad = scene.Define<Mesh>("/Quad", {.points = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}},
                                                       .faceVertexIndices = {0, 1, 2},
                                                       .faceVertexCounts = {3}});
        scene.Publish().Wait();
        std::optional<MeshShape> shape = SceneImpl::Get(scene).GetMeshes().Find(quad.Index());
        ASSERT_TRUE(shape && shape->blas);
        const bvh::Blas* blas = shape->blas.Get();
        shape.reset();
        const Snapshot held = scene.Acquire(Stream::Host());

        scene.Remove(quad);
        scene.Publish().Wait();
        EXPECT_FALSE(SceneImpl::Get(scene).GetMeshes().Find(quad.Index()));
        bool pinned = false;
        for (const Ref<const RefCounted>& item : PinnedTables::From(held).held) {
            pinned = pinned || item.Get() == blas;
        }
        ASSERT_TRUE(pinned);
        EXPECT_EQ(blas->RefCount(), 1u);
        scene.Release(held);
    });
}

TEST(SnapshotTest, AcquiringWithoutAPublishSharesOnePin) {
    ForEachScene([](Scene& scene, device::Device&) {
        scene.Define<Sphere>("/Ball");
        scene.Publish().Wait();
        const Snapshot first = scene.Acquire(Stream::Host());
        const Snapshot second = scene.Acquire(Stream::Host());
        EXPECT_EQ(first.pinned, second.pinned);
        scene.Publish().Wait();
        const Snapshot idle = scene.Acquire(Stream::Host());
        EXPECT_EQ(idle.tables, first.tables);
        for (const Snapshot* snapshot : {&first, &second, &idle}) {
            scene.Release(*snapshot);
        }
    });
}

TEST(SnapshotTest, VersionsMoveOnlyWithTheTablesThatChanged) {
    ForEachScene([](Scene& scene, device::Device&) {
        const Prim ball = scene.Define<Sphere>("/Ball");
        scene.Publish().Wait();
        const auto versions = [&] {
            const Snapshot snapshot = scene.Acquire(Stream::Host());
            const std::array<uint64_t, 3> read = {snapshot.Version(), snapshot.Version("hierarchy"),
                                                  snapshot.Version("materials")};
            scene.Release(snapshot);
            return read;
        };
        const std::array<uint64_t, 3> first = versions();
        scene.Publish().Wait();
        EXPECT_EQ(versions(), first);

        ball.Set(Xformable::kTranslate, Double3{1, 0, 0});
        scene.Publish().Wait();
        const std::array<uint64_t, 3> moved = versions();
        EXPECT_GT(moved[0], first[0]);
        EXPECT_GT(moved[1], first[1]);
        EXPECT_EQ(moved[2], first[2]);

        const Snapshot snapshot = scene.Acquire(Stream::Host());
        EXPECT_THROW((void)snapshot.Version("nothing"), Error);
        scene.Release(snapshot);
        EXPECT_EQ(Snapshot{}.Version(), 0u);
    });
}

}
}
