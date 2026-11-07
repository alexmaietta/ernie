// SPDX-License-Identifier: BSD-3-Clause
#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/device/device.h"
#include "ernie/device/queue.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/scene_impl.h"
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
    device.GetQueue().Read(PinnedTables::From(snapshot).header->Slice(0, sizeof(tables)), &tables);
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
        EXPECT_EQ(std::memcmp(&header, &pinned.tables, sizeof(header)), 0);
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
        EXPECT_EQ(PinnedTables::From(relaid).tables.worldCount, 1u);
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
        const uint64_t layout = PinnedTables::From(held).tables.instances.address;
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
        std::optional<MeshShape> shape = SceneImpl::Get(scene).GetGeometry().Find(quad.Index());
        ASSERT_TRUE(shape && shape->blas);
        const bvh::Blas* blas = shape->blas.Get();
        shape.reset();
        const Snapshot held = scene.Acquire(Stream::Host());

        scene.Remove(quad);
        scene.Publish().Wait();
        EXPECT_FALSE(SceneImpl::Get(scene).GetGeometry().Find(quad.Index()));
        bool pinned = false;
        for (const Ref<const RefCounted>& item : PinnedTables::From(held).held) {
            pinned = pinned || item.Get() == blas;
        }
        ASSERT_TRUE(pinned);
        EXPECT_EQ(blas->RefCount(), 1u);
        scene.Release(held);
    });
}

}
}
