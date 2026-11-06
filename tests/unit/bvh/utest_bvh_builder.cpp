// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/bvh/builder.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/bvh/common.h"
#include "ernie/bvh/inspect.h"
#include "ernie/bvh/software_builder.h"
#include "ernie/device/queue.h"
#include "support/buffers.h"
#include "support/devices.h"
#include "support/geometry.h"

namespace ernie::bvh {
namespace {

using device::Buffer;
using device::BufferRange;
using device::CommandEncoder;
using device::Device;
using device::Traversal;
using test::ForEachDevice;
using test::Upload;

const std::vector<uint32_t> kCapacities = {4, 0, 30, 7};
const std::vector<uint32_t> kCounts = {4, 0, 21, 1};

void Submit(Device& device, const std::function<void(CommandEncoder&)>& record) {
    CommandEncoder encoder = device.GetQueue().CreateEncoder();
    record(encoder);
    device.GetQueue().Submit(encoder.Finish());
}

std::vector<Triangle> Moved(std::vector<Triangle> mesh, float offset) {
    for (Triangle& triangle : mesh) {
        for (float* corner : {triangle.a, triangle.b, triangle.c}) {
            corner[0] += offset;
        }
    }
    return mesh;
}

struct Meshes {
    std::vector<Triangle> soup = test::Soup(200, 1);
    std::vector<BvhBox> boxes = test::Boxes(100, 2);
    Ref<Buffer> soupBuffer;
    Ref<Buffer> boxBuffer;
    Ref<Buffer> gridBuffer;
    std::vector<Ref<Blas>> blases;

    Meshes(Device& device, Builder& builder) {
        soupBuffer = Upload(device, soup);
        boxBuffer = Upload(device, boxes);
        gridBuffer = Upload(device, test::Grid(4));
        blases = {
            builder.CreateBlas(device::AccelTriangles{
                .vertices = *soupBuffer, .vertexCount = 600, .count = 200, .refittable = true}),
            builder.CreateBlas(
                device::AccelBoxes{.boxes = *boxBuffer, .count = 100, .refittable = true}),
            builder.CreateBlas(
                device::AccelTriangles{.vertices = *gridBuffer, .vertexCount = 96, .count = 32})};
        Submit(device, [&](CommandEncoder& encoder) { builder.Build(encoder, blases); });
    }
};

struct World {
    std::vector<InstanceRecord> records;
    std::vector<BvhBox> boxes;
};

std::vector<InstanceRecord> Place(std::span<const Ref<Blas>> blases, std::span<const BvhBox> bounds,
                                  std::vector<World>& worlds) {
    std::vector<InstanceRecord> all;
    for (uint32_t w = 0; w < kCapacities.size(); ++w) {
        test::Instances instances = test::Scatter(kCapacities[w], w + 3);
        World world;
        for (size_t i = 0; i < instances.records.size(); ++i) {
            instances.records[i].blas = blases[i % blases.size()]->Handle();
            world.boxes.push_back(BvhInstanceBox(instances.records[i], bounds[i % bounds.size()]));
        }
        world.records = instances.records;
        all.insert(all.end(), world.records.begin(), world.records.end());
        worlds.push_back(std::move(world));
    }
    return all;
}

std::vector<std::byte> SoftwareBytes(Device& device, const Blas& blas) {
    return test::ReadBytes(device, static_cast<const SoftwareBlas&>(blas).Storage());
}

void ExpectWorld(Device& device, const Tlas& tlas, const World& world, uint32_t w, uint32_t count) {
    SCOPED_TRACE(testing::Message() << "world " << w);
    const std::vector<std::byte> bytes =
        test::ReadBytes(device, static_cast<const SoftwareTlas&>(tlas).World(w));
    const Inspection found = Inspect(bytes, std::span(world.boxes).first(count));
    ASSERT_TRUE(found.errors.empty()) << found.errors.front();
    EXPECT_EQ(found.records, count);
}

TEST(BuilderTest, SharedHelpersLayOutWorlds) {
    EXPECT_EQ(Total(kCapacities), 41u);
    EXPECT_EQ(Offsets(kCapacities), (std::vector<uint32_t>{0, 4, 4, 34, 41}));
    EXPECT_EQ(StorageBytes(0, kBvhTriangles), sizeof(BvhHeader) + sizeof(BvhNode));
    EXPECT_EQ(StorageBytes(10, kBvhInstances),
              sizeof(BvhHeader) + 10 * sizeof(BvhNode) + 10 * sizeof(BvhInstance));
    EXPECT_STREQ(FormatName(Traversal::Hardware), "hardware");
    EXPECT_STREQ(FormatName(Traversal::Software), "software");
}

TEST(BuilderTest, CreatesTheDevicesFormatAndRefusesWhatItCannotTrace) {
    ForEachDevice([](Device& device) {
        EXPECT_EQ(Builder::Create(device)->Format(), device.GetTraversal());
        EXPECT_EQ(Builder::Create(device, Traversal::Software)->Format(), Traversal::Software);
        if (device.GetTraversal() != Traversal::Hardware) {
            EXPECT_THROW(Builder::Create(device, Traversal::Hardware), UnsupportedError);
        }
    });
}

TEST(BuilderTest, SoftwareBlasesBuildAndRefitInPlace) {
    ForEachDevice([](Device& device) {
        const std::unique_ptr<Builder> builder = Builder::Create(device, Traversal::Software);
        builder->Prepare();
        Meshes meshes(device, *builder);
        for (size_t b = 0; b < meshes.blases.size(); ++b) {
            SCOPED_TRACE(b);
            const Inspection found = Inspect(SoftwareBytes(device, *meshes.blases[b]));
            ASSERT_TRUE(found.errors.empty()) << found.errors.front();
        }

        const std::vector<Triangle> moved = Moved(meshes.soup, 5);
        std::vector<BvhBox> boxes = meshes.boxes;
        for (BvhBox& box : boxes) {
            box.lower[0] += 5;
            box.upper[0] += 5;
        }
        device.GetQueue().Write(*meshes.soupBuffer, moved.data());
        device.GetQueue().Write(*meshes.boxBuffer, boxes.data());
        const std::vector<Ref<Blas>> refitted = {meshes.blases[0], meshes.blases[1]};
        Submit(device, [&](CommandEncoder& encoder) { builder->Refit(encoder, refitted); });
        for (const Ref<Blas>& blas : refitted) {
            const std::vector<std::byte> bytes = SoftwareBytes(device, *blas);
            const Inspection found = Inspect(bytes);
            ASSERT_TRUE(found.errors.empty()) << found.errors.front();
            EXPECT_GT(test::Bounds(bytes).lower[0], 4.9f);
        }

        CommandEncoder encoder = device.GetQueue().CreateEncoder();
        EXPECT_THROW(builder->Refit(encoder, std::span(meshes.blases).last(1)), Error);
        builder->Build(encoder, std::span<const Ref<Blas>>());
        builder->Refit(encoder, std::span<const Ref<Blas>>());
    });
}

TEST(BuilderTest, SoftwareTlasesBuildEveryWorldOrSomeAndRefit) {
    ForEachDevice([](Device& device) {
        const std::unique_ptr<Builder> builder = Builder::Create(device, Traversal::Software);
        Meshes meshes(device, *builder);
        std::vector<BvhBox> bounds;
        for (const Ref<Blas>& blas : meshes.blases) {
            bounds.push_back(test::Bounds(SoftwareBytes(device, *blas)));
        }
        std::vector<World> worlds;
        std::vector<InstanceRecord> all = Place(meshes.blases, bounds, worlds);
        const Ref<Buffer> instances = Upload(device, all);
        const Ref<Buffer> counts = Upload(device, kCounts);
        const Ref<Tlas> tlas = builder->CreateTlas(kCapacities);
        ASSERT_EQ(tlas->Count(), 4u);
        EXPECT_EQ(tlas->Handles().Bytes(), 4 * sizeof(uint64_t));

        Submit(device, [&](CommandEncoder& e) { builder->Build(e, *tlas, *instances, *counts); });
        for (uint32_t w = 0; w < tlas->Count(); ++w) {
            ExpectWorld(device, *tlas, worlds[w], w, kCounts[w]);
        }

        const std::vector<uint32_t> some = {2, 0};
        Submit(device, [&](CommandEncoder& e) { builder->Build(e, *tlas, *instances, {}, some); });
        ExpectWorld(device, *tlas, worlds[2], 2, kCapacities[2]);
        ExpectWorld(device, *tlas, worlds[0], 0, kCapacities[0]);
        Submit(device,
               [&](CommandEncoder& e) { builder->Build(e, *tlas, *instances, *counts, some); });
        ExpectWorld(device, *tlas, worlds[2], 2, kCounts[2]);

        for (uint32_t i = 34; i < 41; ++i) {
            all[i].transform[3] += 20;
            worlds[3].boxes[i - 34] = BvhInstanceBox(all[i], bounds[(i - 34) % bounds.size()]);
        }
        device.GetQueue().Write(*instances, all.data());
        Submit(device, [&](CommandEncoder& e) {
            builder->Build(e, *tlas, *instances, {}, std::vector<uint32_t>{3});
            builder->Refit(e, *tlas, *instances, {}, std::vector<uint32_t>{3});
        });
        ExpectWorld(device, *tlas, worlds[3], 3, kCapacities[3]);

        CommandEncoder encoder = device.GetQueue().CreateEncoder();
        EXPECT_THROW(builder->Build(encoder, *tlas, *instances, {}, std::vector<uint32_t>{4}),
                     Error);
        EXPECT_THROW(builder->Build(encoder, *tlas, *instances, {}, std::vector<uint32_t>{1, 1}),
                     Error);
        EXPECT_THROW(builder->Build(encoder, *tlas, instances->Slice(0, 64)), Error);
        EXPECT_THROW(tlas->Handle(4), Error);
        EXPECT_THROW(builder->CreateTlas({}), Error);
    });
}

TEST(BuilderTest, HardwareStructuresBuildAndRefit) {
    ForEachDevice([](Device& device) {
        if (device.GetTraversal() != Traversal::Hardware) {
            return;
        }
        const std::unique_ptr<Builder> builder = Builder::Create(device, Traversal::Hardware);
        const std::unique_ptr<Builder> software = Builder::Create(device, Traversal::Software);
        builder->Prepare();
        Meshes meshes(device, *builder);
        Submit(device, [&](CommandEncoder& encoder) {
            builder->Refit(encoder, std::span(meshes.blases).first(2));
        });
        for (const Ref<Blas>& blas : meshes.blases) {
            EXPECT_NE(blas->Handle(), 0u);
        }

        std::vector<World> worlds;
        const std::vector<BvhBox> bounds = {test::Boxes(1, 1)[0]};
        const std::vector<InstanceRecord> all = Place(meshes.blases, bounds, worlds);
        const Ref<Buffer> instances = Upload(device, all);
        const Ref<Buffer> counts = Upload(device, kCounts);
        const Ref<Tlas> tlas = builder->CreateTlas(kCapacities);
        Submit(device, [&](CommandEncoder& e) {
            builder->Build(e, *tlas, *instances, *counts);
            builder->Build(e, *tlas, *instances);
            builder->Build(e, *tlas, *instances, {}, std::vector<uint32_t>{2, 1});
            builder->Refit(e, *tlas, *instances, {}, std::vector<uint32_t>{2});
        });
        for (uint32_t w = 0; w < tlas->Count(); ++w) {
            EXPECT_NE(tlas->Handle(w), 0u);
        }

        const Meshes other(device, *software);
        CommandEncoder encoder = device.GetQueue().CreateEncoder();
        EXPECT_THROW(builder->Refit(encoder, std::span(meshes.blases).last(1)), Error);
        EXPECT_THROW(builder->Build(encoder, other.blases), Error);
        EXPECT_THROW(software->Build(encoder, meshes.blases), Error);
        EXPECT_THROW(software->Build(encoder, *tlas, *instances), Error);
    });
}

}
}
