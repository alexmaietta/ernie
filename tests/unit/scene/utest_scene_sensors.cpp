// SPDX-License-Identifier: BSD-3-Clause
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/device/device.h"
#include "ernie/device/queue.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/scene_impl.h"
#include "support/devices.h"

namespace ernie {
namespace {

template <class T>
T ReadRecord(device::Device& device, const Snapshot& snapshot, uint64_t table, uint32_t row) {
    for (const Ref<const RefCounted>& held : PinnedTables::From(snapshot).held) {
        auto* buffer = const_cast<device::Buffer*>(dynamic_cast<const device::Buffer*>(held.Get()));
        if (buffer != nullptr && buffer->Address() == table) {
            T record{};
            device.GetQueue().Read(buffer->Slice(uint64_t(row) * sizeof(T), sizeof(T)), &record);
            return record;
        }
    }
    ADD_FAILURE() << "the snapshot doesn't hold the table";
    return {};
}

TEST(SensorsTest, RecordsHoldResolvedParametersAndFollowEdits) {
    test::ForEachDevice([](device::Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        const Prim camera = scene->Define<Camera>("/Camera", {.focalLength = 50, .fStop = 2});
        const Prim lidar = scene->Define<Lidar>("/Lidar", {.beamSamples = 10});
        scene->Define<Radar>("/Radar", {.fieldOfView = {90, 30}});
        scene->Publish().Wait();
        const Prims& prims = SceneImpl::Get(*scene).GetCatalog().GetPrims();
        const uint32_t cameraRow = prims.Row(camera.Index(), SchemaId::Camera);

        Snapshot snapshot = scene->Acquire(Stream::Host());
        const RenderTables& tables = PinnedTables::From(snapshot).tables;
        const CameraRecord before =
            ReadRecord<CameraRecord>(device, snapshot, tables.cameras.address, cameraRow);
        EXPECT_FLOAT_EQ(before.focalLength, 50);
        EXPECT_FLOAT_EQ(before.apertureRadius, 50 * 0.001f / 4);
        const LidarRecord beams = ReadRecord<LidarRecord>(
            device, snapshot, tables.lidars.address, prims.Row(lidar.Index(), SchemaId::Lidar));
        EXPECT_EQ(beams.samples, 10u);
        EXPECT_EQ(beams.latticeStep, 7u);
        const RadarRecord radar =
            ReadRecord<RadarRecord>(device, snapshot, tables.radars.address, 0);
        EXPECT_NEAR(radar.fieldOfView.x, 1.5707964f, 1e-6f);
        scene->Release(snapshot);

        camera.Set(Camera::kFocalLength, 25.0f);
        scene->Publish().Wait();
        snapshot = scene->Acquire(Stream::Host());
        const CameraRecord after = ReadRecord<CameraRecord>(
            device, snapshot, PinnedTables::From(snapshot).tables.cameras.address, cameraRow);
        EXPECT_FLOAT_EQ(after.focalLength, 25);
        EXPECT_FLOAT_EQ(after.apertureRadius, 25 * 0.001f / 4);
        scene->Release(snapshot);
    });
}

TEST(SensorsTest, LightShapesKeepAuthoredSizesForGuides) {
    test::ForEachDevice([](device::Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        const Prim rect =
            scene->Define<RectLight>("/Panel", {.scale = {2, 2, 2}, .width = 3, .height = 4});
        const Prim bulb = scene->Define<SphereLight>("/Bulb", {.radius = 0.5f});
        scene->Publish().Wait();
        const Prims& prims = SceneImpl::Get(*scene).GetCatalog().GetPrims();
        const auto shape = [&](const Prim& light) {
            const Snapshot snapshot = scene->Acquire(Stream::Host());
            const LightShapeRecord record = ReadRecord<LightShapeRecord>(
                device, snapshot, PinnedTables::From(snapshot).tables.lightShapes.address,
                prims.Row(light.Index(), SchemaId::LightAPI));
            scene->Release(snapshot);
            return record;
        };
        EXPECT_FLOAT_EQ(shape(rect).width, 3);
        EXPECT_FLOAT_EQ(shape(rect).height, 4);
        EXPECT_FLOAT_EQ(shape(bulb).radius, 0.5f);
        bulb.Set(SphereLight::kRadius, 2.0f);
        scene->Publish().Wait();
        EXPECT_FLOAT_EQ(shape(bulb).radius, 2);
    });
}

}
}
