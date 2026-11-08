// SPDX-License-Identifier: BSD-3-Clause
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/device/device.h"
#include "ernie/device/queue.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/scene_impl.h"
#include "ernie/scene/tables/table.h"
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
        const Prims& prims = SceneImpl::Get(*scene).GetStructure().GetPrims();
        const uint32_t cameraRow = prims.Row(camera.Index(), SchemaId::Camera);

        Snapshot snapshot = scene->Acquire(Stream::Host());
        const RenderTables& tables = PinnedTables::From(snapshot).Header<RenderTables>();
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
            device, snapshot, PinnedTables::From(snapshot).Header<RenderTables>().cameras.address,
            cameraRow);
        EXPECT_FLOAT_EQ(after.focalLength, 25);
        EXPECT_FLOAT_EQ(after.apertureRadius, 25 * 0.001f / 4);
        scene->Release(snapshot);
    });
}

TEST(SensorsTest, DetectorRecordsAverageQeOverTheCamerasBands) {
    test::ForEachDevice([](device::Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        const Prim automotive =
            scene->Define<Camera>("/Automotive", {.detectorPattern = DetectorPattern::Rccc,
                                                  .detectorSpectralRange = {400, 800},
                                                  .detectorQeRed = {{400, 0}, {800, 1}},
                                                  .detectorQeClear = {{500, 0.5f}, {600, 0.5f}},
                                                  .detectorTemperature = 300});
        const Prim generic = scene->Define<Camera>("/Generic", {});
        const Prim spectral =
            scene->Define<Camera>("/Spectral", {.detectorPattern = DetectorPattern::Bands});
        scene->Publish().Wait();
        const Prims& prims = SceneImpl::Get(*scene).GetStructure().GetPrims();
        const auto detector = [&](const Prim& camera) {
            const Snapshot snapshot = scene->Acquire(Stream::Host());
            const DetectorRecord record = ReadRecord<DetectorRecord>(
                device, snapshot,
                PinnedTables::From(snapshot).Header<RenderTables>().detectors.address,
                prims.Row(camera.Index(), SchemaId::DetectorAPI));
            scene->Release(snapshot);
            return record;
        };
        const auto band = [](const DetectorRecord& record, uint32_t channel, uint32_t index) {
            const Float4& four = record.qe[2 * channel + index / 4];
            const float values[] = {four.x, four.y, four.z, four.w};
            return values[index % 4];
        };

        const DetectorRecord rccc = detector(automotive);
        EXPECT_EQ(rccc.channels, 2u);
        EXPECT_FLOAT_EQ(rccc.bands.first, 400);
        EXPECT_FLOAT_EQ(rccc.bands.width, 50);
        EXPECT_FLOAT_EQ(rccc.temperature, 300);
        for (uint32_t index = 0; index < kDetectorBands; ++index) {
            EXPECT_NEAR(band(rccc, 0, index), (50.0f * index + 25) / 400, 1e-5f) << index;
            EXPECT_NEAR(band(rccc, 1, index), index == 2 || index == 3 ? 0.5f : 0, 1e-6f) << index;
            EXPECT_EQ(band(rccc, 2, index), 0) << index;
        }

        const DetectorRecord rgb = detector(generic);
        EXPECT_EQ(rgb.channels, 3u);
        const uint32_t peaks[] = {4, 3, 1};
        for (uint32_t channel = 0; channel < 3; ++channel) {
            uint32_t peak = 0;
            for (uint32_t index = 1; index < kDetectorBands; ++index) {
                peak = band(rgb, channel, index) > band(rgb, channel, peak) ? index : peak;
            }
            EXPECT_EQ(peak, peaks[channel]) << channel;
        }

        EXPECT_EQ(detector(spectral).channels, kDetectorBands);
        EXPECT_GT(band(detector(spectral), 0, 7), 0);
        automotive.Set(Camera::kDetectorPattern, DetectorPattern::Mono);
        scene->Publish().Wait();
        EXPECT_EQ(detector(automotive).channels, 1u);
        EXPECT_NEAR(band(detector(automotive), 0, 2), 0.5f, 1e-6f);
    });
}

TEST(SensorsTest, LightShapesKeepAuthoredSizesForGuides) {
    test::ForEachDevice([](device::Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        const Prim rect =
            scene->Define<RectLight>("/Panel", {.scale = {2, 2, 2}, .width = 3, .height = 4});
        const Prim bulb = scene->Define<SphereLight>("/Bulb", {.radius = 0.5f});
        scene->Publish().Wait();
        const Prims& prims = SceneImpl::Get(*scene).GetStructure().GetPrims();
        const auto shape = [&](const Prim& light) {
            const Snapshot snapshot = scene->Acquire(Stream::Host());
            const LightShapeRecord record = ReadRecord<LightShapeRecord>(
                device, snapshot,
                PinnedTables::From(snapshot).Header<RenderTables>().lightShapes.address,
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
