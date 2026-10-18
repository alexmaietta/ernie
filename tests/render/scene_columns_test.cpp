// SPDX-License-Identifier: BSD-3-Clause
/// @file scene_columns_test.cpp
/// @brief Tests of scene columns on the device: they follow their buckets' swap-removes and
/// additions without losing what the device wrote, and take what the host authors.
#include "ernie/render/scene_columns.h"
#include "ernie/scene/schemas/geom.h"

#include "tests/device/common.h"

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

namespace ernie {
namespace {

using test::ExpectClean;

std::vector<float> Read(DeviceContext& context, const DeviceColumn& column, const Ticket& after) {
    Readback readback(context, context.Main(), column.Storage(), column.Range().offset,
                      column.Bytes(), std::span(&after, 1));
    std::vector<float> values(column.Bytes() / sizeof(float));
    std::memcpy(values.data(), readback.Bytes().data(), column.Bytes());
    return values;
}

TEST(SceneColumns, FollowTheirBucketKeepingWhatTheDeviceWrote) {
    ERNIE_CONTEXT(context);
    Scene scene;
    std::vector<PrimId> balls;
    for (int i = 0; i < 5; ++i) {
        balls.push_back(scene.Define("/Ball_" + std::to_string(i), "Sphere"));
        scene.Set(balls.back(), Sphere::kRadius, 1.0 + i);
    }
    const BucketId bucket = scene.Store().PrimBucket(balls.front());
    Arenas arenas(*context, MemoryUse::Device, "scene", 1 << 16);
    SceneColumns columns(*context, arenas, scene);
    const DeviceColumn& radii = columns.Bind(bucket, Sphere::kRadius);
    EXPECT_EQ(&columns.Bind(bucket, Sphere::kRadius), &radii) << "bound once";
    EXPECT_EQ(columns.Find(bucket, Sphere::kRadius), &radii);
    EXPECT_EQ(Read(*context, radii, columns.Sync()), (std::vector<float>{1, 2, 3, 4, 5}));

    // The device takes over the values, as a simulation writing in place would.
    Uploader device(*context, context->Main());
    const std::vector<float> simulated{100, 101, 102, 103, 104};
    device.Write(radii.Storage(), radii.Range().offset, std::span<const float>(simulated));
    const Ticket simulatedTicket = device.Flush();

    // Ball_1 goes (Ball_4 moves into its row), Ball_5 comes, and Ball_0 is authored anew.
    scene.Remove(Path("/Ball_1"));
    scene.Set(scene.Define("/Ball_5", "Sphere"), Sphere::kRadius, 9.0);
    scene.Set(balls[0], Sphere::kRadius, 50.0);
    const Transfers before = context->Transfers();
    const Ticket synced = columns.Sync(std::span(&simulatedTicket, 1));
    EXPECT_EQ(context->Transfers().uploaded - before.uploaded, 2 * sizeof(float))
        << "only the authored and the added rows go up";
    EXPECT_EQ(Read(*context, radii, synced), (std::vector<float>{50, 104, 102, 103, 9}));

    EXPECT_THROW(columns.Bind(bucket, "nonsense"), std::invalid_argument);
    context->WaitIdle();
    ExpectClean(*context);
}

TEST(SceneColumns, ConvertTheScenesDoublesToFloats) {
    EXPECT_EQ(ToDeviceElement(ColumnType::Quat)->shape, (std::vector<std::int64_t>{4}));
    EXPECT_EQ(ToDeviceElement(ColumnType::Matrix)->shape, (std::vector<std::int64_t>{4, 4}));
    EXPECT_EQ(ToDeviceElement(ColumnType::Int)->type, DType::Int32);
    EXPECT_FALSE(ToDeviceElement(ColumnType::TokenArray));
}

}  // namespace
}  // namespace ernie
