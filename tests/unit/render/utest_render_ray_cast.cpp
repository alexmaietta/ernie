// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/render/ray_cast/ray_cast.h"

#include <cmath>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/device/queue.h"
#include "ernie/scene/scene.h"
#include "ernie/util/error.h"
#include "support/devices.h"

namespace ernie {
namespace {

using device::Device;
using device::DType;
using device::Tensor;
using test::ForEachDevice;

template <class T>
Tensor Upload(Device& device, DType type, const std::vector<T>& values,
              std::vector<int64_t> shape) {
    Tensor tensor = device.CreateTensor(type, std::move(shape), "rays");
    device.GetQueue().Write(tensor.GetBuffer().Slice(tensor.Offset(), values.size() * sizeof(T)),
                            values.data());
    return tensor;
}

Tensor Floats(Device& device, const std::vector<float>& values, int64_t columns) {
    return Upload(device, DType::Float32, values, {int64_t(values.size()) / columns, columns});
}

Ref<Scene> TwoBalls(Device& device) {
    const Ref<Scene> scene = Scene::Create(device);
    scene->Define<Sphere>("/Near", {.translate = {3, 0, 0}, .radius = 1});
    scene->Define<Sphere>("/Far", {.translate = {6, 0, 0}, .radius = 0.5});
    return scene;
}

TEST(RayCastTest, FindsTheClosestHitAndItsSurface) {
    ForEachDevice([](Device& device) {
        RayCaster caster(TwoBalls(device));
        Rays rays;
        rays.origins = Floats(device, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 3);
        rays.directions = Floats(device, {1, 0, 0, -1, 0, 0, 0, 1, 0, 1, 0, 0}, 3);
        rays.tMins = Upload(device, DType::Float32, std::vector<float>{0, 0, 0, 4.5f}, {4});
        const RayHits hits =
            caster.Cast(rays, {.outputs = RayOutput::Distance | RayOutput::Prim |
                                          RayOutput::Normal | RayOutput::Position});
        hits.done.Wait();

        const std::vector<uint32_t> hit = hits.hit.ToHost<uint32_t>();
        const std::vector<float> distance = hits.distance.ToHost<float>();
        const std::vector<uint32_t> prim = hits.prim.ToHost<uint32_t>();
        const std::vector<float> normal = hits.normal.ToHost<float>();
        const std::vector<float> position = hits.position.ToHost<float>();
        EXPECT_EQ(hit, (std::vector<uint32_t>{1, 0, 0, 1}));
        EXPECT_NEAR(distance[0], 2, 1e-4);
        EXPECT_NEAR(distance[3], 5.5, 1e-4);
        EXPECT_NE(prim[0], prim[3]);
        EXPECT_NEAR(normal[0], -1, 1e-4);
        EXPECT_NEAR(position[0], 2, 1e-4);
        EXPECT_NEAR(normal[9], -1, 1e-4);
        EXPECT_NEAR(position[9], 5.5, 1e-4);
    });
}

TEST(RayCastTest, HitsQuadricsPreciselyFromFarAway) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        scene->Define<Cylinder>("/Cylinder", {.translate = {0, 0, 0}});
        scene->Define<Cone>("/Cone", {.translate = {0, 10, 0}});
        scene->Define<Capsule>("/Capsule", {.translate = {0, 20, 0}});
        RayCaster caster(scene);
        Rays rays;
        rays.origins = Floats(device, {-1000, 0.3f, 0, -1000, 10.3f, 0, -1000, 20.3f, 0}, 3);
        rays.directions = Floats(device, {1, 0, 0, 1, 0, 0, 1, 0, 0}, 3);
        const std::vector<float> distance = caster.Cast(rays).distance.ToHost<float>();
        EXPECT_NEAR(distance[0], 1000 - std::sqrt(0.91), 1e-3);
        EXPECT_NEAR(distance[1], 999.6, 1e-3);
        EXPECT_NEAR(distance[2], 999.6, 1e-3);
    });
}

TEST(RayCastTest, AnyHitsAndDistanceLimits) {
    ForEachDevice([](Device& device) {
        RayCaster caster(TwoBalls(device));
        Rays rays;
        rays.origins = Floats(device, {0, 0, 0, 0, 0, 0}, 3);
        rays.directions = Floats(device, {1, 0, 0, 1, 0, 0}, 3);
        rays.tMaxes = Upload(device, DType::Float32, std::vector<float>{10, 1.5f}, {2});
        EXPECT_EQ(caster.Cast(rays, {.mode = RayCastMode::Any}).hit.ToHost<uint32_t>(),
                  (std::vector<uint32_t>{1, 0}));

        rays.tMaxes = {};
        rays.maxDistance = 1.5f;
        EXPECT_EQ(caster.Cast(rays).hit.ToHost<uint32_t>(), (std::vector<uint32_t>{0, 0}));
        rays.maxDistance = INFINITY;
        rays.minDistance = 4.6f;
        const std::vector<float> beyond = caster.Cast(rays).distance.ToHost<float>();
        EXPECT_NEAR(beyond[0], 5.5, 1e-4);
    });
}

TEST(RayCastTest, CastsIntoEachRaysWorld) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        scene->Define<World>("/env_0");
        scene->Define<World>("/env_1");
        scene->Define<Sphere>("/env_0/Ball", {.translate = {3, 0, 0}, .radius = 1});
        scene->Define<Sphere>("/env_1/Ball", {.translate = {6, 0, 0}, .radius = 1});
        RayCaster caster(scene);
        Rays rays;
        rays.origins = Floats(device, {0, 0, 0, 0, 0, 0}, 3);
        rays.directions = Floats(device, {1, 0, 0, 1, 0, 0}, 3);
        rays.worlds = Upload(device, DType::UInt32, std::vector<uint32_t>{0, 1}, {2});
        const std::vector<float> distance = caster.Cast(rays).distance.ToHost<float>();
        EXPECT_NEAR(distance[0], 2, 1e-4);
        EXPECT_NEAR(distance[1], 5, 1e-4);
        EXPECT_EQ(&caster.GetScene(), scene.Get());
    });
}

TEST(RayCastTest, RejectsRaysItCannotCast) {
    ForEachDevice([](Device& device) {
        RayCaster caster(TwoBalls(device));
        Rays rays;
        EXPECT_THROW(caster.Cast(rays), Error);
        rays.origins = Floats(device, {0, 0, 0, 0, 0, 0}, 3);
        rays.directions = Floats(device, {1, 0, 0}, 3);
        EXPECT_THROW(caster.Cast(rays), Error);
        rays.directions =
            Upload(device, DType::Int32, std::vector<int32_t>{1, 0, 0, 1, 0, 0}, {2, 3});
        EXPECT_THROW(caster.Cast(rays), Error);
        rays.directions = Floats(device, {1, 0, 0, 1, 0, 0}, 3);
        rays.tMins = Upload(device, DType::Float32, std::vector<float>{0}, {1});
        EXPECT_THROW(caster.Cast(rays), Error);
        EXPECT_THROW(RayCaster(nullptr), Error);
    });
}

}
}
