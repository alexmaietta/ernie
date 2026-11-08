// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/render/ray_cast/ray_cast.h"

#include <cmath>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/device/queue.h"
#include "ernie/scene/records/points.h"
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

TEST(RayCastTest, HidingAPrimOrItsParentRemovesItFromCasts) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        scene->Define<Xform>("/Group");
        scene->Define<Sphere>("/Group/Ball", {.translate = {3, 0, 0}, .radius = 1});
        RayCaster caster(scene);
        Rays rays;
        rays.origins = Floats(device, {0, 0, 0}, 3);
        rays.directions = Floats(device, {1, 0, 0}, 3);
        const auto hit = [&] { return caster.Cast(rays).hit.ToHost<uint32_t>().front(); };
        EXPECT_EQ(hit(), 1u);
        scene->Select("/Group").Get<Visibility>("visibility").Set(Visibility::Invisible);
        EXPECT_EQ(hit(), 0u);
        scene->Select("/Group").Get<Visibility>("visibility").Set(Visibility::Inherited);
        EXPECT_EQ(hit(), 1u);
        scene->Select("/Group/Ball").Get<Purpose>("purpose").Set(Purpose::Guide);
        EXPECT_EQ(hit(), 0u);
    });
}

TEST(RayCastTest, ReshapedMeshesHitTheirNewPoints) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        const Prim mesh = scene->Define<Mesh>(
            "/Mesh", {.points = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}},
                      .faceVertexIndices = {0, 1, 2, 3},
                      .faceVertexCounts = {4}});
        RayCaster caster(scene);
        Rays rays;
        rays.origins = Floats(device, {0.25f, 0.25f, 5}, 3);
        rays.directions = Floats(device, {0, 0, -1}, 3);
        const RayCastDesc options{.outputs = RayOutput::Distance | RayOutput::Position};
        EXPECT_NEAR(caster.Cast(rays, options).distance.ToHost<float>().front(), 5, 1e-4);

        mesh.Set(Mesh::kPoints,
                 std::vector<Float3>{{-2, -2, 2}, {2, -2, 2}, {0, 2, 2}, {9, 9, 9}, {8, 8, 8}});
        mesh.Set(Mesh::kFaceVertexCounts, std::vector<int32_t>{3});
        mesh.Set(Mesh::kFaceVertexIndices, std::vector<int32_t>{0, 1, 2});
        const RayHits hits = caster.Cast(rays, options);
        EXPECT_NEAR(hits.distance.ToHost<float>().front(), 3, 1e-4);
        EXPECT_NEAR(hits.position.ToHost<float>()[2], 2, 1e-4);
    });
}

TEST(RayCastTest, HitsEachPointAsASphereAsWideAsItsWidth) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        scene->Define<Points>(
            "/Spread", {.points = {{3, 0, 0}, {3, 5, 0}, {3, 10, 0}}, .widths = {2, 1, 0.5f}});
        scene->Define<Points>(
            "/Shared", {.translate = {0, 15, 0}, .points = {{3, 0, 0}, {3, 2, 0}}, .widths = {1}});
        scene->Define<Points>("/Default", {.translate = {0, 20, 0}, .points = {{3, 0, 0}}});
        RayCaster caster(scene);
        Rays rays;
        rays.origins =
            Floats(device, {0, 0, 0, 0, 5, 0, 0, 10, 0, 0, 2.5f, 0, 0, 17, 0, 0, 20, 0}, 3);
        rays.directions = Floats(device, {1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0}, 3);
        const RayHits hits =
            caster.Cast(rays, {.outputs = RayOutput::Distance | RayOutput::Prim |
                                          RayOutput::Normal | RayOutput::Position});
        const std::vector<uint32_t> hit = hits.hit.ToHost<uint32_t>();
        const std::vector<float> distance = hits.distance.ToHost<float>();
        const std::vector<uint32_t> prim = hits.prim.ToHost<uint32_t>();
        const std::vector<float> normal = hits.normal.ToHost<float>();
        const std::vector<float> position = hits.position.ToHost<float>();
        EXPECT_EQ(hit, (std::vector<uint32_t>{1, 1, 1, 0, 1, 1}));
        EXPECT_NEAR(distance[0], 2, 1e-4);
        EXPECT_NEAR(distance[1], 2.5, 1e-4);
        EXPECT_NEAR(distance[2], 2.75, 1e-4);
        EXPECT_NEAR(distance[4], 2.5, 1e-4);
        EXPECT_NEAR(distance[5], 3 - kDefaultPointWidth / 2, 1e-4);
        EXPECT_EQ(prim[0], prim[1]);
        EXPECT_NE(prim[0], prim[4]);
        EXPECT_NEAR(normal[3], -1, 1e-4);
        EXPECT_NEAR(position[3], 2.5, 1e-4);
        EXPECT_NEAR(position[4], 5, 1e-4);
    });
}

TEST(RayCastTest, MovedAndRecountedPointsHitWhereTheyAre) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        const Prim cloud =
            scene->Define<Points>("/Cloud", {.points = {{3, 0, 0}, {3, 4, 0}}, .widths = {1}});
        RayCaster caster(scene);
        Rays rays;
        rays.origins = Floats(device, {0, 0, 0, 0, 4, 0, 0, 8, 0}, 3);
        rays.directions = Floats(device, {1, 0, 0, 1, 0, 0, 1, 0, 0}, 3);
        const auto cast = [&] { return caster.Cast(rays).distance.ToHost<float>(); };
        std::vector<float> distance = cast();
        EXPECT_NEAR(distance[0], 2.5, 1e-4);
        EXPECT_NEAR(distance[1], 2.5, 1e-4);
        EXPECT_TRUE(std::isinf(distance[2]));

        cloud.Set(Points::kPoints, std::vector<Float3>{{6, 0, 0}, {3, 8, 0}});
        distance = cast();
        EXPECT_NEAR(distance[0], 5.5, 1e-4);
        EXPECT_TRUE(std::isinf(distance[1]));
        EXPECT_NEAR(distance[2], 2.5, 1e-4);

        cloud.Set(Points::kPoints, std::vector<Float3>{{3, 0, 0}, {4, 4, 0}, {5, 8, 0}});
        cloud.Set(Points::kWidths, std::vector<float>{2, 1, 3});
        distance = cast();
        EXPECT_NEAR(distance[0], 2, 1e-4);
        EXPECT_NEAR(distance[1], 3.5, 1e-4);
        EXPECT_NEAR(distance[2], 3.5, 1e-4);

        cloud.Set(Points::kPoints, std::vector<Float3>{});
        distance = cast();
        EXPECT_TRUE(std::isinf(distance[0]) && std::isinf(distance[1]) && std::isinf(distance[2]));
    });
}

TEST(RayCastTest, InstancedMeshCopiesHitAtTheirOwnPlaces) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        scene->Define<PointInstancer>("/Inst", {.prototypes = {"/Inst/Protos/Tile"},
                                                .protoIndices = {0, 0},
                                                .positions = {{0, 0, 0}, {10, 0, 3}}});
        scene->Define<Scope>("/Inst/Protos");
        scene->Define<Mesh>("/Inst/Protos/Tile",
                            {.points = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}},
                             .faceVertexIndices = {0, 1, 2, 3},
                             .faceVertexCounts = {4}});
        RayCaster caster(scene);
        Rays rays;
        rays.origins = Floats(device, {0.5f, 0.5f, 5, 10.5f, -0.5f, 5}, 3);
        rays.directions = Floats(device, {0, 0, -1, 0, 0, -1}, 3);
        const RayHits hits = caster.Cast(
            rays, {.outputs = RayOutput::Distance | RayOutput::Position | RayOutput::Normal});
        const std::vector<float> distance = hits.distance.ToHost<float>();
        const std::vector<float> position = hits.position.ToHost<float>();
        const std::vector<float> normal = hits.normal.ToHost<float>();
        EXPECT_NEAR(distance[0], 5, 1e-4);
        EXPECT_NEAR(distance[1], 2, 1e-4);
        EXPECT_NEAR(position[3], 10.5f, 1e-4);
        EXPECT_NEAR(position[5], 3, 1e-4);
        EXPECT_NEAR(normal[5], 1, 1e-4);
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
