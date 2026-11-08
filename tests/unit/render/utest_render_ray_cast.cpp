// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/render/ray_cast/ray_cast.h"

#include <cmath>
#include <cstdint>
#include <format>
#include <span>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/device/queue.h"
#include "ernie/scene/primvar.h"
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

const std::pair<device::Traversal, uint32_t> kPointLayouts[] = {
    {device::Traversal::Auto, 1},
    {device::Traversal::Software, 1},
    {device::Traversal::Auto, 3},
    {device::Traversal::Software, 3},
};

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

TEST(RayCastTest, AGridOfPointsIsHitOncePerPointAndMissedBetween) {
    ForEachDevice([](Device& device) {
        constexpr uint32_t kAcross = 16;
        const Ref<Scene> scene = Scene::Create(device);
        std::vector<Float3> points;
        for (uint32_t y = 0; y < kAcross; ++y) {
            for (uint32_t z = 0; z < kAcross; ++z) {
                points.push_back({3, float(y), float(z)});
            }
        }
        scene->Define<Points>("/Grid", {.points = points, .widths = {0.5f}});
        std::vector<float> origins;
        std::vector<float> directions;
        for (uint32_t y = 0; y < 2 * kAcross - 1; ++y) {
            for (uint32_t z = 0; z < 2 * kAcross - 1; ++z) {
                origins.insert(origins.end(), {0, 0.5f * float(y), 0.5f * float(z)});
                directions.insert(directions.end(), {1, 0, 0});
            }
        }
        RayCaster caster(scene);
        Rays rays;
        rays.origins = Floats(device, origins, 3);
        rays.directions = Floats(device, directions, 3);
        const std::vector<uint32_t> hit = caster.Cast(rays).hit.ToHost<uint32_t>();
        uint32_t hits = 0;
        uint32_t strays = 0;
        for (uint32_t y = 0; y < 2 * kAcross - 1; ++y) {
            for (uint32_t z = 0; z < 2 * kAcross - 1; ++z) {
                const uint32_t h = hit[y * (2 * kAcross - 1) + z];
                (y % 2 == 0 && z % 2 == 0 ? hits : strays) += h;
            }
        }
        EXPECT_EQ(hits, kAcross * kAcross);
        EXPECT_EQ(strays, 0u);
    });
}

TEST(RayCastTest, PointsWrittenThroughAMappingHitWhereTheyWereWritten) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        scene->Define<Points>("/Cloud", {.points = {{3, 0, 0}, {3, 4, 0}}, .widths = {1}});
        RayCaster caster(scene);
        Rays rays;
        rays.origins = Floats(device, {0, 0, 0, 0, 4, 0, 0, 8, 0}, 3);
        rays.directions = Floats(device, {1, 0, 0, 1, 0, 0, 1, 0, 0}, 3);
        ASSERT_NEAR(caster.Cast(rays).distance.ToHost<float>()[0], 2.5, 1e-4);

        const std::vector<Float3> simulated = {{6, 0, 0}, {3, 8, 0}};
        const Ref<device::Buffer> source =
            device.CreateBuffer({.label = "simulated points"}, std::span<const Float3>(simulated));
        Mapping mapping = scene->Select("/Cloud").GetArray<Float3>("points").Map(MapAccess::Write);
        const Tensor& values = mapping.Values();
        device::Queue& queue = device.GetQueue();
        device::CommandEncoder encoder = queue.CreateEncoder();
        encoder.Copy(source->Slice(0),
                     values.GetBuffer().Slice(values.Offset(), sizeof(Float3) * simulated.size()));
        queue.Submit(encoder.Finish());
        mapping.Close();

        const std::vector<float> distance = caster.Cast(rays).distance.ToHost<float>();
        EXPECT_NEAR(distance[0], 5.5, 1e-4);
        EXPECT_TRUE(std::isinf(distance[1]));
        EXPECT_NEAR(distance[2], 2.5, 1e-4);
    });
}

TEST(RayCastTest, RecountedPointsHitOnlyTheLivingAcrossCapacityChanges) {
    ForEachDevice([](Device& device) {
        constexpr uint32_t kRays = 96;
        for (const auto& [traversal, perBox] : kPointLayouts) {
            SCOPED_TRACE(
                std::format("traversal {} with {} points per box", int(traversal), perBox));
            const Ref<Scene> scene =
                Scene::Create(device, {.traversal = traversal, .pointsPerBox = perBox});
            const auto line = [](uint32_t count, float x, float z) {
                std::vector<Float3> points;
                for (uint32_t i = 0; i < count; ++i) {
                    points.push_back({x, float(i), z});
                }
                return points;
            };
            const Prim near =
                scene->Define<Points>("/Near", {.points = line(64, 3, 0), .widths = {0.5f}});
            const Prim far =
                scene->Define<Points>("/Far", {.points = line(32, 3, 2), .widths = {0.5f}});
            std::vector<float> origins;
            std::vector<float> directions;
            for (const float z : {0.0f, 2.0f}) {
                for (uint32_t i = 0; i < 2 * kRays; ++i) {
                    origins.insert(origins.end(), {0, 0.5f * float(i), z});
                    directions.insert(directions.end(), {1, 0, 0});
                }
            }
            RayCaster caster(scene);
            Rays rays;
            rays.origins = Floats(device, origins, 3);
            rays.directions = Floats(device, directions, 3);
            const std::pair<uint32_t, float> steps[] = {
                {64, 3}, {40, 3}, {40, 4}, {50, 4}, {64, 3}, {64, 5},
                {90, 3}, {70, 3}, {70, 4}, {20, 3}, {0, 3},  {33, 6},
            };
            for (const auto& [count, x] : steps) {
                SCOPED_TRACE(count);
                const uint32_t other = (count * 7) % kRays;
                near.Set(Points::kPoints, line(count, x, 0));
                far.Set(Points::kPoints, line(other, x + 1, 2));
                const std::vector<float> distance = caster.Cast(rays).distance.ToHost<float>();
                for (uint32_t row = 0; row < 2; ++row) {
                    const uint32_t living = row == 0 ? count : other;
                    const float expected = (row == 0 ? x : x + 1) - 0.25f;
                    uint32_t wrong = 0;
                    for (uint32_t i = 0; i < 2 * kRays; ++i) {
                        const float d = distance[row * 2 * kRays + i];
                        const bool hit = i % 2 == 0 && i / 2 < living;
                        wrong += hit ? std::abs(d - expected) > 1e-4f : !std::isinf(d);
                    }
                    EXPECT_EQ(wrong, 0u) << "row " << row;
                }
            }
        }
    });
}

TEST(RayCastTest, PointsWithNormalsAreDisksFacingThem) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        const Prim facing = scene->Define<Points>(
            "/Facing", {.points = {{3, 0, 0}, {3, 4, 0}}, .normals = {{-2, 0, 0}}, .widths = {1}});
        scene->Define<Points>(
            "/Each",
            {.points = {{3, 8, 0}, {3, 12, 0}}, .normals = {{1, 0, 0}, {0, 0, 1}}, .widths = {1}});
        RayCaster caster(scene);
        Rays rays;
        rays.origins =
            Floats(device, {0, 0, 0, 0, 0.45f, 0, 0, 0.45f, 0.45f, 0, 8, 0, 0, 12, 0, 3, -5, 0}, 3);
        rays.directions = Floats(device, {1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0}, 3);
        const auto cast = [&] {
            return caster.Cast(rays, {.outputs = RayOutput::Distance | RayOutput::Normal});
        };
        RayHits hits = cast();
        std::vector<float> distance = hits.distance.ToHost<float>();
        const std::vector<float> normal = hits.normal.ToHost<float>();
        EXPECT_NEAR(distance[0], 3, 1e-4);
        EXPECT_NEAR(distance[1], 3, 1e-4);
        EXPECT_TRUE(std::isinf(distance[2]));
        EXPECT_NEAR(distance[3], 3, 1e-4);
        EXPECT_TRUE(std::isinf(distance[4]));
        EXPECT_TRUE(std::isinf(distance[5]));
        EXPECT_NEAR(std::abs(normal[0]), 1, 1e-4);
        EXPECT_NEAR(normal[1], 0, 1e-4);

        facing.Set(Points::kNormals, std::vector<Float3>{});
        distance = cast().distance.ToHost<float>();
        EXPECT_NEAR(distance[0], 2.5, 1e-4);
        EXPECT_NEAR(distance[5], 4.5, 1e-4);
    });
}

TEST(RayCastTest, TranslucentPointsBlockRaysInProportionToTheirOpacity) {
    ForEachDevice([](Device& device) {
        constexpr uint32_t kRays = 4096;
        for (const auto& [traversal, perBox] : kPointLayouts) {
            SCOPED_TRACE(
                std::format("traversal {} with {} points per box", int(traversal), perBox));
            const Ref<Scene> scene =
                Scene::Create(device, {.traversal = traversal, .pointsPerBox = perBox});
            const Prim veil = scene->Define<Points>(
                "/Veil", {.displayOpacity = 0, .points = {{3, 0, 0}, {3, 10, 0}}, .widths = {2}});
            scene->Define<Points>("/Wall", {.points = {{6, 0, 0}, {6, 10, 0}}, .widths = {2}});
            std::vector<float> origins;
            std::vector<float> directions;
            for (uint32_t i = 0; i < kRays; ++i) {
                const float y = 0.4f * (float(i % 64) / 63 - 0.5f);
                const float z = 0.4f * (float(i / 64) / 63 - 0.5f);
                origins.insert(origins.end(), {0, y + (i % 2 == 0 ? 0.0f : 10.0f), z});
                directions.insert(directions.end(), {1, 0, 0});
            }
            RayCaster caster(scene);
            Rays rays;
            rays.origins = Floats(device, origins, 3);
            rays.directions = Floats(device, directions, 3);
            const auto blocked = [&] {
                const std::vector<float> distance = caster.Cast(rays).distance.ToHost<float>();
                std::pair<uint32_t, uint32_t> counts{0, 0};
                for (uint32_t i = 0; i < kRays; ++i) {
                    EXPECT_LT(distance[i], 6) << i;
                    (i % 2 == 0 ? counts.first : counts.second) += distance[i] < 4;
                }
                return counts;
            };
            EXPECT_EQ(blocked(), (std::pair<uint32_t, uint32_t>{0, 0}));

            veil.Set(Points::kDisplayOpacity, 1.0f);
            EXPECT_EQ(blocked(), (std::pair<uint32_t, uint32_t>{kRays / 2, kRays / 2}));

            veil.Set(Points::kDisplayOpacity, 0.5f);
            const auto [near, far] = blocked();
            EXPECT_NEAR(double(near) / (kRays / 2), 0.5, 0.06);
            EXPECT_NEAR(double(far) / (kRays / 2), 0.5, 0.06);

            veil.Set(Points::kPrimvars,
                     std::vector<Primvar>{Primvar::Create("displayOpacity", Interpolation::Vertex,
                                                          std::vector<float>{0, 1})});
            EXPECT_EQ(blocked(), (std::pair<uint32_t, uint32_t>{0, kRays / 2}));

            Mapping mapping = scene->Select("/Veil")
                                  .Get("primvars")
                                  .MapPrimvar("displayOpacity", MapAccess::Write);
            const std::vector<float> simulated = {1, 0};
            const Tensor& values = mapping.Values();
            device.GetQueue().Write(
                values.GetBuffer().Slice(values.Offset(), sizeof(float) * simulated.size()),
                simulated.data());
            mapping.Close();
            EXPECT_EQ(blocked(), (std::pair<uint32_t, uint32_t>{kRays / 2, 0}));
        }
    });
}

TEST(RayCastTest, SmallPointsFarAwayHitAtTheirEdges) {
    ForEachDevice([](Device& device) {
        for (const auto& [traversal, perBox] : kPointLayouts) {
            SCOPED_TRACE(
                std::format("traversal {} with {} points per box", int(traversal), perBox));
            const Ref<Scene> scene =
                Scene::Create(device, {.traversal = traversal, .pointsPerBox = perBox});
            scene->Define<Points>(
                "/Far",
                {.points = {{10000, 0, 0}, {10000, 0.05f, 0}, {10000, 0.1f, 0}, {10000, 0.15f, 0}},
                 .widths = {0.02f}});
            RayCaster caster(scene);
            Rays rays;
            rays.origins =
                Floats(device, {0, 0, 0, 0, 0.0095f, 0, 0, 0.0105f, 0, 0, 0.15f, 0.0095f}, 3);
            rays.directions = Floats(device, {1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0}, 3);
            const std::vector<float> distance = caster.Cast(rays).distance.ToHost<float>();
            EXPECT_NEAR(distance[0], 9999.99f, 2e-3f);
            EXPECT_LT(distance[1], 10000.01f);
            EXPECT_TRUE(std::isinf(distance[2]));
            EXPECT_LT(distance[3], 10000.01f);
        }
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
