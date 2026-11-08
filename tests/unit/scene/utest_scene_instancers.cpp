// SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/bvh/kernels/bvh.h"
#include "ernie/device/device.h"
#include "ernie/device/queue.h"
#include "ernie/scene/kernels/instance_layout.h"
#include "ernie/scene/kernels/lights.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/scene_impl.h"
#include "support/devices.h"

namespace ernie {
namespace {

struct Slots {
    std::vector<SlotHit> hits;
    std::vector<InstanceRecord> records;
    uint32_t count = 0;
};

Slots Read(Scene& scene, device::Device& device) {
    scene.Publish({.wait = true});
    const InstanceTable& instances = SceneImpl::Get(scene).GetInstances();
    Slots slots;
    slots.count = instances.InstanceCounts()[0];
    slots.hits.resize(slots.count);
    slots.records.resize(slots.count);
    device.GetQueue().Read(instances.Hits()->Slice(0, slots.count * sizeof(SlotHit)),
                           slots.hits.data());
    device.GetQueue().Read(instances.Records()->Slice(0, slots.count * sizeof(InstanceRecord)),
                           slots.records.data());
    return slots;
}

Float3 Translation(const InstanceRecord& record) {
    return {record.transform[3], record.transform[7], record.transform[11]};
}

void Prototype(Scene& scene) {
    scene.Define<Scope>("/Inst/Protos");
    scene.Define<Xform>("/Inst/Protos/Box", {.translate = {0, 0, 1}});
    scene.Define<Cube>("/Inst/Protos/Box/Cube", {.size = 1});
    scene.Define<Sphere>("/Inst/Protos/Box/Ball", {.translate = {0, 1, 0}, .radius = 0.5});
}

TEST(InstancersTest, PointInstancersDrawEachInstancesPrototypeAndHidePrototypesInPlace) {
    test::ForEachDevice([](device::Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        const Prim inst =
            scene->Define<PointInstancer>("/Inst", {.prototypes = {"/Inst/Protos/Box"},
                                                    .protoIndices = {0, 0, 0},
                                                    .positions = {{0, 0, 0}, {5, 0, 0}, {10, 0, 0}},
                                                    .scales = {{1, 1, 1}, {2, 2, 2}, {1, 1, 1}}});
        Prototype(*scene);
        const uint32_t cube = scene->Find("/Inst/Protos/Box/Cube")->Index();
        const uint32_t ball = scene->Find("/Inst/Protos/Box/Ball")->Index();

        Slots slots = Read(*scene, device);
        ASSERT_EQ(slots.count, 6u);
        for (uint32_t slot = 0; slot < 6; ++slot) {
            EXPECT_EQ(slots.hits[slot].instancer, inst.Index());
            EXPECT_EQ(slots.hits[slot].instance, slot / 2);
            EXPECT_TRUE(slots.hits[slot].prim == cube || slots.hits[slot].prim == ball);
            EXPECT_NE(slots.records[slot].blas, 0u);
        }
        const uint32_t scaledCube = slots.hits[2].prim == cube ? 2 : 3;
        const Float3 moved = Translation(slots.records[scaledCube]);
        EXPECT_FLOAT_EQ(moved.x, 5);
        EXPECT_FLOAT_EQ(moved.y, 0);
        EXPECT_FLOAT_EQ(moved.z, 2);
        const uint32_t scaledBall = 5 - scaledCube;
        EXPECT_FLOAT_EQ(Translation(slots.records[scaledBall]).y, 2);

        inst.Set(PointInstancer::kInvisibleIds, std::vector<int64_t>{1});
        slots = Read(*scene, device);
        EXPECT_EQ(slots.records[2].blas, 0u);
        EXPECT_EQ(slots.records[3].blas, 0u);
        EXPECT_NE(slots.records[4].blas, 0u);

        inst.Set(PointInstancer::kInvisibleIds, std::vector<int64_t>{});
        inst.Set(PointInstancer::kProtoIndices, std::vector<int32_t>{0, 0, 7});
        slots = Read(*scene, device);
        EXPECT_NE(slots.records[2].blas, 0u);
        EXPECT_EQ(slots.records[4].blas, 0u);
        EXPECT_EQ(slots.records[5].blas, 0u);

        inst.Set(PointInstancer::kPositions, std::vector<Float3>{{0, 0, 0}});
        EXPECT_EQ(Read(*scene, device).count, 2u);
    });
}

TEST(InstancersTest, PrototypeApiDrawsOneCopyWithoutTheRootsTransform) {
    test::ForEachDevice([](device::Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        scene->Define<PointInstancer>("/Inst", {.prototypes = {"/Inst/Protos/Box"}});
        Prototype(*scene);
        const Prim copy = scene->Define<Xform>(
            "/Copy", {.translate = {0, 10, 0},
                      .prototype = PrototypeAPI::Desc{.prototype = {"/Inst/Protos/Box"}}});
        const uint32_t cube = scene->Find("/Inst/Protos/Box/Cube")->Index();
        Slots slots = Read(*scene, device);
        ASSERT_EQ(slots.count, 2u);
        const uint32_t at = slots.hits[0].prim == cube ? 0 : 1;
        EXPECT_EQ(slots.hits[at].instancer, copy.Index());
        const Float3 moved = Translation(slots.records[at]);
        EXPECT_FLOAT_EQ(moved.y, 10);
        EXPECT_FLOAT_EQ(moved.z, 0);

        scene->Unapply<PrototypeAPI>(copy);
        EXPECT_EQ(Read(*scene, device).count, 0u);
    });
}

TEST(InstancersTest, CountChangesAndHiddenIdsKeepTheLayoutAndBoundsFollowCopies) {
    test::ForEachDevice([](device::Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        const Prim inst = scene->Define<PointInstancer>(
            "/Inst", {.prototypes = {"/Inst/Protos/Box"},
                      .protoIndices = {0, 0, 0},
                      .positions = {{5, 0, 0}, {10, 0, 0}, {15, 0, 0}},
                      .invisibleIds = {2, 0}});
        Prototype(*scene);
        Slots slots = Read(*scene, device);
        ASSERT_EQ(slots.count, 6u);
        EXPECT_EQ(slots.records[0].blas, 0u);
        EXPECT_NE(slots.records[2].blas, 0u);
        EXPECT_EQ(slots.records[4].blas, 0u);

        const InstanceTable& instances = SceneImpl::Get(*scene).GetInstances();
        const uint32_t relayouts = instances.Relayouts();
        inst.Set(PointInstancer::kInvisibleIds, std::vector<int64_t>{});
        inst.Set(PointInstancer::kPositions, std::vector<Float3>{{5, 0, 0}, {10, 0, 0}});
        inst.Set(PointInstancer::kProtoIndices, std::vector<int32_t>{0, 0});
        slots = Read(*scene, device);
        EXPECT_EQ(slots.count, 4u);
        EXPECT_EQ(instances.Relayouts(), relayouts);

        const std::optional<Box> world = scene->Bounds();
        ASSERT_TRUE(world.has_value());
        EXPECT_NEAR(world->lower.x, 4.5f, 1e-4f);
        EXPECT_NEAR(world->upper.x, 10.5f, 1e-4f);
        const Box box = scene->PrimBounds(std::vector<std::string>{inst.Path()}).front();
        const Float3 lower = box.lower;
        const Float3 upper = box.upper;
        EXPECT_NEAR(lower.x, 4.5f, 1e-4f);
        EXPECT_NEAR(upper.x, 10.5f, 1e-4f);
        EXPECT_NEAR(upper.y, 1.5f, 1e-4f);
    });
}

TEST(InstancersTest, RestackingAlongsideAGlobalMoveRebuildsInsteadOfRefitting) {
    test::ForEachDevice([](device::Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        const Prim inst = scene->Define<PointInstancer>(
            "/Inst", {.prototypes = {"/Inst/Protos/Box"},
                      .protoIndices = {0, 0, 0},
                      .positions = {{5, 0, 0}, {10, 0, 0}, {15, 0, 0}}});
        Prototype(*scene);
        const Prim sun = scene->Define<Sphere>("/Sun");
        scene->Publish({.wait = true});
        const InstanceTable& instances = SceneImpl::Get(*scene).GetInstances();
        const uint32_t relayouts = instances.Relayouts();

        inst.Set(PointInstancer::kPositions, std::vector<Float3>{{5, 0, 0}, {10, 0, 0}});
        sun.Set(Xformable::kTranslate, Double3{0, 0, 9});
        scene->Publish({.wait = true});
        EXPECT_EQ(instances.Relayouts(), relayouts);
        EXPECT_EQ(instances.Stats().built, instances.Stats().tlases);
        EXPECT_EQ(instances.Stats().refit, 0u);
    });
}

TEST(InstancersTest, NestedInstancersMultiplyThroughEveryLevel) {
    test::ForEachDevice([](device::Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        const Prim outer = scene->Define<PointInstancer>(
            "/Outer", {.prototypes = {"/Outer/Protos/Row"},
                       .protoIndices = {0, 0, 0},
                       .positions = {{0, 0, 0}, {0, 5, 0}, {0, 10, 0}}});
        scene->Define<Scope>("/Outer/Protos");
        scene->Define<Xform>("/Outer/Protos/Row");
        scene->Define<PointInstancer>("/Outer/Protos/Row/Inner",
                                      {.prototypes = {"/Outer/Protos/Row/Inner/Box"},
                                       .protoIndices = {0, 0},
                                       .positions = {{0, 0, 0}, {1, 0, 0}}});
        scene->Define<Cube>("/Outer/Protos/Row/Inner/Box", {.size = 1});
        const uint32_t box = scene->Find("/Outer/Protos/Row/Inner/Box")->Index();

        const Slots slots = Read(*scene, device);
        ASSERT_EQ(slots.count, 6u);
        for (uint32_t slot = 0; slot < 6; ++slot) {
            EXPECT_EQ(slots.hits[slot].prim, box);
            EXPECT_EQ(slots.hits[slot].instancer, outer.Index());
            EXPECT_EQ(slots.hits[slot].instance, slot / 2);
            const Float3 at = Translation(slots.records[slot]);
            EXPECT_FLOAT_EQ(at.x, float(slot % 2)) << slot;
            EXPECT_FLOAT_EQ(at.y, float(5 * (slot / 2))) << slot;
        }
    });
}

TEST(InstancersTest, LightsInPrototypesLightFromEveryCopyAndNotInPlace) {
    test::ForEachDevice([](device::Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        const Prim inst =
            scene->Define<PointInstancer>("/Lamps", {.prototypes = {"/Lamps/Lamp"},
                                                     .protoIndices = {0, 0},
                                                     .positions = {{0, 0, 2}, {5, 0, 2}}});
        scene->Define<SphereLight>("/Lamps/Lamp", {.intensity = 3, .treatAsPoint = true});
        scene->Publish({.wait = true});
        const LightTable& lights = SceneImpl::Get(*scene).GetLights();
        uint32_t total = 0;
        for (const uint32_t count : lights.Counts()) {
            total += count;
        }
        ASSERT_EQ(total, 2u);
        std::vector<LightRecord> records(total);
        device.GetQueue().Read(lights.Records()->Slice(0, total * sizeof(LightRecord)),
                               records.data());
        std::vector<float> xs;
        for (const LightRecord& record : records) {
            EXPECT_EQ(record.instancer, inst.Index());
            EXPECT_GT(record.power, 0.0f);
            EXPECT_FLOAT_EQ(record.position.z, 2);
            xs.push_back(record.position.x);
        }
        std::ranges::sort(xs);
        EXPECT_FLOAT_EQ(xs[0], 0);
        EXPECT_FLOAT_EQ(xs[1], 5);
    });
}

TEST(InstancersTest, LightCopiesFollowCountChangesThatKeepTheLayout) {
    test::ForEachDevice([](device::Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        const Prim inst =
            scene->Define<PointInstancer>("/Lamps", {.prototypes = {"/Lamps/Lamp"},
                                                     .protoIndices = {0, 0},
                                                     .positions = {{0, 0, 2}, {5, 0, 2}}});
        scene->Define<SphereLight>("/Lamps/Lamp", {.intensity = 3, .treatAsPoint = true});
        scene->Publish({.wait = true});
        SceneImpl& impl = SceneImpl::Get(*scene);
        const uint32_t relayouts = impl.GetInstances().Relayouts();
        const auto total = [&] {
            uint32_t count = 0;
            for (const uint32_t lights : impl.GetLights().Counts()) {
                count += lights;
            }
            return count;
        };
        ASSERT_EQ(total(), 2u);

        inst.Set(PointInstancer::kPositions, std::vector<Float3>{{0, 0, 2}, {5, 0, 2}, {9, 0, 2}});
        inst.Set(PointInstancer::kProtoIndices, std::vector<int32_t>{0, 0, 0});
        scene->Publish({.wait = true});
        EXPECT_EQ(impl.GetInstances().Relayouts(), relayouts);
        EXPECT_EQ(total(), 3u);
    });
}

}
}
