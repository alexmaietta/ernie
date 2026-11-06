// SPDX-License-Identifier: BSD-3-Clause
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/device/device.h"
#include "ernie/scene/scene.h"
#include "support/devices.h"

namespace ernie {
namespace {

template <class Test> void ForEachScene(Test test) {
    test::ForEachDevice([&](device::Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        test(*scene);
    });
}

Double3 WorldPosition(Scene& scene, std::string_view path) {
    scene.Publish();
    const WorldTransform frame = scene.Select(path).Get(Xformable::kWorldTransform).Get().front();
    return {double(frame.high.x) + frame.low.x, double(frame.high.y) + frame.low.y,
            double(frame.high.z) + frame.low.z};
}

TEST(MoveTest, RenamedPrimsKeepTheirIndexAndWhatPointsAtThem) {
    ForEachScene([](Scene& scene) {
        scene.Define<Scope>("/Looks");
        const Prim red = scene.Define<Material>("/Looks/Red", {.id = "UsdPreviewSurface"});
        scene.Define<Sphere>("/Ball", {.materialBinding = {"/Looks/Red"}});
        const uint64_t before = scene.StructureVersion();

        const Prim crimson = scene.Rename(red, "Crimson");
        EXPECT_EQ(crimson.Index(), red.Index());
        EXPECT_EQ(crimson.Path(), "/Looks/Crimson");
        EXPECT_FALSE(scene.Find("/Looks/Red"));
        EXPECT_EQ(scene.Find("/Looks/Crimson")->Index(), red.Index());
        EXPECT_GT(scene.StructureVersion(), before);
        EXPECT_EQ(scene.Select("/Ball").Get("materialBinding").GetTargets(0),
                  std::vector<std::string>{"/Looks/Crimson"});
        EXPECT_THROW(scene.Rename(red, "Again"), Error);
    });
}

TEST(MoveTest, MovedSubtreesFollowTheirNewParent) {
    ForEachScene([](Scene& scene) {
        scene.Define<Xform>("/A", {.translate = {1, 0, 0}});
        scene.Define<Xform>("/B", {.translate = {0, 5, 0}});
        scene.Define<Xform>("/A/Arm", {.translate = {0, 0, 2}});
        scene.Define<Sphere>("/A/Arm/Hand");
        EXPECT_EQ(WorldPosition(scene, "/A/Arm/Hand").x, 1.0);

        const Prim arm = scene.Move(*scene.Find("/A/Arm"), "/B/Limb");
        EXPECT_EQ(arm.Path(), "/B/Limb");
        EXPECT_EQ(arm.Parent()->Path(), "/B");
        EXPECT_FALSE(scene.Find("/A/Arm/Hand"));
        const Double3 hand = WorldPosition(scene, "/B/Limb/Hand");
        EXPECT_EQ(hand.x, 0.0);
        EXPECT_EQ(hand.y, 5.0);
        EXPECT_EQ(hand.z, 2.0);

        scene.Move(arm, "/Limb");
        EXPECT_EQ(WorldPosition(scene, "/Limb/Hand").y, 0.0);
        EXPECT_EQ(scene.Find("/Limb")->Parent(), std::nullopt);
    });
}

TEST(MoveTest, MovesInsideClonesAndOntoPendingTargets) {
    ForEachScene([](Scene& scene) {
        scene.Define<Xform>("/Robot");
        scene.Define<Sphere>("/Robot/Wheel");
        scene.Clone(*scene.Find("/Robot"), std::vector<std::string>{"/Copy0", "/Copy1"});
        scene.Rename(*scene.Find("/Copy1/Wheel"), "Tire");
        EXPECT_TRUE(scene.Find("/Copy1/Tire"));
        EXPECT_TRUE(scene.Find("/Copy0/Wheel"));
        EXPECT_FALSE(scene.Find("/Copy1/Wheel"));

        scene.Define<Camera>("/Cam");
        scene.Define<RenderProduct>("/View", {.camera = {"/Later"}});
        EXPECT_TRUE(scene.Select("/View").Get("camera").GetTargets(0).empty());
        scene.Move(*scene.Find("/Cam"), "/Later");
        EXPECT_EQ(scene.Select("/View").Get("camera").GetTargets(0),
                  std::vector<std::string>{"/Later"});
    });
}

TEST(MoveTest, BadMovesChangeNothing) {
    ForEachScene([](Scene& scene) {
        scene.Define<Xform>("/A");
        scene.Define<Xform>("/A/B");
        scene.Define<Xform>("/C");
        scene.Define<World>("/env_0");
        scene.Define<Sphere>("/env_0/Ball");
        scene.Define<World>("/env_1");
        const Prim a = *scene.Find("/A");
        EXPECT_THROW(scene.Move(a, "/A/B/A"), Error);
        EXPECT_THROW(scene.Move(a, "/C"), Error);
        EXPECT_THROW(scene.Move(a, "/Missing/A"), Error);
        EXPECT_THROW(scene.Move(a, "/A B"), Error);
        EXPECT_THROW(scene.Move(a, "/env_0/A"), Error);
        EXPECT_THROW(scene.Move(*scene.Find("/env_0/Ball"), "/Ball"), Error);
        EXPECT_THROW(scene.Move(*scene.Find("/env_0"), "/env_1/env"), Error);
        EXPECT_EQ(scene.Find("/A/B")->Path(), "/A/B");
        EXPECT_TRUE(a.Valid());
        EXPECT_EQ(scene.Move(*scene.Find("/env_0"), "/env_9").Path(), "/env_9");
        EXPECT_TRUE(scene.Find("/env_9/Ball"));
    });
}

TEST(RemoveTest, RelationshipsToRemovedPrimsWaitForTheirPath) {
    ForEachScene([](Scene& scene) {
        scene.Define<Scope>("/Looks");
        const Prim red = scene.Define<Material>("/Looks/Red", {.id = "UsdPreviewSurface"});
        scene.Define<Material>("/Looks/Blue", {.id = "UsdPreviewSurface"});
        scene.Define<Sphere>("/Ball", {.materialBinding = {"/Looks/Red"}});
        scene.Define<Sphere>("/Pair", {.materialBinding = {"/Looks/Red", "/Looks/Blue"}});
        const uint32_t index = red.Index();
        scene.Remove(red);
        EXPECT_TRUE(scene.Select("/Ball").Get("materialBinding").GetTargets(0).empty());
        EXPECT_EQ(scene.Select("/Pair").Get("materialBinding").GetTargets(0),
                  std::vector<std::string>{"/Looks/Blue"});

        const Prim reused = scene.Define<Material>("/Looks/Green", {.id = "UsdPreviewSurface"});
        EXPECT_EQ(reused.Index(), index);
        EXPECT_TRUE(scene.Select("/Ball").Get("materialBinding").GetTargets(0).empty());

        scene.Define<Material>("/Looks/Red", {.id = "UsdPreviewSurface"});
        EXPECT_EQ(scene.Select("/Ball").Get("materialBinding").GetTargets(0),
                  std::vector<std::string>{"/Looks/Red"});
        EXPECT_EQ(scene.Select("/Pair").Get("materialBinding").GetTargets(0),
                  (std::vector<std::string>{"/Looks/Red", "/Looks/Blue"}));
    });
}

TEST(CopyTest, CopiesPasteWithTheirValuesAndRelationships) {
    ForEachScene([](Scene& scene) {
        scene.Define<Scope>("/Looks");
        scene.Define<Material>("/Looks/Shared", {.id = "UsdPreviewSurface"});
        scene.Define<Xform>("/Robot", {.translate = {1, 2, 3}});
        scene.Define<Material>("/Robot/Paint", {.id = "UsdPreviewSurface"});
        scene.Define<Sphere>("/Robot/Head", {.materialBinding = {"/Robot/Paint"}, .radius = 0.25f});
        scene.Define<Cube>("/Robot/Body", {.materialBinding = {"/Looks/Shared"}, .size = 2});

        const PrimCopy copy = scene.Copy(*scene.Find("/Robot"));
        EXPECT_EQ(copy.Name(), "Robot");
        EXPECT_EQ(copy.PrimCount(), 4u);
        const Prim pasted = scene.Paste(copy, "/Twin");
        EXPECT_EQ(pasted.Path(), "/Twin");
        EXPECT_EQ(scene.Select("/Twin").Get<Double3>("translate").Get().front(),
                  (Double3{1, 2, 3}));
        EXPECT_EQ(scene.Select("/Twin/Head").Get<float>("radius").Get().front(), 0.25f);
        EXPECT_EQ(scene.Select("/Twin/Head").Get("materialBinding").GetTargets(0),
                  std::vector<std::string>{"/Twin/Paint"});
        EXPECT_EQ(scene.Select("/Twin/Body").Get("materialBinding").GetTargets(0),
                  std::vector<std::string>{"/Looks/Shared"});
        EXPECT_EQ(scene.Select("/Robot/Head").Get("materialBinding").GetTargets(0),
                  std::vector<std::string>{"/Robot/Paint"});
        EXPECT_THROW(scene.Paste(copy, "/Twin"), Error);
        EXPECT_THROW(scene.Paste(PrimCopy(), "/Empty"), Error);
    });
}

TEST(CopyTest, PastingARemovedSubtreeBackReconnectsWhatPointedAtIt) {
    ForEachScene([](Scene& scene) {
        scene.Define<Scope>("/Looks");
        scene.Define<Material>("/Looks/Red", {.id = "UsdPreviewSurface"});
        scene.Define<Sphere>("/Ball", {.materialBinding = {"/Looks/Red"}});
        const PrimCopy removed = scene.Copy(*scene.Find("/Looks"));
        scene.Remove(*scene.Find("/Looks"));
        EXPECT_TRUE(scene.Select("/Ball").Get("materialBinding").GetTargets(0).empty());
        scene.Paste(removed, "/Looks");
        EXPECT_EQ(scene.Select("/Ball").Get("materialBinding").GetTargets(0),
                  std::vector<std::string>{"/Looks/Red"});
    });
}

TEST(CopyTest, CopiesPasteIntoOtherScenes) {
    test::ForEachDevice([](device::Device& device) {
        const Ref<Scene> from = Scene::Create(device);
        const Ref<Scene> to = Scene::Create(device);
        from->Define<Material>("/Paint", {.id = "UsdPreviewSurface"});
        from->Define<Sphere>("/Ball", {.materialBinding = {"/Paint"}, .radius = 3});
        to->Paste(from->Copy(*from->Find("/Paint")), "/Paint");
        to->Paste(from->Copy(*from->Find("/Ball")), "/Ball");
        EXPECT_EQ(to->Select("/Ball").Get<float>("radius").Get().front(), 3.0f);
        EXPECT_EQ(to->Select("/Ball").Get("materialBinding").GetTargets(0),
                  std::vector<std::string>{"/Paint"});
    });
}
}
}
