// SPDX-License-Identifier: BSD-3-Clause
#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/scene/rotation.h"
#include "ernie/scene/scene.h"
#include "ernie/util/error.h"
#include "ernie/xr/rig.h"
#include "ernie/xr/session.h"
#include "xr_device.h"

namespace ernie::xr {
namespace {

using test::RenderDevice;

SessionDesc Simulated() {
    SessionDesc desc;
    desc.backend = Backend::Simulated;
    desc.simulated.profile.resolution = {8, 8};
    return desc;
}

void ExpectNear(Double3 actual, Double3 expected, double tolerance = 1e-3) {
    EXPECT_NEAR(actual.x, expected.x, tolerance);
    EXPECT_NEAR(actual.y, expected.y, tolerance);
    EXPECT_NEAR(actual.z, expected.z, tolerance);
}

struct Placed {
    Double3 position;
    Double3 x;
    Double3 z;
};

Placed RootWorld(Scene& scene, Rig& rig) {
    scene.Publish().Wait();
    const WorldTransform world =
        scene.WorldTransforms(std::vector<std::string>{rig.Root().Path()}).front();
    return {{double(world.high.x) + world.low.x, double(world.high.y) + world.low.y,
             double(world.high.z) + world.low.z},
            {world.x.x, world.x.y, world.x.z},
            {world.z.x, world.z.y, world.z.z}};
}

Frame Settle(Session& session, Rig& rig, int frames = 4) {
    Frame frame;
    for (int i = 0; i < frames; ++i) {
        frame = session.BeginFrame();
        rig.Update(frame);
    }
    return frame;
}

#define NEED_DEVICE()                                                                              \
    device::Device* device = RenderDevice();                                                       \
    if (device == nullptr) {                                                                       \
        GTEST_SKIP() << "no supported adapter";                                                    \
    }

TEST(AnchorTest, PrimAnchorsPutTheStageOnTheirTargetAsItMoves) {
    NEED_DEVICE();
    Session session(Simulated());
    const Ref<Scene> scene = Scene::Create(*device);
    const Prim car = scene->Define<Xform>(
        "/Car", {.translate = {10, 0, 3}, .orient = FromAxisAngle({0, 0, 1}, 90)});
    Rig rig(scene, session);
    rig.SetAnchor({.mode = Anchor::Prim, .target = "/Car"});
    Settle(session, rig);
    Placed root = RootWorld(*scene, rig);
    ExpectNear(root.position, {10, 0, 3});
    ExpectNear(root.x, {0, 1, 0});
    ExpectNear(rig.WorldPose({{1, 0, 0}, {0, 0, 0, 1}, true}).position, {10, 1, 3});

    car.Set(Xformable::kTranslate, Double3{20, 0, 3});
    ExpectNear(RootWorld(*scene, rig).position, {20, 0, 3});
    EXPECT_THROW(rig.SetAnchor({.mode = Anchor::Prim, .target = "/Nowhere"}), Error);
    EXPECT_THROW(rig.SetAnchor({.mode = Anchor::Follow}), Error);
    EXPECT_EQ(rig.GetAnchor().mode, Anchor::Prim);
}

TEST(AnchorTest, FollowAnchorsKeepTheStageLevelAndFixedHeightsHoldTheHead) {
    NEED_DEVICE();
    Session session(Simulated());
    const Ref<Scene> scene = Scene::Create(*device);
    scene->Define<Xform>("/Car",
                         {.translate = {10, 0, 3}, .orient = FromEulerDegrees({30, 0, 90})});
    Rig rig(scene, session);
    rig.SetAnchor({.mode = Anchor::Follow, .target = "/Car", .fixedHeight = 1.0});
    const Frame frame = Settle(session, rig);
    const Placed root = RootWorld(*scene, rig);
    ExpectNear(root.x, {0, 1, 0});
    ExpectNear(root.z, {0, 0, 1});
    ExpectNear(rig.WorldPose(frame.head).position, {10, 0, 4});
}

TEST(AnchorTest, CameraAnchorsLandTheHeadOnTheCamera) {
    NEED_DEVICE();
    Session session(Simulated());
    const Ref<Scene> scene = Scene::Create(*device);
    scene->Define<Camera>("/Shot", {.translate = {5, 5, 2}, .orient = LookAlong({1, 0, 0})});
    Rig rig(scene, session);
    rig.SetAnchor({.mode = Anchor::Camera, .target = "/Shot"});
    const Frame frame = Settle(session, rig);
    const Pose head = rig.WorldPose(frame.head);
    ExpectNear(head.position, {5, 5, 2});
    ExpectNear(Rotate(head.orientation, {0, 0, -1}), {1, 0, 0});
}

TEST(AnchorTest, TeleportsOntoBodiesRideThemAndOffThemLeave) {
    NEED_DEVICE();
    Session session(Simulated());
    const Ref<Scene> scene = Scene::Create(*device);
    const Prim floor = scene->Define<Plane>("/Floor", {.width = 0, .length = 0});
    const Prim truck = scene->Define<Xform>("/Truck", {.translate = {0, 4, 0}});
    const Prim bed = scene->Define<Cube>("/Truck/Bed", {.translate = {0, 0, 0.5}, .size = 1});
    Rig rig(scene, session);
    rig.SetLocomotion({.bodies = {truck}});
    Frame frame = Settle(session, rig);

    PointerHit onto;
    onto.valid = true;
    onto.prim = bed.Index();
    onto.position = {0.2, 4.1, 1};
    EXPECT_TRUE(rig.TeleportTo(frame, onto));
    ASSERT_TRUE(rig.Body().has_value());
    EXPECT_EQ(rig.Body()->Index(), truck.Index());
    frame = Settle(session, rig);
    ExpectNear(rig.WorldPose(frame.head).position, {0.2, 4.1, 2.6});

    truck.Set(Xformable::kTranslate, Double3{10, 4, 0});
    ExpectNear(RootWorld(*scene, rig).position, {10.2, 4.1, 1});

    PointerHit off;
    off.valid = true;
    off.prim = floor.Index();
    off.position = {-3, 0, 0};
    frame = Settle(session, rig);
    EXPECT_TRUE(rig.TeleportTo(frame, off));
    EXPECT_FALSE(rig.Body().has_value());
    truck.Set(Xformable::kTranslate, Double3{30, 4, 0});
    frame = Settle(session, rig);
    ExpectNear(rig.WorldPose(frame.head).position, {-3, 0, 1.6});
}

}
}
