// SPDX-License-Identifier: BSD-3-Clause
#include <cmath>
#include <optional>

#include <gtest/gtest.h>

#include "ernie/device/device.h"
#include "ernie/scene/rotation.h"
#include "ernie/scene/scene.h"
#include "ernie/util/error.h"
#include "ernie/xr/rig.h"
#include "ernie/xr/session.h"
#include "xr_device.h"

namespace ernie::xr {
namespace {

using test::RenderDevice;

SessionDesc Simulated(Double3 head = {0, 0, 1.6}) {
    SessionDesc desc;
    desc.backend = Backend::Simulated;
    desc.simulated.profile.resolution = {8, 8};
    desc.simulated.placeControllers = true;
    desc.simulated.script = [head](uint64_t, double, SimulatedInput& input) {
        input.head.position = head;
    };
    return desc;
}

struct World {
    Ref<Scene> scene;
    Prim ball;
    Prim wall;
};

World Room(device::Device& device, const char* parent = nullptr) {
    World world{Scene::Create(device), {}, {}};
    world.scene->Define<Plane>("/Floor", {.width = 0, .length = 0});
    world.ball = world.scene->Define<Sphere>("/Ball", {.translate = {0.2, 5, 0.96}, .radius = 1});
    world.wall = world.scene->Define<Cube>("/Wall", {.translate = {0, 12, 1}, .size = 2});
    if (parent != nullptr) {
        world.scene->Define<Xform>(parent, {.translate = {10, 0, 0}});
    }
    return world;
}

#define NEED_DEVICE()                                                                              \
    device::Device* device = RenderDevice();                                                       \
    if (device == nullptr) {                                                                       \
        GTEST_SKIP() << "no supported adapter";                                                    \
    }

TEST(LocomotionTest, LasersHitWhatTheControllerAimsAtAndPassIgnoredPrims) {
    NEED_DEVICE();
    Session session(Simulated());
    const World world = Room(*device);
    Rig rig(world.scene, session);
    rig.Update(session.BeginFrame());
    PointerHit laser = rig.Laser(Side::Right);
    ASSERT_TRUE(laser.active);
    ASSERT_TRUE(laser.valid);
    EXPECT_EQ(laser.prim, world.ball.Index());
    EXPECT_NEAR(laser.localPosition.y, 4, 1e-3);
    EXPECT_NEAR(laser.position.y, 4, 1e-3);
    EXPECT_NEAR(laser.distance, 3.7, 1e-3);
    EXPECT_EQ(laser.path.size(), 2u);

    PointerDesc pointers;
    pointers.ignore = {world.ball};
    rig.SetPointers(pointers);
    rig.Update(session.BeginFrame());
    laser = rig.Laser(Side::Right);
    EXPECT_EQ(laser.prim, world.wall.Index());
    EXPECT_NEAR(laser.localPosition.y, 11, 1e-3);
}

TEST(LocomotionTest, TeleportArcsLandOnLevelGroundAndPutTheHeadThere) {
    NEED_DEVICE();
    Session session(Simulated({0.5, -0.5, 1.6}));
    const World world = Room(*device);
    Rig rig(world.scene, session);
    Frame frame = session.BeginFrame();
    rig.Update(frame);
    const PointerHit arc = rig.Teleport(Side::Left);
    ASSERT_TRUE(arc.valid);
    EXPECT_NEAR(arc.localPosition.z, 0, 1e-3);
    EXPECT_GT(arc.path.size(), 2u);
    EXPECT_TRUE(rig.TeleportTo(frame, arc));
    EXPECT_NEAR(rig.RootPose().position.x, arc.localPosition.x - 0.5, 1e-6);
    EXPECT_NEAR(rig.RootPose().position.y, arc.localPosition.y + 0.5, 1e-6);
    const Pose head = rig.WorldPose(frame.head);
    EXPECT_NEAR(head.position.x, arc.position.x, 1e-3);
    EXPECT_NEAR(head.position.y, arc.position.y, 1e-3);
    EXPECT_FALSE(rig.TeleportTo(frame, PointerHit()));
}

TEST(LocomotionTest, SnapTurnsAndScalingKeepTheHeadOverItsFloor) {
    NEED_DEVICE();
    Session session(Simulated({1, 0, 1.6}));
    const World world = Room(*device);
    Rig rig(world.scene, session);
    const Frame frame = session.BeginFrame();
    rig.Update(frame);
    const Pose before = rig.WorldPose(frame.head);
    rig.SnapTurn(frame, 1);
    Pose after = rig.WorldPose(frame.head);
    EXPECT_NEAR(after.position.x, before.position.x, 1e-9);
    EXPECT_NEAR(after.position.y, before.position.y, 1e-9);
    EXPECT_NEAR(ToEulerDegrees(rig.RootPose().orientation).z, 30, 1e-6);

    rig.SetUserScale(2);
    after = rig.WorldPose(frame.head);
    EXPECT_NEAR(after.position.x, before.position.x, 1e-9);
    EXPECT_NEAR(after.position.y, before.position.y, 1e-9);
    EXPECT_NEAR(after.position.z, 3.2, 1e-9);
    EXPECT_EQ(rig.Root().Scale(), (Float3{2, 2, 2}));
    EXPECT_THROW(rig.SetUserScale(0), Error);
}

TEST(LocomotionTest, MovesAlongTheHeadingAndStepsBack) {
    NEED_DEVICE();
    Session session(Simulated());
    const World world = Room(*device);
    Rig rig(world.scene, session);
    const Frame frame = session.BeginFrame();
    rig.Update(frame);
    rig.Move(frame, {0, 1});
    EXPECT_NEAR(rig.RootPose().position.y, 1.5 / 90, 1e-9);
    rig.StepBack(frame);
    EXPECT_NEAR(rig.RootPose().position.y, 1.5 / 90 - 1, 1e-9);
    rig.SetLocomotion({.moveMode = MoveMode::Fly});
    rig.Move(frame, {1, 0}, 1);
    EXPECT_NEAR(rig.RootPose().position.x, 3.0 / 90, 1e-9);
    EXPECT_NEAR(rig.RootPose().position.z, 3.0 / 90, 1e-9);
}

TEST(LocomotionTest, GroundedMovesLiftTheFloorOntoTheSurfaceBelow) {
    NEED_DEVICE();
    Session session(Simulated());
    const World world = Room(*device);
    world.scene->Define<Cube>("/Step", {.translate = {0, 0, -0.5}, .size = 2});
    Rig rig(world.scene, session);
    rig.SetLocomotion({.moveMode = MoveMode::Grounded});
    Frame frame = session.BeginFrame();
    rig.Update(frame);
    rig.Move(frame, {0, 0});
    for (int i = 0; i < 20 && std::abs(rig.RootPose().position.z - 0.5) > 1e-3; ++i) {
        frame = session.BeginFrame();
        rig.Update(frame);
    }
    EXPECT_NEAR(rig.RootPose().position.z, 0.5, 1e-3);
}

TEST(LocomotionTest, PointersCastFromARigUnderAMovedParent) {
    NEED_DEVICE();
    Session session(Simulated());
    const World world = Room(*device, "/Car");
    world.scene->Define<Sphere>("/Far", {.translate = {10.2, 5, 0.96}, .radius = 1});
    Rig rig(world.scene, session, {.path = "/Car/XR"});
    rig.Update(session.BeginFrame());
    const PointerHit laser = rig.Laser(Side::Right);
    ASSERT_TRUE(laser.valid);
    EXPECT_NEAR(laser.position.x, 10.2, 1e-3);
    EXPECT_NEAR(laser.localPosition.x, 0.2, 1e-3);
    for (int i = 0; i < 4; ++i) {
        rig.Update(session.BeginFrame());
    }
    EXPECT_NEAR(rig.WorldPose({{0, 0, 0}, {0, 0, 0, 1}, true}).position.x, 10, 1e-6);
}

}
}
