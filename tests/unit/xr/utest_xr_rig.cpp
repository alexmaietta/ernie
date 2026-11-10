// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/xr/rig.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/device/device.h"
#include "ernie/render/render_output.h"
#include "ernie/render/renderer.h"
#include "ernie/scene/scene.h"
#include "ernie/util/error.h"
#include "ernie/xr/session.h"
#include "xr_device.h"

namespace ernie::xr {
namespace {

using test::RenderDevice;

SessionDesc Simulated() {
    SessionDesc desc;
    desc.backend = Backend::Simulated;
    desc.simulated.profile.resolution = {21, 22};
    desc.simulated.placeControllers = true;
    return desc;
}

Ref<Scene> BallAhead(device::Device& device) {
    const Ref<Scene> scene = Scene::Create(device);
    scene->Define<Sphere>("/Ball", {.translate = {0, 4, 1.6}, .radius = 1});
    return scene;
}

device::Tensor RunFrames(Session& session, Rig& rig, Renderer& renderer, int frames) {
    device::Tensor color;
    for (int i = 0; i < frames; ++i) {
        const Frame frame = session.BeginFrame();
        rig.Update(frame);
        RenderOutput output = renderer.Step(rig.Product(), frame.deltaTime);
        session.EndFrame(frame, output);
        color = output[rig.Product()].frames.back().RenderVar(rig.ColorVar()).tensor;
    }
    return color;
}

TEST(RigTest, RendersBothEyesAndPresentsThem) {
    device::Device* device = RenderDevice();
    if (device == nullptr) {
        GTEST_SKIP() << "no supported adapter";
    }
    Session session(Simulated());
    const Ref<Scene> scene = BallAhead(*device);
    Rig rig(scene, session);
    EXPECT_EQ(rig.RecommendedResolution(), (Uint2{21, 22}));
    Renderer renderer(scene);
    const device::Tensor color = RunFrames(session, rig, renderer, 2);

    const device::Tensor presented = session.Presented();
    ASSERT_FALSE(presented.Empty());
    EXPECT_EQ(std::vector<int64_t>(presented.Shape().begin(), presented.Shape().end()),
              (std::vector<int64_t>{2, 22, 21, 4}));
    const std::vector<uint8_t> pixels = presented.ToHost<uint8_t>();
    EXPECT_EQ(pixels, color.ToHost<uint8_t>());
    const auto ballColumn = [&](size_t eye) {
        double columns = 0;
        uint32_t count = 0;
        for (size_t row = 0; row < 22; ++row) {
            for (size_t column = 0; column < 21; ++column) {
                if (pixels[((eye * 22 + row) * 21 + column) * 4 + 3] != 0) {
                    columns += double(column);
                    ++count;
                }
            }
        }
        EXPECT_GT(count, 0u) << "eye " << eye;
        return count > 0 ? columns / count : 10.0;
    };
    EXPECT_GT(ballColumn(0), 10.5);
    EXPECT_LT(ballColumn(1), 9.5);
    EXPECT_EQ(session.Status().framesPresented, 2u);
    EXPECT_EQ(session.Stats().presented, 2u);
}

TEST(RigTest, SubmitsEachEyesDepthFromNearToFar) {
    device::Device* device = RenderDevice();
    if (device == nullptr) {
        GTEST_SKIP() << "no supported adapter";
    }
    Session session(Simulated());
    const Ref<Scene> scene = BallAhead(*device);
    Rig rig(scene, session);
    EXPECT_EQ(rig.Vars().size(), 2u);
    EXPECT_TRUE(session.Status().depthLayer);
    Renderer renderer(scene);
    RunFrames(session, rig, renderer, 1);

    const device::Tensor depth = session.PresentedDepth();
    ASSERT_FALSE(depth.Empty());
    EXPECT_EQ(std::vector<int64_t>(depth.Shape().begin(), depth.Shape().end()),
              (std::vector<int64_t>{2, 22, 21}));
    const std::vector<float> values = depth.ToHost<float>();
    const std::vector<uint8_t> color = session.Presented().ToHost<uint8_t>();
    uint32_t ball = 0;
    for (size_t pixel = 0; pixel < values.size(); ++pixel) {
        if (color[pixel * 4 + 3] != 0) {
            ++ball;
            EXPECT_GT(values[pixel], 0.9f);
            EXPECT_LT(values[pixel], 1.0f);
        } else {
            EXPECT_EQ(values[pixel], 1.0f);
        }
    }
    EXPECT_GT(ball, 0u);

    SessionDesc flat = Simulated();
    flat.depthLayer = false;
    Session without(flat);
    Rig plain(BallAhead(*device), without, {.path = "/Flat"});
    EXPECT_EQ(plain.Vars().size(), 1u);
    EXPECT_FALSE(without.Status().depthLayer);
}

TEST(RigTest, DrawsItsVisualsIntoTheEyes) {
    device::Device* device = RenderDevice();
    if (device == nullptr) {
        GTEST_SKIP() << "no supported adapter";
    }
    Session session(Simulated());
    const Ref<Scene> scene = BallAhead(*device);
    Rig rig(scene, session);
    Renderer renderer(scene);
    const Frame frame = session.BeginFrame();
    rig.Update(frame);
    rig.Draw(*renderer.GetOverlays(), frame, {.teleport = true});
    EXPECT_EQ(renderer.GetOverlays()->Names(), (std::vector<std::string>{"xr"}));
    RenderOutput output = renderer.Step(rig.Product(), frame.deltaTime);
    session.EndFrame(frame, output);
    const std::vector<uint8_t> raw =
        output[rig.Product()].frames.back().RenderVar(rig.ColorVar()).tensor.ToHost<uint8_t>();
    const std::vector<uint8_t> shown = session.Presented().ToHost<uint8_t>();
    ASSERT_EQ(raw.size(), shown.size());
    EXPECT_NE(raw, shown);
}

TEST(RigTest, AimsEachEyeAlongItsViewAndMovesTheHands) {
    device::Device* device = RenderDevice();
    if (device == nullptr) {
        GTEST_SKIP() << "no supported adapter";
    }
    Session session(Simulated());
    const Ref<Scene> scene = BallAhead(*device);
    Rig rig(scene, session);
    const Frame frame = session.BeginFrame();
    rig.Update(frame);

    const CameraPrim left = rig.Eye(Side::Left);
    EXPECT_NEAR(left.Translate().x, -0.0315, 1e-6);
    EXPECT_NEAR(rig.Eye(Side::Right).Translate().x, 0.0315, 1e-6);
    const Fov fov = frame.Eye(Side::Left).fov;
    EXPECT_NEAR(left.HorizontalAperture(), 50 * (std::tan(fov.right) - std::tan(fov.left)), 1e-3);
    EXPECT_LT(left.HorizontalApertureOffset(), 0);
    EXPECT_GT(rig.Eye(Side::Right).HorizontalApertureOffset(), 0);
    EXPECT_NEAR(rig.Head().Translate().z, 1.6, 1e-6);
    EXPECT_NEAR(rig.Hand(Side::Right).Translate().x, 0.2, 1e-6);

    rig.SetRootPose({{3, 0, 0}, {0, 0, 0, 1}, true});
    EXPECT_EQ(rig.Root().Translate(), (Double3{3, 0, 0}));
    EXPECT_EQ(rig.RootPose().position, (Double3{3, 0, 0}));
}

TEST(RigTest, PresentsOnTheSessionsDeviceWhenTheSceneRendersOnAnother) {
    device::Device* device = RenderDevice();
    if (device == nullptr) {
        GTEST_SKIP() << "no supported adapter";
    }
    Session session(Simulated());
    const Ref<device::Device> headset = session.CreateDevice({.adapter = device->Adapter().alias});
    ASSERT_NE(headset.Get(), device);
    const Ref<Scene> scene = BallAhead(*device);
    Rig rig(scene, session);
    Renderer renderer(scene);
    const device::Tensor color = RunFrames(session, rig, renderer, 1);

    const device::Tensor presented = session.Presented();
    EXPECT_EQ(&presented.GetDevice(), headset.Get());
    EXPECT_EQ(presented.ToHost<uint8_t>(), color.ToHost<uint8_t>());
}

}
}
