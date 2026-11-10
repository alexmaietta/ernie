// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/xr/session.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/util/error.h"

namespace ernie::xr {
namespace {

SessionDesc Simulated() {
    SessionDesc desc;
    desc.backend = Backend::Simulated;
    return desc;
}

TEST(SessionTest, SimulatesAHeadsetStandingAtItsEyeHeight) {
    Session session(Simulated());
    EXPECT_TRUE(session.Running());
    const Frame frame = session.BeginFrame();
    EXPECT_TRUE(frame.shouldRender);
    EXPECT_EQ(frame.state, SessionState::Focused);
    EXPECT_TRUE(frame.head.valid);
    EXPECT_NEAR(frame.head.position.z, 1.6, 1e-6);
    const double separation =
        frame.Eye(Side::Right).pose.position.x - frame.Eye(Side::Left).pose.position.x;
    EXPECT_NEAR(separation, 0.063, 1e-6);
    EXPECT_NEAR(frame.deltaTime, 1.0 / 90, 1e-6);
    const SessionStatus status = session.Status();
    EXPECT_EQ(status.backend, Backend::Simulated);
    EXPECT_EQ(status.resolution, (Uint2{2064, 2208}));
    EXPECT_EQ(status.state, "focused");
    EXPECT_EQ(status.framesBegun, 1u);
}

TEST(SessionTest, StepsThroughStartupAndEndsOnRequest) {
    SessionDesc desc = Simulated();
    desc.simulated.skipStartup = false;
    Session session(desc);
    std::vector<SessionState> states;
    for (int i = 0; i < 5; ++i) {
        const Frame frame = session.BeginFrame();
        states.push_back(frame.state);
        EXPECT_EQ(frame.shouldRender,
                  frame.state == SessionState::Visible || frame.state == SessionState::Focused);
    }
    EXPECT_EQ(states, (std::vector<SessionState>{SessionState::Ready, SessionState::Synchronized,
                                                 SessionState::Visible, SessionState::Focused,
                                                 SessionState::Focused}));
    session.RequestExit();
    EXPECT_EQ(session.BeginFrame().state, SessionState::Stopping);
    EXPECT_FALSE(session.Running());
}

TEST(SessionTest, ScriptsDriveTheHeadControllersAndSessionLoss) {
    SessionDesc desc = Simulated();
    desc.simulated.script = [](uint64_t frame, double, SimulatedInput& input) {
        input.head.position = {1, 2, 1.5};
        input.Press(Side::Right, "trigger/value", 0.8f);
        input.Press(Side::Left, "thumbstick/y", -1);
        input.visible = frame != 1;
        input.loseSession = frame == 3;
    };
    Session session(desc);
    Frame frame = session.BeginFrame();
    EXPECT_EQ(frame.head.position, (Double3{1, 2, 1.5}));
    EXPECT_FLOAT_EQ(frame.Controller(Side::Right).trigger, 0.8f);
    EXPECT_FLOAT_EQ(frame.Controller(Side::Left).thumbstick.y, -1);
    EXPECT_EQ(frame.profiles[1], "/interaction_profiles/oculus/touch_controller");
    frame = session.BeginFrame();
    EXPECT_EQ(frame.state, SessionState::Synchronized);
    EXPECT_FALSE(frame.Controller(Side::Right).active);
    EXPECT_TRUE(session.BeginFrame().shouldRender);
    EXPECT_EQ(session.BeginFrame().state, SessionState::LossPending);
    EXPECT_FALSE(session.Running());

    SimulatedInput input;
    input.profile = "/interaction_profiles/oculus/touch_controller";
    EXPECT_THROW(input.Press(Side::Left, "a/click"), Error);
}

TEST(SessionTest, BindingLayersHideTheBindingsBelowThem) {
    SessionDesc desc = Simulated();
    desc.simulated.script = [](uint64_t frame, double, SimulatedInput& input) {
        if (frame == 1 || frame == 2) {
            input.Press(Side::Right, "trigger/value", frame == 1 ? 1.0f : 0.45f);
        }
        if (frame == 4) {
            input.Press(Side::Right, "a/click");
        }
    };
    Session session(desc);
    session.PushBindings("base", {{.action = "select", .input = Input::Trigger}});
    EXPECT_FALSE(session.BeginFrame().Held("select"));
    Frame frame = session.BeginFrame();
    EXPECT_TRUE(frame.Pressed("select"));
    ASSERT_EQ(frame.events.size(), 1u);
    EXPECT_EQ(frame.events[0].hand, Side::Right);
    frame = session.BeginFrame();
    EXPECT_TRUE(frame.Held("select"));
    EXPECT_FALSE(frame.Pressed("select"));
    EXPECT_TRUE(session.BeginFrame().Released("select"));

    session.PushBindings("menu", {{.action = "select", .input = Input::Primary}});
    EXPECT_EQ(session.BindingLayers(), (std::vector<std::string>{"base", "menu"}));
    EXPECT_TRUE(session.BeginFrame().Pressed("select"));
    session.PopBindings("menu");
    EXPECT_EQ(session.BindingLayers(), (std::vector<std::string>{"base"}));
    EXPECT_TRUE(session.BeginFrame().Released("select"));
}

TEST(SessionTest, FiltersOffsetAndSmoothControllerPoses) {
    SessionDesc desc = Simulated();
    desc.simulated.placeControllers = true;
    desc.simulated.script = [](uint64_t frame, double, SimulatedInput& input) {
        input.controllers[1].pose.position.x = frame == 0 ? 0.0 : 1.0;
    };
    Session session(desc);
    session.SetPoseFilter(Side::Right,
                          {.offset = {{0, 0, 0}, {0, 0, 0, 1}, true}, .smoothing = 0.1f});
    EXPECT_FLOAT_EQ(session.GetPoseFilter(Side::Right).smoothing, 0.1f);
    EXPECT_NEAR(session.BeginFrame().Controller(Side::Right).pose.position.x, 0, 1e-9);
    const Controller moved = session.BeginFrame().Controller(Side::Right);
    const double share = 1 - std::exp(-(1.0 / 90) / 0.1);
    EXPECT_NEAR(moved.pose.position.x, share, 1e-6);
    EXPECT_NEAR(moved.rawPose.position.x, 1, 1e-9);
}

TEST(SessionTest, SwitchesOnlyToOfferedRefreshRates) {
    Session session(Simulated());
    EXPECT_TRUE(session.SetRefreshRate(120));
    EXPECT_FALSE(session.SetRefreshRate(60));
    EXPECT_NEAR(session.BeginFrame().deltaTime, 1.0 / 120, 1e-6);
    EXPECT_FLOAT_EQ(session.Status().refreshRate, 120);

    SessionDesc desc = Simulated();
    desc.refreshRate = 60;
    EXPECT_THROW(Session{desc}, Error);
    desc.simulated.offers.refreshRates = false;
    Session fixed(desc);
    EXPECT_FALSE(fixed.SetRefreshRate(120));
    EXPECT_FLOAT_EQ(fixed.Status().refreshRate, 90);
}

TEST(SessionTest, ReportsOnlyFeaturesTheHeadsetOffers) {
    SessionDesc desc = Simulated();
    desc.features = {Feature::HandTracking, Feature::EyeGaze};
    desc.simulated.offers.eyeGaze = false;
    desc.simulated.script = [](uint64_t, double, SimulatedInput& input) {
        input.PoseHand(Side::Left, HandPose::Pinch);
        input.gaze = input.head;
    };
    Session session(desc);
    const Frame frame = session.BeginFrame();
    EXPECT_TRUE(frame.Has(Feature::HandTracking));
    EXPECT_FALSE(frame.Has(Feature::EyeGaze));
    EXPECT_FALSE(frame.gaze.has_value());
    const HandJoints& hand = frame.Hand(Side::Left);
    ASSERT_TRUE(hand.active);
    EXPECT_EQ(hand[HandJoint::ThumbTip].pose.position, hand[HandJoint::IndexTip].pose.position);
}

TEST(SessionTest, CountsMissedRefreshesAndWritesTimings) {
    SessionDesc desc = Simulated();
    desc.simulated.script = [](uint64_t frame, double, SimulatedInput& input) {
        input.skipRefreshes = frame == 1 ? 2 : 0;
    };
    Session session(desc);
    session.BeginFrame();
    const Frame skipped = session.BeginFrame();
    EXPECT_NEAR(skipped.deltaTime, 3.0 / 90, 1e-6);
    const SessionStats stats = session.Stats();
    EXPECT_EQ(stats.frames, 2u);
    EXPECT_EQ(stats.missed, 2u);
    EXPECT_NEAR(stats.period, 1.0 / 90, 1e-6);

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "ernie_xr_timings.csv";
    session.WriteTimings(path);
    std::ifstream file(path);
    std::string header;
    std::getline(file, header);
    EXPECT_EQ(header.rfind("index,displayTime,wait", 0), 0u);
    file.close();
    std::filesystem::remove(path);
}

}
}
