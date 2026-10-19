// SPDX-License-Identifier: BSD-3-Clause
/// @file timeline_test.cpp
/// @brief Tests of the timeline: playing in real time or a frame a step, looping or stopping,
/// stepping and seeking the scene's clock.
#include "ernie/app/timeline.h"

#include <gtest/gtest.h>

namespace ernie {
namespace {

TEST(Timeline, PlaysInRealTimeAndLoops) {
    Scene scene;
    app::Timeline timeline;
    timeline.SetRange(0.0, 48.0);
    timeline.SetFramesPerSecond(24.0);
    EXPECT_EQ(timeline.Advance(scene, 1.0), 0.0) << "paused";
    timeline.Play();
    EXPECT_NEAR(timeline.Advance(scene, 1.0), 1.0, 1e-12);
    EXPECT_NEAR(scene.Time(), 24.0, 1e-12) << "a second is 24 frames";
    timeline.Advance(scene, 1.5);
    EXPECT_NEAR(scene.Time(), 12.0, 1e-12) << "60 frames loop to 12";
    timeline.looping = false;
    timeline.Advance(scene, 10.0);
    EXPECT_EQ(scene.Time(), 48.0);
    EXPECT_FALSE(timeline.Playing()) << "stops at the end";
}

TEST(Timeline, StepsFramesAndPlaysEveryFrame) {
    Scene scene;
    app::Timeline timeline;
    timeline.SetRange(10.0, 20.0);
    timeline.Seek(scene, 10.0);
    timeline.StepFrames(scene, 3);
    EXPECT_EQ(scene.Time(), 13.0);
    timeline.StepFrames(scene, -50);
    EXPECT_EQ(scene.Time(), 10.0) << "within the range";
    timeline.everyFrame = true;
    timeline.Play();
    timeline.Advance(scene, 5.0);
    EXPECT_EQ(scene.Time(), 11.0) << "a frame, whatever the time";
    timeline.Stop(scene);
    EXPECT_EQ(scene.Time(), 10.0);
    timeline.SetRange(5.0, 5.0);
    EXPECT_EQ(timeline.End(), 6.0) << "at least a frame long";
}

}  // namespace
}  // namespace ernie
