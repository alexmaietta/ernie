// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/timeline.h"
#include "ernie/scene/stage_schemas.h"

#include <gtest/gtest.h>

using ernie::Scene;
using ernie::Stage;
using ernie::app::Timeline;

namespace {

/// A stage over time codes 10 to 20 at 24 time codes and frames a
/// second.
Scene RangeScene() {
    Scene scene;
    Stage(scene).SetStartTimeCode(10.0).SetEndTimeCode(20.0);
    return scene;
}

}  // namespace

TEST(Timeline, RangeIsTheStagesOrAHundredTimeCodesFromItsStart) {
    Scene scene;
    EXPECT_EQ(Timeline::Range(scene), std::make_pair(0.0, 100.0));
    scene = RangeScene();
    EXPECT_EQ(Timeline::Range(scene), std::make_pair(10.0, 20.0));
}

TEST(Timeline, PausedItHoldsAndPlayingItRunsInRealTime) {
    Scene scene = RangeScene();
    Timeline timeline;
    EXPECT_EQ(timeline.Advance(scene, 0.5), 0.0);
    EXPECT_EQ(Timeline::TimeCode(scene), 0.0) << "held while paused";

    timeline.Play();
    // From outside the range, its start, then on in real time.
    EXPECT_DOUBLE_EQ(timeline.Advance(scene, 0.25), 0.25);
    EXPECT_DOUBLE_EQ(Timeline::TimeCode(scene), 16.0);
    timeline.Pause();
    timeline.Advance(scene, 1.0);
    EXPECT_DOUBLE_EQ(Timeline::TimeCode(scene), 16.0);
}

TEST(Timeline, LoopsOrStopsAtTheEnd) {
    Scene scene = RangeScene();
    Timeline timeline;
    timeline.Seek(scene, 18.0);
    timeline.Play();
    timeline.Advance(scene, 0.25);  // 6 time codes: 24 wraps to 14
    EXPECT_DOUBLE_EQ(Timeline::TimeCode(scene), 14.0);
    EXPECT_TRUE(timeline.IsPlaying());

    timeline.SetLooping(false);
    timeline.Seek(scene, 18.0);
    timeline.Advance(scene, 0.25);
    EXPECT_DOUBLE_EQ(Timeline::TimeCode(scene), 20.0);
    EXPECT_FALSE(timeline.IsPlaying()) << "stopped at the end";
}

TEST(Timeline, PlaysEveryFrameAFrameAStep) {
    Scene scene = RangeScene();
    Stage(scene).SetTimeCodesPerSecond(48.0);  // two codes a frame
    Timeline timeline;
    timeline.SetPlayEveryFrame(true);
    timeline.Seek(scene, 10.0);
    timeline.Play();
    EXPECT_DOUBLE_EQ(timeline.Advance(scene, 3.0), 1.0 / 24.0)
        << "a frame however long the step took";
    EXPECT_DOUBLE_EQ(Timeline::TimeCode(scene), 12.0);
}

TEST(Timeline, StepsFramesWithinTheRangeAndStopsAtItsStart) {
    Scene scene = RangeScene();
    Timeline timeline;
    timeline.Seek(scene, 19.0);
    timeline.Play();
    timeline.StepFrames(scene, 3);
    EXPECT_FALSE(timeline.IsPlaying()) << "stepping pauses";
    EXPECT_DOUBLE_EQ(Timeline::TimeCode(scene), 20.0);
    timeline.StepFrames(scene, -2);
    EXPECT_DOUBLE_EQ(Timeline::TimeCode(scene), 18.0);

    timeline.Play();
    timeline.Stop(scene);
    EXPECT_FALSE(timeline.IsPlaying());
    EXPECT_DOUBLE_EQ(Timeline::TimeCode(scene), 10.0);
}
