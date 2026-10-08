#include "ernie/session/frame_params.h"

#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

using ernie::Camera;
using ernie::FrameParams;

namespace {

void ExpectNear(const glm::dmat4& a, const glm::dmat4& b, double tolerance = 1e-9) {
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            EXPECT_NEAR(a[c][r], b[c][r], tolerance) << "column " << c << ", row " << r;
        }
    }
}

}  // namespace

TEST(Shutter, SubframesSpanTheShutterAndReadout) {
    Camera camera;
    EXPECT_EQ(camera.Subframes(), 1) << "closed by default";
    EXPECT_EQ(camera.SubframeTime(0), 0.0);
    camera.shutterOpen = -0.5;
    camera.shutterSamples = 5;
    EXPECT_EQ(camera.Subframes(), 5);
    EXPECT_DOUBLE_EQ(camera.SubframeTime(0), -0.5);
    EXPECT_DOUBLE_EQ(camera.SubframeTime(4), 0.0);
    EXPECT_DOUBLE_EQ(camera.SubframeTime(2), -0.25);
    // A rolling readout opens the top row earlier.
    camera.rollingShutter = 0.5;
    EXPECT_DOUBLE_EQ(camera.SubframeTime(0), -1.0);
    EXPECT_DOUBLE_EQ(camera.SubframeTime(4), 0.0);
    // One sample still blurs: at least two sub-frames.
    camera.shutterSamples = 1;
    EXPECT_EQ(camera.Subframes(), 2);

    camera.shutterClose = -0.75;
    EXPECT_THROW(camera.Check(), std::invalid_argument) << "closes before it opens";
    camera.shutterClose = 0.0;
    camera.rollingShutter = -1.0;
    EXPECT_THROW(camera.Check(), std::invalid_argument);
}

TEST(Shutter, PosesInterpolateAlongTheArcAndLine) {
    const glm::dmat4 from = glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 0.0, 1.0));
    const glm::dmat4 to = glm::rotate(glm::translate(glm::dmat4(1.0), glm::dvec3(4.0, 0.0, 1.0)),
                                      M_PI / 2.0, glm::dvec3(0.0, 0.0, 1.0));
    ExpectNear(ernie::InterpolatePose(from, to, 0.0), from);
    ExpectNear(ernie::InterpolatePose(from, to, 1.0), to);
    ExpectNear(ernie::InterpolatePose(from, to, 0.5),
               glm::rotate(glm::translate(glm::dmat4(1.0), glm::dvec3(2.0, 0.0, 1.0)), M_PI / 4.0,
                           glm::dvec3(0.0, 0.0, 1.0)));
    // Past the ends it extrapolates.
    ExpectNear(ernie::InterpolatePose(from, to, 2.0),
               glm::rotate(glm::translate(glm::dmat4(1.0), glm::dvec3(8.0, 0.0, 1.0)), M_PI,
                           glm::dvec3(0.0, 0.0, 1.0)));
}

TEST(Shutter, SubframesExpandEachViewAtItsTimes) {
    FrameParams frame;
    frame.camera.shutterOpen = -1.0;
    frame.camera.shutterSamples = 3;
    const glm::dmat4 a = glm::translate(glm::dmat4(1.0), glm::dvec3(2.0, 0.0, 0.0));
    const glm::dmat4 b = glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 2.0, 0.0));
    frame.poses = {a, b};
    frame.worlds = {3, 4};
    frame.previousPoses = {glm::dmat4(1.0), glm::dmat4(1.0)};
    const FrameParams subframes = frame.Subframes();
    ASSERT_EQ(subframes.Views(), 6u);
    EXPECT_EQ(subframes.worlds, (std::vector<std::int32_t>{3, 4, 3, 4, 3, 4}));
    ExpectNear(subframes.poses[0], glm::dmat4(1.0));
    ExpectNear(subframes.poses[3],
               glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 1.0, 0.0)));
    ExpectNear(subframes.poses[4], a);
    // Without a last frame the camera stands still.
    frame.previousPoses.clear();
    EXPECT_EQ(frame.Subframes().poses, (std::vector<glm::dmat4>{a, b, a, b, a, b}));
    frame.previousPoses = {glm::dmat4(1.0)};
    EXPECT_THROW(frame.Subframes(), std::invalid_argument);
    // A closed shutter keeps the frame.
    frame.camera.shutterOpen = 0.0;
    EXPECT_EQ(frame.Subframes().Views(), 2u);
}
