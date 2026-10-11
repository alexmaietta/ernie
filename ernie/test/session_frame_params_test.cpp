// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/session/frame_params.h"

#include "ernie/scene/wide_view.h"

#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

using ernie::CameraParams;
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
    CameraParams camera;
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
    ExpectNear(subframes.poses[3], glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 1.0, 0.0)));
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

TEST(Pushbroom, StripRunsFromTheLastPoseToThisOne) {
    FrameParams params;
    params.camera.width = 100;
    params.camera.height = 50;
    params.camera.pushbroom = true;
    params.camera.lineOffsetMm = 5.0;
    params.camera.transform = glm::translate(glm::dmat4(1.0), glm::dvec3(10.0, 20.0, 30.0));
    params.previousPoses = {glm::translate(glm::dmat4(1.0), glm::dvec3(10.0, 0.0, 30.0))};
    params.origin = glm::dvec3(10.0, 10.0, 0.0);
    EXPECT_NO_THROW(params.camera.Check());
    const ernie::PushbroomStrip strip = params.Strip();
    EXPECT_EQ(strip.from, glm::vec3(0.0f, -10.0f, 30.0f)) << "relative to the origin";
    EXPECT_EQ(strip.to, glm::vec3(0.0f, 10.0f, 30.0f));
    EXPECT_FLOAT_EQ(strip.line, 0.1f) << "5 mm over the 50 mm focal length";
    EXPECT_NEAR(strip.scale, 2.0 * params.camera.FocalLengthPixels() / 100.0, 1e-5);
    EXPECT_NEAR(strip.shift, 0.0f, 1e-6f);
    const glm::mat4 packed = strip.Packed();
    EXPECT_EQ(glm::vec3(packed[1]), strip.to);
    EXPECT_FLOAT_EQ(packed[1][3], strip.farPlane);
    EXPECT_FLOAT_EQ(packed[2][3], 1.0f) << "the identity rotation's w";

    params.camera.shutterOpen = -0.5;
    EXPECT_THROW(params.camera.Check(), std::invalid_argument) << "rows are its readout";
}

TEST(Pushbroom, EnvelopeHoldsEveryRayOfTheStrip) {
    CameraParams camera;
    camera.width = 64;
    camera.height = 64;
    camera.pushbroom = true;
    camera.lineOffsetMm = 2.0;
    camera.nearPlane = 1.0;
    camera.farPlane = 500.0;
    // Moving 40 along y while turning 10 degrees about it.
    const glm::dmat4 from = glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, -20.0, 100.0));
    const glm::dmat4 to = glm::rotate(glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 20.0, 100.0)),
                                      glm::radians(10.0), glm::dvec3(0.0, 1.0, 0.0));
    const ernie::CameraView envelope = ernie::PushbroomEnvelope(camera, from, to);
    const glm::dmat4 viewProjection(envelope.ViewProjection());
    const double f = camera.FocalLengthPixels();
    const double line = camera.lineOffsetMm / camera.focalLengthMm;
    int inside = 0;
    for (int i = 0; i <= 40; ++i) {
        const glm::dmat4 pose = ernie::InterpolatePose(from, to, i / 40.0);
        for (int column = 0; column <= 64; column += 8) {
            const double x = (column - 32.0) / f;
            for (const double depth : {1.5, 20.0, 250.0, 499.0}) {
                const glm::dvec4 world = pose * glm::dvec4(x * depth, line * depth, -depth, 1.0);
                const glm::dvec4 clip = viewProjection * world;
                EXPECT_GT(clip.w, 0.0);
                const glm::dvec3 ndc = glm::dvec3(clip) / clip.w;
                EXPECT_LE(std::abs(ndc.x), 1.0) << i << ", " << column << ", " << depth;
                EXPECT_LE(std::abs(ndc.y), 1.0) << i << ", " << column << ", " << depth;
                EXPECT_GE(ndc.z, 0.0);
                EXPECT_LE(ndc.z, 1.0);
                ++inside;
            }
        }
    }
    EXPECT_EQ(inside, 41 * 9 * 4);
}

TEST(FrameParams, CubeFacesTurnEachViewOntoItsFaces) {
    ernie::FrameParams frame;
    frame.camera.width = 640;
    frame.camera.height = 480;
    frame.camera.lens.projection = ernie::LensProjection::Fisheye;
    frame.poses = {glm::translate(glm::dmat4(1.0), glm::dvec3(1.0, 2.0, 3.0)),
                   glm::rotate(glm::dmat4(1.0), 0.5, glm::dvec3(0.0, 0.0, 1.0))};
    frame.worlds = {4, 7};
    const ernie::FrameParams faces = frame.Faces(ernie::kCubeFaces);
    ASSERT_EQ(faces.Views(), 2u * ernie::kCubeFaces);
    EXPECT_EQ(faces.camera.lens.projection, ernie::LensProjection::Perspective);
    // A centred square pinhole: 90 degrees plus the margin across the image's height.
    const glm::dvec3 face = ernie::FrameParams::FaceIntrinsics(640, 480);
    EXPECT_NEAR(faces.camera.FocalLengthPixels(), face.x, 1e-9);
    EXPECT_EQ(faces.camera.PrincipalPoint(), glm::dvec2(320.0, 240.0));
    EXPECT_NEAR(240.0 / face.x, ernie::CubeFaceExtent(480.0), 1e-12);
    for (std::size_t v = 0; v < 2; ++v) {
        for (int f = 0; f < ernie::kCubeFaces; ++f) {
            const std::size_t i = v * ernie::kCubeFaces + static_cast<std::size_t>(f);
            const glm::dmat4& pose = faces.poses[i];
            // The view's position, looking along its face's direction.
            EXPECT_EQ(glm::dvec3(pose[3]), glm::dvec3(frame.poses[v][3]));
            const glm::dvec3 look = -glm::dvec3(pose[2]);
            const glm::dvec3 expected = glm::dmat3(frame.poses[v]) *
                                        -ernie::CubeFaceAxes(static_cast<ernie::CubeFace>(f))[2];
            EXPECT_NEAR(glm::distance(look, expected), 0.0, 1e-12) << v << ", " << f;
            EXPECT_EQ(faces.ViewWorld(i), frame.worlds[v]);
        }
    }
    EXPECT_EQ(frame.Faces(1).Views(), 2u);
    EXPECT_THROW(frame.Faces(3), std::invalid_argument);
    EXPECT_EQ(frame.Faces(ernie::kAllCubeFaces).Views(), 2u * ernie::kAllCubeFaces);
    // Posed on the GPU, the faces turn the host's estimates (views.slang turns the poses).
    frame.posesOnGpu = true;
    EXPECT_EQ(frame.Faces(ernie::kCubeFaces).Views(), 2u * ernie::kCubeFaces);
}
