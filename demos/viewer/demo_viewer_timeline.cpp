// SPDX-License-Identifier: BSD-3-Clause
// A moon round a planet and a swinging pendulum posed from the
// stage's time code, so the timeline plays, pauses, scrubs, steps and
// loops them (Space plays; the Capture panel records the loop).

#include "ernie/app/demo.h"
#include "ernie/app/viewer.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/stage_schemas.h"
#include "ernie/scene/xform.h"
#include "util.h"

#include <glm/glm.hpp>

#include <cmath>
#include <numbers>

namespace {

using namespace ernie;
using demos::Check;

constexpr double kFps = 24.0;
constexpr double kLoop =
    10.0;  // s, the range; every period divides it
constexpr double kOrbit = 4.0;  // m, the planet round the origin
constexpr double kOrbitPeriod = kLoop;
constexpr double kMoon = 1.2;  // m, the moon round the planet
constexpr double kMoonPeriod = kLoop / 4.0;
const glm::dvec3 kPivot{-7.0, 0.0, 6.0};  // where the pendulum hangs
constexpr double kArm = 3.5;              // m
constexpr double kSwing = 40.0;           // degrees
constexpr double kSwingPeriod = kLoop / 2.0;

/// Where everything is `t` seconds into the loop.
struct Pose {
    glm::dvec3 planet;
    glm::dvec3 moon;
    double swing;  ///< The pendulum's angle, degrees.
};

Pose PoseAt(double t) {
    constexpr double kTau = 2.0 * std::numbers::pi;
    const double a = kTau * t / kOrbitPeriod;
    const glm::dvec3 planet(kOrbit * std::cos(a),
                            kOrbit * std::sin(a), 1.0);
    const double b = kTau * t / kMoonPeriod;
    return {planet,
            planet +
                kMoon * glm::dvec3(std::cos(b), std::sin(b), 0.0),
            kSwing * std::cos(kTau * t / kSwingPeriod)};
}

class TimelineDemo : public app::Demo {
  public:
    TimelineDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 240;
        Scene& scene = GetScene();
        // Ten seconds at 24 frames a second: the timeline's range.
        Stage(scene)
            .SetTimeCodesPerSecond(kFps)
            .SetFramesPerSecond(kFps)
            .SetStartTimeCode(0.0)
            .SetEndTimeCode(kLoop * kFps)
            .SetTimeCode(0.0);
        Xform::Define(scene, "/World");
        GroundPlane::Define(scene, "/World/Ground").SetChecker(true);
        DistantLight::Define(scene, "/World/Sun")
            .SetTransform(DistantLightTransform({0.3, -0.5, 0.8}));
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(20.0)
            .SetTransform(LookAtTransform({-2.0, -17.0, 8.0},
                                          {-2.0, 0.0, 2.0},
                                          {0.0, 0.0, 1.0}));
        Sphere::Define(scene, "/World/Planet")
            .SetRadius(0.8)
            .SetDisplayColor({0.2, 0.4, 0.9});
        Sphere::Define(scene, "/World/Moon")
            .SetRadius(0.3)
            .SetDisplayColor({0.8, 0.8, 0.75});
        Cylinder::Define(scene, "/World/Rod")
            .SetRadius(0.05)
            .SetHeight(kArm);
        Sphere::Define(scene, "/World/Bob")
            .SetRadius(0.5)
            .SetDisplayColor({0.9, 0.5, 0.1});
        Place(0.0);
        if (viewer != nullptr) {
            // The timeline drives the poses: after it moves on,
            // before the frame renders.
            viewer->OnFrame(
                [this](app::Viewer&, const app::FrameInfo& info) {
                    Place(info.timeCode / kFps);
                });
            viewer->GetTimeline().SetLooping(true);
            viewer->GetTimeline().Play();
        }
    }

    void Step() override {
        if (GetViewer() != nullptr) {
            return;
        }
        // Without a viewer, the demo plays the timeline itself.
        Stage stage(GetScene());
        const double code = stage.GetTimeCode() + frameDt * kFps;
        stage.SetTimeCode(std::fmod(code, kLoop * kFps));
        Place(stage.GetTimeCode() / kFps);
    }

    void TestPostStep() override {
        const double t = Stage(GetScene()).GetTimeCode() / kFps;
        const Pose pose = PoseAt(t);
        const glm::dvec3 planet(
            GetScene().WorldTransform("/World/Planet")[3]);
        Check(glm::distance(planet, pose.planet) < 1e-9,
              "the planet is where the time code puts it");
        Check(std::abs(glm::distance(pose.planet, pose.moon) -
                       kMoon) < 1e-9,
              "the moon keeps its orbit");
    }

    void TestFinal() override {
        const double code = Stage(GetScene()).GetTimeCode();
        Check(code >= 0.0 && code < kLoop * kFps,
              "the time code stays in range");
        // The loop is seamless: its end poses as its start.
        const Pose start = PoseAt(0.0);
        const Pose end = PoseAt(kLoop);
        Check(glm::distance(start.planet, end.planet) < 1e-9 &&
                  glm::distance(start.moon, end.moon) < 1e-9 &&
                  std::abs(start.swing - end.swing) < 1e-9,
              "the loop ends as it starts");
    }

  private:
    void Place(double t) {
        Scene& scene = GetScene();
        const Pose pose = PoseAt(t);
        SetXform(scene, "/World/Planet", {.translate = pose.planet});
        SetXform(scene, "/World/Moon", {.translate = pose.moon});
        const double theta = glm::radians(pose.swing);
        const glm::dvec3 down(std::sin(theta), 0.0, -std::cos(theta));
        SetXform(scene, "/World/Bob",
                 {.translate = kPivot + kArm * down});
        // The rod, an upright cylinder, tilted along the arm.
        SetXform(scene, "/World/Rod",
                 {.translate = kPivot + 0.5 * kArm * down,
                  .rotate = glm::dvec3(0.0, -pose.swing, 0.0)});
    }
};

}  // namespace

ERNIE_REGISTER_DEMO(TimelineDemo, "viewer", "timeline",
                    "A moon, a planet and a pendulum posed from the "
                    "timeline's time code: play, scrub, step and "
                    "loop.");
