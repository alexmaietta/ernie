// SPDX-License-Identifier: BSD-3-Clause
// A bouncing ball whose height, speed and energy are logged each step
// as live plots (LogScalar).

#include "ernie/app/demo.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/xform.h"
#include "util.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr double kGravity = 9.81;
constexpr double kRadius = 0.5;
constexpr double kDrop = 4.0;  // metres, the ball's lowest point
constexpr double kRestitution = 0.8;

class LogScalarDemo : public app::Demo {
  public:
    LogScalarDemo(app::Viewer* viewer,
                  const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 240;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        GroundPlane::Define(scene, "/World/Ground")
            .SetChecker(true)
            .SetCheckerSize(1.0);
        DistantLight::Define(scene, "/World/Sun")
            .SetTransform(DistantLightTransform({0.4, -0.6, 0.7}));
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(24.0)
            .SetTransform(LookAtTransform(
                {0.0, -12.0, 4.0}, {0.0, 0.0, 2.0}, {0.0, 0.0, 1.0}));
        Sphere::Define(scene, "/World/Ball")
            .SetRadius(kRadius)
            .SetDisplayColor({0.9, 0.35, 0.2});
        Place();
        Widgets().Slider(
            "Restitution", 0.1, 1.0, kRestitution,
            [this](double value) { restitution_ = value; });
        Widgets().Button("Drop again", [this] {
            height_ = kDrop;
            velocity_ = 0.0;
        });
    }

    void Step() override {
        // Semi-implicit Euler, the ball bouncing off the ground with
        // the restitution.
        velocity_ -= kGravity * frameDt;
        height_ += velocity_ * frameDt;
        if (height_ < 0.0) {
            height_ = 0.0;
            velocity_ = -velocity_ * restitution_;
        }
        Place();
        LogScalar("ball/height", height_);
        LogScalar("ball/speed", std::abs(velocity_));
        LogScalar("ball/energy", Energy());
    }

    void TestFinal() override {
        const std::vector<double> heights =
            LoggedScalars("ball/height");
        const std::vector<double> energies =
            LoggedScalars("ball/energy");
        Check(heights.size() == frame, "a height every step");
        Check(*std::min_element(heights.begin(), heights.end()) >=
                      0.0 &&
                  *std::max_element(heights.begin(), heights.end()) <=
                      kDrop,
              "the ball stays above the ground");
        int bounces = 0;
        for (std::size_t i = 1; i < heights.size(); ++i) {
            bounces +=
                heights[i - 1] > 0.0 && heights[i] == 0.0 ? 1 : 0;
        }
        Check(bounces >= 2, "the ball should bounce at least twice");
        Check(energies.back() < 0.75 * energies.front(),
              "each bounce should lose energy");
        for (const double energy : energies) {
            Check(std::isfinite(energy),
                  "the energy should stay finite");
        }
    }

  private:
    void Place() {
        SetXform(
            GetScene(), "/World/Ball",
            {.translate = glm::dvec3(0.0, 0.0, height_ + kRadius)});
    }

    /// The ball's energy per unit mass as a fraction of the drop's.
    double Energy() const {
        return (kGravity * height_ + 0.5 * velocity_ * velocity_) /
               (kGravity * kDrop);
    }

    double height_ = kDrop;
    double velocity_ = 0.0;
    double restitution_ = kRestitution;
};

}  // namespace

ERNIE_REGISTER_DEMO(LogScalarDemo, "viewer", "log scalar",
                    "A bouncing ball's height, speed and energy "
                    "logged as live plots.");
