// SPDX-License-Identifier: BSD-3-Clause
/// @file demo_basics_bounce.cpp
/// @brief A ball dropped on the ground, bouncing under gravity, each bounce lower.
#include "ernie/app/demo.h"
#include "ernie/scene/schema/camera.h"
#include "ernie/scene/schema/geom.h"
#include "ernie/scene/schema/planet.h"
#include "ernie/scene/schema/render.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"

#include <imgui.h>

namespace ernie::app {
namespace {

class Bounce final : public Demo {
  public:
    explicit Bounce(Scene& scene) : Demo(scene) {
        description = "A ball dropped from 4 m bounces on the ground, losing a share of its speed "
                      "each bounce.";
        GroundPlane::Define(scene, "/World/Ground");
        Sphere::Define(scene, "/World/Ball").SetRadius(kRadius).SetDisplayColor({0.85, 0.35, 0.2});
        Camera::Define(scene, "/World/Camera");
        scene.Set("/World/Camera", tokens::kTransform,
                  LookAtTransform({0.0, -12.0, 4.0}, {0.0, 0.0, 2.0}, {0.0, 0.0, 1.0}));
        RenderSettings::Define(scene, "/Render/Settings").SetCamera("/World/Camera");
        Place();
    }

    void Step() override {
        velocity_ -= kGravity * fixedDt;
        height_ += velocity_ * fixedDt;
        if (height_ < kRadius) {
            height_ = kRadius;
            velocity_ = -velocity_ * restitution_;
        }
        Place();
    }

    void Gui() override {
        ImGui::SliderFloat("Restitution", &restitution_, 0.0f, 1.0f);
        if (ImGui::Button("Drop Again")) {
            height_ = 4.0;
            velocity_ = 0.0;
        }
    }

    void TestFinal() override {
        if (height_ < kRadius - 1e-9) {
            throw std::runtime_error("The ball fell through the ground");
        }
    }

  private:
    static constexpr double kRadius = 0.5;
    static constexpr double kGravity = 9.81;

    void Place() {
        SetXform(GetScene(), "/World/Ball", {.translate = glm::dvec3(0.0, 0.0, height_)});
    }

    double height_ = 4.0;
    double velocity_ = 0.0;
    float restitution_ = 0.8f;
};

}  // namespace

ERNIE_REGISTER_DEMO(Bounce, "basics", "bounce", "A ball bouncing on the ground, lower each time.");

}  // namespace ernie::app
