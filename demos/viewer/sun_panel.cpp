// SPDX-License-Identifier: BSD-3-Clause
// Opens the viewer on a scene built in code, with options in the
// Demos panel that move the sun. The C++ twin of sun_panel.py.
//
//     ernie --demo viewer/sun_panel [--viewer null --output sun.png]

#include "ernie/app/demo.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/xform.h"
#include "util.h"

#include <glm/trigonometric.hpp>
#include <imgui.h>

#include <cmath>
#include <string>

namespace {

using namespace ernie;

class SunPanelDemo : public app::Demo {
  public:
    SunPanelDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        GroundPlane::Define(scene, "/World/Ground");
        for (int i = 0; i < 5; ++i) {
            const Path box("/World/Box" + std::to_string(i));
            Cube::Define(scene, box)
                .SetSize(1.0 + 0.25 * i)
                .SetDisplayColor(
                    {0.2 + 0.15 * i, 0.4, 0.8 - 0.15 * i});
            SetXform(
                scene, box,
                {.translate = glm::dvec3(3.0 * (i - 2), 0.0, 1.0)});
        }
        DistantLight::Define(scene, "/World/Sun");
        // The viewer frames the scene with its own camera; this one
        // renders it without a viewer
        // (--viewer null, --benchmark) and for --test.
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(24.0)
            .SetTransform(LookAtTransform(
                {0.0, -14.0, 6.0}, {0.0, 0.0, 0.5}, {0.0, 0.0, 1.0}));
        AimSun();
    }

    // The demo's options, in the Demos panel: sliders drawn with Dear
    // ImGui that move the sun, which the next frame renders.
    void Gui() override {
        bool changed =
            ImGui::SliderFloat("Elevation", &elevation_, 1.0f, 90.0f);
        changed |=
            ImGui::SliderFloat("Azimuth", &azimuth_, 0.0f, 360.0f);
        if (changed) {
            AimSun();
        }
    }

    void TestFinal() override {
        demos::Check(demos::ColourDeviation(RenderImage()) > 1.0,
                     "the boxes and ground should not render flat");
    }

  private:
    void AimSun() {
        const double e =
            glm::radians(static_cast<double>(elevation_));
        const double a = glm::radians(static_cast<double>(azimuth_));
        DistantLight(GetScene(), "/World/Sun")
            .SetTransform(DistantLightTransform(
                {std::cos(e) * std::cos(a), std::cos(e) * std::sin(a),
                 std::sin(e)}));
    }

    float elevation_ = 45.0f;
    float azimuth_ = 30.0f;
};

}  // namespace

ERNIE_REGISTER_DEMO(
    SunPanelDemo, "viewer", "sun panel",
    "A scene built in code, with options that move the sun.");
