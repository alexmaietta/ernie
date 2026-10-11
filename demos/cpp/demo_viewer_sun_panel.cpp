// SPDX-License-Identifier: BSD-3-Clause
// Opens the viewer on a scene built in code, with a panel of its own that moves the sun: a C++
// demo on ernie::app::Demo, taking the options every demo takes (--viewer null, --headless,
// --test, --output, --benchmark...).
//
//     ./build/bin/demo_viewer_sun_panel [--viewer null --output sun.png]

#include "ernie/app/demo.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"

#include <glm/trigonometric.hpp>
#include <imgui.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

using namespace ernie;

class SunPanelDemo : public app::Demo {
  public:
    SunPanelDemo(app::Viewer* viewer, const app::DemoOptions& options) : Demo(viewer, options) {
        numFrames = 1;
        description = "The viewer on a scene built in code, with a panel that moves the sun.";
        Scene& scene = GetScene();
        GroundPlane::Define(scene, "/World/Ground");
        for (int i = 0; i < 5; ++i) {
            const Path box("/World/Box" + std::to_string(i));
            Cube::Define(scene, box)
                .SetSize(1.0 + 0.25 * i)
                .SetDisplayColor({0.2 + 0.15 * i, 0.4, 0.8 - 0.15 * i});
            SetXform(scene, box, {.translate = glm::dvec3(3.0 * (i - 2), 0.0, 1.0)});
        }
        DistantLight::Define(scene, "/World/Sun");
        // The viewer frames the scene with its own camera; this one renders it without a viewer
        // (--viewer null, --benchmark) and for --test.
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(24.0)
            .SetTransform(LookAtTransform({0.0, -14.0, 6.0}, {0.0, 0.0, 0.5}, {0.0, 0.0, 1.0}));
        AimSun();
        if (viewer != nullptr) {
            // Panels draw with Dear ImGui and edit the scene, which the next frame renders. The
            // demo registers in a scope of its own, so switching demos removes the panel.
            viewer->AddPanel("Sun", app::Dock::Left, [this](app::Viewer&) {
                bool changed = ImGui::SliderFloat("Elevation", &elevation_, 1.0f, 90.0f);
                changed |= ImGui::SliderFloat("Azimuth", &azimuth_, 0.0f, 360.0f);
                if (changed) {
                    AimSun();
                }
            });
        }
    }

    void TestFinal() override {
        const Tensor image = RenderImage();
        const auto* texels = static_cast<const std::uint8_t*>(image.Data());
        const std::int64_t pixels = image.Shape()[0] * image.Shape()[1];
        double sum = 0.0;
        double squares = 0.0;
        for (std::int64_t i = 0; i < pixels * 4; ++i) {
            if (i % 4 != 3) {
                sum += texels[i];
                squares += static_cast<double>(texels[i]) * texels[i];
            }
        }
        const double mean = sum / static_cast<double>(pixels * 3);
        if (std::sqrt(squares / static_cast<double>(pixels * 3) - mean * mean) <= 1.0) {
            throw std::runtime_error("The boxes and ground should not render flat");
        }
    }

  private:
    void AimSun() {
        const double e = glm::radians(static_cast<double>(elevation_));
        const double a = glm::radians(static_cast<double>(azimuth_));
        GetScene().Set("/World/Sun", tokens::kTransform,
                       DistantLightTransform(
                           {std::cos(e) * std::cos(a), std::cos(e) * std::sin(a), std::sin(e)}));
    }

    float elevation_ = 45.0f;
    float azimuth_ = 30.0f;
};

}  // namespace

int main(int argc, char** argv) {
    return ernie::app::DemoMain(argc, argv,
                                {ernie::app::MakeDemoInfo<SunPanelDemo>("viewer", "sun panel")});
}
