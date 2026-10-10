// SPDX-License-Identifier: BSD-3-Clause
// Opens the viewer on a scene built in code, with a panel of its own that moves the sun.
//
//     ./build/bin/demo_viewer_sun_panel

#include "ernie/app/app.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"

#include <glm/trigonometric.hpp>
#include <imgui.h>

#include <cmath>
#include <cstdio>
#include <exception>

int main() {
    using namespace ernie;
    Scene scene;
    GroundPlane::Define(scene, "/World/Ground");
    for (int i = 0; i < 5; ++i) {
        const Path box("/World/Box" + std::to_string(i));
        Cube::Define(scene, box)
            .SetSize(1.0 + 0.25 * i)
            .SetDisplayColor({0.2 + 0.15 * i, 0.4, 0.8 - 0.15 * i});
        SetXform(scene, box, {.translate = glm::dvec3(3.0 * (i - 2), 0.0, 1.0)});
    }
    DistantLight::Define(scene, "/World/Sun");

    try {
        app::Viewer viewer(scene, {.title = "Ernie demo"});
        float elevation = 45.0f;
        float azimuth = 30.0f;
        // Panels draw with Dear ImGui and edit the scene, which the next frame renders.
        viewer.AddPanel("Sun", app::Dock::Left, [&](app::Viewer& v) {
            ImGui::SliderFloat("Elevation", &elevation, 1.0f, 90.0f);
            ImGui::SliderFloat("Azimuth", &azimuth, 0.0f, 360.0f);
        });
        viewer.OnPreRender([&](app::Viewer& v, const app::FrameInfo&) {
            const double e = glm::radians(static_cast<double>(elevation));
            const double a = glm::radians(static_cast<double>(azimuth));
            v.GetScene().Set("/World/Sun", tokens::kTransform,
                             DistantLightTransform({std::cos(e) * std::cos(a),
                                                    std::cos(e) * std::sin(a), std::sin(e)}));
        });
        viewer.Run();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
