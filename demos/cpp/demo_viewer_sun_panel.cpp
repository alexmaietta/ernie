// Opens the viewer on a scene built in code, with a panel of its own that moves the sun.
//
//     ./build/bin/demo_viewer_sun_panel

#include "ernie/app/app.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"

#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>

#include <cmath>
#include <cstdio>
#include <exception>

int main() {
    using namespace ernie;
    Scene scene;
    scene.Define("/World/Ground", tokens::kGroundPlane);
    for (int i = 0; i < 5; ++i) {
        const double x = 3.0 * (i - 2);
        scene.Define(Path("/World/Box" + std::to_string(i)), tokens::kCube)
            .Set(tokens::kSize, 1.0 + 0.25 * i)
            .Set(tokens::kTransform, glm::translate(glm::dmat4(1.0), glm::dvec3(x, 0.0, 1.0)))
            .Set(tokens::kDisplayColor, glm::vec3(0.2f + 0.15f * i, 0.4f, 0.8f - 0.15f * i));
    }
    scene.Define("/World/Sun", tokens::kDistantLight);

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
