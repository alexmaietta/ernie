// SPDX-License-Identifier: BSD-3-Clause
// A drone camera's live image shown where it was seen from: on its
// frustum a few metres out, with the frustum's edges, and as a
// picture in the picture, rendered each step from its own settings.

#include "ernie/app/demo.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/overlay.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"
#include "util.h"

#include <array>
#include <cmath>
#include <numbers>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr double kBillboard = 3.0;  // metres in front of the drone
const glm::dvec2 kDroneImage{320, 180};

class CameraBillboardDemo : public app::Demo {
  public:
    CameraBillboardDemo(app::Viewer* viewer,
                        const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 120;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        GroundPlane::Define(scene, "/World/Ground")
            .SetChecker(true)
            .SetCheckerSize(1.0);
        DistantLight::Define(scene, "/World/Sun")
            .SetTransform(DistantLightTransform({0.4, -0.6, 0.7}));
        // What the drone looks at: a few shapes, one bouncing.
        Cube::Define(scene, "/World/Red")
            .SetDisplayColor({0.85, 0.2, 0.15});
        SetXform(scene, "/World/Red",
                 {.translate = glm::dvec3(-1.5, 0, 0.5)});
        Cube::Define(scene, "/World/Blue")
            .SetDisplayColor({0.2, 0.35, 0.9});
        SetXform(scene, "/World/Blue",
                 {.translate = glm::dvec3(1.5, 1, 0.5)});
        Sphere::Define(scene, "/World/Ball")
            .SetRadius(0.5)
            .SetDisplayColor({0.95, 0.8, 0.2});
        // The view over it all, and the drone's own camera and
        // settings, which draw no overlays.
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(22.0)
            .SetTransform(LookAtTransform({-9.0, -14.0, 9.0},
                                          {0.0, 0.0, 1.5},
                                          {0.0, 0.0, 1.0}));
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution(glm::dvec2(960, 540))
            .SetOverlays("*");
        Camera::Define(scene, "/World/Drone")
            .SetFocalLength(18.0)
            .SetHorizontalAperture(20.955)
            .SetVerticalAperture(20.955 * kDroneImage.y /
                                 kDroneImage.x);
        RenderSettings::Define(scene, "/Render/Drone")
            .SetCamera(Path("/World/Drone"))
            .SetResolution(kDroneImage);
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
        }
        Place();
    }

    void Step() override {
        Place();
    }

    void TestFinal() override {
        Check(demos::ColourDeviation(droneImage_) > 5.0,
              "the drone should see the scene");
        const Overlay* drone = GetScene().FindOverlay("drone");
        Check(drone != nullptr && drone->Batches().size() == 2,
              "its image on its frustum and in the corner");
    }

  private:
    /// Flies the drone, renders what it sees and shows it.
    void Place() {
        Scene& scene = GetScene();
        const double t = time;
        SetXform(scene, "/World/Ball",
                 {.translate = glm::dvec3(
                      0.0, -1.0,
                      0.5 + 1.5 * std::abs(std::sin(2.5 * t)))});
        const double angle = 0.4 * t;
        const glm::dvec3 eye(6.0 * std::cos(angle),
                             6.0 * std::sin(angle), 3.5);
        scene.Set(
            "/World/Drone", tokens::kTransform,
            LookAtTransform(eye, {0.0, 0.0, 0.5}, {0.0, 0.0, 1.0}));
        droneImage_ =
            GetRenderer().Render(scene, Path("/Render/Drone"));
        // Its image where it was seen from, the frustum's edges to
        // it, and the same image in the top right corner.
        Overlay& drone = scene.GetOverlay("drone").Clear();
        const std::array<glm::dvec3, 4> corners = CameraImageCorners(
            scene, Path("/World/Drone"), kBillboard);
        OverlayStyle edge;
        edge.color = {1.0f, 0.85f, 0.3f, 1.0f};
        edge.width = 2.0f;
        for (int k = 0; k < 4; ++k) {
            drone.Line(eye, corners[k], edge);
            drone.Line(corners[k], corners[(k + 1) % 4], edge);
        }
        OverlayStyle image;
        image.color = {1.0f, 1.0f, 1.0f, 0.95f};
        drone.Image(corners, droneImage_, image);
        const ScreenPlacement corner{.at = {1.0f, 0.0f},
                                     .offset = {-12.0f, 12.0f}};
        OverlayStyle border;
        border.color = edge.color;
        drone.Rect({.at = corner.at,
                    .offset = corner.offset + glm::vec2(3.0f, -3.0f)},
                   glm::vec2(kDroneImage) + 6.0f, border);
        drone.ScreenImage(corner, droneImage_, {},
                          glm::vec2(kDroneImage));
        OverlayStyle label;
        label.size = 13.0f;
        drone.Text({.at = {1.0f, 0.0f},
                    .offset = {-12.0f, 24.0f + static_cast<float>(
                                                   kDroneImage.y)}},
                   "drone camera", label);
    }

    Tensor droneImage_;
};

}  // namespace

ERNIE_REGISTER_DEMO(
    CameraBillboardDemo, "overlays", "camera billboard",
    "A drone camera's live image on its frustum where it "
    "was seen from, and as a picture in the picture.");
