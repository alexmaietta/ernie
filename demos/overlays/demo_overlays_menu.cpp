// SPDX-License-Identifier: BSD-3-Clause
// A heads-up menu of buttons over bouncing balls: pause, reset, add a
// ball and toggle trails and labels, each button lit while hovered.

#include "ernie/app/demo.h"
#include "ernie/app/viewer.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/overlay.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/xform.h"
#include "util.h"

#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr int kMaxBalls = 8;
constexpr double kGravity = 9.81;
constexpr double kRadius = 0.4;
constexpr std::size_t kTrail = 60;
constexpr std::uint32_t kWidth = 640;
constexpr std::uint32_t kHeight = 360;

// The menu's buttons, by id.
enum : std::uint32_t { kPause = 1, kReset, kAdd, kTrails, kLabels };
constexpr glm::vec2 kButton{150.0f, 26.0f};

/// Where button `row` of the menu sits on the image.
ScreenPlacement ButtonAt(int row) {
    return {
        .offset = {16.0f, 16.0f + 34.0f * static_cast<float>(row)},
        .align = glm::vec2(0.0f)};
}

class MenuDemo : public app::Demo {
  public:
    MenuDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 120;
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
                {2.0, -13.0, 4.0}, {2.0, 0.0, 1.5}, {0.0, 0.0, 1.0}));
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution(glm::dvec2(kWidth, kHeight))
            .SetOverlays("*");
        Reset();
        // Clicks on the menu's buttons, in the viewer.
        if (viewer != nullptr) {
            viewer->AddOverlayClickHandler(
                [this](app::Viewer&, const std::string& overlay,
                       std::uint32_t id) {
                    if (overlay == "menu") {
                        Click(id);
                    }
                });
        }
    }

    void Step() override {
        if (!paused_) {
            for (Ball& ball : balls_) {
                ball.velocity -= kGravity * frameDt;
                ball.height += ball.velocity * frameDt;
                if (ball.height < 0.0) {
                    ball.height = 0.0;
                    ball.velocity = -0.9 * ball.velocity;
                }
                ball.trail.push_back(ball.Position());
                if (ball.trail.size() > kTrail) {
                    ball.trail.erase(ball.trail.begin());
                }
            }
        }
        Draw();
    }

    void TestFinal() override {
        // Each button is hit where it is drawn, and a click on it
        // does what it says.
        const Overlay& menu = *GetScene().FindOverlay("menu");
        const glm::vec2 extent(kWidth, kHeight);
        for (std::uint32_t id = kPause; id <= kLabels; ++id) {
            const ScreenPlacement at =
                ButtonAt(static_cast<int>(id) - 1);
            Check(menu.ScreenItemAt(at.offset + 0.5f * kButton,
                                    extent) == id,
                  "button " + std::to_string(id) +
                      " is hit at its centre");
        }
        const std::size_t balls = balls_.size();
        Click(menu.ScreenItemAt(ButtonAt(2).offset + 1.0f, extent));
        Check(balls_.size() == balls + 1, "Add ball adds one");
        const double height = balls_.front().height;
        Click(kPause);
        Step();
        Check(balls_.front().height == height,
              "paused balls stay put");
        Click(kTrails);
        Check(!GetScene().FindOverlay("trails")->IsVisible(),
              "the trails button hides the trails");
        Click(kReset);
        Check(balls_.size() == 3 && !paused_, "reset restarts");
    }

  private:
    struct Ball {
        Path path;
        double x = 0.0;
        double height = 0.0;
        double velocity = 0.0;
        std::vector<glm::dvec3> trail;

        glm::dvec3 Position() const {
            return {x, 0.0, height + kRadius};
        }
    };

    void Click(std::uint32_t id) {
        Scene& scene = GetScene();
        switch (id) {
        case kPause:
            paused_ = !paused_;
            break;
        case kReset:
            Reset();
            break;
        case kAdd:
            AddBall();
            break;
        case kTrails:
            scene.GetOverlay("trails").SetVisible(
                !scene.GetOverlay("trails").IsVisible());
            break;
        case kLabels:
            scene.GetOverlay("labels").SetVisible(
                !scene.GetOverlay("labels").IsVisible());
            break;
        default:
            break;
        }
        Draw();
    }

    void Reset() {
        Scene& scene = GetScene();
        for (const Ball& ball : balls_) {
            scene.Remove(ball.path);
        }
        balls_.clear();
        scene.GetOverlay("labels").Clear().SetVisible(true);
        scene.GetOverlay("trails").SetVisible(true);
        paused_ = false;
        for (int i = 0; i < 3; ++i) {
            AddBall();
        }
        Draw();
    }

    void AddBall() {
        if (balls_.size() >= kMaxBalls) {
            return;
        }
        Scene& scene = GetScene();
        const auto i = static_cast<int>(balls_.size());
        Ball ball;
        ball.path = Path("/World/Ball" + std::to_string(next_++));
        ball.x = -1.0 + 1.2 * i;
        ball.height = 1.0 + 0.6 * i;
        Sphere::Define(scene, ball.path)
            .SetRadius(kRadius)
            .SetDisplayColor({0.9, 0.3 + 0.08 * i, 0.2});
        // A label that follows the ball.
        OverlayStyle label;
        label.anchor = ball.path;
        label.depth = OverlayDepth::Always;
        scene.GetOverlay("labels").Label(
            {0.0, 0.0, kRadius}, "ball " + std::to_string(i), label);
        balls_.push_back(std::move(ball));
    }

    void Draw() {
        Scene& scene = GetScene();
        for (const Ball& ball : balls_) {
            SetXform(scene, ball.path,
                     {.translate = ball.Position()});
        }
        Overlay& trails = scene.GetOverlay("trails").Clear();
        OverlayStyle trail;
        trail.color = {0.3f, 0.7f, 1.0f, 0.8f};
        for (const Ball& ball : balls_) {
            trails.Polyline(ball.trail, trail);
        }
        // The menu: a button a row, lit while hovered.
        Overlay& menu = scene.GetOverlay("menu");
        const std::uint32_t hovered = menu.Hovered();
        menu.Clear().SetHovered(hovered);
        const std::array<std::string, 5> labels{
            paused_ ? "Resume" : "Pause", "Reset", "Add ball",
            std::string("Trails: ") +
                (scene.GetOverlay("trails").IsVisible() ? "on"
                                                        : "off"),
            std::string("Labels: ") +
                (scene.GetOverlay("labels").IsVisible() ? "on"
                                                        : "off")};
        OverlayStyle button;
        button.size = 13.0f;
        button.background = {0.1f, 0.12f, 0.16f, 0.85f};
        button.hover = glm::vec4(0.25f, 0.45f, 0.8f, 0.95f);
        for (std::size_t i = 0; i < labels.size(); ++i) {
            button.id = static_cast<std::uint32_t>(i) + kPause;
            menu.Button(ButtonAt(static_cast<int>(i)), labels[i],
                        button, kButton);
        }
        OverlayStyle status;
        status.size = 13.0f;
        menu.Text({.at = {1.0f, 0.0f}, .offset = {-16.0f, 16.0f}},
                  std::to_string(balls_.size()) + " balls" +
                      (paused_ ? ", paused" : ""),
                  status);
    }

    std::vector<Ball> balls_;
    int next_ = 0;
    bool paused_ = false;
};

}  // namespace

ERNIE_REGISTER_DEMO(
    MenuDemo, "overlays", "menu",
    "A heads-up menu of buttons, lit while hovered, that "
    "pauses, resets and adds to bouncing balls.");
