// SPDX-License-Identifier: BSD-3-Clause
// Bouncing balls under a heads-up display: screen text and an energy
// bar redrawn each step, labels posed under each ball that follow it
// unchanged, and each ball's height printed on the GPU from an array.

#include "ernie/app/demo.h"
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
#include <cstdio>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr int kBalls = 5;
constexpr double kGravity = 9.81;
constexpr double kRadius = 0.5;
constexpr double kRestitution = 0.85;
constexpr std::uint32_t kWidth = 640;
constexpr std::uint32_t kHeight = 360;
constexpr float kBarWidth = 240.0f;
constexpr glm::vec4 kBarColor{0.25f, 0.85f, 0.35f, 1.0f};

class HudDemo : public app::Demo {
  public:
    HudDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 300;
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
                {0.0, -14.0, 5.0}, {0.0, 0.0, 2.0}, {0.0, 0.0, 1.0}));
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution(glm::dvec2(kWidth, kHeight))
            .SetOverlays("*");

        // Labels and rings in each ball's space: added once, they
        // follow the balls with no redraw.
        Overlay& tracked = scene.GetOverlay("tracked");
        for (int i = 0; i < kBalls; ++i) {
            const Path ball("/World/Ball" + std::to_string(i));
            Sphere::Define(scene, ball)
                .SetRadius(kRadius)
                .SetDisplayColor({0.9, 0.3 + 0.12 * i, 0.2});
            heights_[i] = 1.0 + 0.8 * i;
            OverlayStyle style;
            style.anchor = ball;
            style.depth = OverlayDepth::Always;
            tracked.Label({0.0, 0.0, kRadius},
                          "ball " + std::to_string(i), style);
            style.size = 28.0f;
            style.color = {1.0f, 1.0f, 1.0f, 0.6f};
            tracked.Marker({0.0, 0.0, 0.0}, style, MarkerShape::Ring);
            anchors_.push_back(ball);
        }
        // Each ball's height, formatted on the GPU from an array
        // written in place each step.
        OverlayBatch numbers;
        numbers.kind = OverlayBatchKind::Numbers;
        numbers.points = Tensor::Zeros(DType::Float32, {kBalls, 3});
        numbers.values = Tensor::Zeros(DType::Float32, {kBalls});
        numbers.format = {
            .prefix = "h ", .suffix = " m", .decimals = 2};
        numbers.anchors = anchors_;
        numbers.style.color = {0.6f, 0.9f, 1.0f, 1.0f};
        numbers.style.depth = OverlayDepth::Always;
        auto* below = static_cast<float*>(numbers.points.Data());
        for (int i = 0; i < kBalls; ++i) {
            below[3 * i + 2] = static_cast<float>(-2.5 * kRadius);
        }
        tracked.Batch(numbers);
        Place();
    }

    void Step() override {
        for (int i = 0; i < kBalls; ++i) {
            velocities_[i] -= kGravity * frameDt;
            heights_[i] += velocities_[i] * frameDt;
            if (heights_[i] < 0.0) {
                heights_[i] = 0.0;
                velocities_[i] = -velocities_[i] * kRestitution;
            }
        }
        Place();
    }

    void TestFinal() override {
        const Scene& scene = GetScene();
        const Overlay* hud = scene.FindOverlay("hud");
        Check(hud != nullptr && !hud->Sprites().empty(),
              "the HUD should hold text");
        Check(scene.FindOverlay("tracked")->Anchors().size() ==
                  static_cast<std::size_t>(kBalls),
              "a label anchored to each ball");
        // The energy bar's fill, left of the bottom centre.
        const Tensor& image = Image();
        if (image.Empty()) {
            return;
        }
        const auto* texels =
            static_cast<const std::uint8_t*>(image.Data());
        const std::size_t x = kWidth / 2 - kBarWidth / 2 + 4;
        const std::size_t y = kHeight - 16 - 7;
        const std::uint8_t* texel = texels + 4 * (y * kWidth + x);
        Check(texel[1] > 150 && texel[0] < 120 && texel[2] < 140,
              "the energy bar should be green at its left end");
    }

  private:
    /// Poses the balls, writes their heights for the GPU's text and
    /// redraws the screen text.
    void Place() {
        Scene& scene = GetScene();
        Overlay& tracked = scene.GetOverlay("tracked");
        auto* values =
            static_cast<float*>(tracked.Batches()[0].values.Data());
        double energy = 0.0;
        for (int i = 0; i < kBalls; ++i) {
            SetXform(
                scene, anchors_[i],
                {.translate = glm::dvec3(-4.0 + 2.0 * i, 0.0,
                                         heights_[i] + kRadius)});
            values[i] = static_cast<float>(heights_[i]);
            energy += kGravity * heights_[i] +
                      0.5 * velocities_[i] * velocities_[i];
        }
        // Host arrays written in place: read again once touched.
        tracked.Touch();
        if (initialEnergy_ == 0.0) {
            initialEnergy_ = energy;
        }
        Overlay& hud = scene.GetOverlay("hud").Clear();
        OverlayStyle text;
        text.size = 16.0f;
        char line[96];
        std::snprintf(line, sizeof(line), "t = %.2f s\nstep %llu",
                      time, static_cast<unsigned long long>(frame));
        hud.Text({.offset = {12.0f, 12.0f}}, line, text);
        hud.Text({.at = {1.0f, 0.0f}, .offset = {-12.0f, 12.0f}},
                 std::to_string(kBalls) + " balls", text);
        // The energy left, as a bar over a dark track.
        const float share =
            static_cast<float>(energy / initialEnergy_);
        const ScreenPlacement bar{.at = {0.5f, 1.0f},
                                  .offset = {-kBarWidth / 2, -16.0f},
                                  .align = glm::vec2(0.0f, 1.0f)};
        OverlayStyle track;
        track.color = {0.0f, 0.0f, 0.0f, 0.6f};
        hud.Rect(bar, {kBarWidth, 14.0f}, track);
        OverlayStyle fill;
        fill.color = kBarColor;
        hud.Rect(bar,
                 {kBarWidth * std::clamp(share, 0.0f, 1.0f), 14.0f},
                 fill);
        text.size = 12.0f;
        text.background.a = 0.0f;
        hud.Text({.at = {0.5f, 1.0f},
                  .offset = {0.0f, -34.0f},
                  .align = glm::vec2(0.5f, 1.0f)},
                 "energy", text);
    }

    std::array<double, kBalls> heights_{};
    std::array<double, kBalls> velocities_{};
    std::vector<Path> anchors_;
    double initialEnergy_ = 0.0;
};

}  // namespace

ERNIE_REGISTER_DEMO(HudDemo, "overlays", "hud",
                    "Screen text and an energy bar over bouncing "
                    "balls, labels following each ball, and heights "
                    "printed on the GPU.");
