// SPDX-License-Identifier: BSD-3-Clause
// A scene built in code with a custom panel of options: every kind of
// widget the Demos panel offers, each changing the scene.

#include "ernie/app/demo.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/xform.h"
#include "util.h"

#include <glm/trigonometric.hpp>
#include <imgui.h>

#include <array>
#include <cmath>
#include <optional>
#include <string>
#include <vector>

namespace {

using namespace ernie;

constexpr int kBoxes = 5;
// How far the boxes rise and fall when bobbing, in metres.
constexpr double kBob = 0.5;

std::array<double, 3> BoxColour(int i) {
    return {0.2 + 0.15 * i, 0.4, 0.8 - 0.15 * i};
}

Path BoxPath(int i) {
    return Path("/World/Box" + std::to_string(i));
}

class CustomPanelDemo : public app::Demo {
  public:
    CustomPanelDemo(app::Viewer* viewer,
                    const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        GroundPlane::Define(scene, "/World/Ground");
        for (int i = 0; i < kBoxes; ++i) {
            const std::array<double, 3> c = BoxColour(i);
            Cube::Define(scene, BoxPath(i))
                .SetSize(1.0 + 0.25 * i)
                .SetDisplayColor({c[0], c[1], c[2]});
            SetXform(
                scene, BoxPath(i),
                {.translate = glm::dvec3(3.0 * (i - 2), 0.0, 1.0)});
            colours_.push_back(c);
            shown_.push_back(true);
        }
        DistantLight::Define(scene, "/World/Sun");
        // The viewer frames the scene with its own camera; this one
        // renders it without a viewer and for --test.
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(24.0)
            .SetTransform(LookAtTransform(
                {0.0, -14.0, 6.0}, {0.0, 0.0, 0.5}, {0.0, 0.0, 1.0}));
        AimSun();
        // A panel of the demo's own, docked beside the viewport; the
        // Demos panel's options without a viewer.
        if (viewer != nullptr) {
            panel_ = &viewer->AddWidgetPanel("Custom panel",
                                             app::Dock::Right);
            viewer->FocusPanel("Custom panel");
        } else {
            panel_ = &Widgets();
        }
        AddWidgets();
    }

    // C++ demos can also draw Dear ImGui directly, in the Demos
    // panel's options.
    void Gui() override {
        ImGui::TextDisabled("Drawn with ImGui in Gui()");
    }

    void Step() override {
        double height = 1.0;
        for (int i = 0; i < kBoxes; ++i) {
            double z = 1.0;
            if (bobbing_) {
                z += kBob * std::sin(2.0 * time + i);
                SetXform(
                    GetScene(), BoxPath(i),
                    {.translate = glm::dvec3(3.0 * (i - 2), 0.0, z)});
            }
            if (i == selected_) {
                height = z;
            }
        }
        // The last 240 heights, for the plot.
        heights_.push_back(static_cast<float>(height));
        if (heights_.size() > 240) {
            heights_.erase(heights_.begin());
        }
    }

    void TestFinal() override {
        demos::Check(demos::ColourDeviation(RenderImage()) > 1.0,
                     "the boxes and ground should not render flat");
    }

  private:
    void AddWidgets() {
        app::WidgetPanel& w = *panel_;
        // Text, and a field whose text it shows.
        caption_ = w.Text("Five boxes under a sun.");
        w.InputText("Caption", "Five boxes under a sun.",
                    [this](const std::string& text) {
                        panel_->Set(caption_, text);
                    });
        w.Separator();
        // Sliders, and a button that sets them back.
        elevationSlider_ = w.Slider(
            "Elevation", 1.0, 90.0, elevation_,
            [this](double value) { AimSun(value, std::nullopt); });
        azimuthSlider_ = w.Slider(
            "Azimuth", 0.0, 360.0, azimuth_,
            [this](double value) { AimSun(std::nullopt, value); });
        w.Button("Reset the sun", [this] { ResetSun(); });
        w.Separator();
        // A drop-down of the boxes, and the chosen one's colour and
        // visibility.
        std::vector<std::string> names;
        for (int i = 0; i < kBoxes; ++i) {
            names.push_back("Box " + std::to_string(i));
        }
        w.Combo("Box", names, 0,
                [this](std::size_t index) { Select(index); });
        colour_ = w.Color(
            "Colour", colours_[0],
            [this](const std::array<double, 3>& rgb) { Paint(rgb); });
        visible_ = w.Checkbox("Visible", true,
                              [this](bool shown) { Show(shown); });
        w.Separator();
        // A checkbox that animates the boxes, and a plot of the
        // chosen box's height.
        w.Checkbox("Bob", bobbing_,
                   [this](bool bobbing) { bobbing_ = bobbing; });
        w.Plot("Height", [this] { return heights_; });
    }

    void AimSun(std::optional<double> elevation = std::nullopt,
                std::optional<double> azimuth = std::nullopt) {
        elevation_ = elevation.value_or(elevation_);
        azimuth_ = azimuth.value_or(azimuth_);
        const double e = glm::radians(elevation_);
        const double a = glm::radians(azimuth_);
        DistantLight(GetScene(), "/World/Sun")
            .SetTransform(DistantLightTransform(
                {std::cos(e) * std::cos(a), std::cos(e) * std::sin(a),
                 std::sin(e)}));
    }

    void ResetSun() {
        AimSun(45.0, 30.0);
        panel_->Set(elevationSlider_, elevation_);
        panel_->Set(azimuthSlider_, azimuth_);
    }

    void Select(std::size_t index) {
        selected_ = static_cast<int>(index);
        panel_->Set(colour_, colours_[index]);
        panel_->Set(visible_, static_cast<bool>(shown_[index]));
        heights_.clear();
    }

    void Paint(const std::array<double, 3>& rgb) {
        colours_[static_cast<std::size_t>(selected_)] = rgb;
        Cube(GetScene(), BoxPath(selected_))
            .SetDisplayColor({rgb[0], rgb[1], rgb[2]});
    }

    void Show(bool shown) {
        shown_[static_cast<std::size_t>(selected_)] = shown;
        Cube(GetScene(), BoxPath(selected_))
            .SetVisibility(std::string(shown ? tokens::kInherited
                                             : tokens::kInvisible));
    }

    double elevation_ = 45.0;
    double azimuth_ = 30.0;
    std::vector<std::array<double, 3>> colours_;
    std::vector<char> shown_;
    int selected_ = 0;
    bool bobbing_ = false;
    std::vector<float> heights_;
    // The custom panel, or the Demos panel's options.
    app::WidgetPanel* panel_ = nullptr;
    app::WidgetPanel::Id caption_ = 0;
    app::WidgetPanel::Id elevationSlider_ = 0;
    app::WidgetPanel::Id azimuthSlider_ = 0;
    app::WidgetPanel::Id colour_ = 0;
    app::WidgetPanel::Id visible_ = 0;
};

}  // namespace

ERNIE_REGISTER_DEMO(CustomPanelDemo, "viewer", "custom panel",
                    "A scene built in code with a custom panel of "
                    "every kind of option.");
