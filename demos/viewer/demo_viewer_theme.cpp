// SPDX-License-Identifier: BSD-3-Clause
// The viewer restyled while it runs: a light, a dark and a
// high-contrast theme, then the accent, background and text colours,
// text size, corner rounding and spacing each changed live. The Demos
// panel puts the viewer's own theme back when another demo starts.
//
// References:
// - W3C, "Web Content Accessibility Guidelines (WCAG) 2.1", 2018,
//   success criteria 1.4.3 and 1.4.6 (contrast ratios).

#include "ernie/app/demo.h"
#include "ernie/app/theme.h"
#include "ernie/app/viewer.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/xform.h"
#include "util.h"

#include <glm/glm.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

const std::vector<std::string> kPresets = {"Light", "Dark",
                                           "High contrast"};
// The contrast each preset's text needs against its panels: WCAG's AA
// for body text, AAA for the high-contrast theme, and 3 for muted
// labels.
constexpr std::array<double, 3> kContrast = {4.5, 4.5, 7.0};
constexpr double kMutedContrast = 3.0;

/// The viewer's default theme with preset `index`'s colours (sRGB,
/// alpha last) and metrics.
app::Theme Themed(std::size_t index) {
    app::Theme theme;
    if (index == 1) {
        theme.background = {0.13f, 0.14f, 0.15f, 1.0f};
        theme.panel = {0.17f, 0.18f, 0.19f, 1.0f};
        theme.header = {0.22f, 0.23f, 0.25f, 1.0f};
        theme.field = {0.10f, 0.11f, 0.12f, 1.0f};
        theme.fieldHovered = {0.14f, 0.15f, 0.17f, 1.0f};
        theme.button = {0.24f, 0.26f, 0.28f, 1.0f};
        theme.buttonLit = {0.25f, 0.38f, 0.55f, 1.0f};
        theme.separator = {0.28f, 0.30f, 0.32f, 1.0f};
        theme.border = {0.33f, 0.35f, 0.37f, 1.0f};
        theme.text = {0.90f, 0.91f, 0.92f, 1.0f};
        theme.textMuted = {0.62f, 0.64f, 0.66f, 1.0f};
        theme.selection = {0.22f, 0.33f, 0.48f, 1.0f};
        theme.accent = {0.38f, 0.62f, 0.95f, 1.0f};
        theme.overlay = {0.17f, 0.18f, 0.19f, 0.9f};
        theme.overlayBorder = {0.33f, 0.35f, 0.37f, 1.0f};
        theme.checkBox = {0.10f, 0.11f, 0.12f, 1.0f};
        theme.checkMark = {0.90f, 0.91f, 0.92f, 1.0f};
        theme.rounding = 4.0f;
        theme.pillRounding = 8.0f;
    } else if (index == 2) {
        theme.background = {1.0f, 1.0f, 1.0f, 1.0f};
        theme.panel = {1.0f, 1.0f, 1.0f, 1.0f};
        theme.header = {0.85f, 0.85f, 0.85f, 1.0f};
        theme.field = {1.0f, 1.0f, 1.0f, 1.0f};
        theme.fieldHovered = {0.92f, 0.92f, 0.92f, 1.0f};
        theme.button = {0.90f, 0.90f, 0.90f, 1.0f};
        theme.buttonLit = {0.70f, 0.80f, 1.0f, 1.0f};
        theme.separator = {0.0f, 0.0f, 0.0f, 1.0f};
        theme.border = {0.0f, 0.0f, 0.0f, 1.0f};
        theme.text = {0.0f, 0.0f, 0.0f, 1.0f};
        theme.textMuted = {0.25f, 0.25f, 0.25f, 1.0f};
        theme.selection = {0.75f, 0.85f, 1.0f, 1.0f};
        theme.accent = {0.0f, 0.25f, 0.75f, 1.0f};
        theme.overlay = {1.0f, 1.0f, 1.0f, 0.95f};
        theme.overlayBorder = {0.0f, 0.0f, 0.0f, 1.0f};
        theme.checkBox = {1.0f, 1.0f, 1.0f, 1.0f};
        theme.checkMark = {0.0f, 0.0f, 0.0f, 1.0f};
        theme.frameBorder = 2.0f;
        theme.fontSize = 16.0f;
    }
    return theme;
}

/// WCAG's relative luminance of an sRGB colour.
double Luminance(const glm::vec4& rgb) {
    const auto linear = [](double c) {
        return c <= 0.04045 ? c / 12.92
                            : std::pow((c + 0.055) / 1.055, 2.4);
    };
    return 0.2126 * linear(rgb.r) + 0.7152 * linear(rgb.g) +
           0.0722 * linear(rgb.b);
}

/// WCAG's contrast ratio of two sRGB colours, 1 to 21.
double Contrast(const glm::vec4& a, const glm::vec4& b) {
    const double high = std::max(Luminance(a), Luminance(b));
    const double low = std::min(Luminance(a), Luminance(b));
    return (high + 0.05) / (low + 0.05);
}

std::array<double, 3> Rgb(const glm::vec4& colour) {
    return {colour.r, colour.g, colour.b};
}

class ThemeDemo : public app::Demo {
  public:
    ThemeDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        GroundPlane::Define(scene, "/World/Ground").SetChecker(true);
        const std::array<glm::dvec3, 3> colours = {
            glm::dvec3(0.8, 0.3, 0.2), glm::dvec3(0.2, 0.6, 0.3),
            glm::dvec3(0.2, 0.4, 0.8)};
        for (int i = 0; i < 3; ++i) {
            const Path ball("/World/Ball" + std::to_string(i));
            Sphere::Define(scene, ball)
                .SetRadius(0.6)
                .SetDisplayColor(
                    colours[static_cast<std::size_t>(i)]);
            SetXform(
                scene, ball,
                {.translate = glm::dvec3(1.6 * (i - 1), 0.0, 0.6)});
        }
        DistantLight::Define(scene, "/World/Sun")
            .SetTransform(DistantLightTransform(
                glm::normalize(glm::dvec3(0.4, -0.6, 0.7))));
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(24.0)
            .SetTransform(LookAtTransform(
                {0.0, -7.0, 3.0}, {0.0, 0.0, 0.6}, {0.0, 0.0, 1.0}));
        theme_ = Themed(0);
        app::WidgetPanel& w = Widgets();
        w.Combo("Preset", kPresets, 0,
                [this](std::size_t index) { Choose(index); });
        w.Separator();
        accent_ = w.Color("Accent", Rgb(theme_.accent),
                          [this](const std::array<double, 3>& rgb) {
                              Paint(theme_.accent, rgb);
                          });
        background_ =
            w.Color("Background", Rgb(theme_.panel),
                    [this](const std::array<double, 3>& rgb) {
                        // The docks behind the panels too.
                        Paint(theme_.background, rgb);
                        Paint(theme_.panel, rgb);
                    });
        text_ = w.Color("Text", Rgb(theme_.text),
                        [this](const std::array<double, 3>& rgb) {
                            Paint(theme_.text, rgb);
                        });
        font_ = w.Slider("Text size", 10.0, 22.0, theme_.fontSize,
                         [this](double size) {
                             theme_.fontSize =
                                 static_cast<float>(size);
                             Apply();
                         });
        rounding_ = w.Slider("Rounding", 0.0, 12.0, theme_.rounding,
                             [this](double radius) {
                                 theme_.rounding =
                                     static_cast<float>(radius);
                                 Apply();
                             });
        pill_ = w.Slider("Toolbar rounding", 0.0, 16.0,
                         theme_.pillRounding, [this](double radius) {
                             theme_.pillRounding =
                                 static_cast<float>(radius);
                             Apply();
                         });
        spacing_ = w.Slider("Spacing", 2.0, 12.0,
                            theme_.itemSpacing.x, [this](double gap) {
                                theme_.itemSpacing = glm::vec2(
                                    static_cast<float>(gap));
                                Apply();
                            });
        contrast_ = w.Text("");
        Apply();
    }

    void TestFinal() override {
        for (std::size_t i = 0; i < kPresets.size(); ++i) {
            const app::Theme theme = Themed(i);
            const double text = Contrast(theme.text, theme.panel);
            const double muted =
                Contrast(theme.textMuted, theme.panel);
            std::printf("%s: text %.1f:1, muted %.1f:1\n",
                        kPresets[i].c_str(), text, muted);
            Check(text >= kContrast[i],
                  kPresets[i] + "'s text should read");
            Check(muted >= kMutedContrast,
                  kPresets[i] + "'s labels should read");
        }
        // Changing the accent keeps the rest of the theme.
        Choose(1);
        Paint(theme_.accent, {0.9, 0.5, 0.1});
        Check(glm::all(glm::lessThan(
                  glm::abs(theme_.accent -
                           glm::vec4(0.9f, 0.5f, 0.1f, 1.0f)),
                  glm::vec4(1e-6f))),
              "the accent should change");
        Check(theme_.panel == Themed(1).panel,
              "the panels should keep the dark theme's colour");
    }

  private:
    /// Starts again from preset `index`, its controls showing it.
    void Choose(std::size_t index) {
        theme_ = Themed(index);
        app::WidgetPanel& w = Widgets();
        w.Set(accent_, Rgb(theme_.accent));
        w.Set(background_, Rgb(theme_.panel));
        w.Set(text_, Rgb(theme_.text));
        w.Set(font_, static_cast<double>(theme_.fontSize));
        w.Set(rounding_, static_cast<double>(theme_.rounding));
        w.Set(pill_, static_cast<double>(theme_.pillRounding));
        w.Set(spacing_, static_cast<double>(theme_.itemSpacing.x));
        Apply();
    }

    void Paint(glm::vec4& colour, const std::array<double, 3>& rgb) {
        colour = glm::vec4(static_cast<float>(rgb[0]),
                           static_cast<float>(rgb[1]),
                           static_cast<float>(rgb[2]), colour.a);
        Apply();
    }

    /// Shows the theme in the viewer, and its text's contrast.
    void Apply() {
        char ratio[64];
        std::snprintf(ratio, sizeof(ratio), "Text contrast %.1f:1",
                      Contrast(theme_.text, theme_.panel));
        Widgets().Set(contrast_, std::string(ratio));
        if (GetViewer() != nullptr) {
            GetViewer()->SetTheme(theme_);
        }
    }

    app::Theme theme_;
    app::WidgetPanel::Id accent_ = 0;
    app::WidgetPanel::Id background_ = 0;
    app::WidgetPanel::Id text_ = 0;
    app::WidgetPanel::Id font_ = 0;
    app::WidgetPanel::Id rounding_ = 0;
    app::WidgetPanel::Id pill_ = 0;
    app::WidgetPanel::Id spacing_ = 0;
    app::WidgetPanel::Id contrast_ = 0;
};

}  // namespace

ERNIE_REGISTER_DEMO(ThemeDemo, "viewer", "theme",
                    "The viewer restyled live: light, dark and "
                    "high-contrast themes, colours, text size, "
                    "rounding and spacing.");
