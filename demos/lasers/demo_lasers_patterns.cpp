// SPDX-License-Identifier: BSD-3-Clause
// A 532 nm projector over a table through each of its patterns, one
// of them written in Slang at run time.

#include "ernie/app/demo.h"
#include "ernie/app/viewer.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_pattern.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"
#include "util.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr std::array<const char*, 6> kPatterns = {
    "spot", "line", "grid", "dots", "fringes", "rings"};

/// Concentric rings, as a diffractive axicon or a ring light makes:
/// one ring every `period` of tan off the axis, each a thin line as
/// wide as the pixel's footprint and the beam, any scale (Ernie
/// normalises it over its extent).
constexpr const char* kRingsSlang = R"(
import light_pattern;

public struct Rings : ILightPattern {
    public float emission(LightPatternInput input, LightPatternParams p) {
        let period = p.get(0);
        let r = length(input.t);
        let off = r - period * round(r / period);
        let width = max(length(input.footprint), 2.0 * max(input.beam.x, input.beam.y));
        return r > 0.5 * period && abs(off) < 0.5 * width ? 1.0 / width : 0.0;
    }
}
)";

LightPatternModel Rings() {
    LightPatternModel model;
    model.id = "rings";
    model.name = "Rings";
    model.description = "Concentric rings every `period` of tan.";
    model.typeName = "Rings";
    model.slang = kRingsSlang;
    model.params = {{"period", 0.05f, "tan between rings"}};
    model.extent = 0.4;
    return model;
}

/// A PNG of vertical sinusoidal fringes, `periods` across 256 x 16
/// texels, written to the demos' cache; returns its path.
std::filesystem::path Fringes(int periods) {
    constexpr std::int64_t kCols = 256;
    constexpr std::int64_t kRows = 16;
    Tensor image = Tensor::Zeros(DType::UInt8, {kRows, kCols, 3});
    auto* data = static_cast<std::uint8_t*>(image.Data());
    for (std::int64_t row = 0; row < kRows; ++row) {
        for (std::int64_t col = 0; col < kCols; ++col) {
            const double phase =
                2.0 * M_PI * periods * (col + 0.5) / kCols;
            const auto value = static_cast<std::uint8_t>(
                std::lround(255.0 * (0.5 + 0.5 * std::cos(phase))));
            for (int k = 0; k < 3; ++k) {
                data[(row * kCols + col) * 3 + k] = value;
            }
        }
    }
    const std::filesystem::path path =
        demos::CacheDir() / "laser_fringes.png";
    std::filesystem::create_directories(path.parent_path());
    app::WritePng(path, image);
    return path;
}

void Box(Scene& scene, const Path& path, glm::dvec3 centre,
         glm::dvec3 size, glm::dvec3 colour) {
    Cube::Define(scene, path).SetSize(1.0).SetDisplayColor(colour);
    SetXform(scene, path, {.translate = centre, .scale = size});
}

class PatternsDemo : public app::Demo {
  public:
    PatternsDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        fringes_ = Fringes(12);
        // The custom pattern, with whichever renderers draw the
        // scene.
        if (!GetRenderer().Delegate().HasLightPattern("rings")) {
            GetRenderer().Delegate().AddLightPattern(Rings());
        }
        if (viewer != nullptr &&
            !viewer->Delegate().HasLightPattern("rings")) {
            viewer->Delegate().AddLightPattern(Rings());
        }
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        GroundPlane::Define(scene, "/World/Floor")
            .SetChecker(true)
            .SetCheckerColor0(glm::dvec3(0.6))
            .SetCheckerColor1(glm::dvec3(0.5));
        Box(scene, "/World/Wall", {0.0, 1.5, 1.0}, {4.0, 0.1, 2.0},
            glm::dvec3(0.7));
        Box(scene, "/World/Block", {-0.5, 0.3, 0.2}, {0.4, 0.4, 0.4},
            glm::dvec3(0.75));
        Sphere::Define(scene, "/World/Ball")
            .SetRadius(0.25)
            .SetDisplayColor(glm::dvec3(0.8));
        SetXform(scene, "/World/Ball",
                 {.translate = glm::dvec3(0.45, 0.2, 0.25)});
        // A dim room light.
        SphereLight::Define(scene, "/World/Lamp")
            .SetRadius(0.1)
            .SetPhotometricPower(300.0)
            .SetTransform(LookAtTransform(
                {-1.5, -1.0, 2.5}, {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}));
        // The projector 2 m up and back, looking down onto the table:
        // 30 mW at 532 nm, 1 mm waist.
        LaserLight::Define(scene, "/World/Projector")
            .SetWavelength(532.0)
            .SetPower(0.03)
            .SetWaistRadius(glm::dvec2(0.001))
            .SetTransform(LookAtTransform(
                {0.0, -1.6, 2.2}, {0.0, 0.4, 0.2}, {0.0, 0.0, 1.0}));
        // Exposed for the patterns' tens of lux (30 mW of green is
        // about 18 lumens), the same with the laser on or off.
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(20.0)
            .SetSensorNoise(false)
            .SetExposureIso(800.0)
            .SetExposureFStop(2.0)
            .SetExposureTime(1.0 / 30.0)
            .SetTransform(LookAtTransform(
                {1.6, -2.6, 1.7}, {0.0, 0.4, 0.3}, {0.0, 0.0, 1.0}));
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution({448, 300})
            .SetAmbient(0.0);
        Show(0);

        std::vector<std::string> names(kPatterns.begin(),
                                       kPatterns.end());
        Widgets().Combo("Pattern", std::move(names), 0,
                        [this](std::size_t index) { Show(index); });
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
        }
    }

    /// Every pattern side by side, two rows of three.
    void SaveOutput(const std::filesystem::path& path) override {
        const std::vector<Tensor> images = RenderAll();
        app::WritePng(path, demos::SideBySide(images));
    }

    void TestFinal() override {
        LaserLight laser(GetScene(), "/World/Projector");
        laser.SetPower(0.0);
        const Tensor dark = GetRenderer().Render(GetScene());
        laser.SetPower(0.03);
        const std::vector<Tensor> images = RenderAll();
        std::array<double, kPatterns.size()> lit{};
        std::array<double, kPatterns.size()> aspect{};
        for (std::size_t i = 0; i < images.size(); ++i) {
            const Lit seen = LitBy(images[i], dark);
            lit[i] = seen.fraction;
            aspect[i] = seen.aspect;
            std::printf("%s: %.2f%% of the image lit, %.1f wide to "
                        "tall\n",
                        kPatterns[i], 100.0 * lit[i], aspect[i]);
            Check(lit[i] > 0.0,
                  std::string("the ") + kPatterns[i] +
                      " pattern should light the scene");
        }
        Check(lit[0] < 0.01, "the bare beam should light a spot");
        for (std::size_t i = 1; i < lit.size(); ++i) {
            Check(lit[i] > lit[0],
                  std::string("the ") + kPatterns[i] +
                      " pattern should light more than the spot");
        }
        Check(aspect[1] > 2.0, "the line should be wider than tall");
        Check(lit[4] > lit[2],
              "the fringes should light more than the grid");
    }

  private:
    void Show(std::size_t index) {
        index_ = index;
        LaserLight laser(GetScene(), "/World/Projector");
        laser.SetPattern(kPatterns[index]).SetZeroOrder(0.0);
        const glm::dvec2 field(50.0, 40.0);
        switch (index) {
        case 1:  // a flat 50 degree line from a Powell lens
            laser.SetFanAngle(50.0).SetLineProfile("flat");
            break;
        case 2:  // a 9 x 9 diffractive grid, 3 degrees apart
            laser.SetGridColumns(9)
                .SetGridRows(9)
                .SetGridPitch({3.0, 3.0})
                .SetEfficiency(0.85)
                .SetZeroOrder(0.05);
            break;
        case 3:  // 40 x 30 dots in 3 x 3 tiles, as a depth camera's
            laser.SetPattern("dots")
                .SetDotColumns(40)
                .SetDotRows(30)
                .SetDotTileColumns(3)
                .SetDotTileRows(3)
                .SetPatternFieldOfView(field)
                .SetEfficiency(0.8);
            break;
        case 4:  // fringes from an image, as a phase-shift
                 // projector's
            laser.SetPattern("image")
                .SetPatternImage(fringes_.string())
                .SetPatternFieldOfView(field);
            break;
        case 5:  // rings, a pattern registered at run time
            laser.SetPatternParams({0.05f});
            break;
        default:
            break;
        }
    }

    /// Every pattern's image in turn, RGBA uint8; leaves the pattern
    /// as it was.
    std::vector<Tensor> RenderAll() {
        const std::size_t current = index_;
        std::vector<Tensor> images;
        for (std::size_t i = 0; i < kPatterns.size(); ++i) {
            Show(i);
            images.push_back(GetRenderer().Render(GetScene()));
        }
        Show(current);
        return images;
    }

    struct Lit {
        double fraction = 0.0;  // of the image
        double aspect = 0.0;    // its extent's width over its height
    };

    /// What the laser lights in `image` against `dark`, the scene
    /// without it: pixels whose green rose by more than 25.
    static Lit LitBy(const Tensor& image, const Tensor& dark) {
        const auto rows = image.Shape()[0];
        const auto cols = image.Shape()[1];
        const auto channels = image.Shape()[2];
        const auto* a =
            static_cast<const std::uint8_t*>(image.Data());
        const auto* b = static_cast<const std::uint8_t*>(dark.Data());
        std::int64_t count = 0;
        std::int64_t left = cols;
        std::int64_t right = -1;
        std::int64_t top = rows;
        std::int64_t bottom = -1;
        for (std::int64_t r = 0; r < rows; ++r) {
            for (std::int64_t c = 0; c < cols; ++c) {
                const std::int64_t i = (r * cols + c) * channels + 1;
                if (static_cast<int>(a[i]) - static_cast<int>(b[i]) >
                    25) {
                    ++count;
                    left = std::min(left, c);
                    right = std::max(right, c);
                    top = std::min(top, r);
                    bottom = std::max(bottom, r);
                }
            }
        }
        if (count == 0) {
            return {};
        }
        return {static_cast<double>(count) /
                    static_cast<double>(rows * cols),
                static_cast<double>(right - left + 1) /
                    static_cast<double>(bottom - top + 1)};
    }

    std::filesystem::path fringes_;
    std::size_t index_ = 0;
};

}  // namespace

ERNIE_REGISTER_DEMO(
    PatternsDemo, "lasers", "patterns",
    "A projector's patterns: spot, line, grid, dots, fringes and "
    "rings written in Slang.");
