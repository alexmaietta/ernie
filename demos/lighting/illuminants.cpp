// SPDX-License-Identifier: BSD-3-Clause
// One spot light, four illuminants (physical:illuminant), after the Omniverse RTX physical
// lighting page's illuminant figure: "white", "illuminant D" at 4000 K, "blackbody" at 2000 K and
// a "custom" fluorescent spectrum, each at the same 1600 lumens (photometric:power), lighting a
// checkered wall with a colour chart and two spheres in a dark room. The C++ twin of
// illuminants.py.
//
//     ernie --demo lighting/illuminants [--viewer null --output illuminants.png]
//
// The lumens stay the same, so each image has about the same brightness; the illuminant changes
// only the colour of the light, and how the chart's patches render under it. The page's custom
// illuminant is CIE F5; here the custom spectrum is fluorescent-like rather than F5 itself:
// mercury emission lines at 405, 436, 546 and 578 nm over a broad halophosphate band.
//
// References:
// - NVIDIA, "Omniverse RTX Renderer: Lighting, Physical Lighting",
//   https://docs.omniverse.nvidia.com/materials-and-rendering/latest/lighting.html
// - D. Pascale, "RGB coordinates of the Macbeth ColorChecker", BabelColor, 2006 (the patches'
//   sRGB values).

#include "ernie/app/demo.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"
#include "util.h"

#include <imgui.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

// The ColorChecker's 24 patches in sRGB, row by row from dark skin to black.
constexpr std::array<std::array<int, 3>, 24> kChart = {{
    {115, 82, 68},   {194, 150, 130}, {98, 122, 157}, {87, 108, 67},   {133, 128, 177},
    {103, 189, 170}, {214, 126, 44},  {80, 91, 166},  {193, 90, 99},   {94, 60, 108},
    {157, 188, 64},  {224, 163, 46},  {56, 61, 150},  {70, 148, 73},   {175, 54, 60},
    {231, 199, 31},  {187, 86, 149},  {8, 133, 161},  {243, 243, 242}, {200, 200, 200},
    {160, 160, 160}, {122, 122, 121}, {85, 85, 85},   {52, 52, 52},
}};

struct Illuminant {
    const char* name;
    std::string_view token;
    double kelvin;
};

const std::array<Illuminant, 4> kIlluminants = {{
    {"white", tokens::kWhite, 6500.0},
    {"illuminant D 4000 K", tokens::kIlluminantD, 4000.0},
    {"blackbody 2000 K", tokens::kBlackbody, 2000.0},
    {"custom (fluorescent)", tokens::kCustom, 6500.0},
}};

/// (nm, value) pairs of a fluorescent-like spectrum: mercury lines over broad yellow and blue
/// phosphor bands, weighted to CIE F5's white point (x 0.311, y 0.337 against F5's 0.314, 0.335).
std::vector<float> Fluorescent() {
    std::vector<float> pairs;
    for (int nm = 380; nm <= 780; ++nm) {
        const double w = nm;
        double band = std::exp(-0.5 * std::pow((w - 570.0) / 80.0, 2.0));
        band += 0.8 * std::exp(-0.5 * std::pow((w - 470.0) / 30.0, 2.0));
        for (const auto [line, weight] : {std::pair{404.7, 0.8}, std::pair{435.8, 2.0},
                                          std::pair{546.1, 0.5}, std::pair{578.0, 0.2}}) {
            band += weight * std::exp(-0.5 * std::pow((w - line) / 1.5, 2.0));
        }
        pairs.push_back(static_cast<float>(w));
        pairs.push_back(static_cast<float>(band));
    }
    return pairs;
}

void Box(Scene& scene, const Path& path, glm::dvec3 centre, glm::dvec3 size, glm::dvec3 colour) {
    Cube::Define(scene, path).SetSize(1.0).SetDisplayColor(colour);
    SetXform(scene, path, {.translate = centre, .scale = size});
}

class IlluminantsDemo : public app::Demo {
  public:
    IlluminantsDemo(app::Viewer* viewer, const app::DemoOptions& options) : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        GroundPlane::Define(scene, "/World/Floor")
            .SetChecker(true)
            .SetCheckerSize(0.25)
            .SetCheckerColor0({0.35, 0.35, 0.35})
            .SetCheckerColor1({0.2, 0.2, 0.2});
        // A checkered back wall 3 m wide and 3 m high at y = 1.
        Xform::Define(scene, "/World/Wall");
        for (int i = 0; i < 12; ++i) {
            for (int j = 0; j < 12; ++j) {
                const double grey = (i + j) % 2 == 0 ? 0.32 : 0.2;
                Box(scene, "/World/Wall/T" + std::to_string(i) + "_" + std::to_string(j),
                    {-1.5 + 0.25 * (i + 0.5), 1.05, 0.25 * (j + 0.5)}, {0.25, 0.1, 0.25},
                    glm::dvec3(grey));
            }
        }
        // The colour chart on the wall, 6 x 4 patches of 10 cm on a black card.
        Xform::Define(scene, "/World/Chart");
        Box(scene, "/World/Chart/Card", {-0.25, 0.99, 1.3}, {0.68, 0.01, 0.47}, glm::dvec3(0.03));
        for (int k = 0; k < static_cast<int>(kChart.size()); ++k) {
            const int col = k % 6;
            const int row = k / 6;
            const glm::dvec3 srgb(kChart[k][0] / 255.0, kChart[k][1] / 255.0, kChart[k][2] / 255.0);
            Box(scene, "/World/Chart/P" + std::to_string(k),
                {-0.525 + 0.11 * col, 0.98, 1.465 - 0.11 * row}, {0.1, 0.02, 0.1},
                demos::SrgbToLinear(srgb));
        }
        Sphere::Define(scene, "/World/Ball").SetRadius(0.15).SetDisplayColor(glm::dvec3(0.5));
        SetXform(scene, "/World/Ball", {.translate = glm::dvec3(0.25, 0.6, 0.15)});
        Sphere::Define(scene, "/World/Ball2").SetRadius(0.15).SetDisplayColor(glm::dvec3(0.8));
        SetXform(scene, "/World/Ball2", {.translate = glm::dvec3(-0.1, 0.55, 0.15)});
        // A spot near the ceiling at the left, grazing the wall: 1600 lumens.
        SphereLight::Define(scene, "/World/Spot")
            .SetRadius(0.05)
            .SetShapingConeAngle(40.0)
            .SetShapingConeSoftness(0.4)
            .SetPhotometricPower(1600.0)
            .SetPhysicalCustomIlluminant(Fluorescent())
            .SetTransform(LookAtTransform({-1.2, 0.6, 2.9}, {-0.25, 1.0, 1.0}, {0.0, 1.0, 0.0}));
        // ISO 400, f/2, 1/125 s: the spot's lumens fill only its 40 degree cone, so the wall in it
        // is bright; the cone's centre just clips.
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(18.0)
            .SetSensorNoise(false)
            .SetExposureTime(1.0 / 125.0)
            .SetExposureIso(400.0)
            .SetExposureFStop(2.0)
            .SetTransform(LookAtTransform({0.0, -2.6, 1.3}, {0.0, 1.0, 1.2}, {0.0, 0.0, 1.0}));
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution({448, 300})
            .SetAmbient(0.0);
        LightWith(0);

        if (viewer != nullptr) {
            // In the viewer, a panel switches the illuminant.
            viewer->SetActiveCamera("/World/Camera");
            viewer->AddPanel("Illuminant", app::Dock::Right, [this](app::Viewer&) {
                int index = illuminant_;
                for (int i = 0; i < static_cast<int>(kIlluminants.size()); ++i) {
                    if (ImGui::RadioButton(kIlluminants[i].name, i == index)) {
                        index = i;
                    }
                }
                if (index != illuminant_) {
                    LightWith(index);
                }
            });
        }
    }

    /// The four illuminants side by side.
    void SaveOutput(const std::filesystem::path& path) override {
        app::WritePng(path, demos::SideBySide(RenderAll()));
    }

    void TestFinal() override {
        std::array<double, 4> luminance{};
        std::array<double, 4> warmth{};
        const std::vector<Tensor> images = RenderAll();
        for (std::size_t i = 0; i < images.size(); ++i) {
            const glm::dvec3 mean = demos::MeanLinearRgb(images[i]);
            luminance[i] = 0.2126 * mean.r + 0.7152 * mean.g + 0.0722 * mean.b;
            warmth[i] = mean.r / std::max(mean.b, 1e-6);
            std::printf("%s: mean luminance %.4f, red/blue %.2f\n", kIlluminants[i].name,
                        luminance[i], warmth[i]);
        }
        // The same 1600 lumens: about the same brightness under every illuminant.
        Check(luminance[0] > 0.005, "the spot should light the wall");
        Check(*std::max_element(luminance.begin(), luminance.end()) <
                  1.2 * *std::min_element(luminance.begin(), luminance.end()),
              "the same lumens should look as bright");
        // Only the colour changes: 4000 K warmer than white, 2000 K warmer still.
        Check(warmth[0] > 0.7 && warmth[0] < 1.4, "the white light should be about neutral");
        Check(warmth[1] > 1.15 * warmth[0], "illuminant D at 4000 K should be warmer than white");
        Check(warmth[2] > 1.5 * warmth[1], "a 2000 K blackbody should be warmer than 4000 K");
        // The mercury lines at 405 and 436 nm tip the fluorescent spectrum blue of white.
        Check(warmth[3] < 0.9 * warmth[0], "the fluorescent light should be cooler than white");
    }

  private:
    void LightWith(int index) {
        illuminant_ = index;
        SphereLight(GetScene(), "/World/Spot")
            .SetPhysicalIlluminant(std::string(kIlluminants[index].token))
            .SetColorTemperature(kIlluminants[index].kelvin);
    }

    /// The scene under each illuminant in turn, RGBA uint8; leaves the light as it was.
    std::vector<Tensor> RenderAll() {
        const int current = illuminant_;
        std::vector<Tensor> images;
        for (int i = 0; i < static_cast<int>(kIlluminants.size()); ++i) {
            LightWith(i);
            images.push_back(GetRenderer().Render(GetScene()));
        }
        LightWith(current);
        return images;
    }

    int illuminant_ = 0;
};

}  // namespace

ERNIE_REGISTER_DEMO(IlluminantsDemo, "lighting", "illuminants",
                    "One spot light under four illuminants at the same 1600 lumens.");
