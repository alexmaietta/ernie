// SPDX-License-Identifier: BSD-3-Clause
// Measured materials under the Sun, seen from overhead by three
// cameras: visible RGB, a near-infrared camera (760 to 1000 nm) and a
// shortwave-infrared InGaAs camera filtered to 1.57 to 1.65 um, as
// Landsat 8's band 6 tells snow from cloud. Each patch
// reflects as the material does (`inputs:reflectanceLibrary`, from
// the ECOSTRESS Spectral Library): grass dark in the visible and
// bright in the near infrared, snow white in the visible and dark
// past 1.4 um, asphalt dark throughout.
//
// References:
// - S. K. Meerdink et al., "The ECOSTRESS spectral library version
//   1.0", Remote Sensing of Environment 230:111196, 2019.
// - G. P. Asner, "Biophysical and biochemical sources of variability
//   in canopy reflectance", Remote Sensing of Environment 64(3), 1998
//   (the red edge).

#include "ernie/app/demo.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"
#include "util.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr std::array<const char*, 10> kMaterials = {
    "grass",     "dry_grass", "loam",  "sand",
    "concrete",  "asphalt",   "brick", "white_coating",
    "aluminium", "fine_snow"};

// The cameras: their spectral range and bands, and for a mono one its
// quantum efficiency as (nm, value) pairs.
struct Band {
    const char* name;
    const char* product;
    glm::dvec2 range;
    int bands;
    std::vector<float> qe;  // empty: the RGB camera's own
};

const std::array<Band, 3> kBands = {{
    {"visible RGB", "/Render/Rgb", {380.0, 720.0}, 17, {}},
    // A visible-blocking filter on silicon.
    {"NIR 760-1000 nm",
     "/Render/Nir",
     {760.0, 1000.0},
     12,
     {760.0f, 0.0f, 779.0f, 0.0f, 781.0f, 0.35f, 900.0f, 0.25f,
      1000.0f, 0.05f}},
    // InGaAs behind a 1.57 to 1.65 um filter (Landsat 8's band 6).
    {"SWIR 1.57-1.65 um",
     "/Render/Swir",
     {1550.0, 1670.0},
     6,
     {1550.0f, 0.0f, 1569.0f, 0.0f, 1571.0f, 0.8f, 1649.0f, 0.8f,
      1651.0f, 0.0f, 1670.0f, 0.0f}},
}};

constexpr glm::dvec2 kResolution(480.0, 120.0);
constexpr double kHeight = 9.0;  // the cameras above the patches
constexpr double kFocalLength = 24.0;  // mm
constexpr double kAperture = 20.955;   // mm, USD's horizontal default
constexpr double kPitch = 0.6;         // between patches' centres, m
constexpr double kPatch = 0.5;         // a patch's side, m

double PatchX(std::size_t i) {
    return kPitch *
           (static_cast<double>(i) -
            0.5 * static_cast<double>(kMaterials.size() - 1));
}

class NirSwirDemo : public app::Demo {
  public:
    NirSwirDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        // Black ground, so only the patches show.
        GroundPlane::Define(scene, "/World/Ground")
            .SetChecker(true)
            .SetCheckerColor0(glm::dvec3(0.02))
            .SetCheckerColor1(glm::dvec3(0.02));
        Xform::Define(scene, "/World/Looks");
        Xform::Define(scene, "/World/Patches");
        for (std::size_t i = 0; i < kMaterials.size(); ++i) {
            const std::string id = kMaterials[i];
            const Path look("/World/Looks/" + id);
            Material::Define(scene, look).SetReflectanceLibrary(id);
            const Path patch("/World/Patches/" + id);
            Cube::Define(scene, patch)
                .SetSize(1.0)
                .SetMaterialBinding(look);
            SetXform(scene, patch,
                     {.translate = glm::dvec3(PatchX(i), 0.0, 0.01),
                      .scale = glm::dvec3(kPatch, kPatch, 0.02)});
        }
        // The Sun, its measured spectrum, 50 degrees up.
        DistantLight::Define(scene, "/World/Sun")
            .SetIrradiance(1000.0)
            .SetPhysicalIlluminant(std::string(tokens::kSun))
            .SetAngle(0.53)
            .SetTransform(DistantLightTransform(
                glm::normalize(glm::dvec3(0.3, -0.5, 1.0))));
        // The cameras straight down, exposed for sunlight by the
        // sunny 16 rule: f/16, 1/100 s at ISO 100; the SWIR camera's
        // narrow band four times as long.
        for (std::size_t c = 0; c < kBands.size(); ++c) {
            const Band& band = kBands[c];
            const Path camera = CameraPath(c);
            Camera view = Camera::Define(scene, camera);
            view.SetFocalLength(kFocalLength)
                .SetHorizontalAperture(kAperture)
                .SetVerticalAperture(kAperture * kResolution.y /
                                     kResolution.x)
                .SetSensorNoise(false)
                .SetSpectralRange(band.range)
                .SetSpectralBands(band.bands)
                .SetExposureFStop(16.0)
                .SetExposureTime(c == 2 ? 0.04 : 0.01)
                .SetExposureIso(100.0)
                .SetTransform(LookAtTransform({0.0, 0.0, kHeight},
                                              {0.0, 0.0, 0.0},
                                              {0.0, 1.0, 0.0}));
            if (!band.qe.empty()) {
                view.SetSensorType("mono").SetSensorQeMono(band.qe);
            }
            const Path ldr = Path(band.product).Child("LdrColor");
            RenderVar::Define(scene, ldr).SetSourceName("LdrColor");
            RenderProduct::Define(scene, band.product)
                .SetCamera(camera)
                .SetResolution(kResolution)
                .SetOrderedVars({ldr});
        }
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(CameraPath(0))
            .SetResolution(kResolution)
            .SetAmbient(0.0);

        std::vector<std::string> names;
        for (const Band& band : kBands) {
            names.emplace_back(band.name);
        }
        Widgets().Combo(
            "Camera", std::move(names), 0, [this](std::size_t index) {
                if (GetViewer() != nullptr) {
                    GetViewer()->SetActiveCamera(CameraPath(index));
                }
            });
        if (viewer != nullptr) {
            viewer->SetActiveCamera(CameraPath(0));
        }
    }

    /// The three cameras' images, one above another.
    void SaveOutput(const std::filesystem::path& path) override {
        app::WritePng(path, demos::SideBySide(RenderAll()));
    }

    void TestFinal() override {
        const std::vector<Tensor> images = RenderAll();
        std::array<std::array<double, kMaterials.size()>, 3> mean{};
        for (std::size_t c = 0; c < images.size(); ++c) {
            std::printf("%s:", kBands[c].name);
            for (std::size_t i = 0; i < kMaterials.size(); ++i) {
                mean[c][i] = Brightness(images[c], PatchX(i));
                std::printf(" %s %.2f", kMaterials[i], mean[c][i]);
            }
            std::printf("\n");
        }
        const auto index = [](std::string_view id) {
            return static_cast<std::size_t>(
                std::find(kMaterials.begin(), kMaterials.end(), id) -
                kMaterials.begin());
        };
        const std::size_t grass = index("grass");
        const std::size_t concrete = index("concrete");
        const std::size_t snow = index("fine_snow");
        const std::size_t asphalt = index("asphalt");
        Check(
            mean[0][grass] < mean[0][concrete],
            "grass should look darker than concrete in the visible");
        Check(mean[1][grass] > mean[1][concrete],
              "grass should outshine concrete in the near infrared");
        Check(mean[0][snow] > mean[0][concrete],
              "snow should be the brighter in the visible");
        Check(mean[2][snow] < mean[2][concrete],
              "snow should be darker than concrete in the SWIR");
        for (std::size_t c = 0; c < images.size(); ++c) {
            Check(
                mean[c][asphalt] < mean[c][concrete],
                std::string("asphalt should be darker than concrete "
                            "in the ") +
                    kBands[c].name);
        }
    }

  private:
    static Path CameraPath(std::size_t c) {
        return Path("/World/Camera" + std::to_string(c));
    }

    /// Every camera's LdrColor, uint8 (rows, cols, 4).
    std::vector<Tensor> RenderAll() {
        std::vector<Path> products;
        for (const Band& band : kBands) {
            products.emplace_back(band.product);
        }
        RenderProductSetOutputs outputs =
            GetRenderer().Step(GetScene(), products);
        std::vector<Tensor> images;
        for (const Band& band : kBands) {
            images.push_back(outputs.at(Path(band.product))
                                 .vars.at("LdrColor")
                                 .Map(MapDevice::Cpu)
                                 .tensor);
        }
        return images;
    }

    /// The mean value (0 to 1, of the channels' mean) over the middle
    /// half of the patch centred at `x`.
    static double Brightness(const Tensor& image, double x) {
        const auto rows = static_cast<int>(image.Shape()[0]);
        const auto cols = static_cast<int>(image.Shape()[1]);
        const auto channels = static_cast<int>(image.Shape()[2]);
        const double pixels =
            kFocalLength / kAperture * cols / kHeight;
        const int col = static_cast<int>(0.5 * cols + x * pixels);
        const int half = static_cast<int>(0.25 * kPatch * pixels);
        const auto* data =
            static_cast<const std::uint8_t*>(image.Data());
        double sum = 0.0;
        int count = 0;
        for (int r = rows / 2 - half; r <= rows / 2 + half; ++r) {
            for (int c = col - half; c <= col + half; ++c) {
                for (int k = 0; k < 3; ++k) {
                    sum += data[(r * cols + c) * channels + k];
                    ++count;
                }
            }
        }
        return sum / count / 255.0;
    }
};

}  // namespace

ERNIE_REGISTER_DEMO(
    NirSwirDemo, "cameras", "nir_swir",
    "Measured materials under the Sun through visible, near-infrared "
    "and shortwave-infrared cameras.");
