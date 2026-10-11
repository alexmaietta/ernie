// SPDX-License-Identifier: BSD-3-Clause
// Perseverance's cameras on a patch of Mars: the Navcam and front
// Hazcam stereo pairs (f-theta lenses on the CMV-20000, 12-bit
// numbers demosaiced, sent as JPEG) and the Mastcam-Z pair at its
// widest zoom, toed in, with its filter wheels' narrow bands into the
// near infrared, all as published, under the Sun's measured spectrum
// at Mars's distance through its dusty air. Each camera is binned to
// 320 x 240 (16 x 16 of the engineering cameras' pixels, 5 x 5 of
// Mastcam-Z's).
//
// References:
// - J. N. Maki et al., "The Mars 2020 Engineering Cameras and
//   Microphone on the Perseverance Rover: A Next-Generation Imaging
//   System for Mars Exploration", Space Science Reviews 216:137,
//   2020, https://doi.org/10.1007/s11214-020-00765-9: Table 2 and the
//   sections on the detector and flight software.
// - J. F. Bell III et al., "The Mars 2020 Perseverance Rover Mast
//   Camera Zoom (Mastcam-Z) Multispectral, Stereoscopic Imaging
//   Investigation", Space Science Reviews 217:24, 2021,
//   https://doi.org/10.1007/s11214-020-00755-x: Tables 3 and 5 and
//   the focal plane section.
// - J. F. Bell III et al., "Mineralogic and compositional properties
//   of Martian soil and dust: Results from Mars Pathfinder", Journal
//   of Geophysical Research 105(E1):1721-1755, 2000 (soil
//   reflectance, dark in the blue and about 0.3 in the red and near
//   infrared).
// - J. Kannala and S. S. Brandt, "A Generic Camera Model and
//   Calibration Method for Conventional, Wide-Angle, and Fish-Eye
//   Lenses", IEEE TPAMI 28(8):1335-1340, 2006 (the f-theta lens's
//   k1).

#include "ernie/app/demo.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/terrain_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"
#include "util.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numbers>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr double kPi = std::numbers::pi;
constexpr double kDegree = kPi / 180.0;

/// The Sun's irradiance at Mars's mean distance, 1.5237 au, and its
/// angular diameter there, degrees.
constexpr double kMarsIrradiance = 1361.0 / (1.5237 * 1.5237);
constexpr double kSunAngle = 0.35;

/// Every product's size: the detectors binned.
constexpr int kWidth = 320;
constexpr int kHeight = 240;

// ------------------------------------------------- engineering
// cameras

/// The Navcams' and Hazcams' CMV-20000 (Maki et al. 2020, Table 2):
/// 5120 x 3840 pixels of 6.4 um, 0.24 DN per electron, 8 e- dark
/// noise, a 15000 e- full well and a 12-bit ADC.
constexpr double kCmvPixelUm = 6.4;
constexpr int kCmvWidth = 5120;
constexpr int kCmvHeight = 3840;
constexpr double kCmvElectronsPerDn = 1.0 / 0.24;
constexpr double kCmvDarkNoise = 8.0;
constexpr double kCmvFullWell = 15000.0;

/// An engineering camera's optics (Table 2): focal length (mm),
/// f-number, best focus (m), stereo baseline (m) and fields of view
/// (horizontal, vertical, diagonal; degrees).
struct Optics {
    const char* name;
    double focalMm;
    double fNumber;
    double focusM;
    double baselineM;
    std::array<double, 3> fieldsDeg;
};

constexpr Optics kNavcam{"Navcam", 19.1,  12.0,
                         3.5,      0.424, {96.0, 73.0, 120.0}};
constexpr Optics kHazcam{"Hazcam", 14.0,  12.0,
                         0.9,      0.248, {136.0, 102.0, 172.0}};
/// The rear Hazcams: the front pair's optics, 93.4 cm apart.
constexpr Optics kRearHazcam{
    "RearHazcam", 14.0, 12.0, 0.9, 0.934, {136.0, 102.0, 172.0}};

/// The half widths the fields reach on the detector, mm: across,
/// down and to the corner.
std::array<double, 3> CmvReaches() {
    const double halfWidth = 0.5e-3 * kCmvPixelUm * kCmvWidth;
    const double halfHeight = 0.5e-3 * kCmvPixelUm * kCmvHeight;
    return {halfWidth, halfHeight, std::hypot(halfWidth, halfHeight)};
}

/// The f-theta lens's Kannala-Brandt k1 for the published fields: the
/// least squares of theta (1 + k1 theta^2) = r / f over the three,
/// theta each field's half angle and r the half width it reaches.
/// The paper gives the calibrated model only as CAHVORE, without its
/// coefficients.
double FThetaK1(const Optics& optics) {
    const std::array<double, 3> reaches = CmvReaches();
    double numerator = 0.0;
    double denominator = 0.0;
    for (std::size_t i = 0; i < 3; ++i) {
        const double theta = 0.5 * optics.fieldsDeg[i] * kDegree;
        const double r = reaches[i] / optics.focalMm;
        const double cube = theta * theta * theta;
        numerator += cube * (r - theta);
        denominator += cube * cube;
    }
    return numerator / denominator;
}

/// The angle off axis (radians) that lands `r` focal lengths out
/// through the f-theta lens: theta (1 + k1 theta^2) = r, by Newton.
double FThetaAngle(double r, double k1) {
    double theta = r;
    for (int i = 0; i < 20; ++i) {
        const double f = theta * (1.0 + k1 * theta * theta) - r;
        theta -= f / (1.0 + 3.0 * k1 * theta * theta);
    }
    return theta;
}

// ------------------------------------------------------------
// Mastcam-Z

/// The KAI-2020CM (Bell et al. 2021, Table 5): 1600 x 1200 pixels of
/// 7.4 um, 15.6 e-/DN and an 11-bit ADC; each camera's full well and
/// read noise.
constexpr double kKaiPixelUm = 7.4;
constexpr int kKaiWidth = 1600;
constexpr int kKaiHeight = 1200;
constexpr double kKaiElectronsPerDn = 15.6;
constexpr double kKaiFullWell[2] = {21826.0, 21818.0};
constexpr double kKaiReadNoise[2] = {22.0, 21.0};

/// The zoom (26 mm at f/6.7 to 110 mm at f/9.5), the cameras' 24.4 cm
/// baseline and each one's 1.15 degree toe-in (2.3 between them), and
/// the detector's ~40% peak QE through its Bayer microlenses.
constexpr double kShortestFocalMm = 26.0;
constexpr double kLongestFocalMm = 110.0;
constexpr double kWidestFNumber = 6.7;
constexpr double kNarrowestFNumber = 9.5;
constexpr double kMastcamBaselineM = 0.244;
constexpr double kToeInDeg = 1.15;
constexpr double kPeakQe = 0.40;

/// The f-number at focal length `focalMm`, linearly between the
/// zoom's ends (the paper gives only those).
double FNumberAt(double focalMm) {
    const double t = (focalMm - kShortestFocalMm) /
                     (kLongestFocalMm - kShortestFocalMm);
    return kWidestFNumber + t * (kNarrowestFNumber - kWidestFNumber);
}

/// A filter's Table 3 effective wavelength and half width at half
/// maximum, nm, and its neutral density.
struct Filter {
    double effectiveNm;
    double hwhmNm;
    double neutralDensity;
};

/// Each camera's filter wheel (Table 3): 0 the Bayer RGB's near-IR
/// cut-off (by its green channel), 1 to 6 narrow bands, 7 the solar
/// filter; the left's visible, the right's reaching 1022 nm.
constexpr Filter kFilters[2][8] = {{{544.0, 41.0, 0.0},
                                    {800.0, 9.0, 0.0},
                                    {754.0, 10.0, 0.0},
                                    {677.0, 11.0, 0.0},
                                    {605.0, 9.0, 0.0},
                                    {528.0, 11.0, 0.0},
                                    {442.0, 12.0, 0.0},
                                    {590.0, 88.0, 6.0}},
                                   {{544.0, 42.0, 0.0},
                                    {800.0, 9.0, 0.0},
                                    {866.0, 10.0, 0.0},
                                    {910.0, 12.0, 0.0},
                                    {939.0, 12.0, 0.0},
                                    {978.0, 10.0, 0.0},
                                    {1022.0, 19.0, 0.0},
                                    {880.0, 10.0, 5.0}}};

/// Filter 0's red, green and blue Bayer channels (Table 3).
constexpr Filter kBayer[2][3] = {
    {{630.0, 43.0, 0.0}, {544.0, 41.0, 0.0}, {480.0, 46.0, 0.0}},
    {{631.0, 43.0, 0.0}, {544.0, 42.0, 0.0}, {480.0, 46.0, 0.0}}};

/// A generic front-illuminated silicon response (Ernie's mono
/// sensor's), for the detector behind the narrow bands: the paper
/// publishes no curve.
constexpr std::array<double, 9> kSiliconNm = {
    350.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0};
constexpr std::array<double, 9> kSiliconQe = {
    0.15, 0.35, 0.55, 0.62, 0.55, 0.40, 0.22, 0.06, 0.0};

double Silicon(double nm) {
    if (nm <= kSiliconNm.front()) {
        return kSiliconQe.front();
    }
    for (std::size_t i = 1; i < kSiliconNm.size(); ++i) {
        if (nm <= kSiliconNm[i]) {
            const double t = (nm - kSiliconNm[i - 1]) /
                             (kSiliconNm[i] - kSiliconNm[i - 1]);
            return kSiliconQe[i - 1] +
                   t * (kSiliconQe[i] - kSiliconQe[i - 1]);
        }
    }
    return kSiliconQe.back();
}

// ---------------------------------------------------------- the
// scene

/// Mars soil's reflectance, roughly (Bell et al. 2000): dark in the
/// blue, rising to about 0.3 by 750 nm and flat into the near
/// infrared; and darker rock.
const std::vector<float> kSoil{350.0f, 0.05f, 450.0f,  0.08f,
                               550.0f, 0.14f, 650.0f,  0.25f,
                               750.0f, 0.30f, 1100.0f, 0.30f};
const std::vector<float> kRock{350.0f, 0.06f, 550.0f,  0.09f,
                               750.0f, 0.14f, 1100.0f, 0.15f};

/// Rocks on the ground ahead of the rover: x, y and radius, m.
constexpr std::array<std::array<double, 3>, 6> kRocks = {{
    {-1.2, 3.0, 0.25},
    {0.9, 4.5, 0.4},
    {-2.5, 7.0, 0.6},
    {2.0, 9.0, 0.5},
    {0.3, 1.6, 0.12},
    {-0.6, 2.2, 0.18},
}};

/// The rigs, each looking down its -z with +y up, a heading
/// (degrees clockwise from north, +y) and a tilt down: the mast's
/// Navcams and Mastcam-Z about 2 m up looking 15 degrees down ahead
/// (north), the front Hazcams 0.7 m up and 30 degrees down ahead, the
/// rear ones behind the rover looking back.
struct Rig {
    const char* path;
    glm::dvec3 eye;
    double downDeg;
    double headingDeg;
};
constexpr Rig kNavcamRig{
    "/World/Rover/Navcam", {0.0, 0.0, 2.0}, 15.0, 0.0};
constexpr Rig kHazcamRig{
    "/World/Rover/Hazcam", {0.0, 0.6, 0.7}, 30.0, 0.0};
constexpr Rig kRearHazcamRig{
    "/World/Rover/RearHazcam", {0.0, -0.9, 0.7}, 30.0, 180.0};
constexpr Rig kMastcamRig{
    "/World/Rover/MastcamZ", {0.3, 0.0, 2.0}, 15.0, 0.0};

glm::dmat4 RigTransform(const Rig& rig) {
    const double down = rig.downDeg * kDegree;
    const double heading = rig.headingDeg * kDegree;
    const glm::dvec3 forward(std::sin(heading) * std::cos(down),
                             std::cos(heading) * std::cos(down),
                             -std::sin(down));
    return LookAtTransform(rig.eye, rig.eye + forward,
                           {0.0, 0.0, 1.0});
}

/// Rig-local pose of an eye `x` along the rig's x, turned `yaw`
/// (radians) about its +y.
glm::dmat4 EyePose(double x, double yaw) {
    glm::dmat4 pose =
        glm::rotate(glm::dmat4(1.0), yaw, glm::dvec3(0.0, 1.0, 0.0));
    pose[3] = glm::dvec4(x, 0.0, 0.0, 1.0);
    return pose;
}

/// Product `product`'s var `var` on the host.
Tensor Var(RenderProductSetOutputs& outputs, const Path& product,
           const char* var) {
    return outputs.at(product)
        .vars.at(var)
        .Map(MapDevice::Cpu)
        .tensor;
}

constexpr const char* kEyes[2] = {"Left", "Right"};

/// The cameras the viewer can show: name and camera (a pair's left
/// eye).
const std::array<std::pair<const char*, const char*>, 5> kViews = {{
    {"Overview", "/World/View"},
    {"Navcam", "/World/Rover/Navcam/Left"},
    {"Front Hazcam", "/World/Rover/Hazcam/Left"},
    {"Rear Hazcam", "/World/Rover/RearHazcam/Left"},
    {"Mastcam-Z", "/World/Rover/MastcamZ/Left"},
}};

/// Filter `filter` of eye `e`'s wheel, as Table 3 names it: L0 to
/// L7, R0 to R7.
std::string FilterName(int e, int filter) {
    const Filter& band = kFilters[e][filter];
    std::string name = std::string(e == 0 ? "L" : "R") +
                       std::to_string(filter) + " ";
    if (filter == 0) {
        return name + "Bayer RGB";
    }
    name +=
        std::to_string(static_cast<int>(band.effectiveNm)) + " nm";
    if (band.neutralDensity > 0.0) {
        name +=
            ", solar (ND " +
            std::to_string(static_cast<int>(band.neutralDensity)) +
            ")";
    }
    return name;
}

class Mars2020Demo : public app::Demo {
  public:
    Mars2020Demo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        Atmosphere::Define(scene, "/World/Air")
            .SetPreset(std::string(tokens::kMars))
            .SetDustOpticalDepth(0.5)
            .SetGroundAlbedo(0.25);
        // The Sun's measured spectrum at Mars's distance, 40 degrees
        // up in the south-east.
        DistantLight::Define(scene, "/World/Sun")
            .SetIrradiance(kMarsIrradiance)
            .SetPhysicalIlluminant(std::string(tokens::kSun))
            .SetAngle(kSunAngle);
        const double up = 40.0 * kDegree;
        const double azimuth = 135.0 * kDegree;
        scene.Set("/World/Sun", tokens::kTransform,
                  DistantLightTransform(
                      glm::dvec3(std::sin(azimuth) * std::cos(up),
                                 std::cos(azimuth) * std::cos(up),
                                 std::sin(up))));
        Material::Define(scene, "/World/Looks/Soil")
            .SetReflectance(kSoil);
        Material::Define(scene, "/World/Looks/Rock")
            .SetReflectance(kRock);
        GroundPlane::Define(scene, "/World/Ground")
            .SetMaterialBinding(Path("/World/Looks/Soil"));
        for (std::size_t i = 0; i < kRocks.size(); ++i) {
            const Path path("/World/Rock" + std::to_string(i));
            Sphere::Define(scene, path)
                .SetRadius(kRocks[i][2])
                .SetMaterialBinding(Path("/World/Looks/Rock"));
            SetXform(
                scene, path,
                {.translate = glm::dvec3(kRocks[i][0], kRocks[i][1],
                                         0.5 * kRocks[i][2])});
        }

        Xform::Define(scene, "/World/Rover");
        DefineEngineeringPair(kNavcamRig, kNavcam);
        DefineEngineeringPair(kHazcamRig, kHazcam);
        DefineEngineeringPair(kRearHazcamRig, kRearHazcam);
        DefineMastcamPair();

        // The viewer's own view: the rover's patch from behind.
        Camera::Define(scene, "/World/View")
            .SetFocalLength(18.0)
            .SetSensorNoise(false)
            .SetExposureTime(1.0 / 250.0);
        scene.Set("/World/View", tokens::kTransform,
                  LookAtTransform({1.5, -4.0, 2.5}, {0.0, 4.0, 0.0},
                                  {0.0, 0.0, 1.0}));
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/View"))
            .SetResolution(glm::dvec2(480, 320))
            .SetAmbient(0.0);
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/View");
        }

        // The camera the viewer shows (a pair's left eye),
        // Mastcam-Z's zoom and each eye's filter wheel.
        std::vector<std::string> cameras;
        for (const auto& [name, path] : kViews) {
            cameras.emplace_back(name);
        }
        Widgets().Combo("Camera", std::move(cameras), 0,
                        [viewer](std::size_t index) {
                            if (viewer != nullptr) {
                                viewer->SetActiveCamera(
                                    kViews[index].second);
                            }
                        });
        Widgets().Slider("Mastcam-Z focal length (mm)",
                         kShortestFocalMm, kLongestFocalMm,
                         kShortestFocalMm, [this](double focal) {
                             focalMm_ = focal;
                             for (int e = 0; e < 2; ++e) {
                                 UseMastcamFilter(MastcamCamera(e), e,
                                                  filters_[e]);
                             }
                         });
        for (int e = 0; e < 2; ++e) {
            std::vector<std::string> names;
            for (int f = 0; f < 8; ++f) {
                names.push_back(FilterName(e, f));
            }
            Widgets().Combo(
                std::string("Mastcam-Z ") +
                    (e == 0 ? "left" : "right") + " filter",
                std::move(names), 0, [this, e](std::size_t index) {
                    SetMastcamFilter(e, static_cast<int>(index));
                });
        }
    }

    /// The left Navcam, Hazcam and Mastcam-Z images and the right
    /// Navcam's, side by side, as the rover sends them.
    void SaveOutput(const std::filesystem::path& path) override {
        const Path products[] = {
            "/Render/NavcamLeft", "/Render/NavcamRight",
            "/Render/HazcamLeft", "/Render/MastcamZLeft"};
        RenderProductSetOutputs outputs =
            GetRenderer().Step(GetScene(), products);
        std::vector<Tensor> images;
        for (const Path& product : products) {
            images.push_back(Var(outputs, product, "LdrColor"));
        }
        app::WritePng(path, demos::SideBySide(images));
    }

    void TestFinal() override {
        Scene& scene = GetScene();
        for (const char* rig :
             {kNavcamRig.path, kHazcamRig.path, kRearHazcamRig.path,
              kMastcamRig.path}) {
            for (const char* eye : kEyes) {
                Camera(scene, Path(rig).Child(eye))
                    .SetSensorNoise(false);
            }
        }
        CheckFieldsOfView(kNavcam, kNavcamRig, 1.0);
        CheckFieldsOfView(kHazcam, kHazcamRig, 1.5);
        CheckStereo(kNavcam, kNavcamRig, 5.0, 0.1);
        CheckStereo(kHazcam, kHazcamRig, 1.0, 0.05);
        CheckStereo(kRearHazcam, kRearHazcamRig, 1.0, 0.05);
        CheckMastcamRig();
        CheckSoilIsRed();
        CheckNearInfrared();
        CheckEveryFilterAndZoom();
    }

  private:
    /// Product `product` of `camera` at the binned size, with its
    /// readout chain of SensorFilter prims of `types` under it.
    std::vector<Path>
    DefineProduct(const Path& product, const Path& camera,
                  const std::vector<std::string>& types,
                  bool photographic = true) {
        Scene& scene = GetScene();
        std::vector<std::string> names = {"RawDN", "Electrons",
                                          "PrimId"};
        if (photographic) {
            names.emplace_back("LdrColor");
        }
        std::vector<Path> vars;
        for (const std::string& var : names) {
            const Path path = product.Child(var);
            RenderVar::Define(scene, path).SetSourceName(var);
            vars.push_back(path);
        }
        std::vector<Path> filters;
        if (scene.Has(product.Child("Chain"))) {
            scene.Remove(product.Child("Chain"));
        }
        for (std::size_t i = 0; i < types.size(); ++i) {
            const Path path = product.Child("Chain").Child(
                "Step" + std::to_string(i));
            SensorFilterPrim::Define(scene, path)
                .SetFilterType(types[i]);
            filters.push_back(path);
        }
        RenderProduct::Define(scene, product)
            .SetCamera(camera)
            .SetResolution(glm::dvec2(kWidth, kHeight))
            .SetOrderedVars(vars)
            .SetFilters(filters);
        return filters;
    }

    /// A Navcam or Hazcam pair (Maki et al. 2020, Table 2) on its
    /// rig, `optics.baselineM` apart along the rig's x, parallel: the
    /// f-theta lens over the full frame, focused at its best focus,
    /// on the CMV-20000; its readout qe, bayer (RGGB; the paper does
    /// not give the layout), noise, adc, demosaic (RawDN the 12-bit
    /// numbers), then cie, exposure and jpeg at quality 85 for the
    /// 8-bit images the rover sends (the paper gives the quality as
    /// selectable). The detector's QE is published only as 64.5% at
    /// 550 nm, monochrome, so the channels keep Ernie's generic RGB
    /// curves.
    void DefineEngineeringPair(const Rig& rig, const Optics& optics) {
        Scene& scene = GetScene();
        Xform::Define(scene, rig.path);
        scene.Set(rig.path, tokens::kTransform, RigTransform(rig));
        const double k1 = FThetaK1(optics);
        for (int e = 0; e < 2; ++e) {
            const Path camera = Path(rig.path).Child(kEyes[e]);
            Camera::Define(scene, camera)
                .SetLensProjection(std::string(tokens::kFisheye))
                .SetLensFisheye(std::vector<float>{
                    static_cast<float>(k1), 0.0f, 0.0f, 0.0f})
                .SetFocalLength(optics.focalMm)
                .SetHorizontalAperture(1e-3 * kCmvPixelUm * kCmvWidth)
                .SetVerticalAperture(1e-3 * kCmvPixelUm * kCmvHeight)
                .SetClippingRange(glm::dvec2(0.05, 1e4))
                .SetFStop(optics.fNumber)
                .SetFocusDistance(optics.focusM)
                .SetExposureFStop(optics.fNumber)
                .SetExposureTime(4e-5)
                .SetExposureMode("auto")  // the 8-bit images
                .SetExposureIso(100.0)    // the gain is the ADC's
                .SetSensorType("rgb")
                .SetSensorElectronsPerDn(kCmvElectronsPerDn)
                .SetSensorReadNoise(kCmvDarkNoise)
                .SetSensorFullWell(kCmvFullWell)
                .SetSensorAdcBits(12)
                .SetSensorBlackLevel(0.0)
                .SetSensorNoise(true);
            scene.Set(
                camera, tokens::kTransform,
                EyePose((e == 0 ? -0.5 : 0.5) * optics.baselineM,
                        0.0));
            const std::vector<Path> filters = DefineProduct(
                Path(std::string("/Render/") + optics.name +
                     kEyes[e]),
                camera,
                {"qe", "bayer", "noise", "adc", "demosaic", "cie",
                 "exposure", "jpeg"});
            scene.Set(filters[7], "inputs:quality", 85.0);
        }
    }

    /// Mastcam-Z's pair (Bell et al. 2021, Table 5) on its rig at the
    /// widest zoom, 24.4 cm apart, each toed in 1.15 degrees, through
    /// filter 0.
    void DefineMastcamPair() {
        Scene& scene = GetScene();
        Xform::Define(scene, kMastcamRig.path);
        scene.Set(kMastcamRig.path, tokens::kTransform,
                  RigTransform(kMastcamRig));
        for (int e = 0; e < 2; ++e) {
            const Path camera =
                Path(kMastcamRig.path).Child(kEyes[e]);
            Camera::Define(scene, camera);
            // The left turns towards +x, the right towards -x.
            scene.Set(
                camera, tokens::kTransform,
                EyePose((e == 0 ? -0.5 : 0.5) * kMastcamBaselineM,
                        (e == 0 ? -1.0 : 1.0) * kToeInDeg * kDegree));
            SetMastcamFilter(e, 0);
        }
    }

    Path MastcamCamera(int e) const {
        return Path(kMastcamRig.path).Child(kEyes[e]);
    }

    /// Mastcam-Z eye `e` through filter `filter`, with its readout:
    /// qe, bayer, noise, adc and demosaic for filter 0's colour, qe,
    /// noise and adc for a narrow band; then cie and exposure for the
    /// image to show.
    void SetMastcamFilter(int e, int filter) {
        filters_[e] = filter;
        UseMastcamFilter(MastcamCamera(e), e, filter);
        DefineProduct(
            Path(std::string("/Render/MastcamZ") + kEyes[e]),
            MastcamCamera(e),
            filter == 0
                ? std::vector<std::string>{"qe", "bayer", "noise",
                                           "adc", "demosaic", "cie",
                                           "exposure"}
                : std::vector<std::string>{"qe", "noise", "adc",
                                           "cie", "exposure"});
    }

    /// Mastcam-Z camera `camera` (eye `e`) through filter `filter` at
    /// the zoom's focal length: filter 0 an RGB sensor whose channels
    /// are Gaussians at Table 3's Bayer effective wavelengths and
    /// half widths peaking at ~40%; 1 to 7 one channel, the filter's
    /// passband (2 x HWHM about its effective wavelength, ramped over
    /// 2 nm about the half maxima, times 10^-ND) on silicon scaled to
    /// the ~40% peak, the spectral range narrowed to it.
    void UseMastcamFilter(const Path& camera, int e, int filter) {
        Camera prim(GetScene(), camera);
        prim.SetFocalLength(focalMm_)
            .SetHorizontalAperture(1e-3 * kKaiPixelUm * kKaiWidth)
            .SetVerticalAperture(1e-3 * kKaiPixelUm * kKaiHeight)
            .SetClippingRange(glm::dvec2(0.05, 1e4))
            .SetExposureFStop(FNumberAt(focalMm_))
            .SetExposureTime(2e-5)
            .SetExposureMode("auto")  // the 8-bit images
            .SetExposureIso(100.0)    // the gain is the ADC's
            .SetSensorElectronsPerDn(kKaiElectronsPerDn)
            .SetSensorReadNoise(kKaiReadNoise[e])
            .SetSensorFullWell(kKaiFullWell[e])
            .SetSensorAdcBits(11)
            .SetSensorBlackLevel(0.0)
            .SetSensorNoise(true);
        if (filter == 0) {
            std::array<std::vector<float>, 3> curves;
            for (std::size_t c = 0; c < 3; ++c) {
                const Filter& channel = kBayer[e][c];
                const double sigma =
                    channel.hwhmNm / std::sqrt(2.0 * std::log(2.0));
                for (int w = 380; w <= 780; w += 5) {
                    const double x =
                        (w - channel.effectiveNm) / sigma;
                    curves[c].push_back(static_cast<float>(w));
                    curves[c].push_back(static_cast<float>(
                        kPeakQe * std::exp(-0.5 * x * x)));
                }
            }
            prim.SetSensorType("rgb")
                .SetSensorQeR(curves[0])
                .SetSensorQeG(curves[1])
                .SetSensorQeB(curves[2])
                .SetSpectralRange(glm::dvec2(380.0, 780.0))
                .SetSpectralBands(16);
            return;
        }
        const Filter& band = kFilters[e][filter];
        const double transmission =
            std::pow(10.0, -band.neutralDensity);
        const double peak =
            *std::max_element(kSiliconQe.begin(), kSiliconQe.end());
        const double low = band.effectiveNm - band.hwhmNm;
        const double high = band.effectiveNm + band.hwhmNm;
        std::vector<float> curve;
        const auto add = [&](double w, double inBand) {
            curve.push_back(static_cast<float>(w));
            curve.push_back(static_cast<float>(
                inBand * transmission * kPeakQe * Silicon(w) / peak));
        };
        add(low - 1.0, 0.0);
        const double step = std::max(1.0, (high - low - 2.0) / 8.0);
        for (double w = low + 1.0; w <= high - 1.0 + 1e-9;
             w += step) {
            add(w, 1.0);
        }
        add(high + 1.0, 0.0);
        prim.SetSensorType("mono")
            .SetSensorQeMono(curve)
            .SetSpectralRange(glm::dvec2(low - 1.0, high + 1.0))
            .SetSpectralBands(4);
    }

    /// The f-theta lens sees the published fields of view: each
    /// field's half width on the detector over the focal length,
    /// through the authored k1, within `tolerance` degrees.
    void CheckFieldsOfView(const Optics& optics, const Rig& rig,
                           double tolerance) {
        const std::vector<float> fisheye =
            Camera(GetScene(), Path(rig.path).Child("Left"))
                .GetLensFisheye();
        const std::array<double, 3> reaches = CmvReaches();
        for (std::size_t i = 0; i < 3; ++i) {
            const double degrees =
                2.0 *
                FThetaAngle(reaches[i] / optics.focalMm,
                            static_cast<double>(fisheye[0])) /
                kDegree;
            std::printf("%s field %zu: %.1f degrees for %.0f\n",
                        optics.name, i, degrees, optics.fieldsDeg[i]);
            Check(std::abs(degrees - optics.fieldsDeg[i]) < tolerance,
                  "the lens should see the published fields of view");
        }
    }

    /// A ball of `radius` on the rig's axis `depth` metres out sits
    /// further right in the left eye's image than the right's, by the
    /// baseline's disparity through the f-theta lens: 2 f theta (1 +
    /// k1 theta^2) pixels, theta = atan(baseline / 2 depth). The
    /// numbers stay within the 12-bit ADC's range, most exposed
    /// within it.
    void CheckStereo(const Optics& optics, const Rig& rig,
                     double depth, double radius) {
        Scene& scene = GetScene();
        const glm::dvec3 target = glm::dvec3(
            RigTransform(rig) * glm::dvec4(0.0, 0.0, -depth, 1.0));
        const Path ball("/World/StereoTarget");
        Sphere::Define(scene, ball).SetRadius(radius);
        SetXform(scene, ball, {.translate = target});
        const Path products[] = {
            Path(std::string("/Render/") + optics.name + "Left"),
            Path(std::string("/Render/") + optics.name + "Right")};
        RenderProductSetOutputs outputs =
            GetRenderer().Step(scene, products);
        double columns[2] = {0.0, 0.0};
        for (int e = 0; e < 2; ++e) {
            const Tensor ids = Var(outputs, products[e], "PrimId");
            const Tensor raw = Var(outputs, products[e], "RawDN");
            double sum = 0.0;
            int hits = 0;
            for (int row = 0; row < kHeight; ++row) {
                for (int col = 0; col < kWidth; ++col) {
                    if (GetRenderer().PrimPath(
                            ids.As<std::uint32_t>()[row * kWidth +
                                                    col]) == ball) {
                        sum += col + 0.5;
                        ++hits;
                    }
                }
            }
            Check(hits > 0, "each eye should see the target");
            columns[e] = sum / hits;
            std::int64_t between = 0;
            for (std::int64_t i = 0; i < raw.Elements(); ++i) {
                const std::uint16_t dn = raw.As<std::uint16_t>()[i];
                Check(dn <= 4095, "the numbers should be 12-bit");
                between += dn > 0 && dn < 4095 ? 1 : 0;
            }
            Check(
                between > raw.Elements() / 2,
                "the image should be exposed within the ADC's range");
        }
        scene.Remove(ball);
        const double k1 = FThetaK1(optics);
        const double pixels = optics.focalMm / (1e-3 * kCmvPixelUm *
                                                kCmvWidth / kWidth);
        const double theta =
            std::atan(0.5 * optics.baselineM / depth);
        const double expected =
            2.0 * pixels * theta * (1.0 + k1 * theta * theta);
        const double disparity = columns[0] - columns[1];
        std::printf("%s disparity at %.1f m: %.2f pixels for %.2f\n",
                    optics.name, depth, disparity, expected);
        Check(std::abs(disparity - expected) < 1.0,
              "the pair should see the baseline's disparity");
    }

    /// Mastcam-Z's eyes 24.4 cm apart, looking 2.3 degrees apart,
    /// each turned in.
    void CheckMastcamRig() {
        Scene& scene = GetScene();
        glm::dvec3 positions[2];
        glm::dvec3 looks[2];
        for (int e = 0; e < 2; ++e) {
            const glm::dmat4 pose = scene.Get<glm::dmat4>(
                Path(kMastcamRig.path).Child(kEyes[e]),
                tokens::kTransform, glm::dmat4(1.0));
            positions[e] = glm::dvec3(pose[3]);
            looks[e] = -glm::dvec3(pose[2]);
        }
        const double baseline =
            glm::length(positions[1] - positions[0]);
        const double between =
            std::acos(glm::dot(looks[0], looks[1])) / kDegree;
        std::printf("Mastcam-Z: %.3f m apart, %.2f degrees between\n",
                    baseline, between);
        Check(std::abs(baseline - kMastcamBaselineM) < 1e-9 &&
                  std::abs(between - 2.0 * kToeInDeg) < 1e-6 &&
                  looks[0].x > 0.0 && looks[1].x < 0.0,
              "Mastcam-Z's eyes should be toed in as published");
    }

    /// Mastcam-Z's colour image of the ground: redder than it is
    /// blue, as Mars's soil.
    void CheckSoilIsRed() {
        const Path products[] = {"/Render/MastcamZLeft"};
        RenderProductSetOutputs outputs =
            GetRenderer().Step(GetScene(), products);
        const glm::dvec3 rgb = demos::MeanLinearRgb(
            Var(outputs, products[0], "LdrColor"));
        std::printf("Mastcam-Z: soil %.3f %.3f %.3f\n", rgb.r, rgb.g,
                    rgb.b);
        Check(rgb.r > 1.5 * rgb.b, "the soil should look red");
    }

    /// Through the right camera's R1 (800 +/- 9 nm) and R6 (1022 +/-
    /// 19 nm) on a camera of their own at the right eye: both see the
    /// lit ground, R6 the dimmer, silicon nearly blind at 1 um
    /// despite its band twice as wide.
    void CheckNearInfrared() {
        Scene& scene = GetScene();
        const Path camera = "/World/Rover/MastcamZ/Narrow";
        const Path product = "/Render/MastcamZNarrow";
        Camera::Define(scene, camera);
        scene.Set(camera, tokens::kTransform,
                  scene.Get<glm::dmat4>(
                      Path(kMastcamRig.path).Child("Right"),
                      tokens::kTransform, glm::dmat4(1.0)));
        DefineProduct(product, camera, {"qe", "noise", "adc"}, false);
        double means[2] = {0.0, 0.0};
        for (int i = 0; i < 2; ++i) {
            UseMastcamFilter(camera, 1, i == 0 ? 1 : 6);
            Camera(scene, camera)
                .SetSensorNoise(false)
                .SetExposureTime(1e-3);
            const Path products[] = {product};
            RenderProductSetOutputs outputs =
                GetRenderer().Step(scene, products);
            const Tensor electrons =
                Var(outputs, product, "Electrons");
            double sum = 0.0;
            for (std::int64_t k = 0; k < electrons.Elements(); ++k) {
                const double value =
                    static_cast<double>(electrons.As<float>()[k]);
                Check(value < kKaiFullWell[1],
                      "the bands should stay below the full well");
                sum += value;
            }
            means[i] =
                sum / static_cast<double>(electrons.Elements());
        }
        scene.Remove(product);
        scene.Remove(camera);
        std::printf("Mastcam-Z R1 %.0f, R6 %.0f electrons\n",
                    means[0], means[1]);
        Check(means[0] > 100.0 && means[1] > 0.0 &&
                  means[1] < 0.5 * means[0],
              "R6 should see less than half of R1's light");
    }

    /// Every filter of both wheels renders an image to show and, but
    /// for the solar filters, collects light; at the long end of the
    /// zoom the camera closes to f/9.5. Back to filter 0 at 26 mm.
    void CheckEveryFilterAndZoom() {
        Scene& scene = GetScene();
        for (int e = 0; e < 2; ++e) {
            const Path product =
                Path(std::string("/Render/MastcamZ") + kEyes[e]);
            for (int f = 1; f < 8; ++f) {
                SetMastcamFilter(e, f);
                Camera(scene, MastcamCamera(e)).SetSensorNoise(false);
                const Path products[] = {product};
                RenderProductSetOutputs outputs =
                    GetRenderer().Step(scene, products);
                const Tensor ldr = Var(outputs, product, "LdrColor");
                const Tensor electrons =
                    Var(outputs, product, "Electrons");
                double sum = 0.0;
                for (std::int64_t k = 0; k < electrons.Elements();
                     ++k) {
                    sum +=
                        static_cast<double>(electrons.As<float>()[k]);
                }
                Check(ldr.Elements() > 0 && (f == 7 || sum > 0.0),
                      "every filter should render");
            }
            SetMastcamFilter(e, 0);
            Camera(scene, MastcamCamera(e)).SetSensorNoise(false);
        }
        focalMm_ = kLongestFocalMm;
        UseMastcamFilter(MastcamCamera(0), 0, 0);
        const Camera left(scene, MastcamCamera(0));
        std::printf("Mastcam-Z at %.0f mm: f/%.1f\n",
                    left.GetFocalLength(), left.GetExposureFStop());
        Check(std::abs(left.GetExposureFStop() - kNarrowestFNumber) <
                  1e-12,
              "the zoom's long end should be f/9.5");
        focalMm_ = kShortestFocalMm;
        UseMastcamFilter(MastcamCamera(0), 0, 0);
        Camera(scene, MastcamCamera(0)).SetSensorNoise(false);
    }

    double focalMm_ = kShortestFocalMm;
    int filters_[2] = {0, 0};
};

}  // namespace

ERNIE_REGISTER_DEMO(Mars2020Demo, "cameras", "mars2020",
                    "Perseverance's Navcams, Hazcams and Mastcam-Z "
                    "as published, on a "
                    "patch of Mars.");
