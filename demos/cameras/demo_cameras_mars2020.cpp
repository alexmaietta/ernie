// SPDX-License-Identifier: BSD-3-Clause
// Perseverance's Navcam, Hazcam and Mastcam-Z stereo pairs on a patch
// of Mars, their detectors as their datasheets give them.
//
// References:
// - J. N. Maki et al., "The Mars 2020 Engineering Cameras and
//   Microphone on the Perseverance Rover", Space Science Reviews
//   216:137, 2020: Tables 2 and 3 and the detector section.
// - J. F. Bell III et al., "The Mars 2020 Perseverance Rover Mast
//   Camera Zoom (Mastcam-Z) Multispectral, Stereoscopic Imaging
//   Investigation", Space Science Reviews 217:24, 2021: Tables 3
//   and 5.
// - ams OSRAM, "CMV20000 20MP CMOS Machine Vision Image Sensor",
//   datasheet DS000440 v3-00, 2026: Table 5 and Figure 4.
// - onsemi, "KAI-2020 Image Sensor", datasheet KAI-2020/D: Table 7
// and
//   Figure 12 (Gen1 colour).
// - J. F. Bell III et al., "Mineralogic and compositional properties
//   of Martian soil and dust: Results from Mars Pathfinder", JGR
//   105(E1):1721-1755, 2000 (soil reflectance, roughly).
// - J. Kannala and S. S. Brandt, "A Generic Camera Model and
//   Calibration Method for Conventional, Wide-Angle, and Fish-Eye
//   Lenses", IEEE TPAMI 28(8):1335-1340, 2006.

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
#include <cstring>
#include <numbers>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr double kPi = std::numbers::pi;
constexpr double kDegree = kPi / 180.0;

/// The Sun's irradiance and angular diameter at Mars's mean distance.
constexpr double kMarsIrradiance = 1361.0 / (1.5237 * 1.5237);
constexpr double kSunAngle = 0.35;

/// `values` at `nm`, linear between the `grid` wavelengths (from 350
/// nm every `step`), zero outside them.
template <std::size_t N>
double Curve(const std::array<double, N>& values, double first,
             double step, double nm) {
    const double x = (nm - first) / step;
    if (x < 0.0 || x > double(N - 1)) {
        return 0.0;
    }
    const auto i = std::min<std::size_t>(std::size_t(x), N - 2);
    return values[i] + (x - double(i)) * (values[i + 1] - values[i]);
}

/// A trapezoid passband over [low, high], ramped over 20 nm.
double Pass(double nm, double low, double high) {
    return std::clamp(std::min(nm - low, high - nm) / 20.0 + 0.5, 0.0,
                      1.0);
}

/// (nm, value) pairs of `f` every 5 nm over [first, last].
template <typename F>
std::vector<float> Sampled(double first, double last, const F& f) {
    std::vector<float> pairs;
    for (double nm = first; nm <= last + 1e-9; nm += 5.0) {
        pairs.push_back(static_cast<float>(nm));
        pairs.push_back(static_cast<float>(f(nm)));
    }
    return pairs;
}

// ------------------------------------------------- engineering
// cameras

/// The CMV20000 (DS000440, Table 5): 6.4 um pixels, 0.25 DN/e-, 8 e-
/// noise, 15000 e- full well, 125 e-/s dark current, 10 e-/s DSNU, 1%
/// PRNU.
constexpr double kCmvPixelUm = 6.4;
constexpr int kCmvWidth = 5120;
constexpr int kCmvHeight = 3840;

/// Figure 4's Bayer QE, read every 25 nm from 350 nm.
constexpr std::array<double, 31> kCmvBlue = {
    .05, .06, .20, .27, .32, .35, .27,  .10, .05, .04, .03,
    .03, .03, .04, .05, .05, .05, .05,  .19, .21, .21, .16,
    .13, .10, .06, .04, .03, .02, .015, .01, .005};
constexpr std::array<double, 31> kCmvGreen = {
    .04, .05, .05, .06, .07, .10, .25,  .42, .44, .30, .12,
    .06, .05, .08, .13, .13, .17, .15,  .20, .21, .21, .16,
    .13, .10, .06, .04, .03, .02, .015, .01, .005};
constexpr std::array<double, 31> kCmvRed = {
    .05, .08, .07, .03, .02, .02, .02,  .02, .03, .03, .44,
    .40, .38, .37, .34, .32, .32, .25,  .27, .22, .20, .16,
    .13, .10, .06, .04, .03, .02, .015, .01, .005};

/// Maki's 8 x 8 panchromatic mode (640 x 480): each pixel the mean of
/// 64 behind a Bayer cell's mean response and the UV and IR blocking
/// filter (its edges unpublished; 400 and 700 nm assumed).
constexpr int kEcamSum = 8;
constexpr int kEcamWidth = kCmvWidth / kEcamSum;
constexpr int kEcamHeight = kCmvHeight / kEcamSum;
constexpr double kEcamExposure = 1e-3;

double EcamQe(double nm) {
    const double cell =
        0.25 * (Curve(kCmvRed, 350.0, 25.0, nm) +
                2.0 * Curve(kCmvGreen, 350.0, 25.0, nm) +
                Curve(kCmvBlue, 350.0, 25.0, nm));
    return cell * Pass(nm, 400.0, 700.0);
}

/// An engineering camera's optics (Maki Table 2): focal length (mm),
/// f-number, best focus (m), baseline (m) and fields of view
/// (degrees).
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
constexpr Optics kRearHazcam{
    "RearHazcam", 14.0, 12.0, 0.9, 0.934, {136.0, 102.0, 172.0}};

/// The detector's half width, half height and half diagonal, mm.
std::array<double, 3> CmvReaches() {
    const double halfWidth = 0.5e-3 * kCmvPixelUm * kCmvWidth;
    const double halfHeight = 0.5e-3 * kCmvPixelUm * kCmvHeight;
    return {halfWidth, halfHeight, std::hypot(halfWidth, halfHeight)};
}

/// The f-theta lens's k1 fitted to the published fields.
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

/// The angle off axis landing `r` focal lengths out, by Newton.
double FThetaAngle(double r, double k1) {
    double theta = r;
    for (int i = 0; i < 20; ++i) {
        const double f = theta * (1.0 + k1 * theta * theta) - r;
        theta -= f / (1.0 + 3.0 * k1 * theta * theta);
    }
    return theta;
}

// -----------------------------------------------------------
// Mastcam-Z

/// The KAI-2020CM (Bell Table 5; datasheet Table 7): 1600 x 1200
/// pixels of 7.4 um, 15.6 e-/DN, 11 bits, 40 e-/s dark current and
/// 2.5% rms PRNU.
constexpr double kKaiPixelUm = 7.4;
constexpr int kKaiWidth = 1600;
constexpr int kKaiHeight = 1200;
constexpr double kKaiElectronsPerDn = 15.6;
constexpr double kKaiFullWell[2] = {21826.0, 21818.0};
constexpr double kKaiReadNoise[2] = {22.0, 21.0};
/// Exposures through the colour filter and the narrow bands.
constexpr double kMastcamExposure = 2e-3;
constexpr double kNarrowExposure = 2e-2;

/// Figure 12's Gen1 colour QE, read every 20 nm from 380 nm.
constexpr std::array<double, 37> kKaiBlue = {
    .02,  .20,  .33,  .40,  .46,  .40,  .25,  .08, .02,  .01,
    .005, .005, .005, .005, .005, .01,  .01,  .01, .01,  .01,
    .03,  .06,  .08,  .07,  .06,  .045, .035, .03, .025, .02,
    .015, .012, .01,  .008, .006, .004, .002};
constexpr std::array<double, 37> kKaiGreen = {
    .01,  .01,  .015, .02,  .03,  .05,  .12,  .25, .41,  .33,
    .12,  .02,  .01,  .01,  .01,  .02,  .03,  .04, .04,  .05,
    .06,  .07,  .08,  .07,  .06,  .045, .035, .03, .025, .02,
    .015, .012, .01,  .008, .006, .004, .002};
constexpr std::array<double, 37> kKaiRed = {
    .01,  .01,  .01,  .01,  .005, .005, .005, .005, .005, .01,
    .05,  .25,  .33,  .30,  .27,  .25,  .22,  .19,  .16,  .14,
    .12,  .10,  .085, .07,  .06,  .045, .035, .03,  .025, .02,
    .015, .012, .01,  .008, .006, .004, .002};

/// Zoom, baseline and toe-in (Bell Table 5).
constexpr double kShortestFocalMm = 26.0;
constexpr double kLongestFocalMm = 110.0;
constexpr double kWidestFNumber = 6.7;
constexpr double kNarrowestFNumber = 9.5;
constexpr double kMastcamBaselineM = 0.244;
constexpr double kToeInDeg = 1.15;

double FNumberAt(double focalMm) {
    const double t = (focalMm - kShortestFocalMm) /
                     (kLongestFocalMm - kShortestFocalMm);
    return kWidestFNumber + t * (kNarrowestFNumber - kWidestFNumber);
}

/// A filter's effective wavelength, half width at half maximum (nm)
/// and neutral density (Bell Table 3).
struct Filter {
    double effectiveNm;
    double hwhmNm;
    double neutralDensity;
};

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

/// Filter 0's infrared cut-off: the red channel's upper half maximum
/// in Table 3, 631 + 43 nm.
constexpr double kIrCutNm = 674.0;

// ----------------------------------------------------------- the
// scene

/// Mars soil and darker rock's reflectance, roughly (Bell 2000).
const std::vector<float> kSoil{350.0f, 0.05f, 450.0f,  0.08f,
                               550.0f, 0.14f, 650.0f,  0.25f,
                               750.0f, 0.30f, 1100.0f, 0.30f};
const std::vector<float> kRock{350.0f, 0.06f, 550.0f,  0.09f,
                               750.0f, 0.14f, 1100.0f, 0.15f};

/// Rocks ahead of the rover: x, y and radius, m.
constexpr std::array<std::array<double, 3>, 6> kRocks = {{
    {-1.2, 3.0, 0.25},
    {0.9, 4.5, 0.4},
    {-2.5, 7.0, 0.6},
    {2.0, 9.0, 0.5},
    {0.3, 1.6, 0.12},
    {-0.6, 2.2, 0.18},
}};

/// A stereo rig: its eye, tilt down and heading (degrees from north).
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

/// An eye `x` along its rig, turned `yaw` radians about the rig's +y.
glm::dmat4 EyePose(double x, double yaw) {
    glm::dmat4 pose =
        glm::rotate(glm::dmat4(1.0), yaw, glm::dvec3(0.0, 1.0, 0.0));
    pose[3] = glm::dvec4(x, 0.0, 0.0, 1.0);
    return pose;
}

Tensor Var(RenderProductSetOutputs& outputs, const Path& product,
           const char* var) {
    return outputs.at(product)
        .vars.at(var)
        .Map(MapDevice::Cpu)
        .tensor;
}

constexpr const char* kEyes[2] = {"Left", "Right"};

/// The cameras the viewer can show.
const std::array<std::pair<const char*, const char*>, 5> kViews = {{
    {"Overview", "/World/View"},
    {"Navcam", "/World/Rover/Navcam/Left"},
    {"Front Hazcam", "/World/Rover/Hazcam/Left"},
    {"Rear Hazcam", "/World/Rover/RearHazcam/Left"},
    {"Mastcam-Z", "/World/Rover/MastcamZ/Left"},
}};

/// Filter `filter` of eye `e`'s wheel as Table 3 names it.
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
        // The Sun, 40 degrees up in the south-east.
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

        // The viewer's own view of the rover's patch.
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

    /// The left Navcam, the right Navcam, the left Hazcam and the
    /// middle of the left Mastcam-Z, as the rover sends them.
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
        images.back() =
            Centre(images.back(), kEcamWidth, kEcamHeight);
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
    /// The central `width` x `height` of a uint8 image.
    static Tensor Centre(const Tensor& image, int width, int height) {
        const std::int64_t rows = image.Shape()[0];
        const std::int64_t cols = image.Shape()[1];
        const std::int64_t channels = image.Shape()[2];
        const std::int64_t top = (rows - height) / 2;
        const std::int64_t left = (cols - width) / 2;
        std::vector<std::uint8_t> pixels(std::size_t(width) *
                                         std::size_t(height) *
                                         std::size_t(channels));
        const auto* from =
            static_cast<const std::uint8_t*>(image.Data());
        for (int r = 0; r < height; ++r) {
            std::memcpy(&pixels[std::size_t(r) * std::size_t(width) *
                                std::size_t(channels)],
                        from + ((top + r) * cols + left) * channels,
                        std::size_t(width * channels));
        }
        return Tensor::Copy<std::uint8_t>(pixels,
                                          {height, width, channels});
    }

    /// Product `product` of `camera`, with a readout chain of
    /// `types`.
    std::vector<Path>
    DefineProduct(const Path& product, const Path& camera, int width,
                  int height, const std::vector<std::string>& types,
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
            .SetResolution(glm::dvec2(width, height))
            .SetOrderedVars(vars)
            .SetFilters(filters);
        return filters;
    }

    /// A Navcam or Hazcam pair in the 8 x 8 mode: the f-theta lens on
    /// the CMV20000, its sums of 64 pixels scaled as means, sent as
    /// JPEG.
    void DefineEngineeringPair(const Rig& rig, const Optics& optics) {
        Scene& scene = GetScene();
        Xform::Define(scene, rig.path);
        scene.Set(rig.path, tokens::kTransform, RigTransform(rig));
        const double k1 = FThetaK1(optics);
        const double n = kEcamSum * kEcamSum;
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
                .SetExposureTime(kEcamExposure)
                .SetExposureMode("auto")  // the 8-bit images
                .SetExposureIso(100.0)    // the gain is the ADC's
                .SetSensorType("mono")
                .SetSensorQeMono(Sampled(350.0, 1100.0, EcamQe))
                .SetSpectralRange(glm::dvec2(380.0, 720.0))
                .SetSpectralBands(16)
                .SetSensorElectronsPerDn(4.0 * n)
                .SetSensorReadNoise(8.0 * kEcamSum)
                .SetSensorFullWell(15000.0 * n)
                .SetSensorDarkCurrent(125.0 * n)
                .SetSensorDsnu(10.0 * kEcamExposure * kEcamSum)
                .SetSensorPrnu(0.01 / kEcamSum)
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
                camera, kEcamWidth, kEcamHeight,
                {"qe", "noise", "adc", "cie", "exposure", "jpeg"});
            scene.Set(filters[5], "inputs:quality", 85.0);
        }
    }

    /// Mastcam-Z's pair at the widest zoom, toed in, through filter
    /// 0.
    void DefineMastcamPair() {
        Scene& scene = GetScene();
        Xform::Define(scene, kMastcamRig.path);
        scene.Set(kMastcamRig.path, tokens::kTransform,
                  RigTransform(kMastcamRig));
        for (int e = 0; e < 2; ++e) {
            const Path camera = MastcamCamera(e);
            Camera::Define(scene, camera);
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

    /// Mastcam-Z eye `e` through filter `filter`, with its readout.
    void SetMastcamFilter(int e, int filter) {
        filters_[e] = filter;
        UseMastcamFilter(MastcamCamera(e), e, filter);
        DefineProduct(
            Path(std::string("/Render/MastcamZ") + kEyes[e]),
            MastcamCamera(e), kKaiWidth, kKaiHeight,
            filter == 0
                ? std::vector<std::string>{"qe", "bayer", "noise",
                                           "adc", "demosaic", "cie",
                                           "exposure"}
                : std::vector<std::string>{"qe", "noise", "adc",
                                           "cie", "exposure"});
    }

    /// Mastcam-Z camera `camera` through filter `filter`: Gen1 colour
    /// under the IR cut-off for filter 0, else the passband on the
    /// Bayer cell's mean response.
    void UseMastcamFilter(const Path& camera, int e, int filter) {
        Camera prim(GetScene(), camera);
        prim.SetFocalLength(focalMm_)
            .SetHorizontalAperture(1e-3 * kKaiPixelUm * kKaiWidth)
            .SetVerticalAperture(1e-3 * kKaiPixelUm * kKaiHeight)
            .SetClippingRange(glm::dvec2(0.05, 1e4))
            .SetExposureFStop(FNumberAt(focalMm_))
            .SetExposureTime(filter == 0 ? kMastcamExposure
                                         : kNarrowExposure)
            .SetExposureMode("auto")  // the 8-bit images
            .SetExposureIso(100.0)    // the gain is the ADC's
            .SetSensorElectronsPerDn(kKaiElectronsPerDn)
            .SetSensorReadNoise(kKaiReadNoise[e])
            .SetSensorFullWell(kKaiFullWell[e])
            .SetSensorDarkCurrent(40.0)
            .SetSensorDsnu(0.0)
            .SetSensorPrnu(0.025)
            .SetSensorAdcBits(11)
            .SetSensorBlackLevel(0.0)
            .SetSensorNoise(true);
        const auto kai = [](const std::array<double, 37>& curve) {
            return [&curve](double nm) {
                return Curve(curve, 380.0, 20.0, nm) *
                       Pass(nm, 330.0, kIrCutNm);
            };
        };
        if (filter == 0) {
            prim.SetSensorType("rgb")
                .SetSensorQeR(Sampled(380.0, 780.0, kai(kKaiRed)))
                .SetSensorQeG(Sampled(380.0, 780.0, kai(kKaiGreen)))
                .SetSensorQeB(Sampled(380.0, 780.0, kai(kKaiBlue)))
                .SetSpectralRange(glm::dvec2(380.0, 780.0))
                .SetSpectralBands(16);
            return;
        }
        const Filter& band = kFilters[e][filter];
        const double transmission =
            std::pow(10.0, -band.neutralDensity);
        const double low = band.effectiveNm - band.hwhmNm;
        const double high = band.effectiveNm + band.hwhmNm;
        // The passband's edges ramped over 2 nm about its half
        // maxima.
        const auto qe = [&](double nm) {
            const double inBand = std::clamp(
                std::min(nm - low, high - nm) / 2.0 + 0.5, 0.0, 1.0);
            const double cell =
                0.25 * (Curve(kKaiRed, 380.0, 20.0, nm) +
                        2.0 * Curve(kKaiGreen, 380.0, 20.0, nm) +
                        Curve(kKaiBlue, 380.0, 20.0, nm));
            return inBand * transmission * cell;
        };
        std::vector<float> curve;
        for (double nm = low - 1.0; nm <= high + 1.0 + 1e-9;
             nm += 0.5) {
            curve.push_back(static_cast<float>(nm));
            curve.push_back(static_cast<float>(qe(nm)));
        }
        prim.SetSensorType("mono")
            .SetSensorQeMono(curve)
            .SetSpectralRange(glm::dvec2(low - 1.0, high + 1.0))
            .SetSpectralBands(4);
    }

    /// The f-theta lens sees the published fields of view.
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

    /// A ball `depth` m out on the rig's axis shows the baseline's
    /// disparity through the f-theta lens, the numbers 12-bit.
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
            for (int row = 0; row < kEcamHeight; ++row) {
                for (int col = 0; col < kEcamWidth; ++col) {
                    if (GetRenderer().PrimPath(
                            ids.As<std::uint32_t>()[row * kEcamWidth +
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
                between > raw.Elements() / 8,
                "the image should be exposed within the ADC's range");
        }
        scene.Remove(ball);
        const double k1 = FThetaK1(optics);
        const double pixels =
            optics.focalMm /
            (1e-3 * kCmvPixelUm * kCmvWidth / kEcamWidth);
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

    /// Mastcam-Z's eyes are 24.4 cm apart and toed in 2.3 degrees.
    void CheckMastcamRig() {
        Scene& scene = GetScene();
        glm::dvec3 positions[2];
        glm::dvec3 looks[2];
        for (int e = 0; e < 2; ++e) {
            const glm::dmat4 pose = scene.Get<glm::dmat4>(
                MastcamCamera(e), tokens::kTransform,
                glm::dmat4(1.0));
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

    /// Mastcam-Z sees Mars's soil redder than it is blue.
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

    /// R6 (1022 nm) sees under half of R1's (800 nm) light: silicon
    /// is nearly blind at 1 um.
    void CheckNearInfrared() {
        Scene& scene = GetScene();
        const Path camera = "/World/Rover/MastcamZ/Narrow";
        const Path product = "/Render/MastcamZNarrow";
        Camera::Define(scene, camera);
        scene.Set(camera, tokens::kTransform,
                  scene.Get<glm::dmat4>(MastcamCamera(1),
                                        tokens::kTransform,
                                        glm::dmat4(1.0)));
        DefineProduct(product, camera, kKaiWidth, kKaiHeight,
                      {"qe", "noise", "adc"}, false);
        double means[2] = {0.0, 0.0};
        for (int i = 0; i < 2; ++i) {
            UseMastcamFilter(camera, 1, i == 0 ? 1 : 6);
            Camera(scene, camera).SetSensorNoise(false);
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

    /// Every filter renders, and the zoom's long end is f/9.5.
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
