// SPDX-License-Identifier: BSD-3-Clause
// A gallery of the built-in sensor effects: one scene through a
// camera and filter chain per effect, each image logged in the viewer
// and laid out in a grid by --output.

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
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr int kCols = 384;
constexpr int kRows = 240;
constexpr std::size_t kGridColumns = 3;
constexpr std::size_t kGap = 4;
constexpr double kPitchMm = 0.01;  // 10 um pixels
constexpr double kExposureTime = 0.00035;
constexpr glm::dvec3 kEye(0.0, -5.5, 2.2);
constexpr glm::dvec3 kTarget(0.0, 1.0, 1.2);

// The cameras the effects look through; the moving ones pan between
// frames.
enum class Rig {
    Clean,
    LowLight,
    Short,
    Defocus,
    Distorted,
    Fisheye,
    Vignetted,
    Chromatic,
    Rgb,
    Rolling,
    Blurred,
    Moving,
    MovingRgb
};

constexpr Rig kRigs[] = {Rig::Clean,     Rig::LowLight,  Rig::Short,
                         Rig::Defocus,   Rig::Distorted, Rig::Fisheye,
                         Rig::Vignetted, Rig::Chromatic, Rig::Rgb,
                         Rig::Rolling,   Rig::Blurred,   Rig::Moving,
                         Rig::MovingRgb};

bool IsColour(Rig rig) {
    return rig == Rig::Rgb || rig == Rig::Chromatic ||
           rig == Rig::MovingRgb;
}

bool IsMoving(Rig rig) {
    return rig == Rig::Rolling || rig == Rig::Blurred ||
           rig == Rig::Moving || rig == Rig::MovingRgb;
}

struct FilterStep {
    std::string type;
    std::map<std::string, float> inputs = {};
    std::map<std::string, std::vector<float>> arrays = {};
};

struct Effect {
    std::string id;
    Rig rig;
    std::string var;  // RawDN, LdrColor or an AOV
    std::vector<FilterStep> steps;
    double scale = 1.0;  // display gain on the 8-bit numbers
};

const FilterStep kQe{"qe"};
const FilterStep kAdc{"adc"};
const FilterStep kCie{"cie"};
const FilterStep kExposure{"exposure"};

// 16 codes of square-root companding from 8 bits: each code's last
// input DN.
std::vector<float> FourBitUppers() {
    std::vector<float> uppers(256, 255.0f);
    for (std::size_t c = 0; c < 16; ++c) {
        const double top = static_cast<double>((c + 1) * (c + 1));
        uppers[c] =
            static_cast<float>(std::min(std::ceil(top) - 1.0, 255.0));
    }
    return uppers;
}

// A long-pass filter's transmission, 360 to 830 nm every 10 nm: red
// light only.
std::vector<float> RedFilter() {
    std::vector<float> curve(48, 0.0f);
    for (std::size_t i = 0; i < curve.size(); ++i) {
        curve[i] = 360.0 + 10.0 * static_cast<double>(i) >= 600.0
                       ? 1.0f
                       : 0.0f;
    }
    return curve;
}

std::vector<Effect> Effects() {
    const float smearTime = static_cast<float>(0.1 * kExposureTime);
    return {
        // The detector.
        {"clean", Rig::Clean, "RawDN", {kQe, kAdc}},
        {"noise", Rig::LowLight, "RawDN", {kQe, {"noise"}, kAdc}},
        {"dark_current",
         Rig::LowLight,
         "RawDN",
         {kQe,
          {"darkCurrent",
           {{"temperature", 60.0f},
            {"referenceRate", 2e3f},
            {"nonuniformity", 0.3f}}},
          {"noise"},
          kAdc}},
        {"defect_pixels",
         Rig::Clean,
         "RawDN",
         {kQe,
          {"defectPixels",
           {{"dead", 3e-3f}, {"hot", 3e-3f}, {"hotRate", 5e6f}}},
          kAdc}},
        {"radiation",
         Rig::Clean,
         "RawDN",
         {kQe,
          {"radiation", {{"flux", 2e6f}, {"thickness", 40.0f}}},
          kAdc}},
        {"detector_pattern",
         Rig::Clean,
         "RawDN",
         {kQe,
          {"detectorPattern", {{"gain", 0.08f}, {"offset", 400.0f}}},
          kAdc}},
        {"readout_taps",
         Rig::Clean,
         "RawDN",
         {kQe,
          kAdc,
          {"readoutTaps", {{"offset1", 12.0f}, {"gain1", 1.1f}}}}},
        {"smear",
         Rig::Clean,
         "RawDN",
         {kQe,
          {"smear",
           {{"transferTime", smearTime},
            {"resetTime", 0.25f * smearTime}}},
          kAdc}},
        {"blooming", Rig::Clean, "RawDN", {kQe, {"blooming"}, kAdc}},
        {"lag",
         Rig::Moving,
         "RawDN",
         {kQe, {"lag", {{"fraction", 0.4f}}}, kAdc}},
        {"tdi",
         Rig::Short,
         "RawDN",
         {kQe, {"tdi", {{"stages", 4.0f}, {"drift", 1.5f}}}, kAdc}},
        // The optics.
        {"flare",
         Rig::Clean,
         "RawDN",
         {kQe, {"flare", {{"strength", 2e-3f}}}, kAdc}},
        {"gaussian_psf",
         Rig::Clean,
         "RawDN",
         {kQe,
          {"gaussianPsf",
           {},
           {{"weights", {0.8f, 0.15f, 0.05f, 0, 0, 0, 0, 0}},
            {"sigmas", {0.0f, 3.0f, 24.0f, 0, 0, 0, 0, 0}}}},
          kAdc}},
        {"airy_psf",
         Rig::Clean,
         "RawDN",
         {{"psf", {{"profile", 1.0f}, {"blur", 0.7f}}}, kQe, kAdc}},
        {"defocus", Rig::Defocus, "RawDN", {{"defocus"}, kQe, kAdc}},
        {"stray_light",
         Rig::Clean,
         "RawDN",
         {{"strayLight", {{"strength", 3e-3f}}}, kQe, kAdc}},
        {"distortion", Rig::Distorted, "RawDN", {kQe, kAdc}},
        {"fisheye", Rig::Fisheye, "RawDN", {kQe, kAdc}},
        {"vignetting", Rig::Vignetted, "RawDN", {kQe, kAdc}},
        {"chromatic", Rig::Chromatic, "LdrColor", {kCie, kExposure}},
        {"red_filter",
         Rig::Clean,
         "RawDN",
         {{"spectralTransmission", {}, {{"curve", RedFilter()}}},
          kQe,
          kAdc},
         2.0},
        // The shutter.
        {"rolling_shutter", Rig::Rolling, "RawDN", {kQe, kAdc}},
        {"motion_blur", Rig::Blurred, "RawDN", {kQe, kAdc}},
        // The digital numbers.
        {"comms_errors",
         Rig::Clean,
         "RawDN",
         {kQe,
          kAdc,
          {"commsErrors",
           {{"packetPixels", static_cast<float>(kCols)},
            {"packetLoss", 0.03f},
            {"slips", 0.05f},
            {"bitErrorRate", 2e-3f}}}}},
        {"compand_sqrt",
         Rig::Clean,
         "RawDN",
         {kQe,
          kAdc,
          {"compand", {{"inputBits", 8.0f}, {"outputBits", 6.0f}}}},
         4.0},
        {"compand_4bit",
         Rig::Clean,
         "RawDN",
         {kQe,
          kAdc,
          {"compandLookup",
           {{"codes", 16.0f}},
           {{"uppers", FourBitUppers()}}}},
         17.0},
        // Colour.
        {"measured_qe",
         Rig::Rgb,
         "RawDN",
         {{"measuredQe"}, kAdc},
         1.8},
        {"bayer_mosaic",
         Rig::Rgb,
         "aov:mosaic",
         {kQe, {"bayer"}, kAdc, {"demosaic", {{"method", 0.0f}}}},
         1.8},
        {"demosaic",
         Rig::Rgb,
         "RawDN",
         {kQe, {"bayer"}, kAdc, {"demosaic", {{"method", 0.0f}}}},
         1.8},
        {"white_balance",
         Rig::Rgb,
         "LdrColor",
         {kCie,
          {"whiteBalance", {{"red", 1.5f}, {"blue", 0.6f}}},
          kExposure}},
        {"color_matrix",
         Rig::Rgb,
         "LdrColor",
         {kCie,
          {"colorMatrix",
           {{"m00", 0.393f},
            {"m01", 0.769f},
            {"m02", 0.189f},
            {"m10", 0.349f},
            {"m11", 0.686f},
            {"m12", 0.168f},
            {"m20", 0.272f},
            {"m21", 0.534f},
            {"m22", 0.131f}}},
          kExposure}},
        {"tone_curve",
         Rig::Rgb,
         "LdrColor",
         {kCie,
          kExposure,
          {"toneCurve",
           {},
           {{"curve",
             {0.0f, 0.02f, 0.08f, 0.3f, 0.7f, 0.92f, 0.98f,
              1.0f}}}}}},
        {"glare",
         Rig::Rgb,
         "LdrColor",
         {kCie, {"glare", {{"strength", 0.1f}}}, kExposure}},
        {"temporal_average",
         Rig::MovingRgb,
         "LdrColor",
         {kCie, {"temporalAverage", {{"alpha", 0.5f}}}, kExposure}},
        {"jpeg",
         Rig::Rgb,
         "LdrColor",
         {kCie, kExposure, {"jpeg", {{"quality", 8.0f}}}}},
    };
}

Path CameraOf(Rig rig) {
    static const char* const kNames[] = {
        "Clean",   "LowLight",  "Short",     "Defocus", "Distorted",
        "Fisheye", "Vignetted", "Chromatic", "Rgb",     "Rolling",
        "Blurred", "Moving",    "MovingRgb"};
    return Path("/World/Cameras")
        .Child(kNames[static_cast<int>(rig)]);
}

Path ProductOf(const Effect& effect) {
    return Path("/Render/Effects").Child(effect.id);
}

// The var's prim under its product: its source name without the
// AOV's colon.
Path VarOf(const Effect& effect) {
    std::string name = effect.var;
    std::replace(name.begin(), name.end(), ':', '_');
    return ProductOf(effect).Child(name);
}

// The moving cameras' pose on step `step`: panned about the vertical.
glm::dmat4 PanTransform(std::uint64_t step) {
    const double yaw =
        0.14 * std::sin(0.8 * static_cast<double>(step));
    const glm::dvec3 look = kTarget - kEye;
    const glm::dvec3 turned(
        look.x * std::cos(yaw) - look.y * std::sin(yaw),
        look.x * std::sin(yaw) + look.y * std::cos(yaw), look.z);
    return LookAtTransform(kEye, kEye + turned, {0.0, 0.0, 1.0});
}

class SensorEffectsDemo : public app::Demo {
  public:
    SensorEffectsDemo(app::Viewer* viewer,
                      const app::DemoOptions& options)
        : Demo(viewer, options), effects_(Effects()) {
        numFrames = 2;  // the moving cameras' last frame and this
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        GroundPlane::Define(scene, "/World/Ground")
            .SetChecker(true)
            .SetCheckerSize(1.0)
            .SetCheckerColor0(glm::dvec3(0.08))
            .SetCheckerColor1(glm::dvec3(0.35));
        const glm::dvec3 colours[] = {
            {0.8, 0.3, 0.2}, {0.3, 0.6, 0.85}, {0.85, 0.8, 0.3}};
        for (int i = 0; i < 3; ++i) {
            const Path ball("/World/Ball" + std::to_string(i));
            Sphere::Define(scene, ball)
                .SetRadius(0.5)
                .SetDisplayColor(colours[i]);
            SetXform(scene, ball,
                     {.translate =
                          glm::dvec3(1.4 * (i - 1), 0.5 * i, 0.5)});
        }
        // A small, very bright beacon, far beyond the full well.
        Material::Define(scene, "/World/Glow")
            .SetEmissiveColor(glm::dvec3(2e7));
        Sphere::Define(scene, "/World/Beacon")
            .SetRadius(0.06)
            .SetMaterialBinding(Path("/World/Glow"));
        SetXform(scene, "/World/Beacon",
                 {.translate = glm::dvec3(0.5, 3.0, 2.6)});
        // The Sun to the side, outside the field, for stray light.
        DistantLight::Define(scene, "/World/Sun")
            .SetIrradiance(1000.0)
            .SetPhysicalIlluminant(std::string(tokens::kSun))
            .SetAngle(0.53)
            .SetTransform(DistantLightTransform(
                glm::normalize(glm::dvec3(-1.0, 0.5, 0.6))));

        Xform::Define(scene, "/World/Cameras");
        for (Rig rig : kRigs) {
            DefineCamera(rig);
        }
        for (const Effect& effect : effects_) {
            DefineProduct(effect);
        }
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(CameraOf(Rig::Rgb))
            .SetResolution(glm::dvec2(kCols * 3, kRows * 3))
            .SetAmbient(0.0);
        if (viewer != nullptr) {
            viewer->SetActiveCamera(CameraOf(Rig::Rgb));
        }
    }

    // Pans the moving cameras and renders every effect, logging each
    // image (new noise, hits and errors every frame).
    void Step() override {
        for (Rig rig : kRigs) {
            if (IsMoving(rig)) {
                Camera(GetScene(), CameraOf(rig))
                    .SetTransform(PanTransform(steps_));
            }
        }
        ++steps_;
        std::vector<Path> products;
        for (const Effect& effect : effects_) {
            products.push_back(ProductOf(effect));
        }
        RenderProductSetOutputs outputs =
            GetRenderer().Step(GetScene(), products);
        tiles_.clear();
        for (const Effect& effect : effects_) {
            const Tensor var = outputs.at(ProductOf(effect))
                                   .vars.at(effect.var)
                                   .Map(MapDevice::Cpu)
                                   .tensor;
            tiles_.push_back(Tile(var, effect));
            LogImage("effects/" + effect.id, tiles_.back());
        }
    }

    // Every effect's image, three to a row in the table's order.
    void SaveOutput(const std::filesystem::path& path) override {
        if (tiles_.empty()) {
            Step();
        }
        app::WritePng(path, Grid());
    }

    void TestFinal() override {
        std::vector<std::vector<std::uint8_t>> pixels;
        for (const Tensor& t : tiles_) {
            const auto* data =
                static_cast<const std::uint8_t*>(t.Data());
            pixels.emplace_back(data, data + t.Elements());
        }
        const auto tile = [&](const std::string& id)
            -> const std::vector<std::uint8_t>& {
            for (std::size_t i = 0; i < effects_.size(); ++i) {
                if (effects_[i].id == id) {
                    return pixels[i];
                }
            }
            throw std::runtime_error("no effect " + id);
        };
        const auto& clean = tile("clean");
        double mean = 0.0;
        for (std::size_t i = 0; i < clean.size(); i += 4) {
            mean += clean[i];
        }
        mean /= kRows * kCols;
        std::printf("clean mean %.1f DN\n", mean);
        Check(mean > 25.0 && mean < 140.0,
              "the clean image should be well exposed");
        for (std::size_t e = 1; e < effects_.size(); ++e) {
            int changed = 0;
            for (std::size_t i = 0; i < clean.size(); i += 4) {
                changed +=
                    std::abs(pixels[e][i] - clean[i]) > 2 ? 1 : 0;
            }
            std::printf("%s: %d pixels changed\n",
                        effects_[e].id.c_str(), changed);
            Check(changed > 20,
                  effects_[e].id + " should change the image");
        }
        // The beacon's column: smear streaks it, above and below.
        std::size_t brightest = 0;
        for (std::size_t i = 0; i < clean.size(); i += 4) {
            brightest = clean[i] > clean[brightest] ? i : brightest;
        }
        const std::size_t column = brightest / 4 % kCols;
        const auto columnMean =
            [&](const std::vector<std::uint8_t>& image,
                std::size_t x) {
                double sum = 0.0;
                for (std::size_t y = 0; y < kRows; ++y) {
                    sum += image[(y * kCols + x) * 4];
                }
                return sum / kRows;
            };
        Check(columnMean(tile("smear"), column) >
                  columnMean(clean, column) + 5.0,
              "smear should streak the beacon's column");
        // Vignetting darkens the bottom corners, which see the
        // ground.
        const auto corners =
            [&](const std::vector<std::uint8_t>& image) {
                double sum = 0.0;
                for (std::size_t y = kRows - 20; y < kRows; ++y) {
                    for (std::size_t x = 0; x < 20; ++x) {
                        sum += image[(y * kCols + x) * 4] +
                               image[(y * kCols + kCols - 1 - x) * 4];
                    }
                }
                return sum;
            };
        Check(corners(tile("vignetting")) < 0.8 * corners(clean),
              "vignetting should darken the corners");
        std::set<std::uint8_t> codes;
        for (std::size_t i = 0; i < clean.size(); i += 4) {
            codes.insert(tile("compand_4bit")[i]);
        }
        Check(codes.size() <= 16,
              "4-bit companding should leave 16 codes at most");
        int lostRun = 0;
        int longest = 0;
        const auto& comms = tile("comms_errors");
        for (std::size_t i = 0; i < comms.size(); i += 4) {
            lostRun = comms[i] == 0 && clean[i] > 0 ? lostRun + 1 : 0;
            longest = std::max(longest, lostRun);
        }
        Check(longest >= kCols / 2,
              "a lost packet should blank much of a line");
    }

  private:
    void DefineCamera(Rig rig) {
        Scene& scene = GetScene();
        const bool low = rig == Rig::LowLight;
        const double seconds = low ? kExposureTime / 16.0
                               : rig == Rig::Short
                                   ? kExposureTime / 4.0
                                   : kExposureTime;
        Camera camera = Camera::Define(scene, CameraOf(rig));
        camera.SetFocalLength(3.2)
            .SetHorizontalAperture(kCols * kPitchMm)
            .SetVerticalAperture(kRows * kPitchMm)
            .SetSpectralBands(8)
            .SetSensorType(IsColour(rig) ? "rgb" : "mono")
            .SetSensorAdcBits(8)
            .SetSensorNoise(low)
            .SetExposureFStop(8.0)
            .SetExposureTime(seconds)
            .SetExposureIso(low ? 1600.0 : 100.0)
            .SetExposure(IsColour(rig) ? 1.5 : 0.0)
            .SetTransform(
                LookAtTransform(kEye, kTarget, {0.0, 0.0, 1.0}));
        switch (rig) {
        case Rig::Defocus:
            camera.SetFStop(1.4).SetFocusDistance(0.25);
            break;
        case Rig::Distorted:
            camera.SetLensDistortion(
                std::vector<float>{-0.25f, 0.05f});
            break;
        case Rig::Fisheye:
            camera.SetLensProjection(std::string(tokens::kFisheye));
            break;
        case Rig::Vignetted:
            camera.SetLensVignetting(true);
            break;
        case Rig::Chromatic:
            camera.SetLensChromaticScale(std::vector<float>{
                400.0f, 1.03f, 550.0f, 1.0f, 700.0f, 0.97f});
            break;
        case Rig::Rolling:
            camera.SetShutterRolling(1.0);
            break;
        case Rig::Blurred:
            camera.SetShutterOpen(-1.0)
                .SetShutterClose(0.0)
                .SetShutterSamples(16);
            break;
        default:
            break;
        }
    }

    void DefineProduct(const Effect& effect) {
        Scene& scene = GetScene();
        const Path product = ProductOf(effect);
        const Path var = VarOf(effect);
        RenderVar::Define(scene, var).SetSourceName(effect.var);
        std::vector<Path> filters;
        for (std::size_t i = 0; i < effect.steps.size(); ++i) {
            const FilterStep& step = effect.steps[i];
            const Path path = product.Child("Chain").Child(
                "Step" + std::to_string(i));
            SensorFilterPrim::Define(scene, path)
                .SetFilterType(step.type);
            for (const auto& [name, value] : step.inputs) {
                scene.Set(path, "inputs:" + name, value);
            }
            for (const auto& [name, values] : step.arrays) {
                scene.Set(path, "inputs:" + name, values);
            }
            filters.push_back(path);
        }
        RenderProduct::Define(scene, product)
            .SetCamera(CameraOf(effect.rig))
            .SetResolution(glm::dvec2(kCols, kRows))
            .SetOrderedVars({var})
            .SetFilters(filters);
    }

    // An 8-bit RGBA image of a product's var: a mono sensor's numbers
    // as grey, an RGB sensor's balanced to a grey world.
    static Tensor Tile(const Tensor& var, const Effect& effect) {
        const std::size_t count = std::size_t(kRows) * kCols;
        const auto channels =
            static_cast<std::size_t>(var.Elements()) / count;
        const bool colour = IsColour(effect.rig);
        const auto at = [&](std::size_t i) -> double {
            switch (var.Type()) {
            case DType::UInt16:
                return var.As<std::uint16_t>()[i];
            case DType::UInt8:
                return var.As<std::uint8_t>()[i];
            default:
                return var.As<float>()[i];
            }
        };
        double gain[3] = {1.0, 1.0, 1.0};
        if (colour && effect.var != "LdrColor") {
            double mean[3] = {0.0, 0.0, 0.0};
            for (std::size_t p = 0; p < count; ++p) {
                for (std::size_t c = 0; c < 3; ++c) {
                    mean[c] += at(p * channels + c);
                }
            }
            for (std::size_t c = 0; c < 3; ++c) {
                gain[c] = mean[1] / std::max(mean[c], 1.0);
            }
        }
        std::vector<std::uint8_t> pixels(count * 4, 255);
        for (std::size_t p = 0; p < count; ++p) {
            for (std::size_t c = 0; c < 3; ++c) {
                const double value =
                    at(p * channels + (colour ? c : 0));
                pixels[p * 4 + c] =
                    static_cast<std::uint8_t>(std::clamp(
                        value * effect.scale * gain[c], 0.0, 255.0));
            }
        }
        return Tensor::Copy<std::uint8_t>(pixels, {kRows, kCols, 4});
    }

    Tensor Grid() const {
        const std::size_t rows =
            (tiles_.size() + kGridColumns - 1) / kGridColumns;
        const std::size_t width =
            kGridColumns * kCols + (kGridColumns - 1) * kGap;
        const std::size_t height = rows * kRows + (rows - 1) * kGap;
        std::vector<std::uint8_t> pixels(width * height * 4, 0);
        for (std::size_t t = 0; t < tiles_.size(); ++t) {
            const std::size_t x0 = t % kGridColumns * (kCols + kGap);
            const std::size_t y0 = t / kGridColumns * (kRows + kGap);
            const auto* tile =
                static_cast<const std::uint8_t*>(tiles_[t].Data());
            for (std::size_t y = 0; y < kRows; ++y) {
                std::copy_n(tile + y * kCols * 4, kCols * 4,
                            &pixels[((y0 + y) * width + x0) * 4]);
            }
        }
        for (std::size_t i = 3; i < pixels.size(); i += 4) {
            pixels[i] = 255;
        }
        return Tensor::Copy<std::uint8_t>(
            pixels, {static_cast<std::int64_t>(height),
                     static_cast<std::int64_t>(width), 4});
    }

    std::vector<Effect> effects_;
    std::vector<Tensor> tiles_;
    std::uint64_t steps_ = 0;
};

}  // namespace

ERNIE_REGISTER_DEMO(
    SensorEffectsDemo, "cameras", "sensor_effects",
    "A gallery of the built-in sensor effects: the detector's noise, "
    "defects, radiation and readout, the optics' blur, flare and "
    "distortion, shutters, downlink errors, companding and colour.");
