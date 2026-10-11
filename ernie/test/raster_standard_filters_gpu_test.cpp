// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/offscreen_target.h"
#include "ernie/raster/delegate.h"
#include "ernie/scene/camera.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/terrain_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/util/astronomy.h"
#include "support/delegate_scenes.h"
#include "support/vulkan_fixture.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <vector>

using ernie::Path;
using ernie::RasterDelegate;
using ernie::Scene;
using ernie::Tensor;
using namespace ernie::testing;
namespace tokens = ernie::tokens;

namespace {

using Steps =
    std::vector<std::pair<std::string, std::map<std::string, float>>>;
/// Float-array inputs by step index, then input name.
using Arrays =
    std::map<std::size_t, std::map<std::string, std::vector<float>>>;

/// Render vars `vars` (source names, such as "Electrons" or
/// "aov:overflow") of MeshScene's camera (without noise) through
/// SensorFilter prims of `steps`, each with its own inputs and
/// `arrays`, as floats.
std::map<std::string, std::vector<float>>
VarsThrough(RasterDelegate& delegate, Scene& scene,
            const std::vector<std::string>& vars, const Steps& steps,
            const Arrays& arrays = {}) {
    ernie::Camera(scene, "/World/Camera").SetSensorNoise(false);
    std::vector<Path> filters;
    for (std::size_t i = 0; i < steps.size(); ++i) {
        const Path path("/Render/Filters/F" + std::to_string(i));
        scene.Remove(path);
        ernie::SensorFilterPrim::Define(scene, path)
            .SetFilterType(steps[i].first);
        for (const auto& [name, value] : steps[i].second) {
            scene.Set(path, "inputs:" + name, value);
        }
        if (const auto found = arrays.find(i);
            found != arrays.end()) {
            for (const auto& [name, values] : found->second) {
                scene.Set(path, "inputs:" + name, values);
            }
        }
        filters.push_back(path);
    }
    std::vector<Path> varPaths;
    for (std::size_t i = 0; i < vars.size(); ++i) {
        const Path varPath("/Render/Vars/V" + std::to_string(i));
        ernie::RenderVar::Define(scene, varPath)
            .SetSourceName(vars[i]);
        varPaths.push_back(varPath);
    }
    scene.Remove("/Render/Many");
    ernie::RenderProduct::Define(scene, "/Render/Many")
        .SetCamera("/World/Camera")
        .SetResolution(glm::dvec2(kSize, kSize));
    scene.Set("/Render/Many", tokens::kOrderedVars, varPaths);
    scene.Set("/Render/Many", tokens::kFilters, filters);
    const Path product[] = {"/Render/Many"};
    auto outputs = delegate.Step(scene, product);
    std::map<std::string, std::vector<float>> values;
    for (const std::string& var : vars) {
        const Tensor tensor = outputs.at("/Render/Many")
                                  .vars.at(var)
                                  .Map(ernie::MapDevice::Cpu)
                                  .tensor;
        std::vector<float>& out = values[var];
        out.resize(static_cast<std::size_t>(tensor.Elements()));
        for (std::size_t i = 0; i < out.size(); ++i) {
            out[i] =
                tensor.Type() == ernie::DType::UInt16
                    ? static_cast<float>(
                          tensor.As<std::uint16_t>()[i])
                : tensor.Type() == ernie::DType::UInt8
                    ? static_cast<float>(tensor.As<std::uint8_t>()[i])
                    : tensor.As<float>()[i];
        }
    }
    return values;
}

/// Render var `var` of MeshScene's camera (without noise) through
/// SensorFilter prims of `types`, each step with its own inputs, as
/// floats.
std::vector<float>
VarThrough(RasterDelegate& delegate, Scene& scene,
           std::string_view var,
           const std::vector<
               std::pair<std::string, std::map<std::string, float>>>&
               steps) {
    ernie::Camera(scene, "/World/Camera").SetSensorNoise(false);
    std::vector<Path> filters;
    for (std::size_t i = 0; i < steps.size(); ++i) {
        const Path path("/Render/Filters/F" + std::to_string(i));
        scene.Remove(path);
        ernie::SensorFilterPrim::Define(scene, path)
            .SetFilterType(steps[i].first);
        for (const auto& [name, value] : steps[i].second) {
            scene.Set(path, "inputs:" + name, value);
        }
        filters.push_back(path);
    }
    const Path varPath("/Render/Vars/" + std::string(var));
    ernie::RenderVar::Define(scene, varPath).SetSourceName(var);
    ernie::RenderProduct::Define(scene, "/Render/Out")
        .SetCamera("/World/Camera")
        .SetResolution(glm::dvec2(kSize, kSize));
    scene.Set("/Render/Out", tokens::kOrderedVars,
              std::vector<Path>{varPath});
    scene.Set("/Render/Out", tokens::kFilters, filters);
    const Path product[] = {"/Render/Out"};
    const Tensor tensor = delegate.Step(scene, product)
                              .at("/Render/Out")
                              .vars.at(std::string(var))
                              .Map(ernie::MapDevice::Cpu)
                              .tensor;
    std::vector<float> values(
        static_cast<std::size_t>(tensor.Elements()));
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] =
            tensor.Type() == ernie::DType::UInt16
                ? static_cast<float>(tensor.As<std::uint16_t>()[i])
                : tensor.As<float>()[i];
    }
    return values;
}

double Sum(const std::vector<float>& values, std::size_t channel) {
    double sum = 0.0;
    for (std::size_t i = channel; i < values.size(); i += 4) {
        sum += static_cast<double>(values[i]);
    }
    return sum;
}

/// Total variation of a channel: the sum of differences between
/// horizontal neighbours.
double Variation(const std::vector<float>& values,
                 std::size_t channel) {
    double total = 0.0;
    for (std::size_t y = 0; y < kSize; ++y) {
        for (std::size_t x = 1; x < kSize; ++x) {
            total += static_cast<double>(
                std::abs(values[(y * kSize + x) * 4 + channel] -
                         values[(y * kSize + x - 1) * 4 + channel]));
        }
    }
    return total;
}

const std::map<std::string, float> kNone;

/// The size of StarThrough's image.
constexpr std::size_t kStarSize = 64;

/// The Bessel function J_n(x) by Bessel's integral, (1 / pi) int_0^pi
/// cos(n t - x sin t) dt, whose periodic integrand the trapezoid rule
/// integrates to near machine precision.
double BesselJ(int n, double x) {
    constexpr int kSteps = 400;
    double sum = 0.0;
    for (int i = 0; i <= kSteps; ++i) {
        const double t = M_PI * i / kSteps;
        sum += (i == 0 || i == kSteps ? 0.5 : 1.0) *
               std::cos(n * t - x * std::sin(t));
    }
    return sum / kSteps;
}

/// The share of an Airy pattern's light within x = pi r / (lambda N)
/// of its centre, 1 - J0(x)^2 - J1(x)^2 (Born and Wolf, Principles of
/// Optics, section 8.5.2).
double AiryEncircled(double x) {
    return 1.0 - BesselJ(0, x) * BesselJ(0, x) -
           BesselJ(1, x) * BesselJ(1, x);
}

/// A magnitude 0 star of the Sun's temperature just off the axis of
/// FlatScene's camera (without the terrain and sun) at f/`fNumber`:
/// its SpectralRadiance through SensorFilter prims of `steps`, as a
/// kStarSize square of each band.
std::vector<float> StarThrough(RasterDelegate& delegate,
                               double fNumber, const Steps& steps) {
    Scene scene = FlatScene();
    scene.Remove("/World/Terrain");
    scene.Remove("/World/Sun");
    scene.Set("/World/Camera", tokens::kTransform,
              ernie::LookAtTransform({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
                                     {0.0, 0.0, 1.0}));
    ernie::Camera(scene, "/World/Camera").SetExposureFStop(fNumber);
    const auto one = [](float value) {
        return Tensor::Copy<float>(std::array{value}, {1});
    };
    ernie::StarField::Define(scene, "/World/Stars")
        .SetRightAscension(one(0.003f))
        .SetDeclination(one(0.002f))
        .SetMagnitude(one(0.0f))
        .SetTemperature(one(5772.0f));
    std::vector<Path> filters;
    for (std::size_t i = 0; i < steps.size(); ++i) {
        const Path path("/Render/Filters/F" + std::to_string(i));
        ernie::SensorFilterPrim::Define(scene, path)
            .SetFilterType(steps[i].first);
        for (const auto& [name, value] : steps[i].second) {
            scene.Set(path, "inputs:" + name, value);
        }
        filters.push_back(path);
    }
    const Path var("/Render/Vars/Star");
    ernie::RenderVar::Define(scene, var)
        .SetSourceName("SpectralRadiance");
    ernie::RenderProduct::Define(scene, "/Render/Star")
        .SetCamera("/World/Camera")
        .SetResolution(glm::dvec2(kStarSize, kStarSize));
    scene.Set("/Render/Star", tokens::kOrderedVars,
              std::vector<Path>{var});
    scene.Set("/Render/Star", tokens::kFilters, filters);
    const Path product[] = {"/Render/Star"};
    const Tensor tensor = delegate.Step(scene, product)
                              .at("/Render/Star")
                              .vars.at("SpectralRadiance")
                              .Map(ernie::MapDevice::Cpu)
                              .tensor;
    return {tensor.As<float>(),
            tensor.As<float>() + tensor.Elements()};
}

/// Band `band` of a StarThrough image at (`row`, `col`).
double StarAt(const std::vector<float>& image, std::size_t band,
              std::size_t row, std::size_t col) {
    return static_cast<double>(
        image[(band * kStarSize + row) * kStarSize + col]);
}

}  // namespace

TEST_F(RasterDelegateTest, TheStandardSensorFiltersAreRegistered) {
    RasterDelegate delegate(*context_);
    for (const char* id :
         {"psf", "defocus", "bayer", "demosaic", "whiteBalance",
          "colorMatrix", "compand", "detectorPattern", "readoutTaps",
          "tdi", "defectPixels", "smear", "darkCurrent", "jpeg"}) {
        EXPECT_NE(delegate.SensorFilters().Find(id), nullptr) << id;
    }
}

TEST_F(RasterDelegateTest,
       ThePointSpreadFunctionSpreadsLightAndKeepsIt) {
    RasterDelegate delegate(*context_);
    Scene scene = MeshScene();
    const auto base =
        VarThrough(delegate, scene, "HdrColor", {{"cie", kNone}});
    const auto blurred =
        VarThrough(delegate, scene, "HdrColor",
                   {{"psf", {{"blur", 2.0f}}}, {"cie", kNone}});
    EXPECT_NEAR(Sum(blurred, 0), Sum(base, 0), 0.01 * Sum(base, 0))
        << "light is kept";
    EXPECT_LT(Variation(blurred, 0), 0.8 * Variation(base, 0))
        << "edges soften";
    // Diffraction alone at f/1 spreads well under a pixel at these
    // pixels: nearly untouched.
    const auto diffraction =
        VarThrough(delegate, scene, "HdrColor",
                   {{"psf", kNone}, {"cie", kNone}});
    EXPECT_NEAR(Sum(diffraction, 0), Sum(base, 0),
                0.01 * Sum(base, 0));
}

TEST_F(
    RasterDelegateTest,
    TheAiryPointSpreadGrowsStarsIntoRingedDiscsAndKeepsTheirLight) {
    RasterDelegate delegate(*context_);
    const ernie::CameraParams camera = ernie::CameraParamsOf(
        FlatScene(), "/World/Camera", kStarSize, kStarSize);
    const double f = camera.FocalLengthPixels();
    const double pitchUm = camera.PixelPitchMetres() * 1e6;
    const auto pixels = [](std::size_t i) {
        return static_cast<double>(i) + 0.5;
    };
    // Band `band`'s spectral irradiance: its pixels' radiance times
    // their solid angles.
    const auto irradiance = [&](const std::vector<float>& image,
                                std::size_t band) {
        double sum = 0.0;
        for (std::size_t row = 0; row < kStarSize; ++row) {
            for (std::size_t col = 0; col < kStarSize; ++col) {
                const double x = (pixels(col) - 0.5 * kStarSize) / f;
                const double y = (pixels(row) - 0.5 * kStarSize) / f;
                const double cosine =
                    1.0 / std::sqrt(1.0 + x * x + y * y);
                sum += StarAt(image, band, row, col) * cosine *
                       cosine * cosine / (f * f);
            }
        }
        return sum;
    };
    // The share of band `band`'s light in pixels whose centres lie
    // between `inner` and `outer` pixels of `centre`.
    const auto within = [&](const std::vector<float>& image,
                            std::size_t band, glm::dvec2 centre,
                            double inner, double outer) {
        double part = 0.0;
        double total = 0.0;
        for (std::size_t row = 0; row < kStarSize; ++row) {
            for (std::size_t col = 0; col < kStarSize; ++col) {
                const double r = glm::length(
                    glm::dvec2(pixels(col), pixels(row)) - centre);
                const double value = StarAt(image, band, row, col);
                part += r >= inner && r < outer ? value : 0.0;
                total += value;
            }
        }
        return part / total;
    };
    // lambda N 3 to 5 pixels, so the star's own pixel or two of
    // spread hardly blurs the pattern.
    constexpr double kFNumber = 1650.0;
    constexpr float kRadius = 30.0f;
    const auto airy = [&](double fNumber, float obstruction) {
        return StarThrough(delegate, fNumber,
                           {{"psf",
                             {{"profile", 1.0f},
                              {"maxRadius", kRadius},
                              {"obstruction", obstruction}}}});
    };
    const std::vector<float> sharp =
        StarThrough(delegate, kFNumber, {});
    const std::vector<float> clear = airy(kFNumber, 0.0f);
    const std::vector<float> slower = airy(2.0 * kFNumber, 0.0f);
    const std::vector<float> obstructed = airy(kFNumber, 0.33f);
    const std::vector<float> gaussian = StarThrough(
        delegate, kFNumber, {{"psf", {{"maxRadius", kRadius}}}});

    // The star where the sharp image puts it.
    glm::dvec2 centre(0.0);
    double weight = 0.0;
    for (std::size_t row = 0; row < kStarSize; ++row) {
        for (std::size_t col = 0; col < kStarSize; ++col) {
            const double value = StarAt(sharp, 12, row, col);
            centre += value * glm::dvec2(pixels(col), pixels(row));
            weight += value;
        }
    }
    ASSERT_GT(weight, 0.0);
    centre /= weight;
    ASSERT_LT(glm::length(centre - glm::dvec2(0.5 * kStarSize)), 4.0)
        << "near the middle";

    // The kernel's square holds the light of the pattern within about
    // its own area's radius, and keeps all of it, so the pattern's
    // shares within it grow by one over that.
    const double kernel = (2.0 * kRadius + 1.0) / std::sqrt(M_PI);
    for (const std::size_t band : {std::size_t{2}, std::size_t{12}}) {
        const double expected = ernie::BlackbodySpectralIrradiance(
            camera.bands.Lower(static_cast<int>(band)) +
                0.5 * camera.bands.Width(),
            5772.0, 2.54e-6);
        for (const auto* image : {&clear, &slower, &obstructed}) {
            EXPECT_NEAR(irradiance(*image, band),
                        irradiance(sharp, band),
                        0.01 * irradiance(sharp, band))
                << "light is kept, band " << band;
        }
        EXPECT_NEAR(irradiance(clear, band), expected,
                    0.03 * expected)
            << "band " << band;

        // 84% of the light within the first dark ring, 1.22 lambda N:
        // wider for redder bands and slower lenses.
        for (const double fNumber : {kFNumber, 2.0 * kFNumber}) {
            const double lambdaN =
                camera.bands.Centre(static_cast<int>(band)) * 1e-3 *
                fNumber / pitchUm;  // pixels
            const auto& image = fNumber == kFNumber ? clear : slower;
            const double folded =
                AiryEncircled(M_PI * kernel / lambdaN);
            EXPECT_NEAR(
                within(image, band, centre, 0.0, 1.2197 * lambdaN),
                AiryEncircled(3.8317) / folded, 0.025)
                << "band " << band << ", f/" << fNumber;
            if (fNumber == kFNumber) {
                // The Gaussian fit, sigma = 0.42 lambda N, has no
                // rings: it holds 1 - exp(-(1.2197 / 0.42)^2 / 2) =
                // 98.5% there, less the pixels' and the star's own
                // spread at a pattern 2 to 3 pixels across, so well
                // above the Airy pattern's.
                EXPECT_GT(within(gaussian, band, centre, 0.0,
                                 1.2197 * lambdaN),
                          0.92);
                // The wings, falling as r^-3, hold a share falling as
                // 1 / r.
                const double wings =
                    (AiryEncircled(M_PI * 20.0 / lambdaN) -
                     AiryEncircled(M_PI * 8.0 / lambdaN)) /
                    folded;
                EXPECT_NEAR(within(image, band, centre, 8.0, 20.0),
                            wings, 0.2 * wings)
                    << "band " << band;
                EXPECT_LT(within(gaussian, band, centre, 8.0, 20.0),
                          0.1 * wings);
                // A secondary mirror a third of the aperture across
                // moves light from the core into the first rings
                // (Mahajan 1986): within 1.22 lambda N about 68% in
                // place of 84%, and from there to 3 lambda N about
                // 24% in place of 10%.
                EXPECT_LT(within(obstructed, band, centre, 0.0,
                                 1.2197 * lambdaN),
                          within(clear, band, centre, 0.0,
                                 1.2197 * lambdaN) -
                              0.1)
                    << "band " << band;
                EXPECT_GT(within(obstructed, band, centre,
                                 1.2197 * lambdaN, 3.0 * lambdaN),
                          1.5 * within(clear, band, centre,
                                       1.2197 * lambdaN,
                                       3.0 * lambdaN))
                    << "band " << band;
            }
        }
    }

    // Rings far finer than a pixel at a fast lens, and about a third
    // of a pixel (as LRO's NAC's at f/3.59 and 7 micrometre pixels):
    // light is kept, nearly all of it near the star.
    for (const double fNumber : {1.0, 160.0}) {
        const std::vector<float> fine = StarThrough(
            delegate, fNumber,
            {{"psf", {{"profile", 1.0f}, {"obstruction", 0.33f}}}});
        const std::vector<float> base =
            StarThrough(delegate, fNumber, {});
        for (const std::size_t band :
             {std::size_t{2}, std::size_t{12}}) {
            EXPECT_NEAR(irradiance(fine, band),
                        irradiance(base, band),
                        0.01 * irradiance(base, band))
                << "f/" << fNumber << ", band " << band;
            EXPECT_GT(within(fine, band, centre, 0.0, 3.0), 0.9)
                << "f/" << fNumber << ", band " << band;
        }
    }
}

TEST_F(RasterDelegateTest,
       DefocusBlursOutOfFocusPointsAndPinholesStaySharp) {
    RasterDelegate delegate(*context_);
    Scene scene = MeshScene();
    const auto base =
        VarThrough(delegate, scene, "HdrColor", {{"cie", kNone}});
    EXPECT_EQ(VarThrough(delegate, scene, "HdrColor",
                         {{"defocus", kNone}, {"cie", kNone}}),
              base)
        << "a pinhole (fStop 0) is untouched";
    // A fast 18 mm lens focused 10 cm away; the scene is 60 m away,
    // so a point there blurs into a disc about 9 pixels across (these
    // pixels are 0.3 mm).
    ernie::Camera(scene, "/World/Camera")
        .SetFStop(1.4)
        .SetFocusDistance(0.1);
    const auto blurred = VarThrough(
        delegate, scene, "HdrColor",
        {{"defocus", {{"maxRadius", 4.0f}}}, {"cie", kNone}});
    EXPECT_LT(Variation(blurred, 0), 0.8 * Variation(base, 0));
    EXPECT_NEAR(Sum(blurred, 0), Sum(base, 0), 0.05 * Sum(base, 0))
        << "light is kept";
}

TEST_F(RasterDelegateTest,
       WhiteBalanceAndColourMatricesScaleAndMixChannels) {
    RasterDelegate delegate(*context_);
    Scene scene = MeshScene();
    const auto base =
        VarThrough(delegate, scene, "HdrColor", {{"cie", kNone}});
    const auto balanced = VarThrough(
        delegate, scene, "HdrColor",
        {{"cie", kNone}, {"whiteBalance", {{"red", 2.0f}}}});
    const auto swapped = VarThrough(delegate, scene, "HdrColor",
                                    {{"cie", kNone},
                                     {"colorMatrix",
                                      {{"m00", 0.0f},
                                       {"m02", 1.0f},
                                       {"m20", 1.0f},
                                       {"m22", 0.0f}}}});
    for (std::size_t i = 0; i < base.size(); i += 4) {
        EXPECT_EQ(balanced[i], 2.0f * base[i]);
        EXPECT_EQ(balanced[i + 1], base[i + 1]);
        EXPECT_EQ(swapped[i], base[i + 2]);
        EXPECT_EQ(swapped[i + 2], base[i]);
    }
}

TEST_F(RasterDelegateTest,
       BayerMosaicsDemosaicRestoresAndCompandingCompresses) {
    RasterDelegate delegate(*context_);
    Scene scene = MeshScene();
    // Short enough not to saturate.
    ernie::Camera(scene, "/World/Camera").SetExposureTime(0.001);
    const auto raw = VarThrough(delegate, scene, "RawDN",
                                {{"qe", kNone}, {"adc", kNone}});
    const auto mosaic =
        VarThrough(delegate, scene, "RawDN",
                   {{"qe", kNone}, {"bayer", kNone}, {"adc", kNone}});
    // RGGB: each pixel's other channels hold the black level alone,
    // its own the full value.
    const ernie::CameraParams camera =
        ernie::CameraParamsOf(scene, "/World/Camera", kSize, kSize);
    const float black =
        static_cast<float>(camera.sensor.blackLevelDn);
    for (std::size_t y = 0; y < kSize; ++y) {
        for (std::size_t x = 0; x < kSize; ++x) {
            const std::size_t own = (y % 2 == 0)
                                        ? (x % 2 == 0 ? 0 : 1)
                                        : (x % 2 == 0 ? 1 : 2);
            for (std::size_t c = 0; c < 3; ++c) {
                const float value = mosaic[(y * kSize + x) * 4 + c];
                if (c == own) {
                    EXPECT_EQ(value, raw[(y * kSize + x) * 4 + c]);
                } else {
                    EXPECT_EQ(value, black)
                        << x << ", " << y << " channel " << c;
                }
            }
        }
    }
    // Demosaicing gives back the full colour where it is uniform:
    // close over the red quad, which then fills the image, both
    // methods restore every channel inside a two-pixel border.
    Scene close = MeshScene();
    close.Set("/World/Camera", tokens::kTransform,
              ernie::LookAtTransform({-12.0, 0.0, 3.0},
                                     {-12.0, 0.0, 0.0},
                                     {0.0, 1.0, 0.0}));
    ernie::Camera(close, "/World/Camera").SetExposureTime(0.001);
    const auto full = VarThrough(delegate, close, "RawDN",
                                 {{"qe", kNone}, {"adc", kNone}});
    ASSERT_GT(full[(kSize / 2 * kSize + kSize / 2) * 4],
              black + 10.0f)
        << "the quad is lit";
    for (const float method : {0.0f, 1.0f}) {
        const auto restored =
            VarThrough(delegate, close, "RawDN",
                       {{"qe", kNone},
                        {"bayer", kNone},
                        {"adc", kNone},
                        {"demosaic", {{"method", method}}}});
        for (std::size_t y = 2; y < kSize - 2; ++y) {
            for (std::size_t x = 2; x < kSize - 2; ++x) {
                for (std::size_t c = 0; c < 3; ++c) {
                    const std::size_t i = (y * kSize + x) * 4 + c;
                    ASSERT_NEAR(restored[i], full[i], 1.0f)
                        << "method " << method << " at " << x << ", "
                        << y << " channel " << c;
                }
            }
        }
    }
    // Companding 12 bits to 8: full scale to 255, and the square root
    // in between.
    const auto companded = VarThrough(
        delegate, scene, "RawDN",
        {{"qe", kNone},
         {"adc", kNone},
         {"compand", {{"inputBits", 12.0f}, {"outputBits", 8.0f}}}});
    for (std::size_t i = 0; i < raw.size(); ++i) {
        if (i % 4 == 3) {
            continue;
        }
        const float expected = std::floor(
            255.0f * std::sqrt(std::min(raw[i] / 4095.0f, 1.0f)) +
            0.5f);
        EXPECT_NEAR(companded[i], expected, 1.0f);
    }
}

TEST_F(RasterDelegateTest, CurveParametersTakeArraysOfAnyLength) {
    RasterDelegate delegate(*context_);
    Scene scene = MeshScene();
    const auto base =
        VarThrough(delegate, scene, "HdrColor", {{"cie", kNone}});
    // A single number fills the curve: half the light in every band.
    EXPECT_EQ(VarThrough(delegate, scene, "HdrColor",
                         {{"spectralTransmission", {{"curve", 0.5f}}},
                          {"cie", kNone}}),
              [&] {
                  std::vector<float> half = base;
                  for (std::size_t i = 0; i < half.size(); ++i) {
                      half[i] *= i % 4 == 3 ? 1.0f : 0.5f;
                  }
                  return half;
              }());

    // An array, resampled: transmission falling to nothing above 570
    // nm (ten samples over 360 to 830 nm) dims white rocks' red and
    // keeps their blue.
    ernie::Mesh(scene, "/World/Rocks/Rock")
        .SetDisplayColor(glm::dvec3(1.0));
    const auto white =
        VarThrough(delegate, scene, "HdrColor", {{"cie", kNone}});
    const auto cut =
        VarsThrough(
            delegate, scene, {"HdrColor"},
            {{"spectralTransmission", kNone}, {"cie", kNone}},
            {{0, {{"curve", {1, 1, 1, 1, 1, 0, 0, 0, 0, 0}}}}})
            .at("HdrColor");
    EXPECT_LT(Sum(cut, 0), 0.3 * Sum(white, 0))
        << "red passes little";
    EXPECT_GT(Sum(cut, 2), 0.7 * Sum(white, 2))
        << "blue mostly passes";

    // An inverting tone curve, as two samples resampled to 33.
    const auto ldr =
        VarsThrough(delegate, scene, {"LdrColor"},
                    {{"cie", kNone}, {"exposure", kNone}})
            .at("LdrColor");
    const auto flipped = VarsThrough(delegate, scene, {"LdrColor"},
                                     {{"cie", kNone},
                                      {"exposure", kNone},
                                      {"toneCurve", kNone}},
                                     {{2, {{"curve", {1.0f, 0.0f}}}}})
                             .at("LdrColor");
    for (std::size_t i = 0; i < ldr.size(); ++i) {
        if (i % 4 != 3) {
            ASSERT_NEAR(flipped[i], 255.0f - ldr[i], 1.0f) << i;
        }
    }
}

TEST_F(RasterDelegateTest,
       BloomingSpillsSaturatedChargeAlongColumns) {
    // MeshScene's red quad at the default 1 s exposure holds far more
    // than the full well.
    RasterDelegate delegate(*context_);
    Scene scene = MeshScene();
    const float well = static_cast<float>(
        ernie::CameraParamsOf(scene, "/World/Camera", kSize, kSize)
            .sensor.fullWellElectrons);
    const auto clipped =
        VarThrough(delegate, scene, "Electrons", {{"qe", kNone}});
    const auto bloomed = VarsThrough(
        delegate, scene, {"Electrons", "aov:overflow"},
        {{"qe", kNone}, {"blooming", {{"maxLength", 64.0f}}}});
    const auto& electrons = bloomed.at("Electrons");
    const auto& overflow = bloomed.at("aov:overflow");
    const auto at = [](const std::vector<float>& image, std::size_t x,
                       std::size_t y) {
        return image[(y * kSize + x) * 4];  // red
    };
    // The quad's brightest column at the middle row, and a column of
    // background beside it.
    std::size_t column = 0;
    for (std::size_t x = 0; x < kSize; ++x) {
        column =
            at(clipped, x, kSize / 2) > at(clipped, column, kSize / 2)
                ? x
                : column;
    }
    ASSERT_EQ(at(clipped, column, kSize / 2), well) << "saturated";
    std::size_t dark = column;
    while (dark > 0 && at(clipped, dark, kSize / 2) > 0.01f * well) {
        --dark;
    }
    --dark;  // a pixel clear of the quad's edge
    std::size_t streaked = 0;
    std::size_t before = 0;
    for (std::size_t y = 0; y < kSize; ++y) {
        streaked += at(electrons, column, y) >= well ? 1u : 0u;
        before += at(clipped, column, y) >= well ? 1u : 0u;
    }
    EXPECT_GT(streaked, before + 10)
        << "a vertical streak beyond the quad";
    EXPECT_LT(at(electrons, dark, kSize / 2), 0.01f * well)
        << "not sideways";
    EXPECT_GT(at(overflow, column, kSize / 2), 0.0f)
        << "the overflow AOV";
    EXPECT_EQ(at(overflow, column, 0), 0.0f)
        << "nothing overflowed above the quad";

    // An overflow drain takes it all: no streak.
    const auto drained = VarThrough(
        delegate, scene, "Electrons",
        {{"qe", kNone}, {"blooming", {{"antiBlooming", 1.0f}}}});
    EXPECT_EQ(drained, clipped);
}

TEST_F(RasterDelegateTest, AovsRunTheirFiltersAndNameTheirOutputs) {
    RasterDelegate delegate(*context_);
    Scene scene = MeshScene();
    ernie::Camera(scene, "/World/Camera").SetExposureTime(0.001);
    const Steps chain = {{"qe", kNone},
                         {"bayer", kNone},
                         {"adc", kNone},
                         {"demosaic", kNone}};
    // RawDN through bayer and adc alone is the mosaic itself.
    const auto mosaic =
        VarThrough(delegate, scene, "RawDN",
                   {{"qe", kNone}, {"bayer", kNone}, {"adc", kNone}});
    // Asking for the AOV alone runs the chain up to demosaic, which
    // writes the mosaic it read.
    const auto aov =
        VarsThrough(delegate, scene, {"aov:mosaic"}, chain)
            .at("aov:mosaic");
    ASSERT_EQ(aov.size(), mosaic.size());
    for (std::size_t y = 0; y < kSize; ++y) {
        for (std::size_t x = 0; x < kSize; ++x) {
            const std::size_t own = (y % 2 == 0)
                                        ? (x % 2 == 0 ? 0 : 1)
                                        : (x % 2 == 0 ? 1 : 2);
            const std::size_t i = (y * kSize + x) * 4 + own;
            ASSERT_EQ(aov[i], mosaic[i]) << x << ", " << y;
        }
    }
    // Both at once: the demosaiced RawDN and the mosaic.
    const auto both =
        VarsThrough(delegate, scene, {"RawDN", "aov:mosaic"}, chain);
    EXPECT_EQ(both.at("aov:mosaic"), aov);
    EXPECT_NE(both.at("RawDN"), mosaic);
    EXPECT_THROW(VarsThrough(delegate, scene, {"aov:missing"}, chain),
                 std::invalid_argument);
}

TEST_F(RasterDelegateTest,
       TheViewRunsTheFirstProductsChainAndShowsItsAovs) {
    RasterDelegate delegate(*context_);
    Scene scene = MeshScene();
    ernie::Camera(scene, "/World/Camera").SetExposureTime(0.001);
    const Steps chain = {{"qe", kNone},
                         {"bayer", kNone},
                         {"adc", kNone},
                         {"demosaic", kNone}};
    VarsThrough(delegate, scene, {"aov:mosaic"},
                chain);  // defines /Render/Many
    scene.Set("/Render/Settings", tokens::kProducts,
              std::vector<Path>{"/Render/Many"});
    const auto draw = [&](std::string_view var) {
        ernie::OffscreenTarget target(*context_, kSize, kSize);
        context_->Submit([&](VkCommandBuffer commands) {
            delegate.Record(commands, target.Prepare(commands), var);
        });
        return target.ReadColor();
    };
    delegate.Update(scene, "/Render/Settings", {kSize, kSize});
    std::vector<std::string> vars = delegate.ViewVars();
    EXPECT_EQ(vars.back(), "aov:mosaic");
    const std::vector<std::uint8_t> mosaic = draw("aov:mosaic");
    EXPECT_GT(std::accumulate(mosaic.begin(), mosaic.end(), 0L), 0L)
        << "the mosaic shows";
    EXPECT_NE(mosaic, draw(ernie::render_vars::kLdrColor));

    // Without products the view runs the default chain, which writes
    // no AOVs.
    scene.Clear("/Render/Settings", tokens::kProducts);
    delegate.Update(scene, "/Render/Settings", {kSize, kSize});
    vars = delegate.ViewVars();
    EXPECT_EQ(std::count(vars.begin(), vars.end(), "aov:mosaic"), 0);
    EXPECT_EQ(draw(ernie::render_vars::kLdrColor).size(),
              mosaic.size());
}

TEST_F(RasterDelegateTest, CustomIntegratorsStandInForQe) {
    // measuredQe given the camera's own red QE curve reproduces qe's
    // red electrons, and no electrons where its curves are zero.
    RasterDelegate delegate(*context_);
    Scene scene = MeshScene();
    ernie::Camera(scene, "/World/Camera")
        .SetExposureTime(1e-6);  // unsaturated
    const ernie::CameraParams camera =
        ernie::CameraParamsOf(scene, "/World/Camera", kSize, kSize);
    std::vector<float> red(48);
    for (std::size_t i = 0; i < red.size(); ++i) {
        const double nm = 360.0 + 10.0 * static_cast<double>(i);
        red[i] = static_cast<float>(
            camera.sensor.channels[0].quantumEfficiency(nm));
    }
    const auto builtin =
        VarThrough(delegate, scene, "Electrons", {{"qe", kNone}});
    const auto measured =
        VarsThrough(delegate, scene, {"Electrons", "RawDN"},
                    {{"measuredQe", kNone}, {"adc", kNone}},
                    {{0,
                      {{"red", red},
                       {"green", std::vector<float>{0.0f}},
                       {"blue", std::vector<float>{0.0f}}}}});
    const auto& electrons = measured.at("Electrons");
    EXPECT_NEAR(Sum(electrons, 0), Sum(builtin, 0),
                0.03 * Sum(builtin, 0));
    EXPECT_LT(Sum(electrons, 1), 1e-2) << "dark current alone";
    EXPECT_GT(Sum(measured.at("RawDN"), 0), 0.0)
        << "adc after a custom integrator";
    EXPECT_THROW(VarsThrough(delegate, scene, {"Electrons"},
                             {{"qe", kNone}, {"measuredQe", kNone}}),
                 std::invalid_argument)
        << "one integrator into electrons";
}

TEST_F(RasterDelegateTest,
       GlareSpreadsAHaloFromItsPyramidAndKeepsTheLight) {
    RasterDelegate delegate(*context_);
    Scene scene = MeshScene();
    const auto base =
        VarThrough(delegate, scene, "HdrColor", {{"cie", kNone}});
    const auto glare =
        VarThrough(delegate, scene, "HdrColor",
                   {{"cie", kNone}, {"glare", {{"strength", 0.2f}}}});
    EXPECT_NEAR(Sum(glare, 0), Sum(base, 0), 0.03 * Sum(base, 0))
        << "light is kept";
    // A dark pixel beside the red quad gains red, less the farther it
    // is.
    const auto at = [](const std::vector<float>& image, std::size_t x,
                       std::size_t y) {
        return image[(y * kSize + x) * 4];
    };
    std::size_t edge =
        0;  // the quad's leftmost lit column on the middle row
    while (edge < kSize && at(base, edge, kSize / 2) <= 0.0f) {
        ++edge;
    }
    ASSERT_GT(edge, 6u);
    EXPECT_EQ(at(base, edge - 2, kSize / 2), 0.0f);
    EXPECT_GT(at(glare, edge - 2, kSize / 2), 0.0f) << "the halo";
    EXPECT_GT(at(glare, edge - 2, kSize / 2),
              at(glare, edge - 6, kSize / 2))
        << "fading with distance";
}

TEST_F(
    RasterDelegateTest,
    FlareGhostsMirrorBrightSourcesThroughTheCentreAndKeepTheLight) {
    RasterDelegate delegate(*context_);
    Scene scene = MeshScene();
    ernie::Camera(scene, "/World/Camera")
        .SetExposureTime(1e-6);  // unsaturated
    const auto base =
        VarThrough(delegate, scene, "Electrons", {{"qe", kNone}});
    EXPECT_EQ(
        VarThrough(delegate, scene, "Electrons",
                   {{"qe", kNone}, {"flare", {{"strength", 0.0f}}}}),
        base)
        << "untouched at zero strength";
    const auto at = [](const std::vector<float>& image, std::size_t x,
                       std::size_t y) {
        return static_cast<double>(
            image[(y * kSize + x) * 4]);  // red
    };
    // The red quad, left of the centre: its light and centroid in
    // red.
    double source = 0.0;
    glm::dvec2 quad(0.0);
    for (std::size_t y = 0; y < kSize; ++y) {
        for (std::size_t x = 0; x < kSize / 2; ++x) {
            source += at(base, x, y);
            quad += at(base, x, y) * (glm::dvec2(x, y) + 0.5);
        }
    }
    ASSERT_GT(source, 0.0);
    quad /= source;
    const ernie::CameraParams camera =
        ernie::CameraParamsOf(scene, "/World/Camera", kSize, kSize);
    const glm::dvec2 centre = camera.PrincipalPoint();
    const float strength = 0.1f;
    for (const float scale : {-0.5f, -1.0f}) {
        // One small grey ghost at `scale`: (1 - strength) of the
        // image plus the ghosts.
        const auto flared =
            VarsThrough(
                delegate, scene, {"Electrons"},
                {{"qe", kNone},
                 {"flare",
                  {{"strength", strength}, {"ghosts", 1.0f}}}},
                {{1,
                  {{"scales", {scale}},
                   {"radii", {0.02f}},
                   {"weights", {1.0f}},
                   {"tintRed", {1.0f}},
                   {"tintGreen", {1.0f}},
                   {"tintBlue", {1.0f}}}}})
                .at("Electrons");
        EXPECT_NEAR(Sum(flared, 0), Sum(base, 0), 0.01 * Sum(base, 0))
            << "light is kept";
        // The quad's ghost, mirrored to the right of the centre.
        double ghost = 0.0;
        glm::dvec2 where(0.0);
        for (std::size_t y = 0; y < kSize; ++y) {
            for (std::size_t x = kSize / 2; x < kSize; ++x) {
                const double gained =
                    at(flared, x, y) -
                    (1.0 - strength) * at(base, x, y);
                ghost += gained;
                where += gained * (glm::dvec2(x, y) + 0.5);
            }
        }
        ASSERT_GT(ghost, 0.0) << scale;
        where /= ghost;
        EXPECT_NEAR(ghost, strength * source,
                    0.03 * strength * source)
            << scale;
        const glm::dvec2 expected =
            centre + static_cast<double>(scale) * (quad - centre);
        EXPECT_NEAR(where.x, expected.x, 0.3) << scale;
        EXPECT_NEAR(where.y, expected.y, 0.3) << scale;
    }
}

TEST_F(RasterDelegateTest, StrayLightComesFromEveryDistantLight) {
    // The sun straight behind the camera gives none; another distant
    // light as bright, 60 degrees off the axis to the right, gives
    // the veil the sun would there (as below).
    RasterDelegate delegate(*context_);
    Scene scene = MeshScene();
    const ernie::DistantLight sun(scene, "/World/Sun");
    ernie::DistantLight::Define(scene, "/World/Tilted")
        .SetIntensity(sun.GetIntensity())
        .SetShadowEnable(false);
    const double theta = glm::radians(60.0);
    scene.Set("/World/Tilted", tokens::kTransform,
              ernie::DistantLightTransform(
                  {std::sin(theta), 0.0, -std::cos(theta)}));
    const ernie::CameraParams camera =
        ernie::CameraParamsOf(scene, "/World/Camera", kSize, kSize);
    const double perPst =
        4.0 * camera.fStop * camera.fStop / camera.lens.transmission;
    const auto lit =
        VarThrough(delegate, scene, "HdrColor", {{"cie", kNone}});
    const auto veiled = VarThrough(
        delegate, scene, "HdrColor",
        {{"strayLight", {{"strength", 1e-3f}, {"gradient", 0.0f}}},
         {"cie", kNone}});
    const auto luminance = [](const std::vector<float>& image) {
        const float* rgb =
            &image[(kSize / 2 * kSize + kSize / 2) * 4];
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
    };
    const double expected = perPst * 1e-3 * std::pow(6.0, -2.0);
    EXPECT_NEAR(luminance(veiled) - luminance(lit), expected,
                0.02 * expected);
}

TEST_F(
    RasterDelegateTest,
    StrayLightRisesAsTheSunNearsTheFieldAndVanishesBehindTheCamera) {
    RasterDelegate delegate(*context_);
    Scene scene = MeshScene();  // the sun straight behind the camera
    const auto base =
        VarThrough(delegate, scene, "HdrColor", {{"cie", kNone}});
    EXPECT_EQ(VarThrough(delegate, scene, "HdrColor",
                         {{"strayLight", kNone}, {"cie", kNone}}),
              base)
        << "none from behind the camera";
    const ernie::CameraParams camera =
        ernie::CameraParamsOf(scene, "/World/Camera", kSize, kSize);
    // The camera looks down -z: the sun `degrees` off its axis, to
    // the right.
    const auto aim = [&scene](double degrees) {
        const double theta = glm::radians(degrees);
        scene.Set("/World/Sun", tokens::kTransform,
                  ernie::DistantLightTransform(
                      {std::sin(theta), 0.0, -std::cos(theta)}));
    };
    const auto luminance = [](const std::vector<float>& image,
                              std::size_t x, std::size_t y) {
        const float* rgb = &image[(y * kSize + x) * 4];
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
    };
    const std::size_t middle =
        kSize / 2;  // the background between the quad and the rocks
    // The veil at the centre without a gradient: the sun's (relative)
    // illuminance pi times the PST, through the camera equation, a
    // luminance of 4 N^2 PST / T.
    const double perPst =
        4.0 * camera.fStop * camera.fStop / camera.lens.transmission;
    double previous = 0.0;
    for (const double degrees : {80.0, 60.0, 40.0}) {
        aim(degrees);
        const auto lit =
            VarThrough(delegate, scene, "HdrColor", {{"cie", kNone}});
        const auto veiled =
            VarThrough(delegate, scene, "HdrColor",
                       {{"strayLight",
                         {{"strength", 1e-3f}, {"gradient", 0.0f}}},
                        {"cie", kNone}});
        const double expected =
            perPst * 1e-3 * std::pow(degrees / 10.0, -2.0);
        const double veil = luminance(veiled, middle, middle) -
                            luminance(lit, middle, middle);
        EXPECT_NEAR(veil, expected, 0.02 * expected) << degrees;
        EXPECT_GT(veil, previous)
            << "rising as the sun nears the field";
        previous = veil;
        EXPECT_EQ(VarThrough(delegate, scene, "HdrColor",
                             {{"strayLight", {{"strength", 0.0f}}},
                              {"cie", kNone}}),
                  lit)
            << "untouched at zero strength";
    }
    // Brighter towards the sun's side (right), the mean kept at the
    // centre.
    const auto lit =
        VarThrough(delegate, scene, "HdrColor", {{"cie", kNone}});
    const auto ramped = VarThrough(
        delegate, scene, "HdrColor",
        {{"strayLight", {{"strength", 1e-3f}, {"gradient", 1.0f}}},
         {"cie", kNone}});
    const double right = luminance(ramped, kSize - 8, middle) -
                         luminance(lit, kSize - 8, middle);
    const double left =
        luminance(ramped, 8, middle) - luminance(lit, 8, middle);
    EXPECT_GT(right, 1.5 * left);
    EXPECT_NEAR(0.5 * (left + right), previous, 0.05 * previous);
    // A PST table instead of the power law: 1e-4 at 30 degrees, 1e-6
    // at 50, so 1e-5 at 40.
    const auto tabled =
        VarsThrough(
            delegate, scene, {"HdrColor"},
            {{"strayLight", {{"strength", 1.0f}, {"gradient", 0.0f}}},
             {"cie", kNone}},
            {{0,
              {{"angles", {30.0f, 50.0f}},
               {"logValues", {-4.0f, -6.0f}}}}})
            .at("HdrColor");
    EXPECT_NEAR(luminance(tabled, middle, middle) -
                    luminance(lit, middle, middle),
                perPst * 1e-5, 0.03 * perPst * 1e-5);
    // Past the cutoff the aperture no longer sees the sun; in the
    // field, it is glare's case.
    for (const double degrees : {100.0, 10.0}) {
        aim(degrees);
        EXPECT_EQ(
            VarThrough(delegate, scene, "HdrColor",
                       {{"strayLight", kNone}, {"cie", kNone}}),
            VarThrough(delegate, scene, "HdrColor", {{"cie", kNone}}))
            << degrees;
    }
}

TEST_F(RasterDelegateTest, RenderVarsCanNameTheirOwnChains) {
    RasterDelegate delegate(*context_);
    Scene scene = MeshScene();
    ernie::Camera(scene, "/World/Camera")
        .SetSensorNoise(false)
        .SetExposureTime(0.001);
    const auto balancedAlone = VarThrough(
        delegate, scene, "HdrColor",
        {{"cie", kNone}, {"whiteBalance", {{"red", 2.0f}}}});
    const auto mosaicAlone =
        VarThrough(delegate, scene, "RawDN",
                   {{"qe", kNone}, {"bayer", kNone}, {"adc", kNone}});

    // One product: HdrColor through a white-balanced chain, RawDN
    // through a mosaicking one.
    const auto define =
        [&](const std::string& name, const std::string& type,
            const std::map<std::string, float>& inputs = {}) {
            const Path path("/Render/Chains/" + name);
            ernie::SensorFilterPrim::Define(scene, path)
                .SetFilterType(type);
            for (const auto& [input, value] : inputs) {
                scene.Set(path, "inputs:" + input, value);
            }
            return path;
        };
    const std::vector<Path> photo = {
        define("Cie", "cie"),
        define("Balance", "whiteBalance", {{"red", 2.0f}})};
    const std::vector<Path> raw = {define("Qe", "qe"),
                                   define("Bayer", "bayer"),
                                   define("Adc", "adc")};
    ernie::RenderVar::Define(scene, "/Render/Vars/Photo")
        .SetSourceName("HdrColor");
    scene.Set("/Render/Vars/Photo", tokens::kFilters, photo);
    ernie::RenderVar::Define(scene, "/Render/Vars/Raw")
        .SetSourceName("RawDN");
    scene.Set("/Render/Vars/Raw", tokens::kFilters, raw);
    ernie::RenderProduct::Define(scene, "/Render/Both")
        .SetCamera("/World/Camera")
        .SetResolution(glm::dvec2(kSize, kSize));
    scene.Set(
        "/Render/Both", tokens::kOrderedVars,
        std::vector<Path>{"/Render/Vars/Photo", "/Render/Vars/Raw"});
    const Path product[] = {"/Render/Both"};
    auto outputs = delegate.Step(scene, product);
    const Tensor hdr = outputs.at("/Render/Both")
                           .vars.at("HdrColor")
                           .Map(ernie::MapDevice::Cpu)
                           .tensor;
    const Tensor dn = outputs.at("/Render/Both")
                          .vars.at("RawDN")
                          .Map(ernie::MapDevice::Cpu)
                          .tensor;
    for (std::size_t i = 0; i < balancedAlone.size(); ++i) {
        ASSERT_EQ(hdr.As<float>()[i], balancedAlone[i]) << i;
        ASSERT_EQ(static_cast<float>(dn.As<std::uint16_t>()[i]),
                  mosaicAlone[i])
            << i;
    }

    // The same source from two different chains is ambiguous.
    ernie::RenderVar::Define(scene, "/Render/Vars/Raw2")
        .SetSourceName("RawDN");
    scene.Set(
        "/Render/Both", tokens::kOrderedVars,
        std::vector<Path>{"/Render/Vars/Raw", "/Render/Vars/Raw2"});
    EXPECT_THROW(delegate.Step(scene, product),
                 std::invalid_argument);
}

TEST_F(RasterDelegateTest,
       TemporalFiltersKeepStateBetweenAProductsFrames) {
    // Two frames of each: the red quad, then the quad gone.
    const auto frames = [&](const Steps& chain,
                            const std::string& var) {
        RasterDelegate delegate(*context_);
        Scene scene = MeshScene();
        ernie::Camera(scene, "/World/Camera")
            .SetExposureTime(1e-6);  // unsaturated
        const auto first = VarThrough(delegate, scene, var, chain);
        scene.Remove("/World/Quad");
        const auto second = VarThrough(delegate, scene, var, chain);
        return std::pair(first, second);
    };
    // The quad's brightest pixel, and its red.
    const auto brightest = [](const std::vector<float>& image) {
        std::size_t best = 0;
        for (std::size_t i = 0; i < image.size(); i += 4) {
            best = image[i] > image[best] ? i : best;
        }
        return best;
    };

    // Lag: 3/4 of the first frame's charge, then a ghost of the other
    // quarter.
    {
        const Steps chain = {{"qe", kNone},
                             {"lag", {{"fraction", 0.25f}}}};
        const auto [first, second] = frames(chain, "Electrons");
        RasterDelegate reference(*context_);
        Scene plain = MeshScene();
        ernie::Camera(plain, "/World/Camera").SetExposureTime(1e-6);
        const auto collected = VarThrough(
            reference, plain, "Electrons", {{"qe", kNone}});
        const std::size_t i = brightest(collected);
        ASSERT_GT(collected[i], 0.0f);
        EXPECT_NEAR(first[i], 0.75f * collected[i],
                    1e-3f * collected[i]);
        EXPECT_NEAR(second[i], 0.25f * collected[i],
                    1e-3f * collected[i])
            << "the ghost";
    }
    // A moving average: the first frame as it is, then half way to
    // the empty scene.
    {
        const Steps chain = {{"cie", kNone},
                             {"temporalAverage", {{"alpha", 0.5f}}}};
        const auto [first, second] = frames(chain, "HdrColor");
        const std::size_t i = brightest(first);
        ASSERT_GT(first[i], 0.0f);
        EXPECT_NEAR(second[i], 0.5f * first[i], 1e-4f * first[i]);
    }
    // A different chain starts a new history: the quad gone at once,
    // not half way.
    {
        RasterDelegate delegate(*context_);
        Scene scene = MeshScene();
        const auto lit = VarThrough(
            delegate, scene, "HdrColor",
            {{"cie", kNone}, {"temporalAverage", {{"alpha", 0.5f}}}});
        const std::size_t i = brightest(lit);
        scene.Remove("/World/Quad");
        const auto fresh =
            VarThrough(delegate, scene, "HdrColor",
                       {{"cie", kNone},
                        {"whiteBalance", kNone},
                        {"temporalAverage", {{"alpha", 0.5f}}}});
        EXPECT_EQ(fresh[i], 0.0f)
            << "no history after the chain changed";
    }
}

TEST_F(RasterDelegateTest, DetectorPatternsStripeAlongTheTrack) {
    RasterDelegate delegate(*context_);
    Scene scene = FlatScene();  // grey terrain filling the image
    ernie::Camera(scene, "/World/Camera")
        .SetExposureTime(1e-6);  // unsaturated
    const auto at = [](std::size_t x, std::size_t y) {
        return (y * kSize + x) * 4 + 1;
    };
    const auto base =
        VarThrough(delegate, scene, "Electrons", {{"qe", kNone}});
    // Gain alone: each column's electrons scale by its detector's
    // gain, all down the column.
    const auto gains = VarThrough(
        delegate, scene, "Electrons",
        {{"qe", kNone},
         {"detectorPattern", {{"gain", 0.05f}, {"offset", 0.0f}}}});
    std::vector<double> columnGain(kSize, 0.0);
    int lit = 0;
    for (std::size_t x = 0; x < kSize; ++x) {
        for (std::size_t y = 0; y < kSize; ++y) {
            if (base[at(x, y)] < 5.0f) {
                continue;
            }
            const double gain = gains[at(x, y)] / base[at(x, y)];
            if (columnGain[x] == 0.0) {
                columnGain[x] = gain;
            }
            EXPECT_NEAR(gain, columnGain[x], 1e-4)
                << "one gain down column " << x;
            ++lit;
        }
    }
    ASSERT_GT(lit, 100);
    std::vector<double> seen;
    std::copy_if(columnGain.begin(), columnGain.end(),
                 std::back_inserter(seen),
                 [](double g) { return g != 0.0; });
    const auto [low, high] =
        std::minmax_element(seen.begin(), seen.end());
    EXPECT_GT(*high - *low, 0.03) << "columns differ";
    // Offset alone with axis 1: a constant per row instead.
    const auto offsets = VarThrough(
        delegate, scene, "Electrons",
        {{"qe", kNone},
         {"detectorPattern",
          {{"gain", 0.0f}, {"offset", 50.0f}, {"axis", 1.0f}}}});
    std::vector<float> rowShift;
    for (std::size_t y = 0; y < kSize; ++y) {
        std::optional<float> shift;
        for (std::size_t x = 0; x < kSize; ++x) {
            if (offsets[at(x, y)] <= 0.0f) {
                continue;  // clipped at no electrons
            }
            const float own = offsets[at(x, y)] - base[at(x, y)];
            if (!shift) {
                shift = own;
                rowShift.push_back(own);
            }
            EXPECT_NEAR(own, *shift, 0.05f)
                << "one offset along row " << y;
        }
    }
    ASSERT_GT(rowShift.size(), 10u);
    const auto [lowShift, highShift] =
        std::minmax_element(rowShift.begin(), rowShift.end());
    EXPECT_GT(*highShift - *lowShift, 20.0f) << "rows differ";
}

TEST_F(RasterDelegateTest, ReadoutTapsOffsetInterleavedPixels) {
    RasterDelegate delegate(*context_);
    Scene scene = MeshScene();
    const auto raw = VarThrough(delegate, scene, "RawDN",
                                {{"qe", kNone}, {"adc", kNone}});
    const auto tapped = VarThrough(
        delegate, scene, "RawDN",
        {{"qe", kNone},
         {"adc", kNone},
         {"readoutTaps", {{"offset0", 7.0f}, {"gain1", 2.0f}}}});
    int checked = 0;
    for (std::size_t y = 0; y < kSize; ++y) {
        for (std::size_t x = 0; x < kSize; ++x) {
            const std::size_t i = (y * kSize + x) * 4 + 1;
            if (raw[i] > 2000.0f) {
                continue;  // doubled past the ADC's range
            }
            EXPECT_EQ(tapped[i],
                      x % 2 == 0 ? raw[i] + 7.0f : 2.0f * raw[i])
                << x << ", " << y;
            ++checked;
        }
    }
    EXPECT_GT(checked, 1000);
}

TEST_F(RasterDelegateTest, TdiSumsItsStagesAndSmearsWhenOutOfStep) {
    RasterDelegate delegate(*context_);
    Scene scene = MeshScene();
    ernie::Camera(scene, "/World/Camera")
        .SetExposureTime(1e-6);  // unsaturated
    const auto at = [](std::size_t x, std::size_t y) {
        return (y * kSize + x) * 4 + 1;
    };
    const auto line =
        VarThrough(delegate, scene, "Electrons", {{"qe", kNone}});
    ASSERT_LT(*std::max_element(line.begin(), line.end()), 2000.0f)
        << "four stages unsaturated";
    const auto inStep =
        VarThrough(delegate, scene, "Electrons",
                   {{"qe", kNone},
                    {"tdi", {{"stages", 4.0f}, {"drift", 0.0f}}}});
    const auto drifting =
        VarThrough(delegate, scene, "Electrons",
                   {{"qe", kNone},
                    {"tdi", {{"stages", 4.0f}, {"drift", 1.0f}}}});
    for (std::size_t y = 0; y + 3 < kSize; ++y) {
        for (std::size_t x = 0; x < kSize; ++x) {
            EXPECT_NEAR(inStep[at(x, y)], 4.0f * line[at(x, y)],
                        1e-3f * line[at(x, y)] + 1e-3f);
            const float smeared =
                line[at(x, y)] + line[at(x, y + 1)] +
                line[at(x, y + 2)] + line[at(x, y + 3)];
            EXPECT_NEAR(drifting[at(x, y)], smeared,
                        1e-3f * smeared + 1e-3f)
                << x << ", " << y;
        }
    }
}

TEST_F(RasterDelegateTest, DefectPixelsAreDeadOrHotAndStayPut) {
    RasterDelegate delegate(*context_);
    Scene scene = FlatScene();
    ernie::Camera(scene, "/World/Camera")
        .SetExposureTime(1e-6);  // unsaturated
    const auto base =
        VarThrough(delegate, scene, "Electrons", {{"qe", kNone}});
    const std::map<std::string, float> defects{
        {"dead", 0.02f}, {"hot", 0.02f}, {"hotRate", 1e8f}};
    const auto first =
        VarThrough(delegate, scene, "Electrons",
                   {{"qe", kNone}, {"defectPixels", defects}});
    const auto second =
        VarThrough(delegate, scene, "Electrons",
                   {{"qe", kNone}, {"defectPixels", defects}});
    int dead = 0;
    int hot = 0;
    for (std::size_t i = 1; i < base.size(); i += 4) {
        ASSERT_GT(base[i], 0.0f);
        if (first[i] == 0.0f) {
            ++dead;
        } else if (first[i] > base[i] + 40.0f) {
            ++hot;  // 50 to 150 electrons over the microsecond
        } else {
            EXPECT_EQ(first[i], base[i]);
        }
    }
    const int pixels = static_cast<int>(kSize * kSize);
    EXPECT_NEAR(dead, 0.02 * pixels, 0.01 * pixels);
    EXPECT_NEAR(hot, 0.02 * pixels, 0.01 * pixels);
    EXPECT_EQ(first, second) << "the same pixels every frame";
}

TEST_F(RasterDelegateTest,
       SmearAddsEachColumnsMeanForTheTransferTime) {
    RasterDelegate delegate(*context_);
    Scene scene = MeshScene();
    ernie::Camera(scene, "/World/Camera")
        .SetExposureTime(1e-6);  // unsaturated
    const auto at = [](std::size_t x, std::size_t y) {
        return (y * kSize + x) * 4 + 1;
    };
    const auto base =
        VarThrough(delegate, scene, "Electrons", {{"qe", kNone}});
    // A transfer as long as half the exposure: half each column's
    // mean on every pixel of it.
    const auto smeared = VarThrough(
        delegate, scene, "Electrons",
        {{"qe", kNone}, {"smear", {{"transferTime", 0.5e-6f}}}});
    for (std::size_t x = 0; x < kSize; ++x) {
        double column = 0.0;
        for (std::size_t y = 0; y < kSize; ++y) {
            column += base[at(x, y)];
        }
        const double added = 0.5 * column / kSize;
        for (std::size_t y = 0; y < kSize; ++y) {
            EXPECT_NEAR(smeared[at(x, y)], base[at(x, y)] + added,
                        1e-3 * (base[at(x, y)] + added) + 1e-3)
                << x << ", " << y;
        }
    }
}

TEST_F(RasterDelegateTest, DarkCurrentDoublesWithTemperature) {
    RasterDelegate delegate(*context_);
    Scene scene = FlatScene();
    ernie::Camera(scene, "/World/Camera").SetExposureTime(1e-6);
    const auto base =
        VarThrough(delegate, scene, "Electrons", {{"qe", kNone}});
    const auto darkAt = [&](float temperature) {
        return VarThrough(delegate, scene, "Electrons",
                          {{"qe", kNone},
                           {"darkCurrent",
                            {{"temperature", temperature},
                             {"referenceTemperature", 20.0f},
                             {"referenceRate", 1e7f},
                             {"doubling", 6.0f},
                             {"nonuniformity", 0.0f}}}});
    };
    const auto reference = darkAt(20.0f);
    const auto warm = darkAt(32.0f);  // two doublings
    for (std::size_t i = 1; i < base.size(); i += 4) {
        EXPECT_NEAR(reference[i] - base[i], 10.0f, 1e-2f)
            << "1e7 e/s over a microsecond";
        EXPECT_NEAR(warm[i] - base[i], 40.0f, 4e-2f);
    }
}

TEST_F(RasterDelegateTest, JpegKeepsQualityAndLosesDetailAsItFalls) {
    RasterDelegate delegate(*context_);
    Scene scene = MeshScene();
    ernie::Camera(scene, "/World/Camera").SetExposureTime(0.001);
    const auto plain =
        VarsThrough(delegate, scene, {"LdrColor"},
                    {{"cie", kNone}, {"exposure", kNone}})
            .at("LdrColor");
    const auto errorAt = [&](float quality) {
        const auto jpeg =
            VarsThrough(delegate, scene, {"LdrColor"},
                        {{"cie", kNone},
                         {"exposure", kNone},
                         {"jpeg", {{"quality", quality}}}})
                .at("LdrColor");
        double total = 0.0;
        double largest = 0.0;
        for (std::size_t i = 0; i < plain.size(); ++i) {
            if (i % 4 != 3) {
                const double e =
                    std::abs(double(jpeg[i]) - double(plain[i]));
                total += e;
                largest = std::max(largest, e);
            }
        }
        return std::pair(total / (plain.size() * 3 / 4), largest);
    };
    const auto [best, bestLargest] = errorAt(100.0f);
    const auto [good, goodLargest] = errorAt(75.0f);
    const auto [poor, poorLargest] = errorAt(5.0f);
    EXPECT_LE(bestLargest, 4.0)
        << "quality 100 is nearly lossless (DN of 255)";
    EXPECT_LT(best, good);
    EXPECT_LT(good, poor);
    EXPECT_GT(poor, 2.0) << "quality 5 blocks visibly";
}

namespace {

/// The NAC's nominal 12- to 8-bit table (Robinson et al. 2010,
/// Appendix A, code 0).
float NacCode0(float x) {
    if (x < 32.0f) {
        return std::floor(x / 2.0f);
    }
    if (x < 136.0f) {
        return std::floor(x / 4.0f + 8.0f);
    }
    if (x < 543.0f) {
        return std::floor(x / 8.0f + 25.0f);
    }
    if (x < 2207.0f) {
        return std::floor(x / 16.0f + 59.0f);
    }
    return std::floor(x / 32.0f + 128.0f);
}

}  // namespace

TEST_F(RasterDelegateTest,
       CompandTableEncodesTheNacsTwelveBitsInEight) {
    RasterDelegate delegate(*context_);
    Scene scene = MeshScene();
    std::set<float> seen;
    // Exposures spanning the 12-bit range, so every segment of the
    // table is crossed.
    for (const double exposure : {1e-7, 1e-6, 1e-5, 1e-4, 1e-3}) {
        ernie::Camera(scene, "/World/Camera")
            .SetExposureTime(exposure);
        const auto raw = VarThrough(delegate, scene, "RawDN",
                                    {{"qe", kNone}, {"adc", kNone}});
        const auto encoded = VarThrough(
            delegate, scene, "RawDN",
            {{"qe", kNone}, {"adc", kNone}, {"compandTable", kNone}});
        for (std::size_t i = 0; i < raw.size(); ++i) {
            if (i % 4 == 3) {
                continue;
            }
            EXPECT_EQ(encoded[i], NacCode0(raw[i]))
                << "12-bit " << raw[i];
            EXPECT_LE(encoded[i], 255.0f);
            seen.insert(raw[i]);
        }
    }
    EXPECT_GT(seen.size(), 15u) << "many 12-bit values";
    EXPECT_LT(*seen.begin(), 32.0f) << "in the first segment";
    EXPECT_GT(*seen.rbegin(), 2207.0f) << "into the last segment";
}
