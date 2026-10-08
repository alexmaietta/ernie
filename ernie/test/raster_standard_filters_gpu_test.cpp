#include "ernie/raster/delegate.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"
#include "support/delegate_scenes.h"
#include "support/vulkan_fixture.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <numeric>
#include <string>
#include <vector>

using ernie::Path;
using ernie::RasterDelegate;
using ernie::Scene;
using ernie::Tensor;
using namespace ernie::testing;
namespace tokens = ernie::tokens;

namespace {

using Steps = std::vector<std::pair<std::string, std::map<std::string, float>>>;
/// Float-array inputs by step index, then input name.
using Arrays = std::map<std::size_t, std::map<std::string, std::vector<float>>>;

/// Render vars `vars` (source names, such as "Electrons" or "aov:overflow") of MeshScene's camera
/// (without noise) through SensorFilter prims of `steps`, each with its own inputs and `arrays`,
/// as floats.
std::map<std::string, std::vector<float>> VarsThrough(RasterDelegate& delegate, Scene& scene,
                                                      const std::vector<std::string>& vars,
                                                      const Steps& steps,
                                                      const Arrays& arrays = {}) {
    scene.Set("/World/Camera", tokens::kSensorNoise, false);
    std::vector<Path> filters;
    for (std::size_t i = 0; i < steps.size(); ++i) {
        const Path path("/Render/Filters/F" + std::to_string(i));
        scene.Remove(path);
        scene.Define(path, tokens::kSensorFilter).Set(tokens::kFilterType, steps[i].first);
        for (const auto& [name, value] : steps[i].second) {
            scene.Set(path, "inputs:" + name, value);
        }
        if (const auto found = arrays.find(i); found != arrays.end()) {
            for (const auto& [name, values] : found->second) {
                scene.Set(path, "inputs:" + name, values);
            }
        }
        filters.push_back(path);
    }
    std::vector<Path> varPaths;
    for (std::size_t i = 0; i < vars.size(); ++i) {
        const Path varPath("/Render/Vars/V" + std::to_string(i));
        scene.Define(varPath, tokens::kRenderVar).Set(tokens::kSourceName, vars[i]);
        varPaths.push_back(varPath);
    }
    scene.Remove("/Render/Many");
    scene.Define("/Render/Many", tokens::kRenderProduct)
        .Set(tokens::kCameraRel, Path("/World/Camera"))
        .Set(tokens::kResolution, glm::ivec2(kSize, kSize))
        .Set(tokens::kOrderedVars, varPaths)
        .Set(tokens::kFilters, filters);
    const Path product[] = {"/Render/Many"};
    auto outputs = delegate.Step(scene, product);
    std::map<std::string, std::vector<float>> values;
    for (const std::string& var : vars) {
        const Tensor tensor =
            outputs.at("/Render/Many").vars.at(var).Map(ernie::MapDevice::Cpu).tensor;
        std::vector<float>& out = values[var];
        out.resize(static_cast<std::size_t>(tensor.Elements()));
        for (std::size_t i = 0; i < out.size(); ++i) {
            out[i] = tensor.Type() == ernie::DType::UInt16 ? static_cast<float>(
                                                                 tensor.As<std::uint16_t>()[i])
                     : tensor.Type() == ernie::DType::UInt8
                         ? static_cast<float>(tensor.As<std::uint8_t>()[i])
                         : tensor.As<float>()[i];
        }
    }
    return values;
}

/// Render var `var` of MeshScene's camera (without noise) through SensorFilter prims of `types`,
/// each step with its own inputs, as floats.
std::vector<float> VarThrough(RasterDelegate& delegate, Scene& scene, std::string_view var,
                              const std::vector<std::pair<std::string,
                                                          std::map<std::string, float>>>& steps) {
    scene.Set("/World/Camera", tokens::kSensorNoise, false);
    std::vector<Path> filters;
    for (std::size_t i = 0; i < steps.size(); ++i) {
        const Path path("/Render/Filters/F" + std::to_string(i));
        scene.Remove(path);
        scene.Define(path, tokens::kSensorFilter).Set(tokens::kFilterType, steps[i].first);
        for (const auto& [name, value] : steps[i].second) {
            scene.Set(path, "inputs:" + name, value);
        }
        filters.push_back(path);
    }
    const Path varPath("/Render/Vars/" + std::string(var));
    scene.Define(varPath, tokens::kRenderVar).Set(tokens::kSourceName, std::string(var));
    scene.Define("/Render/Out", tokens::kRenderProduct)
        .Set(tokens::kCameraRel, Path("/World/Camera"))
        .Set(tokens::kResolution, glm::ivec2(kSize, kSize))
        .Set(tokens::kOrderedVars, std::vector<Path>{varPath})
        .Set(tokens::kFilters, filters);
    const Path product[] = {"/Render/Out"};
    const Tensor tensor = delegate.Step(scene, product)
                              .at("/Render/Out")
                              .vars.at(std::string(var))
                              .Map(ernie::MapDevice::Cpu)
                              .tensor;
    std::vector<float> values(static_cast<std::size_t>(tensor.Elements()));
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = tensor.Type() == ernie::DType::UInt16
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

/// Total variation of a channel: the sum of differences between horizontal neighbours.
double Variation(const std::vector<float>& values, std::size_t channel) {
    double total = 0.0;
    for (std::size_t y = 0; y < kSize; ++y) {
        for (std::size_t x = 1; x < kSize; ++x) {
            total += static_cast<double>(std::abs(values[(y * kSize + x) * 4 + channel] -
                                                  values[(y * kSize + x - 1) * 4 + channel]));
        }
    }
    return total;
}

const std::map<std::string, float> kNone;

}  // namespace

TEST_F(RasterDelegateTest, TheStandardSensorFiltersAreRegistered) {
    RasterDelegate delegate(*context_);
    for (const char* id :
         {"psf", "defocus", "bayer", "demosaic", "whiteBalance", "colorMatrix", "compand"}) {
        EXPECT_NE(delegate.SensorFilters().Find(id), nullptr) << id;
    }
}

TEST_F(RasterDelegateTest, ThePointSpreadFunctionSpreadsLightAndKeepsIt) {
    RasterDelegate delegate(*context_);
    Scene scene = MeshScene();
    const auto base = VarThrough(delegate, scene, "HdrColor", {{"cie", kNone}});
    const auto blurred =
        VarThrough(delegate, scene, "HdrColor", {{"psf", {{"blur", 2.0f}}}, {"cie", kNone}});
    EXPECT_NEAR(Sum(blurred, 0), Sum(base, 0), 0.01 * Sum(base, 0)) << "light is kept";
    EXPECT_LT(Variation(blurred, 0), 0.8 * Variation(base, 0)) << "edges soften";
    // Diffraction alone at f/1 spreads well under a pixel at these pixels: nearly untouched.
    const auto diffraction = VarThrough(delegate, scene, "HdrColor", {{"psf", kNone}, {"cie", kNone}});
    EXPECT_NEAR(Sum(diffraction, 0), Sum(base, 0), 0.01 * Sum(base, 0));
}

TEST_F(RasterDelegateTest, DefocusBlursOutOfFocusPointsAndPinholesStaySharp) {
    RasterDelegate delegate(*context_);
    Scene scene = MeshScene();
    const auto base = VarThrough(delegate, scene, "HdrColor", {{"cie", kNone}});
    EXPECT_EQ(VarThrough(delegate, scene, "HdrColor", {{"defocus", kNone}, {"cie", kNone}}), base)
        << "a pinhole (fStop 0) is untouched";
    // A fast 18 mm lens focused 10 cm away; the scene is 60 m away, so a point there blurs into
    // a disc about 9 pixels across (these pixels are 0.3 mm).
    scene.Set("/World/Camera", tokens::kFStop, 1.4);
    scene.Set("/World/Camera", tokens::kFocusDistance, 0.1);
    const auto blurred = VarThrough(delegate, scene, "HdrColor",
                                    {{"defocus", {{"maxRadius", 4.0f}}}, {"cie", kNone}});
    EXPECT_LT(Variation(blurred, 0), 0.8 * Variation(base, 0));
    EXPECT_NEAR(Sum(blurred, 0), Sum(base, 0), 0.05 * Sum(base, 0)) << "light is kept";
}

TEST_F(RasterDelegateTest, WhiteBalanceAndColourMatricesScaleAndMixChannels) {
    RasterDelegate delegate(*context_);
    Scene scene = MeshScene();
    const auto base = VarThrough(delegate, scene, "HdrColor", {{"cie", kNone}});
    const auto balanced = VarThrough(delegate, scene, "HdrColor",
                                     {{"cie", kNone}, {"whiteBalance", {{"red", 2.0f}}}});
    const auto swapped = VarThrough(
        delegate, scene, "HdrColor",
        {{"cie", kNone},
         {"colorMatrix",
          {{"m00", 0.0f}, {"m02", 1.0f}, {"m20", 1.0f}, {"m22", 0.0f}}}});
    for (std::size_t i = 0; i < base.size(); i += 4) {
        EXPECT_EQ(balanced[i], 2.0f * base[i]);
        EXPECT_EQ(balanced[i + 1], base[i + 1]);
        EXPECT_EQ(swapped[i], base[i + 2]);
        EXPECT_EQ(swapped[i + 2], base[i]);
    }
}

TEST_F(RasterDelegateTest, BayerMosaicsDemosaicRestoresAndCompandingCompresses) {
    RasterDelegate delegate(*context_);
    Scene scene = MeshScene();
    // Short enough not to saturate.
    scene.Set("/World/Camera", tokens::kExposureTime, 0.001);
    const auto raw = VarThrough(delegate, scene, "RawDN", {{"qe", kNone}, {"adc", kNone}});
    const auto mosaic = VarThrough(delegate, scene, "RawDN",
                                   {{"qe", kNone}, {"bayer", kNone}, {"adc", kNone}});
    // RGGB: each pixel's other channels hold the black level alone, its own the full value.
    const ernie::Camera camera = ernie::CameraOf(scene, "/World/Camera", kSize, kSize);
    const float black = static_cast<float>(camera.sensor.blackLevelDn);
    for (std::size_t y = 0; y < kSize; ++y) {
        for (std::size_t x = 0; x < kSize; ++x) {
            const std::size_t own = (y % 2 == 0) ? (x % 2 == 0 ? 0 : 1) : (x % 2 == 0 ? 1 : 2);
            for (std::size_t c = 0; c < 3; ++c) {
                const float value = mosaic[(y * kSize + x) * 4 + c];
                if (c == own) {
                    EXPECT_EQ(value, raw[(y * kSize + x) * 4 + c]);
                } else {
                    EXPECT_EQ(value, black) << x << ", " << y << " channel " << c;
                }
            }
        }
    }
    // Demosaicing gives back the full colour where it is uniform: close over the red quad, which
    // then fills the image, both methods restore every channel inside a two-pixel border.
    Scene close = MeshScene();
    close.Set("/World/Camera", tokens::kTransform,
              ernie::LookAtTransform({-12.0, 0.0, 3.0}, {-12.0, 0.0, 0.0}, {0.0, 1.0, 0.0}));
    close.Set("/World/Camera", tokens::kExposureTime, 0.001);
    const auto full = VarThrough(delegate, close, "RawDN", {{"qe", kNone}, {"adc", kNone}});
    ASSERT_GT(full[(kSize / 2 * kSize + kSize / 2) * 4], black + 10.0f) << "the quad is lit";
    for (const float method : {0.0f, 1.0f}) {
        const auto restored = VarThrough(
            delegate, close, "RawDN",
            {{"qe", kNone}, {"bayer", kNone}, {"adc", kNone}, {"demosaic", {{"method", method}}}});
        for (std::size_t y = 2; y < kSize - 2; ++y) {
            for (std::size_t x = 2; x < kSize - 2; ++x) {
                for (std::size_t c = 0; c < 3; ++c) {
                    const std::size_t i = (y * kSize + x) * 4 + c;
                    ASSERT_NEAR(restored[i], full[i], 1.0f)
                        << "method " << method << " at " << x << ", " << y << " channel " << c;
                }
            }
        }
    }
    // Companding 12 bits to 8: full scale to 255, and the square root in between.
    const auto companded = VarThrough(delegate, scene, "RawDN",
                                      {{"qe", kNone}, {"adc", kNone},
                                       {"compand", {{"inputBits", 12.0f}, {"outputBits", 8.0f}}}});
    for (std::size_t i = 0; i < raw.size(); ++i) {
        if (i % 4 == 3) {
            continue;
        }
        const float expected =
            std::floor(255.0f * std::sqrt(std::min(raw[i] / 4095.0f, 1.0f)) + 0.5f);
        EXPECT_NEAR(companded[i], expected, 1.0f);
    }
}

TEST_F(RasterDelegateTest, CurveParametersTakeArraysOfAnyLength) {
    RasterDelegate delegate(*context_);
    Scene scene = MeshScene();
    const auto base = VarThrough(delegate, scene, "HdrColor", {{"cie", kNone}});
    // A single number fills the curve: half the light in every band.
    EXPECT_EQ(VarThrough(delegate, scene, "HdrColor",
                         {{"spectralTransmission", {{"curve", 0.5f}}}, {"cie", kNone}}),
              [&] {
                  std::vector<float> half = base;
                  for (std::size_t i = 0; i < half.size(); ++i) {
                      half[i] *= i % 4 == 3 ? 1.0f : 0.5f;
                  }
                  return half;
              }());

    // An array, resampled: transmission falling to nothing above 570 nm (ten samples over 360 to
    // 830 nm) dims white rocks' red and keeps their blue.
    scene.Set("/World/Rocks/Rock", tokens::kDisplayColor, glm::vec3(1.0f));
    const auto white = VarThrough(delegate, scene, "HdrColor", {{"cie", kNone}});
    const auto cut =
        VarsThrough(delegate, scene, {"HdrColor"}, {{"spectralTransmission", kNone}, {"cie", kNone}},
                    {{0, {{"curve", {1, 1, 1, 1, 1, 0, 0, 0, 0, 0}}}}})
            .at("HdrColor");
    EXPECT_LT(Sum(cut, 0), 0.3 * Sum(white, 0)) << "red passes little";
    EXPECT_GT(Sum(cut, 2), 0.7 * Sum(white, 2)) << "blue mostly passes";

    // An inverting tone curve, as two samples resampled to 33.
    const auto ldr =
        VarsThrough(delegate, scene, {"LdrColor"}, {{"cie", kNone}, {"exposure", kNone}})
            .at("LdrColor");
    const auto flipped = VarsThrough(delegate, scene, {"LdrColor"},
                                     {{"cie", kNone}, {"exposure", kNone}, {"toneCurve", kNone}},
                                     {{2, {{"curve", {1.0f, 0.0f}}}}})
                             .at("LdrColor");
    for (std::size_t i = 0; i < ldr.size(); ++i) {
        if (i % 4 != 3) {
            ASSERT_NEAR(flipped[i], 255.0f - ldr[i], 1.0f) << i;
        }
    }
}

TEST_F(RasterDelegateTest, BloomingSpillsSaturatedChargeAlongColumns) {
    // MeshScene's red quad at the default 1 s exposure holds far more than the full well.
    RasterDelegate delegate(*context_);
    Scene scene = MeshScene();
    const float well = static_cast<float>(
        ernie::CameraOf(scene, "/World/Camera", kSize, kSize).sensor.fullWellElectrons);
    const auto clipped = VarThrough(delegate, scene, "Electrons", {{"qe", kNone}});
    const auto bloomed =
        VarsThrough(delegate, scene, {"Electrons", "aov:overflow"},
                    {{"qe", kNone}, {"blooming", {{"maxLength", 64.0f}}}});
    const auto& electrons = bloomed.at("Electrons");
    const auto& overflow = bloomed.at("aov:overflow");
    const auto at = [](const std::vector<float>& image, std::size_t x, std::size_t y) {
        return image[(y * kSize + x) * 4];  // red
    };
    // The quad's brightest column at the middle row, and a column of background beside it.
    std::size_t column = 0;
    for (std::size_t x = 0; x < kSize; ++x) {
        column = at(clipped, x, kSize / 2) > at(clipped, column, kSize / 2) ? x : column;
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
    EXPECT_GT(streaked, before + 10) << "a vertical streak beyond the quad";
    EXPECT_LT(at(electrons, dark, kSize / 2), 0.01f * well) << "not sideways";
    EXPECT_GT(at(overflow, column, kSize / 2), 0.0f) << "the overflow AOV";
    EXPECT_EQ(at(overflow, column, 0), 0.0f) << "nothing overflowed above the quad";

    // An overflow drain takes it all: no streak.
    const auto drained =
        VarThrough(delegate, scene, "Electrons", {{"qe", kNone}, {"blooming", {{"antiBlooming", 1.0f}}}});
    EXPECT_EQ(drained, clipped);
}

TEST_F(RasterDelegateTest, AovsRunTheirFiltersAndNameTheirOutputs) {
    RasterDelegate delegate(*context_);
    Scene scene = MeshScene();
    scene.Set("/World/Camera", tokens::kExposureTime, 0.001);
    const Steps chain = {{"qe", kNone}, {"bayer", kNone}, {"adc", kNone}, {"demosaic", kNone}};
    // RawDN through bayer and adc alone is the mosaic itself.
    const auto mosaic =
        VarThrough(delegate, scene, "RawDN", {{"qe", kNone}, {"bayer", kNone}, {"adc", kNone}});
    // Asking for the AOV alone runs the chain up to demosaic, which writes the mosaic it read.
    const auto aov = VarsThrough(delegate, scene, {"aov:mosaic"}, chain).at("aov:mosaic");
    ASSERT_EQ(aov.size(), mosaic.size());
    for (std::size_t y = 0; y < kSize; ++y) {
        for (std::size_t x = 0; x < kSize; ++x) {
            const std::size_t own = (y % 2 == 0) ? (x % 2 == 0 ? 0 : 1) : (x % 2 == 0 ? 1 : 2);
            const std::size_t i = (y * kSize + x) * 4 + own;
            ASSERT_EQ(aov[i], mosaic[i]) << x << ", " << y;
        }
    }
    // Both at once: the demosaiced RawDN and the mosaic.
    const auto both = VarsThrough(delegate, scene, {"RawDN", "aov:mosaic"}, chain);
    EXPECT_EQ(both.at("aov:mosaic"), aov);
    EXPECT_NE(both.at("RawDN"), mosaic);
    EXPECT_THROW(VarsThrough(delegate, scene, {"aov:missing"}, chain), std::invalid_argument);
}
