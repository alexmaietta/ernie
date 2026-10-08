#include "ernie/device/offscreen_target.h"
#include "ernie/geometry/primitives.h"
#include "ernie/raster/delegate.h"
#include "ernie/scene/orbit_controller.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"
#include "ernie/session/renderer.h"
#include "support/delegate_scenes.h"
#include "support/test_raster.h"
#include "support/vulkan_fixture.h"

#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

using ernie::DeviceKind;
using ernie::DType;
using ernie::OrbitController;
using ernie::Path;
using ernie::RasterDelegate;
using ernie::Scene;
using ernie::Tensor;
using namespace ernie::testing;
namespace tokens = ernie::tokens;

namespace {

/// Custom filters: a gain on linear colour (`scale`, 2 by default) and a flat transmission on
/// every band (`transmission`, 1 by default).
ernie::SensorFilterModel GainFilter() {
    ernie::SensorFilterModel model;
    model.id = "gain";
    model.kind = ernie::FilterKind::Pixel;
    model.domain = ernie::FilterDomain::Rgb;
    model.params = {{"scale", 2.0f, ""}};
    model.typeName = "Gain";
    model.slang = R"(import sensor;
struct Gain : IPixelFilter {
    struct Params { float scale; }
    static void apply(inout SensorPixel value, PixelContext pixel, Params params) {
        value.rgb *= params.scale;
    }
})";
    return model;
}

ernie::SensorFilterModel TransmissionFilter() {
    ernie::SensorFilterModel model;
    model.id = "transmission";
    model.kind = ernie::FilterKind::Band;
    model.domain = ernie::FilterDomain::Spectrum;
    model.params = {{"transmission", 1.0f, ""}};
    model.typeName = "Transmission";
    model.slang = R"(import sensor;
struct Transmission : IBandFilter {
    struct Params { float transmission; }
    static float apply(float radiance, uint band, PixelContext pixel, Params params) {
        return radiance * params.transmission;
    }
})";
    return model;
}

/// HdrColor of MeshScene's camera through SensorFilter prims of `types`, each with `inputs`.
std::vector<float> HdrThrough(RasterDelegate& delegate, Scene& scene,
                              const std::vector<std::string>& types,
                              const std::map<std::string, float>& inputs = {}) {
    std::vector<Path> filters;
    for (std::size_t i = 0; i < types.size(); ++i) {
        const Path path("/Render/Filters/F" + std::to_string(i));
        scene.Remove(path);  // a fresh prim: no inputs left from an earlier chain
        scene.Define(path, tokens::kSensorFilter).Set(tokens::kFilterType, types[i]);
        for (const auto& [name, value] : inputs) {
            if (types[i] != "cie") {
                scene.Set(path, "inputs:" + name, value);
            }
        }
        filters.push_back(path);
    }
    scene.Define("/Render/Vars/Hdr", tokens::kRenderVar).Set(tokens::kSourceName, "HdrColor");
    scene.Define("/Render/Hdr", tokens::kRenderProduct)
        .Set(tokens::kCameraRel, Path("/World/Camera"))
        .Set(tokens::kResolution, glm::ivec2(kSize, kSize))
        .Set(tokens::kOrderedVars, std::vector<Path>{"/Render/Vars/Hdr"})
        .Set(tokens::kFilters, filters);
    const Path product[] = {"/Render/Hdr"};
    const Tensor hdr = delegate.Step(scene, product)
                           .at("/Render/Hdr")
                           .vars.at("HdrColor")
                           .Map(ernie::MapDevice::Cpu)
                           .tensor;
    return {hdr.As<float>(), hdr.As<float>() + hdr.Elements()};
}

/// `values` with every colour channel (not alpha) times `factor`.
std::vector<float> Scaled(std::vector<float> values, float factor) {
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i % 4 != 3) {
            values[i] *= factor;
        }
    }
    return values;
}

}  // namespace

TEST_F(RasterDelegateTest, CustomSensorFiltersRunInTheirChains) {
    RasterDelegate delegate(*context_);
    delegate.AddSensorFilter(GainFilter());
    delegate.AddSensorFilter(TransmissionFilter());
    Scene scene = MeshScene();
    const std::vector<float> base = HdrThrough(delegate, scene, {"cie"});
    ASSERT_GT(*std::max_element(base.begin(), base.end()), 0.0f);

    EXPECT_EQ(HdrThrough(delegate, scene, {"cie", "gain"}), Scaled(base, 2.0f))
        << "the default scale";
    EXPECT_EQ(HdrThrough(delegate, scene, {"cie", "gain"}, {{"scale", 3.0f}}), Scaled(base, 3.0f))
        << "inputs:scale";
    // A power of two scales every band's product and their sum exactly.
    EXPECT_EQ(HdrThrough(delegate, scene, {"transmission", "cie"}, {{"transmission", 0.25f}}),
              Scaled(base, 0.25f));
    EXPECT_THROW(HdrThrough(delegate, scene, {"gain", "cie"}), std::invalid_argument)
        << "gain before there is colour";
}

TEST_F(RasterDelegateTest, CustomSensorFiltersThatFailToCompileAreNotRegistered) {
    RasterDelegate delegate(*context_);
    ernie::SensorFilterModel broken = GainFilter();
    broken.slang = R"(import sensor;
struct Gain : IPixelFilter {
    struct Params { float scale; }
    static void apply(inout SensorPixel value, PixelContext pixel, Params params) {
        value.rgb *= undefinedThing;
    }
})";
    try {
        delegate.AddSensorFilter(broken);
        ADD_FAILURE() << "a broken filter registered";
    } catch (const std::invalid_argument& error) {
        EXPECT_NE(std::string(error.what()).find("undefinedThing"), std::string::npos)
            << "the compiler's diagnostics: " << error.what();
    }
    EXPECT_EQ(delegate.SensorFilters().Find("gain"), nullptr);
    Scene scene = MeshScene();
    EXPECT_THROW(HdrThrough(delegate, scene, {"cie", "gain"}), std::invalid_argument)
        << "an unknown filter type";
    delegate.AddSensorFilter(GainFilter());
    EXPECT_NO_THROW(HdrThrough(delegate, scene, {"cie", "gain"})) << "the fixed filter registers";
}

namespace {

/// Spatial filters: linear colour from the pixel `dx` to the left, and each band from the pixel
/// itself.
ernie::SensorFilterModel ShiftFilter() {
    ernie::SensorFilterModel model;
    model.id = "shift";
    model.kind = ernie::FilterKind::Spatial;
    model.domain = ernie::FilterDomain::Rgb;
    model.params = {{"dx", 1.0f, ""}};
    model.typeName = "Shift";
    model.slang = R"(import sensor;
struct Shift : ISpatialPixelFilter {
    struct Params { float dx; }
    static void apply(inout SensorPixel value, PixelContext pixel, SpatialImage input,
                      Params params) {
        value.rgb = input.values(pixel, int2(pixel.id.xy) - int2(int(params.dx), 0)).xyz;
    }
})";
    return model;
}

ernie::SensorFilterModel CopySpectrumFilter() {
    ernie::SensorFilterModel model;
    model.id = "copySpectrum";
    model.kind = ernie::FilterKind::Spatial;
    model.domain = ernie::FilterDomain::Spectrum;
    model.typeName = "CopySpectrum";
    model.slang = R"(import sensor;
struct CopySpectrum : ISpatialBandFilter {
    struct Params {}
    static float band(uint band, PixelContext pixel, SpatialImage input, Params params) {
        return input.band(pixel, int2(pixel.id.xy), band);
    }
})";
    return model;
}

/// `hdr` (kSize x kSize RGBA rows) with each pixel taken from `dx` to its left, clamped.
std::vector<float> ShiftedRight(const std::vector<float>& hdr, int dx) {
    std::vector<float> shifted(hdr.size());
    for (int y = 0; y < static_cast<int>(kSize); ++y) {
        for (int x = 0; x < static_cast<int>(kSize); ++x) {
            const int from = std::max(x - dx, 0);
            for (int c = 0; c < 4; ++c) {
                shifted[(static_cast<std::size_t>(y) * kSize + static_cast<std::size_t>(x)) * 4 +
                        static_cast<std::size_t>(c)] =
                    hdr[(static_cast<std::size_t>(y) * kSize + static_cast<std::size_t>(from)) * 4 +
                        static_cast<std::size_t>(c)];
            }
        }
    }
    return shifted;
}

}  // namespace

TEST_F(RasterDelegateTest, SpatialSensorFiltersReadThePixelsAroundTheirOwn) {
    RasterDelegate delegate(*context_);
    delegate.AddSensorFilter(ShiftFilter());
    delegate.AddSensorFilter(CopySpectrumFilter());
    Scene scene = MeshScene();
    const std::vector<float> base = HdrThrough(delegate, scene, {"cie"});

    EXPECT_EQ(HdrThrough(delegate, scene, {"cie", "shift"}), ShiftedRight(base, 1));
    EXPECT_EQ(HdrThrough(delegate, scene, {"cie", "shift"}, {{"dx", 3.0f}}), ShiftedRight(base, 3));
    EXPECT_EQ(HdrThrough(delegate, scene, {"cie", "shift", "shift"}), ShiftedRight(base, 2))
        << "two in a row: three kernels, the images alternating";
    EXPECT_EQ(HdrThrough(delegate, scene, {"copySpectrum", "cie"}), base)
        << "the spectrum passes between kernels exactly";
}
