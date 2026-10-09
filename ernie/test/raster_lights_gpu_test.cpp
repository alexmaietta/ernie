#include "ernie/device/offscreen_target.h"
#include "ernie/geometry/primitives.h"
#include "ernie/raster/delegate.h"
#include "ernie/scene/orbit_controller.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"
#include "ernie/session/renderer.h"
#include "ernie/session/resolve.h"
#include "ernie/util/astronomy.h"
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

/// FlatScene lit only by a point lamp `height` above the origin pointing down, with a radiant flux
/// of 1000 W over a flat spectrum and a cone of `coneAngle` degrees (180: unshaped).
Scene LampScene(double height, float coneAngle = 180.0f) {
    Scene scene = FlatScene();
    scene.Remove("/World/Sun");
    scene.Define("/World/Lamp", tokens::kSphereLight)
        .Set(tokens::kTransform,
             ernie::LookAtTransform({0.0, 0.0, height}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}))
        .Set(tokens::kTreatAsPoint, true)
        .Set(tokens::kLightPower, 1000.0)
        .Set(tokens::kShapingConeAngle, coneAngle);
    return scene;
}

float RadianceAt(RasterDelegate& delegate, const Scene& scene, float x, float y) {
    ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
    const Tensor spectral =
        outputs.at("/Render/Product").vars.at("SpectralRadiance").Map(ernie::MapDevice::Cpu).tensor;
    const auto [row, col] = PixelOf(scene, x, y);
    return static_cast<const float*>(
        spectral.Data())[8 * kVarSize * kVarSize + row * kVarSize + col];
}

}  // namespace

TEST_F(RasterDelegateTest, LampsLightByTheInverseSquareLaw) {
    RasterDelegate delegate(*context_);
    const Scene near = LampScene(10.0);
    const float intensity = static_cast<float>(1000.0 / (4.0 * M_PI) / 470.0);
    const float x = 3.0f;
    const float distance2 = 100.0f + x * x;
    const float expected =
        intensity * (10.0f / std::sqrt(distance2)) / distance2 * 0.5f / static_cast<float>(M_PI);
    EXPECT_NEAR(RadianceAt(delegate, near, x, 0.0f), expected, 0.04f * expected)
        << "L = I cos(i) / d^2 * albedo / pi";
    EXPECT_FALSE(delegate.Prims().lights.empty());

    const float twice = RadianceAt(delegate, LampScene(20.0), 0.0f, 0.0f);
    const float once = RadianceAt(delegate, LampScene(10.0), 0.0f, 0.0f);
    EXPECT_NEAR(once / twice, 4.0f, 0.1f);

    const Scene spot = LampScene(10.0, 10.0f);
    EXPECT_NEAR(RadianceAt(delegate, spot, 0.0f, 0.0f), once, 0.02f * once) << "inside the cone";
    EXPECT_EQ(RadianceAt(delegate, spot, 6.0f, 0.0f), 0.0f) << "outside a 10 degree cone";
}

TEST_F(RasterDelegateTest, LampsCastShadowsFromSpotAndCubeMaps) {
    RasterDelegate delegate(*context_);
    for (const float cone : {60.0f, 180.0f}) {
        Scene scene = LampScene(10.0, cone);
        const float lit = RadianceAt(delegate, scene, 3.0f, 0.0f);
        DefineQuad(scene, "/World/Blocker");
        scene.Set("/World/Blocker", tokens::kTransform,
                  glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 0.0, 5.0)));
        EXPECT_LT(RadianceAt(delegate, scene, 3.0f, 0.0f), 0.1f * lit)
            << "behind the blocker, cone " << cone;
        EXPECT_NEAR(RadianceAt(delegate, scene, 6.0f, 6.0f),
                    RadianceAt(delegate, LampScene(10.0, cone), 6.0f, 6.0f), 0.05f * lit)
            << "beside the shadow, cone " << cone;
        scene.Set("/World/Lamp", tokens::kShadowEnable, false);
        EXPECT_NEAR(RadianceAt(delegate, scene, 3.0f, 0.0f), lit, 0.02f * lit);
    }
    EXPECT_EQ(delegate.Renderer().LightShadowViews().size(), 0u) << "shadows off";
}

TEST_F(RasterDelegateTest, OtherDistantLightsAddToTheSun) {
    Scene scene = FlatScene();
    scene.Define("/World/Zenith", tokens::kDistantLight)
        .Set(tokens::kTransform, ernie::DistantLightTransform({0.0, 0.0, 1.0}))
        .Set(tokens::kIntensity, 0.5f);
    RasterDelegate delegate(*context_);
    ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
    const auto vars = MapVars(outputs);
    const std::size_t c = kVarSize / 2;
    EXPECT_NEAR(At<float>(vars.at("HdrColor"), c, c, 1), 0.75f, 0.03f)
        << "0.5 cd/m^2 from the sun and 0.25 from Earthshine";
    EXPECT_EQ(delegate.Renderer().LightShadowViews().size(), 1u);
}

TEST_F(RasterDelegateTest, TheSunsDiskWidensPenumbrae) {
    Scene scene = FlatScene();
    scene.Set("/World/Sun", tokens::kShadowEnable, true);
    scene.Set("/World/Sun", tokens::kTransform, ernie::DistantLightTransform({0.3, 0.0, 1.0}));
    DefineQuad(scene, "/World/Blocker", 4.0f);
    scene.Set("/World/Blocker", tokens::kTransform,
              glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 0.0, 30.0)));
    RasterDelegate delegate(*context_);
    const auto partial = [&](float angle) {
        scene.Set("/World/Sun", tokens::kAngle, angle);
        ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
        const Tensor hdr =
            outputs.at("/Render/Product").vars.at("HdrColor").Map(ernie::MapDevice::Cpu).tensor;
        const Tensor ids =
            outputs.at("/Render/Product").vars.at("PrimId").Map(ernie::MapDevice::Cpu).tensor;
        const std::uint32_t ground = At<std::uint32_t>(ids, 0, 0);
        float lit = 0.0f;
        for (std::size_t row = 0; row < kVarSize; ++row) {
            for (std::size_t col = 0; col < kVarSize; ++col) {
                lit = std::max(lit, At<float>(hdr, row, col, 1));
            }
        }
        int count = 0;
        int shadowed = 0;
        for (std::size_t row = 0; row < kVarSize; ++row) {
            for (std::size_t col = 0; col < kVarSize; ++col) {
                if (At<std::uint32_t>(ids, row, col) != ground) {
                    continue;
                }
                const float value = At<float>(hdr, row, col, 1) / lit;
                count += value > 0.1f && value < 0.9f ? 1 : 0;
                shadowed += value <= 0.1f ? 1 : 0;
            }
        }
        return std::pair(count, shadowed);
    };
    const auto [hard, umbra] = partial(0.0f);
    EXPECT_GT(umbra, 20) << "the blocker's shadow is in view";
    const auto [soft, smaller] = partial(4.0f);
    EXPECT_GT(soft, hard + 10) << "a wider disk widens the penumbra";
    EXPECT_LT(smaller, umbra);
}

namespace {

/// FlatScene lit only by an area light of `type` 10 m above the origin facing down, with a
/// luminance of 1000 cd/m^2.
Scene AreaLightScene(std::string_view type) {
    Scene scene = FlatScene();
    scene.Remove("/World/Sun");
    scene.Define("/World/Panel", type)
        .Set(tokens::kTransform,
             ernie::LookAtTransform({0.0, 0.0, 10.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}))
        .Set(tokens::kIntensity, 1000.0)
        .Set(tokens::kShadowEnable, false);
    return scene;
}

/// Spectral radiance of a flat 1 cd/m^2 white emitter, per band.
double UnitLuminanceRadiance() {
    double sum = 0.0;
    for (double w = 360.5; w < 830.0; w += 1.0) {
        sum += ernie::CieXyz(w).y;
    }
    return 1.0 / (ernie::kLuminousEfficacy * sum);
}

/// Form factor from a point to a parallel rectangle `a` x `b` whose corner lies `h` above it.
double CornerFormFactor(double a, double b, double h) {
    const double x = a / h;
    const double y = b / h;
    const double sx = std::sqrt(1.0 + x * x);
    const double sy = std::sqrt(1.0 + y * y);
    return (x / sx * std::atan(y / sx) + y / sy * std::atan(x / sy)) / (2.0 * M_PI);
}

}  // namespace

TEST_F(RasterDelegateTest, RectAndDiskLightsMatchTheirFormFactors) {
    RasterDelegate delegate(*context_);
    const double radiance = 1000.0 * UnitLuminanceRadiance();

    Scene rect = AreaLightScene(tokens::kRectLight);
    rect.Set("/World/Panel", tokens::kWidth, 8.0);
    rect.Set("/World/Panel", tokens::kHeight, 6.0);
    const double rectExpected =
        0.5 / M_PI * radiance * M_PI * 4.0 * CornerFormFactor(4.0, 3.0, 10.0);
    EXPECT_NEAR(RadianceAt(delegate, rect, 0.0f, 0.0f), rectExpected, 0.03 * rectExpected)
        << "albedo / pi times E = pi L F";

    Scene disk = AreaLightScene(tokens::kDiskLight);
    disk.Set("/World/Panel", tokens::kRadius, 5.0);
    const double diskExpected = 0.5 * radiance * 25.0 / (25.0 + 100.0);
    EXPECT_NEAR(RadianceAt(delegate, disk, 0.0f, 0.0f), diskExpected, 0.03 * diskExpected)
        << "E = pi L r^2 / (r^2 + h^2) on the axis";

    disk.Set("/World/Panel", tokens::kTransform,
             ernie::LookAtTransform({0.0, 0.0, 10.0}, {0.0, 0.0, 20.0}, {0.0, 1.0, 0.0}));
    EXPECT_EQ(RadianceAt(delegate, disk, 0.0f, 0.0f), 0.0f) << "facing away";
    disk.Set("/World/Panel", tokens::kTransform,
             ernie::LookAtTransform({0.0, 0.0, 0.5}, {10.0, 0.0, 0.5}, {0.0, 0.0, 1.0}));
    const float grazing = RadianceAt(delegate, disk, 3.0f, 0.0f);
    EXPECT_GT(grazing, 0.0f) << "half the disk is above the ground's horizon";
    EXPECT_LT(grazing, static_cast<float>(diskExpected));
}

TEST_F(RasterDelegateTest, AreaLightsCastShadows) {
    RasterDelegate delegate(*context_);
    Scene scene = AreaLightScene(tokens::kRectLight);
    scene.Set("/World/Panel", tokens::kShadowEnable, true);
    const float lit = RadianceAt(delegate, scene, 3.0f, 0.0f);
    DefineQuad(scene, "/World/Blocker");
    scene.Set("/World/Blocker", tokens::kTransform,
              glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 0.0, 5.0)));
    EXPECT_LT(RadianceAt(delegate, scene, 3.0f, 0.0f), 0.2f * lit) << "behind the blocker";
    EXPECT_EQ(delegate.Renderer().LightShadowViews().size(), 6u) << "a cube at the centre";
}

TEST_F(RasterDelegateTest, DomesLightEverySurfaceDiffusely) {
    Scene scene = FlatScene();
    scene.Remove("/World/Sun");
    scene.Define("/World/Sky", tokens::kDomeLight).Set(tokens::kIntensity, 1.0);
    RasterDelegate delegate(*context_);
    ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
    const auto vars = MapVars(outputs);
    const std::size_t c = kVarSize / 2;
    EXPECT_NEAR(At<float>(vars.at("HdrColor"), c, c, 1), 0.5f, 0.02f)
        << "a uniform 1 cd/m^2 dome gives albedo 0.5 a luminance of 0.5";
    EXPECT_FALSE(delegate.Renderer().Bands().count == 0);

    std::vector<float> map(8 * 16 * 3, 0.0f);
    std::fill(map.begin(), map.begin() + map.size() / 2, 1.0f);
    scene.Write("/World/Sky", tokens::kLightTexture, Tensor::Copy<float>(map, {8, 16, 3}));
    ernie::RenderProductSetOutputs upper = delegate.Step(scene);
    EXPECT_NEAR(At<float>(MapVars(upper).at("HdrColor"), c, c, 1), 0.5f, 0.03f)
        << "the upper hemisphere alone lights upward surfaces fully";
    scene.Set("/World/Sky", tokens::kTransform,
              glm::rotate(glm::dmat4(1.0), M_PI, glm::dvec3(1.0, 0.0, 0.0)));
    ernie::RenderProductSetOutputs lower = delegate.Step(scene);
    EXPECT_LT(At<float>(MapVars(lower).at("HdrColor"), c, c, 1), 0.03f) << "turned upside down";
}

TEST_F(RasterDelegateTest, MetalsReflectDomesAndAmbientLightThroughTheSplitSum) {
    // A white metal under a uniform dome is a white furnace: with multiple scattering restored it
    // reflects the dome's radiance whole at any roughness, as a white Lambertian floor does; the
    // sun's ambient fraction alike. Before, metals (no diffuse albedo) went black under both, and
    // without the compensation lost a third of the light when rough.
    Scene scene = FlatScene();
    scene.Remove("/World/Sun");
    scene.Define("/World/Sky", tokens::kDomeLight).Set(tokens::kIntensity, 1.0);
    scene.Set("/Looks/Grey", tokens::kAlbedo, glm::vec3(1.0f));
    RasterDelegate delegate(*context_);
    const std::size_t c = kVarSize / 2;
    const auto centre = [&] {
        ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
        return At<float>(MapVars(outputs).at("HdrColor"), c, c, 1);
    };
    const float lambert = centre();
    ASSERT_GT(lambert, 0.5f);
    scene.Set("/Looks/Grey", tokens::kBrdf, std::string("preview_surface"));
    scene.Set("/Looks/Grey", "inputs:metallic", 1.0);
    for (const float roughness : {0.2f, 0.6f, 1.0f}) {
        scene.Set("/Looks/Grey", "inputs:roughness", static_cast<double>(roughness));
        EXPECT_NEAR(centre() / lambert, 1.0f, 0.03f) << "dome, roughness " << roughness;
    }

    // The ambient fraction of a sun below the horizon, which lights nothing directly.
    scene.Remove("/World/Sky");
    scene.Define("/World/Sun", tokens::kDistantLight)
        .Set(tokens::kTransform, ernie::DistantLightTransform({0.0, 0.0, -1.0}))
        .Set(tokens::kShadowEnable, false);
    scene.Define("/Render/Settings", tokens::kRenderSettings).Set(tokens::kAmbient, 0.5f);
    scene.Set("/Looks/Grey", tokens::kBrdf, std::string("lambert"));
    const float ambient = centre();
    ASSERT_GT(ambient, 0.0f);
    scene.Set("/Looks/Grey", tokens::kBrdf, std::string("preview_surface"));
    EXPECT_NEAR(centre() / ambient, 1.0f, 0.03f) << "ambient";

    // A smooth metal mirrors the dome's map: facing up under a map bright above the horizon it
    // shows the bright half, and with the dome turned over the dark one.
    scene.Remove("/World/Sun");
    scene.Define("/Render/Settings", tokens::kRenderSettings).Set(tokens::kAmbient, 0.0f);
    std::vector<float> map(8 * 16 * 3, 0.0f);
    std::fill(map.begin(), map.begin() + map.size() / 2, 1.0f);
    scene.Define("/World/Sky", tokens::kDomeLight)
        .Set(tokens::kIntensity, 1.0)
        .Write(tokens::kLightTexture, Tensor::Copy<float>(map, {8, 16, 3}));
    scene.Set("/Looks/Grey", "inputs:roughness", 0.05);
    const float above = centre();
    EXPECT_GT(above, 0.8f * lambert);
    scene.Set("/World/Sky", tokens::kTransform,
              glm::rotate(glm::dmat4(1.0), M_PI, glm::dvec3(1.0, 0.0, 0.0)));
    EXPECT_LT(centre(), 0.05f * above) << "the dark half mirrored";
}

TEST_F(RasterDelegateTest, PhotometricLightsLightByTheirLuxAndLumens) {
    // FlatScene's grey floor (albedo 0.5) under a 1 m square panel 5 m up giving 1000 lux at 5 m:
    // the floor below receives about 990 lux (the panel is not quite a point), so its luminance
    // is 0.5 x 990 / pi cd/m^2.
    Scene scene = FlatScene();
    scene.Remove("/World/Sun");
    // A plain Lambertian floor, not the hillshaded terrain.
    scene.Remove("/World/Terrain");
    scene.Define("/World/Floor", tokens::kGroundPlane)
        .Set(tokens::kMaterialBinding, Path("/Looks/Grey"));
    // Close enough that the centre pixel samples the floor within a few centimetres of the axis,
    // where the illuminance has not yet fallen off (by cos^4 of the angle off it).
    scene.Set("/World/Camera", tokens::kTransform,
              ernie::LookAtTransform({0.0, 0.0, 20.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}));
    scene.Define("/World/Panel", tokens::kRectLight)
        .Set(tokens::kTransform,
             ernie::LookAtTransform({0.0, 0.0, 5.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}))
        .Set(tokens::kPhotometricIlluminance, 1000.0)
        .Set(tokens::kPhotometricIlluminanceDistance, 5.0)
        .Set(tokens::kShadowEnable, false);
    RasterDelegate delegate(*context_);
    auto outputs = delegate.Step(scene);
    const std::size_t c = kVarSize / 2;
    const glm::vec3 hdr(At<float>(MapVars(outputs).at("HdrColor"), c, c, 0),
                        At<float>(MapVars(outputs).at("HdrColor"), c, c, 1),
                        At<float>(MapVars(outputs).at("HdrColor"), c, c, 2));
    const float luminance = glm::dot(hdr, glm::vec3(0.2126f, 0.7152f, 0.0722f));
    EXPECT_NEAR(M_PI * luminance / 0.5, 990.0, 25.0);
}

TEST_F(RasterDelegateTest, ARenderersFirstFrameHasItsLightsShadows) {
    // The shading of a renderer's first frame once read the light shadow maps' size before the
    // lights made them, losing the spot's shadow for that frame.
    Scene scene = LampScene(10.0, 60.0f);
    DefineQuad(scene, "/World/Blocker");
    scene.Set("/World/Blocker", tokens::kTransform,
              glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 0.0, 5.0)));
    RasterDelegate first(*context_);
    const std::vector<std::uint8_t> once = Bytes(first.Render(scene));
    EXPECT_EQ(Bytes(first.Render(scene)), once);
}

TEST_F(RasterDelegateTest, MeteringMeasuresTheScenesLuminance) {
    // The grey floor (albedo 0.5) under a 20000 lux sun straight above: 0.5 x 20000 / pi cd/m^2
    // everywhere, which the meter reads a frame or two later.
    Scene scene = FlatScene();
    scene.Remove("/World/Terrain");
    scene.Define("/World/Floor", tokens::kGroundPlane)
        .Set(tokens::kMaterialBinding, Path("/Looks/Grey"));
    scene.Set("/World/Sun", tokens::kPhotometricIlluminance, 20000.0);
    RasterDelegate delegate(*context_);
    delegate.Render(scene);
    EXPECT_FALSE(delegate.MeteredLuminance()) << "metering is off by default";
    scene.Set("/Render/Settings", tokens::kMeter, true);
    for (int i = 0; i < 3; ++i) {
        delegate.Render(scene);
    }
    ASSERT_TRUE(delegate.MeteredLuminance());
    // Within a quarter of the histogram's half-stop bins.
    EXPECT_NEAR(std::log2(*delegate.MeteredLuminance()), std::log2(0.5 * 20000.0 / M_PI), 0.3);

    // Exposure compensation brightens the photographic image by its stops.
    const auto centre = [&] {
        const Tensor image = delegate.Render(scene);
        return static_cast<int>(image.As<std::uint8_t>()[4 * (kVarSize / 2 * kVarSize + kVarSize / 2)]);
    };
    scene.Set("/World/Camera", tokens::kExposureTime, 1.0 / 20000.0);
    const int base = centre();
    scene.Set("/Render/Settings", tokens::kExposureCompensation, 2.0);
    EXPECT_GT(centre(), base + 40);
}

namespace {

/// FlatScene without its terrain or sun: an empty world, the camera at the origin looking along
/// `forward`.
Scene SkyScene(glm::dvec3 forward, glm::dvec3 up) {
    Scene scene = FlatScene();
    scene.Remove("/World/Terrain");
    scene.Remove("/World/Sun");
    scene.Set("/World/Camera", tokens::kTransform,
              ernie::LookAtTransform({0.0, 0.0, 0.0}, forward, up));
    return scene;
}

/// Band `band` of SpectralRadiance, (bands, rows, cols), at (`row`, `col`) of a `size` square.
float SpectralAt(const Tensor& spectral, std::size_t band, std::size_t row, std::size_t col,
                 std::size_t size) {
    return static_cast<const float*>(spectral.Data())[(band * size + row) * size + col];
}

}  // namespace

TEST_F(RasterDelegateTest, DomesShowBehindTheScene) {
    Scene scene = SkyScene({0.0, 0.0, 1.0}, {0.0, 1.0, 0.0});
    scene.Define("/World/Sky", tokens::kDomeLight).Set(tokens::kIntensity, 1.0);
    RasterDelegate delegate(*context_);
    ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
    auto vars = MapVars(outputs);
    const ernie::SpectralBands bands =
        ernie::CameraOf(scene, "/World/Camera", kVarSize, kVarSize).bands;
    const std::vector<float> dome = ernie::DomeOf(scene, "/World/Sky", bands).radiance;
    const std::size_t c = kVarSize / 2;
    for (int band = 0; band < bands.count; ++band) {
        EXPECT_NEAR(SpectralAt(vars.at("SpectralRadiance"), band, c, c, kVarSize), dome[band],
                    0.02f * dome[band])
            << "band " << band;
    }
    const auto luminance = [&](std::map<std::string, Tensor, std::less<>>& of) {
        return 0.2126f * At<float>(of.at("HdrColor"), c, c, 0) +
               0.7152f * At<float>(of.at("HdrColor"), c, c, 1) +
               0.0722f * At<float>(of.at("HdrColor"), c, c, 2);
    };
    EXPECT_NEAR(luminance(vars), 1.0f, 0.03f) << "a 1 cd/m^2 dome looks 1 cd/m^2";
    EXPECT_EQ(At<std::uint32_t>(vars.at("PrimId"), c, c), 0u) << "still no prim";

    // Red above the horizon, blue below: looking up sees red, until the dome turns over.
    std::vector<float> map(8 * 16 * 3, 0.0f);
    for (std::size_t texel = 0; texel < 8 * 16; ++texel) {
        map[3 * texel + (texel < 4 * 16 ? 0 : 2)] = 1.0f;
    }
    scene.Write("/World/Sky", tokens::kLightTexture, Tensor::Copy<float>(map, {8, 16, 3}));
    ernie::RenderProductSetOutputs upper = delegate.Step(scene);
    auto up = MapVars(upper);
    EXPECT_GT(At<float>(up.at("HdrColor"), c, c, 0), 10.0f * At<float>(up.at("HdrColor"), c, c, 2));
    scene.Set("/World/Sky", tokens::kTransform,
              glm::rotate(glm::dmat4(1.0), M_PI, glm::dvec3(1.0, 0.0, 0.0)));
    ernie::RenderProductSetOutputs turned = delegate.Step(scene);
    auto down = MapVars(turned);
    EXPECT_GT(At<float>(down.at("HdrColor"), c, c, 2),
              10.0f * At<float>(down.at("HdrColor"), c, c, 0));

    scene.Set("/World/Sky", tokens::kVisibleInPrimaryRay, false);
    ernie::RenderProductSetOutputs hidden = delegate.Step(scene);
    EXPECT_EQ(SpectralAt(MapVars(hidden).at("SpectralRadiance"), 8, c, c, kVarSize), 0.0f)
        << "not in camera rays, though it still lights";
}

TEST_F(RasterDelegateTest, StarsKeepTheirIrradianceAtAnyResolutionBehindTheScene) {
    // A magnitude 0 star of the Sun's temperature a little off the view's axis: summing the
    // radiance of its pixels times their solid angles gives back its spectral irradiance, at any
    // resolution.
    Scene scene = SkyScene({1.0, 0.0, 0.0}, {0.0, 0.0, 1.0});
    scene.Define("/World/Stars", tokens::kStarField)
        .Set(tokens::kStarsRightAscension, std::vector<float>{0.7f})
        .Set(tokens::kStarsDeclination, std::vector<float>{-0.4f})
        .Set(tokens::kStarsMagnitude, std::vector<float>{0.0f})
        .Set(tokens::kStarsTemperature, std::vector<float>{5772.0f});
    RasterDelegate delegate(*context_);
    const auto irradiance = [&](std::size_t size, std::size_t band) {
        scene.Set("/Render/Settings", tokens::kResolution,
                  glm::ivec2(static_cast<int>(size), static_cast<int>(size)));
        ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
        const Tensor spectral =
            outputs.at("/Render/Product").vars.at("SpectralRadiance").Map(ernie::MapDevice::Cpu).tensor;
        const ernie::Camera camera = ernie::CameraOf(scene, "/World/Camera", size, size);
        const double f = camera.FocalLengthPixels();
        double sum = 0.0;
        for (std::size_t row = 0; row < size; ++row) {
            for (std::size_t col = 0; col < size; ++col) {
                const double x = (static_cast<double>(col) + 0.5 - 0.5 * size) / f;
                const double y = (static_cast<double>(row) + 0.5 - 0.5 * size) / f;
                const double cosine = 1.0 / std::sqrt(1.0 + x * x + y * y);
                sum += SpectralAt(spectral, band, row, col, size) * cosine * cosine * cosine /
                       (f * f);
            }
        }
        return sum;
    };
    const ernie::SpectralBands bands = ernie::CameraOf(scene, "/World/Camera", 32, 32).bands;
    for (const std::size_t band : {std::size_t{2}, std::size_t{12}}) {
        const double expected = ernie::BlackbodySpectralIrradiance(
            bands.Lower(static_cast<int>(band)) + 0.5 * bands.Width(), 5772.0, 2.54e-6);
        for (const std::size_t size : {32u, 57u, 90u}) {
            EXPECT_NEAR(irradiance(size, band), expected, 0.03 * expected)
                << size << " pixels, band " << band;
        }
    }

    // A quad in front hides it.
    DefineQuad(scene, "/World/Blocker");
    scene.Set("/World/Blocker", tokens::kTransform,
              glm::rotate(glm::translate(glm::dmat4(1.0), glm::dvec3(5.0, 0.0, 0.0)), M_PI / 2.0,
                          glm::dvec3(0.0, 1.0, 0.0)));
    EXPECT_EQ(irradiance(32, 12), 0.0) << "behind the scene";
}
