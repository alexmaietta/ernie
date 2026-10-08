#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"
#include "ernie/session/resolve.h"

#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

using ernie::Light;
using ernie::LightType;
using ernie::Path;
using ernie::Scene;
namespace tokens = ernie::tokens;

namespace {

double LuminanceIntegral() {
    double sum = 0.0;
    for (double w = 360.5; w < 830.0; w += 1.0) {
        sum += ernie::CieXyz(w).y;
    }
    return sum;
}

}  // namespace

TEST(LightOf, SphereLightsInRelativeUnitsCarryTheirSurfaceLuminance) {
    Scene scene;
    scene.Define("/Lamp", tokens::kSphereLight)
        .Set(tokens::kTransform,
             ernie::LookAtTransform({1.0, 2.0, 10.0}, {1.0, 2.0, 0.0}, {0.0, 1.0, 0.0}))
        .Set(tokens::kIntensity, 4.0)
        .Set(tokens::kRadius, 0.5);
    const ernie::SpectralBands bands;
    const Light lamp = ernie::LightOf(scene, "/Lamp", bands);
    EXPECT_EQ(lamp.type, LightType::Sphere);
    EXPECT_EQ(lamp.position, glm::dvec3(1.0, 2.0, 10.0));
    EXPECT_NEAR(lamp.direction.z, -1.0f, 1e-6f) << "emits along its -z axis";
    EXPECT_EQ(lamp.radius, 0.5f);
    EXPECT_EQ(lamp.coneAngle, 180.0f) << "unshaped";
    ASSERT_EQ(lamp.spectrum.size(), 16u);
    const double luminous = 4.0 * M_PI * 0.25;
    EXPECT_NEAR(lamp.spectrum[8], luminous / (ernie::kLuminousEfficacy * LuminanceIntegral()), 1e-7)
        << "luminous intensity = luminance x pi r^2";

    scene.Set("/Lamp", tokens::kTreatAsPoint, true);
    scene.Set("/Lamp", tokens::kShapingConeAngle, 30.0f);
    scene.Set("/Lamp", tokens::kShapingConeSoftness, 0.2f);
    scene.Set("/Lamp", tokens::kShadowEnable, false);
    const Light spot = ernie::LightOf(scene, "/Lamp", bands);
    EXPECT_EQ(spot.radius, 0.0f);
    EXPECT_EQ(spot.coneAngle, 30.0f);
    EXPECT_FLOAT_EQ(spot.coneSoftness, 0.2f);
    EXPECT_FALSE(spot.shadows);
    EXPECT_NEAR(spot.spectrum[8], 4.0 / (ernie::kLuminousEfficacy * LuminanceIntegral()), 1e-7)
        << "a point's luminous intensity is its intensity";
}

TEST(LightOf, PhysicalLampsSpreadTheirPowerOverTheSphere) {
    Scene scene;
    scene.Define("/Lamp", tokens::kSphereLight).Set(tokens::kLightPower, 470.0);
    const ernie::SpectralBands bands;
    for (const float value : ernie::LightOf(scene, "/Lamp", bands).spectrum) {
        EXPECT_NEAR(value, 1.0 / (4.0 * M_PI), 1e-6) << "470 W flat over 470 nm, over 4 pi sr";
    }
    scene.Set("/Lamp", tokens::kLightSpectrum, std::vector<float>{400.0f, 2.0f, 700.0f, 2.0f});
    for (const float value : ernie::LightOf(scene, "/Lamp", {400.0, 700.0, 3}).spectrum) {
        EXPECT_NEAR(value, 470.0 / 300.0 / (4.0 * M_PI), 1e-5) << "the spectrum's shape";
    }
    scene.Clear("/Lamp", tokens::kLightPower);
    EXPECT_EQ(ernie::LightOf(scene, "/Lamp", {400.0, 700.0, 3}).spectrum,
              (std::vector<float>{2.0f, 2.0f, 2.0f}))
        << "a spectrum alone is the intensity";

    scene.Set("/Lamp", tokens::kLightPower, -1.0);
    EXPECT_THROW(ernie::LightOf(scene, "/Lamp", bands), std::invalid_argument);
    scene.Define("/Sky", tokens::kDomeLight);
    EXPECT_THROW(ernie::LightOf(scene, "/Sky", bands), std::invalid_argument) << "see DomeOf";
}

TEST(LightOf, DistantLightsShineFromTheirAxis) {
    Scene scene;
    scene.Define("/Earth", tokens::kDistantLight)
        .Set(tokens::kTransform, ernie::DistantLightTransform({0.0, 1.0, 1.0}))
        .Set(tokens::kLightIrradiance, 0.1);
    const ernie::SpectralBands bands;
    const Light earth = ernie::LightOf(scene, "/Earth", bands);
    EXPECT_EQ(earth.type, LightType::Distant);
    EXPECT_NEAR(earth.direction.y, std::sqrt(0.5f), 1e-6f);
    EXPECT_EQ(earth.spectrum, ernie::SunIrradianceOf(scene, "/Earth", bands));
}

TEST(FrameParamsOf, ResolvesEveryLightAndTheSunsDisk) {
    Scene scene;
    scene.Define("/World/Camera", tokens::kCamera);
    scene.Define("/World/Lamp", tokens::kSphereLight);
    scene.Define("/World/Off", tokens::kSphereLight).Set(tokens::kVisibility, tokens::kInvisible);
    ernie::RenderPrims prims = ernie::ResolveRenderPrims(scene, {});
    EXPECT_EQ(prims.light, Path());
    EXPECT_EQ(prims.lights, std::vector<Path>{"/World/Lamp"});
    ernie::FrameParams frame = ernie::FrameParamsOf(scene, prims, 8, 8);
    EXPECT_FALSE(frame.sun) << "lamps alone light the scene";
    ASSERT_EQ(frame.lights.size(), 1u);

    scene.Define("/World/Sun", tokens::kDistantLight).Set(tokens::kAngle, 2.0f);
    scene.Define("/World/Earth", tokens::kDistantLight);
    prims = ernie::ResolveRenderPrims(scene, {});
    EXPECT_EQ(prims.light, Path("/World/Earth")) << "the first distant light is the sun";
    EXPECT_EQ(prims.lights, (std::vector<Path>{"/World/Lamp", "/World/Sun"}));
    frame = ernie::FrameParamsOf(scene, prims, 8, 8);
    EXPECT_TRUE(frame.sun);
    EXPECT_EQ(frame.sunAngle, 0.53f) << "UsdLux's default angle";
    EXPECT_EQ(frame.lights.size(), 2u);
    scene.Set("/World/Earth", tokens::kAngle, 2.0f);
    EXPECT_EQ(ernie::FrameParamsOf(scene, prims, 8, 8).sunAngle, 2.0f);
}

TEST(LightOf, RectAndDiskLightsTakeTheirSizeAndRadiance) {
    Scene scene;
    scene.Define("/Panel", tokens::kRectLight)
        .Set(tokens::kTransform,
             glm::scale(ernie::LookAtTransform({0.0, 0.0, 5.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}),
                        glm::dvec3(2.0, 1.0, 1.0)))
        .Set(tokens::kWidth, 3.0)
        .Set(tokens::kHeight, 4.0)
        .Set(tokens::kIntensity, 10.0);
    const ernie::SpectralBands bands;
    const Light panel = ernie::LightOf(scene, "/Panel", bands);
    EXPECT_EQ(panel.type, LightType::Rect);
    EXPECT_EQ(panel.size, glm::vec2(6.0f, 4.0f)) << "scaled by the transform";
    EXPECT_NEAR(panel.direction.z, -1.0f, 1e-6f);
    EXPECT_NEAR(glm::dot(panel.tangent, panel.direction), 0.0f, 1e-6f);
    EXPECT_NEAR(panel.spectrum[8], 10.0 / (ernie::kLuminousEfficacy * LuminanceIntegral()), 1e-7)
        << "relative units are the face's luminance";

    scene.Set("/Panel", tokens::kLightPower, 470.0 * M_PI * 24.0);
    EXPECT_NEAR(ernie::LightOf(scene, "/Panel", bands).spectrum[3], 1.0, 1e-5)
        << "power = pi x area x radiance for a Lambertian face";

    scene.Define("/Disk", tokens::kDiskLight).Set(tokens::kRadius, 2.0);
    const Light disk = ernie::LightOf(scene, "/Disk", bands);
    EXPECT_EQ(disk.type, LightType::Disk);
    EXPECT_EQ(disk.radius, 2.0f);
    scene.Set("/Disk", tokens::kRadius, 0.0);
    EXPECT_THROW(ernie::LightOf(scene, "/Disk", bands), std::invalid_argument);
}

namespace {

/// The irradiance factor of `sh` at unit normal `n`.
glm::vec3 Evaluate(const std::array<glm::vec3, 9>& sh, const glm::vec3& n) {
    const float basis[9] = {0.282095f,
                            0.488603f * n.y,
                            0.488603f * n.z,
                            0.488603f * n.x,
                            1.092548f * n.x * n.y,
                            1.092548f * n.y * n.z,
                            0.315392f * (3.0f * n.z * n.z - 1.0f),
                            1.092548f * n.x * n.z,
                            0.546274f * (n.x * n.x - n.y * n.y)};
    glm::vec3 sum(0.0f);
    for (int k = 0; k < 9; ++k) {
        sum += sh[k] * basis[k];
    }
    return sum;
}

ernie::Tensor UpperHemisphere(std::int64_t rows, std::int64_t cols) {
    std::vector<float> map(static_cast<std::size_t>(rows * cols * 4), 0.0f);
    std::fill(map.begin(), map.begin() + static_cast<std::ptrdiff_t>(map.size() / 2), 1.0f);
    return ernie::Tensor::Copy<float>(map, {rows, cols, 4});
}

}  // namespace

TEST(DomeIrradianceSh, ProjectsUniformAndHemisphericalMaps) {
    std::vector<float> ones(32 * 64 * 3, 1.0f);
    const auto uniform =
        ernie::DomeIrradianceSh(ernie::Tensor::Copy<float>(ones, {32, 64, 3}), glm::dmat3(1.0));
    EXPECT_NEAR(uniform[0].g, std::sqrt(4.0 * M_PI), 0.01);
    for (const glm::vec3 n : {glm::vec3(0, 0, 1), glm::vec3(1, 0, 0), glm::vec3(0, -1, 0)}) {
        EXPECT_NEAR(Evaluate(uniform, n).r, 1.0f, 0.01f) << "E = pi L from every side";
    }

    const auto upper = ernie::DomeIrradianceSh(UpperHemisphere(64, 128), glm::dmat3(1.0));
    EXPECT_NEAR(Evaluate(upper, {0, 0, 1}).g, 1.0f, 0.05f) << "a sky facing up";
    EXPECT_NEAR(Evaluate(upper, {0, 0, -1}).g, 0.0f, 0.05f) << "and nothing facing down";
    EXPECT_NEAR(Evaluate(upper, {1, 0, 0}).g, 0.5f, 0.01f) << "half from a wall";

    const glm::dmat3 flip(glm::rotate(glm::dmat4(1.0), M_PI, glm::dvec3(1.0, 0.0, 0.0)));
    const auto flipped = ernie::DomeIrradianceSh(UpperHemisphere(64, 128), flip);
    EXPECT_NEAR(Evaluate(flipped, {0, 0, -1}).g, 1.0f, 0.05f);

    EXPECT_THROW(
        ernie::DomeIrradianceSh(ernie::Tensor::Copy<float>(ones, {32 * 64, 3}), glm::dmat3(1.0)),
        std::invalid_argument);
}

TEST(DomeOf, ResolvesTheDomeAndTurnsOffTheDefaultSun) {
    Scene scene;
    scene.Define("/World/Camera", tokens::kCamera);
    scene.Define("/World/Sky", tokens::kDomeLight).Set(tokens::kIntensity, 2.0);
    const ernie::RenderPrims prims = ernie::ResolveRenderPrims(scene, {});
    EXPECT_EQ(prims.dome, Path("/World/Sky"));
    const ernie::FrameParams frame = ernie::FrameParamsOf(scene, prims, 8, 8);
    EXPECT_FALSE(frame.sun);
    ASSERT_TRUE(frame.dome.has_value());
    EXPECT_NEAR(frame.dome->radiance[8], 2.0 / (ernie::kLuminousEfficacy * LuminanceIntegral()),
                1e-7);
    EXPECT_NEAR(frame.dome->sh[0].r, std::sqrt(4.0 * M_PI), 1e-5);
    EXPECT_THROW(ernie::DomeOf(scene, "/World/Camera", {}), std::invalid_argument);
}

namespace {

/// The luminous value of a binned spectrum: 683 lm/W times its integral against the y function.
double Luminous(const std::vector<float>& spectrum, const ernie::SpectralBands& bands) {
    return ernie::kLuminousEfficacy * ernie::ToXyz(bands, spectrum).y;
}

glm::dvec2 Chromaticity(const std::vector<float>& spectrum, const ernie::SpectralBands& bands) {
    const glm::dvec3 xyz = ernie::ToXyz(bands, spectrum);
    return glm::dvec2(xyz.x, xyz.y) / (xyz.x + xyz.y + xyz.z);
}

}  // namespace

TEST(LightOf, PhotometricUnitsSetTheBrightnessInLuxAndLumens) {
    const ernie::SpectralBands bands{380.0, 780.0, 40};
    Scene scene;
    // A sun of 50000 lux, whatever its intensity and exposure say.
    scene.Define("/Sun", tokens::kDistantLight)
        .Set(tokens::kIntensity, 7.0)
        .Set(tokens::kExposure, 3.0)
        .Set(tokens::kPhotometricIlluminance, 50000.0);
    EXPECT_NEAR(Luminous(ernie::LightOf(scene, "/Sun", bands).spectrum, bands), 50000.0, 500.0);

    // A sphere of 1600 lumens: 1600 / 4 pi candela; or 100 lux at 2 m: 400 cd.
    scene.Define("/Bulb", tokens::kSphereLight).Set(tokens::kPhotometricPower, 1600.0);
    EXPECT_NEAR(Luminous(ernie::LightOf(scene, "/Bulb", bands).spectrum, bands),
                1600.0 / (4.0 * M_PI), 1.5);
    scene.Set("/Bulb", tokens::kPhotometricIlluminance, 100.0);
    scene.Set("/Bulb", tokens::kPhotometricIlluminanceDistance, 2.0);
    EXPECT_NEAR(Luminous(ernie::LightOf(scene, "/Bulb", bands).spectrum, bands), 400.0, 4.0);

    // A 2 m x 1 m panel of 1000 lumens: luminance 1000 / (pi 2) nits; or 50 lux at 3 m.
    scene.Define("/Panel", tokens::kRectLight)
        .Set(tokens::kWidth, 2.0)
        .Set(tokens::kHeight, 1.0)
        .Set(tokens::kPhotometricPower, 1000.0);
    EXPECT_NEAR(Luminous(ernie::LightOf(scene, "/Panel", bands).spectrum, bands),
                1000.0 / (2.0 * M_PI), 1.6);
    scene.Set("/Panel", tokens::kPhotometricIlluminance, 50.0);
    scene.Set("/Panel", tokens::kPhotometricIlluminanceDistance, 3.0);
    EXPECT_NEAR(Luminous(ernie::LightOf(scene, "/Panel", bands).spectrum, bands), 225.0, 2.3);

    // A uniform dome giving 20000 lux on the ground: luminance 20000 / pi.
    scene.Define("/Sky", tokens::kDomeLight).Set(tokens::kPhotometricIlluminance, 20000.0);
    EXPECT_NEAR(Luminous(ernie::DomeOf(scene, "/Sky", bands).radiance, bands), 20000.0 / M_PI,
                64.0);
}

TEST(LightOf, IlluminantsShapeTheSpectrum) {
    const ernie::SpectralBands bands{380.0, 780.0, 40};
    Scene scene;
    scene.Define("/Sun", tokens::kDistantLight).Set(tokens::kPhotometricIlluminance, 1000.0);
    const auto shape = [&](std::string_view illuminant) {
        scene.Set("/Sun", tokens::kPhysicalIlluminant, std::string(illuminant));
        return ernie::LightOf(scene, "/Sun", bands).spectrum;
    };
    const glm::dvec2 white = Chromaticity(shape(tokens::kWhite), bands);
    EXPECT_NEAR(white.x, 1.0 / 3.0, 0.005) << "flat: the equal-energy point";
    scene.Set("/Sun", tokens::kColorTemperature, 6504.0);
    const glm::dvec2 daylight = Chromaticity(shape(tokens::kIlluminantD), bands);
    EXPECT_NEAR(daylight.x, 0.3127, 0.005);
    EXPECT_NEAR(daylight.y, 0.3290, 0.005);
    scene.Set("/Sun", tokens::kColorTemperature, 3000.0);
    const glm::dvec2 warm = Chromaticity(shape(tokens::kBlackbody), bands);
    EXPECT_NEAR(warm.x, 0.4369, 0.005) << "the Planckian locus at 3000 K";
    EXPECT_THROW(shape(tokens::kIlluminantD), std::invalid_argument) << "below 4000 K";

    scene.Set("/Sun", tokens::kPhysicalCustomIlluminant,
              std::vector<float>{540.0f, 1.0f, 560.0f, 1.0f});
    const std::vector<float> custom = shape(tokens::kCustom);
    EXPECT_EQ(custom.front(), 0.0f) << "nothing outside the pairs";
    EXPECT_GT(custom[17], 0.0f) << "the 550 nm band";
    EXPECT_NEAR(Luminous(custom, bands), 1000.0, 10.0) << "still 1000 lux";
    EXPECT_THROW(shape("candle"), std::invalid_argument);

    // Without photometric units, an illuminant only shapes the spectrum.
    Scene relative;
    relative.Define("/Sun", tokens::kDistantLight)
        .Set(tokens::kIntensity, 2.0)
        .Set(tokens::kPhysicalIlluminant, std::string(tokens::kBlackbody))
        .Set(tokens::kColorTemperature, 3000.0);
    const std::vector<float> spectrum = ernie::LightOf(relative, "/Sun", bands).spectrum;
    EXPECT_NEAR(Luminous(spectrum, bands), 2.0 * M_PI, 0.1) << "pi x intensity lux, as before";
    EXPECT_NEAR(Chromaticity(spectrum, bands).x, 0.4369, 0.005);
}
