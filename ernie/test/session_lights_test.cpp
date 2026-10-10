// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/terrain_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/session/resolve.h"
#include "ernie/util/astronomy.h"

#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

using ernie::LightParams;
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

TEST(LightParamsOf, SphereLightsInRelativeUnitsCarryTheirSurfaceLuminance) {
    Scene scene;
    ernie::SphereLight sphere =
        ernie::SphereLight::Define(scene, "/Lamp").SetIntensity(4.0).SetRadius(0.5);
    scene.Set("/Lamp", tokens::kTransform,
              ernie::LookAtTransform({1.0, 2.0, 10.0}, {1.0, 2.0, 0.0}, {0.0, 1.0, 0.0}));
    const ernie::SpectralBands bands;
    const LightParams lamp = ernie::LightParamsOf(scene, "/Lamp", bands);
    EXPECT_EQ(lamp.type, LightType::Sphere);
    EXPECT_EQ(lamp.position, glm::dvec3(1.0, 2.0, 10.0));
    EXPECT_NEAR(lamp.direction.z, -1.0f, 1e-6f) << "emits along its -z axis";
    EXPECT_EQ(lamp.radius, 0.5f);
    EXPECT_EQ(lamp.coneAngle, 180.0f) << "unshaped";
    ASSERT_EQ(lamp.spectrum.size(), 16u);
    const double luminous = 4.0 * M_PI * 0.25;
    EXPECT_NEAR(lamp.spectrum[8], luminous / (ernie::kLuminousEfficacy * LuminanceIntegral()), 1e-7)
        << "luminous intensity = luminance x pi r^2";

    sphere.SetTreatAsPoint(true)
        .SetShapingConeAngle(30.0)
        .SetShapingConeSoftness(0.2)
        .SetShadowEnable(false);
    const LightParams spot = ernie::LightParamsOf(scene, "/Lamp", bands);
    EXPECT_EQ(spot.radius, 0.0f);
    EXPECT_EQ(spot.coneAngle, 30.0f);
    EXPECT_FLOAT_EQ(spot.coneSoftness, 0.2f);
    EXPECT_FALSE(spot.shadows);
    EXPECT_NEAR(spot.spectrum[8], 4.0 / (ernie::kLuminousEfficacy * LuminanceIntegral()), 1e-7)
        << "a point's luminous intensity is its intensity";
}

TEST(LightParamsOf, PhysicalLampsSpreadTheirPowerOverTheSphere) {
    Scene scene;
    ernie::SphereLight lamp = ernie::SphereLight::Define(scene, "/Lamp").SetPower(470.0);
    const ernie::SpectralBands bands;
    for (const float value : ernie::LightParamsOf(scene, "/Lamp", bands).spectrum) {
        EXPECT_NEAR(value, 1.0 / (4.0 * M_PI), 1e-6) << "470 W flat over 470 nm, over 4 pi sr";
    }
    lamp.SetSpectrum({400.0f, 2.0f, 700.0f, 2.0f});
    for (const float value : ernie::LightParamsOf(scene, "/Lamp", {400.0, 700.0, 3}).spectrum) {
        EXPECT_NEAR(value, 470.0 / 300.0 / (4.0 * M_PI), 1e-5) << "the spectrum's shape";
    }
    scene.Clear("/Lamp", tokens::kLightPower);
    EXPECT_EQ(ernie::LightParamsOf(scene, "/Lamp", {400.0, 700.0, 3}).spectrum,
              (std::vector<float>{2.0f, 2.0f, 2.0f}))
        << "a spectrum alone is the intensity";

    lamp.SetPower(-1.0);
    EXPECT_THROW(ernie::LightParamsOf(scene, "/Lamp", bands), std::invalid_argument);
    ernie::DomeLight::Define(scene, "/Sky");
    EXPECT_THROW(ernie::LightParamsOf(scene, "/Sky", bands), std::invalid_argument)
        << "see DomeParamsOf";
}

TEST(LightParamsOf, DistantLightsShineFromTheirAxis) {
    Scene scene;
    ernie::DistantLight::Define(scene, "/Earth").SetIrradiance(0.1);
    scene.Set("/Earth", tokens::kTransform, ernie::DistantLightTransform({0.0, 1.0, 1.0}));
    const ernie::SpectralBands bands;
    const LightParams earth = ernie::LightParamsOf(scene, "/Earth", bands);
    EXPECT_EQ(earth.type, LightType::Distant);
    EXPECT_NEAR(earth.direction.y, std::sqrt(0.5f), 1e-6f);
    EXPECT_EQ(earth.spectrum, ernie::SunIrradianceOf(scene, "/Earth", bands));
}

TEST(FrameParamsOf, ResolvesEveryLightAndTheSunsDisk) {
    Scene scene;
    ernie::Camera::Define(scene, "/World/Camera");
    ernie::SphereLight::Define(scene, "/World/Lamp");
    ernie::SphereLight::Define(scene, "/World/Off").SetVisibility(tokens::kInvisible);
    ernie::RenderPrims prims = ernie::ResolveRenderPrims(scene, {});
    EXPECT_EQ(prims.light, Path());
    EXPECT_EQ(prims.lights, std::vector<Path>{"/World/Lamp"});
    ernie::FrameParams frame = ernie::FrameParamsOf(scene, prims, 8, 8);
    EXPECT_FALSE(frame.sun) << "lamps alone light the scene";
    ASSERT_EQ(frame.lights.size(), 1u);

    ernie::DistantLight::Define(scene, "/World/Sun").SetAngle(2.0);
    ernie::DistantLight earth = ernie::DistantLight::Define(scene, "/World/Earth");
    prims = ernie::ResolveRenderPrims(scene, {});
    EXPECT_EQ(prims.light, Path("/World/Earth")) << "the first distant light is the sun";
    EXPECT_EQ(prims.lights, (std::vector<Path>{"/World/Lamp", "/World/Sun"}));
    frame = ernie::FrameParamsOf(scene, prims, 8, 8);
    EXPECT_TRUE(frame.sun);
    EXPECT_EQ(frame.sunAngle, 0.53f) << "UsdLux's default angle";
    EXPECT_EQ(frame.lights.size(), 2u);
    earth.SetAngle(2.0);
    EXPECT_EQ(ernie::FrameParamsOf(scene, prims, 8, 8).sunAngle, 2.0f);
}

TEST(LightParamsOf, RectAndDiskLightsTakeTheirSizeAndRadiance) {
    Scene scene;
    ernie::RectLight rect =
        ernie::RectLight::Define(scene, "/Panel").SetWidth(3.0).SetHeight(4.0).SetIntensity(10.0);
    scene.Set("/Panel", tokens::kTransform,
              glm::scale(ernie::LookAtTransform({0.0, 0.0, 5.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}),
                         glm::dvec3(2.0, 1.0, 1.0)));
    const ernie::SpectralBands bands;
    const LightParams panel = ernie::LightParamsOf(scene, "/Panel", bands);
    EXPECT_EQ(panel.type, LightType::Rect);
    EXPECT_EQ(panel.size, glm::vec2(6.0f, 4.0f)) << "scaled by the transform";
    EXPECT_NEAR(panel.direction.z, -1.0f, 1e-6f);
    EXPECT_NEAR(glm::dot(panel.tangent, panel.direction), 0.0f, 1e-6f);
    EXPECT_NEAR(panel.spectrum[8], 10.0 / (ernie::kLuminousEfficacy * LuminanceIntegral()), 1e-7)
        << "relative units are the face's luminance";

    rect.SetPower(470.0 * M_PI * 24.0);
    EXPECT_NEAR(ernie::LightParamsOf(scene, "/Panel", bands).spectrum[3], 1.0, 1e-5)
        << "power = pi x area x radiance for a Lambertian face";

    ernie::DiskLight diskLight = ernie::DiskLight::Define(scene, "/Disk").SetRadius(2.0);
    const LightParams disk = ernie::LightParamsOf(scene, "/Disk", bands);
    EXPECT_EQ(disk.type, LightType::Disk);
    EXPECT_EQ(disk.radius, 2.0f);
    diskLight.SetRadius(0.0);
    EXPECT_THROW(ernie::LightParamsOf(scene, "/Disk", bands), std::invalid_argument);
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

TEST(DomeParamsOf, ResolvesTheDomeAndTurnsOffTheDefaultSun) {
    Scene scene;
    ernie::Camera::Define(scene, "/World/Camera");
    ernie::DomeLight::Define(scene, "/World/Sky").SetIntensity(2.0);
    const ernie::RenderPrims prims = ernie::ResolveRenderPrims(scene, {});
    EXPECT_EQ(prims.dome, Path("/World/Sky"));
    const ernie::FrameParams frame = ernie::FrameParamsOf(scene, prims, 8, 8);
    EXPECT_FALSE(frame.sun);
    ASSERT_TRUE(frame.dome.has_value());
    EXPECT_NEAR(frame.dome->radiance[8], 2.0 / (ernie::kLuminousEfficacy * LuminanceIntegral()),
                1e-7);
    EXPECT_NEAR(frame.dome->sh[0].r, std::sqrt(4.0 * M_PI), 1e-5);
    EXPECT_THROW(ernie::DomeParamsOf(scene, "/World/Camera", {}), std::invalid_argument);
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

TEST(LightParamsOf, PhotometricUnitsSetTheBrightnessInLuxAndLumens) {
    const ernie::SpectralBands bands{380.0, 780.0, 40};
    Scene scene;
    // A sun of 50000 lux, whatever its intensity and exposure say.
    ernie::DistantLight::Define(scene, "/Sun")
        .SetIntensity(7.0)
        .SetExposure(3.0)
        .SetPhotometricIlluminance(50000.0);
    EXPECT_NEAR(Luminous(ernie::LightParamsOf(scene, "/Sun", bands).spectrum, bands), 50000.0,
                500.0);

    // A sphere of 1600 lumens: 1600 / 4 pi candela; or 100 lux at 2 m: 400 cd.
    ernie::SphereLight bulb =
        ernie::SphereLight::Define(scene, "/Bulb").SetPhotometricPower(1600.0);
    EXPECT_NEAR(Luminous(ernie::LightParamsOf(scene, "/Bulb", bands).spectrum, bands),
                1600.0 / (4.0 * M_PI), 1.5);
    bulb.SetPhotometricIlluminance(100.0).SetPhotometricIlluminanceDistance(2.0);
    EXPECT_NEAR(Luminous(ernie::LightParamsOf(scene, "/Bulb", bands).spectrum, bands), 400.0, 4.0);

    // A 2 m x 1 m panel of 1000 lumens: luminance 1000 / (pi 2) nits; or 50 lux at 3 m.
    ernie::RectLight panel = ernie::RectLight::Define(scene, "/Panel")
                                 .SetWidth(2.0)
                                 .SetHeight(1.0)
                                 .SetPhotometricPower(1000.0);
    EXPECT_NEAR(Luminous(ernie::LightParamsOf(scene, "/Panel", bands).spectrum, bands),
                1000.0 / (2.0 * M_PI), 1.6);
    panel.SetPhotometricIlluminance(50.0).SetPhotometricIlluminanceDistance(3.0);
    EXPECT_NEAR(Luminous(ernie::LightParamsOf(scene, "/Panel", bands).spectrum, bands), 225.0, 2.3);

    // A uniform dome giving 20000 lux on the ground: luminance 20000 / pi.
    ernie::DomeLight::Define(scene, "/Sky").SetPhotometricIlluminance(20000.0);
    EXPECT_NEAR(Luminous(ernie::DomeParamsOf(scene, "/Sky", bands).radiance, bands), 20000.0 / M_PI,
                64.0);
}

TEST(LightParamsOf, IlluminantsShapeTheSpectrum) {
    const ernie::SpectralBands bands{380.0, 780.0, 40};
    Scene scene;
    ernie::DistantLight sun =
        ernie::DistantLight::Define(scene, "/Sun").SetPhotometricIlluminance(1000.0);
    const auto shape = [&](std::string_view illuminant) {
        sun.SetPhysicalIlluminant(illuminant);
        return ernie::LightParamsOf(scene, "/Sun", bands).spectrum;
    };
    const glm::dvec2 white = Chromaticity(shape(tokens::kWhite), bands);
    EXPECT_NEAR(white.x, 1.0 / 3.0, 0.005) << "flat: the equal-energy point";
    sun.SetColorTemperature(6504.0);
    const glm::dvec2 daylight = Chromaticity(shape(tokens::kIlluminantD), bands);
    EXPECT_NEAR(daylight.x, 0.3127, 0.005);
    EXPECT_NEAR(daylight.y, 0.3290, 0.005);
    sun.SetColorTemperature(3000.0);
    const glm::dvec2 warm = Chromaticity(shape(tokens::kBlackbody), bands);
    EXPECT_NEAR(warm.x, 0.4369, 0.005) << "the Planckian locus at 3000 K";
    EXPECT_THROW(shape(tokens::kIlluminantD), std::invalid_argument) << "below 4000 K";

    sun.SetPhysicalCustomIlluminant({540.0f, 1.0f, 560.0f, 1.0f});
    const std::vector<float> custom = shape(tokens::kCustom);
    EXPECT_EQ(custom.front(), 0.0f) << "nothing outside the pairs";
    EXPECT_GT(custom[17], 0.0f) << "the 550 nm band";
    EXPECT_NEAR(Luminous(custom, bands), 1000.0, 10.0) << "still 1000 lux";
    EXPECT_THROW(shape("candle"), std::invalid_argument);

    // Without photometric units, an illuminant only shapes the spectrum.
    Scene relative;
    ernie::DistantLight::Define(relative, "/Sun")
        .SetIntensity(2.0)
        .SetPhysicalIlluminant(tokens::kBlackbody)
        .SetColorTemperature(3000.0);
    const std::vector<float> spectrum = ernie::LightParamsOf(relative, "/Sun", bands).spectrum;
    EXPECT_NEAR(Luminous(spectrum, bands), 2.0 * M_PI, 0.1) << "pi x intensity lux, as before";
    EXPECT_NEAR(Chromaticity(spectrum, bands).x, 0.4369, 0.005);
}

TEST(StarsOf, StarsShineByTheirMagnitudeAndColour) {
    Scene scene;
    scene.Define("/Sky", tokens::kStarField)
        .Set(tokens::kStarsRightAscension, std::vector<float>{0.0f, 90.0f, 180.0f})
        .Set(tokens::kStarsDeclination, std::vector<float>{0.0f, 45.0f, -30.0f})
        .Set(tokens::kStarsMagnitude, std::vector<float>{0.0f, 5.0f, 9.0f})
        .Set(tokens::kStarsColorIndex, std::vector<float>{0.65f, 0.0f, 1.5f});
    std::vector<ernie::Star> stars = ernie::StarsOf(scene, "/Sky");
    ASSERT_EQ(stars.size(), 3u);
    EXPECT_NEAR(stars[1].rightAscension, M_PI / 2.0, 1e-6);
    EXPECT_NEAR(stars[1].declination, M_PI / 4.0, 1e-6);
    EXPECT_NEAR(stars[0].temperature, 5778.0f, 5.0f) << "B-V 0.65";
    EXPECT_NEAR(stars[0].irradiance555,
                ernie::BlackbodySpectralIrradiance(555.0, stars[0].temperature, 2.54e-6), 1e-15);
    EXPECT_GT(stars[2].temperature, 0.0f);
    EXPECT_LT(stars[2].temperature, stars[1].temperature) << "redder is cooler";

    ernie::StarField sky(scene, "/Sky");
    sky.SetMagnitudeLimit(6.0);
    EXPECT_EQ(ernie::StarsOf(scene, "/Sky").size(), 2u) << "fainter than the limit";
    sky.SetExposure(1.0);
    EXPECT_NEAR(ernie::StarsOf(scene, "/Sky")[0].irradiance555 / stars[0].irradiance555, 2.0, 1e-5)
        << "2^exposure";

    scene.Set("/Sky", tokens::kStarsDeclination, std::vector<float>{0.0f});
    EXPECT_THROW(ernie::StarsOf(scene, "/Sky"), std::invalid_argument) << "lengths differ";
    Scene directions;
    directions.Define("/Sky", tokens::kStarField)
        .Set(tokens::kStarsDirection, std::vector<float>{0.0f, 0.0f, 2.0f})
        .Set(tokens::kStarsMagnitude, std::vector<float>{1.0f});
    EXPECT_NEAR(ernie::StarsOf(directions, "/Sky")[0].declination, M_PI / 2.0, 1e-6);
    EXPECT_NEAR(ernie::StarsOf(directions, "/Sky")[0].temperature, 5772.0f, 1e-3f)
        << "the Sun's without a colour";
    directions.Define("/Bare", tokens::kStarField)
        .Set(tokens::kStarsMagnitude, std::vector<float>{1.0f});
    EXPECT_THROW(ernie::StarsOf(directions, "/Bare"), std::invalid_argument) << "no places";
}

TEST(FrameParamsOf, DomesShowAndStarFieldsTurnWithTheirTransforms) {
    Scene scene;
    ernie::Camera::Define(scene, "/Camera");
    ernie::DomeLight dome = ernie::DomeLight::Define(scene, "/Dome");
    scene.Set("/Dome", tokens::kTransform,
              glm::rotate(glm::dmat4(1.0), M_PI / 2.0, glm::dvec3(0, 0, 1)));
    scene.Define("/Stars", tokens::kStarField)
        .Set(tokens::kTransform, glm::rotate(glm::dmat4(1.0), M_PI / 2.0, glm::dvec3(1, 0, 0)));
    const ernie::RenderPrims prims = ernie::ResolveRenderPrims(scene, {});
    EXPECT_EQ(prims.stars, Path("/Stars"));
    ernie::FrameParams params = ernie::FrameParamsOf(scene, prims, 16, 16);
    ASSERT_TRUE(params.dome && params.stars);
    EXPECT_TRUE(params.dome->visible);
    // World +y is the dome's +x after a quarter turn about z.
    const glm::vec3 local = params.dome->toDome * glm::vec3(0.0f, 1.0f, 0.0f);
    EXPECT_NEAR(local.x, 1.0f, 1e-6f);
    // World +y is the catalogue's -z after a quarter turn about x.
    EXPECT_NEAR((params.stars->toCatalogue * glm::vec3(0.0f, 1.0f, 0.0f)).z, -1.0f, 1e-6f);
    dome.SetVisibleInPrimaryRay(false);
    EXPECT_FALSE(ernie::FrameParamsOf(scene, prims, 16, 16).dome->visible);
}
