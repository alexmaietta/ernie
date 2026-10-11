// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/terrain_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/session/resolve.h"
#include "ernie/util/astronomy.h"
#include "ernie/util/ies.h"
#include "ernie/util/solar.h"

#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <stb_image_write.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
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

TEST(LightParamsOf,
     SphereLightsInRelativeUnitsCarryTheirSurfaceLuminance) {
    Scene scene;
    ernie::SphereLight sphere =
        ernie::SphereLight::Define(scene, "/Lamp")
            .SetIntensity(4.0)
            .SetRadius(0.5);
    scene.Set("/Lamp", tokens::kTransform,
              ernie::LookAtTransform({1.0, 2.0, 10.0},
                                     {1.0, 2.0, 0.0},
                                     {0.0, 1.0, 0.0}));
    const ernie::SpectralBands bands;
    const LightParams lamp =
        ernie::LightParamsOf(scene, "/Lamp", bands);
    EXPECT_EQ(lamp.type, LightType::Sphere);
    EXPECT_EQ(lamp.position, glm::dvec3(1.0, 2.0, 10.0));
    EXPECT_NEAR(lamp.direction.z, -1.0f, 1e-6f)
        << "emits along its -z axis";
    EXPECT_EQ(lamp.radius, 0.5f);
    EXPECT_EQ(lamp.coneAngle, 180.0f) << "unshaped";
    ASSERT_EQ(lamp.spectrum.size(), 16u);
    const double luminous = 4.0 * M_PI * 0.25;
    EXPECT_NEAR(lamp.spectrum[8],
                luminous /
                    (ernie::kLuminousEfficacy * LuminanceIntegral()),
                1e-7)
        << "luminous intensity = luminance x pi r^2";

    sphere.SetTreatAsPoint(true)
        .SetShapingConeAngle(30.0)
        .SetShapingConeSoftness(0.2)
        .SetShadowEnable(false);
    const LightParams spot =
        ernie::LightParamsOf(scene, "/Lamp", bands);
    EXPECT_EQ(spot.radius, 0.0f);
    EXPECT_EQ(spot.coneAngle, 30.0f);
    EXPECT_FLOAT_EQ(spot.coneSoftness, 0.2f);
    EXPECT_FALSE(spot.shadows);
    EXPECT_NEAR(
        spot.spectrum[8],
        4.0 / (ernie::kLuminousEfficacy * LuminanceIntegral()), 1e-7)
        << "a point's luminous intensity is its intensity";
}

TEST(LightParamsOf, PhysicalLampsSpreadTheirPowerOverTheSphere) {
    Scene scene;
    ernie::SphereLight lamp =
        ernie::SphereLight::Define(scene, "/Lamp").SetPower(470.0);
    const ernie::SpectralBands bands;
    for (const float value :
         ernie::LightParamsOf(scene, "/Lamp", bands).spectrum) {
        EXPECT_NEAR(value, 1.0 / (4.0 * M_PI), 1e-6)
            << "470 W flat over 470 nm, over 4 pi sr";
    }
    lamp.SetSpectrum({400.0f, 2.0f, 700.0f, 2.0f});
    for (const float value :
         ernie::LightParamsOf(scene, "/Lamp", {400.0, 700.0, 3})
             .spectrum) {
        EXPECT_NEAR(value, 470.0 / 300.0 / (4.0 * M_PI), 1e-5)
            << "the spectrum's shape";
    }
    scene.Clear("/Lamp", tokens::kLightPower);
    EXPECT_EQ(ernie::LightParamsOf(scene, "/Lamp", {400.0, 700.0, 3})
                  .spectrum,
              (std::vector<float>{2.0f, 2.0f, 2.0f}))
        << "a spectrum alone is the intensity";

    lamp.SetPower(-1.0);
    EXPECT_THROW(ernie::LightParamsOf(scene, "/Lamp", bands),
                 std::invalid_argument);
    ernie::DomeLight::Define(scene, "/Sky");
    EXPECT_THROW(ernie::LightParamsOf(scene, "/Sky", bands),
                 std::invalid_argument)
        << "see DomeParamsOf";
}

TEST(LightParamsOf, DistantLightsShineFromTheirAxis) {
    Scene scene;
    ernie::DistantLight::Define(scene, "/Earth").SetIrradiance(0.1);
    scene.Set("/Earth", tokens::kTransform,
              ernie::DistantLightTransform({0.0, 1.0, 1.0}));
    const ernie::SpectralBands bands;
    const LightParams earth =
        ernie::LightParamsOf(scene, "/Earth", bands);
    EXPECT_EQ(earth.type, LightType::Distant);
    EXPECT_NEAR(earth.direction.y, std::sqrt(0.5f), 1e-6f);
    EXPECT_EQ(earth.spectrum,
              ernie::SunIrradianceOf(scene, "/Earth", bands));
}

TEST(FrameParamsOf, ResolvesEveryLightAndTheSunsDisk) {
    Scene scene;
    ernie::Camera::Define(scene, "/World/Camera");
    ernie::SphereLight::Define(scene, "/World/Lamp");
    ernie::SphereLight::Define(scene, "/World/Off")
        .SetVisibility(tokens::kInvisible);
    ernie::RenderPrims prims = ernie::ResolveRenderPrims(scene, {});
    EXPECT_EQ(prims.light, Path());
    EXPECT_EQ(prims.lights, std::vector<Path>{"/World/Lamp"});
    ernie::FrameParams frame =
        ernie::FrameParamsOf(scene, prims, 8, 8);
    EXPECT_FALSE(frame.sun) << "lamps alone light the scene";
    ASSERT_EQ(frame.lights.size(), 1u);

    ernie::DistantLight::Define(scene, "/World/Sun").SetAngle(2.0);
    ernie::DistantLight earth =
        ernie::DistantLight::Define(scene, "/World/Earth");
    prims = ernie::ResolveRenderPrims(scene, {});
    EXPECT_EQ(prims.light, Path("/World/Earth"))
        << "the first distant light is the sun";
    EXPECT_EQ(prims.lights,
              (std::vector<Path>{"/World/Lamp", "/World/Sun"}));
    frame = ernie::FrameParamsOf(scene, prims, 8, 8);
    EXPECT_TRUE(frame.sun);
    EXPECT_EQ(frame.sunAngle, 0.53f) << "UsdLux's default angle";
    EXPECT_EQ(frame.lights.size(), 2u);
    earth.SetAngle(2.0);
    EXPECT_EQ(ernie::FrameParamsOf(scene, prims, 8, 8).sunAngle,
              2.0f);
}

TEST(LightParamsOf, RectAndDiskLightsTakeTheirSizeAndRadiance) {
    Scene scene;
    ernie::RectLight rect = ernie::RectLight::Define(scene, "/Panel")
                                .SetWidth(3.0)
                                .SetHeight(4.0)
                                .SetIntensity(10.0);
    scene.Set("/Panel", tokens::kTransform,
              glm::scale(ernie::LookAtTransform({0.0, 0.0, 5.0},
                                                {0.0, 0.0, 0.0},
                                                {0.0, 1.0, 0.0}),
                         glm::dvec3(2.0, 1.0, 1.0)));
    const ernie::SpectralBands bands;
    const LightParams panel =
        ernie::LightParamsOf(scene, "/Panel", bands);
    EXPECT_EQ(panel.type, LightType::Rect);
    EXPECT_EQ(panel.size, glm::vec2(6.0f, 4.0f))
        << "scaled by the transform";
    EXPECT_NEAR(panel.direction.z, -1.0f, 1e-6f);
    EXPECT_NEAR(glm::dot(panel.tangent, panel.direction), 0.0f,
                1e-6f);
    EXPECT_NEAR(
        panel.spectrum[8],
        10.0 / (ernie::kLuminousEfficacy * LuminanceIntegral()), 1e-7)
        << "relative units are the face's luminance";

    rect.SetPower(470.0 * M_PI * 24.0);
    EXPECT_NEAR(
        ernie::LightParamsOf(scene, "/Panel", bands).spectrum[3], 1.0,
        1e-5)
        << "power = pi x area x radiance for a Lambertian face";

    ernie::DiskLight diskLight =
        ernie::DiskLight::Define(scene, "/Disk").SetRadius(2.0);
    const LightParams disk =
        ernie::LightParamsOf(scene, "/Disk", bands);
    EXPECT_EQ(disk.type, LightType::Disk);
    EXPECT_EQ(disk.radius, 2.0f);
    diskLight.SetRadius(0.0);
    EXPECT_THROW(ernie::LightParamsOf(scene, "/Disk", bands),
                 std::invalid_argument);
}

namespace {

/// The irradiance factor of `sh` at unit normal `n`.
glm::vec3 Evaluate(const std::array<glm::vec3, 9>& sh,
                   const glm::vec3& n) {
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
    std::vector<float> map(static_cast<std::size_t>(rows * cols * 4),
                           0.0f);
    std::fill(map.begin(),
              map.begin() +
                  static_cast<std::ptrdiff_t>(map.size() / 2),
              1.0f);
    return ernie::Tensor::Copy<float>(map, {rows, cols, 4});
}

}  // namespace

TEST(DomeIrradianceSh, ProjectsUniformAndHemisphericalMaps) {
    std::vector<float> ones(32 * 64 * 3, 1.0f);
    const auto uniform = ernie::DomeIrradianceSh(
        ernie::Tensor::Copy<float>(ones, {32, 64, 3}),
        glm::dmat3(1.0));
    EXPECT_NEAR(uniform[0].g, std::sqrt(4.0 * M_PI), 0.01);
    for (const glm::vec3 n : {glm::vec3(0, 0, 1), glm::vec3(1, 0, 0),
                              glm::vec3(0, -1, 0)}) {
        EXPECT_NEAR(Evaluate(uniform, n).r, 1.0f, 0.01f)
            << "E = pi L from every side";
    }

    const auto upper = ernie::DomeIrradianceSh(
        UpperHemisphere(64, 128), glm::dmat3(1.0));
    EXPECT_NEAR(Evaluate(upper, {0, 0, 1}).g, 1.0f, 0.05f)
        << "a sky facing up";
    EXPECT_NEAR(Evaluate(upper, {0, 0, -1}).g, 0.0f, 0.05f)
        << "and nothing facing down";
    EXPECT_NEAR(Evaluate(upper, {1, 0, 0}).g, 0.5f, 0.01f)
        << "half from a wall";

    const glm::dmat3 flip(glm::rotate(glm::dmat4(1.0), M_PI,
                                      glm::dvec3(1.0, 0.0, 0.0)));
    const auto flipped =
        ernie::DomeIrradianceSh(UpperHemisphere(64, 128), flip);
    EXPECT_NEAR(Evaluate(flipped, {0, 0, -1}).g, 1.0f, 0.05f);

    EXPECT_THROW(ernie::DomeIrradianceSh(
                     ernie::Tensor::Copy<float>(ones, {32 * 64, 3}),
                     glm::dmat3(1.0)),
                 std::invalid_argument);
}

TEST(DomeParamsOf, ResolvesTheDomeAndTurnsOffTheDefaultSun) {
    Scene scene;
    ernie::Camera::Define(scene, "/World/Camera");
    ernie::DomeLight::Define(scene, "/World/Sky").SetIntensity(2.0);
    const ernie::RenderPrims prims =
        ernie::ResolveRenderPrims(scene, {});
    EXPECT_EQ(prims.dome, Path("/World/Sky"));
    const ernie::FrameParams frame =
        ernie::FrameParamsOf(scene, prims, 8, 8);
    EXPECT_FALSE(frame.sun);
    ASSERT_TRUE(frame.dome.has_value());
    EXPECT_NEAR(
        frame.dome->radiance[8],
        2.0 / (ernie::kLuminousEfficacy * LuminanceIntegral()), 1e-7);
    EXPECT_NEAR(frame.dome->sh[0].r, std::sqrt(4.0 * M_PI), 1e-5);
    EXPECT_THROW(ernie::DomeParamsOf(scene, "/World/Camera", {}),
                 std::invalid_argument);
}

namespace {

/// The luminous value of a binned spectrum: 683 lm/W times its
/// integral against the y function.
double Luminous(const std::vector<float>& spectrum,
                const ernie::SpectralBands& bands) {
    return ernie::kLuminousEfficacy * ernie::ToXyz(bands, spectrum).y;
}

glm::dvec2 Chromaticity(const std::vector<float>& spectrum,
                        const ernie::SpectralBands& bands) {
    const glm::dvec3 xyz = ernie::ToXyz(bands, spectrum);
    return glm::dvec2(xyz.x, xyz.y) / (xyz.x + xyz.y + xyz.z);
}

/// Illuminance per unit luminance on a surface facing a flat
/// Lambertian emitter on its axis `distance` away, by brute force:
/// the integral of d^2 / r^4 over the points of a grid over (-half,
/// half) that `inside` says are on the emitter.
template <typename Inside>
double OnAxisIlluminancePerLuminance(glm::dvec2 half, double distance,
                                     const Inside& inside) {
    constexpr int kSteps = 1000;
    const glm::dvec2 step = 2.0 * half / static_cast<double>(kSteps);
    double sum = 0.0;
    for (int i = 0; i < kSteps; ++i) {
        for (int j = 0; j < kSteps; ++j) {
            const glm::dvec2 p =
                -half + (glm::dvec2(i, j) + 0.5) * step;
            if (inside(p)) {
                const double r2 =
                    glm::dot(p, p) + distance * distance;
                sum += distance * distance / (r2 * r2);
            }
        }
    }
    return sum * step.x * step.y;
}

/// The solid angle under a cone's falloff, by brute force: its
/// smoothstep in the cosine off the axis (as the shading's)
/// integrated over the sphere.
double ConeSolidAngleBySum(double angleDegrees, double softness) {
    const double outer = std::cos(glm::radians(angleDegrees));
    const double inner =
        std::cos(glm::radians(angleDegrees * (1.0 - softness)));
    constexpr int kSteps = 100000;
    double sum = 0.0;
    for (int i = 0; i < kSteps; ++i) {
        const double theta = (i + 0.5) * M_PI / kSteps;
        const double t = std::clamp(
            (std::cos(theta) - outer) / (inner - outer), 0.0, 1.0);
        sum += t * t * (3.0 - 2.0 * t) * std::sin(theta);
    }
    return 2.0 * M_PI * sum * M_PI / kSteps;
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
    EXPECT_NEAR(
        Luminous(ernie::LightParamsOf(scene, "/Sun", bands).spectrum,
                 bands),
        50000.0, 500.0);

    // A sphere of 1600 lumens: 1600 / 4 pi candela; or 100 lux at 2
    // m: 400 cd.
    ernie::SphereLight bulb =
        ernie::SphereLight::Define(scene, "/Bulb")
            .SetPhotometricPower(1600.0);
    EXPECT_NEAR(
        Luminous(ernie::LightParamsOf(scene, "/Bulb", bands).spectrum,
                 bands),
        1600.0 / (4.0 * M_PI), 1.5);
    bulb.SetPhotometricIlluminance(100.0)
        .SetPhotometricIlluminanceDistance(2.0);
    EXPECT_NEAR(
        Luminous(ernie::LightParamsOf(scene, "/Bulb", bands).spectrum,
                 bands),
        400.0, 4.0);

    // A 2 m x 1 m panel of 1000 lumens: luminance 1000 / (pi 2) nits;
    // or 50 lux at 3 m, a little more than 50 x 3^2 / 2 nits, since
    // its edges light the axis at a slant.
    ernie::RectLight panel = ernie::RectLight::Define(scene, "/Panel")
                                 .SetWidth(2.0)
                                 .SetHeight(1.0)
                                 .SetPhotometricPower(1000.0);
    EXPECT_NEAR(
        Luminous(
            ernie::LightParamsOf(scene, "/Panel", bands).spectrum,
            bands),
        1000.0 / (2.0 * M_PI), 1.6);
    panel.SetPhotometricIlluminance(50.0)
        .SetPhotometricIlluminanceDistance(3.0);
    const double panelLuminance =
        50.0 / OnAxisIlluminancePerLuminance(
                   {1.0, 0.5}, 3.0, [](glm::dvec2) { return true; });
    EXPECT_GT(panelLuminance, 225.0);
    EXPECT_NEAR(
        Luminous(
            ernie::LightParamsOf(scene, "/Panel", bands).spectrum,
            bands),
        panelLuminance, 0.01 * panelLuminance);

    // A uniform dome giving 20000 lux on the ground: luminance 20000
    // / pi.
    ernie::DomeLight::Define(scene, "/Sky")
        .SetPhotometricIlluminance(20000.0);
    EXPECT_NEAR(
        Luminous(ernie::DomeParamsOf(scene, "/Sky", bands).radiance,
                 bands),
        20000.0 / M_PI, 64.0);
}

TEST(LightParamsOf, PhotometricAreaLightsFollowOmniversesRules) {
    const ernie::SpectralBands bands{380.0, 780.0, 40};
    Scene scene;
    const auto luminous = [&](std::string_view path) {
        return Luminous(
            ernie::LightParamsOf(scene, path, bands).spectrum, bands);
    };

    // An illuminance counts only with a distance, both above 0;
    // otherwise the power does.
    ernie::RectLight panel =
        ernie::RectLight::Define(scene, "/Panel")
            .SetWidth(2.0)
            .SetHeight(1.0)
            .SetPhotometricPower(1000.0)
            .SetPhotometricIlluminanceDistance(3.0);
    EXPECT_NEAR(luminous("/Panel"), 1000.0 / (2.0 * M_PI), 1.6)
        << "a distance alone";
    panel.ClearPhotometricIlluminanceDistance()
        .SetPhotometricIlluminance(50.0);
    EXPECT_NEAR(luminous("/Panel"), 1000.0 / (2.0 * M_PI), 1.6)
        << "an illuminance alone";

    // Close up, the illuminance is still exactly what a surface
    // facing the light on its axis gets, far from the distance
    // squared over the area.
    panel.SetPhotometricIlluminanceDistance(0.5);
    const double rect =
        50.0 / OnAxisIlluminancePerLuminance(
                   {1.0, 0.5}, 0.5, [](glm::dvec2) { return true; });
    EXPECT_GT(rect, 1.5 * 50.0 * 0.25 / 2.0);
    EXPECT_NEAR(luminous("/Panel"), rect, 0.01 * rect);
    ernie::DiskLight::Define(scene, "/Disk")
        .SetRadius(1.0)
        .SetPhotometricIlluminance(500.0)
        .SetPhotometricIlluminanceDistance(0.5);
    const double disk =
        500.0 / OnAxisIlluminancePerLuminance(
                    {1.0, 1.0}, 0.5, [](glm::dvec2 p) {
                        return glm::dot(p, p) <= 1.0;
                    });
    EXPECT_NEAR(luminous("/Disk"), disk, 0.01 * disk);

    // A spotlight of 1600 lumens emits them all within its cone, so a
    // narrower cone is brighter.
    ernie::SphereLight spot =
        ernie::SphereLight::Define(scene, "/Spot")
            .SetPhotometricPower(1600.0)
            .SetShapingConeAngle(40.0)
            .SetShapingConeSoftness(0.5);
    const double cone = ConeSolidAngleBySum(40.0, 0.5);
    EXPECT_NEAR(luminous("/Spot"), 1600.0 / cone,
                0.015 * 1600.0 / cone);
    spot.SetShapingConeSoftness(0.0);
    const double hard = ConeSolidAngleBySum(40.0, 0.0);
    EXPECT_NEAR(luminous("/Spot"), 1600.0 / hard,
                0.015 * 1600.0 / hard);
    spot.SetPhotometricIlluminance(100.0)
        .SetPhotometricIlluminanceDistance(2.0);
    EXPECT_NEAR(luminous("/Spot"), 400.0, 4.0)
        << "on its axis, the cone does not matter";

    // So does a physical lamp its watts.
    ernie::SphereLight lamp =
        ernie::SphereLight::Define(scene, "/Lamp")
            .SetPower(470.0)
            .SetShapingConeAngle(40.0)
            .SetShapingConeSoftness(0.5);
    for (const float value :
         ernie::LightParamsOf(scene, "/Lamp", {}).spectrum) {
        EXPECT_NEAR(value, 1.0 / cone, 1e-3 / cone)
            << "470 W flat over 470 nm, over the cone";
    }
    lamp.SetShapingConeAngle(180.0);
    EXPECT_NEAR(ernie::LightParamsOf(scene, "/Lamp", {}).spectrum[0],
                1.0 / (4.0 * M_PI), 1e-6);
}

TEST(LightParamsOf, NormalizedLightsDivideTheirIntensityByTheirSize) {
    const ernie::SpectralBands bands{380.0, 780.0, 40};
    Scene scene;
    const auto luminous = [&](std::string_view path) {
        return Luminous(
            ernie::LightParamsOf(scene, path, bands).spectrum, bands);
    };
    // A 2 m x 0.5 m panel of intensity 8: luminance 8, or 8 / 1 m^2
    // normalized; scaled up by 2, the normalized one keeps its power.
    ernie::RectLight panel = ernie::RectLight::Define(scene, "/Panel")
                                 .SetIntensity(8.0)
                                 .SetWidth(2.0)
                                 .SetHeight(0.5);
    EXPECT_NEAR(luminous("/Panel"), 8.0, 0.08);
    panel.SetNormalize(true);
    EXPECT_NEAR(luminous("/Panel"), 8.0, 0.08);
    panel.SetTransform(glm::scale(glm::dmat4(1.0), glm::dvec3(2.0)));
    EXPECT_NEAR(luminous("/Panel"), 2.0, 0.02)
        << "a quarter of the luminance on 4 times the area";

    // A sphere of radius 0.5: luminous intensity 8 pi 0.25 = 2 pi,
    // normalized over 4 pi 0.25.
    ernie::SphereLight bulb =
        ernie::SphereLight::Define(scene, "/Bulb")
            .SetIntensity(8.0)
            .SetRadius(0.5);
    EXPECT_NEAR(luminous("/Bulb"), 2.0 * M_PI, 0.07);
    bulb.SetNormalize(true);
    EXPECT_NEAR(luminous("/Bulb"), 2.0, 0.02);

    // A distant light's normalized intensity is its illuminance;
    // un-normalized, pi times it.
    ernie::DistantLight sun =
        ernie::DistantLight::Define(scene, "/Sun").SetIntensity(3.0);
    EXPECT_NEAR(luminous("/Sun"), 3.0 * M_PI, 0.1);
    sun.SetNormalize(true);
    EXPECT_NEAR(luminous("/Sun"), 3.0, 0.03);

    // Powers and photometric units are sizeless already.
    panel.SetPhotometricPower(1000.0);
    EXPECT_NEAR(luminous("/Panel"), 1000.0 / (4.0 * M_PI), 1.6)
        << "4 m^2 after the scale";
}

namespace {

/// A downlight's IES profile: 100 cd at the nadir falling as cos to 0
/// at 90 degrees, every 15.
constexpr const char* kDownlight = R"(IESNA:LM-63-2002
TILT=NONE
1 -1 1 7 1 1 2 0 0 0
1 1 10
0 15 30 45 60 75 90
0
100 96.592583 86.602540 70.710678 50 25.881905 0
)";

/// `text` in a file of the temporary directory named after `name`.
std::filesystem::path WriteTemporary(const std::string& name,
                                     const char* text) {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("ernie_session_lights_" + name + ".ies");
    std::ofstream(path) << text;
    return path;
}

}  // namespace

TEST(LightParamsOf, IesTiltDataScaleATiltedLuminaire) {
    // A vertical lamp losing a fifth of its output at 45 degrees.
    const std::filesystem::path file =
        WriteTemporary("tilted", R"(IESNA:LM-63-2002
TILT=INCLUDE
1
3
0 45 90
1 0.8 0.5
1 -1 1 2 1 1 2 0 0 0
1 1 10
0 90
0
100 0
)");
    const ernie::SpectralBands bands{380.0, 780.0, 40};
    Scene scene;
    ernie::SphereLight lamp =
        ernie::SphereLight::Define(scene, "/Lamp")
            .SetTreatAsPoint(true)
            .SetShapingIesFile(file.string());
    EXPECT_FLOAT_EQ(
        ernie::LightParamsOf(scene, "/Lamp", bands).profile[0],
        100.0f)
        << "hanging straight down";
    lamp.SetTransform(glm::rotate(glm::dmat4(1.0), glm::radians(45.0),
                                  glm::dvec3(1, 0, 0)));
    EXPECT_NEAR(
        ernie::LightParamsOf(scene, "/Lamp", bands).profile[0], 80.0f,
        1e-4f)
        << "tilted 45 degrees";
    std::filesystem::remove(file);
}

TEST(LightParamsOf, IesProfilesShapeLightsInEveryUnit) {
    const ernie::SpectralBands bands{380.0, 780.0, 40};
    const std::filesystem::path file =
        WriteTemporary("downlight", kDownlight);
    const double flux = ernie::ReadIes(file).Flux();
    Scene scene;
    const auto luminous = [&](std::string_view path) {
        return Luminous(
            ernie::LightParamsOf(scene, path, bands).spectrum, bands);
    };
    ernie::SphereLight lamp =
        ernie::SphereLight::Define(scene, "/Lamp")
            .SetTreatAsPoint(true)
            .SetIntensity(2.0)
            .SetShapingIesFile(file.string());
    const LightParams params =
        ernie::LightParamsOf(scene, "/Lamp", bands);
    constexpr int columns = LightParams::kProfileColumns;
    ASSERT_EQ(params.profile.size(),
              static_cast<std::size_t>(LightParams::kProfileRows *
                                       columns));
    EXPECT_FLOAT_EQ(params.profile[0], 100.0f) << "along the axis";
    EXPECT_NEAR(params.profile[32 * columns + 5], 70.710678f, 1e-4f)
        << "45 degrees off it";
    EXPECT_EQ(params.profile[100 * columns], 0.0f) << "behind it";

    // Its candela times the intensity; normalized, the flux of 2 cd
    // in every direction.
    EXPECT_NEAR(luminous("/Lamp"), 2.0, 0.02);
    lamp.SetShapingIesNormalize(true);
    EXPECT_NEAR(luminous("/Lamp"), 2.0 * 4.0 * M_PI / flux,
                0.01 * 2.0 * 4.0 * M_PI / flux);
    // In photometric units, 1000 lumens in all; or 50 lux at 2 m
    // along the axis.
    lamp.SetPhotometricPower(1000.0);
    EXPECT_NEAR(luminous("/Lamp"), 1000.0 / flux,
                0.01 * 1000.0 / flux);
    lamp.SetPhotometricIlluminance(50.0)
        .SetPhotometricIlluminanceDistance(2.0);
    EXPECT_NEAR(luminous("/Lamp"), 50.0 * 4.0 / 100.0, 0.02);

    // An angle scale of 0.5 squeezes the profile into half the angle:
    // 22.5 degrees (row 16) reads its 45, and 45 degrees its 90.
    lamp.SetShapingIesAngleScale(0.5);
    const std::vector<float> squeezed =
        ernie::LightParamsOf(scene, "/Lamp", bands).profile;
    EXPECT_NEAR(squeezed[16 * columns], 70.710678f, 1e-3f);
    EXPECT_NEAR(squeezed[32 * columns], 0.0f, 1e-3f);
    std::filesystem::remove(file);
}

namespace {

/// The flux per unit luminance of a Lambertian cylinder side `length`
/// long along x and `radius` wide, its emission times `g` of the
/// direction (a unit vector, the light's axis -z), by brute force
/// over its surface normals and the hemisphere each faces.
template <typename G>
double CylinderFlux(double length, double radius, const G& g) {
    constexpr int kAround = 96;
    constexpr int kTheta = 96;
    constexpr int kPhi = 192;
    const double dA = 2.0 * M_PI / kAround;
    const double dTheta = 0.5 * M_PI / kTheta;
    const double dPhi = 2.0 * M_PI / kPhi;
    double sum = 0.0;
    for (int a = 0; a < kAround; ++a) {
        const double angle = (a + 0.5) * dA;
        const glm::dvec3 n(0.0, std::cos(angle), std::sin(angle));
        const glm::dvec3 t1(1.0, 0.0, 0.0);
        const glm::dvec3 t2 = glm::cross(n, t1);
        for (int i = 0; i < kTheta; ++i) {
            const double theta = (i + 0.5) * dTheta;
            for (int j = 0; j < kPhi; ++j) {
                const double phi = (j + 0.5) * dPhi;
                const glm::dvec3 w =
                    std::sin(theta) *
                        (std::cos(phi) * t1 + std::sin(phi) * t2) +
                    std::cos(theta) * n;
                sum += std::cos(theta) * std::sin(theta) * g(w);
            }
        }
    }
    return sum * dTheta * dPhi * radius * dA * length;
}

}  // namespace

TEST(LightParamsOf,
     CylinderLightsTakeFocusAndProfilesKeepingTheirPower) {
    const ernie::SpectralBands bands{380.0, 780.0, 40};
    Scene scene;
    const auto luminous = [&](std::string_view path) {
        return Luminous(
            ernie::LightParamsOf(scene, path, bands).spectrum, bands);
    };
    // 2 m long and 0.25 m in radius: 2 r l = 1 m^2 seen side on.
    ernie::CylinderLight tube =
        ernie::CylinderLight::Define(scene, "/Tube")
            .SetLength(2.0)
            .SetRadius(0.25)
            .SetPhotometricPower(1000.0)
            .SetShapingFocus(2.0);
    const double focused =
        1000.0 / CylinderFlux(2.0, 0.25, [](const glm::dvec3& w) {
            return w.z * w.z;
        });
    EXPECT_NEAR(luminous("/Tube"), focused, 0.01 * focused)
        << "1000 lumens, focused";
    EXPECT_EQ(ernie::LightParamsOf(scene, "/Tube", bands).focus,
              2.0f);

    // A downlight profile, its candela over the tube's -z axis.
    const std::filesystem::path file =
        WriteTemporary("tube", kDownlight);
    const ernie::IesProfile profile = ernie::ReadIes(file);
    tube.ClearShapingFocus().SetShapingIesFile(file.string());
    const double profiled =
        1000.0 / CylinderFlux(2.0, 0.25, [&](const glm::dvec3& w) {
            return profile.Sample(glm::degrees(std::acos(-w.z)),
                                  glm::degrees(std::atan2(w.y, w.x)));
        });
    EXPECT_NEAR(luminous("/Tube"), profiled, 0.01 * profiled)
        << "1000 lumens, profiled";
    // In relative units, the candela times the intensity over 2 r l.
    tube.ClearPhotometricPower().SetIntensity(3.0);
    EXPECT_NEAR(luminous("/Tube"), 3.0, 0.03);
    std::filesystem::remove(file);
}

TEST(LightParamsOf, LaserPatternsAndTheirShadowCones) {
    const ernie::SpectralBands bands{380.0, 780.0, 16};
    Scene scene;
    ernie::LaserLight laser =
        ernie::LaserLight::Define(scene, "/Laser");
    EXPECT_TRUE(
        ernie::LightParamsOf(scene, "/Laser", bands).beam.IsSpot());

    // A 60 degree line: its half fan and profile, unbounded, and a
    // shadow cone over it.
    laser.SetPattern("line").SetFanAngle(60.0).SetLineProfile(
        "gaussian");
    const LightParams line =
        ernie::LightParamsOf(scene, "/Laser", bands);
    EXPECT_EQ(line.beam.pattern, "line");
    ASSERT_EQ(line.beam.patternParams.size(), 2u);
    EXPECT_NEAR(line.beam.patternParams[0], glm::radians(30.0f),
                1e-6f);
    EXPECT_EQ(line.beam.patternParams[1], 1.0f);
    EXPECT_EQ(line.beam.extent, glm::vec2(0.0f));
    EXPECT_GT(line.coneAngle, 30.0f);
    EXPECT_LT(line.coneAngle, 32.0f);

    // A 5 x 3 grid 4 degrees apart, 10% in its zero order: its cone
    // reaches the corner beamlets, atan(|(tan 8, tan 4)|).
    laser.SetPattern("grid")
        .SetGridColumns(5)
        .SetGridRows(3)
        .SetGridPitch({4.0, 4.0})
        .SetEfficiency(0.8)
        .SetZeroOrder(0.1);
    const LightParams grid =
        ernie::LightParamsOf(scene, "/Laser", bands);
    EXPECT_EQ(grid.beam.pattern, "grid");
    ASSERT_EQ(grid.beam.patternParams.size(), 4u);
    EXPECT_EQ(grid.beam.patternParams[0], 5.0f);
    EXPECT_EQ(grid.beam.patternParams[1], 3.0f);
    EXPECT_NEAR(grid.beam.patternParams[3], glm::radians(4.0f),
                1e-6f);
    EXPECT_FLOAT_EQ(grid.beam.efficiency, 0.8f);
    EXPECT_FLOAT_EQ(grid.beam.zeroOrder, 0.1f);
    const double corner = glm::degrees(std::atan(
        glm::length(glm::dvec2(std::tan(glm::radians(8.0)),
                               std::tan(glm::radians(4.0))))));
    EXPECT_GT(grid.coneAngle, corner);
    EXPECT_LT(grid.coneAngle, corner + 1.5);

    laser.SetZeroOrder(0.5);
    EXPECT_THROW(ernie::LightParamsOf(scene, "/Laser", bands),
                 std::invalid_argument)
        << "more than its power";
    // Any other name is a custom pattern, for the renderer to find,
    // with its parameters and data as given.
    laser.SetZeroOrder(0.0)
        .SetPattern("rings")
        .SetPatternParams({0.5f, 2.0f})
        .SetPatternData({1.0f, 2.0f, 3.0f});
    const LightParams rings =
        ernie::LightParamsOf(scene, "/Laser", bands);
    EXPECT_EQ(rings.beam.pattern, "rings");
    EXPECT_EQ(rings.beam.patternParams,
              (std::vector<float>{0.5f, 2.0f}));
    EXPECT_EQ(rings.beam.patternData,
              (std::vector<float>{1.0f, 2.0f, 3.0f}));
    EXPECT_EQ(rings.beam.patternImage, nullptr);
    EXPECT_FLOAT_EQ(rings.beam.wavelength, 532.0f);

    // Dots: per tile, tiles, jitter, the field's half tangents and
    // the seed; unbounded, as each dot's beam carries its share.
    laser.SetPattern("dots")
        .SetDotColumns(8)
        .SetDotRows(6)
        .SetDotTileColumns(3)
        .SetDotTileRows(3)
        .SetDotSeed(7);
    const LightParams dots =
        ernie::LightParamsOf(scene, "/Laser", bands);
    ASSERT_EQ(dots.beam.patternParams.size(), 8u);
    EXPECT_EQ(dots.beam.patternParams[0], 8.0f);
    EXPECT_EQ(dots.beam.patternParams[3], 3.0f);
    EXPECT_FLOAT_EQ(dots.beam.patternParams[4], 0.9f);
    EXPECT_NEAR(dots.beam.patternParams[5],
                std::tan(glm::radians(20.0)), 1e-6);
    EXPECT_EQ(dots.beam.patternParams[7], 7.0f);
    EXPECT_EQ(dots.beam.extent, glm::vec2(0.0f));
    EXPECT_GT(dots.coneAngle, 24.0f)
        << "reaching the field's corners";
    laser.SetDotJitter(1.5);
    EXPECT_THROW(ernie::LightParamsOf(scene, "/Laser", bands),
                 std::invalid_argument);
    laser.SetDotJitter(0.9);

    // An image needs its file; over 40 x 30 degrees it reaches
    // tan 20 and tan 15 across the tangent plane.
    laser.SetPattern("image");
    EXPECT_THROW(ernie::LightParamsOf(scene, "/Laser", bands),
                 std::invalid_argument)
        << "no image";
    const std::filesystem::path file =
        std::filesystem::temp_directory_path() /
        "ernie_light_pattern_test.png";
    const std::array<std::uint8_t, 8> grey = {51,  51,  51,  51,
                                              153, 153, 153, 153};
    ASSERT_NE(stbi_write_png(file.string().c_str(), 4, 2, 1,
                             grey.data(), 4),
              0);
    laser.SetPatternImage(file.string());
    const LightParams image =
        ernie::LightParamsOf(scene, "/Laser", bands);
    ASSERT_NE(image.beam.patternImage, nullptr);
    EXPECT_EQ(image.beam.patternImage->width, 4u);
    EXPECT_EQ(image.beam.patternImage->height, 2u);
    // Linear, 0.2 above 0.6; then 2 x 1 and 1 x 1 levels of 0.4.
    const std::vector<float> texels = {0.2f, 0.2f, 0.2f, 0.2f,
                                       0.6f, 0.6f, 0.6f, 0.6f,
                                       0.4f, 0.4f, 0.4f};
    ASSERT_EQ(image.beam.patternImage->texels.size(), texels.size());
    for (std::size_t i = 0; i < texels.size(); ++i) {
        EXPECT_NEAR(image.beam.patternImage->texels[i], texels[i],
                    1e-6f)
            << "texel " << i;
    }
    EXPECT_NEAR(image.beam.extent.x, std::tan(glm::radians(20.0)),
                1e-6);
    EXPECT_NEAR(image.beam.extent.y, std::tan(glm::radians(15.0)),
                1e-6);
    EXPECT_EQ(image.beam.patternImage,
              ernie::LightParamsOf(scene, "/Laser", bands)
                  .beam.patternImage)
        << "read once";
    std::filesystem::remove(file);
}

TEST(LightParamsOf, LasersCarryTheirBeamAndLine) {
    // 5 mW at 532 nm, 0.1 nm wide, in bands of 25 nm from 380.
    const ernie::SpectralBands bands{380.0, 780.0, 16};
    Scene scene;
    ernie::LaserLight laser =
        ernie::LaserLight::Define(scene, "/Laser")
            .SetPower(0.005)
            .SetWaistRadius({0.001, 0.002})
            .SetBeamQuality({1.0, 2.0})
            .SetWaistDistance(3.0);
    const LightParams params =
        ernie::LightParamsOf(scene, "/Laser", bands);
    EXPECT_EQ(params.type, ernie::LightType::Laser);
    // All its power in the 530-555 nm band, as W/nm.
    double total = 0.0;
    for (int i = 0; i < bands.count; ++i) {
        total += params.spectrum[static_cast<std::size_t>(i)] *
                 bands.Width();
    }
    EXPECT_NEAR(total, 0.005, 1e-8);
    EXPECT_NEAR(params.spectrum[6], 0.005 / 25.0, 1e-9)
        << "530 to 555 nm";
    EXPECT_EQ(params.spectrum[5], 0.0f);
    // z_R = pi w0^2 / (M^2 lambda).
    EXPECT_FLOAT_EQ(params.beam.waist.x, 0.001f);
    EXPECT_NEAR(params.beam.rayleigh.x, M_PI * 1e-6 / 532e-9, 1e-4);
    EXPECT_NEAR(params.beam.rayleigh.y, M_PI * 4e-6 / (2.0 * 532e-9),
                1e-4);
    EXPECT_FLOAT_EQ(params.beam.waistDistance, 3.0f);
    EXPECT_NEAR(laser.Divergence().x, 532e-9 / (M_PI * 0.001), 1e-9)
        << "theta = M^2 lambda / (pi w0)";

    // A line straddling two bands splits between them by its
    // Gaussian: centred on 530 nm, half below, half above.
    laser.SetWavelength(530.0).SetLinewidth(2.0);
    const std::vector<float> split =
        ernie::LightParamsOf(scene, "/Laser", bands).spectrum;
    EXPECT_NEAR(split[5] * 25.0, 0.0025, 1e-7);
    EXPECT_NEAR(split[6] * 25.0, 0.0025, 1e-7);

    // An aperture as wide as the beam there (1/e^2 radii 1 and 2 mm
    // at the waist, which here is the aperture) passes
    // 1 - exp(-2 a^2 / (w_x w_y)) of its power.
    laser.SetWaistDistance(0.0).SetApertureRadius(
        std::sqrt(0.001 * 0.002));
    const std::vector<float> clipped =
        ernie::LightParamsOf(scene, "/Laser", bands).spectrum;
    EXPECT_NEAR((clipped[5] + clipped[6]) * 25.0,
                0.005 * (1.0 - std::exp(-2.0)), 1e-7);
    EXPECT_EQ(ernie::ResolveRenderPrims(scene, {}).lights,
              std::vector<Path>{Path("/Laser")});

    laser.SetWaistRadius({0.0, 0.001});
    EXPECT_THROW(ernie::LightParamsOf(scene, "/Laser", bands),
                 std::invalid_argument);
}

TEST(LightParamsOf, CylinderLightsEmitFromTheirSides) {
    const ernie::SpectralBands bands{380.0, 780.0, 40};
    Scene scene;
    const auto luminous = [&](std::string_view path) {
        return Luminous(
            ernie::LightParamsOf(scene, path, bands).spectrum, bands);
    };
    // 2 m long, 0.25 m in radius, scaled up 2 along its length.
    ernie::CylinderLight tube =
        ernie::CylinderLight::Define(scene, "/Tube")
            .SetLength(1.0)
            .SetRadius(0.25)
            .SetIntensity(5.0)
            .SetTransform(
                glm::scale(glm::dmat4(1.0), glm::dvec3(2, 1, 1)));
    const LightParams params =
        ernie::LightParamsOf(scene, "/Tube", bands);
    EXPECT_EQ(params.type, ernie::LightType::Cylinder);
    EXPECT_FLOAT_EQ(params.size.x, 2.0f);
    EXPECT_FLOAT_EQ(params.radius, 0.25f);
    EXPECT_NEAR(params.tangent.x, 1.0f, 1e-6f) << "along its x axis";
    EXPECT_NEAR(luminous("/Tube"), 5.0, 0.05) << "its luminance";
    EXPECT_EQ(ernie::ResolveRenderPrims(scene, {}).lights,
              std::vector<Path>{Path("/Tube")});

    // Normalized, over its side's area, 2 pi r l = pi m^2.
    tube.SetNormalize(true);
    EXPECT_NEAR(luminous("/Tube"), 5.0 / M_PI, 0.05 / M_PI);
    // 1000 lumens from its side: over pi times its area; or 50 lux 3
    // m from its middle, through its 2 m by 0.5 m silhouette.
    tube.SetPhotometricPower(1000.0);
    EXPECT_NEAR(luminous("/Tube"), 1000.0 / (M_PI * M_PI),
                0.01 * 1000.0 / (M_PI * M_PI));
    tube.SetPhotometricIlluminance(50.0)
        .SetPhotometricIlluminanceDistance(3.0);
    const double luminance =
        50.0 / OnAxisIlluminancePerLuminance(
                   {1.0, 0.25}, 3.0, [](glm::dvec2) { return true; });
    EXPECT_NEAR(luminous("/Tube"), luminance, 0.01 * luminance);
}

TEST(LightParamsOf, FocusedLightsKeepTheirPower) {
    const ernie::SpectralBands bands{380.0, 780.0, 40};
    Scene scene;
    const auto luminous = [&](std::string_view path) {
        return Luminous(
            ernie::LightParamsOf(scene, path, bands).spectrum, bands);
    };

    // |cos|^4 over the sphere integrates to 4 pi / 5: 1600 lumens
    // give 5 times 1600 / 4 pi cd along the axis.
    ernie::SphereLight bulb =
        ernie::SphereLight::Define(scene, "/Bulb")
            .SetPhotometricPower(1600.0)
            .SetShapingFocus(4.0);
    EXPECT_NEAR(luminous("/Bulb"), 5.0 * 1600.0 / (4.0 * M_PI),
                0.015 * 5.0 * 1600.0 / (4.0 * M_PI));
    const LightParams focused =
        ernie::LightParamsOf(scene, "/Bulb", bands);
    EXPECT_EQ(focused.focus, 4.0f);
    EXPECT_TRUE(focused.focusTint.empty()) << "black";

    // A white tint undoes it; a grey one halfway.
    bulb.SetShapingFocusTint({1.0, 1.0, 1.0});
    EXPECT_NEAR(luminous("/Bulb"), 1600.0 / (4.0 * M_PI),
                0.015 * 1600.0 / (4.0 * M_PI));
    bulb.SetShapingFocusTint({0.5, 0.5, 0.5});
    const LightParams grey =
        ernie::LightParamsOf(scene, "/Bulb", bands);
    ASSERT_EQ(grey.focusTint.size(), 40u);
    EXPECT_NEAR(grey.focusTint[20], 0.5f, 0.02f);
    const double half = 1600.0 / (4.0 * M_PI * (0.5 + 0.5 / 5.0));
    EXPECT_NEAR(luminous("/Bulb"), half, 0.015 * half);

    // A panel's cos x cos^2 over the hemisphere integrates to pi / 2:
    // 1000 lumens from 2 m^2 give 1000 / pi nits along its axis.
    ernie::RectLight panel = ernie::RectLight::Define(scene, "/Panel")
                                 .SetWidth(2.0)
                                 .SetHeight(1.0)
                                 .SetPhotometricPower(1000.0)
                                 .SetShapingFocus(2.0);
    EXPECT_NEAR(luminous("/Panel"), 1000.0 / M_PI,
                0.015 * 1000.0 / M_PI);
    panel.SetPhotometricIlluminance(50.0)
        .SetPhotometricIlluminanceDistance(3.0);
    panel.ClearShapingFocus();
    const double unfocused = luminous("/Panel");
    panel.SetShapingFocus(2.0);
    EXPECT_NEAR(luminous("/Panel"), unfocused, 1e-6 * unfocused)
        << "on its axis, focus is moot";
}

TEST(LightParamsOf, IlluminantsShapeTheSpectrum) {
    const ernie::SpectralBands bands{380.0, 780.0, 40};
    Scene scene;
    ernie::DistantLight sun =
        ernie::DistantLight::Define(scene, "/Sun")
            .SetPhotometricIlluminance(1000.0);
    const auto shape = [&](std::string_view illuminant) {
        sun.SetPhysicalIlluminant(illuminant);
        return ernie::LightParamsOf(scene, "/Sun", bands).spectrum;
    };
    const glm::dvec2 white =
        Chromaticity(shape(tokens::kWhite), bands);
    EXPECT_NEAR(white.x, 1.0 / 3.0, 0.005)
        << "flat: the equal-energy point";
    sun.SetColorTemperature(6504.0);
    const glm::dvec2 daylight =
        Chromaticity(shape(tokens::kIlluminantD), bands);
    EXPECT_NEAR(daylight.x, 0.3127, 0.005);
    EXPECT_NEAR(daylight.y, 0.3290, 0.005);
    sun.SetColorTemperature(3000.0);
    const glm::dvec2 warm =
        Chromaticity(shape(tokens::kBlackbody), bands);
    EXPECT_NEAR(warm.x, 0.4369, 0.005)
        << "the Planckian locus at 3000 K";
    EXPECT_THROW(shape(tokens::kIlluminantD), std::invalid_argument)
        << "below 4000 K";

    sun.SetPhysicalCustomIlluminant({540.0f, 1.0f, 560.0f, 1.0f});
    const std::vector<float> custom = shape(tokens::kCustom);
    EXPECT_EQ(custom.front(), 0.0f) << "nothing outside the pairs";
    EXPECT_GT(custom[17], 0.0f) << "the 550 nm band";
    EXPECT_NEAR(Luminous(custom, bands), 1000.0, 10.0)
        << "still 1000 lux";
    EXPECT_THROW(shape("candle"), std::invalid_argument);

    // White and black body light spans 380 to 780 nm, as Omniverse's;
    // unset, the whole range.
    const ernie::SpectralBands wide{360.0, 840.0, 48};
    for (const std::string_view illuminant :
         {tokens::kWhite, tokens::kBlackbody}) {
        sun.SetPhysicalIlluminant(illuminant);
        const std::vector<float> spectrum =
            ernie::LightParamsOf(scene, "/Sun", wide).spectrum;
        EXPECT_EQ(spectrum[1], 0.0f)
            << illuminant << ": 370 to 380 nm";
        EXPECT_GT(spectrum[2], 0.0f)
            << illuminant << ": 380 to 390 nm";
        EXPECT_GT(spectrum[41], 0.0f)
            << illuminant << ": 770 to 780 nm";
        EXPECT_EQ(spectrum[42], 0.0f)
            << illuminant << ": 780 to 790 nm";
        EXPECT_NEAR(Luminous(spectrum, wide), 1000.0, 10.0)
            << illuminant;
    }
    sun.ClearPhysicalIlluminant();
    const std::vector<float> flat =
        ernie::LightParamsOf(scene, "/Sun", wide).spectrum;
    EXPECT_GT(flat.front(), 0.0f);
    EXPECT_GT(flat.back(), 0.0f);

    // Without photometric units, an illuminant only shapes the
    // spectrum.
    Scene relative;
    ernie::DistantLight::Define(relative, "/Sun")
        .SetIntensity(2.0)
        .SetPhysicalIlluminant(tokens::kBlackbody)
        .SetColorTemperature(3000.0);
    const std::vector<float> spectrum =
        ernie::LightParamsOf(relative, "/Sun", bands).spectrum;
    EXPECT_NEAR(Luminous(spectrum, bands), 2.0 * M_PI, 0.1)
        << "pi x intensity lux, as before";
    EXPECT_NEAR(Chromaticity(spectrum, bands).x, 0.4369, 0.005);
}

TEST(StarsOf, StarsShineByTheirMagnitudeAndColour) {
    Scene scene;
    scene.Define("/Sky", tokens::kStarField)
        .Set(tokens::kStarsRightAscension,
             std::vector<float>{0.0f, 90.0f, 180.0f})
        .Set(tokens::kStarsDeclination,
             std::vector<float>{0.0f, 45.0f, -30.0f})
        .Set(tokens::kStarsMagnitude,
             std::vector<float>{0.0f, 5.0f, 9.0f})
        .Set(tokens::kStarsColorIndex,
             std::vector<float>{0.65f, 0.0f, 1.5f});
    std::vector<ernie::Star> stars = ernie::StarsOf(scene, "/Sky");
    ASSERT_EQ(stars.size(), 3u);
    EXPECT_NEAR(stars[1].rightAscension, M_PI / 2.0, 1e-6);
    EXPECT_NEAR(stars[1].declination, M_PI / 4.0, 1e-6);
    EXPECT_NEAR(stars[0].temperature, 5778.0f, 5.0f) << "B-V 0.65";
    EXPECT_NEAR(stars[0].irradiance555,
                ernie::BlackbodySpectralIrradiance(
                    555.0, stars[0].temperature, 2.54e-6),
                1e-15);
    EXPECT_GT(stars[2].temperature, 0.0f);
    EXPECT_LT(stars[2].temperature, stars[1].temperature)
        << "redder is cooler";

    ernie::StarField sky(scene, "/Sky");
    sky.SetMagnitudeLimit(6.0);
    EXPECT_EQ(ernie::StarsOf(scene, "/Sky").size(), 2u)
        << "fainter than the limit";
    sky.SetExposure(1.0);
    EXPECT_NEAR(ernie::StarsOf(scene, "/Sky")[0].irradiance555 /
                    stars[0].irradiance555,
                2.0, 1e-5)
        << "2^exposure";

    scene.Set("/Sky", tokens::kStarsDeclination,
              std::vector<float>{0.0f});
    EXPECT_THROW(ernie::StarsOf(scene, "/Sky"), std::invalid_argument)
        << "lengths differ";
    Scene directions;
    directions.Define("/Sky", tokens::kStarField)
        .Set(tokens::kStarsDirection,
             std::vector<float>{0.0f, 0.0f, 2.0f})
        .Set(tokens::kStarsMagnitude, std::vector<float>{1.0f});
    EXPECT_NEAR(ernie::StarsOf(directions, "/Sky")[0].declination,
                M_PI / 2.0, 1e-6);
    EXPECT_NEAR(ernie::StarsOf(directions, "/Sky")[0].temperature,
                5772.0f, 1e-3f)
        << "the Sun's without a colour";
    directions.Define("/Bare", tokens::kStarField)
        .Set(tokens::kStarsMagnitude, std::vector<float>{1.0f});
    EXPECT_THROW(ernie::StarsOf(directions, "/Bare"),
                 std::invalid_argument)
        << "no places";
}

TEST(FrameParamsOf, DomesShowAndStarFieldsTurnWithTheirTransforms) {
    Scene scene;
    ernie::Camera::Define(scene, "/Camera");
    ernie::DomeLight dome = ernie::DomeLight::Define(scene, "/Dome");
    scene.Set("/Dome", tokens::kTransform,
              glm::rotate(glm::dmat4(1.0), M_PI / 2.0,
                          glm::dvec3(0, 0, 1)));
    scene.Define("/Stars", tokens::kStarField)
        .Set(tokens::kTransform,
             glm::rotate(glm::dmat4(1.0), M_PI / 2.0,
                         glm::dvec3(1, 0, 0)));
    const ernie::RenderPrims prims =
        ernie::ResolveRenderPrims(scene, {});
    EXPECT_EQ(prims.stars, Path("/Stars"));
    ernie::FrameParams params =
        ernie::FrameParamsOf(scene, prims, 16, 16);
    ASSERT_TRUE(params.dome && params.stars);
    EXPECT_TRUE(params.dome->visible);
    // World +y is the dome's +x after a quarter turn about z.
    const glm::vec3 local =
        params.dome->toDome * glm::vec3(0.0f, 1.0f, 0.0f);
    EXPECT_NEAR(local.x, 1.0f, 1e-6f);
    // World +y is the catalogue's -z after a quarter turn about x.
    EXPECT_NEAR(
        (params.stars->toCatalogue * glm::vec3(0.0f, 1.0f, 0.0f)).z,
        -1.0f, 1e-6f);
    dome.SetVisibleInPrimaryRay(false);
    EXPECT_FALSE(
        ernie::FrameParamsOf(scene, prims, 16, 16).dome->visible);
}

// The "sun" illuminant: the measured solar spectrum per band, scaled
// so its whole range carries the irradiance; at Mars's mean 1.524 au,
// 1361 / 1.524^2 W/m^2 takes the same shape.
TEST(SunIrradianceOf, TheSunIlluminantIsTheMeasuredSolarSpectrum) {
    Scene scene;
    ernie::DistantLight::Define(scene, "/Sun")
        .SetPhysicalIlluminant(std::string(tokens::kSun))
        .SetIrradiance(ernie::TotalSolarIrradiance());
    const ernie::SpectralBands bands{380.0, 780.0, 16};
    const std::vector<float> irradiance =
        ernie::SunIrradianceOf(scene, "/Sun", bands);
    const std::vector<float> measured =
        ernie::SolarSpectralIrradiance().Bin(bands);
    ASSERT_EQ(irradiance.size(), measured.size());
    for (std::size_t b = 0; b < irradiance.size(); ++b) {
        EXPECT_NEAR(irradiance[b] / measured[b], 1.0f, 1e-6f);
    }
    // 380-405 nm, below a black body's share: the calcium and Balmer
    // lines (1.2107 W m^-2 nm^-1 at 1 au).
    EXPECT_NEAR(irradiance[0], 1.2107f, 1e-3f);

    ernie::DistantLight(scene, "/Sun")
        .SetIrradiance(1361.0 / (1.524 * 1.524));
    const std::vector<float> mars =
        ernie::SunIrradianceOf(scene, "/Sun", bands);
    EXPECT_NEAR(mars[8] / irradiance[8],
                1361.0 / (1.524 * 1.524) /
                    ernie::TotalSolarIrradiance(),
                1e-6);
}
