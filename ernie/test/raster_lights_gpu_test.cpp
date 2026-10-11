// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/offscreen_target.h"
#include "ernie/geometry/primitives.h"
#include "ernie/raster/delegate.h"
#include "ernie/scene/camera.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/orbit_controller.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/terrain_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/wide_view.h"
#include "ernie/session/renderer.h"
#include "ernie/session/resolve.h"
#include "ernie/util/astronomy.h"
#include "ernie/util/ies.h"
#include "support/delegate_scenes.h"
#include "support/test_raster.h"
#include "support/vulkan_fixture.h"

#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <stb_image_write.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using ernie::DeviceKind;
using ernie::DType;
using ernie::OrbitController;
using ernie::RasterDelegate;
using ernie::Scene;
using ernie::Tensor;
using namespace ernie::testing;
namespace tokens = ernie::tokens;

namespace {

/// FlatScene lit only by a point lamp `height` above the origin
/// pointing down, with a radiant flux of 1000 W over a flat spectrum
/// and a cone of `coneAngle` degrees (180: unshaped).
Scene LampScene(double height, float coneAngle = 180.0f) {
    Scene scene = FlatScene();
    scene.Remove("/World/Sun");
    ernie::SphereLight::Define(scene, "/World/Lamp")
        .SetTreatAsPoint(true)
        .SetPower(1000.0)
        .SetShapingConeAngle(coneAngle);
    scene.Set("/World/Lamp", tokens::kTransform,
              ernie::LookAtTransform({0.0, 0.0, height},
                                     {0.0, 0.0, 0.0},
                                     {0.0, 1.0, 0.0}));
    return scene;
}

float RadianceAt(RasterDelegate& delegate, const Scene& scene,
                 float x, float y) {
    ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
    const Tensor spectral = outputs.at("/Render/Product")
                                .vars.at("SpectralRadiance")
                                .Map(ernie::MapDevice::Cpu)
                                .tensor;
    const auto [row, col] = PixelOf(scene, x, y);
    return static_cast<const float*>(
        spectral
            .Data())[8 * kVarSize * kVarSize + row * kVarSize + col];
}

}  // namespace

TEST_F(RasterDelegateTest, LampsLightByTheInverseSquareLaw) {
    RasterDelegate delegate(*context_);
    const Scene near = LampScene(10.0);
    const float intensity =
        static_cast<float>(1000.0 / (4.0 * M_PI) / 470.0);
    const float x = 3.0f;
    const float distance2 = 100.0f + x * x;
    const float expected =
        intensity * (10.0f / std::sqrt(distance2)) / distance2 *
        0.5f / static_cast<float>(M_PI);
    EXPECT_NEAR(RadianceAt(delegate, near, x, 0.0f), expected,
                0.04f * expected)
        << "L = I cos(i) / d^2 * albedo / pi";
    EXPECT_FALSE(delegate.Prims().lights.empty());

    const float twice =
        RadianceAt(delegate, LampScene(20.0), 0.0f, 0.0f);
    const float once =
        RadianceAt(delegate, LampScene(10.0), 0.0f, 0.0f);
    EXPECT_NEAR(once / twice, 4.0f, 0.1f);

    // The 1000 W go into the cone: 4 pi over its solid angle brighter
    // inside it.
    const Scene spot = LampScene(10.0, 10.0f);
    const float concentrated =
        once * static_cast<float>(
                   2.0 / (1.0 - std::cos(glm::radians(10.0))));
    EXPECT_NEAR(RadianceAt(delegate, spot, 0.0f, 0.0f), concentrated,
                0.02f * concentrated)
        << "inside the cone";
    EXPECT_EQ(RadianceAt(delegate, spot, 6.0f, 0.0f), 0.0f)
        << "outside a 10 degree cone";
}

TEST_F(RasterDelegateTest, IesProfilesShapeLamps) {
    // Every 15 degrees: 100 cd x cos within 30 degrees of the 0
    // degree plane (the lamp's +x, world +x here) and (1 + 2 sin)
    // times that within 30 of the 90 degree plane (+y), mirrored by
    // quadrant symmetry, so the coarse pixels' bearings matter
    // little; the planes agree at the nadir.
    const char* text = R"(IESNA:LM-63-2002
TILT=NONE
1 -1 1 7 4 1 2 0 0 0
1 1 10
0 15 30 45 60 75 90
0 30 60 90
100 96.592583 86.602540 70.710678 50 25.881905 0
100 96.592583 86.602540 70.710678 50 25.881905 0
100 146.592583 173.205081 170.710678 136.602540 75.881905 0
100 146.592583 173.205081 170.710678 136.602540 75.881905 0
)";
    // The reference: 100 cd every way, so each pixel's ratio to it is
    // the profile there over 100.
    const char* uniform = R"(IESNA:LM-63-2002
TILT=NONE
1 -1 1 2 1 1 2 0 0 0
1 1 10
0 180
0
100 100
)";
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path();
    const std::filesystem::path file =
        directory / "ernie_raster_lights_quadrant.ies";
    const std::filesystem::path flat =
        directory / "ernie_raster_lights_uniform.ies";
    std::ofstream(file) << text;
    std::ofstream(flat) << uniform;
    const ernie::IesProfile profile = ernie::ParseIes(text);

    RasterDelegate delegate(*context_);
    const float x = 6.0f;
    // Each pixel's angle off the lamp's axis, from the unshaped lamp:
    // E ~ cos^3.
    const float below =
        RadianceAt(delegate, LampScene(10.0), 0.0f, 0.0f);
    const auto angle = [&](float px, float py) {
        const double c3 =
            RadianceAt(delegate, LampScene(10.0), px, py) / below;
        return glm::degrees(std::acos(std::cbrt(c3)));
    };
    // Relative units: the candela times the intensity.
    const auto shaped = [&](const std::filesystem::path& ies) {
        Scene scene = LampScene(10.0);
        ernie::SphereLight(scene, "/World/Lamp")
            .ClearPower()
            .SetIntensity(1.0)
            .SetShapingIesFile(ies.string());
        return scene;
    };
    const Scene quadrant = shaped(file);
    const Scene reference = shaped(flat);
    const auto ratio = [&](float px, float py) {
        return RadianceAt(delegate, quadrant, px, py) /
               RadianceAt(delegate, reference, px, py);
    };
    const double onX = profile.Sample(angle(x, 0.0f), 0.0) / 100.0;
    EXPECT_NEAR(ratio(x, 0.0f), onX, 0.02 * onX)
        << "the 0 degree plane";
    const double onY = profile.Sample(angle(0.0f, x), 90.0) / 100.0;
    EXPECT_NEAR(ratio(0.0f, x), onY, 0.02 * onY)
        << "the 90 degree plane, brighter";
    std::filesystem::remove(file);
    std::filesystem::remove(flat);
}

TEST_F(RasterDelegateTest, FocusedLampsFallOffAsAPowerOfTheCosine) {
    RasterDelegate delegate(*context_);
    const float x = 6.0f;
    const float plain =
        RadianceAt(delegate, LampScene(10.0), x, 0.0f);
    const float below =
        RadianceAt(delegate, LampScene(10.0), 0.0f, 0.0f);
    // On the floor E = I cos(i) / d^2 = I c^3 / h^2, c the cosine off
    // the lamp's axis, so the ratio gives c^3 at the pixel sampled.
    const float c3 = plain / below;

    // |cos|^3 off the axis, its 1000 W kept: over the sphere |cos|^3
    // integrates to pi, so it is 4 times brighter along the axis.
    Scene scene = LampScene(10.0);
    ernie::SphereLight(scene, "/World/Lamp").SetShapingFocus(3.0);
    EXPECT_NEAR(RadianceAt(delegate, scene, 0.0f, 0.0f), 4.0f * below,
                0.03f * 4.0f * below)
        << "along the axis";
    const float off = 4.0f * plain * c3;
    EXPECT_NEAR(RadianceAt(delegate, scene, x, 0.0f), off,
                0.03f * off)
        << "off the axis";

    // A white tint undoes the focus.
    ernie::SphereLight(scene, "/World/Lamp")
        .SetShapingFocusTint({1.0, 1.0, 1.0});
    EXPECT_NEAR(RadianceAt(delegate, scene, x, 0.0f), plain,
                0.03f * plain)
        << "white tint";
}

TEST_F(RasterDelegateTest, LampsCastShadowsFromSpotAndCubeMaps) {
    RasterDelegate delegate(*context_);
    for (const float cone : {60.0f, 180.0f}) {
        Scene scene = LampScene(10.0, cone);
        const float lit = RadianceAt(delegate, scene, 3.0f, 0.0f);
        DefineQuad(scene, "/World/Blocker");
        scene.Set("/World/Blocker", tokens::kTransform,
                  glm::translate(glm::dmat4(1.0),
                                 glm::dvec3(0.0, 0.0, 5.0)));
        EXPECT_LT(RadianceAt(delegate, scene, 3.0f, 0.0f), 0.1f * lit)
            << "behind the blocker, cone " << cone;
        EXPECT_NEAR(
            RadianceAt(delegate, scene, 6.0f, 6.0f),
            RadianceAt(delegate, LampScene(10.0, cone), 6.0f, 6.0f),
            0.05f * lit)
            << "beside the shadow, cone " << cone;
        ernie::SphereLight(scene, "/World/Lamp")
            .SetShadowEnable(false);
        EXPECT_NEAR(RadianceAt(delegate, scene, 3.0f, 0.0f), lit,
                    0.02f * lit);
    }
    EXPECT_EQ(delegate.Renderer().LightShadowViews().size(), 0u)
        << "shadows off";
}

TEST_F(RasterDelegateTest, OtherDistantLightsAddToTheSun) {
    Scene scene = FlatScene();
    ernie::DistantLight::Define(scene, "/World/Zenith")
        .SetIntensity(0.5);
    scene.Set("/World/Zenith", tokens::kTransform,
              ernie::DistantLightTransform({0.0, 0.0, 1.0}));
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
    ernie::DistantLight sun(scene, "/World/Sun");
    sun.SetShadowEnable(true);
    scene.Set("/World/Sun", tokens::kTransform,
              ernie::DistantLightTransform({0.3, 0.0, 1.0}));
    DefineQuad(scene, "/World/Blocker", 4.0f);
    scene.Set(
        "/World/Blocker", tokens::kTransform,
        glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 0.0, 30.0)));
    RasterDelegate delegate(*context_);
    const auto partial = [&](float angle) {
        sun.SetAngle(angle);
        ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
        const Tensor hdr = outputs.at("/Render/Product")
                               .vars.at("HdrColor")
                               .Map(ernie::MapDevice::Cpu)
                               .tensor;
        const Tensor ids = outputs.at("/Render/Product")
                               .vars.at("PrimId")
                               .Map(ernie::MapDevice::Cpu)
                               .tensor;
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

/// FlatScene lit only by an area light of `type` 10 m above the
/// origin facing down, with a luminance of 1000 cd/m^2.
Scene AreaLightScene(std::string_view type) {
    Scene scene = FlatScene();
    scene.Remove("/World/Sun");
    scene.Define("/World/Panel", type)
        .Set(tokens::kTransform,
             ernie::LookAtTransform({0.0, 0.0, 10.0}, {0.0, 0.0, 0.0},
                                    {0.0, 1.0, 0.0}));
    ernie::Light(scene, "/World/Panel")
        .SetIntensity(1000.0)
        .SetShadowEnable(false);
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

/// Form factor from a point to a parallel rectangle `a` x `b` whose
/// corner lies `h` above it.
double CornerFormFactor(double a, double b, double h) {
    const double x = a / h;
    const double y = b / h;
    const double sx = std::sqrt(1.0 + x * x);
    const double sy = std::sqrt(1.0 + y * y);
    return (x / sx * std::atan(y / sx) + y / sy * std::atan(x / sy)) /
           (2.0 * M_PI);
}

/// Irradiance per unit radiance on the ground (normal +z) at `q` from
/// the side of a Lambertian cylinder `length` long along x and
/// `radius` wide, its axis `height` above the origin: the integral of
/// cos cos / d^2 over the part of its surface that faces `q`, by
/// brute force.
double CylinderIrradiance(double length, double radius, double height,
                          const glm::dvec3& q) {
    constexpr int kAround = 720;
    constexpr int kAlong = 800;
    const double dPhi = 2.0 * M_PI / kAround;
    const double dT = length / kAlong;
    double sum = 0.0;
    for (int i = 0; i < kAround; ++i) {
        const double phi = (i + 0.5) * dPhi;
        const glm::dvec3 normal(0.0, std::cos(phi), std::sin(phi));
        for (int j = 0; j < kAlong; ++j) {
            const glm::dvec3 p =
                glm::dvec3(-0.5 * length + (j + 0.5) * dT, 0.0,
                           height) +
                radius * normal;
            const glm::dvec3 d = q - p;
            const double d2 = glm::dot(d, d);
            const double out = glm::dot(normal, d);
            const double in = -d.z;
            if (out > 0.0 && in > 0.0) {
                sum += out * in / (d2 * d2);
            }
        }
    }
    return sum * radius * dPhi * dT;
}

/// The ground point (z = 0) at the centre of FlatScene's var pixel
/// holding (`x`, `y`).
glm::dvec3 PixelCentre(const Scene& scene, float x, float y) {
    const auto [row, col] = PixelOf(scene, x, y);
    const glm::mat4 inverse = glm::inverse(
        ernie::CameraParamsOf(scene, "/World/Camera", kSize, kSize)
            .RasterView()
            .ViewProjection());
    const glm::vec2 ndc(
        (static_cast<float>(col) + 0.5f) / kVarSize * 2.0f - 1.0f,
        1.0f - (static_cast<float>(row) + 0.5f) / kVarSize * 2.0f);
    glm::vec4 near = inverse * glm::vec4(ndc, 0.0f, 1.0f);
    glm::vec4 far = inverse * glm::vec4(ndc, 1.0f, 1.0f);
    near /= near.w;
    far /= far.w;
    const float t = near.z / (near.z - far.z);
    return glm::dvec3(glm::vec3(near + t * (far - near)));
}

/// FlatScene lit only by a laser 10 m above the origin pointing down:
/// 1 W of a line at the centre of the camera's band 8 (so all of it
/// there, `*bandWidth` nm wide), its waist `waist` m at the aperture.
Scene LaserScene(double waist, double* bandWidth) {
    Scene scene = FlatScene();
    scene.Remove("/World/Sun");
    const ernie::SpectralBands bands =
        ernie::CameraParamsOf(scene, "/World/Camera", kVarSize,
                              kVarSize)
            .bands;
    *bandWidth = bands.Width();
    ernie::LaserLight::Define(scene, "/World/Laser")
        .SetPower(1.0)
        .SetWavelength(bands.Lower(8) + 0.5 * bands.Width())
        .SetLinewidth(0.0)
        .SetWaistRadius(glm::dvec2(waist));
    scene.Set("/World/Laser", tokens::kTransform,
              ernie::LookAtTransform({0.0, 0.0, 10.0},
                                     {0.0, 0.0, 0.0},
                                     {0.0, 1.0, 0.0}));
    return scene;
}

/// Band 8 of FlatScene's SpectralRadiance, rows of `kVarSize`.
std::vector<float> Band8(RasterDelegate& delegate,
                         const Scene& scene) {
    ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
    const Tensor spectral = outputs.at("/Render/Product")
                                .vars.at("SpectralRadiance")
                                .Map(ernie::MapDevice::Cpu)
                                .tensor;
    const auto* data = static_cast<const float*>(spectral.Data()) +
                       8 * kVarSize * kVarSize;
    return {data, data + kVarSize * kVarSize};
}

/// A Gaussian beam's 1D profile, 1/e^2 radius `w`, averaged over
/// `width` centred at `u` (std::erf, apart from the shader's).
double BeamAverage(double u, double w, double width) {
    const double k = std::sqrt(2.0) / w;
    return 0.5 *
           (std::erf(k * (u + 0.5 * width)) -
            std::erf(k * (u - 0.5 * width))) /
           width;
}

}  // namespace

TEST_F(RasterDelegateTest, LaserLinesAndGridsSpreadTheirPower) {
    RasterDelegate delegate(*context_);
    double bandWidth = 0.0;
    // A 1 mm beam 10 m up, fanned into a flat 60 degree line along x.
    Scene scene = LaserScene(0.001, &bandWidth);
    ernie::LaserLight laser(scene, "/World/Laser");
    laser.SetPattern("line").SetFanAngle(60.0).SetEfficiency(0.9);
    const auto [r0, left] = PixelOf(scene, -10.0f, 0.0f);
    const auto [r1, right] = PixelOf(scene, 10.0f, 0.0f);
    const double pixel = (PixelCentre(scene, 10.0f, 0.0f).x -
                          PixelCentre(scene, -10.0f, 0.0f).x) /
                         static_cast<double>(right - left);
    // L = 0.5 / pi E on the grey floor, 1 W / band width in the band.
    const double unit = 0.5 / M_PI / bandWidth;
    const auto column = [&](const std::vector<float>& band,
                            std::size_t col) {
        double power = 0.0;
        for (std::size_t row = 0; row < kVarSize; ++row) {
            power += band[row * kVarSize + col] * pixel * pixel;
        }
        return power / unit;
    };
    std::vector<float> band = Band8(delegate, scene);
    // Per metre of line, P g(a) cos^2 a / z: flat g = 1 / fan.
    for (const float x : {0.0f, 4.0f}) {
        const double centre = PixelCentre(scene, x, 0.0f).x;
        const double a = std::atan(centre / 10.0);
        const double want = 0.9 / glm::radians(60.0) * std::cos(a) *
                            std::cos(a) / 10.0 * pixel;
        EXPECT_NEAR(column(band, PixelOf(scene, x, 0.0f).second),
                    want, 0.03 * want)
            << "at x = " << x;
    }
    EXPECT_LT(column(band, PixelOf(scene, 8.0f, 0.0f).second), 1e-4)
        << "past the fan's edge, 5.8 m out";

    // A 3 x 3 grid 20 degrees apart: 90% shared by its beamlets, 10%
    // in the zero order at its centre.
    laser.SetPattern("grid")
        .SetGridColumns(3)
        .SetGridRows(3)
        .SetGridPitch({20.0, 20.0})
        .SetZeroOrder(0.1);
    band = Band8(delegate, scene);
    const double offset = 10.0 * std::tan(glm::radians(20.0));
    const auto dot = [&](double x, double y) {
        const auto [row, col] = PixelOf(scene, static_cast<float>(x),
                                        static_cast<float>(y));
        double power = 0.0;
        for (std::size_t r = row - 1; r <= row + 1; ++r) {
            for (std::size_t c = col - 1; c <= col + 1; ++c) {
                power += band[r * kVarSize + c] * pixel * pixel;
            }
        }
        return power / unit;
    };
    EXPECT_NEAR(dot(0.0, 0.0), 0.1 + 0.1, 0.006) << "the centre";
    for (const glm::dvec2 at :
         {glm::dvec2(offset, 0.0), glm::dvec2(-offset, offset),
          glm::dvec2(offset, -offset)}) {
        EXPECT_NEAR(dot(at.x, at.y), 0.1, 0.003)
            << "at " << at.x << ", " << at.y;
    }
}

TEST_F(RasterDelegateTest, CustomLaserPatternsKeepTheirPower) {
    RasterDelegate delegate(*context_);
    double bandWidth = 0.0;
    Scene scene = LaserScene(0.001, &bandWidth);
    ernie::LaserLight laser(scene, "/World/Laser");
    laser.SetPattern("square").SetEfficiency(0.8);
    EXPECT_THROW(delegate.Step(scene), std::invalid_argument)
        << "not registered yet";

    // A uniform square 0.2 across the beam on each side of its axis,
    // a 4 m square on the floor 10 m down; its level any scale.
    ernie::LightPatternModel square;
    square.id = "square";
    square.typeName = "Square";
    square.slang = R"(import light_pattern;
public struct Square : ILightPattern {
    public float emission(LightPatternInput input,
                          LightPatternParams p) {
        return p.get(0);
    }
};
)";
    square.params = {{"level", 1.0f, "Its emission"}};
    square.extent = 0.2;
    delegate.AddLightPattern(square);
    laser.SetPatternParams({3.0f});

    const auto [r0, left] = PixelOf(scene, -10.0f, 0.0f);
    const auto [r1, right] = PixelOf(scene, 10.0f, 0.0f);
    const double pixel = (PixelCentre(scene, 10.0f, 0.0f).x -
                          PixelCentre(scene, -10.0f, 0.0f).x) /
                         static_cast<double>(right - left);
    const double unit = 0.5 / M_PI / bandWidth;
    const std::vector<float> band = Band8(delegate, scene);
    double total = 0.0;
    for (const float value : band) {
        total += value * pixel * pixel;
    }
    EXPECT_NEAR(total / unit, 0.8, 0.04) << "all of its power";
    // 0.8 W over 16 m^2, inside; nothing outside.
    for (const glm::dvec2 at :
         {glm::dvec2(0.0, 0.0), glm::dvec2(1.5, -1.0)}) {
        const auto [row, col] =
            PixelOf(scene, static_cast<float>(at.x),
                    static_cast<float>(at.y));
        EXPECT_NEAR(band[row * kVarSize + col] / unit, 0.05, 0.0015)
            << "at " << at.x << ", " << at.y;
    }
    const auto [row, col] = PixelOf(scene, 3.0f, 0.0f);
    EXPECT_EQ(band[row * kVarSize + col], 0.0f) << "past its edge";
}

TEST_F(RasterDelegateTest, LaserPatternsReadTheirDataAndImage) {
    RasterDelegate delegate(*context_);
    double bandWidth = 0.0;
    Scene scene = LaserScene(0.001, &bandWidth);
    const auto [r0, left] = PixelOf(scene, -10.0f, 0.0f);
    const auto [r1, right] = PixelOf(scene, 10.0f, 0.0f);
    const double pixel = (PixelCentre(scene, 10.0f, 0.0f).x -
                          PixelCentre(scene, -10.0f, 0.0f).x) /
                         static_cast<double>(right - left);
    const double unit = 0.5 / M_PI / bandWidth;
    // The power (per watt) on the floor within a pixel of (x, y).
    const auto around = [&](const std::vector<float>& band, double x,
                            double y) {
        const auto [row, col] = PixelOf(scene, static_cast<float>(x),
                                        static_cast<float>(y));
        double power = 0.0;
        for (std::size_t r = row - 1; r <= row + 1; ++r) {
            for (std::size_t c = col - 1; c <= col + 1; ++c) {
                power += band[r * kVarSize + c] * pixel * pixel;
            }
        }
        return power / unit;
    };

    // Dots where the laser's data put them, sharing its power: a
    // pattern of beamlets, normalized.
    ernie::LightPatternModel dots;
    dots.id = "dots";
    dots.typeName = "Dots";
    dots.normalized = true;
    dots.slang = R"(import light_pattern;
public struct Dots : ILightPattern {
    public float emission(LightPatternInput input,
                          LightPatternParams p) {
        let count = p.dataCount() / 2;
        var sum = 0.0;
        for (uint i = 0; i < count; ++i) {
            let centre = float2(p.data(2 * i), p.data(2 * i + 1));
            sum += beamlet(centre, input);
        }
        return sum / float(count);
    }
};
)";
    delegate.AddLightPattern(dots);
    ernie::LaserLight laser(scene, "/World/Laser");
    laser.SetPattern("dots")
        .SetPatternData({0.4f, 0.0f, -0.3f, 0.3f})
        .SetEfficiency(0.9)
        .SetZeroOrder(0.1);
    std::vector<float> band = Band8(delegate, scene);
    EXPECT_NEAR(around(band, 4.0, 0.0), 0.45, 0.015);
    EXPECT_NEAR(around(band, -3.0, 3.0), 0.45, 0.015);
    EXPECT_NEAR(around(band, 0.0, 0.0), 0.1, 0.005)
        << "the zero order";

    // An image 4 m x 2 m across the floor 10 m down, a quarter of its
    // power in its left half and the rest in its right.
    const std::filesystem::path file =
        std::filesystem::temp_directory_path() /
        "ernie_laser_image_test.png";
    std::array<std::uint8_t, 32> texels{};
    for (std::size_t i = 0; i < texels.size(); ++i) {
        texels[i] = i % 8 < 4 ? 51 : 153;
    }
    ASSERT_NE(stbi_write_png(file.string().c_str(), 8, 4, 1,
                             texels.data(), 8),
              0);
    laser.SetPattern("image")
        .SetPatternImage(file.string())
        .SetPatternFieldOfView(
            glm::degrees(2.0 * glm::atan(glm::dvec2(0.2, 0.1))))
        .SetEfficiency(0.8)
        .SetZeroOrder(0.0);
    band = Band8(delegate, scene);
    double total = 0.0;
    for (const float value : band) {
        total += value * pixel * pixel;
    }
    EXPECT_NEAR(total / unit, 0.8, 0.04) << "all of its power";
    for (const auto& [x, want] :
         {std::pair{-1.0f, 0.05}, std::pair{1.0f, 0.15}}) {
        const auto [row, col] = PixelOf(scene, x, 0.0f);
        EXPECT_NEAR(band[row * kVarSize + col] / unit, want,
                    0.03 * want)
            << "at x = " << x;
    }
    const auto [row, col] = PixelOf(scene, 0.0f, 2.0f);
    EXPECT_EQ(band[row * kVarSize + col], 0.0f) << "past its edge";
    std::filesystem::remove(file);
}

TEST_F(RasterDelegateTest, LasersLightAGaussianSpot) {
    RasterDelegate delegate(*context_);
    double bandWidth = 0.0;
    // The floor's pixels: their width from two pixels 20 m apart.
    Scene wide = LaserScene(4.0, &bandWidth);
    const auto [r0, left] = PixelOf(wide, -10.0f, 0.0f);
    const auto [r1, right] = PixelOf(wide, 10.0f, 0.0f);
    const double pixel = (PixelCentre(wide, 10.0f, 0.0f).x -
                          PixelCentre(wide, -10.0f, 0.0f).x) /
                         static_cast<double>(right - left);
    // Grey floor (albedo 0.5): L = 0.5 / pi E, E per watt the beam's
    // average over the pixel, times 1 W / band width in the band.
    const auto expected = [&](const glm::dvec3& at, double w) {
        return 0.5 / M_PI / bandWidth * BeamAverage(at.x, w, pixel) *
               BeamAverage(at.y, w, pixel);
    };

    // A collimated beam 4 m in radius (its Rayleigh range thousands
    // of km): its Gaussian across the floor.
    for (const glm::dvec2 at :
         {glm::dvec2(0.0, 0.0), glm::dvec2(4.0, 0.0),
          glm::dvec2(0.0, -6.0)}) {
        const auto x = static_cast<float>(at.x);
        const auto y = static_cast<float>(at.y);
        const double want = expected(PixelCentre(wide, x, y), 4.0);
        EXPECT_NEAR(RadianceAt(delegate, wide, x, y), want,
                    0.03 * want)
            << "at " << at.x << ", " << at.y;
    }

    // A beam 1 mm at its waist, 2 mm across 10 m on, a sliver of a
    // pixel: the pixels it falls on still carry its whole power.
    Scene narrow = LaserScene(0.001, &bandWidth);
    const std::vector<float> band = Band8(delegate, narrow);
    double total = 0.0;
    for (const float value : band) {
        total += value * pixel * pixel;
    }
    EXPECT_NEAR(total, 0.5 / M_PI / bandWidth,
                0.03 * 0.5 / M_PI / bandWidth)
        << "the spot's power, whatever the pixels";

    // Slanted 45 degrees, its spot stretches but keeps its power; a
    // small cube halfway along the beam (out of the camera's way to
    // the spot) shadows it.
    narrow.Set("/World/Laser", tokens::kTransform,
               ernie::LookAtTransform({-10.0, 0.0, 10.0},
                                      {0.0, 0.0, 0.0},
                                      {0.0, 1.0, 0.0}));
    // On the floor within 2 m of the spot: not the cube's own lit
    // faces, which the camera sees too.
    const auto [top, first] = PixelOf(narrow, -2.0f, 2.0f);
    const auto [bottom, last] = PixelOf(narrow, 2.0f, -2.0f);
    const auto sum = [&](const Scene& scene) {
        const std::vector<float> values = Band8(delegate, scene);
        double power = 0.0;
        for (std::size_t row = top; row <= bottom; ++row) {
            for (std::size_t col = first; col <= last; ++col) {
                power += values[row * kVarSize + col] * pixel * pixel;
            }
        }
        return power;
    };
    const double open = sum(narrow);
    EXPECT_NEAR(open, 0.5 / M_PI / bandWidth,
                0.03 * 0.5 / M_PI / bandWidth)
        << "slanted";
    ernie::Cube::Define(narrow, "/World/Block")
        .SetSize(0.5)
        .SetXform({.translate = glm::dvec3(-5.0, 0.0, 5.0)});
    EXPECT_LT(sum(narrow), 0.01 * open) << "the cube's shadow";
}

TEST_F(RasterDelegateTest, CylinderLightsLightAsTheirSidesDo) {
    // A tube 8 m long and 2 m across, 10 m up along x, lit from its
    // side only (no caps).
    RasterDelegate delegate(*context_);
    const double radiance = 1000.0 * UnitLuminanceRadiance();
    Scene scene = AreaLightScene(tokens::kCylinderLight);
    ernie::CylinderLight(scene, "/World/Panel")
        .SetLength(8.0)
        .SetRadius(1.0);
    for (const glm::dvec2 at :
         {glm::dvec2(0.0, 0.0), glm::dvec2(6.0, 0.0),
          glm::dvec2(0.0, 6.0)}) {
        // At the pixel's centre, which the coarse image puts up to
        // half a pixel off `at`.
        const glm::dvec3 centre =
            PixelCentre(scene, static_cast<float>(at.x),
                        static_cast<float>(at.y));
        const double expected =
            0.5 / M_PI * radiance *
            CylinderIrradiance(8.0, 1.0, 10.0, centre);
        EXPECT_NEAR(RadianceAt(delegate, scene,
                               static_cast<float>(at.x),
                               static_cast<float>(at.y)),
                    expected, 0.03 * expected)
            << "at " << at.x << ", " << at.y;
    }

    // Focused in relative units: times |cos|^2 off its -z axis, by
    // the direction from its centre.
    ernie::CylinderLight(scene, "/World/Panel").SetShapingFocus(2.0);
    for (const glm::dvec2 at :
         {glm::dvec2(6.0, 0.0), glm::dvec2(0.0, 6.0)}) {
        const glm::dvec3 centre =
            PixelCentre(scene, static_cast<float>(at.x),
                        static_cast<float>(at.y));
        const double c =
            10.0 / glm::length(centre - glm::dvec3(0.0, 0.0, 10.0));
        const double expected =
            0.5 / M_PI * radiance *
            CylinderIrradiance(8.0, 1.0, 10.0, centre) * c * c;
        EXPECT_NEAR(RadianceAt(delegate, scene,
                               static_cast<float>(at.x),
                               static_cast<float>(at.y)),
                    expected, 0.03 * expected)
            << "focused, at " << at.x << ", " << at.y;
    }

    // An IES profile: a downlight's candela against 100 cd every way,
    // by the direction from its centre off its -z axis.
    const char* downlight = R"(IESNA:LM-63-2002
TILT=NONE
1 -1 1 7 1 1 2 0 0 0
1 1 10
0 15 30 45 60 75 90
0
100 96.592583 86.602540 70.710678 50 25.881905 0
)";
    const char* uniform = R"(IESNA:LM-63-2002
TILT=NONE
1 -1 1 2 1 1 2 0 0 0
1 1 10
0 180
0
100 100
)";
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path();
    const std::filesystem::path shaped =
        directory / "ernie_raster_lights_tube_downlight.ies";
    const std::filesystem::path flat =
        directory / "ernie_raster_lights_tube_uniform.ies";
    std::ofstream(shaped) << downlight;
    std::ofstream(flat) << uniform;
    const ernie::IesProfile profile = ernie::ParseIes(downlight);
    Scene reference = scene;
    ernie::CylinderLight(scene, "/World/Panel")
        .ClearShapingFocus()
        .SetShapingIesFile(shaped.string());
    ernie::CylinderLight(reference, "/World/Panel")
        .ClearShapingFocus()
        .SetShapingIesFile(flat.string());
    for (const glm::dvec2 at :
         {glm::dvec2(6.0, 0.0), glm::dvec2(0.0, 6.0)}) {
        const auto x = static_cast<float>(at.x);
        const auto y = static_cast<float>(at.y);
        const glm::dvec3 centre = PixelCentre(scene, x, y);
        const double theta = glm::degrees(std::acos(
            10.0 / glm::length(centre - glm::dvec3(0.0, 0.0, 10.0))));
        const double expected = profile.Sample(theta, 0.0) / 100.0;
        EXPECT_NEAR(RadianceAt(delegate, scene, x, y) /
                        RadianceAt(delegate, reference, x, y),
                    expected, 0.02 * expected)
            << "profiled, at " << at.x << ", " << at.y;
    }
    std::filesystem::remove(shaped);
    std::filesystem::remove(flat);
}

TEST_F(RasterDelegateTest, RectAndDiskLightsMatchTheirFormFactors) {
    RasterDelegate delegate(*context_);
    const double radiance = 1000.0 * UnitLuminanceRadiance();

    Scene rect = AreaLightScene(tokens::kRectLight);
    ernie::RectLight(rect, "/World/Panel")
        .SetWidth(8.0)
        .SetHeight(6.0);
    const double rectExpected = 0.5 / M_PI * radiance * M_PI * 4.0 *
                                CornerFormFactor(4.0, 3.0, 10.0);
    EXPECT_NEAR(RadianceAt(delegate, rect, 0.0f, 0.0f), rectExpected,
                0.03 * rectExpected)
        << "albedo / pi times E = pi L F";

    Scene disk = AreaLightScene(tokens::kDiskLight);
    ernie::DiskLight(disk, "/World/Panel").SetRadius(5.0);
    const double diskExpected =
        0.5 * radiance * 25.0 / (25.0 + 100.0);
    EXPECT_NEAR(RadianceAt(delegate, disk, 0.0f, 0.0f), diskExpected,
                0.03 * diskExpected)
        << "E = pi L r^2 / (r^2 + h^2) on the axis";

    disk.Set("/World/Panel", tokens::kTransform,
             ernie::LookAtTransform({0.0, 0.0, 10.0},
                                    {0.0, 0.0, 20.0},
                                    {0.0, 1.0, 0.0}));
    EXPECT_EQ(RadianceAt(delegate, disk, 0.0f, 0.0f), 0.0f)
        << "facing away";
    disk.Set("/World/Panel", tokens::kTransform,
             ernie::LookAtTransform({0.0, 0.0, 0.5}, {10.0, 0.0, 0.5},
                                    {0.0, 0.0, 1.0}));
    const float grazing = RadianceAt(delegate, disk, 3.0f, 0.0f);
    EXPECT_GT(grazing, 0.0f)
        << "half the disk is above the ground's horizon";
    EXPECT_LT(grazing, static_cast<float>(diskExpected));
}

TEST_F(RasterDelegateTest, AreaLightsCastShadows) {
    RasterDelegate delegate(*context_);
    Scene scene = AreaLightScene(tokens::kRectLight);
    ernie::RectLight(scene, "/World/Panel").SetShadowEnable(true);
    const float lit = RadianceAt(delegate, scene, 3.0f, 0.0f);
    DefineQuad(scene, "/World/Blocker");
    scene.Set(
        "/World/Blocker", tokens::kTransform,
        glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 0.0, 5.0)));
    EXPECT_LT(RadianceAt(delegate, scene, 3.0f, 0.0f), 0.2f * lit)
        << "behind the blocker";
    EXPECT_EQ(delegate.Renderer().LightShadowViews().size(), 6u)
        << "a cube at the centre";
}

TEST_F(RasterDelegateTest, LargerLightsCastWiderPenumbrae) {
    // A lamp at (40, 0, 40) and a blocker halfway to the origin, over
    // x in [4, 20], whose edge throws its shadow's edge onto x = 0,
    // out of the way of the camera looking down: a point's edge is
    // sharp, and a sphere of radius 4 spreads it over about 5.6 to
    // each side (its radius times the receiver's gap to the blocker
    // over the blocker's to the light, slanted onto the ground).
    const auto fractions = [&](double radius) {
        RasterDelegate delegate(*context_);
        Scene scene = LampScene(40.0);
        scene.Set("/World/Lamp", tokens::kTransform,
                  ernie::LookAtTransform({40.0, 0.0, 40.0},
                                         {0.0, 0.0, 0.0},
                                         {0.0, 1.0, 0.0}));
        ernie::SphereLight lamp(scene, "/World/Lamp");
        lamp.SetShadowEnable(true);
        if (radius > 0.0) {
            lamp.SetTreatAsPoint(false).SetRadius(radius);
        }
        const float inLit = RadianceAt(delegate, scene, -2.5f, 0.0f);
        const float outLit = RadianceAt(delegate, scene, 2.5f, 0.0f);
        DefineQuad(scene, "/World/Blocker", 8.0f);
        scene.Set("/World/Blocker", tokens::kTransform,
                  glm::translate(glm::dmat4(1.0),
                                 glm::dvec3(12.0, 0.0, 20.0)));
        return std::pair{
            RadianceAt(delegate, scene, -2.5f, 0.0f) / inLit,
            RadianceAt(delegate, scene, 2.5f, 0.0f) / outLit};
    };
    const auto [pointIn, pointOut] = fractions(0.0);
    EXPECT_LT(pointIn, 0.1f) << "inside a point's shadow";
    EXPECT_GT(pointOut, 0.9f) << "outside it";
    const auto [sphereIn, sphereOut] = fractions(4.0);
    EXPECT_GT(sphereIn, 0.15f) << "the penumbra reaches inside";
    EXPECT_LT(sphereOut, 0.85f) << "and outside";
}

TEST_F(RasterDelegateTest, WiderDistantLightsCastWiderPenumbrae) {
    // A distant light besides the sun, from 45 degrees towards +x,
    // past a blocker 20 up over x in [4, 20] whose edge throws its
    // shadow's edge onto x = 0: sharp from a point in the sky, and
    // from one 10 degrees across spread over about 3.5 to each side
    // (the 28 between the edge and the ground along the light times
    // the tangent of its angular radius, slanted onto the ground).
    const auto fractions = [&](double angle) {
        RasterDelegate delegate(*context_);
        Scene scene = FlatScene();
        // The first distant light by path is the sun, which this one
        // follows.
        ernie::DistantLight(scene, "/World/Sun").SetIntensity(0.0);
        ernie::DistantLight::Define(scene, "/World/Tilted")
            .SetShadowEnable(true)
            .SetAngle(angle);
        scene.Set("/World/Tilted", tokens::kTransform,
                  ernie::DistantLightTransform({1.0, 0.0, 1.0}));
        const float inLit = RadianceAt(delegate, scene, -1.5f, 0.0f);
        const float outLit = RadianceAt(delegate, scene, 1.5f, 0.0f);
        DefineQuad(scene, "/World/Blocker", 8.0f);
        scene.Set("/World/Blocker", tokens::kTransform,
                  glm::translate(glm::dmat4(1.0),
                                 glm::dvec3(12.0, 0.0, 20.0)));
        const float in = RadianceAt(delegate, scene, -1.5f, 0.0f);
        const float out = RadianceAt(delegate, scene, 1.5f, 0.0f);
        EXPECT_EQ(delegate.Renderer().LightShadowViews().size(), 1u)
            << "a map of its own, not the sun's cascades";
        return std::pair{in / inLit, out / outLit};
    };
    const auto [pointIn, pointOut] = fractions(0.0);
    EXPECT_LT(pointIn, 0.1f) << "inside a point's shadow";
    EXPECT_GT(pointOut, 0.9f) << "outside it";
    const auto [sphereIn, sphereOut] = fractions(10.0);
    EXPECT_GT(sphereIn, 0.15f) << "the penumbra reaches inside";
    EXPECT_LT(sphereOut, 0.85f) << "and outside";
}

TEST_F(RasterDelegateTest, DomesLightEverySurfaceDiffusely) {
    Scene scene = FlatScene();
    scene.Remove("/World/Sun");
    ernie::DomeLight sky =
        ernie::DomeLight::Define(scene, "/World/Sky")
            .SetIntensity(1.0);
    RasterDelegate delegate(*context_);
    ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
    const auto vars = MapVars(outputs);
    const std::size_t c = kVarSize / 2;
    EXPECT_NEAR(At<float>(vars.at("HdrColor"), c, c, 1), 0.5f, 0.02f)
        << "a uniform 1 cd/m^2 dome gives albedo 0.5 a luminance of "
           "0.5";
    EXPECT_FALSE(delegate.Renderer().Bands().count == 0);

    std::vector<float> map(8 * 16 * 3, 0.0f);
    std::fill(map.begin(), map.begin() + map.size() / 2, 1.0f);
    sky.SetTexture(Tensor::Copy<float>(map, {8, 16, 3}));
    ernie::RenderProductSetOutputs upper = delegate.Step(scene);
    EXPECT_NEAR(At<float>(MapVars(upper).at("HdrColor"), c, c, 1),
                0.5f, 0.03f)
        << "the upper hemisphere alone lights upward surfaces fully";
    scene.Set("/World/Sky", tokens::kTransform,
              glm::rotate(glm::dmat4(1.0), M_PI,
                          glm::dvec3(1.0, 0.0, 0.0)));
    ernie::RenderProductSetOutputs lower = delegate.Step(scene);
    EXPECT_LT(At<float>(MapVars(lower).at("HdrColor"), c, c, 1),
              0.03f)
        << "turned upside down";
}

TEST_F(RasterDelegateTest,
       MetalsReflectDomesAndAmbientLightThroughTheSplitSum) {
    // A white metal under a uniform dome is a white furnace: with
    // multiple scattering restored it reflects the dome's radiance
    // whole at any roughness, as a white Lambertian floor does; the
    // sun's ambient fraction alike. Before, metals (no diffuse
    // albedo) went black under both, and without the compensation
    // lost a third of the light when rough.
    Scene scene = FlatScene();
    scene.Remove("/World/Sun");
    ernie::DomeLight::Define(scene, "/World/Sky").SetIntensity(1.0);
    ernie::Material grey(scene, "/Looks/Grey");
    grey.SetAlbedo(glm::dvec3(1.0));
    RasterDelegate delegate(*context_);
    const std::size_t c = kVarSize / 2;
    const auto centre = [&] {
        ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
        return At<float>(MapVars(outputs).at("HdrColor"), c, c, 1);
    };
    const float lambert = centre();
    ASSERT_GT(lambert, 0.5f);
    grey.SetBrdf("preview_surface");
    scene.Set("/Looks/Grey", "inputs:metallic", 1.0);
    for (const float roughness : {0.2f, 0.6f, 1.0f}) {
        scene.Set("/Looks/Grey", "inputs:roughness",
                  static_cast<double>(roughness));
        EXPECT_NEAR(centre() / lambert, 1.0f, 0.03f)
            << "dome, roughness " << roughness;
    }

    // The ambient fraction of a sun below the horizon, which lights
    // nothing directly.
    scene.Remove("/World/Sky");
    ernie::DistantLight::Define(scene, "/World/Sun")
        .SetShadowEnable(false);
    scene.Set("/World/Sun", tokens::kTransform,
              ernie::DistantLightTransform({0.0, 0.0, -1.0}));
    ernie::RenderSettings::Define(scene, "/Render/Settings")
        .SetAmbient(0.5);
    grey.SetBrdf("lambert");
    const float ambient = centre();
    ASSERT_GT(ambient, 0.0f);
    grey.SetBrdf("preview_surface");
    EXPECT_NEAR(centre() / ambient, 1.0f, 0.03f) << "ambient";

    // A smooth metal mirrors the dome's map: facing up under a map
    // bright above the horizon it shows the bright half, and with the
    // dome turned over the dark one.
    scene.Remove("/World/Sun");
    ernie::RenderSettings::Define(scene, "/Render/Settings")
        .SetAmbient(0.0);
    std::vector<float> map(8 * 16 * 3, 0.0f);
    std::fill(map.begin(), map.begin() + map.size() / 2, 1.0f);
    ernie::DomeLight::Define(scene, "/World/Sky")
        .SetIntensity(1.0)
        .SetTexture(Tensor::Copy<float>(map, {8, 16, 3}));
    scene.Set("/Looks/Grey", "inputs:roughness", 0.05);
    const float above = centre();
    EXPECT_GT(above, 0.8f * lambert);
    scene.Set("/World/Sky", tokens::kTransform,
              glm::rotate(glm::dmat4(1.0), M_PI,
                          glm::dvec3(1.0, 0.0, 0.0)));
    EXPECT_LT(centre(), 0.05f * above) << "the dark half mirrored";
}

TEST_F(RasterDelegateTest, DomeLightModesShadowLimitAndApproximate) {
    // FlatScene's floor under a uniform dome with a 6 m cube resting
    // on it. The approximate mode (the default) lights the floor
    // beside the cube as fully as open floor; the full one darkens it
    // there (the cube hides part of the sky) while open floor stays
    // lit, unless the dome's shadows are off; the limited one gives
    // the Lambertian floor no dome light at all.
    Scene scene = FlatScene();
    scene.Remove("/World/Sun");
    ernie::DomeLight sky =
        ernie::DomeLight::Define(scene, "/World/Sky")
            .SetIntensity(1.0);
    ernie::Cube::Define(scene, "/World/Box")
        .SetSize(6.0)
        .SetMaterialBinding("/Looks/Grey");
    scene.Set(
        "/World/Box", tokens::kTransform,
        glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 0.0, 3.0)));
    RasterDelegate delegate(*context_);
    // Floor a metre and a half from the cube's side, and floor beyond
    // the occlusion radius of it.
    const auto [nearRow, nearCol] = PixelOf(scene, 4.5f, 0.0f);
    const auto [farRow, farCol] = PixelOf(scene, -12.0f, 12.0f);
    const auto floor = [&] {
        ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
        const auto vars = MapVars(outputs);
        return std::pair(
            At<float>(vars.at("HdrColor"), nearRow, nearCol, 1),
            At<float>(vars.at("HdrColor"), farRow, farCol, 1));
    };
    const auto [approximateNear, approximateFar] = floor();
    EXPECT_NEAR(approximateNear, approximateFar,
                0.02f * approximateFar)
        << "unoccluded";

    ernie::RenderSettings settings =
        ernie::RenderSettings::Define(scene, "/Render/Settings")
            .SetDomeLightMode("full")
            .SetDomeShadowRadius(8.0);
    const auto [fullNear, fullFar] = floor();
    EXPECT_LT(fullNear, 0.9f * approximateNear)
        << "the cube hides some of the sky";
    EXPECT_NEAR(fullFar, approximateFar, 0.02f * approximateFar)
        << "open floor sees it all";
    // Floor alongside the cube's face, at the same distance from it,
    // is as occluded everywhere: the filter smooths the per-pixel
    // rotations' noise away.
    {
        ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
        const auto vars = MapVars(outputs);
        float low = 1e9f;
        float high = 0.0f;
        for (const float y :
             {-1.0f, 0.0f,
              1.0f}) {  // the face's middle, away from its corners
            const auto [row, col] = PixelOf(scene, 4.5f, y);
            const float value =
                At<float>(vars.at("HdrColor"), row, col, 1);
            low = std::min(low, value);
            high = std::max(high, value);
        }
        EXPECT_LT(high - low, 0.05f * high)
            << "smooth along the face";
    }

    sky.SetShadowEnable(false);
    EXPECT_NEAR(floor().first, approximateNear,
                0.02f * approximateNear)
        << "shadows off";

    settings.SetDomeLightMode("limited");
    EXPECT_LT(floor().second, 0.01f * approximateFar)
        << "no diffuse dome light";

    settings.SetDomeLightMode("bright");
    EXPECT_THROW(delegate.Step(scene), std::invalid_argument);
}

TEST_F(RasterDelegateTest, TheFullDomeModeShadowsTheSunOfAnHdrSky) {
    // A sky of 1 with a sun of 3000 in one 2-degree texel, 60 degrees
    // up towards +x, over FlatScene's floor and a 6 m cube. The full
    // mode lights the sun as a shadowed sun of its own: the cube
    // shadows the floor on its -x side; the approximate mode leaves
    // the sun in the map, lighting both sides alike. Open floor gets
    // about the same light either way.
    Scene scene = FlatScene();
    scene.Remove("/World/Sun");
    const std::int64_t rows = 90;
    const std::int64_t cols = 180;
    std::vector<float> map(static_cast<std::size_t>(rows * cols * 3),
                           1.0f);
    const std::int64_t sunRow = 15;  // 30-32 degrees from the zenith
    for (int c = 0; c < 3; ++c) {
        map[static_cast<std::size_t>((sunRow * cols + 0) * 3 + c)] =
            3000.0f;  // azimuth 0, +x
    }
    ernie::DomeLight::Define(scene, "/World/Sky")
        .SetIntensity(1.0)
        .SetTexture(Tensor::Copy<float>(map, {rows, cols, 3}));
    ernie::Cube::Define(scene, "/World/Box")
        .SetSize(6.0)
        .SetMaterialBinding("/Looks/Grey");
    scene.Set(
        "/World/Box", tokens::kTransform,
        glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 0.0, 3.0)));
    RasterDelegate delegate(*context_);
    const auto [shadeRow, shadeCol] = PixelOf(scene, -4.5f, 0.0f);
    const auto [sunnyRow, sunnyCol] = PixelOf(scene, 4.5f, 0.0f);
    const auto [openRow, openCol] = PixelOf(scene, -12.0f, 12.0f);
    struct Floor {
        float shade;
        float sunny;
        float open;
    };
    const auto floor = [&] {
        ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
        const auto vars = MapVars(outputs);
        const Tensor& hdr = vars.at("HdrColor");
        return Floor{At<float>(hdr, shadeRow, shadeCol, 1),
                     At<float>(hdr, sunnyRow, sunnyCol, 1),
                     At<float>(hdr, openRow, openCol, 1)};
    };
    const Floor approximate = floor();
    EXPECT_NEAR(approximate.shade, approximate.sunny,
                0.05f * approximate.sunny)
        << "the sun in the map casts no shadow";

    ernie::RenderSettings::Define(scene, "/Render/Settings")
        .SetDomeLightMode("full");
    const Floor full = floor();
    // The sun gives about a third of the light here, so its shadow
    // leaves about two thirds.
    EXPECT_LT(full.shade, 0.75f * full.sunny)
        << "the cube shadows the sun";
    EXPECT_GT(full.shade, 0.5f * full.sunny)
        << "the rest of the sky still lights the shadow";
    // Beside the cube, 1.5 m from its face, the floor is out of the
    // sun's shadow but the cube hides about a quarter of the sky
    // (rising 76 degrees across +-63 degrees of azimuth), which gives
    // two thirds of the light here: about 0.83 of the open floor's.
    EXPECT_NEAR(full.sunny / full.open, 0.83f, 0.05f)
        << "the cube hides some sky beside it";
    EXPECT_NEAR(full.open, approximate.open, 0.05f * approximate.open)
        << "the sun's light moved out of the map, not added";
}

TEST_F(RasterDelegateTest,
       TheDirectionsModeShadowsTheDomeFromTheWholeScene) {
    // FlatScene's floor and a 6 m cube under a uniform dome. In the
    // directions mode the dome's diffuse light comes from 16 shadowed
    // directions: open floor gets as much as unshadowed, while floor
    // beside the cube loses the directions the cube blocks, from the
    // cube's shadows rather than the (1 m) screen-space radius.
    Scene scene = FlatScene();
    scene.Remove("/World/Sun");
    ernie::DomeLight::Define(scene, "/World/Sky").SetIntensity(1.0);
    ernie::Cube::Define(scene, "/World/Box")
        .SetSize(6.0)
        .SetMaterialBinding("/Looks/Grey");
    scene.Set(
        "/World/Box", tokens::kTransform,
        glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 0.0, 3.0)));
    RasterDelegate delegate(*context_);
    const auto [nearRow, nearCol] = PixelOf(scene, 4.0f, 0.0f);
    const auto [openRow, openCol] = PixelOf(scene, -12.0f, 12.0f);
    const auto floor = [&] {
        ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
        const auto vars = MapVars(outputs);
        return std::pair(
            At<float>(vars.at("HdrColor"), nearRow, nearCol, 1),
            At<float>(vars.at("HdrColor"), openRow, openCol, 1));
    };
    const auto [approximateNear, approximateOpen] = floor();
    ernie::RenderSettings::Define(scene, "/Render/Settings")
        .SetDomeLightMode("directions");
    const auto [directionsNear, directionsOpen] = floor();
    EXPECT_NEAR(directionsOpen, approximateOpen,
                0.08f * approximateOpen)
        << "the directions carry the dome's diffuse light";
    EXPECT_LT(directionsNear, 0.85f * approximateNear)
        << "the cube blocks some of them";
}

TEST_F(RasterDelegateTest,
       TerrainHorizonsShadowTheDomeFromOffScreen) {
    // FlatScene's floor, 65 x 65 cells of 2 m, with a cliff 50 m tall
    // at x = 25 to 35 m, outside the view (which spans about 16 m
    // either side of the centre), under a uniform dome. The full
    // mode's horizon map sees the cliff: floor near it (x = 14) loses
    // much of its sky, floor farther (x = -14) less; the approximate
    // mode lights both alike. The screen-space radius stays at its 1
    // m default, so the horizon map does the shadowing.
    Scene scene = FlatScene();
    scene.Remove("/World/Sun");
    std::vector<float> heights(65 * 65, 0.0f);
    for (int row = 0; row < 65; ++row) {
        for (int col = 0; col < 65; ++col) {
            const double x = -65.0 + (col + 0.5) * 2.0;
            if (x >= 25.0 && x <= 35.0) {
                heights[static_cast<std::size_t>(row * 65 + col)] =
                    50.0f;
            }
        }
    }
    ernie::Terrain(scene, "/World/Terrain")
        .SetHeights(Tensor::Copy<float>(heights, {65, 65}));
    ernie::DomeLight::Define(scene, "/World/Sky").SetIntensity(1.0);
    RasterDelegate delegate(*context_);
    const auto [nearRow, nearCol] = PixelOf(scene, 14.0f, 0.0f);
    const auto [farRow, farCol] = PixelOf(scene, -14.0f, 0.0f);
    const auto floor = [&] {
        ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
        const auto vars = MapVars(outputs);
        return std::pair(
            At<float>(vars.at("HdrColor"), nearRow, nearCol, 1),
            At<float>(vars.at("HdrColor"), farRow, farCol, 1));
    };
    const auto [approximateNear, approximateFar] = floor();
    EXPECT_NEAR(approximateNear, approximateFar,
                0.02f * approximateFar);

    ernie::RenderSettings::Define(scene, "/Render/Settings")
        .SetDomeLightMode("full");
    const auto [fullNear, fullFar] = floor();
    EXPECT_LT(fullNear, 0.8f * approximateNear)
        << "the cliff hides much of the sky";
    EXPECT_GT(fullFar, fullNear + 0.05f * approximateFar)
        << "less so farther from it";
}

TEST_F(RasterDelegateTest, WhatStandsOnTheTerrainHidesItsSky) {
    // Flat terrain, 65 x 65 cells of 2 m, and a wall 40 m tall and 60
    // m long standing on it at x = 24 to 30 m, outside the view,
    // under a uniform dome. The full mode's horizon map over what
    // stands on the terrain sees the wall: floor near it (x = 14)
    // loses much of its sky, floor farther (x = -14) less; the
    // approximate mode lights both alike.
    Scene scene = FlatScene();
    scene.Remove("/World/Sun");
    ernie::Terrain(scene, "/World/Terrain")
        .SetHeights(Tensor::Copy<float>(
            std::vector<float>(65 * 65, 0.0f), {65, 65}));
    ernie::Cube::Define(scene, "/World/Wall")
        .SetSize(1.0)
        .SetXform({.translate = glm::dvec3(27.0, 0.0, 20.0),
                   .scale = glm::dvec3(6.0, 60.0, 40.0)});
    ernie::DomeLight::Define(scene, "/World/Sky").SetIntensity(1.0);
    RasterDelegate delegate(*context_);
    const auto [nearRow, nearCol] = PixelOf(scene, 14.0f, 0.0f);
    const auto [farRow, farCol] = PixelOf(scene, -14.0f, 0.0f);
    const auto floor = [&] {
        ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
        const auto vars = MapVars(outputs);
        return std::pair(
            At<float>(vars.at("HdrColor"), nearRow, nearCol, 1),
            At<float>(vars.at("HdrColor"), farRow, farCol, 1));
    };
    const auto [approximateNear, approximateFar] = floor();
    EXPECT_NEAR(approximateNear, approximateFar,
                0.02f * approximateFar);

    ernie::RenderSettings::Define(scene, "/Render/Settings")
        .SetDomeLightMode("full");
    const auto [fullNear, fullFar] = floor();
    EXPECT_LT(fullNear, 0.8f * approximateNear)
        << "the wall hides much of the sky";
    EXPECT_GT(fullFar, fullNear + 0.05f * approximateFar)
        << "less so farther from it";

    // Sunk below the ground, it no longer does.
    ernie::Cube(scene, "/World/Wall")
        .SetXform({.translate = glm::dvec3(27.0, 0.0, -100.0),
                   .scale = glm::dvec3(6.0, 60.0, 40.0)});
    const auto [movedNear, movedFar] = floor();
    EXPECT_GT(movedNear, 0.95f * approximateNear)
        << "the horizon follows what moves";
}

TEST_F(RasterDelegateTest,
       TerrainHorizonsShadowWhatStandsOnTheTerrain) {
    // The cliff of TerrainHorizonsShadowTheDomeFromOffScreen, and a
    // block 6 m wide whose top stands 30 m up, 13 m from the cliff.
    // From the top the cliff rises 57 degrees at most, hiding about a
    // sixth of its cosine-weighted sky; from the floor beside it, 11
    // m from the cliff, 78 degrees, hiding over a third. Only the
    // horizon map at the top's height sees that.
    Scene scene = FlatScene();
    scene.Remove("/World/Sun");
    std::vector<float> heights(65 * 65, 0.0f);
    for (int row = 0; row < 65; ++row) {
        for (int col = 0; col < 65; ++col) {
            const double x = -65.0 + (col + 0.5) * 2.0;
            if (x >= 25.0 && x <= 35.0) {
                heights[static_cast<std::size_t>(row * 65 + col)] =
                    50.0f;
            }
        }
    }
    ernie::Terrain(scene, "/World/Terrain")
        .SetHeights(Tensor::Copy<float>(heights, {65, 65}));
    ernie::Cube::Define(scene, "/World/Block")
        .SetSize(1.0)
        .SetXform({.translate = glm::dvec3(12.0, 0.0, 15.0),
                   .scale = glm::dvec3(6.0, 6.0, 30.0)});
    ernie::DomeLight::Define(scene, "/World/Sky").SetIntensity(1.0);
    RasterDelegate delegate(*context_);
    // The pixels seeing the block's top at (12, 0, 30) and the floor
    // at (14, -8).
    const glm::mat4 viewProjection =
        ernie::CameraParamsOf(scene, "/World/Camera", kSize, kSize)
            .RasterView()
            .ViewProjection();
    const glm::vec4 clip =
        viewProjection * glm::vec4(12.0f, 0.0f, 30.0f, 1.0f);
    const glm::vec2 ndc = glm::vec2(clip) / clip.w;
    const auto topRow =
        static_cast<std::size_t>((0.5f - ndc.y * 0.5f) * kVarSize);
    const auto topCol =
        static_cast<std::size_t>((ndc.x * 0.5f + 0.5f) * kVarSize);
    const auto [floorRow, floorCol] = PixelOf(scene, 14.0f, -8.0f);
    const auto lit = [&] {
        ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
        const auto vars = MapVars(outputs);
        return std::pair(
            At<float>(vars.at("HdrColor"), topRow, topCol, 1),
            At<float>(vars.at("HdrColor"), floorRow, floorCol, 1));
    };
    const auto [approximateTop, approximateFloor] = lit();
    ernie::RenderSettings::Define(scene, "/Render/Settings")
        .SetDomeLightMode("full");
    const auto [fullTop, fullFloor] = lit();
    EXPECT_LT(fullTop, 0.92f * approximateTop)
        << "the cliff hides some of the top's sky";
    EXPECT_GT(fullTop / approximateTop,
              1.1f * fullFloor / approximateFloor)
        << "but less than the floor's below it";
}

TEST_F(RasterDelegateTest, EmissiveSurfacesShowTheirLuminanceUnlit) {
    // With no light at all, a floor emitting white of 2 cd/m^2 shows
    // a luminance of 2 (HdrColor is in cd/m^2), and red emission
    // reads red.
    Scene scene = FlatScene();
    scene.Remove("/World/Sun");
    ernie::Material grey(scene, "/Looks/Grey");
    grey.SetEmissiveColor(glm::dvec3(2.0));
    RasterDelegate delegate(*context_);
    const std::size_t c = kVarSize / 2;
    ernie::RenderProductSetOutputs white = delegate.Step(scene);
    const auto vars = MapVars(white);
    EXPECT_NEAR(At<float>(vars.at("HdrColor"), c, c, 1), 2.0f, 0.06f);
    EXPECT_NEAR(At<float>(vars.at("HdrColor"), c, c, 0), 2.0f, 0.1f);

    grey.SetEmissiveColor(glm::dvec3(1.0, 0.0, 0.0));
    ernie::RenderProductSetOutputs red = delegate.Step(scene);
    const auto reds = MapVars(red);
    EXPECT_GT(At<float>(reds.at("HdrColor"), c, c, 0),
              3.0f * At<float>(reds.at("HdrColor"), c, c, 1));
}

TEST_F(RasterDelegateTest,
       PhotometricLightsLightByTheirLuxAndLumens) {
    // FlatScene's grey floor (albedo 0.5) under a 1 m square panel 5
    // m up giving 1000 lux at 5 m: the floor below receives about 990
    // lux (the panel is not quite a point), so its luminance is 0.5 x
    // 990 / pi cd/m^2.
    Scene scene = FlatScene();
    scene.Remove("/World/Sun");
    // A plain Lambertian floor, not the hillshaded terrain.
    scene.Remove("/World/Terrain");
    ernie::GroundPlane::Define(scene, "/World/Floor")
        .SetMaterialBinding("/Looks/Grey");
    // Close enough that the centre pixel samples the floor within a
    // few centimetres of the axis, where the illuminance has not yet
    // fallen off (by cos^4 of the angle off it).
    scene.Set("/World/Camera", tokens::kTransform,
              ernie::LookAtTransform({0.0, 0.0, 20.0},
                                     {0.0, 0.0, 0.0},
                                     {0.0, 1.0, 0.0}));
    ernie::RectLight::Define(scene, "/World/Panel")
        .SetPhotometricIlluminance(1000.0)
        .SetPhotometricIlluminanceDistance(5.0)
        .SetShadowEnable(false);
    scene.Set("/World/Panel", tokens::kTransform,
              ernie::LookAtTransform({0.0, 0.0, 5.0}, {0.0, 0.0, 0.0},
                                     {0.0, 1.0, 0.0}));
    RasterDelegate delegate(*context_);
    auto outputs = delegate.Step(scene);
    const std::size_t c = kVarSize / 2;
    const glm::vec3 hdr(
        At<float>(MapVars(outputs).at("HdrColor"), c, c, 0),
        At<float>(MapVars(outputs).at("HdrColor"), c, c, 1),
        At<float>(MapVars(outputs).at("HdrColor"), c, c, 2));
    const float luminance =
        glm::dot(hdr, glm::vec3(0.2126f, 0.7152f, 0.0722f));
    EXPECT_NEAR(M_PI * luminance / 0.5, 990.0, 25.0);
}

TEST_F(RasterDelegateTest, ARenderersFirstFrameHasItsLightsShadows) {
    // The shading of a renderer's first frame once read the light
    // shadow maps' size before the lights made them, losing the
    // spot's shadow for that frame.
    Scene scene = LampScene(10.0, 60.0f);
    DefineQuad(scene, "/World/Blocker");
    scene.Set(
        "/World/Blocker", tokens::kTransform,
        glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 0.0, 5.0)));
    RasterDelegate first(*context_);
    const std::vector<std::uint8_t> once = Bytes(first.Render(scene));
    EXPECT_EQ(Bytes(first.Render(scene)), once);
}

TEST_F(RasterDelegateTest, MeteringMeasuresWhatACubeCameraSees) {
    // The grey floor under a 20000 lux sun straight above, Lambertian
    // so 0.5 x 20000 / pi cd/m^2 from every direction, through an
    // equidistant fisheye reaching 80 degrees off axis on cube faces:
    // metered as through a frame camera.
    Scene scene = FlatScene();
    scene.Remove("/World/Terrain");
    ernie::GroundPlane::Define(scene, "/World/Floor")
        .SetMaterialBinding("/Looks/Grey");
    ernie::DistantLight(scene, "/World/Sun")
        .SetPhotometricIlluminance(20000.0);
    const double aperture = 20.0;
    const double corner = std::sqrt(2.0) * 0.5 * kVarSize;
    ernie::Camera(scene, "/World/Camera")
        .SetLensProjection(tokens::kFisheye)
        .SetVerticalAperture(aperture)
        .SetHorizontalAperture(aperture)
        .SetFocalLength(aperture * corner / (80.0 * M_PI / 180.0) /
                        kVarSize);
    ASSERT_EQ(ernie::CubeFacesOf(ernie::CameraParamsOf(
                  scene, "/World/Camera", kVarSize, kVarSize)),
              ernie::kCubeFaces);
    ernie::RenderSettings(scene, "/Render/Settings").SetMeter(true);
    RasterDelegate delegate(*context_);
    for (int i = 0; i < 3; ++i) {
        delegate.Render(scene);
    }
    ASSERT_TRUE(delegate.MeteredLuminance());
    EXPECT_NEAR(std::log2(*delegate.MeteredLuminance()),
                std::log2(0.5 * 20000.0 / M_PI), 0.3);
}

TEST_F(RasterDelegateTest, MeteringMeasuresTheScenesLuminance) {
    // The grey floor (albedo 0.5) under a 20000 lux sun straight
    // above: 0.5 x 20000 / pi cd/m^2 everywhere, which the meter
    // reads a frame or two later.
    Scene scene = FlatScene();
    scene.Remove("/World/Terrain");
    ernie::GroundPlane::Define(scene, "/World/Floor")
        .SetMaterialBinding("/Looks/Grey");
    ernie::DistantLight(scene, "/World/Sun")
        .SetPhotometricIlluminance(20000.0);
    RasterDelegate delegate(*context_);
    delegate.Render(scene);
    EXPECT_FALSE(delegate.MeteredLuminance())
        << "metering is off by default";
    ernie::RenderSettings settings(scene, "/Render/Settings");
    settings.SetMeter(true);
    for (int i = 0; i < 3; ++i) {
        delegate.Render(scene);
    }
    ASSERT_TRUE(delegate.MeteredLuminance());
    // Within a quarter of the histogram's half-stop bins.
    EXPECT_NEAR(std::log2(*delegate.MeteredLuminance()),
                std::log2(0.5 * 20000.0 / M_PI), 0.3);

    // Exposure compensation brightens the photographic image by its
    // stops.
    const auto centre = [&] {
        const Tensor image = delegate.Render(scene);
        return static_cast<int>(
            image.As<std::uint8_t>()[4 * (kVarSize / 2 * kVarSize +
                                          kVarSize / 2)]);
    };
    ernie::Camera(scene, "/World/Camera")
        .SetExposureTime(1.0 / 20000.0);
    const int base = centre();
    settings.SetExposureCompensation(2.0);
    EXPECT_GT(centre(), base + 40);
}

TEST_F(RasterDelegateTest,
       AutomaticExposureKeysTheSceneToMiddleGrey) {
    // The grey floor fills the view, so automatic exposure shows it
    // at middle grey, 0.18 linear, 118 in sRGB: from the first frame
    // (metered first), and whatever lights it.
    Scene scene = FlatScene();
    scene.Remove("/World/Terrain");
    ernie::GroundPlane::Define(scene, "/World/Floor")
        .SetMaterialBinding("/Looks/Grey");
    ernie::DistantLight sun(scene, "/World/Sun");
    sun.SetPhotometricIlluminance(20000.0);
    ernie::Camera camera(scene, "/World/Camera");
    camera.SetExposureMode(std::string(tokens::kAuto));
    RasterDelegate delegate(*context_);
    const auto centre = [&] {
        const Tensor image = delegate.Render(scene);
        return static_cast<int>(
            image.As<std::uint8_t>()[4 * (kVarSize / 2 * kVarSize +
                                          kVarSize / 2)]);
    };
    EXPECT_NEAR(centre(), 118, 8) << "the first frame";
    // A tenth of the light: the next frames meter it and catch up.
    sun.SetPhotometricIlluminance(2000.0);
    for (int i = 0; i < 3; ++i) {
        centre();
    }
    EXPECT_NEAR(centre(), 118, 8) << "a tenth of the light";

    // Manual exposure shows the light as it is: exposed for this
    // light, then ten times as much.
    camera.SetExposureMode(std::string(tokens::kManual))
        .SetExposureTime(1.0 / 2000.0);
    const int manual = centre();
    sun.SetPhotometricIlluminance(20000.0);
    EXPECT_GT(centre(), manual + 40) << "ten times the light";

    // Adapting over frames: a change shows at first, then fades.
    camera.SetExposureMode(std::string(tokens::kAuto))
        .SetExposureAdaptation(8.0);
    for (int i = 0; i < 4; ++i) {
        centre();
    }
    sun.SetPhotometricIlluminance(2000.0);
    centre();
    centre();
    EXPECT_LT(centre(), 100) << "still dim just after";
    for (int i = 0; i < 60; ++i) {
        centre();
    }
    EXPECT_NEAR(centre(), 118, 8) << "adapted";

    // Without products, the settings' own of LdrColor alone: it holds
    // HdrColor besides, for the meter.
    scene.Set("/Render/Settings", tokens::kProducts,
              std::vector<ernie::Path>{});
    camera.SetExposureAdaptation(0.0);
    RasterDelegate fresh(*context_);
    const Tensor image = fresh.Render(scene);
    EXPECT_NEAR(
        static_cast<int>(
            image.As<std::uint8_t>()[4 * (kVarSize / 2 * kVarSize +
                                          kVarSize / 2)]),
        118, 8)
        << "LdrColor alone";
}

namespace {

/// FlatScene without its terrain or sun: an empty world, the camera
/// at the origin looking along `forward`.
Scene SkyScene(glm::dvec3 forward, glm::dvec3 up) {
    Scene scene = FlatScene();
    scene.Remove("/World/Terrain");
    scene.Remove("/World/Sun");
    scene.Set("/World/Camera", tokens::kTransform,
              ernie::LookAtTransform({0.0, 0.0, 0.0}, forward, up));
    return scene;
}

/// Band `band` of SpectralRadiance, (bands, rows, cols), at (`row`,
/// `col`) of a `size` square.
float SpectralAt(const Tensor& spectral, std::size_t band,
                 std::size_t row, std::size_t col, std::size_t size) {
    return static_cast<const float*>(
        spectral.Data())[(band * size + row) * size + col];
}

}  // namespace

TEST_F(RasterDelegateTest, DomesShowBehindTheScene) {
    Scene scene = SkyScene({0.0, 0.0, 1.0}, {0.0, 1.0, 0.0});
    ernie::DomeLight sky =
        ernie::DomeLight::Define(scene, "/World/Sky")
            .SetIntensity(1.0);
    RasterDelegate delegate(*context_);
    ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
    auto vars = MapVars(outputs);
    const ernie::SpectralBands bands =
        ernie::CameraParamsOf(scene, "/World/Camera", kVarSize,
                              kVarSize)
            .bands;
    const std::vector<float> dome =
        ernie::DomeParamsOf(scene, "/World/Sky", bands).radiance;
    const std::size_t c = kVarSize / 2;
    for (int band = 0; band < bands.count; ++band) {
        EXPECT_NEAR(SpectralAt(vars.at("SpectralRadiance"), band, c,
                               c, kVarSize),
                    dome[band], 0.02f * dome[band])
            << "band " << band;
    }
    const auto luminance =
        [&](std::map<std::string, Tensor, std::less<>>& of) {
            return 0.2126f * At<float>(of.at("HdrColor"), c, c, 0) +
                   0.7152f * At<float>(of.at("HdrColor"), c, c, 1) +
                   0.0722f * At<float>(of.at("HdrColor"), c, c, 2);
        };
    EXPECT_NEAR(luminance(vars), 1.0f, 0.03f)
        << "a 1 cd/m^2 dome looks 1 cd/m^2";
    EXPECT_EQ(At<std::uint32_t>(vars.at("PrimId"), c, c), 0u)
        << "still no prim";

    // Red above the horizon, blue below: looking up sees red, until
    // the dome turns over.
    std::vector<float> map(8 * 16 * 3, 0.0f);
    for (std::size_t texel = 0; texel < 8 * 16; ++texel) {
        map[3 * texel + (texel < 4 * 16 ? 0 : 2)] = 1.0f;
    }
    sky.SetTexture(Tensor::Copy<float>(map, {8, 16, 3}));
    ernie::RenderProductSetOutputs upper = delegate.Step(scene);
    auto up = MapVars(upper);
    EXPECT_GT(At<float>(up.at("HdrColor"), c, c, 0),
              10.0f * At<float>(up.at("HdrColor"), c, c, 2));
    scene.Set("/World/Sky", tokens::kTransform,
              glm::rotate(glm::dmat4(1.0), M_PI,
                          glm::dvec3(1.0, 0.0, 0.0)));
    ernie::RenderProductSetOutputs turned = delegate.Step(scene);
    auto down = MapVars(turned);
    EXPECT_GT(At<float>(down.at("HdrColor"), c, c, 2),
              10.0f * At<float>(down.at("HdrColor"), c, c, 0));

    sky.SetVisibleInPrimaryRay(false);
    ernie::RenderProductSetOutputs hidden = delegate.Step(scene);
    EXPECT_EQ(SpectralAt(MapVars(hidden).at("SpectralRadiance"), 8, c,
                         c, kVarSize),
              0.0f)
        << "not in camera rays, though it still lights";
}

TEST_F(RasterDelegateTest,
       StarsKeepTheirIrradianceAtAnyResolutionBehindTheScene) {
    // A magnitude 0 star of the Sun's temperature a little off the
    // view's axis: summing the radiance of its pixels times their
    // solid angles gives back its spectral irradiance, at any
    // resolution.
    Scene scene = SkyScene({1.0, 0.0, 0.0}, {0.0, 0.0, 1.0});
    const auto one = [](float value) {
        return Tensor::Copy<float>(std::array{value}, {1});
    };
    ernie::StarField::Define(scene, "/World/Stars")
        .SetRightAscension(one(0.7f))
        .SetDeclination(one(-0.4f))
        .SetMagnitude(one(0.0f))
        .SetTemperature(one(5772.0f));
    RasterDelegate delegate(*context_);
    const auto irradiance = [&](std::size_t size, std::size_t band) {
        ernie::RenderSettings(scene, "/Render/Settings")
            .SetResolution(glm::dvec2(static_cast<double>(size),
                                      static_cast<double>(size)));
        ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
        const Tensor spectral = outputs.at("/Render/Product")
                                    .vars.at("SpectralRadiance")
                                    .Map(ernie::MapDevice::Cpu)
                                    .tensor;
        const ernie::CameraParams camera =
            ernie::CameraParamsOf(scene, "/World/Camera", size, size);
        const double f = camera.FocalLengthPixels();
        double sum = 0.0;
        for (std::size_t row = 0; row < size; ++row) {
            for (std::size_t col = 0; col < size; ++col) {
                const double x =
                    (static_cast<double>(col) + 0.5 - 0.5 * size) / f;
                const double y =
                    (static_cast<double>(row) + 0.5 - 0.5 * size) / f;
                const double cosine =
                    1.0 / std::sqrt(1.0 + x * x + y * y);
                sum += SpectralAt(spectral, band, row, col, size) *
                       cosine * cosine * cosine / (f * f);
            }
        }
        return sum;
    };
    const ernie::SpectralBands bands =
        ernie::CameraParamsOf(scene, "/World/Camera", 32, 32).bands;
    for (const std::size_t band : {std::size_t{2}, std::size_t{12}}) {
        const double expected = ernie::BlackbodySpectralIrradiance(
            bands.Lower(static_cast<int>(band)) + 0.5 * bands.Width(),
            5772.0, 2.54e-6);
        for (const std::size_t size : {32u, 57u, 90u}) {
            EXPECT_NEAR(irradiance(size, band), expected,
                        0.03 * expected)
                << size << " pixels, band " << band;
        }
    }

    // A quad in front hides it.
    DefineQuad(scene, "/World/Blocker");
    scene.Set("/World/Blocker", tokens::kTransform,
              glm::rotate(glm::translate(glm::dmat4(1.0),
                                         glm::dvec3(5.0, 0.0, 0.0)),
                          M_PI / 2.0, glm::dvec3(0.0, 1.0, 0.0)));
    EXPECT_EQ(irradiance(32, 12), 0.0) << "behind the scene";
}

TEST_F(RasterDelegateTest, PushbroomRowsSeeStarsFromTheirOwnPoses) {
    // A magnitude 0 star in the detector line's plane of view
    // (declination 0, the camera looking along +x with +z up), a
    // little across the line.
    Scene scene = SkyScene({1.0, 0.0, 0.0}, {0.0, 0.0, 1.0});
    const auto one = [](float value) {
        return Tensor::Copy<float>(std::array{value}, {1});
    };
    ernie::StarField::Define(scene, "/World/Stars")
        .SetRightAscension(one(0.05f))
        .SetDeclination(one(0.0f))
        .SetMagnitude(one(0.0f))
        .SetTemperature(one(5772.0f));
    ernie::Camera(scene, "/World/Camera").SetScan(tokens::kPushbroom);
    constexpr std::size_t kSize = 32;
    ernie::RenderSettings(scene, "/Render/Settings")
        .SetResolution(glm::dvec2(kSize, kSize));
    const ernie::CameraParams camera =
        ernie::CameraParamsOf(scene, "/World/Camera", kSize, kSize);
    const double f = camera.FocalLengthPixels();
    constexpr std::size_t kBand = 12;
    const double expected = ernie::BlackbodySpectralIrradiance(
        camera.bands.Lower(static_cast<int>(kBand)) +
            0.5 * camera.bands.Width(),
        5772.0, 2.54e-6);
    // Each row's spectral irradiance: its pixels' radiance times
    // their detector pixels' solid angles.
    const auto rows = [&](double fromElevation, double toElevation) {
        const auto at = [](double y, double elevation) {
            return ernie::LookAtTransform(
                {0.0, y, 0.0},
                {std::cos(elevation), y, std::sin(elevation)},
                {0.0, 0.0, 1.0});
        };
        RasterDelegate delegate(*context_);
        scene.Set("/World/Camera", tokens::kTransform,
                  at(-1.0, fromElevation));
        delegate.Step(scene);  // the strip's start
        scene.Set("/World/Camera", tokens::kTransform,
                  at(1.0, toElevation));
        ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
        const Tensor spectral = outputs.at("/Render/Product")
                                    .vars.at("SpectralRadiance")
                                    .Map(ernie::MapDevice::Cpu)
                                    .tensor;
        std::vector<double> sums(kSize, 0.0);
        for (std::size_t row = 0; row < kSize; ++row) {
            for (std::size_t col = 0; col < kSize; ++col) {
                const double x = (static_cast<double>(col) + 0.5 -
                                  camera.PrincipalPoint().x) /
                                 f;
                const double cosine = 1.0 / std::sqrt(1.0 + x * x);
                sums[row] +=
                    SpectralAt(spectral, kBand, row, col, kSize) *
                    cosine * cosine * cosine / (f * f);
            }
        }
        return sums;
    };

    // Held still, every row sees it: a streak down its column.
    for (const double sum : rows(0.0, 0.0)) {
        EXPECT_NEAR(sum, expected, 0.03 * expected);
    }
    // Turning a pixel's angle a row, the rows sweep past it as a
    // frame camera's image holds it: all of it once, over a row or
    // two.
    const double sweep = static_cast<double>(kSize) / f;
    const std::vector<double> swept = rows(0.5 * sweep, -0.5 * sweep);
    double total = 0.0;
    std::size_t lit = 0;
    for (const double sum : swept) {
        total += sum;
        lit += sum > 0.0 ? 1 : 0;
    }
    EXPECT_NEAR(total, expected, 0.03 * expected);
    EXPECT_LE(lit, 2u);
}
