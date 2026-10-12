// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/raster/delegate.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/terrain_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/volume_schemas.h"
#include "ernie/scene/xform.h"
#include "support/delegate_scenes.h"
#include "support/vulkan_fixture.h"

#include <glm/glm.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <string>
#include <vector>

using ernie::Path;
using ernie::RasterDelegate;
using ernie::Scene;
using ernie::Tensor;
using namespace ernie::testing;
namespace tokens = ernie::tokens;

namespace {

// tools/volume_fixtures.py: 16^3 ones in 0.1 m voxels about the
// origin, so a ray along an axis through it crosses 1.6 m of density
// 1 (trilinear ramps at its faces included).
constexpr double kDepth = 1.6;

std::string CubeFile() {
    return (std::filesystem::path(ERNIE_TEST_FIXTURES) / "volumes" /
            "cube.nvdb")
        .string();
}

// FlatScene's camera 3 m above the ground, and the cube centred 1 m
// above it, its material's density `sigma` (1/m) and emission
// `emission` (cd/m^2 per m, white).
Scene VolumeScene(double sigma, double emission) {
    Scene scene = FlatScene();
    scene.Set("/World/Camera", tokens::kTransform,
              ernie::LookAtTransform({0.0, 0.0, 3.0}, {0.0, 0.0, 0.0},
                                     {0.0, 1.0, 0.0}));
    ernie::Volume volume =
        ernie::Volume::Define(scene, "/World/Cube");
    volume.GetPrim().SetXform(
        {.translate = glm::dvec3(0.0, 0.0, 1.0)});
    ernie::OpenVDBAsset::Define(scene, "/World/Cube/Density")
        .SetFilePath(CubeFile());
    volume.SetField("density", "/World/Cube/Density")
        .SetMaterialBinding("/Looks/Gas");
    ernie::Material::Define(scene, "/Looks/Gas")
        .SetDensity(sigma)
        .SetEmissionStrength(emission);
    return scene;
}

std::vector<double> CentreSpectrum(RasterDelegate& delegate,
                                   const Scene& scene) {
    ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
    Tensor spectral = outputs.at("/Render/Product")
                          .vars.at("SpectralRadiance")
                          .Map(ernie::MapDevice::Cpu)
                          .tensor;
    const auto* values = static_cast<const float*>(spectral.Data());
    const auto bands =
        static_cast<std::size_t>(spectral.Shape().front());
    std::vector<double> spectrum;
    constexpr std::size_t kCentre = kVarSize / 2;
    for (std::size_t b = 0; b < bands; ++b) {
        spectrum.push_back(
            values[(b * kVarSize + kCentre) * kVarSize + kCentre]);
    }
    return spectrum;
}

}  // namespace

TEST_F(RasterDelegateTest, AVolumeDimsWhatIsBehindIt) {
    constexpr double kSigma = 1.2;
    RasterDelegate delegate(*context_);
    Scene scene = VolumeScene(kSigma, 0.0);
    const std::vector<double> through =
        CentreSpectrum(delegate, scene);
    EXPECT_EQ(delegate.Renderer().Volumes().GridCount(), 1u);
    scene.Set("/World/Cube", tokens::kVisibility,
              std::string(tokens::kInvisible));
    const std::vector<double> bare = CentreSpectrum(delegate, scene);
    ASSERT_EQ(through.size(), bare.size());
    const double expected = std::exp(-kSigma * kDepth);
    for (std::size_t b = 0; b < bare.size(); ++b) {
        ASSERT_GT(bare[b], 0.0) << "band " << b;
        EXPECT_NEAR(through[b] / bare[b], expected, 0.02 * expected)
            << "band " << b;
    }
}

TEST_F(RasterDelegateTest, AVolumeEmitsAlongItsDepth) {
    // Against the ground, which the emission is added to: L =
    // L_ground T + e (1 - T) / sigma, e per unit density.
    RasterDelegate delegate(*context_);
    Scene scene = VolumeScene(1e-4, 0.0);
    const std::vector<double> ground =
        CentreSpectrum(delegate, scene);
    constexpr double kEmission = 5.0;
    scene = VolumeScene(1e-4, kEmission);
    const std::vector<double> thin = CentreSpectrum(delegate, scene);
    constexpr double kSigma = 2.0;
    scene = VolumeScene(kSigma, kEmission);
    const std::vector<double> dense = CentreSpectrum(delegate, scene);
    const double transmittance = std::exp(-kSigma * kDepth);
    for (std::size_t b = 0; b < ground.size(); ++b) {
        // Thin, all of it shows: e times the depth.
        const double e = (thin[b] - ground[b]) / kDepth;
        ASSERT_GT(e, 0.0) << "band " << b;
        const double expected = ground[b] * transmittance +
                                e * (1.0 - transmittance) / kSigma;
        EXPECT_NEAR(dense[b], expected, 0.02 * expected)
            << "band " << b;
    }
}

TEST_F(RasterDelegateTest, AViewSeesOnlyItsWorldsVolumes) {
    RasterDelegate delegate(*context_);
    Scene scene = VolumeScene(3.0, 0.0);
    scene.Set("/World/Camera", tokens::kWorld, 0);
    scene.Set("/World/Cube", tokens::kWorld, 1);
    const std::vector<double> other = CentreSpectrum(delegate, scene);
    scene.Set("/World/Cube", tokens::kWorld, 0);
    const std::vector<double> own = CentreSpectrum(delegate, scene);
    scene.Set("/World/Cube", tokens::kVisibility,
              std::string(tokens::kInvisible));
    const std::vector<double> bare = CentreSpectrum(delegate, scene);
    for (std::size_t b = 0; b < bare.size(); ++b) {
        EXPECT_NEAR(other[b], bare[b], 1e-6 * bare[b])
            << "band " << b;
        EXPECT_LT(own[b], 0.05 * bare[b]) << "band " << b;
    }
}

namespace {

// The sun overhead scattered once towards a camera straight above:
// the light and the view cross the same column, so whatever its
// density, L = p E (1 - exp(-2 tau)) / 2 for a white albedo, phase
// p at phase angle 0 and the sun's irradiance E. E comes from the
// ground without the cube (Lambertian, albedo 0.5: L = 0.5 E / pi);
// the scattering is seen against the empty background.
std::vector<double> ScatteredOverEach(RasterDelegate& delegate,
                                      Scene scene, double sigma,
                                      double phase) {
    // The column shadows itself: the sun casts shadows.
    ernie::DistantLight(scene, "/World/Sun").SetShadowEnable(true);
    scene.Set("/World/Cube", tokens::kVisibility,
              std::string(tokens::kInvisible));
    const std::vector<double> ground =
        CentreSpectrum(delegate, scene);
    scene.Set("/World/Cube", tokens::kVisibility,
              std::string(tokens::kInherited));
    scene.Remove("/World/Terrain");
    const std::vector<double> scattered =
        CentreSpectrum(delegate, scene);
    const double fraction =
        phase * (1.0 - std::exp(-2.0 * sigma * kDepth)) / 2.0;
    std::vector<double> ratio;
    for (std::size_t b = 0; b < ground.size(); ++b) {
        const double irradiance = ground[b] * std::numbers::pi / 0.5;
        ratio.push_back(scattered[b] / (fraction * irradiance));
    }
    return ratio;
}

}  // namespace

TEST_F(RasterDelegateTest, AVolumeScattersTheSun) {
    constexpr double kSigma = 1.0;
    constexpr double kFourPi = 4.0 * std::numbers::pi;
    struct Case {
        std::string phase;
        double anisotropy;
        double phaseAtZero;  // p(g = 0) times 4 pi
    };
    // Henyey-Greenstein at phase angle 0, scattering straight back:
    // (1 - g^2) / (1 + g)^3; Hapke's double lobe with b = 0.3 and
    // c = 0.4 weighs its backward lobe (1 + c) / 2.
    constexpr double b = 0.3;
    constexpr double c = 0.4;
    const Case cases[] = {
        {std::string(tokens::kIsotropic), 0.0, 1.0},
        {std::string(tokens::kHenyeyGreenstein), 0.5,
         0.75 / (1.5 * 1.5 * 1.5)},
        {std::string(tokens::kHenyeyGreenstein), -0.5,
         0.75 / (0.5 * 0.5 * 0.5)},
        {std::string(tokens::kDoubleHenyeyGreenstein), 0.0,
         0.5 * (1.0 + c) * (1.0 - b * b) / std::pow(1.0 - b, 3.0) +
             0.5 * (1.0 - c) * (1.0 - b * b) /
                 std::pow(1.0 + b, 3.0)},
        {std::string(tokens::kRayleigh), 0.0, 1.5},
    };
    RasterDelegate delegate(*context_);
    for (const Case& test : cases) {
        Scene scene = VolumeScene(kSigma, 0.0);
        ernie::Material(scene, "/Looks/Gas")
            .SetVolumeAlbedo(glm::dvec3(1.0))
            .SetPhaseFunction(test.phase)
            .SetAnisotropy(test.anisotropy);
        scene.Set("/Looks/Gas", tokens::kPhaseB, b);
        scene.Set("/Looks/Gas", tokens::kPhaseC, c);
        const std::vector<double> ratio = ScatteredOverEach(
            delegate, scene, kSigma, test.phaseAtZero / kFourPi);
        for (std::size_t band = 0; band < ratio.size(); ++band) {
            EXPECT_NEAR(ratio[band], 1.0, 0.03)
                << test.phase << " g " << test.anisotropy << ", band "
                << band;
        }
    }
}

namespace {

// The cube seen side on against the empty background (no ground),
// from 4 m along -y at its height, a white isotropic scatterer of
// density `sigma`, lit by FlatScene's sun alone, which casts shadows
// (volumes' too).
Scene SideScene(double sigma) {
    Scene scene = VolumeScene(sigma, 0.0);
    scene.Remove("/World/Terrain");
    ernie::DistantLight(scene, "/World/Sun").SetShadowEnable(true);
    scene.Set("/World/Camera", tokens::kTransform,
              ernie::LookAtTransform({0.0, -4.0, 1.0},
                                     {0.0, 0.0, 1.0},
                                     {0.0, 0.0, 1.0}));
    ernie::Material(scene, "/Looks/Gas")
        .SetVolumeAlbedo(glm::dvec3(1.0))
        .SetPhaseFunction(std::string(tokens::kIsotropic));
    return scene;
}

double Sum(const std::vector<double>& values) {
    double sum = 0.0;
    for (const double value : values) {
        sum += value;
    }
    return sum;
}

}  // namespace

TEST_F(RasterDelegateTest, GeometryShadowsAVolume) {
    RasterDelegate delegate(*context_);
    Scene scene = SideScene(1.0);
    ernie::DistantLight(scene, "/World/Sun").SetShadowEnable(true);
    const double lit = Sum(CentreSpectrum(delegate, scene));
    ASSERT_GT(lit, 0.0);
    // A roof between the sun and the cube.
    DefineQuad(scene, "/World/Roof", 2.0f);
    scene.GetPrim("/World/Roof")
        .SetXform({.translate = glm::dvec3(0.0, 0.0, 2.6)});
    const double shaded = Sum(CentreSpectrum(delegate, scene));
    EXPECT_LT(shaded, 0.02 * lit);
}

TEST_F(RasterDelegateTest, AVolumeShadowsAnother) {
    // A second cube above the first, out of the camera's view,
    // between it and the sun overhead: the light reaching the first
    // crosses the second's whole column, so its scattering drops by
    // exp(-1.6 sigma) in every band.
    constexpr double kAbove = 0.8;
    RasterDelegate delegate(*context_);
    Scene scene = SideScene(1.0);
    const std::vector<double> alone = CentreSpectrum(delegate, scene);
    ernie::Volume upper =
        ernie::Volume::Define(scene, "/World/Upper");
    // Twice as wide, so over the whole first cube its columns are
    // whole (not its ramps at its sides).
    upper.GetPrim().SetXform({.translate = glm::dvec3(0.0, 0.0, 3.5),
                              .scale = glm::dvec3(2.0, 2.0, 1.0)});
    // Its own field: a field's transform places its grid.
    ernie::OpenVDBAsset::Define(scene, "/World/Upper/Density")
        .SetFilePath(CubeFile());
    upper.SetField("density", "/World/Upper/Density")
        .SetMaterialBinding("/Looks/Upper");
    ernie::Material::Define(scene, "/Looks/Upper").SetDensity(kAbove);
    const std::vector<double> under = CentreSpectrum(delegate, scene);
    const double expected = std::exp(-kAbove * kDepth);
    for (std::size_t b = 0; b < alone.size(); ++b) {
        ASSERT_GT(alone[b], 0.0) << "band " << b;
        EXPECT_NEAR(under[b] / alone[b], expected, 0.03 * expected)
            << "band " << b;
    }
}

TEST_F(RasterDelegateTest, TheAirDimsAndReddensTheSunInAVolume) {
    // The sun 5 degrees up: what the cube scatters, less the sky
    // behind it dimmed through it, against the same without air.
    constexpr double kSigma = 1.0;
    constexpr double kElevation = 5.0 * std::numbers::pi / 180.0;
    RasterDelegate delegate(*context_);
    Scene scene = SideScene(kSigma);
    scene.Set("/World/Sun", tokens::kTransform,
              ernie::DistantLightTransform(
                  {std::cos(kElevation), 0.0, std::sin(kElevation)}));
    const std::vector<double> bare = CentreSpectrum(delegate, scene);
    ernie::Atmosphere::Define(scene, "/World/Air");
    const std::vector<double> withAir =
        CentreSpectrum(delegate, scene);
    scene.Set("/World/Cube", tokens::kVisibility,
              std::string(tokens::kInvisible));
    const std::vector<double> sky = CentreSpectrum(delegate, scene);
    const double through = std::exp(-kSigma * kDepth);
    std::vector<double> ratio;
    for (std::size_t b = 0; b < bare.size(); ++b) {
        ASSERT_GT(bare[b], 0.0) << "band " << b;
        ratio.push_back((withAir[b] - sky[b] * through) / bare[b]);
        EXPECT_LT(ratio.back(), 1.0) << "band " << b;
        EXPECT_GT(ratio.back(), 0.0) << "band " << b;
    }
    // Blue light, scattered out on the long way through the air, dims
    // most.
    EXPECT_LT(ratio.front(), 0.7 * ratio.back());
}

TEST_F(RasterDelegateTest, ARectLightScattersAsASphereOfItsArea) {
    // Far above, a 1 m^2 rect facing down and a sphere of the same
    // area, of one intensity: the same light along their axes.
    RasterDelegate delegate(*context_);
    std::vector<double> spectra[2];
    for (int i = 0; i < 2; ++i) {
        Scene scene = SideScene(1.0);
        scene.Remove("/World/Sun");
        const auto type =
            i == 0 ? tokens::kRectLight : tokens::kSphereLight;
        scene.Define("/World/Lamp", type)
            .Set(tokens::kTransform,
                 ernie::LookAtTransform({0.0, 0.0, 9.0},
                                        {0.0, 0.0, 0.0},
                                        {0.0, 1.0, 0.0}));
        ernie::Light(scene, "/World/Lamp")
            .SetIntensity(1000.0)
            .SetShadowEnable(false);
        if (i == 1) {
            ernie::SphereLight(scene, "/World/Lamp")
                .SetRadius(1.0 / std::sqrt(std::numbers::pi));
        }
        spectra[i] = CentreSpectrum(delegate, scene);
    }
    for (std::size_t b = 0; b < spectra[0].size(); ++b) {
        ASSERT_GT(spectra[1][b], 0.0) << "band " << b;
        EXPECT_NEAR(spectra[0][b] / spectra[1][b], 1.0, 0.03)
            << "band " << b;
    }
}

TEST_F(RasterDelegateTest, ALaserBeamShowsInAVolume) {
    RasterDelegate delegate(*context_);
    Scene scene = SideScene(1.0);
    scene.Remove("/World/Sun");
    ernie::LaserLight::Define(scene, "/World/Laser")
        .SetPower(1.0)
        .SetWaistRadius(glm::dvec2(0.05))
        .SetShadowEnable(false);
    // Down through the cube's centre, then away from it.
    scene.Set("/World/Laser", tokens::kTransform,
              ernie::LookAtTransform({0.0, 0.0, 6.0}, {0.0, 0.0, 0.0},
                                     {0.0, 1.0, 0.0}));
    const double through = Sum(CentreSpectrum(delegate, scene));
    scene.Set("/World/Laser", tokens::kTransform,
              ernie::LookAtTransform({0.0, 0.0, 6.0},
                                     {0.0, 0.0, 12.0},
                                     {0.0, 1.0, 0.0}));
    const double away = Sum(CentreSpectrum(delegate, scene));
    EXPECT_GT(through, 0.0);
    EXPECT_EQ(away, 0.0);
}

TEST_F(RasterDelegateTest, LightLinksHoldForVolumes) {
    RasterDelegate delegate(*context_);
    Scene scene = SideScene(1.0);
    const std::vector<double> sun = CentreSpectrum(delegate, scene);
    // A second sun overhead, as bright and as shadowed: twice the
    // light, unless its light link leaves the volume out.
    ernie::DistantLight::Define(scene, "/World/Moon")
        .SetShadowEnable(true);
    scene.Set("/World/Moon", tokens::kTransform,
              ernie::DistantLightTransform({0.0, 0.0, 1.0}));
    const std::vector<double> both = CentreSpectrum(delegate, scene);
    scene.Set("/World/Moon", tokens::kLightLinkIncludeRoot, false);
    const std::vector<double> unlinked =
        CentreSpectrum(delegate, scene);
    for (std::size_t b = 0; b < sun.size(); ++b) {
        EXPECT_NEAR(both[b] / sun[b], 2.0, 0.02) << "band " << b;
        EXPECT_NEAR(unlinked[b] / sun[b], 1.0, 0.02) << "band " << b;
    }

    // A volume the sun's shadow link leaves out shadows nothing.
    scene.Remove("/World/Moon");
    ernie::Volume upper =
        ernie::Volume::Define(scene, "/World/Upper");
    upper.GetPrim().SetXform({.translate = glm::dvec3(0.0, 0.0, 3.5),
                              .scale = glm::dvec3(2.0, 2.0, 1.0)});
    ernie::OpenVDBAsset::Define(scene, "/World/Upper/Density")
        .SetFilePath(CubeFile());
    upper.SetField("density", "/World/Upper/Density");
    const std::vector<double> shadowed =
        CentreSpectrum(delegate, scene);
    scene.Set("/World/Sun", tokens::kShadowLinkExcludes,
              std::vector<Path>{"/World/Upper"});
    const std::vector<double> unshadowed =
        CentreSpectrum(delegate, scene);
    for (std::size_t b = 0; b < sun.size(); ++b) {
        EXPECT_LT(shadowed[b], 0.5 * sun[b]) << "band " << b;
        EXPECT_NEAR(unshadowed[b] / sun[b], 1.0, 0.02)
            << "band " << b;
    }
}

TEST_F(RasterDelegateTest,
       ALightWithoutShadowsCastsNoneThroughVolumes) {
    RasterDelegate delegate(*context_);
    Scene scene = SideScene(1.0);
    ernie::DistantLight(scene, "/World/Sun").SetShadowEnable(false);
    const std::vector<double> alone = CentreSpectrum(delegate, scene);
    ernie::Volume upper =
        ernie::Volume::Define(scene, "/World/Upper");
    upper.GetPrim().SetXform({.translate = glm::dvec3(0.0, 0.0, 3.5),
                              .scale = glm::dvec3(2.0, 2.0, 1.0)});
    ernie::OpenVDBAsset::Define(scene, "/World/Upper/Density")
        .SetFilePath(CubeFile());
    upper.SetField("density", "/World/Upper/Density");
    const std::vector<double> under = CentreSpectrum(delegate, scene);
    for (std::size_t b = 0; b < alone.size(); ++b) {
        EXPECT_NEAR(under[b] / alone[b], 1.0, 0.02) << "band " << b;
    }
}

TEST_F(RasterDelegateTest, IesProfilesShapeLampsInAVolume) {
    // A lamp 9 m above the cube, which it sees within 6 degrees of
    // its axis: a profile of 50 cd out to 10 degrees against a
    // uniform 100 cd one.
    const char* half = R"(IESNA:LM-63-2002
TILT=NONE
1 -1 1 3 1 1 2 0 0 0
1 1 10
0 10 180
0
50 50 100
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
    const std::filesystem::path files[] = {
        directory / "ernie_raster_volumes_half.ies",
        directory / "ernie_raster_volumes_uniform.ies"};
    std::ofstream(files[0]) << half;
    std::ofstream(files[1]) << uniform;
    RasterDelegate delegate(*context_);
    std::vector<double> spectra[2];
    for (int i = 0; i < 2; ++i) {
        Scene scene = SideScene(1.0);
        scene.Remove("/World/Sun");
        ernie::SphereLight::Define(scene, "/World/Lamp")
            .SetIntensity(1.0)
            .SetShapingIesFile(files[i].string())
            .SetShadowEnable(false);
        scene.Set("/World/Lamp", tokens::kTransform,
                  ernie::LookAtTransform({0.0, 0.0, 10.0},
                                         {0.0, 0.0, 0.0},
                                         {0.0, 1.0, 0.0}));
        spectra[i] = CentreSpectrum(delegate, scene);
    }
    for (std::size_t b = 0; b < spectra[0].size(); ++b) {
        ASSERT_GT(spectra[1][b], 0.0) << "band " << b;
        EXPECT_NEAR(spectra[0][b] / spectra[1][b], 0.5, 0.01)
            << "band " << b;
    }
    std::filesystem::remove(files[0]);
    std::filesystem::remove(files[1]);
}
