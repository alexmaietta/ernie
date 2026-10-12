// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/raster/delegate.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/terrain_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/volume_schemas.h"
#include "ernie/scene/xform.h"
#include "ernie/util/nanovdb.h"
#include "support/delegate_scenes.h"
#include "support/vulkan_fixture.h"

#include <glm/glm.hpp>
#include <gtest/gtest.h>

#include <array>
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

TEST_F(RasterDelegateTest, AVolumeShadowsTheGroundBelowIt) {
    // The cube 3 m up, absorbing, over the ground the camera sees
    // from low beside it (its view clear of the cube): the overhead
    // sun reaches the ground below through the cube's whole column.
    constexpr double kSigma = 0.8;
    RasterDelegate delegate(*context_);
    Scene scene = VolumeScene(kSigma, 0.0);
    ernie::DistantLight(scene, "/World/Sun").SetShadowEnable(true);
    scene.GetPrim("/World/Cube")
        .SetXform({.translate = glm::dvec3(0.0, 0.0, 3.0)});
    scene.Set("/World/Camera", tokens::kTransform,
              ernie::LookAtTransform({4.0, 0.0, 0.5}, {0.0, 0.0, 0.0},
                                     {0.0, 0.0, 1.0}));
    const std::vector<double> shaded =
        CentreSpectrum(delegate, scene);
    scene.Set("/World/Cube", tokens::kVisibility,
              std::string(tokens::kInvisible));
    const std::vector<double> lit = CentreSpectrum(delegate, scene);
    const double expected = std::exp(-kSigma * kDepth);
    for (std::size_t b = 0; b < lit.size(); ++b) {
        ASSERT_GT(lit[b], 0.0) << "band " << b;
        EXPECT_NEAR(shaded[b] / lit[b], expected, 0.05 * expected)
            << "band " << b;
    }
}

TEST_F(RasterDelegateTest, EachWorldOfABatchShadowsWithItsOwnVolumes) {
    // Three views from above of a shared floor, of world 0, world 1
    // and every world, under a sun 45 degrees up towards +x: world 0's
    // dense cube 10 m up at x = 10 shades the floor's centre only in
    // the views that see world 0.
    Scene scene;
    const glm::dmat4 above = ernie::LookAtTransform(
        {0.0, 0.0, 60.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0});
    std::vector<double> poses;
    for (int view = 0; view < 3; ++view) {
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                poses.push_back(above[col][row]);
            }
        }
    }
    ernie::Camera::Define(scene, "/World/Camera")
        .SetSensorNoise(false)
        .SetPoses(Tensor::Copy<double>(poses, {3, 4, 4}))
        .SetWorlds(Tensor::Copy<std::int32_t>(
            std::vector<std::int32_t>{0, 1, -1}, {3}));
    scene.Define("/World/Sun", tokens::kDistantLight)
        .Set(tokens::kTransform,
             ernie::DistantLightTransform({1.0, 0.0, 1.0}));
    DefineQuad(scene, "/World/Floor", 20.0f);
    ernie::Xform::Define(scene, "/Envs/Env0").SetWorld(0);
    ernie::Volume volume =
        ernie::Volume::Define(scene, "/Envs/Env0/Smoke");
    volume.GetPrim().SetXform({.translate = glm::dvec3(10.0, 0.0, 10.0),
                               .scale = glm::dvec3(3.0)});
    ernie::OpenVDBAsset::Define(scene, "/Envs/Env0/Smoke/Density")
        .SetFilePath(CubeFile());
    volume.SetField("density", "/Envs/Env0/Smoke/Density")
        .SetMaterialBinding("/Looks/Smoke");
    ernie::Material::Define(scene, "/Looks/Smoke").SetDensity(3.0);
    ernie::Xform::Define(scene, "/Envs/Env1").SetWorld(1);
    ernie::RenderProduct::Define(scene, "/Render/Batch")
        .SetCamera("/World/Camera")
        .SetResolution(glm::dvec2(kSize, kSize));
    ernie::RenderSettings::Define(scene, "/Render/Settings")
        .SetAmbient(0.1);

    RasterDelegate delegate(*context_);
    const Path batchOnly[] = {"/Render/Batch"};
    const Tensor images = delegate.Step(scene, batchOnly)
                              .at("/Render/Batch")
                              .vars.at("LdrColor")
                              .Map(ernie::MapDevice::Cpu)
                              .tensor;
    const auto centre = [&](std::size_t view) {
        const std::uint8_t* texel =
            images.As<std::uint8_t>() +
            ((view * kSize + kSize / 2) * kSize + kSize / 2) * 4;
        return texel[0] + texel[1] + texel[2];
    };
    EXPECT_LT(centre(0), centre(1) / 2)
        << "world 0's smoke shades the floor in its view";
    EXPECT_LT(centre(2), centre(1) / 2)
        << "a view of every world sees every world's shadows";
    EXPECT_GT(centre(1), 100)
        << "world 1 has no smoke, so its floor is lit";
}

TEST_F(RasterDelegateTest, APointInstancerPlacesVolumesInTheirWorlds) {
    // The cube as an instancer's prototype, two instances where it
    // stood, in worlds 1 and 0: the camera's world 0 sees one of
    // them, and the prototype shows only through its instances.
    constexpr double kSigma = 1.2;
    RasterDelegate delegate(*context_);
    Scene scene = VolumeScene(kSigma, 0.0);
    scene.Set("/World/Camera", tokens::kWorld, 0);
    scene.Set("/World/Cube", tokens::kVisibility,
              std::string(tokens::kInvisible));
    const std::vector<double> bare = CentreSpectrum(delegate, scene);
    scene.Set("/World/Cube", tokens::kVisibility,
              std::string(tokens::kInherited));
    const std::array<float, 6> positions{0, 0, 0, 0, 0, 0};
    ernie::PointInstancer::Define(scene, "/World/Smoke")
        .SetPositions(Tensor::Copy<float>(positions, {2, 3}));
    scene.Set("/World/Smoke", tokens::kPrototypes,
              std::vector<Path>{"/World/Cube"});
    const auto worlds = [&](std::int32_t first, std::int32_t second) {
        scene.Write("/World/Smoke", tokens::kWorlds,
                    Tensor::Copy<std::int32_t>(
                        std::vector<std::int32_t>{first, second}, {2}));
        return CentreSpectrum(delegate, scene);
    };
    const std::vector<double> one = worlds(1, 0);
    const std::vector<double> none = worlds(1, 1);
    const double expected = std::exp(-kSigma * kDepth);
    for (std::size_t b = 0; b < bare.size(); ++b) {
        ASSERT_GT(bare[b], 0.0) << "band " << b;
        EXPECT_NEAR(one[b] / bare[b], expected, 0.02 * expected)
            << "band " << b;
        EXPECT_NEAR(none[b], bare[b], 1e-6 * bare[b]) << "band " << b;
    }
}

TEST_F(RasterDelegateTest, AVolumeReadsItsGridFromAnArray) {
    // The cube's grid written as an array (as Warp's Volume.array()
    // gives it) shades as the file does, and is read again only when
    // written again.
    constexpr double kSigma = 1.2;
    RasterDelegate delegate(*context_);
    Scene scene = VolumeScene(kSigma, 0.0);
    const std::vector<double> fromFile =
        CentreSpectrum(delegate, scene);
    const std::vector<std::uint8_t> bytes =
        ernie::ReadNanoVdb(CubeFile()).front().buffer;
    ernie::NanoVDBField::Define(scene, "/World/Cube/Grids")
        .SetBuffer(Tensor::Copy<std::uint8_t>(
            bytes, {static_cast<std::int64_t>(bytes.size())}));
    ernie::Volume(scene, "/World/Cube")
        .SetField("density", "/World/Cube/Grids");
    const std::vector<double> fromArray =
        CentreSpectrum(delegate, scene);
    EXPECT_EQ(delegate.Renderer().Volumes().GridCount(), 1u);
    for (std::size_t b = 0; b < fromFile.size(); ++b) {
        EXPECT_NEAR(fromArray[b], fromFile[b], 1e-4 * fromFile[b])
            << "band " << b;
    }
    // Only the field's prim names it, so a new array replaces it.
    scene.Write("/World/Cube/Grids", tokens::kNanoVdbBuffer,
                Tensor::Copy<std::uint8_t>(
                    bytes, {static_cast<std::int64_t>(bytes.size())}));
    const std::vector<double> again = CentreSpectrum(delegate, scene);
    EXPECT_EQ(delegate.Renderer().Volumes().GridCount(), 1u);
    EXPECT_NEAR(again[0], fromArray[0], 1e-4 * fromArray[0]);
}

TEST_F(RasterDelegateTest, AVolumeReadsItsGridFromADeviceArray) {
    // The cube's grid in CUDA memory, as a simulation writes it each
    // step, shades as the file does; written again at its size, it is
    // copied over itself in the pool.
    constexpr double kSigma = 1.2;
    RasterDelegate delegate(*context_);
    std::string why;
    ernie::ComputeInterop* cuda = delegate.Compute(&why);
    if (cuda == nullptr) {
        GTEST_SKIP() << "No CUDA interop: " << why;
    }
    Scene scene = VolumeScene(kSigma, 0.0);
    const std::vector<double> fromFile =
        CentreSpectrum(delegate, scene);
    const std::vector<std::uint8_t> bytes =
        ernie::ReadNanoVdb(CubeFile()).front().buffer;
    const Tensor host = Tensor::Copy<std::uint8_t>(
        bytes, {static_cast<std::int64_t>(bytes.size())});
    ernie::NanoVDBField::Define(scene, "/World/Cube/Grids");
    scene.Write("/World/Cube/Grids", tokens::kNanoVdbBuffer,
                cuda->Upload(host));
    ernie::Volume(scene, "/World/Cube")
        .SetField("density", "/World/Cube/Grids");
    const std::vector<double> fromDevice =
        CentreSpectrum(delegate, scene);
    const std::size_t poolBytes =
        delegate.Renderer().Volumes().PoolBytes();
    scene.Write("/World/Cube/Grids", tokens::kNanoVdbBuffer,
                cuda->Upload(host));
    const std::vector<double> again = CentreSpectrum(delegate, scene);
    EXPECT_EQ(delegate.Renderer().Volumes().PoolBytes(), poolBytes);
    for (std::size_t b = 0; b < fromFile.size(); ++b) {
        EXPECT_NEAR(fromDevice[b], fromFile[b], 1e-4 * fromFile[b])
            << "band " << b;
        EXPECT_NEAR(again[b], fromFile[b], 1e-4 * fromFile[b])
            << "band " << b;
    }
}

TEST_F(RasterDelegateTest, AGridBuiltFromDenseValuesShadesAsTheFile) {
    // The cube's 16^3 ones built by Ernie (DenseNanoVdbGrid) rather
    // than Warp: the same values at the same place.
    constexpr double kSigma = 1.2;
    RasterDelegate delegate(*context_);
    Scene scene = VolumeScene(kSigma, 0.0);
    const std::vector<double> fromFile =
        CentreSpectrum(delegate, scene);
    const std::vector<std::uint8_t> bytes = ernie::DenseNanoVdbGrid(
        std::vector<float>(16 * 16 * 16, 1.0f), glm::ivec3(16), 0.1,
        glm::dvec3(-0.75));
    ernie::NanoVDBField::Define(scene, "/World/Cube/Grids")
        .SetBuffer(Tensor::Copy<std::uint8_t>(
            bytes, {static_cast<std::int64_t>(bytes.size())}));
    ernie::Volume(scene, "/World/Cube")
        .SetField("density", "/World/Cube/Grids");
    const std::vector<double> built = CentreSpectrum(delegate, scene);
    for (std::size_t b = 0; b < fromFile.size(); ++b) {
        EXPECT_NEAR(built[b], fromFile[b], 1e-4 * fromFile[b])
            << "band " << b;
    }
}

TEST_F(RasterDelegateTest, OctavesAddLightScatteredManyTimes) {
    // Wrenninge's octaves in the overhead column: order i scatters
    // a^i of the light, attenuated b^i as much towards the sun, so an
    // isotropic column of optical depth tau gives
    // sum a^i (1 - exp(-(1 + b^i) tau)) / (1 + b^i) of what single
    // scattering's (1 - exp(-2 tau)) / 2 normalizes.
    constexpr double kSigma = 2.0;
    constexpr int kOctaves = 4;
    // a at most b, as energy allows (a larger a is clamped to b).
    constexpr double a = 0.4;
    constexpr double b = 0.6;
    RasterDelegate delegate(*context_);
    Scene scene = VolumeScene(kSigma, 0.0);
    ernie::Material(scene, "/Looks/Gas")
        .SetVolumeAlbedo(glm::dvec3(1.0))
        .SetPhaseFunction(std::string(tokens::kIsotropic))
        .SetScatteringOctaves(kOctaves)
        .SetOctaveAttenuation(a)
        .SetOctaveExtinction(b);
    const std::vector<double> ratio = ScatteredOverEach(
        delegate, scene, kSigma, 1.0 / (4.0 * std::numbers::pi));
    const double tau = kSigma * kDepth;
    double expected = 0.0;
    for (int i = 0; i < kOctaves; ++i) {
        const double bi = std::pow(b, i);
        expected += std::pow(a, i) *
                    (1.0 - std::exp(-(1.0 + bi) * tau)) / (1.0 + bi);
    }
    expected /= (1.0 - std::exp(-2.0 * tau)) / 2.0;
    for (std::size_t band = 0; band < ratio.size(); ++band) {
        EXPECT_NEAR(ratio[band], expected, 0.04 * expected)
            << "band " << band;
    }
    EXPECT_GT(expected, 1.5) << "the octaves add light";
}

TEST_F(RasterDelegateTest,
       TheSkyAndGroundLightAVolumeInAnAtmosphere) {
    // Under Earth's air, a white scatterer takes the sky's and the
    // ground's light besides the sun's; its sky light scale off, the
    // sun's alone.
    RasterDelegate delegate(*context_);
    Scene scene = VolumeScene(1.0, 0.0);
    ernie::Atmosphere::Define(scene, "/World/Air")
        .SetPreset(std::string(tokens::kEarth));
    ernie::Material(scene, "/Looks/Gas")
        .SetVolumeAlbedo(glm::dvec3(1.0))
        .SetPhaseFunction(std::string(tokens::kIsotropic));
    scene.Remove("/World/Terrain");
    const std::vector<double> lit = CentreSpectrum(delegate, scene);
    ernie::Material(scene, "/Looks/Gas").SetSkyLight(0.0);
    const std::vector<double> sunOnly =
        CentreSpectrum(delegate, scene);
    double litSum = 0.0;
    double sunSum = 0.0;
    for (std::size_t b = 0; b < lit.size(); ++b) {
        litSum += lit[b];
        sunSum += sunOnly[b];
    }
    ASSERT_GT(sunSum, 0.0);
    // The overhead sun's sky gives about a sixth of its light, the
    // ground (black here, no terrain: the atmosphere's albedo) more.
    EXPECT_GT(litSum, 1.03 * sunSum);
    EXPECT_LT(litSum, 1.6 * sunSum);
}

TEST_F(RasterDelegateTest, TheAirInFrontOfAVolumeShows) {
    // A black wall of absorbing smoke 20 km off across a level view:
    // the air between shows in front of it, rather than the wall
    // blacking out the sky's air light along the whole ray.
    RasterDelegate delegate(*context_);
    Scene scene = VolumeScene(0.01, 0.0);
    ernie::Atmosphere::Define(scene, "/World/Air")
        .SetPreset(std::string(tokens::kEarth));
    ernie::DistantLight(scene, "/World/Sun")
        .SetTransform(ernie::DistantLightTransform({0.3, -1.0, 0.6}));
    scene.Remove("/World/Terrain");
    scene.GetPrim("/World/Cube")
        .SetXform({.translate = glm::dvec3(0.0, 20000.0, 500.0),
                   .scale = glm::dvec3(2000.0)});
    scene.Set("/World/Camera", tokens::kTransform,
              ernie::LookAtTransform({0.0, 0.0, 500.0},
                                     {0.0, 1.0, 500.0},
                                     {0.0, 0.0, 1.0}));
    const std::vector<double> walled =
        CentreSpectrum(delegate, scene);
    scene.Set("/World/Cube", tokens::kVisibility,
              std::string(tokens::kInvisible));
    const std::vector<double> sky = CentreSpectrum(delegate, scene);
    double walledSum = 0.0;
    double skySum = 0.0;
    for (std::size_t b = 0; b < sky.size(); ++b) {
        walledSum += walled[b];
        skySum += sky[b];
    }
    // 20 km of the 8 km scale height's air at 500 m: much of the
    // horizon's light, but less than the sky's along the whole ray.
    EXPECT_GT(walledSum, 0.3 * skySum);
    EXPECT_LT(walledSum, 0.95 * skySum);
}
