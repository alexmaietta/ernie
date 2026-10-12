// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/raster/delegate.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/volume_schemas.h"
#include "ernie/scene/xform.h"
#include "support/delegate_scenes.h"
#include "support/vulkan_fixture.h"

#include <glm/glm.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
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
