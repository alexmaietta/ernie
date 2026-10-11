// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/raster/delegate.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/terrain_schemas.h"
#include "ernie/scene/tokens.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>

using ernie::BrdfRegistry;
using ernie::DType;
using ernie::ImageEncoding;
using ernie::Path;
using ernie::RenderPrims;
using ernie::Scene;
using ernie::Tensor;
using ernie::TerrainColorMode;
namespace tokens = ernie::tokens;

namespace {

Scene MinimalScene() {
    Scene scene;
    scene.Define("/World/Camera", tokens::kCamera)
        .Set(tokens::kTransform,
             ernie::LookAtTransform({0.0, -100.0, 100.0},
                                    {0.0, 0.0, 0.0},
                                    {0.0, 0.0, 1.0}));
    ernie::Terrain::Define(scene, "/World/Terrain");
    return scene;
}

}  // namespace

TEST(FrameParamsOf, DefaultsMatchTheRendererAndAnUnlitSceneHasNoSun) {
    const Scene scene = MinimalScene();
    const RenderPrims prims = ernie::ResolveRenderPrims(scene, {});
    const ernie::FrameParams params =
        ernie::FrameParamsOf(scene, prims, 64, 32);
    const ernie::FrameParams defaults;
    EXPECT_FALSE(params.sun)
        << "a scene without lights renders black";
    EXPECT_TRUE(params.lights.empty());
    EXPECT_EQ(params.sunDirection, defaults.sunDirection);
    EXPECT_EQ(params.ambient, defaults.ambient);
    EXPECT_EQ(params.exposure, 1.0f);
    EXPECT_TRUE(params.shadows.enabled);
    EXPECT_EQ(params.View().aspect, 2.0f);
    EXPECT_TRUE(params.sunIrradiance.empty());
    EXPECT_EQ(params.camera.width, 64u);
    EXPECT_DOUBLE_EQ(params.overscan, 1.0);
    EXPECT_DOUBLE_EQ(params.camera.LinearExposureScale(), 1.0);
    const ernie::TerrainParams terrain =
        ernie::TerrainParamsOf(scene, prims);
    EXPECT_EQ(terrain.colorMode, ernie::TerrainParams().colorMode);
    EXPECT_EQ(terrain.exaggeration, 1.0f);
    const ernie::MaterialSettings material =
        ernie::MaterialSettingsOf(scene, prims.material,
                                  BrdfRegistry::WithBuiltins());
    EXPECT_EQ(material.brdf, "lambert");
    EXPECT_EQ(material.albedo, glm::vec3(1.0f));

    Scene empty;
    EXPECT_THROW(
        ernie::FrameParamsOf(
            empty, ernie::ResolveRenderPrims(empty, {}), 8, 8),
        std::invalid_argument);
}

TEST(FrameParamsOf, ReadsLightMaterialTerrainAndSettings) {
    Scene scene = MinimalScene();
    ernie::DistantLight::Define(scene, "/World/Sun")
        .SetIntensity(2.0)
        .SetExposure(1.0)
        .SetShadowEnable(false);
    scene.Set("/World/Sun", tokens::kTransform,
              ernie::DistantLightTransform({1.0, 0.0, 1.0}));
    ernie::Camera(scene, "/World/Camera").SetExposure(1.0);
    ernie::Material::Define(scene, "/Looks/Rock")
        .SetBrdf("hapke")
        .SetAlbedo({0.5, 0.4, 0.3})
        .SetAlbedoMapEnable(false)
        .SetDetailTileSize(8.0)
        .SetDetailHexTiling(false);
    scene.Set("/Looks/Rock", "inputs:b",
              0.33);  // a BRDF parameter: an open namespace
    ernie::Terrain(scene, "/World/Terrain")
        .SetMaterialBinding("/Looks/Rock")
        .SetExaggeration(3.0);
    ernie::RenderSettings::Define(scene, "/Render/Settings")
        .SetAmbient(0.3)
        .SetWireframe(tokens::kOverlay)
        .SetShadowCascades(2)
        .SetShadowResolution(1024)
        .SetLodEnable(false)
        .SetLodDetail(6.0);

    const BrdfRegistry brdfs = BrdfRegistry::WithBuiltins();
    const RenderPrims prims = ernie::ResolveRenderPrims(scene, {});
    const ernie::FrameParams frame =
        ernie::FrameParamsOf(scene, prims, 32, 32);
    const float diagonal = std::sqrt(0.5f);
    EXPECT_NEAR(frame.sunDirection.x, diagonal, 1e-6f);
    EXPECT_NEAR(frame.sunDirection.z, diagonal, 1e-6f);
    EXPECT_DOUBLE_EQ(frame.camera.LinearExposureScale(), 2.0)
        << "camera exposure 1 stop";
    ASSERT_EQ(frame.sunIrradiance.size(), 16u);
    double luminanceIntegral = 0.0;
    for (double w = 360.5; w < 830.0; w += 1.0) {
        luminanceIntegral += ernie::CieXyz(w).y;
    }
    EXPECT_NEAR(frame.sunIrradiance[5],
                M_PI * 4.0 / (683.002 * luminanceIntegral), 1e-6)
        << "a flat sun whose white Lambertian luminance is 2 x 2^1 "
           "cd/m^2";
    EXPECT_FALSE(frame.shadows.enabled);
    EXPECT_EQ(frame.ambient, 0.3f);
    EXPECT_EQ(frame.wireframe, ernie::Wireframe::Overlay);
    EXPECT_EQ(frame.shadows.cascades, 2);
    EXPECT_EQ(frame.shadows.resolution, 1024u);

    const ernie::MaterialSettings material =
        ernie::MaterialSettingsOf(scene, prims.material, brdfs);
    EXPECT_EQ(material.brdf, "hapke");
    const ernie::BrdfModel* hapke = brdfs.Find("hapke");
    ASSERT_NE(hapke, nullptr);
    ASSERT_EQ(material.brdfParams.size(), hapke->params.size());
    for (std::size_t i = 0; i < hapke->params.size(); ++i) {
        const float expected = hapke->params[i].name == "b"
                                   ? 0.33f
                                   : hapke->params[i].defaultValue;
        EXPECT_FLOAT_EQ(material.brdfParams[i], expected)
            << hapke->params[i].name;
    }
    EXPECT_EQ(material.albedo, glm::vec3(0.5f, 0.4f, 0.3f));
    EXPECT_FALSE(material.useAlbedoMap);
    EXPECT_EQ(material.detail.tileSize, 8.0f);
    EXPECT_FALSE(material.detail.hexTiling);

    ernie::TerrainParams terrain =
        ernie::TerrainParamsOf(scene, prims);
    EXPECT_EQ(terrain.exaggeration, 3.0f);
    EXPECT_EQ(terrain.colorMode, TerrainColorMode::Shaded);
    EXPECT_FALSE(terrain.lod.enabled);
    EXPECT_EQ(terrain.lod.detail, 6.0f);

    ernie::Terrain(scene, "/World/Terrain")
        .SetColorMode(tokens::kColormap);
    EXPECT_EQ(ernie::TerrainParamsOf(scene, prims).colorMode,
              TerrainColorMode::Colormap);
    ernie::Terrain(scene, "/World/Terrain")
        .SetColorMode(tokens::kHillshade)
        .SetHillshadeAzimuth(270.0)
        .SetHillshadeAltitude(30.0);
    terrain = ernie::TerrainParamsOf(scene, prims);
    EXPECT_EQ(terrain.colorMode, TerrainColorMode::Hillshade);
    EXPECT_EQ(terrain.hillshadeAzimuth, 270.0f);
    const glm::vec3 towards = ernie::HillshadeDirection(terrain);
    EXPECT_NEAR(towards.x, -std::cos(glm::radians(30.0f)), 1e-6f);
    EXPECT_NEAR(towards.y, 0.0f, 1e-6f);
    EXPECT_NEAR(towards.z, 0.5f, 1e-6f);
    ernie::Material(scene, "/Looks/Rock").SetBrdf("nope");
    EXPECT_THROW(
        ernie::MaterialSettingsOf(scene, prims.material, brdfs),
        std::invalid_argument);
}

TEST(GroundParamsOf, ReadsHeightAndCheckerboard) {
    Scene scene;
    ernie::GroundPlane ground =
        ernie::GroundPlane::Define(scene, "/Ground");
    ernie::GroundParams params =
        ernie::GroundParamsOf(scene, "/Ground");
    EXPECT_EQ(params.height, 0.0f);
    EXPECT_FALSE(params.checker);
    EXPECT_EQ(params.checkerColors[1], glm::vec3(0.5f));
    ground.SetHeight(-2.5)
        .SetChecker(true)
        .SetCheckerSize(10.0)
        .SetCheckerColor0({0.2, 0.3, 0.4});
    params = ernie::GroundParamsOf(scene, "/Ground");
    EXPECT_EQ(params.height, -2.5f);
    EXPECT_TRUE(params.checker);
    EXPECT_EQ(params.checkerSize, 10.0f);
    EXPECT_EQ(params.checkerColors[0], glm::vec3(0.2f, 0.3f, 0.4f));
    EXPECT_EQ(ernie::ResolveRenderPrims(scene, {}).ground,
              Path("/Ground"));
    ground.SetCheckerSize(0.0);
    EXPECT_THROW(ernie::GroundParamsOf(scene, "/Ground"),
                 std::invalid_argument);
}
