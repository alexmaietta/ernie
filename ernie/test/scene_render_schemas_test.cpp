// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/render_schemas.h"

#include "ernie/raster/terrain_drawable.h"
#include "ernie/scene/prim_type.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/tokens.h"
#include "ernie/session/frame_params.h"
#include "ernie/session/resolve.h"

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

namespace {

using ernie::AttributeKind;
using ernie::Path;
using ernie::PrimTypeDef;
using ernie::Scene;
using ernie::Value;
namespace tokens = ernie::tokens;

TEST(RenderSchemas, FallbacksAreTheRenderDefaults) {
    // Each declared fallback against the default of the field it sets, so they cannot drift.
    Scene scene;
    const ernie::RenderSettings settings = ernie::RenderSettings::Define(scene, "/Render/Settings");
    const ernie::FrameParams frame;
    const ernie::ShadowSettings& shadows = frame.shadows;
    EXPECT_FLOAT_EQ(static_cast<float>(settings.GetAmbient()), frame.ambient);
    EXPECT_FLOAT_EQ(static_cast<float>(settings.GetDomeShadowRadius()), frame.domeShadowRadius);
    EXPECT_EQ(static_cast<std::uint32_t>(settings.GetDomeDirections()), frame.domeDirections);
    EXPECT_EQ(settings.GetDomeLightMode(), "approximate");
    EXPECT_EQ(frame.domeMode, ernie::DomeLightMode::Approximate);
    EXPECT_FLOAT_EQ(static_cast<float>(std::exp2(settings.GetExposureCompensation())),
                    frame.exposure);
    EXPECT_EQ(settings.GetMeter(), frame.meter);
    EXPECT_FLOAT_EQ(static_cast<float>(settings.GetLodMeshError()), frame.meshLodError);
    EXPECT_EQ(glm::vec3(settings.GetClearColor()), frame.clearColor);
    EXPECT_EQ(glm::vec3(settings.GetMinimalConstantColor()), frame.minimalColor);
    EXPECT_EQ(settings.GetOverlays(), frame.overlays);
    EXPECT_EQ(settings.GetShadowCascades(), shadows.cascades);
    EXPECT_EQ(static_cast<std::uint32_t>(settings.GetShadowResolution()), shadows.resolution);
    EXPECT_EQ(static_cast<std::uint32_t>(settings.GetShadowWorldResolution()),
              shadows.worldResolution);
    EXPECT_FLOAT_EQ(static_cast<float>(settings.GetShadowSoftness()), shadows.softness);
    EXPECT_FLOAT_EQ(static_cast<float>(settings.GetShadowNormalOffset()), shadows.normalOffset);
    EXPECT_FLOAT_EQ(static_cast<float>(settings.GetShadowSplitLambda()), shadows.splitLambda);
    EXPECT_EQ(settings.GetShadowShowCascades(), shadows.showCascades);
    EXPECT_FLOAT_EQ(static_cast<float>(settings.GetShadowWorldMargin()), shadows.worldMargin);
    EXPECT_EQ(settings.GetShadowWorlds(), tokens::kSeparate);
    EXPECT_EQ(shadows.worlds, ernie::WorldShadows::Separate);
    EXPECT_EQ(settings.GetShadowWorldFit(), tokens::kScene);
    EXPECT_EQ(shadows.worldFit, ernie::WorldShadowFit::Scene);
    EXPECT_EQ(settings.GetShadowFit(), tokens::kScene);
    EXPECT_EQ(shadows.fit, ernie::ShadowFit::Scene);

    // The token options, through the functions that parse them, against their defaults.
    const Path path = settings.GetPath();
    EXPECT_EQ(ernie::CullingOf(scene, path), frame.culling);
    EXPECT_EQ(ernie::OcclusionOf(scene, path), frame.occlusion);
    EXPECT_EQ(ernie::WireframeOf(scene, path), frame.wireframe);
    EXPECT_EQ(settings.GetMinimalMode(), tokens::kOff);
    EXPECT_EQ(frame.minimal, ernie::MinimalMode::Off);

    // The terrain's.
    const ernie::TerrainParams terrain;
    const ernie::LodSettings& lod = terrain.lod;
    EXPECT_EQ(settings.GetColorMode(), tokens::kShadedColormap);
    EXPECT_EQ(terrain.colorMode, ernie::TerrainColorMode::ShadedColormap);
    EXPECT_EQ(settings.GetLodEnable(), lod.enabled);
    EXPECT_FLOAT_EQ(static_cast<float>(settings.GetLodDetail()), lod.detail);
    EXPECT_FLOAT_EQ(static_cast<float>(settings.GetLodMorphStart()), lod.morphStart);
    EXPECT_EQ(settings.GetLodFreeze(), lod.freeze);
    EXPECT_EQ(settings.GetLodShowLevels(), lod.showLevels);

    // UsdRender's resolution; the other types'.
    EXPECT_EQ(settings.GetResolution(), glm::dvec2(2048.0, 1080.0));
    EXPECT_TRUE(settings.GetCamera().empty());
    EXPECT_EQ(settings.GetFrame(), 0);
    EXPECT_EQ(ernie::RenderVar::Define(scene, "/Render/Var").GetSourceName(), tokens::kLdrColor);
    EXPECT_EQ(ernie::SensorFilterPrim::Define(scene, "/Render/Filter").GetFilterType(), "");
    const ernie::RenderProduct product = ernie::RenderProduct::Define(scene, "/Render/Product");
    EXPECT_EQ(product.GetResolution(), glm::dvec2(2048.0, 1080.0));
    EXPECT_FALSE(product.HasResolution());
}

TEST(RenderSchemas, SetsAndReadsByUsdName) {
    Scene scene;
    scene.Define("/Cam", tokens::kCamera);
    ernie::RenderSettings settings = ernie::RenderSettings::Define(scene, "/Render/Settings");
    settings.SetCamera(Path("/Cam"))
        .SetResolution({640.0, 480.0})
        .SetAmbient(0.3)
        .SetCulling(tokens::kCpu)
        .SetShadowCascades(2);
    EXPECT_EQ(*scene.Get("/Render/Settings", tokens::kCameraRel), Value(Path("/Cam")));
    EXPECT_EQ(*scene.Get("/Render/Settings", tokens::kAmbient), Value(0.3));
    EXPECT_EQ(settings.GetCamera(), std::vector<Path>{Path("/Cam")});
    EXPECT_EQ(settings.GetResolution(), glm::dvec2(640.0, 480.0));
    EXPECT_EQ(settings.GetCulling(), "cpu");
    EXPECT_EQ(settings.GetShadowCascades(), 2);
    EXPECT_TRUE(settings.HasAmbient());
    EXPECT_FALSE(settings.HasOcclusion());
    // An int resolution, as the viewer's panel writes, reads as the declared 2-vector.
    scene.Set("/Render/Settings", tokens::kResolution, glm::ivec2(320, 200));
    EXPECT_EQ(settings.GetResolution(), glm::dvec2(320.0, 200.0));

    ernie::RenderVar var = ernie::RenderVar::Define(scene, "/Render/Vars/Depth");
    var.SetSourceName(ernie::render_vars::kDistanceToCamera);
    EXPECT_EQ(*scene.Get("/Render/Vars/Depth", tokens::kSourceName),
              Value(std::string("DistanceToCamera")));
    ernie::SensorFilterPrim filter = ernie::SensorFilterPrim::Define(scene, "/Render/Qe");
    filter.SetFilterType("qe");
    EXPECT_EQ(*scene.Get("/Render/Qe", tokens::kFilterType), Value(std::string("qe")));
    ernie::RenderProduct product = ernie::RenderProduct::Define(scene, "/Render/Product");
    product.SetOverlays("*").SetMinimalMode(tokens::kConstantDiffuse);
    EXPECT_TRUE(product.HasOverlays());
    EXPECT_EQ(product.GetOverlays(), "*");
    EXPECT_EQ(ernie::MinimalModeOf(scene, "/Render/Product", ernie::MinimalMode::Off),
              ernie::MinimalMode::Constant);
}

TEST(RenderSchemas, ViewsCheckTypes) {
    Scene scene;
    ernie::RenderSettings::Define(scene, "/Settings");
    ernie::RenderProduct::Define(scene, "/Product");
    scene.Define("/Cam", tokens::kCamera);
    EXPECT_NO_THROW(ernie::RenderSettingsBase(scene, "/Settings"));
    EXPECT_NO_THROW(ernie::RenderSettingsBase(scene, "/Product"));
    EXPECT_THROW(ernie::RenderSettings(scene, "/Product"), std::invalid_argument);
    EXPECT_THROW(ernie::RenderProduct(scene, "/Settings"), std::invalid_argument);
    EXPECT_THROW(ernie::RenderVar(scene, "/Cam"), std::invalid_argument);
    EXPECT_THROW(ernie::SensorFilterPrim(scene, "/Missing"), std::invalid_argument);

    // Without a settings prim, a read-only view of the fallbacks.
    ernie::RenderSettings none = ernie::RenderSettings::OrFallbacks(scene, Path());
    EXPECT_EQ(none.GetShadowCascades(), 3);
    EXPECT_FALSE(none.HasShadowCascades());
    EXPECT_THROW(none.SetShadowCascades(2), std::logic_error);
    EXPECT_EQ(ernie::RenderSettings::OrFallbacks(scene, "/Missing").GetCulling(), tokens::kGpu);
    EXPECT_EQ(ernie::RenderSettings::OrFallbacks(scene, "/Settings").GetPath(), Path("/Settings"));
    EXPECT_THROW(ernie::RenderSettings::OrFallbacks(scene, "/Cam"), std::invalid_argument);
}

TEST(RenderSchemas, TypedRelationshipsChooseWhatRenders) {
    Scene scene;
    scene.Define("/Cams/A", tokens::kCamera);
    scene.Define("/Cams/B", tokens::kCamera);
    scene.Define("/Sun", tokens::kDistantLight);
    scene.Define("/Lamp", tokens::kSphereLight);
    ernie::RenderSettings settings = ernie::RenderSettings::Define(scene, "/Render/Settings")
                                         .SetCamera("/Cams/B")
                                         .SetProducts({"/Render/Product"});
    const ernie::RenderVar var = ernie::RenderVar::Define(scene, "/Render/Product/Color");
    ernie::SensorFilterPrim::Define(scene, "/Render/Mine").SetFilterType("mine");
    ernie::RenderProduct::Define(scene, "/Render/Product")
        .SetCamera({"/Cams/A", "/Cams/B"})
        .SetOrderedVars({var.GetPath()})
        .SetFilters({"/Render/Mine"});

    ernie::RenderPrims prims = ernie::ResolveRenderPrims(scene, "/Render/Settings");
    EXPECT_EQ(prims.camera, Path("/Cams/B"));
    EXPECT_EQ(prims.light, Path("/Sun")) << "every visible light while ernie:lights is unset";
    EXPECT_EQ(prims.lights, std::vector<Path>{Path("/Lamp")});
    settings.SetLights({});
    prims = ernie::ResolveRenderPrims(scene, "/Render/Settings");
    EXPECT_TRUE(prims.light.IsEmpty()) << "an empty list names none";
    EXPECT_TRUE(prims.lights.empty());
    settings.SetLights("/Lamp");
    prims = ernie::ResolveRenderPrims(scene, "/Render/Settings");
    EXPECT_TRUE(prims.light.IsEmpty());
    EXPECT_EQ(prims.lights, std::vector<Path>{Path("/Lamp")});
    settings.ClearLights().SetCamera({"/Cams/A", "/Cams/B"});
    prims = ernie::ResolveRenderPrims(scene, "/Render/Settings");
    EXPECT_EQ(prims.light, Path("/Sun"));
    EXPECT_EQ(prims.camera, Path("/Cams/A")) << "the first of a list";

    std::vector<ernie::RenderProductSpec> specs =
        ernie::ResolveRenderProducts(scene, {}, "/Render/Settings");
    ASSERT_EQ(specs.size(), 1u);
    EXPECT_EQ(specs[0].product, Path("/Render/Product"));
    EXPECT_EQ(specs[0].cameras, (std::vector<Path>{Path("/Cams/A"), Path("/Cams/B")}));
    EXPECT_EQ(specs[0].camera, Path("/Cams/A"));
    ASSERT_EQ(specs[0].vars.size(), 1u);
    EXPECT_EQ(specs[0].vars[0].first, var.GetPath());
    ASSERT_EQ(specs[0].filters.size(), 1u);
    EXPECT_EQ(specs[0].filters[0].type, "mine");
    EXPECT_FALSE(specs[0].varFilters[0].has_value()) << "the var's filters unset: the product's";
    // A var's own empty chain is a chain of its own, unlike none.
    ernie::RenderVar(scene, var.GetPath()).SetFilters({});
    specs = ernie::ResolveRenderProducts(scene, {}, "/Render/Settings");
    ASSERT_TRUE(specs[0].varFilters[0].has_value());
    EXPECT_TRUE(specs[0].varFilters[0]->empty());
}

TEST(RenderSchemas, RegistryDescribesTheTypes) {
    const PrimTypeDef& base = ernie::GetPrimType("RenderSettingsBase");
    EXPECT_TRUE(base.abstract);
    EXPECT_EQ(base.base, &ernie::Imageable::Definition());
    const PrimTypeDef& settings = ernie::GetPrimType(tokens::kRenderSettings);
    const PrimTypeDef& product = ernie::GetPrimType(tokens::kRenderProduct);
    EXPECT_FALSE(settings.abstract);
    EXPECT_EQ(settings.base, &base);
    EXPECT_EQ(product.base, &base);
    ASSERT_EQ(product.attributes.size(), 2u) << "its vars and filters; the rest its base's";
    EXPECT_EQ(product.attributes[0].name, tokens::kOrderedVars);
    EXPECT_EQ(product.attributes[0].pyName, "ordered_vars");
    EXPECT_EQ(product.attributes[1].name, tokens::kFilters);
    EXPECT_EQ(&ernie::GetPrimType(tokens::kSensorFilter), &ernie::SensorFilterPrim::Definition());
    EXPECT_TRUE(ernie::GetPrimType(tokens::kRenderVar).IsA(ernie::Imageable::Definition()));

    const ernie::AttributeDef* camera = product.Find(tokens::kCameraRel);
    ASSERT_NE(camera, nullptr);
    EXPECT_EQ(camera->kind, AttributeKind::Relationships) << "one camera, or a batch";
    const ernie::AttributeDef* cascades = settings.Find(tokens::kShadowCascades);
    ASSERT_NE(cascades, nullptr);
    EXPECT_EQ(cascades->pyName, "shadow_cascades");
    EXPECT_EQ(cascades->kind, AttributeKind::Int);
    EXPECT_EQ(cascades->label, "Cascades");
    EXPECT_EQ(cascades->meta.group, "Shadows");
    EXPECT_EQ(cascades->fallback, Value(3));
    EXPECT_EQ(settings.Find(tokens::kResolution)->kind, AttributeKind::Vec2);
    EXPECT_EQ(settings.Find(tokens::kMinimalMode)->meta.options.size(), 4u);
    EXPECT_EQ(settings.Find(tokens::kProducts)->kind, AttributeKind::Relationships);
    EXPECT_EQ(settings.Find(tokens::kLightsRel)->pyName, "lights");
    EXPECT_EQ(ernie::GetPrimType(tokens::kRenderVar).Find(tokens::kFilters)->kind,
              AttributeKind::Relationships);
    const ernie::AttributeDef* source =
        ernie::GetPrimType(tokens::kRenderVar).Find(tokens::kSourceName);
    ASSERT_NE(source, nullptr);
    EXPECT_EQ(source->meta.options.size(), ernie::kRenderVarNames.size());
    EXPECT_EQ(ernie::SensorFilterPrim::Definition().Find(tokens::kFilterType)->pyName,
              "filter_type");
}

}  // namespace
