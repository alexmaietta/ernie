// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"
#include "ernie/session/resolve.h"

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
namespace tokens = ernie::tokens;

namespace {

Scene MinimalScene() {
    Scene scene;
    scene.Define("/World/Camera", tokens::kCamera)
        .Set(tokens::kTransform,
             ernie::LookAtTransform({0.0, -100.0, 100.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}));
    scene.Define("/World/Terrain", tokens::kTerrain);
    return scene;
}

}  // namespace

TEST(WorldOf, InheritsAndResolvesViewAndInstanceWorlds) {
    Scene scene;
    scene.Define("/Envs/Env3", tokens::kXform).Set(tokens::kWorld, 3);
    scene.Define("/Envs/Env3/Rock", tokens::kMesh);
    scene.Define("/Shared", tokens::kMesh);
    EXPECT_EQ(ernie::WorldOf(scene, "/Envs/Env3/Rock"), 3) << "inherited";
    EXPECT_EQ(ernie::WorldOf(scene, "/Shared"), -1) << "shared by every world";

    scene.Define("/Envs/Env3/Cam", tokens::kCamera);
    scene.Define("/Cam", tokens::kCamera);
    const Path two[] = {"/Envs/Env3/Cam", "/Cam"};
    EXPECT_EQ(ernie::ViewWorldsOf(scene, two, 2), (std::vector<std::int32_t>{3, -1}));
    const Path one[] = {"/Cam"};
    scene.Write("/Cam", tokens::kPoses,
                Tensor::Copy<float>(std::vector<float>(32, 0.0f), {2, 4, 4}));
    EXPECT_EQ(ernie::ViewWorldsOf(scene, one, 2), (std::vector<std::int32_t>{-1, -1}));
    scene.Write("/Cam", tokens::kWorlds,
                Tensor::Copy<std::int32_t>(std::vector<std::int32_t>{4, 7}, {2}));
    EXPECT_EQ(ernie::ViewWorldsOf(scene, one, 2), (std::vector<std::int32_t>{4, 7}));
    EXPECT_THROW(ernie::ViewWorldsOf(scene, one, 3), std::invalid_argument) << "one per view";

    scene.Define("/Envs/Env3/Rocks", tokens::kPointInstancer);
    EXPECT_EQ(ernie::InstanceWorldsOf(scene, "/Envs/Env3/Rocks", 2),
              (std::vector<std::int32_t>{3, 3}));
    scene.Write("/Envs/Env3/Rocks", tokens::kWorlds,
                Tensor::Copy<std::int64_t>(std::vector<std::int64_t>{1, 0}, {2}));
    EXPECT_EQ(ernie::InstanceWorldsOf(scene, "/Envs/Env3/Rocks", 2),
              (std::vector<std::int32_t>{1, 0}));
}

TEST(WireframeOf, ReadsTokensAndBools) {
    using ernie::Wireframe;
    Scene scene;
    const Path settings("/Render/Settings");
    scene.Define(settings, tokens::kRenderSettings);
    EXPECT_EQ(ernie::WireframeOf(scene, settings), Wireframe::Off);
    EXPECT_EQ(ernie::WireframeOf(scene, Path()), Wireframe::Off);
    for (const Wireframe mode : {Wireframe::Off, Wireframe::Overlay, Wireframe::HiddenLine}) {
        scene.Set(settings, tokens::kWireframe, std::string(ernie::WireframeToken(mode)));
        EXPECT_EQ(ernie::WireframeOf(scene, settings), mode);
    }
    scene.Set(settings, tokens::kWireframe, true);
    EXPECT_EQ(ernie::WireframeOf(scene, settings), Wireframe::Overlay);
    scene.Set(settings, tokens::kWireframe, false);
    EXPECT_EQ(ernie::WireframeOf(scene, settings), Wireframe::Off);
    scene.Set(settings, tokens::kWireframe, std::string("dotted"));
    EXPECT_THROW(ernie::WireframeOf(scene, settings), std::invalid_argument);
}

TEST(RenderPrims, ResolveFollowsSettingsBindingsAndVisibility) {
    Scene scene = MinimalScene();
    scene.Define("/World/Other", tokens::kCamera);
    scene.Define("/World/Sun", tokens::kDistantLight);
    scene.Define("/Looks/Rock", tokens::kMaterial);
    scene.Define("/Looks/NotAMaterial", tokens::kXform);
    scene.Set("/World/Terrain", tokens::kMaterialBinding, Path("/Looks/Rock"));

    RenderPrims prims = ernie::ResolveRenderPrims(scene, {});
    EXPECT_TRUE(prims.settings.IsEmpty());
    EXPECT_EQ(prims.camera, Path("/World/Camera"));
    EXPECT_EQ(prims.terrain, Path("/World/Terrain"));
    EXPECT_EQ(prims.material, Path("/Looks/Rock"));
    EXPECT_EQ(prims.light, Path("/World/Sun"));

    scene.Define("/Render/Settings", tokens::kRenderSettings)
        .Set(tokens::kCameraRel, Path("/World/Other"));
    scene.Set("/World/Sun", tokens::kVisibility, tokens::kInvisible);
    scene.Set("/World/Terrain", tokens::kMaterialBinding, Path("/Looks/NotAMaterial"));
    prims = ernie::ResolveRenderPrims(scene, {});
    EXPECT_EQ(prims.settings, Path("/Render/Settings"));
    EXPECT_EQ(prims.camera, Path("/World/Other"));
    EXPECT_TRUE(prims.light.IsEmpty());
    EXPECT_TRUE(prims.material.IsEmpty());

    scene.Set("/World", tokens::kVisibility, tokens::kInvisible);
    EXPECT_TRUE(ernie::ResolveRenderPrims(scene, "/Render/Settings").terrain.IsEmpty());
    EXPECT_THROW(ernie::ResolveRenderPrims(scene, "/World/Camera"), std::invalid_argument);
    EXPECT_THROW(ernie::ResolveRenderPrims(scene, "/Missing"), std::invalid_argument);
}

TEST(RenderPrims, FallbacksSkipTheSessionRoot) {
    Scene scene;
    scene.Define("/__session/Perspective", tokens::kCamera);
    scene.Define("/__session/RenderSettings", tokens::kRenderSettings);
    scene.Define("/__session/Lights/Default", tokens::kDistantLight);
    scene.Define("/__session/Lights/Dome", tokens::kDomeLight);
    RenderPrims prims = ernie::ResolveRenderPrims(scene, {});
    EXPECT_TRUE(prims.settings.IsEmpty());
    EXPECT_TRUE(prims.camera.IsEmpty());
    EXPECT_TRUE(prims.light.IsEmpty());
    EXPECT_TRUE(prims.dome.IsEmpty());

    scene.Define("/z/Camera", tokens::kCamera);
    scene.Define("/z/Settings", tokens::kRenderSettings);
    scene.Define("/z/Lamp", tokens::kSphereLight);
    prims = ernie::ResolveRenderPrims(scene, {});
    EXPECT_EQ(prims.settings, Path("/z/Settings"));
    EXPECT_EQ(prims.camera, Path("/z/Camera"));
    EXPECT_EQ(prims.lights, std::vector<Path>{Path("/z/Lamp")});

    prims = ernie::ResolveRenderPrims(scene, "/__session/RenderSettings");
    EXPECT_EQ(prims.settings, Path("/__session/RenderSettings"));
    EXPECT_EQ(prims.camera, Path("/z/Camera"));
}

TEST(RenderPrims, LightsRelationshipChoosesTheLights) {
    Scene scene = MinimalScene();
    scene.Define("/World/Sun", tokens::kDistantLight);
    scene.Define("/World/Lamp", tokens::kSphereLight);
    scene.Define("/World/Sky", tokens::kDomeLight);
    scene.Define("/__session/Lights/Head", tokens::kDistantLight);
    scene.Define("/__session/Lights/Fill", tokens::kDistantLight);
    scene.Define("/Render/Settings", tokens::kRenderSettings)
        .Set(tokens::kLightsRel,
             std::vector<Path>{Path("/__session/Lights/Head"), Path("/__session/Lights/Fill"),
                               Path("/Missing"), Path("/World/Terrain")});
    RenderPrims prims = ernie::ResolveRenderPrims(scene, {});
    EXPECT_EQ(prims.light, Path("/__session/Lights/Fill"));
    EXPECT_EQ(prims.lights, std::vector<Path>{Path("/__session/Lights/Head")});
    EXPECT_TRUE(prims.dome.IsEmpty());

    scene.Set("/Render/Settings", tokens::kLightsRel, Path("/World/Sky"));
    prims = ernie::ResolveRenderPrims(scene, {});
    EXPECT_TRUE(prims.light.IsEmpty());
    EXPECT_TRUE(prims.lights.empty());
    EXPECT_EQ(prims.dome, Path("/World/Sky"));

    scene.Set("/Render/Settings", tokens::kLightsRel, std::vector<Path>{});
    prims = ernie::ResolveRenderPrims(scene, {});
    EXPECT_TRUE(prims.light.IsEmpty());
    EXPECT_TRUE(prims.lights.empty());
    EXPECT_TRUE(prims.dome.IsEmpty());

    scene.Clear("/Render/Settings", tokens::kLightsRel);
    prims = ernie::ResolveRenderPrims(scene, {});
    EXPECT_EQ(prims.light, Path("/World/Sun"));
    EXPECT_EQ(prims.lights, std::vector<Path>{Path("/World/Lamp")});
    EXPECT_EQ(prims.dome, Path("/World/Sky"));

    scene.Set("/Render/Settings", tokens::kLightsRel, 3.0f);
    EXPECT_THROW(ernie::ResolveRenderPrims(scene, {}), std::invalid_argument);
}

TEST(MeshDataOf, ReadsPointsTopologyNormalsAndVertexColours) {
    Scene scene;
    const std::array<double, 12> points{0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0};
    const std::array<std::int64_t, 1> counts{4};
    const std::array<std::int32_t, 4> indices{0, 1, 2, 3};
    const std::array<float, 12> normals{0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1};
    const std::array<float, 12> colors{1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1};
    scene.Define("/Quad", tokens::kMesh)
        .Write(tokens::kPoints, Tensor::Copy<double>(points, {4, 3}))
        .Write(tokens::kFaceVertexCounts, Tensor::Copy<std::int64_t>(counts, {1}))
        .Write(tokens::kFaceVertexIndices, Tensor::Copy<std::int32_t>(indices, {4}))
        .Write(tokens::kNormals, Tensor::Copy<float>(normals, {12}))
        .Write(tokens::kDisplayColor, Tensor::Copy<float>(colors, {4, 3}));
    const ernie::MeshData data = ernie::MeshDataOf(scene, "/Quad");
    EXPECT_EQ(data.points.size(), 12u);
    EXPECT_EQ(data.points[3], 1.0f);
    EXPECT_EQ(data.indices, (std::vector<std::uint32_t>{0, 1, 2, 0, 2, 3}));
    EXPECT_EQ(data.normals.size(), 12u);
    EXPECT_EQ(data.colors.size(), 12u);

    const std::array<float, 6> faceVarying{0, 0, 1, 0, 0, 1};
    scene.Write("/Quad", tokens::kNormals, Tensor::Copy<float>(faceVarying, {2, 3}));
    scene.Set("/Quad", tokens::kDisplayColor, glm::vec3(0.2f));
    scene.Clear("/Quad", tokens::kDisplayColor);
    EXPECT_TRUE(ernie::MeshDataOf(scene, "/Quad").normals.empty()) << "not one per point";

    scene.Write("/Quad", tokens::kPoints, Tensor::Zeros(DType::Int32, {4, 3}));
    EXPECT_THROW(ernie::MeshDataOf(scene, "/Quad"), std::invalid_argument);
    scene.Write("/Quad", tokens::kPoints, Tensor::Zeros(DType::Float32, {4, 2}));
    EXPECT_THROW(ernie::MeshDataOf(scene, "/Quad"), std::invalid_argument);
    scene.Write("/Quad", tokens::kPoints, Tensor::Copy<double>(points, {4, 3}));
    scene.Write("/Quad", tokens::kFaceVertexIndices, Tensor::Zeros(DType::Float32, {4}));
    EXPECT_THROW(ernie::MeshDataOf(scene, "/Quad"), std::invalid_argument);
    scene.Clear("/Quad", tokens::kFaceVertexIndices);
    EXPECT_THROW(ernie::MeshDataOf(scene, "/Quad"), std::invalid_argument);
}

TEST(MeshDataOf, ReadsTextureCoordinatesPerTriangleCornerWithAnyInterpolation) {
    // Two triangles over four points: 6 face vertices, 2 faces.
    Scene scene;
    const std::array<float, 12> points{0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0};
    const std::array<std::int64_t, 2> counts{3, 3};
    const std::array<std::int32_t, 6> indices{0, 1, 2, 0, 2, 3};
    scene.Define("/Quad", tokens::kMesh)
        .Write(tokens::kPoints, Tensor::Copy<float>(points, {4, 3}))
        .Write(tokens::kFaceVertexCounts, Tensor::Copy<std::int64_t>(counts, {2}))
        .Write(tokens::kFaceVertexIndices, Tensor::Copy<std::int32_t>(indices, {6}));
    EXPECT_TRUE(ernie::MeshDataOf(scene, "/Quad").uvs.empty());

    // One per point: vertex, following the point indices.
    const std::array<float, 8> perPoint{0, 0, 1, 0, 1, 1, 0, 1};
    scene.Write("/Quad", "primvars:st", Tensor::Copy<float>(perPoint, {4, 2}));
    EXPECT_EQ(ernie::MeshDataOf(scene, "/Quad").uvs,
              (std::vector<float>{0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1}));

    // One per face vertex: faceVarying, in face-vertex order.
    const std::array<float, 12> perCorner{0, 0, 1, 0, 1, 1, 5, 5, 6, 6, 7, 7};
    scene.Write("/Quad", "primvars:st", Tensor::Copy<float>(perCorner, {6, 2}));
    EXPECT_EQ(ernie::MeshDataOf(scene, "/Quad").uvs,
              (std::vector<float>{0, 0, 1, 0, 1, 1, 5, 5, 6, 6, 7, 7}));

    // Indexed: the indices' count decides, here one per face (uniform).
    const std::array<float, 4> values{0.25f, 0.5f, 0.75f, 1.0f};
    const std::array<std::int32_t, 2> perFace{1, 0};
    scene.Write("/Quad", "primvars:st", Tensor::Copy<float>(values, {2, 2}));
    scene.Write("/Quad", "primvars:st:indices", Tensor::Copy<std::int32_t>(perFace, {2}));
    const std::vector<float> uniform = ernie::MeshDataOf(scene, "/Quad").uvs;
    EXPECT_EQ(uniform[0], 0.75f) << "the first face reads value 1";
    EXPECT_EQ(uniform[6], 0.25f) << "the second value 0";

    // The interpolation attribute overrides the count, and must agree with it.
    scene.Set("/Quad", "primvars:st:interpolation", std::string("vertex"));
    EXPECT_THROW(ernie::MeshDataOf(scene, "/Quad"), std::invalid_argument);
    const std::array<std::int32_t, 1> one{1};
    scene.Write("/Quad", "primvars:st:indices", Tensor::Copy<std::int32_t>(one, {1}));
    scene.Set("/Quad", "primvars:st:interpolation", std::string("constant"));
    EXPECT_EQ(ernie::MeshDataOf(scene, "/Quad").uvs[10], 0.75f);
    const std::array<std::int32_t, 1> outOfRange{2};
    scene.Write("/Quad", "primvars:st:indices", Tensor::Copy<std::int32_t>(outOfRange, {1}));
    EXPECT_THROW(ernie::MeshDataOf(scene, "/Quad"), std::invalid_argument);

    // Coordinates matching nothing are an error, not silently dropped.
    scene.Clear("/Quad", "primvars:st:indices");
    scene.Clear("/Quad", "primvars:st:interpolation");
    scene.Write("/Quad", "primvars:st", Tensor::Zeros(DType::Float32, {5, 2}));
    EXPECT_THROW(ernie::MeshDataOf(scene, "/Quad"), std::invalid_argument);
}

TEST(MaterialSettingsOf, ReadsAUsdPreviewSurfaceNetwork) {
    Scene scene;
    const BrdfRegistry brdfs = BrdfRegistry::WithBuiltins();
    scene.Define("/Looks/M", tokens::kMaterial);
    scene.Define("/Looks/M/Surface", tokens::kShader)
        .Set(tokens::kInfoId, std::string(tokens::kUsdPreviewSurface))
        .Set(tokens::kDiffuseColor, glm::vec3(0.5f, 0.25f, 0.125f));
    EXPECT_EQ(ernie::MaterialSettingsOf(scene, "/Looks/M", brdfs).albedo, glm::vec3(1.0f))
        << "Ernie's own inputs until the surface is connected";
    ernie::Connect(scene, "/Looks/M", tokens::kSurfaceOutput, "/Looks/M/Surface",
                   tokens::kSurfaceOutput);
    ernie::MaterialSettings settings = ernie::MaterialSettingsOf(scene, "/Looks/M", brdfs);
    EXPECT_EQ(settings.albedo, glm::vec3(0.5f, 0.25f, 0.125f));
    EXPECT_FALSE(settings.albedoTexture.has_value());

    // diffuseColor from a texture, its coordinates through a transform from primvar "uv".
    scene.Define("/Looks/M/Tex", tokens::kShader)
        .Set(tokens::kInfoId, std::string(tokens::kUsdUVTexture))
        .Set(tokens::kFile, std::string("rock.png"))
        .Set(tokens::kWrapS, std::string("clamp"))
        .Set(tokens::kShaderScale, glm::vec4(2.0f))
        .Set(tokens::kSourceColorSpace, std::string("raw"));
    scene.Define("/Looks/M/Place", tokens::kShader)
        .Set(tokens::kInfoId, std::string(tokens::kUsdTransform2d))
        .Set(tokens::kShaderScale, glm::dvec2(4.0, 2.0))
        .Set(tokens::kRotation, 90.0);
    scene.Define("/Looks/M/Uv", tokens::kShader)
        .Set(tokens::kInfoId, std::string(tokens::kUsdPrimvarReaderFloat2))
        .Set(tokens::kVarname, std::string("uv"));
    ernie::Connect(scene, "/Looks/M/Surface", tokens::kDiffuseColor, "/Looks/M/Tex", "outputs:rgb");
    ernie::Connect(scene, "/Looks/M/Tex", tokens::kSt, "/Looks/M/Place", "outputs:result");
    ernie::Connect(scene, "/Looks/M/Place", tokens::kIn, "/Looks/M/Uv", "outputs:result");
    ernie::Connect(scene, "/Looks/M/Surface", tokens::kNormalInput, "/Looks/M/Tex", "outputs:rgb");
    settings = ernie::MaterialSettingsOf(scene, "/Looks/M", brdfs);
    EXPECT_EQ(settings.albedo, glm::vec3(1.0f)) << "the texture carries the colour";
    ASSERT_TRUE(settings.albedoTexture.has_value());
    const ernie::TextureReader& texture = *settings.albedoTexture;
    EXPECT_EQ(texture.file, "rock.png");
    EXPECT_EQ(texture.uvSet, "uv");
    EXPECT_EQ(texture.wrapS, ernie::TextureWrap::Clamp);
    EXPECT_EQ(texture.wrapT, ernie::TextureWrap::Repeat);
    EXPECT_EQ(texture.colorSpace, ernie::TextureColorSpace::Raw);
    EXPECT_EQ(texture.scale, glm::vec4(2.0f));
    EXPECT_EQ(texture.uvScale, glm::vec2(4.0f, 2.0f));
    EXPECT_EQ(texture.uvRotation, 90.0f);
    EXPECT_TRUE(settings.normalTexture.has_value());

    // A tensor in place of the file.
    scene.Write("/Looks/M/Tex", tokens::kShaderTexture, Tensor::Zeros(DType::UInt8, {2, 2, 3}));
    settings = ernie::MaterialSettingsOf(scene, "/Looks/M", brdfs);
    EXPECT_TRUE(settings.albedoTexture->file.empty());
    EXPECT_EQ(settings.albedoTexture->Source(), "/Looks/M/Tex");

    // Malformed networks say what is wrong.
    scene.Set("/Looks/M/Tex", tokens::kWrapT, std::string("wobble"));
    EXPECT_THROW(ernie::MaterialSettingsOf(scene, "/Looks/M", brdfs), std::invalid_argument);
    scene.Set("/Looks/M/Tex", tokens::kWrapT, std::string("repeat"));
    ernie::Connect(scene, "/Looks/M/Surface", tokens::kDiffuseColor, "/Looks/M/Tex", "outputs:r");
    EXPECT_THROW(ernie::MaterialSettingsOf(scene, "/Looks/M", brdfs), std::invalid_argument);
    ernie::Connect(scene, "/Looks/M/Surface", tokens::kDiffuseColor, "/Looks/M/Uv", "outputs:rgb");
    EXPECT_THROW(ernie::MaterialSettingsOf(scene, "/Looks/M", brdfs), std::invalid_argument);
    scene.Set("/Looks/M", std::string(tokens::kSurfaceOutput) + ".connect",
              std::string("not a path"));
    EXPECT_THROW(ernie::MaterialSettingsOf(scene, "/Looks/M", brdfs), std::invalid_argument);

    // A mesh bound to the material reads the coordinate set its textures name.
    scene.Set("/Looks/M", std::string(tokens::kSurfaceOutput) + ".connect",
              std::string("/Looks/M/Surface.outputs:surface"));
    ernie::Connect(scene, "/Looks/M/Surface", tokens::kDiffuseColor, "/Looks/M/Tex", "outputs:rgb");
    const std::array<float, 9> points{0, 0, 0, 1, 0, 0, 0, 1, 0};
    const std::array<std::int64_t, 1> counts{3};
    const std::array<std::int32_t, 3> indices{0, 1, 2};
    const std::array<float, 6> uv{0, 0, 1, 0, 0, 1};
    scene.Define("/Tri", tokens::kMesh)
        .Write(tokens::kPoints, Tensor::Copy<float>(points, {3, 3}))
        .Write(tokens::kFaceVertexCounts, Tensor::Copy<std::int64_t>(counts, {1}))
        .Write(tokens::kFaceVertexIndices, Tensor::Copy<std::int32_t>(indices, {3}))
        .Write("primvars:uv", Tensor::Copy<float>(uv, {3, 2}))
        .Set(tokens::kMaterialBinding, Path("/Looks/M"));
    EXPECT_EQ(ernie::MeshDataOf(scene, "/Tri").uvs.size(), 6u);
}

TEST(MaterialSettingsOf, PreviewSurfacesShadeWithTheirOwnModelAndInputs) {
    Scene scene;
    const BrdfRegistry brdfs = BrdfRegistry::WithBuiltins();
    scene.Define("/Looks/M", tokens::kMaterial);
    scene.Define("/Looks/M/Surface", tokens::kShader)
        .Set(tokens::kInfoId, std::string(tokens::kUsdPreviewSurface))
        .Set(tokens::kRoughness, 0.25)
        .Set(tokens::kUseSpecularWorkflow, true)
        .Set(tokens::kSpecularColor, glm::vec3(0.1f, 0.2f, 0.3f));
    ernie::Connect(scene, "/Looks/M", tokens::kSurfaceOutput, "/Looks/M/Surface",
                   tokens::kSurfaceOutput);
    ernie::MaterialSettings settings = ernie::MaterialSettingsOf(scene, "/Looks/M", brdfs);
    EXPECT_EQ(settings.brdf, "preview_surface");
    const std::vector<float> expected{0.25f, 0.0f, 1.5f, 1.0f, 0.1f, 0.2f, 0.3f, 0.0f, 0.01f};
    ASSERT_EQ(settings.brdfParams.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(settings.brdfParams[i], expected[i]) << i;
    }

    // Roughness and metallic from channels of one texture (glTF's metallicRoughness).
    scene.Define("/Looks/M/Mr", tokens::kShader)
        .Set(tokens::kInfoId, std::string(tokens::kUsdUVTexture))
        .Set(tokens::kFile, std::string("mr.png"));
    ernie::Connect(scene, "/Looks/M/Surface", tokens::kRoughness, "/Looks/M/Mr", "outputs:g");
    ernie::Connect(scene, "/Looks/M/Surface", tokens::kMetallic, "/Looks/M/Mr", "outputs:b");
    settings = ernie::MaterialSettingsOf(scene, "/Looks/M", brdfs);
    ASSERT_TRUE(settings.roughnessTexture.has_value());
    EXPECT_EQ(settings.roughnessTexture->channel, 1);
    EXPECT_EQ(settings.brdfParams[0], 0.0f) << "the texture carries the roughness";
    ASSERT_TRUE(settings.metallicTexture.has_value());
    EXPECT_EQ(settings.metallicTexture->channel, 2);
    EXPECT_EQ(settings.metallicParam, 1);
    ernie::Connect(scene, "/Looks/M/Surface", tokens::kMetallic, "/Looks/M/Mr", "outputs:rgb");
    EXPECT_THROW(ernie::MaterialSettingsOf(scene, "/Looks/M", brdfs), std::invalid_argument)
        << "a scalar reads one channel";

    // Emission and opacity, constants or (opacity) one texture channel.
    scene.Set("/Looks/M/Surface", tokens::kEmissiveColor, glm::vec3(3.0f, 2.0f, 1.0f));
    scene.Set("/Looks/M/Surface", tokens::kOpacityThreshold, 0.5);
    ernie::Connect(scene, "/Looks/M/Surface", tokens::kOpacity, "/Looks/M/Mr", "outputs:a");
    scene.Clear("/Looks/M/Surface", std::string(tokens::kMetallic) + ".connect");
    settings = ernie::MaterialSettingsOf(scene, "/Looks/M", brdfs);
    EXPECT_EQ(settings.emission, glm::vec3(3.0f, 2.0f, 1.0f));
    EXPECT_TRUE(settings.Cutout());
    ASSERT_TRUE(settings.opacityTexture.has_value());
    EXPECT_EQ(settings.opacityTexture->channel, 3);
    ernie::Connect(scene, "/Looks/M/Surface", tokens::kEmissiveColor, "/Looks/M/Mr", "outputs:rgb");
    EXPECT_THROW(ernie::MaterialSettingsOf(scene, "/Looks/M", brdfs), std::invalid_argument)
        << "emissive textures are not supported yet";
    scene.Clear("/Looks/M/Surface", std::string(tokens::kEmissiveColor) + ".connect");

    // A Material naming another model keeps it, with its parameters from the Material.
    scene.Clear("/Looks/M/Surface", std::string(tokens::kMetallic) + ".connect");
    scene.Set("/Looks/M", tokens::kBrdf, std::string("oren_nayar"));
    scene.Set("/Looks/M", "inputs:sigma", 5.0);
    settings = ernie::MaterialSettingsOf(scene, "/Looks/M", brdfs);
    EXPECT_EQ(settings.brdf, "oren_nayar");
    EXPECT_EQ(settings.brdfParams, std::vector<float>{5.0f});
    EXPECT_FALSE(settings.roughnessTexture.has_value());
}

TEST(InstanceDataOf, ReadsArraysAndGroupsInstancesByPrototype) {
    Scene scene;
    const std::array<float, 9> positions{0, 0, 0, 1, 0, 0, 2, 0, 0};
    const std::array<std::int32_t, 3> protoIndices{1, 0, 7};
    scene.Define("/Rocks", tokens::kPointInstancer)
        .Write(tokens::kPositions, Tensor::Copy<float>(positions, {3, 3}))
        .Write(tokens::kScales, Tensor::Copy<float>(positions, {9}));
    const ernie::InstanceData data = ernie::InstanceDataOf(scene, "/Rocks");
    EXPECT_EQ(data.positions.size(), 9u);
    EXPECT_TRUE(data.orientations.empty());
    EXPECT_EQ(data.scales.size(), 9u);

    using Lists = std::vector<std::vector<std::uint32_t>>;
    EXPECT_EQ(ernie::InstancesByPrototype(scene, "/Rocks", 2), (Lists{{0, 1, 2}, {}}))
        << "without protoIndices every instance uses prototype 0";
    scene.Write("/Rocks", tokens::kProtoIndices, Tensor::Copy<std::int32_t>(protoIndices, {3}));
    EXPECT_EQ(ernie::InstancesByPrototype(scene, "/Rocks", 2), (Lists{{1}, {0}}))
        << "out-of-range indices are dropped";
    EXPECT_TRUE(ernie::InstancesByPrototype(scene, "/Rocks", 0).empty());
    scene.Write("/Rocks", tokens::kProtoIndices, Tensor::Zeros(DType::Float32, {3}));
    EXPECT_THROW(ernie::InstancesByPrototype(scene, "/Rocks", 2), std::invalid_argument);
    scene.Write("/Rocks", tokens::kOrientations, Tensor::Zeros(DType::Float32, {3, 3}));
    EXPECT_THROW(ernie::InstanceDataOf(scene, "/Rocks"), std::invalid_argument);
}

TEST(BoundMaterial, FollowsBindingsToMaterialsOnly) {
    Scene scene;
    scene.Define("/Looks/Rock", tokens::kMaterial);
    scene.Define("/Looks/Group", tokens::kScope);
    scene.Define("/Mesh", tokens::kMesh).Set(tokens::kMaterialBinding, Path("/Looks/Rock"));
    EXPECT_EQ(ernie::BoundMaterial(scene, "/Mesh"), Path("/Looks/Rock"));
    scene.Set("/Mesh", tokens::kMaterialBinding, Path("/Looks/Group"));
    EXPECT_TRUE(ernie::BoundMaterial(scene, "/Mesh").IsEmpty());
    EXPECT_TRUE(ernie::BoundMaterial(scene, "/Missing").IsEmpty());
}

TEST(ImageOfTensor, ConvertsFloatAndByteTextures) {
    const std::array<float, 6> rgb{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    const ernie::Image floats =
        ernie::ImageOfTensor(Tensor::Copy<float>(rgb, {1, 2, 3}), ImageEncoding::Srgb);
    EXPECT_EQ(floats.width, 2u);
    EXPECT_EQ(floats.height, 1u);
    EXPECT_EQ(floats.At(1, 0), glm::vec4(0.4f, 0.5f, 0.6f, 1.0f));

    const std::array<std::uint8_t, 4> rgba{255, 128, 0, 51};
    const Tensor bytes = Tensor::Copy<std::uint8_t>(rgba, {1, 1, 4});
    const ernie::Image srgb = ernie::ImageOfTensor(bytes, ImageEncoding::Srgb);
    EXPECT_NEAR(srgb.At(0, 0).g, ernie::SrgbToLinear(128.0f / 255.0f), 1e-6f);
    EXPECT_NEAR(srgb.At(0, 0).a, 0.2f, 1e-6f) << "alpha stays linear";
    const ernie::Image linear = ernie::ImageOfTensor(bytes, ImageEncoding::Linear);
    EXPECT_NEAR(linear.At(0, 0).g, 128.0f / 255.0f, 1e-6f);

    const std::array<float, 2> grey{0.25f, 0.75f};
    const ernie::Image g =
        ernie::ImageOfTensor(Tensor::Copy<float>(grey, {2, 1}), ImageEncoding::Linear);
    EXPECT_EQ(g.At(0, 1), glm::vec4(0.75f, 0.75f, 0.75f, 1.0f));

    EXPECT_THROW(
        ernie::ImageOfTensor(Tensor::Zeros(DType::Float32, {2, 2, 2}), ImageEncoding::Srgb),
        std::invalid_argument);
    EXPECT_THROW(ernie::ImageOfTensor(Tensor::Zeros(DType::Int32, {2, 2}), ImageEncoding::Srgb),
                 std::invalid_argument);
    EXPECT_THROW(ernie::ImageOfTensor(Tensor::Zeros(DType::Float32, {0, 2}), ImageEncoding::Srgb),
                 std::invalid_argument);
    EXPECT_THROW(ernie::ImageOfTensor(Tensor::Zeros(DType::Float32, {4}), ImageEncoding::Srgb),
                 std::invalid_argument);
}

TEST(SunIrradianceOf, PhysicalLightsCarryTheirIrradianceAndSpectrum) {
    Scene scene;
    scene.Define("/Sun", tokens::kDistantLight).Set(tokens::kLightIrradiance, 1361.0);
    const ernie::SpectralBands bands;
    const std::vector<float> sun = ernie::SunIrradianceOf(scene, "/Sun", bands);
    EXPECT_EQ(sun, ernie::BlackbodyIrradiance(bands, 5772.0, 1361.0)) << "the Sun's black body";

    scene.Set("/Sun", tokens::kLightSpectrum, std::vector<float>{400.0f, 1.0f, 800.0f, 1.0f});
    scene.Set("/Sun", tokens::kLightIrradiance, 800.0);
    for (const float value : ernie::SunIrradianceOf(scene, "/Sun", {400.0, 780.0, 4})) {
        EXPECT_NEAR(value, 2.0f, 1e-5f) << "a flat 400 W/m^2 spectrum scaled to 800 W/m^2";
    }

    Scene hot;
    hot.Define("/Sun", tokens::kDistantLight)
        .Set(tokens::kLightIrradiance, 1000.0)
        .Set(tokens::kEnableColorTemperature, true)
        .Set(tokens::kColorTemperature, 3000.0);
    EXPECT_EQ(ernie::SunIrradianceOf(hot, "/Sun", bands),
              ernie::BlackbodyIrradiance(bands, 3000.0, 1000.0));
    EXPECT_THROW(ernie::SunIrradianceOf(hot, "/Sun", {500.0, 400.0, 4}), std::invalid_argument);
}

TEST(SunIrradianceOf, RelativeLightsFollowTheirColourAndTemperature) {
    Scene scene;
    scene.Define("/Sun", tokens::kDistantLight).Set(tokens::kColor, glm::vec3(1.0f, 0.2f, 0.1f));
    const ernie::SpectralBands bands;
    const std::vector<float> red = ernie::SunIrradianceOf(scene, "/Sun", bands);
    EXPECT_GT(red[13], 3.0f * red[2]) << "650 nm against 430 nm";

    scene.Set("/Sun", tokens::kColor, glm::vec3(1.0f));
    scene.Set("/Sun", tokens::kEnableColorTemperature, true);
    scene.Set("/Sun", tokens::kColorTemperature, 2500.0);
    const std::vector<float> warm = ernie::SunIrradianceOf(scene, "/Sun", bands);
    EXPECT_GT(warm[13], warm[2]);
    scene.Set("/Sun", tokens::kColorTemperature, 20000.0);
    const std::vector<float> cool = ernie::SunIrradianceOf(scene, "/Sun", bands);
    EXPECT_LT(cool[13], cool[2]);
}

TEST(OverscanOf, WidensTheRasterViewForDistortion) {
    ernie::CameraParams camera;
    camera.width = 64;
    camera.height = 48;
    EXPECT_EQ(ernie::OverscanOf(camera), 1.0);
    camera.lens.k1 = -0.2;
    const double barrel = ernie::OverscanOf(camera);
    EXPECT_GT(barrel, 1.02) << "barrel distortion pulls wider rays into the sensor";
    camera.lens.k1 = 0.2;
    EXPECT_LT(ernie::OverscanOf(camera), barrel);
    EXPECT_GE(ernie::OverscanOf(camera), 1.0);
}

TEST(PrimitiveMeshOf, ReadsUsdGeomAttributesAndDefaults) {
    Scene scene;
    scene.Define("/Cube", tokens::kCube);
    scene.Define("/Rod", tokens::kCylinder)
        .Set(tokens::kGeomRadius, 0.25)
        .Set(tokens::kGeomHeight, 3.0)
        .Set(tokens::kAxis, "X");
    scene.Define("/Floor", tokens::kPlane).Set(tokens::kGeomWidth, 10.0);
    scene.Define("/Bad", tokens::kCone).Set(tokens::kAxis, "W");

    const auto bounds = [](const ernie::MeshData& mesh) {
        glm::vec3 low(1e30f);
        glm::vec3 high(-1e30f);
        for (std::size_t i = 0; i < mesh.points.size(); i += 3) {
            const glm::vec3 p(mesh.points[i], mesh.points[i + 1], mesh.points[i + 2]);
            low = glm::min(low, p);
            high = glm::max(high, p);
        }
        return std::pair(low, high);
    };
    EXPECT_EQ(bounds(ernie::MeshDataOf(scene, "/Cube")).second, glm::vec3(1.0f))
        << "UsdGeomCube's default size is 2";
    const auto [rodLow, rodHigh] = bounds(ernie::MeshDataOf(scene, "/Rod"));
    EXPECT_FLOAT_EQ(rodHigh.x, 1.5f) << "height along the X axis";
    EXPECT_NEAR(rodHigh.y, 0.25f, 1e-6f);
    EXPECT_FLOAT_EQ(bounds(ernie::MeshDataOf(scene, "/Floor")).second.x, 5.0f);
    EXPECT_FLOAT_EQ(bounds(ernie::MeshDataOf(scene, "/Floor")).second.y, 1.0f);
    EXPECT_THROW(ernie::MeshDataOf(scene, "/Bad"), std::invalid_argument);
    EXPECT_TRUE(ernie::IsMeshType(tokens::kCapsule));
    EXPECT_FALSE(ernie::IsMeshType(tokens::kTerrain));
    scene.Define("/Xform", tokens::kXform);
    EXPECT_THROW(ernie::PrimitiveMeshOf(scene, "/Xform"), std::invalid_argument);
}

TEST(FrameOf, ReadsTheAuthoredFrame) {
    Scene scene;
    scene.Define("/Render/Settings", tokens::kRenderSettings);
    EXPECT_FALSE(ernie::FrameOf(scene, Path("/Render/Settings")).has_value());
    EXPECT_FALSE(ernie::FrameOf(scene, Path()).has_value());
    scene.Set("/Render/Settings", tokens::kFrame, 12);
    EXPECT_EQ(ernie::FrameOf(scene, Path("/Render/Settings")), 12u);
    scene.Set("/Render/Settings", tokens::kFrame, 3.0);
    EXPECT_EQ(ernie::FrameOf(scene, Path("/Render/Settings")), 3u);
    scene.Set("/Render/Settings", tokens::kFrame, -1);
    EXPECT_THROW(ernie::FrameOf(scene, Path("/Render/Settings")), std::invalid_argument);
    scene.Set("/Render/Settings", tokens::kFrame, 2.5);
    EXPECT_THROW(ernie::FrameOf(scene, Path("/Render/Settings")), std::invalid_argument);
}

TEST(ExtractDomeSun, FindsACompactBrightSunAndLeavesTheSky) {
    // A sky of 1 at a degree per texel, with a sun of 5000 a degree in radius, 40 degrees from
    // the zenith at azimuth 60 degrees.
    const std::uint32_t rows = 180;
    const std::uint32_t cols = 360;
    const auto direction = [&](std::uint32_t r, std::uint32_t c) {
        const double theta = (r + 0.5) * M_PI / rows;
        const double phi = (c + 0.5) * 2.0 * M_PI / cols;
        return glm::dvec3(std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi),
                          std::cos(theta));
    };
    const glm::dvec3 towards(std::sin(glm::radians(40.0)) * std::cos(glm::radians(60.0)),
                             std::sin(glm::radians(40.0)) * std::sin(glm::radians(60.0)),
                             std::cos(glm::radians(40.0)));
    ernie::Image map(cols, rows, glm::vec4(1.0f));
    double sunOmega = 0.0;
    for (std::uint32_t r = 0; r < rows; ++r) {
        for (std::uint32_t c = 0; c < cols; ++c) {
            if (glm::dot(direction(r, c), towards) > std::cos(glm::radians(1.0))) {
                map.At(c, r) = glm::vec4(5000.0f, 4000.0f, 3000.0f, 1.0f);
                sunOmega += (std::cos(r * M_PI / rows) - std::cos((r + 1) * M_PI / rows)) *
                            2.0 * M_PI / cols;
            }
        }
    }
    const std::optional<ernie::DomeSunExtraction> found = ernie::ExtractDomeSun(map);
    ASSERT_TRUE(found.has_value());
    EXPECT_GT(glm::dot(found->sun.direction, towards), std::cos(glm::radians(0.5)));
    EXPECT_NEAR(found->sun.angle, 2.0, 0.5) << "a degree in radius";
    EXPECT_NEAR(found->sun.irradiance.r, 4999.0 * sunOmega, 1e-3 * 4999.0 * sunOmega)
        << "what the sun adds above the sky";
    EXPECT_NEAR(found->sun.irradiance.b / found->sun.irradiance.r, 2999.0 / 4999.0, 1e-6);
    for (const glm::vec4& texel : found->lighting.pixels) {
        EXPECT_NEAR(texel.r, 1.0f, 1e-6f) << "the sky goes on where the sun was";
    }

    // No sun: a uniform sky, a broad bright patch, a low-dynamic-range highlight.
    EXPECT_FALSE(ernie::ExtractDomeSun(ernie::Image(cols, rows, glm::vec4(1.0f))).has_value());
    ernie::Image broad(cols, rows, glm::vec4(1.0f));
    ernie::Image mild = broad;
    for (std::uint32_t r = 0; r < rows; ++r) {
        for (std::uint32_t c = 0; c < cols; ++c) {
            const double cosine = glm::dot(direction(r, c), towards);
            if (cosine > std::cos(glm::radians(10.0))) {
                broad.At(c, r) = glm::vec4(5000.0f);
            }
            if (cosine > std::cos(glm::radians(1.0))) {
                mild.At(c, r) = glm::vec4(4.0f);
            }
        }
    }
    EXPECT_FALSE(ernie::ExtractDomeSun(broad).has_value()) << "too wide for a sun";
    EXPECT_FALSE(ernie::ExtractDomeSun(mild).has_value()) << "not bright enough over the sky";
}

TEST(DomeDirectionsOf, SplitsTheDomesEnergyEvenlyAndFollowsItsBrightRegions) {
    // A uniform map: eight directions of about an eighth of the sphere's irradiance each (4 pi /
    // 8, within the snapping of cuts to whole rows and columns), adding up to the whole sphere's
    // exactly and spread so their mean direction vanishes.
    const ernie::Image uniform(64, 32, glm::vec4(1.0f));
    const std::vector<ernie::DomeDirection> even = ernie::DomeDirectionsOf(uniform, 8);
    ASSERT_EQ(even.size(), 8u);
    glm::dvec3 total(0.0);
    glm::dvec3 mean(0.0);
    for (const ernie::DomeDirection& d : even) {
        EXPECT_NEAR(d.irradiance.g, 4.0 * M_PI / 8.0, 0.25 * 4.0 * M_PI / 8.0);
        total += d.irradiance;
        mean += d.direction;
    }
    EXPECT_NEAR(total.g, 4.0 * M_PI, 1e-6);
    EXPECT_LT(glm::length(mean / 8.0), 0.1);

    // A bright upper hemisphere draws the directions up.
    ernie::Image sky(64, 32, glm::vec4(0.01f));
    for (std::uint32_t r = 0; r < 16; ++r) {
        for (std::uint32_t c = 0; c < 64; ++c) {
            sky.At(c, r) = glm::vec4(1.0f);
        }
    }
    int up = 0;
    for (const ernie::DomeDirection& d : ernie::DomeDirectionsOf(sky, 16)) {
        up += d.direction.z > 0.0 ? 1 : 0;
    }
    EXPECT_GE(up, 14);
    EXPECT_TRUE(ernie::DomeDirectionsOf(ernie::Image(64, 32, glm::vec4(0.0f)), 4).empty());
}
