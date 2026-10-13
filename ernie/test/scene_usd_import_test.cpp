// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/usd_import.h"

#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/stage_schemas.h"
#include "ernie/scene/tokens.h"

#include <glm/glm.hpp>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using ernie::Path;
using ernie::Scene;
using ernie::Tensor;
namespace tokens = ernie::tokens;

namespace {

/// A robot in centimetres, Y up: a cube body with a normal per face
/// corner and its top face bound to an OmniPBR material through a
/// GeomSubset, a left-handed triangle, a guide, a ball whose colour a
/// variant picks, two instances of a wheel outside the default prim,
/// and a light.
constexpr const char* kRobot = R"(#usda 1.0
(
    defaultPrim = "Robot"
    upAxis = "Y"
    metersPerUnit = 0.01
)

def Xform "Robot" (
    variants = {
        string paint = "red"
    }
    prepend variantSets = "paint"
)
{
    double3 xformOp:translate = (0, 100, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Mesh "Body" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        uniform token subdivisionScheme = "none"
        point3f[] points = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
        int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
        int[] faceVertexIndices = [0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 2, 3, 7, 6, 1, 2, 6, 5, 0, 4, 7, 3]
        normal3f[] normals = [(0, 0, -1), (0, 0, -1), (0, 0, -1), (0, 0, -1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, -1, 0), (0, -1, 0), (0, -1, 0), (0, -1, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0), (-1, 0, 0), (-1, 0, 0), (-1, 0, 0), (-1, 0, 0)] (
            interpolation = "faceVarying"
        )
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
            interpolation = "faceVarying"
        )
        int[] primvars:st:indices = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
        rel material:binding = </Robot/Looks/Painted>

        def GeomSubset "Top" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        )
        {
            uniform token elementType = "face"
            uniform token familyName = "materialBind"
            int[] indices = [1]
            rel material:binding = </Robot/Looks/Omni>
        }
    }

    def Mesh "Flipped"
    {
        uniform token subdivisionScheme = "none"
        uniform token orientation = "leftHanded"
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
    }

    def Mesh "Smooth"
    {
        point3f[] points = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
        int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
        int[] faceVertexIndices = [0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 2, 3, 7, 6, 1, 2, 6, 5, 0, 4, 7, 3]
        normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (
            interpolation = "vertex"
        )
        int[] creaseIndices = [4, 5, 6]
        int[] creaseLengths = [3]
        float[] creaseSharpnesses = [10]
        int[] holeIndices = [0]
    }

    def Sphere "Guide"
    {
        uniform token purpose = "guide"
    }

    def Sphere "Ball"
    {
        double radius = 5
    }

    def Xform "Wheels"
    {
        def "Wheel1" (
            instanceable = true
            prepend references = </Wheel>
        )
        {
        }

        def "Wheel2" (
            instanceable = true
            prepend references = </Wheel>
        )
        {
        }
    }

    def SphereLight "Lamp"
    {
        float inputs:intensity = 10
        float inputs:radius = 2
    }

    def Scope "Looks"
    {
        def Material "Painted"
        {
            token outputs:surface.connect = </Robot/Looks/Painted/Surface.outputs:surface>

            def Shader "Surface"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = </Robot/Looks/Painted/Tex.outputs:rgb>
                float inputs:roughness = 0.25
                token outputs:surface
            }

            def Shader "Tex"
            {
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @paint.png@
                float2 inputs:st.connect = </Robot/Looks/Painted/St.outputs:result>
                float3 outputs:rgb
            }

            def Shader "St"
            {
                uniform token info:id = "UsdPrimvarReader_float2"
                string inputs:varname = "st"
                float2 outputs:result
            }
        }

        def Material "Omni"
        {
            token outputs:mdl:surface.connect = </Robot/Looks/Omni/Shader.outputs:out>

            def Shader "Shader"
            {
                uniform token info:implementationSource = "sourceAsset"
                uniform asset info:mdl:sourceAsset = @OmniPBR.mdl@
                uniform token info:mdl:sourceAsset:subIdentifier = "OmniPBR"
                color3f inputs:diffuse_color_constant = (0.5, 0.25, 0.125)
                float inputs:metallic_constant = 1
                token outputs:out
            }
        }
    }

    variantSet "paint" = {
        "blue" {
            over "Ball"
            {
                color3f[] primvars:displayColor = [(0, 0, 1)]
            }
        }
        "red" {
            over "Ball"
            {
                color3f[] primvars:displayColor = [(1, 0, 0)]
            }
        }
    }
}

def Xform "Wheel"
{
    def Cube "Hub"
    {
        double size = 2
    }
}
)";

std::vector<float> Floats(const Tensor* tensor) {
    if (tensor == nullptr ||
        tensor->Type() != ernie::DType::Float32) {
        return {};
    }
    std::vector<float> values(
        static_cast<std::size_t>(tensor->Elements()));
    std::memcpy(values.data(), tensor->Contiguous().Data(),
                values.size() * sizeof(float));
    return values;
}

std::vector<std::int32_t> Ints(const Tensor* tensor) {
    if (tensor == nullptr || tensor->Type() != ernie::DType::Int32) {
        return {};
    }
    std::vector<std::int32_t> values(
        static_cast<std::size_t>(tensor->Elements()));
    std::memcpy(values.data(), tensor->Contiguous().Data(),
                values.size() * sizeof(std::int32_t));
    return values;
}

class UsdImportTest : public ::testing::Test {
  protected:
    void SetUp() override {
        std::string reason;
        if (!ernie::UsdImportAvailable(&reason)) {
            GTEST_SKIP() << reason;
        }
        dir_ = std::filesystem::temp_directory_path() /
               ("ernie_usd_import_" +
                std::string(::testing::UnitTest::GetInstance()
                                ->current_test_info()
                                ->name()));
        std::filesystem::create_directories(dir_);
        file_ = dir_ / "robot.usda";
        std::ofstream(file_) << kRobot;
        scene_.Define(Path("/World"), "Xform");
    }

    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
    }

    std::string Token(const char* path, std::string_view attribute) {
        return scene_.Get<std::string>(Path(path), attribute, "");
    }

    std::filesystem::path dir_;
    std::filesystem::path file_;
    Scene scene_;
};

TEST_F(UsdImportTest, DefaultPrimBecomesTheRootInSceneUnits) {
    const ernie::UsdImportResult result =
        ernie::ImportUsd(scene_, Path("/World/robot"), file_);
    EXPECT_EQ(scene_.TypeOf(Path("/World/robot")), "Xform");
    EXPECT_EQ(result.prims.at("/Robot/Body"),
              Path("/World/robot/Body"));
    // 100 cm up Y is 1 m up Z.
    const glm::dvec4 origin =
        scene_.WorldTransform(Path("/World/robot")) *
        glm::dvec4(0.0, 0.0, 0.0, 1.0);
    EXPECT_NEAR(origin.x, 0.0, 1e-9);
    EXPECT_NEAR(origin.y, 0.0, 1e-9);
    EXPECT_NEAR(origin.z, 1.0, 1e-9);
    // Y up turns into Z up, and centimetres into metres.
    const glm::dvec4 up =
        scene_.WorldTransform(Path("/World/robot")) *
        glm::dvec4(0.0, 1.0, 0.0, 0.0);
    EXPECT_NEAR(up.z, 0.01, 1e-12);
    EXPECT_NEAR(up.y, 0.0, 1e-12);
}

TEST_F(UsdImportTest, MeshesSplitHardEdgesAndSubsets) {
    ernie::ImportUsd(scene_, Path("/World/robot"), file_);
    // The five faces outside the subset, a vertex per corner since no
    // two faces share a normal.
    const Path body("/World/robot/Body");
    EXPECT_EQ(scene_.TypeOf(body), "Mesh");
    EXPECT_EQ(
        Ints(scene_.Read(body, tokens::kFaceVertexCounts)).size(),
        5u);
    EXPECT_EQ(Floats(scene_.Read(body, tokens::kPoints)).size(),
              20u * 3);
    EXPECT_EQ(Floats(scene_.Read(body, tokens::kNormals)).size(),
              20u * 3);
    EXPECT_EQ(Floats(scene_.Read(body, "primvars:st")).size(),
              20u * 2);
    EXPECT_EQ(scene_.Get<Path>(body, "material:binding", Path()),
              Path("/World/robot/Looks/Painted"));
    // The top face, bound to its own material.
    const Path top("/World/robot/Body/Top");
    EXPECT_EQ(scene_.TypeOf(top), "Mesh");
    const std::vector<float> normals =
        Floats(scene_.Read(top, tokens::kNormals));
    ASSERT_EQ(normals.size(), 12u);
    EXPECT_EQ(normals[2], 1.0f);
    EXPECT_EQ(scene_.Get<Path>(top, "material:binding", Path()),
              Path("/World/robot/Looks/Omni"));
    // Left-handed faces wind the other way.
    EXPECT_EQ(Ints(scene_.Read(Path("/World/robot/Flipped"),
                               tokens::kFaceVertexIndices)),
              (std::vector<std::int32_t>{0, 1, 2}));
    const std::vector<float> flipped = Floats(
        scene_.Read(Path("/World/robot/Flipped"), tokens::kPoints));
    ASSERT_EQ(flipped.size(), 9u);
    EXPECT_EQ(flipped[3],
              0.0f);  // The second corner is the third point.
    EXPECT_EQ(flipped[4], 1.0f);
}

TEST_F(UsdImportTest, SubdivisionSurfacesKeepTheirTags) {
    ernie::ImportUsd(scene_, Path("/World/robot"), file_);
    // No scheme authored: USD's fallback, Catmull-Clark; authored
    // normals ignored; creases and holes kept, through the points
    // they became.
    const Path smooth("/World/robot/Smooth");
    EXPECT_EQ(
        Token("/World/robot/Smooth", tokens::kSubdivisionScheme),
        "catmullClark");
    EXPECT_EQ(ernie::Mesh(scene_, Path("/World/robot/Body"))
                  .GetSubdivisionScheme(),
              "none");
    EXPECT_EQ(scene_.Read(smooth, tokens::kNormals), nullptr);
    EXPECT_EQ(Floats(scene_.Read(smooth, tokens::kPoints)).size(),
              8u * 3);
    const std::vector<std::int32_t> indices =
        Ints(scene_.Read(smooth, tokens::kFaceVertexIndices));
    const std::vector<float> points =
        Floats(scene_.Read(smooth, tokens::kPoints));
    const std::vector<std::int32_t> crease =
        Ints(scene_.Read(smooth, tokens::kCreaseIndices));
    ASSERT_EQ(crease.size(), 3u);
    // The crease's first point is the source's (-1, -1, 1).
    EXPECT_EQ(points[3 * static_cast<std::size_t>(crease[0]) + 2],
              1.0f);
    EXPECT_EQ(Ints(scene_.Read(smooth, tokens::kCreaseLengths)),
              (std::vector<std::int32_t>{3}));
    EXPECT_EQ(Ints(scene_.Read(smooth, tokens::kHoleIndices)),
              (std::vector<std::int32_t>{0}));
    EXPECT_EQ(indices.size(), 24u);
}

TEST_F(UsdImportTest, PurposesInstancesShapesAndLights) {
    const ernie::UsdImportResult result =
        ernie::ImportUsd(scene_, Path("/World/robot"), file_);
    EXPECT_FALSE(scene_.Has(Path("/World/robot/Guide")));
    // The two instances of the wheel: one instancer of one prototype,
    // holding the hub.
    std::vector<Path> instancers;
    std::vector<Path> hubs;
    for (const auto& [source, path] : result.prims) {
        if (scene_.TypeOf(path) == "PointInstancer") {
            instancers.push_back(path);
        }
        if (scene_.TypeOf(path) == "Cube" &&
            path.String().ends_with("/Hub")) {
            hubs.push_back(path);
        }
    }
    ASSERT_EQ(instancers.size(), 1u);
    EXPECT_EQ(Floats(scene_.Read(instancers[0], "positions")).size(),
              2u * 3);
    EXPECT_EQ(
        scene_.Get<std::vector<Path>>(instancers[0], "prototypes", {})
            .size(),
        1u);
    EXPECT_EQ(hubs.size(), 1u) << "the hub is shared, not copied";
    EXPECT_EQ(
        scene_.Get<double>(Path("/World/robot/Ball"), "radius", 0.0),
        5.0);
    EXPECT_EQ(scene_.TypeOf(Path("/World/robot/Lamp")),
              "SphereLight");
    EXPECT_NE(
        scene_.Get(Path("/World/robot/Lamp"), "inputs:intensity"),
        nullptr);
}

TEST_F(UsdImportTest, MaterialsKeepTheirNetworks) {
    ernie::ImportUsd(scene_, Path("/World/robot"), file_);
    EXPECT_EQ(Token("/World/robot/Looks/Painted",
                    "outputs:surface.connect"),
              "/World/robot/Looks/Painted/Surface.outputs:surface");
    EXPECT_EQ(Token("/World/robot/Looks/Painted/Surface", "info:id"),
              "UsdPreviewSurface");
    EXPECT_EQ(Token("/World/robot/Looks/Painted/Surface",
                    "inputs:diffuseColor.connect"),
              "/World/robot/Looks/Painted/Tex.outputs:rgb");
    EXPECT_TRUE(Token("/World/robot/Looks/Painted/Tex", "inputs:file")
                    .ends_with("paint.png"));
    // OmniPBR, approximated.
    const Path omni("/World/robot/Looks/Omni/ErniePreviewSurface");
    EXPECT_EQ(Token(omni.String().c_str(), "info:id"),
              "UsdPreviewSurface");
    const glm::dvec3 color = scene_.Get<glm::dvec3>(
        omni, "inputs:diffuseColor", glm::dvec3(0.0));
    EXPECT_NEAR(color.y, 0.25, 1e-6);
    EXPECT_NEAR(scene_.Get<double>(omni, "inputs:metallic", 0.0), 1.0,
                1e-6);
}

TEST_F(UsdImportTest, VariantsAreSelectedBeforeTranslating) {
    ernie::ImportUsd(scene_, Path("/World/red"), file_);
    ernie::UsdImportOptions options;
    options.variants.push_back({"/Robot", "paint", "blue"});
    ernie::ImportUsd(scene_, Path("/World/blue"), file_, options);
    EXPECT_EQ(Floats(scene_.Read(Path("/World/red/Ball"),
                                 "primvars:displayColor")),
              (std::vector<float>{1.0f, 0.0f, 0.0f}));
    EXPECT_EQ(Floats(scene_.Read(Path("/World/blue/Ball"),
                                 "primvars:displayColor")),
              (std::vector<float>{0.0f, 0.0f, 1.0f}));
}

// Animated attributes come in as the scene's time samples:
// transforms, a camera's focal length, a light's intensity, a shape's
// size, a constant displayColor and visibility; the file's range
// becomes the stage's.
TEST_F(UsdImportTest, AnimatedAttributesBecomeTimeSamples) {
    const std::filesystem::path file = dir_ / "animated.usda";
    std::ofstream(file) << R"(#usda 1.0
(
    defaultPrim = "Shot"
    metersPerUnit = 1
    upAxis = "Z"
    startTimeCode = 1
    endTimeCode = 24
    timeCodesPerSecond = 24
)
def Xform "Shot"
{
    def Camera "Camera"
    {
        float focalLength.timeSamples = { 1: 20, 24: 66 }
    }
    def SphereLight "Lamp"
    {
        float inputs:intensity.timeSamples = { 1: 1, 24: 47 }
    }
    def Cube "Box"
    {
        double size.timeSamples = { 1: 1, 24: 3 }
        color3f[] primvars:displayColor.timeSamples = { 1: [(1, 0, 0)], 24: [(0, 0, 1)] }
        token visibility.timeSamples = { 1: "inherited", 24: "invisible" }
        double3 xformOp:translate.timeSamples = { 1: (0, 0, 0), 24: (0, 23, 0) }
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
}
)";
    const ernie::UsdImportResult result =
        ernie::ImportUsd(scene_, Path("/World/Shot"), file);
    EXPECT_TRUE(result.warnings.empty())
        << (result.warnings.empty() ? "" : result.warnings.front());
    ASSERT_TRUE(result.timeCodeRange.has_value());
    EXPECT_EQ(result.timeCodeRange->y, 24.0);
    const auto at = [&](const char* path, std::string_view attribute,
                        double time) {
        const ernie::Value value =
            scene_.GetAt(Path(path), attribute, time);
        if (const auto* d = value.Get<double>()) {
            return *d;
        }
        return value.Get<float>() != nullptr
                   ? static_cast<double>(*value.Get<float>())
                   : -1.0;
    };
    EXPECT_NEAR(at("/World/Shot/Camera", tokens::kFocalLength, 12.5),
                43.0, 1e-4);
    EXPECT_NEAR(at("/World/Shot/Lamp", tokens::kIntensity, 12.5),
                24.0, 1e-4);
    EXPECT_NEAR(at("/World/Shot/Box", tokens::kSize, 12.5), 2.0,
                1e-4);
    EXPECT_EQ(
        scene_
            .SamplesOf(Path("/World/Shot/Box"), tokens::kDisplayColor)
            .size(),
        2u);
    EXPECT_EQ(
        scene_.SamplesOf(Path("/World/Shot/Box"), tokens::kVisibility)
            .size(),
        2u);
    ernie::Stage(scene_, Path::Root()).SetTimeCode(12.5);
    EXPECT_NEAR(scene_.WorldTransform(Path("/World/Shot/Box"))[3].y,
                11.5, 1e-6);
}

TEST_F(UsdImportTest, VolumesNameTheirFieldsAndFiles) {
    std::ofstream(dir_ / "smoke.vdb") << "vdb";
    std::ofstream(dir_ / "fire.vdb") << "vdb";
    const std::filesystem::path file = dir_ / "volume.usda";
    std::ofstream(file) << R"(#usda 1.0
(
    defaultPrim = "Fx"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Fx"
{
    def Volume "Smoke"
    {
        rel field:density = </Fx/Smoke/Density>
        rel field:temperature = </Fx/Smoke/Heat>

        def OpenVDBAsset "Density"
        {
            asset filePath = @smoke.vdb@
            token fieldName = "density"
        }
        def OpenVDBAsset "Heat"
        {
            asset filePath = @fire.vdb@
            token fieldName = "temperature"
            int fieldIndex = 1
        }
    }
}
)";
    const ernie::UsdImportResult result =
        ernie::ImportUsd(scene_, Path("/World/Fx"), file);
    EXPECT_EQ(scene_.TypeOf(Path("/World/Fx/Smoke")), "Volume");
    EXPECT_EQ(scene_.TypeOf(Path("/World/Fx/Smoke/Density")),
              "OpenVDBAsset");
    EXPECT_EQ(scene_.Get<Path>(Path("/World/Fx/Smoke"),
                               "field:density", Path()),
              Path("/World/Fx/Smoke/Density"));
    EXPECT_EQ(scene_.Get<Path>(Path("/World/Fx/Smoke"),
                               "field:temperature", Path()),
              Path("/World/Fx/Smoke/Heat"));
    EXPECT_TRUE(Token("/World/Fx/Smoke/Density", tokens::kFilePath)
                    .ends_with("smoke.vdb"));
    EXPECT_EQ(Token("/World/Fx/Smoke/Density", tokens::kFieldName),
              "density");
    EXPECT_EQ(scene_.Get<int>(Path("/World/Fx/Smoke/Heat"),
                              tokens::kFieldIndex, 0),
              1);
    EXPECT_TRUE(Token("/World/Fx/Smoke/Heat", tokens::kFilePath)
                    .ends_with("fire.vdb"));
    EXPECT_TRUE(result.warnings.empty())
        << (result.warnings.empty() ? "" : result.warnings.front());
}

TEST_F(UsdImportTest, LightsKeepTheirLinks) {
    const std::filesystem::path file = dir_ / "linked.usda";
    std::ofstream(file) << R"(#usda 1.0
(
    defaultPrim = "Set"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Set"
{
    def Cube "Hero" {}
    def Cube "Extra" {}
    def SphereLight "Key" (
        prepend apiSchemas = ["CollectionAPI:lightLink", "CollectionAPI:shadowLink"]
    )
    {
        uniform bool collection:lightLink:includeRoot = 0
        rel collection:lightLink:includes = </Set/Hero>
        rel collection:shadowLink:excludes = </Set/Extra>
    }
    def DistantLight "Fill" (
        prepend apiSchemas = ["CollectionAPI:lightLink"]
    )
    {
        rel collection:lightLink:includes = </Set>
    }
}
)";
    const ernie::UsdImportResult result =
        ernie::ImportUsd(scene_, Path("/World/Set"), file);
    EXPECT_TRUE(result.warnings.empty())
        << (result.warnings.empty() ? "" : result.warnings.front());
    const Path key("/World/Set/Key");
    EXPECT_FALSE(
        scene_.Get<bool>(key, tokens::kLightLinkIncludeRoot, true));
    EXPECT_EQ(scene_.Get<std::vector<Path>>(
                  key, tokens::kLightLinkIncludes, {}),
              std::vector<Path>{Path("/World/Set/Hero")});
    EXPECT_EQ(scene_.Get<std::vector<Path>>(
                  key, tokens::kShadowLinkExcludes, {}),
              std::vector<Path>{Path("/World/Set/Extra")});
    EXPECT_TRUE(
        scene_.Get<bool>(key, tokens::kShadowLinkIncludeRoot, true))
        << "unauthored, the default";
    EXPECT_EQ(
        scene_.Get<std::vector<Path>>(Path("/World/Set/Fill"),
                                      tokens::kLightLinkIncludes, {}),
        std::vector<Path>{Path("/World/Set")})
        << "the default prim is the import's root";
}

TEST_F(UsdImportTest, RenderSettingsComeWithTheirCamerasAndVars) {
    const std::filesystem::path file = dir_ / "shot.usda";
    std::ofstream(file) << R"(#usda 1.0
(
    defaultPrim = "Set"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Set"
{
    def Cube "Box" {}
}
def Camera "ShotCam"
{
    float focalLength = 35
}
def Scope "Render"
{
    def RenderSettings "Settings"
    {
        rel camera = </ShotCam>
        uniform int2 resolution = (640, 360)
        rel products = </Render/Beauty>
    }
    def RenderProduct "Beauty"
    {
        rel orderedVars = [</Render/Color>, </Render/Depth>]
    }
    def RenderVar "Color"
    {
        uniform string sourceName = "Ci"
    }
    def RenderVar "Depth"
    {
        uniform string sourceName = "depth"
    }
}
)";
    const ernie::UsdImportResult result =
        ernie::ImportUsd(scene_, Path("/World/Set"), file);
    EXPECT_TRUE(result.warnings.empty())
        << (result.warnings.empty() ? "" : result.warnings.front());
    const Path settings("/World/Set/_external/Render/Settings");
    ASSERT_EQ(scene_.TypeOf(settings), "RenderSettings");
    const glm::dvec2 resolution = scene_.Get<glm::dvec2>(
        settings, tokens::kResolution, glm::dvec2(0.0));
    EXPECT_EQ(resolution, glm::dvec2(640.0, 360.0));
    const std::vector<Path> cameras = scene_.Get<std::vector<Path>>(
        settings, tokens::kCameraRel, {});
    ASSERT_EQ(cameras.size(), 1u);
    EXPECT_EQ(scene_.TypeOf(cameras.front()), "Camera")
        << "the camera outside the default prim comes in too";
    EXPECT_EQ(scene_
                  .Get<std::vector<Path>>(
                      Path("/World/Set/_external/Render/Beauty"),
                      tokens::kOrderedVars, {})
                  .size(),
              2u);
    EXPECT_EQ(Token("/World/Set/_external/Render/Color",
                    tokens::kSourceName),
              "HdrColor");
    EXPECT_EQ(Token("/World/Set/_external/Render/Depth",
                    tokens::kSourceName),
              "DistanceToImagePlane");

    Scene without;
    without.Define(Path("/World"), "Xform");
    ernie::UsdImportOptions options;
    options.renderSettings = false;
    ernie::ImportUsd(without, Path("/World/Set"), file, options);
    EXPECT_TRUE(without.Prims(tokens::kRenderSettings).empty());
}

/// The CRC-32 a zip entry records.
std::uint32_t Crc32(std::string_view bytes) {
    std::uint32_t crc = 0xffffffffu;
    for (const char c : bytes) {
        crc ^= static_cast<std::uint8_t>(c);
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
        }
    }
    return ~crc;
}

/// Writes `files` (name, bytes; the layer first) as a .usdz: an
/// uncompressed zip whose data starts on 64-byte boundaries.
void WriteUsdz(
    const std::filesystem::path& path,
    const std::vector<std::pair<std::string, std::string>>& files) {
    std::string zip;
    std::string directory;
    const auto u16 = [](std::string& out, std::uint32_t v) {
        out.push_back(static_cast<char>(v & 0xff));
        out.push_back(static_cast<char>((v >> 8) & 0xff));
    };
    const auto u32 = [&](std::string& out, std::uint32_t v) {
        u16(out, v & 0xffff);
        u16(out, v >> 16);
    };
    for (const auto& [name, bytes] : files) {
        const auto offset = static_cast<std::uint32_t>(zip.size());
        const auto size = static_cast<std::uint32_t>(bytes.size());
        const std::uint32_t crc = Crc32(bytes);
        // Padding in an extra field (4 bytes of header) aligns data.
        const std::size_t start = zip.size() + 30 + name.size() + 4;
        const std::size_t pad = (64 - start % 64) % 64;
        u32(zip, 0x04034b50u);
        u16(zip, 20);
        u16(zip, 0);
        u16(zip, 0);
        u32(zip, 0);
        u32(zip, crc);
        u32(zip, size);
        u32(zip, size);
        u16(zip, static_cast<std::uint32_t>(name.size()));
        u16(zip, static_cast<std::uint32_t>(4 + pad));
        zip += name;
        u16(zip, 0x1986);
        u16(zip, static_cast<std::uint32_t>(pad));
        zip.append(pad, '\0');
        zip += bytes;
        u32(directory, 0x02014b50u);
        u16(directory, 20);
        u16(directory, 20);
        u16(directory, 0);
        u16(directory, 0);
        u32(directory, 0);
        u32(directory, crc);
        u32(directory, size);
        u32(directory, size);
        u16(directory, static_cast<std::uint32_t>(name.size()));
        u16(directory, 0);
        u16(directory, 0);
        u16(directory, 0);
        u16(directory, 0);
        u32(directory, 0);
        u32(directory, offset);
        directory += name;
    }
    const auto at = static_cast<std::uint32_t>(zip.size());
    zip += directory;
    u32(zip, 0x06054b50u);
    u16(zip, 0);
    u16(zip, 0);
    u16(zip, static_cast<std::uint32_t>(files.size()));
    u16(zip, static_cast<std::uint32_t>(files.size()));
    u32(zip, static_cast<std::uint32_t>(directory.size()));
    u32(zip, at);
    u16(zip, 0);
    std::ofstream(path, std::ios::binary) << zip;
}

void SetEnvironment(const char* name, const std::string& value) {
#if defined(_WIN32)
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

TEST_F(UsdImportTest, TexturesInsidePackagesAreUnpacked) {
    const std::filesystem::path cache = dir_ / "packages";
    SetEnvironment("ERNIE_USD_PACKAGE_CACHE", cache.string());
    const std::string texture = "not really a png";
    WriteUsdz(dir_ / "asset.usdz", {{"asset.usda", R"(#usda 1.0
(
    defaultPrim = "Asset"
)
def Xform "Asset"
{
    def Cube "Box" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        rel material:binding = </Asset/Looks/Paint>
    }
    def Scope "Looks"
    {
        def Material "Paint"
        {
            token outputs:surface.connect = </Asset/Looks/Paint/Surface.outputs:surface>
            def Shader "Surface"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = </Asset/Looks/Paint/Tex.outputs:rgb>
                token outputs:surface
            }
            def Shader "Tex"
            {
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @textures/paint.png@
                float3 outputs:rgb
            }
        }
    }
}
)"},
                                    {"textures/paint.png", texture}});
    const ernie::UsdImportResult result = ernie::ImportUsd(
        scene_, Path("/World/Asset"), dir_ / "asset.usdz");
    SetEnvironment("ERNIE_USD_PACKAGE_CACHE", "");
    EXPECT_TRUE(result.warnings.empty())
        << (result.warnings.empty() ? "" : result.warnings.front());
    const std::filesystem::path unpacked =
        Token("/World/Asset/Looks/Paint/Tex", "inputs:file");
    EXPECT_EQ(unpacked.filename(), "paint.png");
    EXPECT_TRUE(unpacked.string().starts_with(cache.string()))
        << unpacked;
    std::ifstream in(unpacked, std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(in)),
                            std::istreambuf_iterator<char>());
    EXPECT_EQ(bytes, texture);
}

TEST_F(UsdImportTest, MissingFilesThrow) {
    EXPECT_THROW(ernie::ImportUsd(scene_, Path("/World/none"),
                                  dir_ / "missing.usda"),
                 std::runtime_error);
}

}  // namespace
