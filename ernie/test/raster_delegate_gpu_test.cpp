#include "ernie/device/offscreen_target.h"
#include "ernie/geometry/primitives.h"
#include "ernie/raster/delegate.h"
#include "ernie/scene/orbit_controller.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"
#include "ernie/session/renderer.h"
#include "support/delegate_scenes.h"
#include "support/test_raster.h"
#include "support/vulkan_fixture.h"

#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

using ernie::DeviceKind;
using ernie::DType;
using ernie::OrbitController;
using ernie::Path;
using ernie::RasterDelegate;
using ernie::Scene;
using ernie::Tensor;
using namespace ernie::testing;
namespace tokens = ernie::tokens;

namespace {

/// A camera framing the delegate's terrain from 0.8 rad above the horizon.
ernie::Camera FramingCamera(const RasterDelegate& delegate) {
    OrbitController orbit;
    orbit.pitch = 0.8f;
    const auto [low, high] = delegate.Terrain().Bounds(1.0f);
    orbit.Frame(low, high);
    return orbit.MakeCamera(kSize, kSize);
}

}  // namespace

TEST_F(RasterDelegateTest, RendersTheSameImageAsTheRendererDrivenDirectly) {
    const TestRaster raster = RollingTerrain(65, 65, 2.0);
    Scene scene = TerrainScene(raster);
    RasterDelegate delegate(*context_);
    delegate.Sync(scene);
    ASSERT_TRUE(delegate.Terrain().HasHeightfield());
    const ernie::Camera camera = FramingCamera(delegate);
    ernie::SetCamera(scene, "/World/Camera", camera);

    const Tensor image = delegate.Render(scene);
    ASSERT_EQ(image.Shape(), (std::vector<std::int64_t>{kSize, kSize, 4}));
    EXPECT_EQ(delegate.Prims().terrain, Path("/World/Terrain"));

    ernie::RasterRenderer direct(*context_, VK_FORMAT_R8G8B8A8_UNORM);
    ernie::TerrainDrawable terrainDrawable(direct);
    terrainDrawable.SetHeightfield(raster);
    const std::uint32_t material = direct.AddMaterial(
        ernie::MaterialSettingsOf(scene, delegate.Prims().material, direct.Brdfs()));
    ernie::TerrainParams terrainParams = ernie::TerrainParamsOf(scene, delegate.Prims());
    terrainParams.material = material;
    terrainDrawable.SetParams(terrainParams);
    ernie::RasterDrawable* drawables[] = {&terrainDrawable};
    direct.Update(ernie::FrameParamsOf(scene, delegate.Prims(), kSize, kSize), drawables);
    ernie::OffscreenTarget target(*context_, kSize, kSize);
    ernie::RenderTargets targets(*context_, {kSize, kSize}, direct.Bands().count, false);
    const ernie::RasterDrawable* drawn[] = {&terrainDrawable};
    context_->Submit([&](VkCommandBuffer commands) {
        const ernie::RenderingTarget present = target.Prepare(commands);
        direct.Record(commands, targets, drawn, &present);
    });
    const std::vector<std::uint8_t> expected = target.ReadColor();
    ASSERT_EQ(expected.size(), static_cast<std::size_t>(image.Elements()));
    EXPECT_EQ(std::memcmp(expected.data(), image.Data(), expected.size()), 0);

    const auto* pixels = image.As<std::uint8_t>();
    EXPECT_EQ(pixels[2], 255) << "clear colour from ernie:clearColor";
    int terrain = 0;
    for (std::size_t i = 0; i < expected.size(); i += 4) {
        terrain += pixels[i + 2] != 255 ? 1 : 0;
    }
    EXPECT_GT(terrain, static_cast<int>(kSize * kSize / 4)) << "the terrain covers the view";
}

TEST_F(RasterDelegateTest, AnUnlitSceneRendersBlackAndChosenLightsLightIt) {
    const TestRaster raster = RollingTerrain(33, 33, 2.0);
    Scene scene = TerrainScene(raster);
    scene.Set("/Render/Settings", tokens::kClearColor, glm::vec3(0.0f));
    RasterDelegate delegate(*context_);
    delegate.Sync(scene);
    ernie::SetCamera(scene, "/World/Camera", FramingCamera(delegate));
    const auto brightness = [&](const Tensor& image) {
        const auto* pixels = image.As<std::uint8_t>();
        long sum = 0;
        for (std::int64_t i = 0; i < image.Elements(); i += 4) {
            sum += pixels[i] + pixels[i + 1] + pixels[i + 2];
        }
        return sum;
    };
    EXPECT_GT(brightness(delegate.Render(scene)), 0);

    scene.Define("/__session/Lights/Head", tokens::kDistantLight)
        .Set(tokens::kTransform, ernie::DistantLightTransform({0.0, 0.0, 1.0}));
    scene.Set("/Render/Settings", tokens::kLightsRel, std::vector<Path>{});
    EXPECT_EQ(brightness(delegate.Render(scene)), 0) << "no chosen lights";
    scene.Set("/Render/Settings", tokens::kLightsRel, Path("/__session/Lights/Head"));
    EXPECT_GT(brightness(delegate.Render(scene)), 0) << "lit by the chosen session light";

    scene.Clear("/Render/Settings", tokens::kLightsRel);
    scene.Remove("/World/Sun");
    EXPECT_EQ(brightness(delegate.Render(scene)), 0) << "session lights are not stage lights";
}

TEST_F(RasterDelegateTest, WireframeModesDrawEdgesOverShadedOrFlatSurfaces) {
    const TestRaster raster = RollingTerrain(33, 33, 2.0);
    Scene scene = TerrainScene(raster);
    RasterDelegate delegate(*context_);
    delegate.Sync(scene);
    ernie::SetCamera(scene, "/World/Camera", FramingCamera(delegate));
    const auto render = [&](std::string_view mode) {
        scene.Set("/Render/Settings", tokens::kWireframe, std::string(mode));
        const Tensor image = delegate.Render(scene);
        const auto* pixels = image.As<std::uint8_t>();
        return std::vector<std::uint8_t>(pixels, pixels + image.Elements());
    };
    const std::vector<std::uint8_t> shaded = render(tokens::kOff);
    const std::vector<std::uint8_t> overlay = render(tokens::kOverlay);
    const std::vector<std::uint8_t> hidden = render(tokens::kHiddenLine);
    std::size_t darker = 0;
    for (std::size_t i = 0; i < shaded.size(); ++i) {
        ASSERT_LE(overlay[i], shaded[i]) << "edges only darken the shaded image";
        darker += overlay[i] < shaded[i] ? 1u : 0u;
    }
    EXPECT_GT(darker, 0u) << "the overlay draws edges";
    EXPECT_NE(hidden, shaded);

    scene.Set("/World/Sun", tokens::kTransform, ernie::DistantLightTransform({-1.0, 0.2, 0.9}));
    EXPECT_NE(render(tokens::kOff), shaded) << "shading follows the sun";
    EXPECT_EQ(render(tokens::kHiddenLine), hidden) << "hidden line ignores the sun's direction";
}

TEST_F(RasterDelegateTest, OutputsDoNotDependOnThePresentFormat) {
    const TestRaster raster = RollingTerrain(33, 33, 2.0);
    Scene scene = TerrainScene(raster);
    RasterDelegate rgba(*context_);
    rgba.Sync(scene);
    ernie::SetCamera(scene, "/World/Camera", FramingCamera(rgba));
    RasterDelegate bgra(*context_, VK_FORMAT_B8G8R8A8_UNORM);
    const Tensor a = rgba.Render(scene, "/Render/Settings");
    const Tensor b = bgra.Render(scene, "/Render/Settings");
    ASSERT_EQ(a.Elements(), b.Elements());
    EXPECT_EQ(std::memcmp(a.Data(), b.Data(), static_cast<std::size_t>(a.Elements())), 0);

    RasterDelegate wide(*context_, VK_FORMAT_R16G16B16A16_SFLOAT);
    const Tensor c = wide.Render(scene, "/Render/Settings");
    EXPECT_EQ(std::memcmp(a.Data(), c.Data(), static_cast<std::size_t>(a.Elements())), 0);
}

TEST_F(RasterDelegateTest, SyncsHeightfieldEditsRemovalsAndTextures) {
    Scene scene = TerrainScene(RollingTerrain(33, 17, 2.0));
    RasterDelegate delegate(*context_);
    delegate.Sync(scene);
    EXPECT_EQ(delegate.Terrain().Width(), 33u);
    EXPECT_EQ(delegate.Terrain().Height(), 17u);

    scene.Write("/World/Terrain", tokens::kHeights, HeightsOf(RollingTerrain(9, 5, 2.0)));
    delegate.Sync(scene);
    EXPECT_EQ(delegate.Terrain().Width(), 9u);

    scene.Write("/Looks/Rock", tokens::kDetailAlbedo, Tensor::Zeros(DType::UInt8, {8, 8, 3}));
    scene.Write("/Looks/Rock", tokens::kDetailNormal, Tensor::Zeros(DType::Float32, {8, 8, 4}));
    scene.Write("/Looks/Rock", tokens::kAlbedoMap, Tensor::Zeros(DType::Float32, {4, 4}));
    delegate.Sync(scene);
    ASSERT_TRUE(delegate.MaterialId("/Looks/Rock").has_value());
    const ernie::RasterMaterial& rock =
        delegate.Renderer().Material(*delegate.MaterialId("/Looks/Rock"));
    EXPECT_TRUE(rock.HasDetailAlbedo());
    EXPECT_TRUE(rock.HasDetailNormal());
    EXPECT_TRUE(rock.HasAlbedoMap());

    scene.Clear("/Looks/Rock", tokens::kDetailAlbedo);
    delegate.Sync(scene);
    EXPECT_FALSE(rock.HasDetailAlbedo());
    EXPECT_TRUE(rock.HasDetailNormal());
    scene.Remove("/Looks/Rock");
    delegate.Sync(scene);
    EXPECT_FALSE(delegate.MaterialId("/Looks/Rock").has_value());

    scene.Set("/World/Terrain", tokens::kVisibility, tokens::kInvisible);
    delegate.Sync(scene);
    EXPECT_FALSE(delegate.Terrain().HasHeightfield());
    scene.Set("/World/Terrain", tokens::kVisibility, tokens::kInherited);
    delegate.Sync(scene);
    EXPECT_TRUE(delegate.Terrain().HasHeightfield());
}

TEST_F(RasterDelegateTest, ReportsBadTensorsOnceWithThePrimAndAttribute) {
    Scene scene = TerrainScene(RollingTerrain(9, 9, 2.0));
    scene.Write("/World/Terrain", tokens::kHeights, Tensor::Zeros(DType::Float64, {9, 9}));
    RasterDelegate delegate(*context_);
    try {
        delegate.Sync(scene);
        FAIL() << "expected an upload error";
    } catch (const std::invalid_argument& error) {
        EXPECT_NE(std::string(error.what()).find("heights of '/World/Terrain'"), std::string::npos)
            << error.what();
    }
    scene.Set("/World/Terrain", tokens::kExaggeration, 2.0f);
    EXPECT_NO_THROW(delegate.Sync(scene)) << "not retried until the heights change";

    scene.Write("/Looks/Rock", tokens::kDetailNormal, Tensor::Zeros(DType::Int32, {2, 2}));
    EXPECT_THROW(delegate.Sync(scene), std::invalid_argument);
    scene.Write("/Looks/Rock", tokens::kAlbedoMap, Tensor::Zeros(DType::Float32, {4}));
    EXPECT_THROW(delegate.Sync(scene), std::invalid_argument);
}

TEST_F(RasterDelegateTest, SwitchingScenesResyncsEverything) {
    Scene first = TerrainScene(RollingTerrain(33, 17, 2.0));
    for (int i = 0; i < 20; ++i) {
        first.Set("/World/Terrain", tokens::kExaggeration, 1.0f);
    }
    RasterDelegate delegate(*context_);
    delegate.Sync(first);
    EXPECT_EQ(delegate.Terrain().Width(), 33u);

    Scene second = TerrainScene(RollingTerrain(9, 5, 2.0));
    ASSERT_LT(second.Version(), first.Version()) << "an older cursor must not hide it";
    delegate.Sync(second);
    EXPECT_EQ(delegate.Terrain().Width(), 9u);

    Scene empty;
    delegate.Sync(empty);
    EXPECT_FALSE(delegate.Terrain().HasHeightfield());
    delegate.Sync(second);
    EXPECT_TRUE(delegate.Terrain().HasHeightfield());
}

TEST_F(RasterDelegateTest, DrawsMeshesAndInstancersWithTheirMaterials) {
    Scene scene = MeshScene();
    RasterDelegate delegate(*context_);
    Tensor image = delegate.Render(scene);
    EXPECT_EQ(delegate.Meshes().MeshCount(), 2u);
    EXPECT_EQ(delegate.Meshes().InstancerCount(), 1u);
    const std::uint8_t* quad = TexelAt(scene, image, {-12, 0, 0});
    EXPECT_GT(quad[0], 200);
    EXPECT_LT(quad[1] + quad[2], quad[0] / 2) << "red from its material";
    for (const float y : {-10.0f, 0.0f, 10.0f}) {
        const std::uint8_t* rock = TexelAt(scene, image, {12, y, 0});
        EXPECT_GT(rock[1], 200) << "green from displayColor, instance at y " << y;
        EXPECT_LT(rock[0] + rock[2], rock[1] / 2);
    }
    const std::uint8_t* prototype = TexelAt(scene, image, {0, 0, 0});
    EXPECT_EQ(prototype[0] + prototype[1] + prototype[2], 0) << "prototypes draw only as instances";
    EXPECT_TRUE(delegate.Renderer().Cascades().size() > 0) << "meshes alone enable shadows";
}

TEST_F(RasterDelegateTest, MovesInstancesAndDeformsMeshesInPlace) {
    Scene scene = MeshScene();
    RasterDelegate delegate(*context_);
    delegate.Render(scene);

    const std::array<float, 9> moved{0, -10, 0, 12, 0, 0, 12, 10, 0};
    scene.Write("/World/Rocks", tokens::kPositions, Tensor::Copy<float>(moved, {3, 3}));
    const std::array<float, 12> points{-2, 6, 0, 2, 6, 0, 2, 10, 0, -2, 10, 0};
    scene.Write("/World/Quad", tokens::kPoints, Tensor::Copy<float>(points, {4, 3}));
    const Tensor image = delegate.Render(scene);
    EXPECT_EQ(delegate.Meshes().MeshCount(), 2u);
    EXPECT_GT(TexelAt(scene, image, {0, -10, 0})[1], 200) << "the instance moved";
    EXPECT_EQ(TexelAt(scene, image, {12, -10, 0})[1], 0);
    EXPECT_GT(TexelAt(scene, image, {-12, 8, 0})[0], 200) << "the mesh deformed";
    EXPECT_EQ(TexelAt(scene, image, {-12, 0, 0})[0], 0);

    scene.Remove("/World/Rocks");
    const Tensor removed = delegate.Render(scene);
    EXPECT_EQ(delegate.Meshes().InstancerCount(), 0u);
    EXPECT_EQ(TexelAt(scene, removed, {12, 0, 0})[1], 0);
    EXPECT_EQ(delegate.Meshes().MeshCount(), 1u) << "the prototype went with its instancer";
}

TEST_F(RasterDelegateTest, ABadMeshDoesNotBlockTheOthers) {
    Scene scene = MeshScene();
    DefineQuad(scene, "/World/Broken");
    const std::array<std::int64_t, 4> bad{0, 1, 2, 9};
    scene.Write("/World/Broken", tokens::kFaceVertexIndices, Tensor::Copy<std::int64_t>(bad, {4}));
    RasterDelegate delegate(*context_);
    try {
        delegate.Sync(scene);
        FAIL() << "expected an upload error";
    } catch (const std::invalid_argument& error) {
        EXPECT_NE(std::string(error.what()).find("'/World/Broken' mesh"), std::string::npos)
            << error.what();
    }
    const Tensor image = delegate.Render(scene);
    EXPECT_GT(TexelAt(scene, image, {-12, 0, 0})[0], 200) << "the good quad still draws";
}

TEST_F(RasterDelegateTest, CudaArraysAndCudaOutputMatchTheHostPath) {
    RasterDelegate delegate(*context_);
    std::string why;
    ernie::ComputeInterop* cuda = delegate.Compute(&why);
    if (cuda == nullptr) {
        GTEST_SKIP() << "No CUDA interop: " << why;
    }
    Scene host = MeshScene();
    RasterDelegate reference(*context_);
    const Tensor expected = reference.Render(host);

    Scene device = MeshScene();
    device.Write("/World/Rocks", tokens::kPositions,
                 cuda->Upload(*device.Read("/World/Rocks", tokens::kPositions)));
    device.Set("/World/Rocks", tokens::kExtent,
               std::vector<float>{-2.0f, -12.0f, -1.0f, 14.0f, 12.0f, 1.0f});
    device.Write("/World/Quad", tokens::kPoints,
                 cuda->Upload(*device.Read("/World/Quad", tokens::kPoints)));
    const Tensor image = delegate.Render(device);
    ASSERT_EQ(image.Elements(), expected.Elements());
    EXPECT_EQ(
        std::memcmp(image.Data(), expected.Data(), static_cast<std::size_t>(image.Elements())), 0)
        << "CUDA inputs draw the same image";

    const Tensor first = delegate.Render(device, {}, {ernie::DeviceKind::Cuda, cuda->Device()});
    EXPECT_EQ(first.Location().kind, ernie::DeviceKind::Cuda);
    EXPECT_EQ(first.Shape(), expected.Shape());
    const Tensor second = delegate.Render(device, {}, {ernie::DeviceKind::Cuda, cuda->Device()});
    EXPECT_NE(first.Data(), second.Data()) << "a live result is not overwritten";
    const Tensor downloaded = cuda->Download(first);
    EXPECT_EQ(std::memcmp(downloaded.Data(), expected.Data(),
                          static_cast<std::size_t>(expected.Elements())),
              0)
        << "the CUDA result holds the render";

    const std::array<float, 9> moved{12, -10, 0, 0, 0, 0, 12, 10, 0};
    device.Write("/World/Rocks", tokens::kPositions,
                 cuda->Upload(Tensor::Copy<float>(moved, {3, 3})));
    const Tensor after = delegate.Render(device);
    EXPECT_GT(TexelAt(device, after, {0, 0, 0})[1], 200) << "a moved CUDA instance";

    EXPECT_THROW(delegate.Render(device, {}, {ernie::DeviceKind::Cuda, cuda->Device() + 7}),
                 std::invalid_argument);
}

TEST(ExtentOf, ReadsValuesAndArrays) {
    Scene scene;
    scene.Define("/A", tokens::kMesh).Set(tokens::kExtent, std::vector<float>{0, 1, 2, 3, 4, 5});
    const auto extent = ernie::ExtentOf(scene, "/A");
    ASSERT_TRUE(extent.has_value());
    EXPECT_EQ(extent->second, glm::vec3(3.0f, 4.0f, 5.0f));
    const std::array<double, 6> values{-1, -1, -1, 1, 1, 1};
    scene.Write("/A", tokens::kExtent, Tensor::Copy<double>(values, {2, 3}));
    EXPECT_EQ(ernie::ExtentOf(scene, "/A")->first, glm::vec3(-1.0f));
    scene.Set("/A", tokens::kExtent, std::vector<float>{1, 2});
    EXPECT_FALSE(ernie::ExtentOf(scene, "/A").has_value());
    EXPECT_FALSE(ernie::ExtentOf(scene, "/Missing").has_value());
}

TEST_F(RasterDelegateTest, DrawsUsdGeomPrimitivesAndInstancesThem) {
    Scene scene = MeshScene();
    scene.Remove("/World/Quad");
    scene.Remove("/World/Rocks/Rock");
    scene.Define("/World/Ball", tokens::kSphere)
        .Set(tokens::kGeomRadius, 3.0)
        .Set(tokens::kMaterialBinding, Path("/Looks/Red"))
        .Set(tokens::kTransform, glm::translate(glm::dmat4(1.0), glm::dvec3(-12.0, 0.0, 0.0)));
    scene.Define("/World/Rocks/Rock", tokens::kCube)
        .Set(tokens::kSize, 3.0)
        .Set(tokens::kDisplayColor, glm::vec3(0.0f, 1.0f, 0.0f));
    RasterDelegate delegate(*context_);
    const Tensor image = delegate.Render(scene);
    EXPECT_EQ(delegate.Meshes().MeshCount(), 2u);
    const std::uint8_t* ball = TexelAt(scene, image, {-12, 0, 3});
    EXPECT_GT(ball[0], 150);
    EXPECT_LT(ball[1] + ball[2], ball[0] / 2) << "the sphere, red from its material";
    for (const float y : {-10.0f, 0.0f, 10.0f}) {
        const std::uint8_t* rock = TexelAt(scene, image, {12, y, 1.5f});
        EXPECT_GT(rock[1], 150) << "a cube prototype instanced at y " << y;
    }
    const std::uint8_t* origin = TexelAt(scene, image, {0, 0, 0});
    EXPECT_EQ(origin[0] + origin[1] + origin[2], 0) << "the prototype draws only as instances";

    scene.Set("/World/Ball", tokens::kGeomRadius, 0.5);
    const Tensor smaller = delegate.Render(scene);
    const std::uint8_t* rim = TexelAt(scene, smaller, {-10, 0, 0});
    EXPECT_EQ(rim[0], 0) << "editing the radius regenerates the mesh";
}

namespace {

/// FlatScene with its terrain replaced by a grey GroundPlane at `height`.
Scene GroundScene(double height) {
    Scene scene = FlatScene();
    scene.Remove("/World/Terrain");
    scene.Define("/World/Ground", tokens::kGroundPlane)
        .Set(tokens::kGroundHeight, height)
        .Set(tokens::kMaterialBinding, Path("/Looks/Grey"));
    return scene;
}

}  // namespace

TEST_F(RasterDelegateTest, GroundPlanesAreLitAndSortedLikeSurfaces) {
    Scene scene = GroundScene(10.0);
    RasterDelegate delegate(*context_);
    ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
    auto vars = MapVars(outputs);
    const std::size_t c = kVarSize / 2;
    EXPECT_TRUE(delegate.Ground().Enabled());
    EXPECT_NEAR(At<float>(vars.at("DistanceToImagePlane"), c, c), 90.0f, 0.5f);
    EXPECT_NEAR(At<float>(vars.at("DistanceToImagePlane"), 0, 0), 90.0f, 0.5f)
        << "a flat plane under a nadir camera";
    EXPECT_NEAR(At<float>(vars.at("Normal"), c, c, 2), 1.0f, 1e-4f);
    EXPECT_NEAR(At<float>(vars.at("HdrColor"), c, c, 1), 0.5f, 0.02f) << "albedo 0.5 under the sun";
    EXPECT_EQ(delegate.PrimPath(At<std::uint32_t>(vars.at("PrimId"), 0, 0)), Path("/World/Ground"));

    Scene withTerrain = FlatScene();
    withTerrain.Define("/World/Ground", tokens::kGroundPlane).Set(tokens::kGroundHeight, -5.0);
    ernie::RenderProductSetOutputs both = delegate.Step(withTerrain);
    vars = MapVars(both);
    EXPECT_EQ(delegate.PrimPath(At<std::uint32_t>(vars.at("PrimId"), c, c)), Path("/World/Terrain"))
        << "the terrain is in front of a plane below it";
    EXPECT_NEAR(At<float>(vars.at("DistanceToImagePlane"), c, c), 100.0f, 0.5f);

    withTerrain.Set("/World/Ground", tokens::kVisibility, tokens::kInvisible);
    delegate.Step(withTerrain);
    EXPECT_FALSE(delegate.Ground().Enabled());
}

TEST_F(RasterDelegateTest, GroundPlanesReachTheHorizonWithACheckerboard) {
    Scene scene = GroundScene(0.0);
    scene.Set("/World/Ground", tokens::kChecker, true);
    scene.Set("/World/Ground", tokens::kCheckerSize, 4.0);
    scene.Set("/World/Ground", tokens::kCheckerColor1, glm::vec3(0.0f));
    RasterDelegate delegate(*context_);
    ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
    const auto vars = MapVars(outputs);
    int light = 0;
    int dark = 0;
    for (std::size_t row = 0; row < kVarSize; ++row) {
        for (std::size_t col = 0; col < kVarSize; ++col) {
            const float value = At<float>(vars.at("HdrColor"), row, col, 1);
            light += value > 0.4f ? 1 : 0;
            dark += value < 0.1f ? 1 : 0;
        }
    }
    EXPECT_GT(light, 300) << "4 m squares at about 1 m per pixel are crisp";
    EXPECT_GT(dark, 300);

    scene.Set("/World/Camera", tokens::kTransform,
              ernie::LookAtTransform({0.0, 0.0, 2.0}, {0.0, 1000.0, 1.0}, {0.0, 0.0, 1.0}));
    ernie::RenderProductSetOutputs horizon = delegate.Step(scene);
    const auto far = MapVars(horizon);
    const std::size_t c = kVarSize / 2;
    const std::uint32_t top = At<std::uint32_t>(far.at("PrimId"), 0, c);
    const std::uint32_t bottom = At<std::uint32_t>(far.at("PrimId"), kVarSize - 1, c);
    EXPECT_EQ(top, 0u) << "sky above the horizon";
    EXPECT_EQ(delegate.PrimPath(bottom), Path("/World/Ground"));
    std::size_t lastGround = 0;
    for (std::size_t row = 0; row < kVarSize; ++row) {
        if (At<std::uint32_t>(far.at("PrimId"), row, c) != 0) {
            lastGround = row;
            break;
        }
    }
    EXPECT_GT(lastGround, kVarSize / 2 - 3) << "the plane reaches up to the horizon";
    EXPECT_LT(lastGround, kVarSize / 2 + 3);
    const float distant = At<float>(far.at("HdrColor"), lastGround, c, 1);
    EXPECT_NEAR(distant, 0.25f, 0.1f) << "the filtered checker fades to its mean far away";
}

TEST_F(RasterDelegateTest, PrimitivesStandOnTheGroundAndShadowIt) {
    Scene scene = GroundScene(0.0);
    scene.Set("/World/Sun", tokens::kShadowEnable, true);
    scene.Set("/World/Sun", tokens::kTransform, ernie::DistantLightTransform({1.0, 0.0, 1.0}));
    scene.Define("/World/Block", tokens::kCube)
        .Set(tokens::kSize, 6.0)
        .Set(tokens::kTransform, glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 0.0, 3.0)));
    RasterDelegate delegate(*context_);
    ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
    const auto vars = MapVars(outputs);
    const auto [row, col] = PixelOf(scene, 0.0f, 0.0f);
    EXPECT_EQ(delegate.PrimPath(At<std::uint32_t>(vars.at("PrimId"), row, col)),
              Path("/World/Block"));
    const auto [shadowRow, shadowCol] = PixelOf(scene, -5.0f, 0.0f);
    const auto [litRow, litCol] = PixelOf(scene, 5.0f, 5.0f);
    EXPECT_EQ(delegate.PrimPath(At<std::uint32_t>(vars.at("PrimId"), shadowRow, shadowCol)),
              Path("/World/Ground"));
    EXPECT_LT(At<float>(vars.at("HdrColor"), shadowRow, shadowCol, 1),
              0.2f * At<float>(vars.at("HdrColor"), litRow, litCol, 1))
        << "the cube shadows the ground away from the sun";
}

TEST_F(RasterDelegateTest, FloatingCastersShadowTheGroundFarBelowThem) {
    // A block 8 above the ground: its shadow lands below the level of its own bottom, where
    // cascades fitted to the casters alone would not reach.
    Scene scene = GroundScene(0.0);
    scene.Set("/World/Sun", tokens::kShadowEnable, true);
    scene.Set("/World/Sun", tokens::kTransform, ernie::DistantLightTransform({1.0, 0.0, 1.0}));
    scene.Define("/World/Block", tokens::kCube)
        .Set(tokens::kSize, 4.0)
        .Set(tokens::kTransform, glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 0.0, 8.0)));
    RasterDelegate delegate(*context_);
    ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
    const auto vars = MapVars(outputs);
    // Under the sun at 45 degrees, the shadow of the block's centre falls 8 to the -x side.
    const auto [shadowRow, shadowCol] = PixelOf(scene, -8.0f, 0.0f);
    const auto [litRow, litCol] = PixelOf(scene, 5.0f, 5.0f);
    EXPECT_EQ(delegate.PrimPath(At<std::uint32_t>(vars.at("PrimId"), shadowRow, shadowCol)),
              Path("/World/Ground"));
    EXPECT_LT(At<float>(vars.at("HdrColor"), shadowRow, shadowCol, 1),
              0.2f * At<float>(vars.at("HdrColor"), litRow, litCol, 1));
}

namespace {

/// MeshScene under /World moved by `offset`, with a checkered ground plane 1 below the meshes (2
/// unit squares, so offsets in multiples of 4 keep the pattern in place), a lamp and a terrain
/// beside them.
Scene MovedScene(const glm::dvec3& offset) {
    Scene scene = MeshScene();
    scene.Define("/World", tokens::kXform)
        .Set(tokens::kTransform, glm::translate(glm::dmat4(1.0), offset));
    scene.Define("/World/Ground", tokens::kGroundPlane)
        .Set(tokens::kGroundHeight, offset.z - 1.0)
        .Set(tokens::kChecker, true)
        .Set(tokens::kCheckerSize, 2.0);
    scene.Define("/World/Lamp", tokens::kSphereLight)
        .Set(tokens::kTransform, glm::translate(glm::dmat4(1.0), glm::dvec3(4.0, 4.0, 6.0)))
        .Set(tokens::kIntensity, 30.0f);
    return scene;
}

}  // namespace

TEST_F(RasterDelegateTest, ScenesFarFromTheWorldOriginRenderAsNearIt) {
    // About 2000 km out and 3 km up, where float positions are 0.25 m apart: rendered relative to
    // an origin near the camera, the image matches the scene's at the world origin.
    RasterDelegate delegate(*context_);
    Scene near = MovedScene(glm::dvec3(0.0));
    const std::vector<std::uint8_t> expected = Bytes(delegate.Render(near));
    Scene far = MovedScene(glm::dvec3(1'234'564.0, -1'765'432.0, 3000.0));
    const std::vector<std::uint8_t> moved = Bytes(delegate.Render(far));
    EXPECT_LE(BytesDiffering(moved, expected), expected.size() / 1000)
        << "the origin moved with the camera, a rebase";
    // And back near the world origin: the origin follows again.
    EXPECT_LE(BytesDiffering(Bytes(delegate.Render(near)), expected), expected.size() / 1000);
}

TEST_F(RasterDelegateTest, VisibleShadowFitsSpanOnlyTheDepthsTheViewSees) {
    // MeshScene's meshes lie 60 below its camera: fitted to what the view saw, the sun's first
    // cascade starts near them rather than at the near plane.
    RasterDelegate delegate(*context_);
    Scene scene = MeshScene();
    delegate.Render(scene);
    const float nearPlane = delegate.Renderer().Cascades().front().splitNear;
    scene.Set("/Render/Settings", tokens::kShadowFit, std::string(tokens::kVisible));
    for (int i = 0; i < 4; ++i) {
        delegate.Render(scene);
    }
    const float fitted = delegate.Renderer().Cascades().front().splitNear;
    EXPECT_GT(fitted, 50.0f);
    EXPECT_LT(fitted, 60.0f);
    EXPECT_GT(fitted, 10.0f * nearPlane);
    // A view far from the one measured falls back to the scene's fit until it measures again.
    scene.Set("/World/Camera", tokens::kTransform,
              ernie::LookAtTransform({0.0, 0.0, 30.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}));
    delegate.Render(scene);
    EXPECT_EQ(delegate.Renderer().Cascades().front().splitNear, nearPlane);
    scene.Set("/Render/Settings", tokens::kShadowFit, std::string("nearest"));
    EXPECT_THROW(delegate.Render(scene), std::invalid_argument);
}
