#include "ernie/device/offscreen_target.h"
#include "ernie/geometry/primitives.h"
#include "ernie/raster/delegate.h"
#include "ernie/scene/orbit_controller.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"
#include "ernie/session/renderer.h"
#include "support/test_raster.h"
#include "support/vulkan_fixture.h"

#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <vector>

using ernie::DeviceKind;
using ernie::DType;
using ernie::OrbitController;
using ernie::Path;
using ernie::RasterDelegate;
using ernie::Scene;
using ernie::Tensor;
using ernie::testing::RollingTerrain;
using ernie::testing::TestRaster;
namespace tokens = ernie::tokens;

namespace {

constexpr std::uint32_t kSize = 48;

Tensor HeightsOf(const TestRaster& raster) {
    auto owner = std::make_shared<std::vector<float>>(raster.values);
    return Tensor(
        owner->data(), DType::Float32,
        {static_cast<std::int64_t>(raster.height), static_cast<std::int64_t>(raster.width)}, {}, {},
        owner);
}

/// A camera framing the delegate's terrain from 0.8 rad above the horizon.
ernie::Camera FramingCamera(const RasterDelegate& delegate) {
    OrbitController orbit;
    orbit.pitch = 0.8f;
    const auto [low, high] = delegate.Terrain().Bounds(1.0f);
    orbit.Frame(low, high);
    return orbit.MakeCamera(kSize, kSize);
}

/// A terrain lit by a low sun, with render settings for a `kSize` square image.
Scene TerrainScene(const TestRaster& raster) {
    Scene scene;
    scene.Define("/World/Terrain", tokens::kTerrain)
        .Write(tokens::kHeights, HeightsOf(raster))
        .Set(tokens::kOrigin, glm::dvec2(raster.originX, raster.originY))
        .Set(tokens::kCellSize, raster.cellSize)
        .Set(tokens::kMaterialBinding, Path("/Looks/Rock"));
    scene.Define("/Looks/Rock", tokens::kMaterial).Set(tokens::kAlbedo, glm::vec3(0.6f));
    scene.Define("/World/Sun", tokens::kDistantLight)
        .Set(tokens::kTransform, ernie::DistantLightTransform({1.0, 1.0, 0.6}));
    scene.Define("/Render/Settings", tokens::kRenderSettings)
        .Set(tokens::kCameraRel, Path("/World/Camera"))
        .Set(tokens::kResolution, glm::ivec2(kSize, kSize))
        .Set(tokens::kClearColor, glm::vec3(0.0f, 0.0f, 1.0f));
    return scene;
}

class RasterDelegateTest : public ernie::testing::VulkanTest {};

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

namespace {

/// A square of side 4 in the x-y plane centred on the origin, as one quad face.
void DefineQuad(Scene& scene, const Path& path, float half = 2.0f) {
    const std::array<float, 12> points{-half, -half, 0, half,  -half, 0,
                                       half,  half,  0, -half, half,  0};
    const std::array<std::int64_t, 1> counts{4};
    const std::array<std::int64_t, 4> indices{0, 1, 2, 3};
    scene.Define(path, tokens::kMesh)
        .Write(tokens::kPoints, Tensor::Copy<float>(points, {4, 3}))
        .Write(tokens::kFaceVertexCounts, Tensor::Copy<std::int64_t>(counts, {1}))
        .Write(tokens::kFaceVertexIndices, Tensor::Copy<std::int64_t>(indices, {4}));
}

/// Meshes seen from straight above, lit from overhead, against a black background.
Scene MeshScene() {
    Scene scene;
    scene.Define("/World/Camera", tokens::kCamera)
        .Set(tokens::kTransform,
             ernie::LookAtTransform({0.0, 0.0, 60.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}))
        .Set(tokens::kFocalLength, 18.0f);
    scene.Define("/World/Sun", tokens::kDistantLight)
        .Set(tokens::kTransform, ernie::DistantLightTransform({0.0, 0.0, 1.0}));
    scene.Define("/Render/Settings", tokens::kRenderSettings)
        .Set(tokens::kResolution, glm::ivec2(kSize, kSize))
        .Set(tokens::kAmbient, 0.0f);
    scene.Define("/Looks/Red", tokens::kMaterial).Set(tokens::kAlbedo, glm::vec3(1.0f, 0.0f, 0.0f));
    DefineQuad(scene, "/World/Quad");
    scene.Set("/World/Quad", tokens::kTransform,
              glm::translate(glm::dmat4(1.0), glm::dvec3(-12.0, 0.0, 0.0)));
    scene.Set("/World/Quad", tokens::kMaterialBinding, Path("/Looks/Red"));

    const std::array<float, 9> positions{12, -10, 0, 12, 0, 0, 12, 10, 0};
    scene.Define("/World/Rocks", tokens::kPointInstancer)
        .Write(tokens::kPositions, Tensor::Copy<float>(positions, {3, 3}))
        .Set(tokens::kPrototypes, std::vector<Path>{"/World/Rocks/Rock"});
    DefineQuad(scene, "/World/Rocks/Rock", 1.5f);
    scene.Set("/World/Rocks/Rock", tokens::kDisplayColor, glm::vec3(0.0f, 1.0f, 0.0f));
    return scene;
}

/// The RGBA texel showing world point `p` in `image`.
const std::uint8_t* TexelAt(const Scene& scene, const Tensor& image, glm::vec3 p) {
    const glm::vec4 clip =
        ernie::CameraOf(scene, "/World/Camera", kSize, kSize).RasterView().ViewProjection() *
        glm::vec4(p, 1.0f);
    const glm::vec2 ndc = glm::vec2(clip) / clip.w;
    const auto col = static_cast<std::size_t>((ndc.x * 0.5f + 0.5f) * kSize);
    const auto row = static_cast<std::size_t>((0.5f - ndc.y * 0.5f) * kSize);
    return image.As<std::uint8_t>() + 4 * (row * kSize + col);
}

}  // namespace

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
    ernie::CudaInterop* cuda = delegate.Cuda(&why);
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

namespace {

/// MeshScene with two RenderProducts: the overhead camera at 32 x 32, and a second camera to the
/// side at 16 x 24, each with an LdrColor RenderVar.
Scene ProductScene() {
    Scene scene = MeshScene();
    scene.Define("/World/Side", tokens::kCamera)
        .Set(tokens::kTransform,
             ernie::LookAtTransform({60.0, 0.0, 20.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}))
        .Set(tokens::kFocalLength, 18.0f);
    scene.Define("/Render/Vars/Color", tokens::kRenderVar).Set(tokens::kSourceName, "LdrColor");
    scene.Define("/Render/Top", tokens::kRenderProduct)
        .Set(tokens::kCameraRel, Path("/World/Camera"))
        .Set(tokens::kResolution, glm::ivec2(32, 32))
        .Set(tokens::kOrderedVars, std::vector<Path>{"/Render/Vars/Color"});
    scene.Define("/Render/SideView", tokens::kRenderProduct)
        .Set(tokens::kCameraRel, Path("/World/Side"))
        .Set(tokens::kResolution, glm::ivec2(16, 24));
    scene.Set("/Render/Settings", tokens::kProducts,
              std::vector<Path>{"/Render/Top", "/Render/SideView"});
    return scene;
}

}  // namespace

TEST(ResolveRenderProducts, FollowsUsdRender) {
    Scene scene = ProductScene();
    std::vector<ernie::RenderProductSpec> specs = ernie::ResolveRenderProducts(scene, {}, {});
    ASSERT_EQ(specs.size(), 2u);
    EXPECT_EQ(specs[0].product, Path("/Render/Top"));
    EXPECT_EQ(specs[0].resolution.width, 32u);
    EXPECT_EQ(specs[0].vars.front().first, Path("/Render/Vars/Color"));
    EXPECT_EQ(specs[1].camera, Path("/World/Side"));
    EXPECT_EQ(specs[1].resolution.height, 24u);
    EXPECT_EQ(specs[1].vars.front().second, "LdrColor") << "default var";

    const Path only[] = {"/Render/SideView"};
    EXPECT_EQ(ernie::ResolveRenderProducts(scene, only, {}).size(), 1u);
    scene.Clear("/Render/Settings", tokens::kProducts);
    specs = ernie::ResolveRenderProducts(scene, {}, {});
    ASSERT_EQ(specs.size(), 1u);
    EXPECT_EQ(specs[0].product, Path("/Render/Settings")) << "default product";
    EXPECT_EQ(specs[0].resolution.width, kSize);

    const Path bad[] = {"/World/Camera"};
    EXPECT_THROW(ernie::ResolveRenderProducts(scene, bad, {}), std::invalid_argument);
    scene.Set("/Render/Vars/Color", tokens::kSourceName, "Depth");
    const Path top[] = {"/Render/Top"};
    EXPECT_THROW(ernie::ResolveRenderProducts(scene, top, {}), std::invalid_argument);
}

TEST_F(RasterDelegateTest, StepsRenderEveryProductSyncOrAsync) {
    Scene scene = ProductScene();
    RasterDelegate delegate(*context_);
    ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
    ASSERT_EQ(outputs.size(), 2u);
    ernie::RenderVar& top = outputs.at("/Render/Top").vars.at("LdrColor");
    EXPECT_EQ(top.GetPath(), Path("/Render/Vars/Color"));
    const Tensor topImage = top.Map(ernie::MapDevice::Cpu).tensor;
    EXPECT_EQ(topImage.Shape(), (std::vector<std::int64_t>{32, 32, 4}));
    const Tensor side =
        outputs.at("/Render/SideView").vars.at("LdrColor").Map(ernie::MapDevice::Cpu).tensor;
    EXPECT_EQ(side.Shape(), (std::vector<std::int64_t>{24, 16, 4}));
    scene.Define("/Render/TopSmall", tokens::kRenderProduct)
        .Set(tokens::kCameraRel, Path("/World/Camera"))
        .Set(tokens::kResolution, glm::ivec2(16, 24));
    const Path pair[] = {"/Render/TopSmall", "/Render/SideView"};
    ernie::RenderProductSetOutputs both = delegate.Step(scene, pair);
    const Tensor small =
        both.at("/Render/TopSmall").vars.at("LdrColor").Map(ernie::MapDevice::Cpu).tensor;
    const Tensor sideAgain =
        both.at("/Render/SideView").vars.at("LdrColor").Map(ernie::MapDevice::Cpu).tensor;
    EXPECT_NE(
        std::memcmp(small.Data(), sideAgain.Data(), static_cast<std::size_t>(small.Elements())), 0)
        << "each product renders through its own camera";
    EXPECT_EQ(std::memcmp(side.Data(), sideAgain.Data(), static_cast<std::size_t>(side.Elements())),
              0);

    ernie::StepOperation operation = delegate.StepAsync(scene);
    EXPECT_TRUE(operation.Wait());
    EXPECT_TRUE(operation.Done());
    const Tensor again =
        operation.Fetch().at("/Render/Top").vars.at("LdrColor").Map(ernie::MapDevice::Cpu).tensor;
    EXPECT_EQ(
        std::memcmp(again.Data(), topImage.Data(), static_cast<std::size_t>(topImage.Elements())),
        0)
        << "async and sync steps render the same";
}

TEST_F(RasterDelegateTest, FramesHeldByOutputsAreNotRenderedOver) {
    Scene scene = ProductScene();
    RasterDelegate delegate(*context_);
    const Path top[] = {"/Render/Top"};
    ernie::RenderProductSetOutputs first = delegate.Step(scene, top);
    const Tensor before =
        first.at("/Render/Top").vars.at("LdrColor").Map(ernie::MapDevice::Cpu).tensor;
    scene.Set("/Looks/Red", tokens::kAlbedo, glm::vec3(0.0f, 0.0f, 1.0f));
    ernie::RenderProductSetOutputs second = delegate.Step(scene, top);
    const Tensor held =
        first.at("/Render/Top").vars.at("LdrColor").Map(ernie::MapDevice::Cpu).tensor;
    EXPECT_EQ(std::memcmp(held.Data(), before.Data(), static_cast<std::size_t>(before.Elements())),
              0)
        << "the held frame kept its image";
    const Tensor after =
        second.at("/Render/Top").vars.at("LdrColor").Map(ernie::MapDevice::Cpu).tensor;
    EXPECT_NE(std::memcmp(after.Data(), before.Data(), static_cast<std::size_t>(before.Elements())),
              0);
}

TEST_F(RasterDelegateTest, MapsRenderVarsToCudaArraysWithoutCopying) {
    RasterDelegate delegate(*context_);
    std::string why;
    ernie::CudaInterop* cuda = delegate.Cuda(&why);
    if (cuda == nullptr) {
        GTEST_SKIP() << "No CUDA interop: " << why;
    }
    Scene scene = ProductScene();
    const Path top[] = {"/Render/Top"};
    ernie::StepOperation operation = delegate.StepAsync(scene, top);
    ernie::RenderVar var = operation.Outputs().at("/Render/Top").vars.at("LdrColor");
    const Tensor expected = var.Map(ernie::MapDevice::Cpu).tensor;

    const ernie::MappedRenderVar array = var.Map(ernie::MapDevice::CudaArray);
    EXPECT_NE(array.array.array, 0u);
    EXPECT_TRUE(var.IsMapped());
    const Tensor fromArray = cuda->DownloadArray(array.array);
    EXPECT_EQ(std::memcmp(fromArray.Data(), expected.Data(),
                          static_cast<std::size_t>(expected.Elements())),
              0);
    EXPECT_EQ(var.Map(ernie::MapDevice::CudaArray).array.array, array.array.array)
        << "the frame's own image";
    var.Unmap();
    var.Unmap();
    EXPECT_FALSE(var.IsMapped());

    const ernie::MappedRenderVar linear = var.Map(ernie::MapDevice::Cuda);
    EXPECT_EQ(linear.tensor.Location().kind, ernie::DeviceKind::Cuda);
    const Tensor fromLinear = cuda->Download(linear.tensor);
    EXPECT_EQ(std::memcmp(fromLinear.Data(), expected.Data(),
                          static_cast<std::size_t>(expected.Elements())),
              0);

    var = {};
    operation = delegate.StepAsync(scene, top);
    EXPECT_TRUE(operation.Wait());
}

TEST_F(RasterDelegateTest, CudaArraysMappedInTheSceneAreWrittenInPlace) {
    RasterDelegate delegate(*context_);
    std::string why;
    ernie::CudaInterop* cuda = delegate.Cuda(&why);
    if (cuda == nullptr) {
        GTEST_SKIP() << "No CUDA interop: " << why;
    }
    Scene scene = MeshScene();
    scene.Set("/World/Rocks", tokens::kExtent,
              std::vector<float>{-14.0f, -14.0f, -1.0f, 14.0f, 14.0f, 1.0f});
    // A simulation keeps its state in CUDA arrays, written into the scene once.
    const std::array<float, 9> start{12, -10, 0, 12, 0, 0, 12, 10, 0};
    scene.Write("/World/Rocks", tokens::kPositions,
                cuda->Upload(Tensor::Copy<float>(start, {3, 3})));
    Tensor image = delegate.Render(scene);
    EXPECT_GT(TexelAt(scene, image, {12, -10, 0})[1], 200) << "drawn from the CUDA array";

    // Then it maps and writes them in place every step.
    const Tensor positions = scene.Map("/World/Rocks", tokens::kPositions);
    EXPECT_EQ(positions.Location().kind, ernie::DeviceKind::Cuda);
    const std::array<float, 9> moved{0, 0, 0, 12, 0, 0, 12, 10, 0};
    cuda->Copy(cuda->Upload(Tensor::Copy<float>(moved, {3, 3})), positions);
    const std::uint64_t version = scene.AttributeVersion("/World/Rocks", tokens::kPositions);
    scene.Unmap("/World/Rocks", tokens::kPositions);
    EXPECT_GT(scene.AttributeVersion("/World/Rocks", tokens::kPositions), version);
    image = delegate.Render(scene);
    EXPECT_GT(TexelAt(scene, image, {0, 0, 0})[1], 200) << "drawn from the in-place write";
    EXPECT_EQ(TexelAt(scene, image, {12, -10, 0})[1], 0);

    const std::array<float, 12> shifted{-2, 6, 0, 2, 6, 0, 2, 10, 0, -2, 10, 0};
    scene.Write("/World/Quad", tokens::kPoints, cuda->Upload(Tensor::Copy<float>(shifted, {4, 3})));
    DefineQuad(scene, "/World/Extra", 1.0f);
    scene.Set("/World/Extra", tokens::kTransform,
              glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, -12.0, 0.0)));
    image = delegate.Render(scene);
    EXPECT_GT(TexelAt(scene, image, {0, 0, 0})[1], 200) << "CUDA positions survive a rebuild";
    EXPECT_GT(TexelAt(scene, image, {-12, 8, 0})[0], 200) << "CUDA points survive a rebuild";
    EXPECT_GT(TexelAt(scene, image, {0, -12, 0})[0] + TexelAt(scene, image, {0, -12, 0})[1], 0)
        << "the added mesh draws";

    EXPECT_THROW(scene.Map("/World/Camera", tokens::kPoints), std::invalid_argument)
        << "only written arrays map";
}

TEST(ResolveRenderProducts, ReadsSensorFilterChains) {
    Scene scene = ProductScene();
    scene.Define("/Render/Filters/Cie", tokens::kSensorFilter).Set(tokens::kFilterType, "cie");
    scene.Define("/Render/Filters/Exposure", tokens::kSensorFilter)
        .Set(tokens::kFilterType, "exposure");
    scene.Set("/Render/Top", tokens::kFilters,
              std::vector<Path>{"/Render/Filters/Cie", "/Render/Filters/Exposure"});
    const Path top[] = {"/Render/Top"};
    const std::vector<ernie::RenderProductSpec> specs =
        ernie::ResolveRenderProducts(scene, top, {});
    ASSERT_EQ(specs.size(), 1u);
    EXPECT_EQ(specs[0].filters, (std::vector<ernie::SensorFilter>{ernie::SensorFilter::Cie,
                                                                  ernie::SensorFilter::Exposure}));

    scene.Define("/Render/Vars/Dn", tokens::kRenderVar).Set(tokens::kSourceName, "RawDN");
    scene.Set("/Render/Top", tokens::kOrderedVars, std::vector<Path>{"/Render/Vars/Dn"});
    EXPECT_THROW(ernie::ResolveRenderProducts(scene, top, {}), std::invalid_argument)
        << "the chain has no ADC";
    scene.Set("/Render/Top", tokens::kOrderedVars, std::vector<Path>{"/Render/Vars/Color"});
    scene.Set("/Render/Filters/Cie", tokens::kFilterType, "bloom");
    EXPECT_THROW(ernie::ResolveRenderProducts(scene, top, {}), std::invalid_argument);
    scene.Set("/Render/Top", tokens::kFilters, std::vector<Path>{"/World/Camera"});
    EXPECT_THROW(ernie::ResolveRenderProducts(scene, top, {}), std::invalid_argument);
    scene.Define("/Render/Filters/Adc", tokens::kSensorFilter).Set(tokens::kFilterType, "adc");
    scene.Set("/Render/Top", tokens::kFilters, std::vector<Path>{"/Render/Filters/Adc"});
    EXPECT_THROW(ernie::ResolveRenderProducts(scene, top, {}), std::invalid_argument)
        << "adc needs qe";
}

namespace {

constexpr std::uint32_t kVarSize = 32;

/// A flat grey (albedo 0.5) Hillshade terrain 100 m below a camera looking straight down, lit by
/// a relative white sun overhead without shadows, rendering every var at 32 x 32.
Scene FlatScene() {
    Scene scene;
    scene.Define("/World/Terrain", tokens::kTerrain)
        .Write(tokens::kHeights, HeightsOf(TestRaster::Centred(65, 65, 2.0)))
        .Set(tokens::kOrigin, glm::dvec2(-65.0, 65.0))
        .Set(tokens::kCellSize, 2.0)
        .Set(tokens::kMaterialBinding, Path("/Looks/Grey"));
    scene.Define("/Looks/Grey", tokens::kMaterial).Set(tokens::kAlbedo, glm::vec3(0.5f));
    scene.Define("/World/Sun", tokens::kDistantLight)
        .Set(tokens::kTransform, ernie::DistantLightTransform({0.0, 0.0, 1.0}))
        .Set(tokens::kShadowEnable, false);
    scene.Define("/World/Camera", tokens::kCamera)
        .Set(tokens::kTransform,
             ernie::LookAtTransform({0.0, 0.0, 100.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}))
        .Set(tokens::kSensorNoise, false);
    std::vector<Path> vars;
    for (const std::string_view name : ernie::RenderTargets::VarNames()) {
        const Path var = Path("/Render/Vars").Child(std::string(name));
        scene.Define(var, tokens::kRenderVar).Set(tokens::kSourceName, std::string(name));
        vars.push_back(var);
    }
    scene.Define("/Render/Settings", tokens::kRenderSettings)
        .Set(tokens::kCameraRel, Path("/World/Camera"))
        .Set(tokens::kResolution, glm::ivec2(kVarSize, kVarSize))
        .Set(tokens::kColorMode, tokens::kHillshade)
        .Set(tokens::kAmbient, 0.0f);
    scene.Define("/Render/Product", tokens::kRenderProduct).Set(tokens::kOrderedVars, vars);
    scene.Set("/Render/Settings", tokens::kProducts, std::vector<Path>{"/Render/Product"});
    return scene;
}

/// Every var of `outputs`' product mapped to the CPU.
std::map<std::string, Tensor, std::less<>> MapVars(ernie::RenderProductSetOutputs& outputs) {
    std::map<std::string, Tensor, std::less<>> tensors;
    for (auto& [name, var] : outputs.at("/Render/Product").vars) {
        tensors[name] = var.Map(ernie::MapDevice::Cpu).tensor;
    }
    return tensors;
}

template <typename T>
T At(const Tensor& tensor, std::size_t row, std::size_t col, std::size_t channel = 0) {
    const std::size_t channels = tensor.Shape().size() == 3 ? tensor.Shape()[2] : 1;
    return static_cast<const T*>(tensor.Data())[(row * kVarSize + col) * channels + channel];
}

}  // namespace

TEST_F(RasterDelegateTest, RenderVarsMeasureTheScene) {
    Scene scene = FlatScene();
    RasterDelegate delegate(*context_);
    ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
    const auto vars = MapVars(outputs);
    ASSERT_EQ(vars.size(), ernie::RenderTargets::VarNames().size());
    const std::size_t c = kVarSize / 2;

    const Tensor& spectral = vars.at("SpectralRadiance");
    ASSERT_EQ(spectral.Shape(), (std::vector<std::int64_t>{16, kVarSize, kVarSize}));
    EXPECT_EQ(spectral.Type(), DType::Float32);
    double luminanceIntegral = 0.0;
    for (double w = 360.0; w <= 830.0; w += 1.0) {
        luminanceIntegral += ernie::CieXyz(w).y;
    }
    const float* radiance = static_cast<const float*>(spectral.Data());
    const double expected = 0.5 / (ernie::kLuminousEfficacy * luminanceIntegral);
    EXPECT_NEAR(radiance[8 * kVarSize * kVarSize + c * kVarSize + c], expected, 0.03 * expected)
        << "L = E / pi * albedo for a flat sun of white luminance 1 cd/m^2";

    const Tensor& hdr = vars.at("HdrColor");
    EXPECT_EQ(hdr.Shape(), (std::vector<std::int64_t>{kVarSize, kVarSize, 4}));
    for (std::size_t k = 0; k < 3; ++k) {
        EXPECT_NEAR(At<float>(hdr, c, c, k), 0.5f, 0.02f) << "neutral grey of 0.5 cd/m^2";
    }
    const int ldr = At<std::uint8_t>(vars.at("LdrColor"), c, c);
    EXPECT_NEAR(ldr, 188, 3) << "sRGB encoding of 0.5";

    EXPECT_NEAR(At<float>(vars.at("DistanceToImagePlane"), c, c), 100.0f, 0.5f);
    EXPECT_NEAR(At<float>(vars.at("DistanceToCamera"), c, c), 100.0f, 0.5f);
    EXPECT_GT(At<float>(vars.at("DistanceToCamera"), 0, 0),
              At<float>(vars.at("DistanceToImagePlane"), 0, 0))
        << "off-axis pixels are further from the eye than the image plane";
    EXPECT_NEAR(At<float>(vars.at("Normal"), c, c, 2), 1.0f, 1e-3f);

    const std::uint32_t id = At<std::uint32_t>(vars.at("PrimId"), c, c);
    EXPECT_NE(id, 0u);
    EXPECT_EQ(delegate.PrimPath(id), Path("/World/Terrain"));
    EXPECT_EQ(delegate.PrimPath(0), Path());
    EXPECT_EQ(At<std::uint32_t>(vars.at("InstanceId"), c, c), 0u);

    const ernie::Camera camera = ernie::CameraOf(scene, "/World/Camera", kVarSize, kVarSize);
    const float electrons = At<float>(vars.at("Electrons"), c, c, 1);
    EXPECT_GT(electrons, 0.0f);
    EXPECT_EQ(At<float>(vars.at("Electrons"), c, c, 3), 0.0f) << "an RGB sensor has 3 channels";
    const double perDn = camera.sensor.ElectronsPerDn(camera.iso);
    EXPECT_EQ(vars.at("RawDN").Type(), DType::UInt16);
    EXPECT_NEAR(At<std::uint16_t>(vars.at("RawDN"), c, c, 1), electrons / perDn, 0.5 + 1e-3);
}

TEST_F(RasterDelegateTest, ExposureNoiseAndDistortionFollowTheCamera) {
    Scene scene = FlatScene();
    RasterDelegate delegate(*context_);
    ernie::RenderProductSetOutputs plain = delegate.Step(scene);
    const auto before = MapVars(plain);
    const std::size_t c = kVarSize / 2;

    scene.Set("/World/Camera", tokens::kCameraExposure, 1.0f);
    ernie::RenderProductSetOutputs brighter = delegate.Step(scene);
    const auto stop = MapVars(brighter);
    EXPECT_FLOAT_EQ(At<float>(stop.at("HdrColor"), c, c), At<float>(before.at("HdrColor"), c, c))
        << "exposure is photographic only";
    EXPECT_EQ(At<std::uint8_t>(stop.at("LdrColor"), c, c), 255) << "2 x 0.5 is white";

    scene.Set("/World/Camera", tokens::kCameraExposure, 0.0f);
    scene.Set("/World/Camera", tokens::kSensorNoise, true);
    ernie::RenderProductSetOutputs first = delegate.Step(scene);
    ernie::RenderProductSetOutputs second = delegate.Step(scene);
    const Tensor a =
        first.at("/Render/Product").vars.at("Electrons").Map(ernie::MapDevice::Cpu).tensor;
    const Tensor b =
        second.at("/Render/Product").vars.at("Electrons").Map(ernie::MapDevice::Cpu).tensor;
    double mean = 0.0;
    double clean = 0.0;
    int differing = 0;
    for (std::size_t row = 0; row < kVarSize; ++row) {
        for (std::size_t col = 0; col < kVarSize; ++col) {
            mean += At<float>(a, row, col, 1);
            clean += At<float>(before.at("Electrons"), row, col, 1);
            differing += At<float>(a, row, col, 1) != At<float>(b, row, col, 1) ? 1 : 0;
        }
    }
    EXPECT_GT(differing, static_cast<int>(kVarSize * kVarSize / 2)) << "temporal noise";
    EXPECT_NEAR(mean / clean, 1.0, 0.02) << "noise keeps the mean";

    scene.Set("/World/Camera", tokens::kSensorNoise, false);
    scene.Set("/World/Camera", tokens::kLensDistortion, std::vector<float>{-0.3f, 0, 0, 0, 0});
    ernie::RenderProductSetOutputs distorted = delegate.Step(scene);
    const auto barrel = MapVars(distorted);
    EXPECT_NEAR(At<float>(barrel.at("DistanceToCamera"), c, c), 100.0f, 0.5f);
    EXPECT_NEAR(At<float>(barrel.at("HdrColor"), c, c), 0.5f, 0.02f);
    EXPECT_NE(At<std::uint32_t>(barrel.at("PrimId"), 0, 0), 0u) << "corners stay covered";
    EXPECT_NE(At<float>(barrel.at("DistanceToCamera"), 0, 0),
              At<float>(before.at("DistanceToCamera"), 0, 0))
        << "barrel distortion sees wider at the corners";
}

namespace {

/// The Electrons var of `product` in `outputs`, on the CPU.
std::vector<float> ElectronsOf(ernie::RenderProductSetOutputs& outputs, const Path& product) {
    const Tensor tensor =
        outputs.at(product).vars.at("Electrons").Map(ernie::MapDevice::Cpu).tensor;
    const auto* data = static_cast<const float*>(tensor.Data());
    return {data, data + tensor.Elements()};
}

/// FlatScene with sensor noise, and a second product `/Render/Other` of the same camera.
Scene NoisyScene() {
    Scene scene = FlatScene();
    scene.Set("/World/Camera", tokens::kSensorNoise, true);
    scene.Define("/Render/Other", tokens::kRenderProduct)
        .Set(tokens::kOrderedVars, *scene.Get("/Render/Product", tokens::kOrderedVars));
    return scene;
}

}  // namespace

TEST_F(RasterDelegateTest, NoiseIsDeterministicPerProductAndFrame) {
    Scene scene = NoisyScene();
    const Path product("/Render/Product");
    const Path other("/Render/Other");
    const std::array<Path, 1> one = {product};
    const std::array<Path, 1> second = {other};

    RasterDelegate a(*context_);
    std::vector<std::vector<float>> frames;
    for (int i = 0; i < 2; ++i) {
        ernie::RenderProductSetOutputs outputs = a.Step(scene, one);
        frames.push_back(ElectronsOf(outputs, product));
    }
    EXPECT_NE(frames[0], frames[1]) << "consecutive frames have independent noise";

    RasterDelegate b(*context_);
    b.Update(scene, "/Render/Settings", {kVarSize, kVarSize});
    ernie::RenderProductSetOutputs otherFirst = b.Step(scene, second);
    const std::vector<float> otherNoise = ElectronsOf(otherFirst, other);
    for (int i = 0; i < 2; ++i) {
        ernie::RenderProductSetOutputs outputs = b.Step(scene, one);
        EXPECT_EQ(ElectronsOf(outputs, product), frames[static_cast<std::size_t>(i)])
            << "frame " << i << " does not depend on what else rendered";
    }
    EXPECT_NE(otherNoise, frames[0]) << "products of one camera have their own noise";

    scene.Set("/Render/Settings", tokens::kFrame, 7);
    ernie::RenderProductSetOutputs seventh = a.Step(scene, one);
    ernie::RenderProductSetOutputs again = b.Step(scene, one);
    EXPECT_EQ(ElectronsOf(seventh, product), ElectronsOf(again, product))
        << "an authored frame renders the same noise";
    scene.Set("/Render/Settings", tokens::kFrame, 0);
    ernie::RenderProductSetOutputs zeroth = a.Step(scene, one);
    EXPECT_EQ(ElectronsOf(zeroth, product), frames[0]) << "frame 0 again";
}

TEST_F(RasterDelegateTest, IsARenderBackend) {
    const std::unique_ptr<ernie::RenderDelegate> delegate =
        ernie::CreateRenderDelegate("raster", *context_);
    EXPECT_EQ(delegate->Info().name, "raster");
    EXPECT_FALSE(delegate->Info().progressive);
    EXPECT_NE(delegate->Brdfs().Find("hapke"), nullptr);
    EXPECT_FALSE(delegate->SceneBounds().has_value()) << "nothing synced";
    EXPECT_THROW(ernie::CreateRenderDelegate("pathtracer", *context_), std::invalid_argument);

    Scene scene = FlatScene();
    delegate->Update(scene, "/Render/Settings", {kVarSize, kVarSize});
    const std::optional<ernie::Bounds3> bounds = delegate->SceneBounds();
    ASSERT_TRUE(bounds.has_value());
    EXPECT_NEAR(bounds->first.x, -64.0f, 1e-3f);
    EXPECT_NEAR(bounds->second.x, 64.0f, 1e-3f);
    const ernie::RenderStats stats = delegate->Stats();
    EXPECT_GT(stats.gpuBytes, 0u);
    std::vector<std::string> names;
    for (const auto& [name, value] : stats.counters) {
        names.emplace_back(name);
    }
    EXPECT_EQ(names, (std::vector<std::string>{std::string(ernie::raster_stats::kTerrainPatches),
                                               std::string(ernie::raster_stats::kTerrainLodLevels),
                                               std::string(ernie::raster_stats::kMeshDraws),
                                               std::string(ernie::raster_stats::kMeshDrawCalls)}));
    delegate->Step(scene);
    EXPECT_GT(delegate->Stats().triangles, 0u);
    EXPECT_GT(delegate->Stats().gpuBytes, stats.gpuBytes) << "the product's frame";
}

namespace {

/// FlatScene lit only by a point lamp `height` above the origin pointing down, with a radiant flux
/// of 1000 W over a flat spectrum and a cone of `coneAngle` degrees (180: unshaped).
Scene LampScene(double height, float coneAngle = 180.0f) {
    Scene scene = FlatScene();
    scene.Remove("/World/Sun");
    scene.Define("/World/Lamp", tokens::kSphereLight)
        .Set(tokens::kTransform,
             ernie::LookAtTransform({0.0, 0.0, height}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}))
        .Set(tokens::kTreatAsPoint, true)
        .Set(tokens::kLightPower, 1000.0)
        .Set(tokens::kShapingConeAngle, coneAngle);
    return scene;
}

/// The pixel of FlatScene's image showing ground point (x, y).
std::pair<std::size_t, std::size_t> PixelOf(const Scene& scene, float x, float y) {
    const glm::vec4 clip =
        ernie::CameraOf(scene, "/World/Camera", kSize, kSize).RasterView().ViewProjection() *
        glm::vec4(x, y, 0.0f, 1.0f);
    const glm::vec2 ndc = glm::vec2(clip) / clip.w;
    return {static_cast<std::size_t>((0.5f - ndc.y * 0.5f) * kVarSize),
            static_cast<std::size_t>((ndc.x * 0.5f + 0.5f) * kVarSize)};
}

/// Spectral radiance in band 8 at ground point (x, y) of a step of `scene`.
float RadianceAt(RasterDelegate& delegate, const Scene& scene, float x, float y) {
    ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
    const Tensor spectral =
        outputs.at("/Render/Product").vars.at("SpectralRadiance").Map(ernie::MapDevice::Cpu).tensor;
    const auto [row, col] = PixelOf(scene, x, y);
    return static_cast<const float*>(
        spectral.Data())[8 * kVarSize * kVarSize + row * kVarSize + col];
}

}  // namespace

TEST_F(RasterDelegateTest, LampsLightByTheInverseSquareLaw) {
    RasterDelegate delegate(*context_);
    const Scene near = LampScene(10.0);
    const float intensity = static_cast<float>(1000.0 / (4.0 * M_PI) / 470.0);
    const float x = 3.0f;
    const float distance2 = 100.0f + x * x;
    const float expected =
        intensity * (10.0f / std::sqrt(distance2)) / distance2 * 0.5f / static_cast<float>(M_PI);
    EXPECT_NEAR(RadianceAt(delegate, near, x, 0.0f), expected, 0.04f * expected)
        << "L = I cos(i) / d^2 * albedo / pi";
    EXPECT_FALSE(delegate.Prims().lights.empty());

    const float twice = RadianceAt(delegate, LampScene(20.0), 0.0f, 0.0f);
    const float once = RadianceAt(delegate, LampScene(10.0), 0.0f, 0.0f);
    EXPECT_NEAR(once / twice, 4.0f, 0.1f);

    const Scene spot = LampScene(10.0, 10.0f);
    EXPECT_NEAR(RadianceAt(delegate, spot, 0.0f, 0.0f), once, 0.02f * once) << "inside the cone";
    EXPECT_EQ(RadianceAt(delegate, spot, 6.0f, 0.0f), 0.0f) << "outside a 10 degree cone";
}

TEST_F(RasterDelegateTest, LampsCastShadowsFromSpotAndCubeMaps) {
    RasterDelegate delegate(*context_);
    for (const float cone : {60.0f, 180.0f}) {
        Scene scene = LampScene(10.0, cone);
        const float lit = RadianceAt(delegate, scene, 3.0f, 0.0f);
        DefineQuad(scene, "/World/Blocker");
        scene.Set("/World/Blocker", tokens::kTransform,
                  glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 0.0, 5.0)));
        EXPECT_LT(RadianceAt(delegate, scene, 3.0f, 0.0f), 0.1f * lit)
            << "behind the blocker, cone " << cone;
        EXPECT_NEAR(RadianceAt(delegate, scene, 6.0f, 6.0f),
                    RadianceAt(delegate, LampScene(10.0, cone), 6.0f, 6.0f), 0.05f * lit)
            << "beside the shadow, cone " << cone;
        scene.Set("/World/Lamp", tokens::kShadowEnable, false);
        EXPECT_NEAR(RadianceAt(delegate, scene, 3.0f, 0.0f), lit, 0.02f * lit);
    }
    EXPECT_EQ(delegate.Renderer().LightShadowViews().size(), 0u) << "shadows off";
}

TEST_F(RasterDelegateTest, OtherDistantLightsAddToTheSun) {
    Scene scene = FlatScene();
    scene.Define("/World/Zenith", tokens::kDistantLight)
        .Set(tokens::kTransform, ernie::DistantLightTransform({0.0, 0.0, 1.0}))
        .Set(tokens::kIntensity, 0.5f);
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
    scene.Set("/World/Sun", tokens::kShadowEnable, true);
    scene.Set("/World/Sun", tokens::kTransform, ernie::DistantLightTransform({0.3, 0.0, 1.0}));
    DefineQuad(scene, "/World/Blocker", 4.0f);
    scene.Set("/World/Blocker", tokens::kTransform,
              glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 0.0, 30.0)));
    RasterDelegate delegate(*context_);
    const auto partial = [&](float angle) {
        scene.Set("/World/Sun", tokens::kAngle, angle);
        ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
        const Tensor hdr =
            outputs.at("/Render/Product").vars.at("HdrColor").Map(ernie::MapDevice::Cpu).tensor;
        const Tensor ids =
            outputs.at("/Render/Product").vars.at("PrimId").Map(ernie::MapDevice::Cpu).tensor;
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

/// FlatScene lit only by an area light of `type` 10 m above the origin facing down, with a
/// luminance of 1000 cd/m^2.
Scene AreaLightScene(std::string_view type) {
    Scene scene = FlatScene();
    scene.Remove("/World/Sun");
    scene.Define("/World/Panel", type)
        .Set(tokens::kTransform,
             ernie::LookAtTransform({0.0, 0.0, 10.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}))
        .Set(tokens::kIntensity, 1000.0)
        .Set(tokens::kShadowEnable, false);
    return scene;
}

/// Spectral radiance of a flat 1 cd/m^2 white emitter, per band.
double UnitLuminanceRadiance() {
    double sum = 0.0;
    for (double w = 360.0; w <= 830.0; w += 1.0) {
        sum += ernie::CieXyz(w).y;
    }
    return 1.0 / (ernie::kLuminousEfficacy * sum);
}

/// Form factor from a point to a parallel rectangle `a` x `b` whose corner lies `h` above it.
double CornerFormFactor(double a, double b, double h) {
    const double x = a / h;
    const double y = b / h;
    const double sx = std::sqrt(1.0 + x * x);
    const double sy = std::sqrt(1.0 + y * y);
    return (x / sx * std::atan(y / sx) + y / sy * std::atan(x / sy)) / (2.0 * M_PI);
}

}  // namespace

TEST_F(RasterDelegateTest, RectAndDiskLightsMatchTheirFormFactors) {
    RasterDelegate delegate(*context_);
    const double radiance = 1000.0 * UnitLuminanceRadiance();

    Scene rect = AreaLightScene(tokens::kRectLight);
    rect.Set("/World/Panel", tokens::kWidth, 8.0);
    rect.Set("/World/Panel", tokens::kHeight, 6.0);
    const double rectExpected =
        0.5 / M_PI * radiance * M_PI * 4.0 * CornerFormFactor(4.0, 3.0, 10.0);
    EXPECT_NEAR(RadianceAt(delegate, rect, 0.0f, 0.0f), rectExpected, 0.03 * rectExpected)
        << "albedo / pi times E = pi L F";

    Scene disk = AreaLightScene(tokens::kDiskLight);
    disk.Set("/World/Panel", tokens::kRadius, 5.0);
    const double diskExpected = 0.5 * radiance * 25.0 / (25.0 + 100.0);
    EXPECT_NEAR(RadianceAt(delegate, disk, 0.0f, 0.0f), diskExpected, 0.03 * diskExpected)
        << "E = pi L r^2 / (r^2 + h^2) on the axis";

    disk.Set("/World/Panel", tokens::kTransform,
             ernie::LookAtTransform({0.0, 0.0, 10.0}, {0.0, 0.0, 20.0}, {0.0, 1.0, 0.0}));
    EXPECT_EQ(RadianceAt(delegate, disk, 0.0f, 0.0f), 0.0f) << "facing away";
    disk.Set("/World/Panel", tokens::kTransform,
             ernie::LookAtTransform({0.0, 0.0, 0.5}, {10.0, 0.0, 0.5}, {0.0, 0.0, 1.0}));
    const float grazing = RadianceAt(delegate, disk, 3.0f, 0.0f);
    EXPECT_GT(grazing, 0.0f) << "half the disk is above the ground's horizon";
    EXPECT_LT(grazing, static_cast<float>(diskExpected));
}

TEST_F(RasterDelegateTest, AreaLightsCastShadows) {
    RasterDelegate delegate(*context_);
    Scene scene = AreaLightScene(tokens::kRectLight);
    scene.Set("/World/Panel", tokens::kShadowEnable, true);
    const float lit = RadianceAt(delegate, scene, 3.0f, 0.0f);
    DefineQuad(scene, "/World/Blocker");
    scene.Set("/World/Blocker", tokens::kTransform,
              glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 0.0, 5.0)));
    EXPECT_LT(RadianceAt(delegate, scene, 3.0f, 0.0f), 0.2f * lit) << "behind the blocker";
    EXPECT_EQ(delegate.Renderer().LightShadowViews().size(), 6u) << "a cube at the centre";
}

TEST_F(RasterDelegateTest, DomesLightEverySurfaceDiffusely) {
    Scene scene = FlatScene();
    scene.Remove("/World/Sun");
    scene.Define("/World/Sky", tokens::kDomeLight).Set(tokens::kIntensity, 1.0);
    RasterDelegate delegate(*context_);
    ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
    const auto vars = MapVars(outputs);
    const std::size_t c = kVarSize / 2;
    EXPECT_NEAR(At<float>(vars.at("HdrColor"), c, c, 1), 0.5f, 0.02f)
        << "a uniform 1 cd/m^2 dome gives albedo 0.5 a luminance of 0.5";
    EXPECT_FALSE(delegate.Renderer().Bands().count == 0);

    std::vector<float> map(8 * 16 * 3, 0.0f);
    std::fill(map.begin(), map.begin() + map.size() / 2, 1.0f);
    scene.Write("/World/Sky", tokens::kLightTexture, Tensor::Copy<float>(map, {8, 16, 3}));
    ernie::RenderProductSetOutputs upper = delegate.Step(scene);
    EXPECT_NEAR(At<float>(MapVars(upper).at("HdrColor"), c, c, 1), 0.5f, 0.03f)
        << "the upper hemisphere alone lights upward surfaces fully";
    scene.Set("/World/Sky", tokens::kTransform,
              glm::rotate(glm::dmat4(1.0), M_PI, glm::dvec3(1.0, 0.0, 0.0)));
    ernie::RenderProductSetOutputs lower = delegate.Step(scene);
    EXPECT_LT(At<float>(MapVars(lower).at("HdrColor"), c, c, 1), 0.03f) << "turned upside down";
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

namespace {

/// The camera transforms of a batch: around the terrain at `count` headings, 0.8 rad up.
std::vector<glm::dmat4> BatchPoses(std::size_t count) {
    std::vector<glm::dmat4> poses;
    for (std::size_t i = 0; i < count; ++i) {
        const double heading = 0.7 * static_cast<double>(i);
        const glm::dvec3 eye(60.0 * std::cos(heading), 60.0 * std::sin(heading), 50.0);
        poses.push_back(ernie::LookAtTransform(eye, {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}));
    }
    return poses;
}

/// A rolling terrain with a quad on it, lit by a sun without shadows and seen by noiseless
/// cameras: batches and single views then render the same images.
Scene BatchScene() {
    Scene scene = TerrainScene(RollingTerrain(33, 33, 2.0));
    scene.Set("/World/Sun", tokens::kShadowEnable, false);
    DefineQuad(scene, "/World/Quad", 6.0f);
    scene.Set("/World/Quad", tokens::kTransform,
              glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 0.0, 8.0)));
    scene.Define("/Render/Vars/Color", tokens::kRenderVar)
        .Set(tokens::kSourceName, std::string(tokens::kLdrColor));
    scene.Define("/Render/Vars/Distance", tokens::kRenderVar)
        .Set(tokens::kSourceName, std::string("DistanceToCamera"));
    scene.Define("/Render/Vars/Spectral", tokens::kRenderVar)
        .Set(tokens::kSourceName, std::string("SpectralRadiance"));
    return scene;
}

void DefineBatchCamera(Scene& scene, const Path& path, const glm::dmat4& pose) {
    scene.Define(path, tokens::kCamera)
        .Set(tokens::kTransform, pose)
        .Set(tokens::kSensorNoise, false);
}

void DefineProduct(Scene& scene, const Path& path, const ernie::Value& cameras,
                   std::uint32_t size = kSize) {
    scene.Define(path, tokens::kRenderProduct)
        .Set(tokens::kCameraRel, cameras)
        .Set(tokens::kResolution, glm::ivec2(size, size))
        .Set(tokens::kOrderedVars, std::vector<Path>{"/Render/Vars/Color", "/Render/Vars/Distance",
                                                     "/Render/Vars/Spectral"});
}

/// The bytes of view `view` of a batched CPU tensor.
std::vector<std::uint8_t> ViewBytes(const Tensor& batch, std::size_t view) {
    const auto bytes = static_cast<std::size_t>(batch.Elements() / batch.Shape()[0]) *
                       (batch.Type() == DType::UInt8 ? 1 : 4);
    const auto* data = static_cast<const std::uint8_t*>(batch.Data());
    return {data + view * bytes, data + (view + 1) * bytes};
}

std::vector<std::uint8_t> Bytes(const Tensor& image) {
    const auto bytes =
        static_cast<std::size_t>(image.Elements()) * (image.Type() == DType::UInt8 ? 1 : 4);
    const auto* data = static_cast<const std::uint8_t*>(image.Data());
    return {data, data + bytes};
}

}  // namespace

TEST_F(RasterDelegateTest, ABatchOfCamerasRendersAsThoseCamerasDoAlone) {
    Scene scene = BatchScene();
    const std::vector<glm::dmat4> poses = BatchPoses(3);
    std::vector<Path> cameras;
    for (std::size_t i = 0; i < poses.size(); ++i) {
        cameras.emplace_back("/World/Cam" + std::to_string(i));
        DefineBatchCamera(scene, cameras.back(), poses[i]);
        DefineProduct(scene, Path("/Render/Single" + std::to_string(i)), cameras.back());
    }
    DefineProduct(scene, "/Render/Batch", cameras);
    RasterDelegate delegate(*context_);
    const Path batchOnly[] = {"/Render/Batch"};
    ernie::RenderProductSetOutputs batch = delegate.Step(scene, batchOnly);
    ernie::RenderProductOutput& output = batch.at("/Render/Batch");
    EXPECT_EQ(output.vars.at("LdrColor").Views(), 3u);
    EXPECT_EQ(output.vars.at("LdrColor").Shape(), (std::vector<std::int64_t>{3, kSize, kSize, 4}));
    EXPECT_EQ(output.vars.at("DistanceToCamera").Shape(),
              (std::vector<std::int64_t>{3, kSize, kSize}));
    EXPECT_EQ(output.vars.at("SpectralRadiance").Shape(),
              (std::vector<std::int64_t>{3, 16, kSize, kSize}));
    std::map<std::string, Tensor> batched;
    for (const std::string name : {"LdrColor", "DistanceToCamera", "SpectralRadiance"}) {
        batched[name] = output.vars.at(name).Map(ernie::MapDevice::Cpu).tensor;
    }
    EXPECT_NE(ViewBytes(batched["LdrColor"], 0), ViewBytes(batched["LdrColor"], 1))
        << "the views see different things";

    for (std::size_t i = 0; i < poses.size(); ++i) {
        const Path single("/Render/Single" + std::to_string(i));
        const Path one[] = {single};
        ernie::RenderProductSetOutputs alone = delegate.Step(scene, one);
        for (const std::string name : {"LdrColor", "DistanceToCamera", "SpectralRadiance"}) {
            ernie::RenderVar& var = alone.at(single).vars.at(name);
            EXPECT_EQ(var.Views(), 1u);
            const Tensor image = var.Map(ernie::MapDevice::Cpu).tensor;
            EXPECT_EQ(ViewBytes(batched[name], i), Bytes(image)) << name << " of view " << i;
        }
    }
}

TEST_F(RasterDelegateTest, PosesRenderABatchThroughOneCamera) {
    Scene scene = BatchScene();
    const std::vector<glm::dmat4> poses = BatchPoses(4);
    // Math convention, row-major: element (row, col) of pose i at [i][row][col].
    std::vector<double> values;
    for (const glm::dmat4& pose : poses) {
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                values.push_back(pose[col][row]);
            }
        }
    }
    scene.Define("/World/Envs", tokens::kXform)
        .Set(tokens::kTransform, glm::translate(glm::dmat4(1.0), glm::dvec3(5.0, 0.0, 0.0)));
    DefineBatchCamera(scene, "/World/Envs/Cam", glm::dmat4(1.0));
    scene.Write("/World/Envs/Cam", tokens::kPoses,
                Tensor::Copy<double>(values, {static_cast<std::int64_t>(poses.size()), 4, 4}));
    DefineProduct(scene, "/Render/Batch", Path("/World/Envs/Cam"));
    DefineBatchCamera(scene, "/World/Check", glm::dmat4(1.0));
    DefineProduct(scene, "/Render/Check", Path("/World/Check"));

    RasterDelegate delegate(*context_);
    const Path batchOnly[] = {"/Render/Batch"};
    const Tensor batch = delegate.Step(scene, batchOnly)
                             .at("/Render/Batch")
                             .vars.at("LdrColor")
                             .Map(ernie::MapDevice::Cpu)
                             .tensor;
    ASSERT_EQ(batch.Shape(), (std::vector<std::int64_t>{4, kSize, kSize, 4}));
    const Path checkOnly[] = {"/Render/Check"};
    for (std::size_t i = 0; i < poses.size(); ++i) {
        // Poses are local to the camera's parent.
        scene.Set("/World/Check", tokens::kTransform,
                  glm::translate(glm::dmat4(1.0), glm::dvec3(5.0, 0.0, 0.0)) * poses[i]);
        const Tensor alone = delegate.Step(scene, checkOnly)
                                 .at("/Render/Check")
                                 .vars.at("LdrColor")
                                 .Map(ernie::MapDevice::Cpu)
                                 .tensor;
        EXPECT_EQ(ViewBytes(batch, i), Bytes(alone)) << "pose " << i;
    }
}

TEST_F(RasterDelegateTest, BatchesLargerThanAnImageArrayRenderInChunks) {
    Scene scene = BatchScene();
    const std::uint32_t views = ernie::RenderTargets::MaxViews(*context_, 16) + 2;
    std::vector<double> values;
    const glm::dmat4 pose = BatchPoses(1).front();
    for (std::uint32_t i = 0; i < views; ++i) {
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                values.push_back(pose[col][row]);
            }
        }
    }
    DefineBatchCamera(scene, "/World/Cam", glm::dmat4(1.0));
    scene.Write("/World/Cam", tokens::kPoses,
                Tensor::Copy<double>(values, {static_cast<std::int64_t>(views), 4, 4}));
    DefineProduct(scene, "/Render/Batch", Path("/World/Cam"), 4);
    RasterDelegate delegate(*context_);
    const Path batchOnly[] = {"/Render/Batch"};
    ernie::RenderProductSetOutputs outputs = delegate.Step(scene, batchOnly);
    const Tensor batch =
        outputs.at("/Render/Batch").vars.at("LdrColor").Map(ernie::MapDevice::Cpu).tensor;
    ASSERT_EQ(batch.Shape(), (std::vector<std::int64_t>{views, 4, 4, 4}));
    const std::vector<std::uint8_t> first = ViewBytes(batch, 0);
    EXPECT_NE(first, std::vector<std::uint8_t>(first.size(), 0));
    EXPECT_EQ(ViewBytes(batch, views - 1), first) << "the last chunk renders too";
    const Tensor spectral =
        outputs.at("/Render/Batch").vars.at("SpectralRadiance").Map(ernie::MapDevice::Cpu).tensor;
    EXPECT_EQ(spectral.Shape(), (std::vector<std::int64_t>{views, 16, 4, 4}));
}

TEST(ViewPosesOf, RejectsCamerasThatCannotShareABatch) {
    Scene scene;
    scene.Define("/A", tokens::kCamera);
    scene.Define("/B", tokens::kCamera).Set(tokens::kFocalLength, 35.0f);
    scene.Define("/C", tokens::kCamera);
    scene.Define("/NotACamera", tokens::kXform);
    const Path one[] = {"/A"};
    EXPECT_TRUE(ernie::ViewPosesOf(scene, one, 8, 8).empty()) << "a single view";
    const Path same[] = {"/A", "/C"};
    EXPECT_EQ(ernie::ViewPosesOf(scene, same, 8, 8).size(), 2u);
    const Path lenses[] = {"/A", "/B"};
    EXPECT_THROW(ernie::ViewPosesOf(scene, lenses, 8, 8), std::invalid_argument);
    const Path xform[] = {"/A", "/NotACamera"};
    EXPECT_THROW(ernie::ViewPosesOf(scene, xform, 8, 8), std::invalid_argument);
    const std::array<float, 12> flat{};
    scene.Write("/A", tokens::kPoses, Tensor::Copy<float>(flat, {3, 4}));
    EXPECT_THROW(ernie::ViewPosesOf(scene, one, 8, 8), std::invalid_argument);
}

TEST_F(RasterDelegateTest, SyncingAloneGivesTheMeshesBounds) {
    Scene scene = MeshScene();
    RasterDelegate delegate(*context_);
    delegate.Sync(scene);
    const std::optional<ernie::Bounds3> bounds = delegate.SceneBounds();
    ASSERT_TRUE(bounds.has_value()) << "viewers frame the scene right after syncing it";
    EXPECT_LT(bounds->first.x, -12.0f) << "the quad at x = -12";
    EXPECT_GT(bounds->second.x, 12.0f) << "the rocks at x = 12";
}

namespace {

/// Two worlds overlapping at the origin, seen from above by a camera whose two poses are the
/// same view of world 0 and of world 1: world 0 a small red quad, world 1 a large green one, each
/// once as a mesh and once as an instance of one instancer whose instances are in both worlds.
Scene WorldsScene(bool instanced) {
    Scene scene;
    const glm::dmat4 above =
        ernie::LookAtTransform({0.0, 0.0, 60.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0});
    scene.Define("/World/Camera", tokens::kCamera).Set(tokens::kSensorNoise, false);
    std::vector<double> poses;
    for (int view = 0; view < 2; ++view) {
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                poses.push_back(above[col][row]);
            }
        }
    }
    scene.Write("/World/Camera", tokens::kPoses, Tensor::Copy<double>(poses, {2, 4, 4}));
    scene.Write("/World/Camera", tokens::kWorlds,
                Tensor::Copy<std::int32_t>(std::vector<std::int32_t>{0, 1}, {2}));
    scene.Define("/World/Sun", tokens::kDistantLight)
        .Set(tokens::kTransform, ernie::DistantLightTransform({0.0, 0.0, 1.0}));
    scene.Define("/Looks/Red", tokens::kMaterial).Set(tokens::kAlbedo, glm::vec3(1.0f, 0.0f, 0.0f));
    scene.Define("/Looks/Green", tokens::kMaterial)
        .Set(tokens::kAlbedo, glm::vec3(0.0f, 1.0f, 0.0f));
    if (instanced) {
        // Instance 0 in world 1 (big, green), instance 1 in world 0 (small, red): unsorted.
        const std::array<float, 6> positions{0, 0, 0, 0, 0, 0};
        const std::array<float, 6> scales{3, 3, 1, 1, 1, 1};
        const std::array<std::int32_t, 2> protoIndices{1, 0};
        scene.Define("/World/Quads", tokens::kPointInstancer)
            .Write(tokens::kPositions, Tensor::Copy<float>(positions, {2, 3}))
            .Write(tokens::kScales, Tensor::Copy<float>(scales, {2, 3}))
            .Write(tokens::kProtoIndices, Tensor::Copy<std::int32_t>(protoIndices, {2}))
            .Write(tokens::kWorlds,
                   Tensor::Copy<std::int32_t>(std::vector<std::int32_t>{1, 0}, {2}))
            .Set(tokens::kPrototypes, std::vector<Path>{"/World/Quads/Red", "/World/Quads/Green"});
        DefineQuad(scene, "/World/Quads/Red", 2.0f);
        DefineQuad(scene, "/World/Quads/Green", 2.0f);
        scene.Set("/World/Quads/Red", tokens::kMaterialBinding, Path("/Looks/Red"));
        scene.Set("/World/Quads/Green", tokens::kMaterialBinding, Path("/Looks/Green"));
    } else {
        scene.Define("/Envs/Env0", tokens::kXform).Set(tokens::kWorld, 0);
        DefineQuad(scene, "/Envs/Env0/Quad", 2.0f);
        scene.Set("/Envs/Env0/Quad", tokens::kMaterialBinding, Path("/Looks/Red"));
        scene.Clone("/Envs/Env0", "/Envs/Env1");
        scene.Set("/Envs/Env1", tokens::kWorld, 1);
        DefineQuad(scene, "/Envs/Env1/Quad", 6.0f);
        scene.Set("/Envs/Env1/Quad", tokens::kMaterialBinding, Path("/Looks/Green"));
    }
    scene.Define("/Render/Batch", tokens::kRenderProduct)
        .Set(tokens::kCameraRel, Path("/World/Camera"))
        .Set(tokens::kResolution, glm::ivec2(kSize, kSize));
    scene.Define("/Render/Settings", tokens::kRenderSettings).Set(tokens::kAmbient, 0.0f);
    return scene;
}

}  // namespace

TEST_F(RasterDelegateTest, EachViewOfABatchSeesOnlyItsOwnWorld) {
    for (const bool instanced : {false, true}) {
        Scene scene = WorldsScene(instanced);
        RasterDelegate delegate(*context_);
        const Path batchOnly[] = {"/Render/Batch"};
        const Tensor images = delegate.Step(scene, batchOnly)
                                  .at("/Render/Batch")
                                  .vars.at("LdrColor")
                                  .Map(ernie::MapDevice::Cpu)
                                  .tensor;
        ASSERT_EQ(images.Shape(), (std::vector<std::int64_t>{2, kSize, kSize, 4}));
        const auto texel = [&](std::size_t view, std::size_t row, std::size_t col) {
            return images.As<std::uint8_t>() + ((view * kSize + row) * kSize + col) * 4;
        };
        const std::size_t centre = kSize / 2;
        const std::size_t edge = kSize / 2 + kSize / 8;  // inside only the big quad
        EXPECT_GT(texel(0, centre, centre)[0], 100)
            << "world 0's red quad, instanced " << instanced;
        EXPECT_LT(texel(0, centre, centre)[1], 50) << "not world 1's green one";
        EXPECT_EQ(texel(0, edge, edge)[1], 0) << "world 1 is not drawn in view 0";
        EXPECT_GT(texel(1, centre, centre)[1], 100) << "world 1's green quad";
        EXPECT_LT(texel(1, centre, centre)[0], 50) << "not world 0's red one";
        EXPECT_GT(texel(1, edge, edge)[1], 100) << "the green quad is the big one";
    }
}

TEST(Renderer, RendersAScenesProductsWithoutAnyGraphicsSetup) {
    std::unique_ptr<ernie::Renderer> renderer;
    try {
        renderer = std::make_unique<ernie::Renderer>(ernie::RendererOptions{.validation = true});
    } catch (const std::runtime_error& error) {
        GTEST_SKIP() << "No Vulkan device: " << error.what();
    }
    EXPECT_EQ(renderer->Info().name, "raster");
    EXPECT_FALSE(renderer->DeviceName().empty());
    Scene scene = MeshScene();
    const Tensor image = renderer->Render(scene);
    EXPECT_EQ(image.Shape(), (std::vector<std::int64_t>{kSize, kSize, 4}));
    EXPECT_TRUE(renderer->TakeValidationMessages().empty());
    EXPECT_THROW(ernie::Renderer({.backend = "nope"}), std::invalid_argument);
}

TEST_F(RasterDelegateTest, ProductsHoldOnlyTheVarsTheyAskFor) {
    Scene scene = BatchScene();
    DefineBatchCamera(scene, "/World/Cam", BatchPoses(1).front());
    DefineProduct(scene, "/Render/All", Path("/World/Cam"));
    scene.Define("/Render/DepthOnly", tokens::kRenderProduct)
        .Set(tokens::kCameraRel, Path("/World/Cam"))
        .Set(tokens::kResolution, glm::ivec2(kSize, kSize))
        .Set(tokens::kOrderedVars, std::vector<Path>{"/Render/Vars/Distance"});
    RasterDelegate everything(*context_);
    const Path all[] = {"/Render/All"};
    const Tensor full = everything.Step(scene, all)
                            .at("/Render/All")
                            .vars.at("DistanceToCamera")
                            .Map(ernie::MapDevice::Cpu)
                            .tensor;
    RasterDelegate depthAlone(*context_);
    const Path depthOnly[] = {"/Render/DepthOnly"};
    ernie::RenderProductSetOutputs outputs = depthAlone.Step(scene, depthOnly);
    const Tensor depth = outputs.at("/Render/DepthOnly")
                             .vars.at("DistanceToCamera")
                             .Map(ernie::MapDevice::Cpu)
                             .tensor;
    EXPECT_EQ(Bytes(depth), Bytes(full)) << "the same distances without the colour filters";
    EXPECT_EQ(outputs.at("/Render/DepthOnly").vars.count("LdrColor"), 0u);
    const std::uint64_t spectral = std::uint64_t{kSize} * kSize * 16 * sizeof(float);
    EXPECT_GT(everything.Stats().gpuBytes, depthAlone.Stats().gpuBytes + spectral)
        << "the depth-only frame holds no SpectralRadiance or colour vars";
}

TEST_F(RasterDelegateTest, EachWorldOfABatchCastsItsOwnShadows) {
    // The sun 45 degrees up towards +x; world 0's quad 10 m up at x = 10 shadows the shared
    // ground at the origin, the centre of every view.
    Scene scene;
    const glm::dmat4 above =
        ernie::LookAtTransform({0.0, 0.0, 60.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0});
    std::vector<double> poses;
    for (int view = 0; view < 3; ++view) {
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                poses.push_back(above[col][row]);
            }
        }
    }
    scene.Define("/World/Camera", tokens::kCamera).Set(tokens::kSensorNoise, false);
    scene.Write("/World/Camera", tokens::kPoses, Tensor::Copy<double>(poses, {3, 4, 4}));
    scene.Write("/World/Camera", tokens::kWorlds,
                Tensor::Copy<std::int32_t>(std::vector<std::int32_t>{0, 1, -1}, {3}));
    scene.Define("/World/Sun", tokens::kDistantLight)
        .Set(tokens::kTransform, ernie::DistantLightTransform({1.0, 0.0, 1.0}));
    DefineQuad(scene, "/World/Floor", 20.0f);  // shared, and bounded for the sun's map
    scene.Define("/Envs/Env0", tokens::kXform).Set(tokens::kWorld, 0);
    DefineQuad(scene, "/Envs/Env0/Roof", 3.0f);
    scene.Set("/Envs/Env0/Roof", tokens::kTransform,
              glm::translate(glm::dmat4(1.0), glm::dvec3(10.0, 0.0, 10.0)));
    scene.Define("/Envs/Env1", tokens::kXform).Set(tokens::kWorld, 1);
    scene.Define("/Render/Batch", tokens::kRenderProduct)
        .Set(tokens::kCameraRel, Path("/World/Camera"))
        .Set(tokens::kResolution, glm::ivec2(kSize, kSize));
    scene.Define("/Render/Settings", tokens::kRenderSettings).Set(tokens::kAmbient, 0.1f);

    RasterDelegate delegate(*context_);
    const Path batchOnly[] = {"/Render/Batch"};
    const Tensor images = delegate.Step(scene, batchOnly)
                              .at("/Render/Batch")
                              .vars.at("LdrColor")
                              .Map(ernie::MapDevice::Cpu)
                              .tensor;
    const auto centre = [&](std::size_t view) {
        const std::uint8_t* texel =
            images.As<std::uint8_t>() + ((view * kSize + kSize / 2) * kSize + kSize / 2) * 4;
        return texel[0] + texel[1] + texel[2];
    };
    EXPECT_LT(centre(0), centre(1) / 2) << "world 0's roof shades the ground in its view";
    EXPECT_LT(centre(2), centre(1) / 2) << "a view of every world sees every world's shadows";
    EXPECT_GT(centre(1), 100) << "world 1 has no roof, so its ground is lit";
}

namespace {

/// A batch of three identical views from above of a shared floor: of world 0, world 1 and every
/// world.
Scene ThreeViewsOfAFloor() {
    Scene scene;
    const glm::dmat4 above =
        ernie::LookAtTransform({0.0, 0.0, 60.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0});
    std::vector<double> poses;
    for (int view = 0; view < 3; ++view) {
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                poses.push_back(above[col][row]);
            }
        }
    }
    scene.Define("/World/Camera", tokens::kCamera).Set(tokens::kSensorNoise, false);
    scene.Write("/World/Camera", tokens::kPoses, Tensor::Copy<double>(poses, {3, 4, 4}));
    scene.Write("/World/Camera", tokens::kWorlds,
                Tensor::Copy<std::int32_t>(std::vector<std::int32_t>{0, 1, -1}, {3}));
    DefineQuad(scene, "/World/Floor", 20.0f);
    scene.Define("/Render/Batch", tokens::kRenderProduct)
        .Set(tokens::kCameraRel, Path("/World/Camera"))
        .Set(tokens::kResolution, glm::ivec2(kSize, kSize));
    return scene;
}

int CentreOf(const Tensor& images, std::size_t view) {
    const std::uint8_t* texel =
        images.As<std::uint8_t>() + ((view * kSize + kSize / 2) * kSize + kSize / 2) * 4;
    return texel[0] + texel[1] + texel[2];
}

}  // namespace

TEST_F(RasterDelegateTest, LightsShineOnlyInTheirWorld) {
    Scene scene = ThreeViewsOfAFloor();
    scene.Define("/Envs/Env0", tokens::kXform).Set(tokens::kWorld, 0);
    scene.Define("/Envs/Env0/Lamp", tokens::kSphereLight)
        .Set(tokens::kTransform, glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 0.0, 5.0)))
        .Set(tokens::kIntensity, 50.0f);
    RasterDelegate delegate(*context_);
    const Path batchOnly[] = {"/Render/Batch"};
    const Tensor images = delegate.Step(scene, batchOnly)
                              .at("/Render/Batch")
                              .vars.at("LdrColor")
                              .Map(ernie::MapDevice::Cpu)
                              .tensor;
    EXPECT_GT(CentreOf(images, 0), 100) << "world 0's lamp lights its view";
    EXPECT_EQ(CentreOf(images, 1), 0) << "world 1 has no light";
    EXPECT_EQ(CentreOf(images, 2), CentreOf(images, 0)) << "a view of every world sees it";
}

TEST_F(RasterDelegateTest, PosesWrittenAsCudaArraysRenderAsHostPoses) {
    RasterDelegate delegate(*context_);
    std::string why;
    ernie::CudaInterop* cuda = delegate.Cuda(&why);
    if (cuda == nullptr) {
        GTEST_SKIP() << "No CUDA interop: " << why;
    }
    Scene scene = ThreeViewsOfAFloor();
    scene.Define("/World/Sun", tokens::kDistantLight)
        .Set(tokens::kTransform, ernie::DistantLightTransform({0.3, 0.2, 1.0}));
    const Path batchOnly[] = {"/Render/Batch"};
    const auto render = [&] {
        return delegate.Step(scene, batchOnly)
            .at("/Render/Batch")
            .vars.at("LdrColor")
            .Map(ernie::MapDevice::Cpu)
            .tensor;
    };
    const Tensor host = render();
    const Tensor poses = *scene.Read("/World/Camera", tokens::kPoses);
    scene.Write("/World/Camera", tokens::kPoses, cuda->Upload(poses));
    EXPECT_EQ(Bytes(render()), Bytes(host));
}

TEST_F(RasterDelegateTest, CudaWritesOnSeveralStreamsSignalVulkanInOrder) {
    RasterDelegate delegate(*context_);
    std::string why;
    ernie::CudaInterop* cuda = delegate.Cuda(&why);
    if (cuda == nullptr) {
        GTEST_SKIP() << "No CUDA interop: " << why;
    }
    Scene scene = MeshScene();
    scene.Set("/World/Rocks", tokens::kExtent,
              std::vector<float>{-14.0f, -14.0f, -1.0f, 14.0f, 14.0f, 1.0f});
    const std::array<float, 9> start{12, -10, 0, 12, 0, 0, 12, 10, 0};
    const std::array<float, 12> quad{-2, -2, 0, 2, -2, 0, 2, 2, 0, -2, 2, 0};  // local, at x = -12
    constexpr std::uintptr_t kPerThread = 2;
    for (int step = 0; step < 3; ++step) {
        // Two arrays written on two streams in one step: their copies signal Vulkan in order.
        scene.Write("/World/Rocks", tokens::kPositions,
                    cuda->Upload(Tensor::Copy<float>(start, {3, 3})), 1);
        scene.Write("/World/Quad", tokens::kPoints, cuda->Upload(Tensor::Copy<float>(quad, {4, 3})),
                    kPerThread);
        const Tensor image = delegate.Render(scene);
        EXPECT_GT(TexelAt(scene, image, {12, -10, 0})[1], 200) << "step " << step;
        EXPECT_GT(TexelAt(scene, image, {-12, 0, 0})[0], 200) << "step " << step;
    }
}

TEST(CudaInterop, TensorsMayOutliveTheirDelegateAndDevice) {
    std::unique_ptr<ernie::VulkanContext> context;
    try {
        context = ernie::VulkanContext::Create();
    } catch (const std::runtime_error& error) {
        GTEST_SKIP() << "No Vulkan device: " << error.what();
    }
    auto delegate = std::make_unique<RasterDelegate>(*context);
    std::string why;
    ernie::CudaInterop* cuda = delegate->Cuda(&why);
    if (cuda == nullptr) {
        GTEST_SKIP() << "No CUDA interop: " << why;
    }
    Tensor uploaded = cuda->Upload(Tensor::Copy<float>(std::vector<float>{1, 2, 3}, {3}));
    delegate.reset();
    context.reset();
    uploaded = Tensor();  // frees its CUDA memory without touching the destroyed device
    SUCCEED();
}

TEST_F(RasterDelegateTest, MeshesOutsideAViewAreCulled) {
    Scene scene = MeshScene();
    RasterDelegate delegate(*context_);
    const Tensor before = delegate.Render(scene);
    const std::size_t drawn = delegate.Meshes().DrawCount(0);
    EXPECT_EQ(drawn, 2u) << "the quad and the rocks";

    // A quad above the camera, which looks down, and rocks far off to the side.
    DefineQuad(scene, "/World/Behind", 2.0f);
    scene.Set("/World/Behind", tokens::kTransform,
              glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 0.0, 100.0)));
    scene.Define("/World/FarRocks", tokens::kPointInstancer)
        .Write(tokens::kPositions,
               Tensor::Copy<float>(std::vector<float>{500, 0, 0, 600, 0, 0}, {2, 3}))
        .Set(tokens::kPrototypes, std::vector<Path>{"/World/Rocks/Rock"});
    const Tensor after = delegate.Render(scene);
    EXPECT_EQ(delegate.Meshes().DrawCount(0), drawn) << "neither new draw is in view";
    EXPECT_EQ(Bytes(after), Bytes(before));
    bool counted = false;
    for (const auto& [name, value] : delegate.Stats().counters) {
        if (name == ernie::raster_stats::kMeshDraws) {
            EXPECT_EQ(value, drawn);
            counted = true;
        }
    }
    EXPECT_TRUE(counted);
}

namespace {

/// MeshScene plus a row of 41 instances along x every 5 m, most of them outside the view.
Scene CullingScene() {
    Scene scene = MeshScene();
    std::vector<float> positions;
    for (int i = -20; i <= 20; ++i) {
        positions.insert(positions.end(), {5.0f * static_cast<float>(i), -20.0f, 0.0f});
    }
    scene.Define("/World/Row", tokens::kPointInstancer)
        .Write(tokens::kPositions, Tensor::Copy<float>(positions, {41, 3}))
        .Set(tokens::kPrototypes, std::vector<Path>{"/World/Rocks/Rock"});
    return scene;
}

/// How many of the row's 41 instances lie in the view of MeshScene's camera, by projecting their
/// centres: the ones GPU culling must keep (the prototype is small next to their spacing).
std::uint32_t RowInstancesInView(const Scene& scene) {
    const glm::mat4 viewProjection =
        ernie::CameraOf(scene, "/World/Camera", kSize, kSize).RasterView().ViewProjection();
    std::uint32_t inside = 0;
    for (int i = -20; i <= 20; ++i) {
        const glm::vec4 clip =
            viewProjection * glm::vec4(5.0f * static_cast<float>(i), -20.0f, 0.0f, 1.0f);
        inside += std::abs(clip.x) <= clip.w && std::abs(clip.y) <= clip.w ? 1u : 0u;
    }
    return inside;
}

}  // namespace

TEST(CullingOf, ReadsTheSettingsToken) {
    Scene scene;
    scene.Define("/Render/Settings", tokens::kRenderSettings);
    EXPECT_EQ(ernie::CullingOf(scene, "/Render/Settings"), ernie::Culling::Gpu) << "the default";
    for (const auto& [token, mode] : {std::pair{tokens::kOff, ernie::Culling::Off},
                                      std::pair{tokens::kCpu, ernie::Culling::Cpu},
                                      std::pair{tokens::kGpu, ernie::Culling::Gpu}}) {
        scene.Set("/Render/Settings", tokens::kCulling, std::string(token));
        EXPECT_EQ(ernie::CullingOf(scene, "/Render/Settings"), mode);
    }
    scene.Set("/Render/Settings", tokens::kCulling, std::string("magic"));
    EXPECT_THROW(ernie::CullingOf(scene, "/Render/Settings"), std::invalid_argument);
}

TEST_F(RasterDelegateTest, EveryCullingModeRendersTheSameImage) {
    Scene scene = CullingScene();
    RasterDelegate delegate(*context_);
    std::map<std::string, std::vector<std::uint8_t>> images;
    std::map<std::string, std::size_t> draws;
    for (const std::string_view mode : {tokens::kOff, tokens::kCpu, tokens::kGpu}) {
        scene.Set("/Render/Settings", tokens::kCulling, std::string(mode));
        images[std::string(mode)] = Bytes(delegate.Render(scene));
        draws[std::string(mode)] = delegate.Meshes().DrawCount(0);
    }
    EXPECT_EQ(images["cpu"], images["off"]);
    EXPECT_EQ(images["gpu"], images["off"]);
    EXPECT_EQ(draws["off"], draws["cpu"]) << "every draw has some instance in view";
}

TEST_F(RasterDelegateTest, GpuCullingKeepsExactlyTheInstancesInView) {
    Scene scene = CullingScene();
    scene.Set("/World/Sun", tokens::kShadowEnable, false);  // only the view's jobs
    RasterDelegate delegate(*context_);
    delegate.Render(scene);
    // Jobs in order: the quad is no instancer; then the rocks (3 instances, all in view), then
    // the row.
    const std::vector<std::uint32_t> kept = delegate.Meshes().CulledInstanceCounts();
    ASSERT_EQ(kept.size(), 2u);
    EXPECT_EQ(kept[0], 3u);
    const std::uint32_t inView = RowInstancesInView(scene);
    EXPECT_GT(inView, 0u);
    EXPECT_LT(inView, 41u);
    EXPECT_EQ(kept[1], inView);
}

TEST_F(RasterDelegateTest, InstancesWrittenAsCudaArraysAreCulledOnTheGpu) {
    RasterDelegate delegate(*context_);
    std::string why;
    ernie::CudaInterop* cuda = delegate.Cuda(&why);
    if (cuda == nullptr) {
        GTEST_SKIP() << "No CUDA interop: " << why;
    }
    Scene scene = CullingScene();
    scene.Set("/World/Sun", tokens::kShadowEnable, false);
    // With an extent, the positions never come to the host: only the GPU sees each instance.
    scene.Set("/World/Row", tokens::kExtent,
              std::vector<float>{-101.0f, -21.0f, -1.0f, 101.0f, -19.0f, 1.0f});
    scene.Write("/World/Row", tokens::kPositions,
                cuda->Upload(*scene.Read("/World/Row", tokens::kPositions)));
    delegate.Render(scene);
    const std::vector<std::uint32_t> kept = delegate.Meshes().CulledInstanceCounts();
    ASSERT_EQ(kept.size(), 2u);
    EXPECT_EQ(kept[1], RowInstancesInView(scene));
}

namespace {

/// Spheres of radius 1 in a row receding from a camera at the origin looking along +y, every 4 m
/// from 6 m to 400 m, lit from above without shadows.
Scene RecedingSpheres() {
    Scene scene;
    scene.Define("/World/Camera", tokens::kCamera)
        .Set(tokens::kTransform,
             ernie::LookAtTransform({0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}))
        .Set(tokens::kSensorNoise, false);
    scene.Define("/World/Sun", tokens::kDistantLight)
        .Set(tokens::kTransform, ernie::DistantLightTransform({0.2, -0.5, 1.0}))
        .Set(tokens::kShadowEnable, false);
    std::vector<float> positions;
    for (float y = 6.0f; y <= 400.0f; y += 4.0f) {
        positions.insert(positions.end(), {0.0f, y, 0.0f});
    }
    scene.Define("/World/Balls", tokens::kPointInstancer)
        .Write(tokens::kPositions,
               Tensor::Copy<float>(positions, {static_cast<std::int64_t>(positions.size() / 3), 3}))
        .Set(tokens::kPrototypes, std::vector<Path>{"/World/Balls/Ball"});
    scene.Define("/World/Balls/Ball", tokens::kSphere).Set(tokens::kGeomRadius, 1.0);
    scene.Define("/Render/Settings", tokens::kRenderSettings)
        .Set(tokens::kCameraRel, Path("/World/Camera"))
        .Set(tokens::kResolution, glm::ivec2(64, 64));
    return scene;
}

}  // namespace

TEST_F(RasterDelegateTest, DistantInstancesDrawCoarserLevelsOfDetail) {
    Scene scene = RecedingSpheres();
    RasterDelegate delegate(*context_);
    const Tensor detailed = [&] {
        scene.Set("/Render/Settings", tokens::kLodMeshError, 0.0f);
        return delegate.Render(scene);
    }();
    std::vector<std::vector<std::uint32_t>> levels = delegate.Meshes().CulledLevelCounts();
    ASSERT_EQ(levels.size(), 1u);
    const std::uint32_t visible = levels[0][0];
    EXPECT_GT(visible, 10u);
    for (std::size_t l = 1; l < levels[0].size(); ++l) {
        EXPECT_EQ(levels[0][l], 0u) << "an error of 0 draws the full meshes";
    }

    scene.Set("/Render/Settings", tokens::kLodMeshError, 1.0f);
    const Tensor simplified = delegate.Render(scene);
    levels = delegate.Meshes().CulledLevelCounts();
    ASSERT_EQ(levels.size(), 1u);
    ASSERT_GT(levels[0].size(), 3u) << "the sphere has a chain of levels";
    std::uint32_t total = 0;
    for (const std::uint32_t count : levels[0]) {
        total += count;
    }
    EXPECT_EQ(total, visible) << "the same instances, at different levels";
    EXPECT_LT(levels[0][0], visible / 2) << "most spheres are far enough to simplify";
    EXPECT_GT(levels[0].back() + levels[0][levels[0].size() - 2], 0u)
        << "the farthest use the coarsest levels";

    // Within a pixel of error the image barely changes.
    const auto* a = detailed.As<std::uint8_t>();
    const auto* b = simplified.As<std::uint8_t>();
    double difference = 0.0;
    for (std::int64_t i = 0; i < detailed.Elements(); ++i) {
        difference += std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
    }
    EXPECT_LT(difference / static_cast<double>(detailed.Elements()), 2.0);
}

TEST(MeshDrawableLevels, SimplifyOnlyMeshesWorthIt) {
    std::unique_ptr<ernie::VulkanContext> context;
    try {
        context = ernie::VulkanContext::Create();
    } catch (const std::runtime_error& error) {
        GTEST_SKIP() << "No Vulkan device: " << error.what();
    }
    ernie::RasterRenderer renderer(*context);
    ernie::MeshDrawable meshes(renderer);
    const std::uint32_t sphere = meshes.AddMesh(ernie::SphereMesh(1.0f));
    const std::uint32_t quad = meshes.AddMesh(ernie::BoxMesh({2.0f, 2.0f, 2.0f}));
    EXPECT_GT(meshes.LevelCount(sphere), 3u);
    EXPECT_LE(meshes.LevelCount(sphere), ernie::MeshDrawable::kMaxLevels);
    EXPECT_EQ(meshes.LevelCount(quad), 1u) << "12 triangles: only the full mesh";
}

TEST_F(RasterDelegateTest, GpuCulledInstancersShareOneMultiDraw) {
    Scene scene = CullingScene();  // a quad, and two instancers of the same mesh and material
    RasterDelegate delegate(*context_);
    scene.Set("/Render/Settings", tokens::kCulling, std::string(tokens::kCpu));
    const Tensor direct = delegate.Render(scene);
    EXPECT_EQ(delegate.Meshes().DrawCalls(0), 3u) << "one call per draw";
    scene.Set("/Render/Settings", tokens::kCulling, std::string(tokens::kGpu));
    const Tensor batched = delegate.Render(scene);
    EXPECT_EQ(delegate.Meshes().DrawCalls(0), context_->SupportsMultiDrawIndirect() ? 2u : 3u)
        << "the quad, then both instancers in one multi-draw";
    EXPECT_EQ(Bytes(batched), Bytes(direct));
}

namespace {

/// A camera at the origin looking along +y at a wall 20 m away (a mesh, or with `instancedWall`
/// an instance of a cube), with spheres hidden behind it at 40 m and one in front at 10 m.
Scene WallScene(bool instancedWall) {
    Scene scene;
    scene.Define("/World/Camera", tokens::kCamera)
        .Set(tokens::kTransform,
             ernie::LookAtTransform({0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}))
        .Set(tokens::kSensorNoise, false);
    scene.Define("/World/Sun", tokens::kDistantLight)
        .Set(tokens::kTransform, ernie::DistantLightTransform({0.2, -1.0, 0.5}))
        .Set(tokens::kShadowEnable, false);
    if (instancedWall) {
        scene.Define("/World/Walls", tokens::kPointInstancer)
            .Write(tokens::kPositions, Tensor::Copy<float>(std::vector<float>{0, 20, 0}, {1, 3}))
            .Set(tokens::kPrototypes, std::vector<Path>{"/World/Walls/Wall"});
        scene.Define("/World/Walls/Wall", tokens::kCube).Set(tokens::kSize, 60.0);
        scene.Set("/World/Walls/Wall", tokens::kTransform,
                  glm::scale(glm::dmat4(1.0), glm::dvec3(1.0, 0.05, 1.0)));
    } else {
        DefineQuad(scene, "/World/Wall", 30.0f);
        scene.Set("/World/Wall", tokens::kTransform,
                  glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 20.0, 0.0)) *
                      glm::rotate(glm::dmat4(1.0), glm::radians(90.0), glm::dvec3(1.0, 0.0, 0.0)));
    }
    std::vector<float> positions{0, 10, 0};  // in front of the wall
    for (int i = -4; i <= 4; ++i) {
        positions.insert(positions.end(), {static_cast<float>(i), 40.0f, 0.0f});
    }
    scene.Define("/World/Balls", tokens::kPointInstancer)
        .Write(tokens::kPositions, Tensor::Copy<float>(positions, {10, 3}))
        .Set(tokens::kPrototypes, std::vector<Path>{"/World/Balls/Ball"});
    scene.Define("/World/Balls/Ball", tokens::kSphere).Set(tokens::kGeomRadius, 0.5);
    scene.Define("/Render/Settings", tokens::kRenderSettings)
        .Set(tokens::kCameraRel, Path("/World/Camera"))
        .Set(tokens::kResolution, glm::ivec2(64, 64));
    return scene;
}

}  // namespace

TEST(OcclusionOf, ReadsTheSettingsToken) {
    Scene scene;
    scene.Define("/Render/Settings", tokens::kRenderSettings);
    EXPECT_EQ(ernie::OcclusionOf(scene, "/Render/Settings"), ernie::Occlusion::Off);
    scene.Set("/Render/Settings", tokens::kOcclusion, std::string(tokens::kOccluders));
    EXPECT_EQ(ernie::OcclusionOf(scene, "/Render/Settings"), ernie::Occlusion::Occluders);
    scene.Set("/Render/Settings", tokens::kOcclusion, std::string(tokens::kTwoPass));
    EXPECT_EQ(ernie::OcclusionOf(scene, "/Render/Settings"), ernie::Occlusion::TwoPass);
    scene.Set("/Render/Settings", tokens::kOcclusion, std::string("sometimes"));
    EXPECT_THROW(ernie::OcclusionOf(scene, "/Render/Settings"), std::invalid_argument);
}

TEST_F(RasterDelegateTest, OccludersHideTheInstancesBehindThem) {
    Scene scene = WallScene(false);
    RasterDelegate delegate(*context_);
    const Tensor unculled = delegate.Render(scene);
    EXPECT_EQ(delegate.Meshes().CulledInstanceCounts(), std::vector<std::uint32_t>{10u})
        << "frustum culling keeps every ball";

    scene.Set("/Render/Settings", tokens::kOcclusion, std::string(tokens::kOccluders));
    const Tensor occluded = delegate.Render(scene);
    EXPECT_TRUE(delegate.Meshes().CullsOcclusion());
    EXPECT_EQ(delegate.Meshes().CulledInstanceCounts(), std::vector<std::uint32_t>{1u})
        << "only the ball in front of the wall";
    EXPECT_EQ(Bytes(occluded), Bytes(unculled)) << "hidden instances change nothing";
}

TEST_F(RasterDelegateTest, TwoPassOcclusionLetsInstancesHideEachOther) {
    Scene scene = WallScene(true);
    RasterDelegate delegate(*context_);
    const Tensor unculled = delegate.Render(scene);
    scene.Set("/Render/Settings", tokens::kOcclusion, std::string(tokens::kOccluders));
    delegate.Render(scene);
    // Jobs in draw order: the balls, then the wall.
    EXPECT_EQ(delegate.Meshes().CulledInstanceCounts(), (std::vector<std::uint32_t>{10u, 1u}))
        << "an instanced wall occludes nothing in a single pass";

    scene.Set("/Render/Settings", tokens::kOcclusion, std::string(tokens::kTwoPass));
    for (int frame = 0; frame < 3; ++frame) {
        const Tensor image = delegate.Render(scene);
        EXPECT_EQ(Bytes(image), Bytes(unculled)) << "frame " << frame;
    }
    // By now the wall, visible last frame, draws first and hides the balls behind it.
    EXPECT_EQ(delegate.Meshes().CulledInstanceCounts(), (std::vector<std::uint32_t>{1u, 1u}));
}

TEST_F(RasterDelegateTest, OcclusionCullsEveryChunkOfABatch) {
    Scene scene = WallScene(false);
    const std::uint32_t views = ernie::RenderTargets::MaxViews(*context_, 16) + 3;
    std::vector<double> poses;
    for (std::uint32_t v = 0; v < views; ++v) {
        // Views sidestep along x, all still facing the wall.
        const glm::dmat4 pose =
            ernie::LookAtTransform({0.01 * v, 0.0, 0.0}, {0.01 * v, 1.0, 0.0}, {0.0, 0.0, 1.0});
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                poses.push_back(pose[col][row]);
            }
        }
    }
    scene.Write("/World/Camera", tokens::kPoses,
                Tensor::Copy<double>(poses, {static_cast<std::int64_t>(views), 4, 4}));
    scene.Define("/Render/Batch", tokens::kRenderProduct)
        .Set(tokens::kCameraRel, Path("/World/Camera"))
        .Set(tokens::kResolution, glm::ivec2(16, 16));
    RasterDelegate delegate(*context_);
    const Path batchOnly[] = {"/Render/Batch"};
    const auto render = [&] {
        return delegate.Step(scene, batchOnly)
            .at("/Render/Batch")
            .vars.at("LdrColor")
            .Map(ernie::MapDevice::Cpu)
            .tensor;
    };
    const Tensor unculled = render();
    scene.Set("/Render/Settings", tokens::kOcclusion, std::string(tokens::kOccluders));
    const Tensor occluded = render();
    EXPECT_EQ(Bytes(occluded), Bytes(unculled));
    const std::vector<std::uint32_t> kept = delegate.Meshes().CulledInstanceCounts();
    ASSERT_EQ(kept.size(), views) << "one job per view";
    EXPECT_EQ(kept.front(), 1u);
    EXPECT_EQ(kept.back(), 1u) << "the last chunk's views cull against their own pyramids";
}

namespace {

/// Three views from above of a shared floor at the origin, of world 0, world 1 and every world,
/// with world 0's roof shading the floor's centre under a sun 45 degrees up towards +x. With
/// `farWorld`, world 1 has a box 2 km away that stretches the scene.
Scene RoofScene(bool farWorld) {
    Scene scene;
    const glm::dmat4 above =
        ernie::LookAtTransform({0.0, 0.0, 60.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0});
    std::vector<double> poses;
    for (int view = 0; view < 3; ++view) {
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                poses.push_back(above[col][row]);
            }
        }
    }
    scene.Define("/World/Camera", tokens::kCamera).Set(tokens::kSensorNoise, false);
    scene.Write("/World/Camera", tokens::kPoses, Tensor::Copy<double>(poses, {3, 4, 4}));
    scene.Write("/World/Camera", tokens::kWorlds,
                Tensor::Copy<std::int32_t>(std::vector<std::int32_t>{0, 1, -1}, {3}));
    scene.Define("/World/Sun", tokens::kDistantLight)
        .Set(tokens::kTransform, ernie::DistantLightTransform({1.0, 0.0, 1.0}));
    DefineQuad(scene, "/World/Floor", 20.0f);
    scene.Define("/Envs/Env0", tokens::kXform).Set(tokens::kWorld, 0);
    DefineQuad(scene, "/Envs/Env0/Roof", 3.0f);
    scene.Set("/Envs/Env0/Roof", tokens::kTransform,
              glm::translate(glm::dmat4(1.0), glm::dvec3(10.0, 0.0, 10.0)));
    scene.Define("/Envs/Env1", tokens::kXform).Set(tokens::kWorld, 1);
    if (farWorld) {
        DefineQuad(scene, "/Envs/Env1/Far", 3.0f);
        scene.Set("/Envs/Env1/Far", tokens::kTransform,
                  glm::translate(glm::dmat4(1.0), glm::dvec3(2000.0, 0.0, 0.0)));
    }
    scene.Define("/Render/Batch", tokens::kRenderProduct)
        .Set(tokens::kCameraRel, Path("/World/Camera"))
        .Set(tokens::kResolution, glm::ivec2(kSize, kSize));
    scene.Define("/Render/Settings", tokens::kRenderSettings).Set(tokens::kAmbient, 0.1f);
    return scene;
}

}  // namespace

TEST_F(RasterDelegateTest, WorldShadowOptionsChooseWhatCastsIntoEachWorld) {
    Scene scene = RoofScene(false);
    RasterDelegate delegate(*context_);
    const Path batchOnly[] = {"/Render/Batch"};
    const auto centres = [&] {
        const Tensor images = delegate.Step(scene, batchOnly)
                                  .at("/Render/Batch")
                                  .vars.at("LdrColor")
                                  .Map(ernie::MapDevice::Cpu)
                                  .tensor;
        std::array<int, 3> sums{};
        for (std::size_t view = 0; view < 3; ++view) {
            const std::uint8_t* texel =
                images.As<std::uint8_t>() + ((view * kSize + kSize / 2) * kSize + kSize / 2) * 4;
            sums[view] = texel[0] + texel[1] + texel[2];
        }
        return sums;
    };
    const std::array<int, 3> separate = centres();
    EXPECT_LT(separate[0], separate[1] / 2) << "separate: world 0's roof shades its view";

    scene.Set("/Render/Settings", tokens::kShadowWorlds, std::string(tokens::kShared));
    const std::array<int, 3> shared = centres();
    EXPECT_EQ(shared[0], shared[1]) << "shared: worlds' own geometry casts nothing";

    scene.Set("/Render/Settings", tokens::kShadowWorlds, std::string(tokens::kMerged));
    const std::array<int, 3> merged = centres();
    EXPECT_LT(merged[1], shared[1] / 2) << "merged: every world's casters shade every view";
}

TEST_F(RasterDelegateTest, WorldFittedShadowMapsKeepSmallShadowsInLargeScenes) {
    Scene scene = RoofScene(true);
    scene.Set("/Render/Settings", tokens::kShadowWorldResolution, 64);
    RasterDelegate delegate(*context_);
    const Path batchOnly[] = {"/Render/Batch"};
    const auto centre = [&] {
        const Tensor images = delegate.Step(scene, batchOnly)
                                  .at("/Render/Batch")
                                  .vars.at("LdrColor")
                                  .Map(ernie::MapDevice::Cpu)
                                  .tensor;
        const std::uint8_t* texel =
            images.As<std::uint8_t>() + ((kSize / 2) * kSize + kSize / 2) * 4;  // view 0
        return texel[0] + texel[1] + texel[2];
    };
    const int sceneFit = centre();
    scene.Set("/Render/Settings", tokens::kShadowWorldFit, std::string(tokens::kWorldFit));
    const int worldFit = centre();
    EXPECT_LT(worldFit, sceneFit / 2)
        << "a 64-texel map over 2 km misses the roof's shadow; one fitted to world 0 keeps it";
}

TEST(ShadowSettings, SizesWorldMapsToTheImageByDefault) {
    using ernie::ShadowSettings;
    EXPECT_EQ(ShadowSettings{}.worldResolution, 0u) << "automatic";
    EXPECT_EQ(ShadowSettings::AutoWorldResolution(64, 64), 256u);
    EXPECT_EQ(ShadowSettings::AutoWorldResolution(48, 100), 512u);
    EXPECT_EQ(ShadowSettings::AutoWorldResolution(256, 128), 1024u);
    EXPECT_EQ(ShadowSettings::AutoWorldResolution(4096, 2160), 2048u);
}

TEST(FrameParamsOf, RejectsUnknownWorldShadowOptions) {
    Scene scene = RoofScene(false);
    const ernie::RenderPrims prims = ernie::ResolveRenderPrims(scene, "/Render/Settings");
    scene.Set("/Render/Settings", tokens::kShadowWorlds, std::string("some"));
    EXPECT_THROW(ernie::FrameParamsOf(scene, prims, 8, 8), std::invalid_argument);
    scene.Set("/Render/Settings", tokens::kShadowWorlds, std::string(tokens::kShared));
    scene.Set("/Render/Settings", tokens::kShadowWorldFit, std::string("tight"));
    EXPECT_THROW(ernie::FrameParamsOf(scene, prims, 8, 8), std::invalid_argument);
}

namespace {

/// CullingScene seen by a batch of `views` noiseless views from above along the row, under a
/// moved parent, over a ground plane below the meshes; shadows on.
Scene PosedRowScene(std::size_t views) {
    Scene scene = CullingScene();
    scene.Define("/World/Ground", tokens::kGroundPlane).Set(tokens::kGroundHeight, -1.0);
    scene.Define("/World/Rig", tokens::kXform)
        .Set(tokens::kTransform, glm::translate(glm::dmat4(1.0), glm::dvec3(3.0, -4.0, 0.0)));
    // No transform of its own: estimates from it look away from everything.
    scene.Define("/World/Rig/Camera", tokens::kCamera)
        .Set(tokens::kFocalLength, 18.0f)
        .Set(tokens::kSensorNoise, false);
    std::vector<float> poses;
    for (std::size_t v = 0; v < views; ++v) {
        const double x = -36.0 + 24.0 * static_cast<double>(v);
        const glm::dmat4 pose =
            ernie::LookAtTransform({x, -16.0, 40.0}, {x + 4.0, -16.0, 0.0}, {0.0, 1.0, 0.0});
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                poses.push_back(static_cast<float>(pose[col][row]));
            }
        }
    }
    scene.Write("/World/Rig/Camera", tokens::kPoses,
                Tensor::Copy<float>(poses, {static_cast<std::int64_t>(views), 4, 4}));
    scene.Define("/Render/Posed", tokens::kRenderProduct)
        .Set(tokens::kCameraRel, Path("/World/Rig/Camera"))
        .Set(tokens::kResolution, glm::ivec2(kSize, kSize));
    return scene;
}

/// How many bytes of `a` and `b` differ by more than 8.
std::size_t BytesDiffering(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b) {
    std::size_t differing = 0;
    for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        differing += std::abs(int(a[i]) - int(b[i])) > 8 ? 1u : 0u;
    }
    return differing + (a.size() > b.size() ? a.size() - b.size() : b.size() - a.size());
}

}  // namespace

TEST(GpuPosesOf, KeepsContiguousFloat32CudaPosesOnTheGpu) {
    Scene scene = PosedRowScene(2);
    EXPECT_EQ(ernie::GpuPosesOf(scene, "/World/Rig/Camera"), nullptr) << "host poses";
    EXPECT_EQ(ernie::PosesParentOf(scene, "/World/Rig/Camera"),
              glm::translate(glm::dmat4(1.0), glm::dvec3(3.0, -4.0, 0.0)));
}

TEST_F(RasterDelegateTest, PosesOnTheGpuRenderAsPosesOnTheHost) {
    RasterDelegate delegate(*context_);
    std::string why;
    ernie::CudaInterop* cuda = delegate.Cuda(&why);
    if (cuda == nullptr) {
        GTEST_SKIP() << "No CUDA interop: " << why;
    }
    const Path posedOnly[] = {"/Render/Posed"};
    const auto render = [&](const Scene& scene) {
        return Bytes(delegate.Step(scene, posedOnly)
                         .at("/Render/Posed")
                         .vars.at("LdrColor")
                         .Map(ernie::MapDevice::Cpu)
                         .tensor);
    };
    for (const std::size_t views : {std::size_t{1}, std::size_t{4}}) {
        Scene scene = PosedRowScene(views);
        // A single host-posed view fits shadow cascades to its frustum; posed on the GPU, it
        // takes a batch's map fitted to the scene, so its shadows differ in resolution.
        scene.Set("/World/Sun", tokens::kShadowEnable, views > 1);
        const std::vector<std::uint8_t> host = render(scene);
        const Tensor poses = *scene.Read("/World/Rig/Camera", tokens::kPoses);
        scene.Write("/World/Rig/Camera", tokens::kPoses, cuda->Upload(poses));
        ASSERT_NE(ernie::GpuPosesOf(scene, "/World/Rig/Camera"), nullptr);
        // The first step estimates the poses from the camera's own transform, the second from
        // the first's; neither may change what the views see.
        for (int step = 0; step < 2; ++step) {
            const std::vector<std::uint8_t> gpu = render(scene);
            EXPECT_LE(BytesDiffering(gpu, host), host.size() / 200)
                << views << " views, step " << step;
        }
        // Downloading them renders the host's poses exactly.
        scene.Set("/World/Rig/Camera", tokens::kPosesDevice, std::string(tokens::kHost));
        EXPECT_EQ(ernie::GpuPosesOf(scene, "/World/Rig/Camera"), nullptr);
        EXPECT_EQ(render(scene), host) << views << " views";
        scene.Set("/World/Rig/Camera", tokens::kPosesDevice, std::string("magic"));
        EXPECT_THROW(render(scene), std::invalid_argument);
    }
}

TEST_F(RasterDelegateTest, GpuPosedViewsCullInstancesAgainstTheirOwnFrusta) {
    RasterDelegate delegate(*context_);
    std::string why;
    ernie::CudaInterop* cuda = delegate.Cuda(&why);
    if (cuda == nullptr) {
        GTEST_SKIP() << "No CUDA interop: " << why;
    }
    Scene scene = PosedRowScene(4);
    const Path posedOnly[] = {"/Render/Posed"};
    // Every view's and shadow view's instances kept, summed: the GPU poses test every range,
    // the host's only those in view, which keep none of their instances otherwise.
    const auto kept = [&] {
        delegate.Step(scene, posedOnly);
        std::uint32_t sum = 0;
        for (const std::uint32_t count : delegate.Meshes().CulledInstanceCounts()) {
            sum += count;
        }
        return sum;
    };
    const std::uint32_t host = kept();
    ASSERT_GT(host, 0u);
    ASSERT_LT(host, 41u * 4u) << "the views see only part of the row";
    const Tensor poses = *scene.Read("/World/Rig/Camera", tokens::kPoses);
    scene.Write("/World/Rig/Camera", tokens::kPoses, cuda->Upload(poses));
    EXPECT_EQ(kept(), host) << "the GPU poses' frusta keep exactly the host poses' instances";
}

namespace {

/// LdrColor of every step of `scene` after `change(scene, step)`, all submitted before any is
/// fetched (`pipelined`), or each fetched before the next.
std::vector<std::vector<std::uint8_t>> Steps(RasterDelegate& delegate, Scene& scene,
                                             const std::function<void(Scene&, int)>& change,
                                             int steps, bool pipelined) {
    const Path product[] = {"/Render/Posed"};
    std::vector<ernie::StepOperation> operations;
    std::vector<std::vector<std::uint8_t>> images;
    const auto fetch = [&](ernie::StepOperation& operation) {
        images.push_back(Bytes(operation.Fetch()
                                   .at("/Render/Posed")
                                   .vars.at("LdrColor")
                                   .Map(ernie::MapDevice::Cpu)
                                   .tensor));
    };
    for (int step = 0; step < steps; ++step) {
        change(scene, step);
        operations.push_back(delegate.StepAsync(scene, product));
        if (!pipelined) {
            fetch(operations.back());
        }
    }
    if (pipelined) {
        for (ernie::StepOperation& operation : operations) {
            fetch(operation);
        }
    }
    return images;
}

}  // namespace

namespace {

constexpr std::size_t kFourViews = 4;

/// PosedRowScene with a few more views and pixels.
Scene PosedRowSceneOfFour() {
    Scene scene = PosedRowScene(kFourViews);
    scene.Set("/Render/Posed", tokens::kResolution, glm::ivec2(96, 96));
    return scene;
}

}  // namespace

TEST_F(RasterDelegateTest, StepsInFlightRenderAsStepsOneAtATime) {
    // Every step moves the row and the views; later steps are prepared while earlier ones still
    // render, so their host data must not reach the frames in flight.
    const auto change = [](Scene& scene, int step) {
        std::vector<float> positions;
        for (int i = -20; i <= 20; ++i) {
            positions.insert(positions.end(),
                             {5.0f * static_cast<float>(i) + 1.5f * static_cast<float>(step),
                              -20.0f + 2.0f * static_cast<float>(step % 3), 0.0f});
        }
        scene.Write("/World/Row", tokens::kPositions, Tensor::Copy<float>(positions, {41, 3}));
        scene.Set("/World/Rig", tokens::kTransform,
                  glm::translate(glm::dmat4(1.0), glm::dvec3(2.0 * step, -4.0, 0.0)));
    };
    constexpr int kSteps = 6;
    Scene oneAtATime = PosedRowSceneOfFour();
    RasterDelegate reference(*context_);
    const auto expected = Steps(reference, oneAtATime, change, kSteps, false);
    ASSERT_NE(expected.front(), expected.back()) << "the steps differ";

    RasterDelegate delegate(*context_);
    Scene pipelined = PosedRowSceneOfFour();
    const auto images = Steps(delegate, pipelined, change, kSteps, true);
    for (int step = 0; step < kSteps; ++step) {
        EXPECT_EQ(images[step], expected[step]) << "step " << step;
    }

    // Poses copied from CUDA each step, in flight too.
    std::string why;
    ernie::CudaInterop* cuda = delegate.Cuda(&why);
    if (cuda == nullptr) {
        GTEST_SKIP() << "No CUDA interop: " << why;
    }
    // Every step's poses on the GPU up front: uploading or reading them back during the steps
    // would wait for the work queued before. Each step's views turn a little further.
    const Tensor firstPoses = *PosedRowSceneOfFour().Read("/World/Rig/Camera", tokens::kPoses);
    std::vector<Tensor> posesOfStep;
    for (int step = 0; step < kSteps; ++step) {
        std::vector<float> turned(firstPoses.As<float>(),
                                  firstPoses.As<float>() + firstPoses.Elements());
        for (std::size_t v = 0; v < kFourViews; ++v) {
            turned[v * 16 + 3] += 0.5f * static_cast<float>(step);
        }
        posesOfStep.push_back(cuda->Upload(
            Tensor::Copy<float>(turned, {static_cast<std::int64_t>(kFourViews), 4, 4})));
    }
    const auto onGpu = [&](Scene& scene, int step) {
        change(scene, step);
        scene.Write("/World/Rig/Camera", tokens::kPoses,
                    posesOfStep[static_cast<std::size_t>(step)]);
    };
    Scene gpuOneAtATime = PosedRowSceneOfFour();
    Scene gpuPipelined = PosedRowSceneOfFour();
    const auto gpuExpected = Steps(reference, gpuOneAtATime, onGpu, kSteps, false);
    // Long CUDA work queued before the first step's poses holds every frame back on the GPU, as
    // a simulator's would, while the host prepares the later steps as far ahead as it may.
    const std::vector<float> zeros(std::size_t{1} << 26);  // 256 MiB
    const Tensor busy = cuda->Upload(Tensor::Copy<float>(zeros, {1 << 26}));
    const Tensor busyCopy = cuda->Upload(Tensor::Copy<float>(zeros, {1 << 26}));
    const auto stalled = [&](Scene& scene, int step) {
        if (step == 0) {
            for (int i = 0; i < 30; ++i) {
                cuda->Copy(busy, busyCopy);
            }
        }
        onGpu(scene, step);
    };
    const auto gpuImages = Steps(delegate, gpuPipelined, stalled, kSteps, true);
    for (int step = 0; step < kSteps; ++step) {
        EXPECT_EQ(gpuImages[step], gpuExpected[step]) << "GPU poses, step " << step;
    }
}
