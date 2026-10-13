// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/offscreen_target.h"
#include "ernie/device/pipeline.h"
#include "ernie/device/shader_compiler.h"
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
#include "support/delegate_scenes.h"
#include "support/test_raster.h"
#include "support/vulkan_fixture.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <numbers>
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

/// A camera framing the delegate's terrain from 0.8 rad above the
/// horizon.
ernie::CameraParams FramingCamera(const RasterDelegate& delegate) {
    OrbitController orbit;
    orbit.pitch = 0.8f;
    const auto [low, high] = delegate.Terrain().Bounds(1.0f);
    orbit.Frame(low, high);
    return orbit.MakeCamera(kSize, kSize);
}

}  // namespace

TEST_F(RasterDelegateTest,
       RendersTheSameImageAsTheRendererDrivenDirectly) {
    const TestRaster raster = RollingTerrain(65, 65, 2.0);
    Scene scene = TerrainScene(raster);
    RasterDelegate delegate(*context_);
    delegate.Sync(scene);
    ASSERT_TRUE(delegate.Terrain().HasHeightfield());
    const ernie::CameraParams camera = FramingCamera(delegate);
    ernie::SetCamera(scene, "/World/Camera", camera);

    const Tensor image = delegate.Render(scene);
    ASSERT_EQ(image.Shape(),
              (std::vector<std::int64_t>{kSize, kSize, 4}));
    EXPECT_EQ(delegate.Prims().terrain, Path("/World/Terrain"));

    ernie::RasterRenderer direct(*context_, VK_FORMAT_R8G8B8A8_UNORM);
    ernie::TerrainDrawable terrainDrawable(direct);
    terrainDrawable.SetHeightfield(raster);
    const std::uint32_t material =
        direct.AddMaterial(ernie::MaterialSettingsOf(
            scene, delegate.Prims().material, direct.Brdfs()));
    ernie::TerrainParams terrainParams =
        ernie::TerrainParamsOf(scene, delegate.Prims());
    terrainParams.material = material;
    terrainDrawable.SetParams(terrainParams);
    ernie::RasterDrawable* drawables[] = {&terrainDrawable};
    direct.Update(
        ernie::FrameParamsOf(scene, delegate.Prims(), kSize, kSize),
        drawables);
    ernie::OffscreenTarget target(*context_, kSize, kSize);
    ernie::RenderTargets targets(*context_, {kSize, kSize},
                                 direct.Bands().count, false);
    const ernie::RasterDrawable* drawn[] = {&terrainDrawable};
    context_->Submit([&](VkCommandBuffer commands) {
        const ernie::RenderingTarget present =
            target.Prepare(commands);
        direct.Record(commands, targets, drawn, &present);
    });
    const std::vector<std::uint8_t> expected = target.ReadColor();
    ASSERT_EQ(expected.size(),
              static_cast<std::size_t>(image.Elements()));
    EXPECT_EQ(
        std::memcmp(expected.data(), image.Data(), expected.size()),
        0);

    const auto* pixels = image.As<std::uint8_t>();
    EXPECT_EQ(pixels[2], 255) << "clear colour from ernie:clearColor";
    int terrain = 0;
    for (std::size_t i = 0; i < expected.size(); i += 4) {
        terrain += pixels[i + 2] != 255 ? 1 : 0;
    }
    EXPECT_GT(terrain, static_cast<int>(kSize * kSize / 4))
        << "the terrain covers the view";
}

TEST_F(RasterDelegateTest,
       AnUnlitSceneRendersBlackAndChosenLightsLightIt) {
    const TestRaster raster = RollingTerrain(33, 33, 2.0);
    Scene scene = TerrainScene(raster);
    ernie::RenderSettings(scene, "/Render/Settings")
        .SetClearColor(glm::dvec3(0.0));
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
        .Set(tokens::kTransform,
             ernie::DistantLightTransform({0.0, 0.0, 1.0}));
    scene.Set("/Render/Settings", tokens::kLightsRel,
              std::vector<Path>{});
    EXPECT_EQ(brightness(delegate.Render(scene)), 0)
        << "no chosen lights";
    scene.Set("/Render/Settings", tokens::kLightsRel,
              Path("/__session/Lights/Head"));
    EXPECT_GT(brightness(delegate.Render(scene)), 0)
        << "lit by the chosen session light";

    scene.Clear("/Render/Settings", tokens::kLightsRel);
    scene.Remove("/World/Sun");
    EXPECT_EQ(brightness(delegate.Render(scene)), 0)
        << "session lights are not stage lights";
}

TEST_F(RasterDelegateTest,
       WireframeModesDrawEdgesOverShadedOrFlatSurfaces) {
    const TestRaster raster = RollingTerrain(33, 33, 2.0);
    Scene scene = TerrainScene(raster);
    RasterDelegate delegate(*context_);
    delegate.Sync(scene);
    ernie::SetCamera(scene, "/World/Camera", FramingCamera(delegate));
    const auto render = [&](std::string_view mode) {
        ernie::RenderSettings(scene, "/Render/Settings")
            .SetWireframe(mode);
        const Tensor image = delegate.Render(scene);
        const auto* pixels = image.As<std::uint8_t>();
        return std::vector<std::uint8_t>(pixels,
                                         pixels + image.Elements());
    };
    const std::vector<std::uint8_t> shaded = render(tokens::kOff);
    const std::vector<std::uint8_t> overlay =
        render(tokens::kOverlay);
    const std::vector<std::uint8_t> hidden =
        render(tokens::kHiddenLine);
    std::size_t darker = 0;
    for (std::size_t i = 0; i < shaded.size(); ++i) {
        ASSERT_LE(overlay[i], shaded[i])
            << "edges only darken the shaded image";
        darker += overlay[i] < shaded[i] ? 1u : 0u;
    }
    EXPECT_GT(darker, 0u) << "the overlay draws edges";
    EXPECT_NE(hidden, shaded);

    scene.Set("/World/Sun", tokens::kTransform,
              ernie::DistantLightTransform({-1.0, 0.2, 0.9}));
    EXPECT_NE(render(tokens::kOff), shaded)
        << "shading follows the sun";
    EXPECT_EQ(render(tokens::kHiddenLine), hidden)
        << "hidden line ignores the sun's direction";
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
    EXPECT_EQ(std::memcmp(a.Data(), b.Data(),
                          static_cast<std::size_t>(a.Elements())),
              0);

    RasterDelegate wide(*context_, VK_FORMAT_R16G16B16A16_SFLOAT);
    const Tensor c = wide.Render(scene, "/Render/Settings");
    EXPECT_EQ(std::memcmp(a.Data(), c.Data(),
                          static_cast<std::size_t>(a.Elements())),
              0);
}

TEST_F(RasterDelegateTest, StepsMoveTheSensorClockOn) {
    const TestRaster raster = RollingTerrain(9, 9, 2.0);
    Scene scene = TerrainScene(raster);
    RasterDelegate delegate(*context_);
    delegate.Sync(scene);
    ernie::SetCamera(scene, "/World/Camera", FramingCamera(delegate));
    // Each step's frame spans the clock from the last frame's end.
    ernie::RenderProductSetOutputs outputs =
        delegate.Step(scene, {}, "/Render/Settings", 0.125);
    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_EQ(outputs.begin()->second.startTime, 0.0);
    EXPECT_EQ(outputs.begin()->second.time, 0.125);
    outputs = delegate.Step(scene, {}, "/Render/Settings", 0.25);
    EXPECT_EQ(outputs.begin()->second.startTime, 0.125);
    EXPECT_EQ(outputs.begin()->second.time, 0.375);
    EXPECT_EQ(delegate.Time(), 0.375);
    // A reset starts the next frame there; no delta, none spanned.
    delegate.ResetTime(7.0);
    outputs = delegate.Step(scene, {}, "/Render/Settings");
    EXPECT_EQ(outputs.begin()->second.startTime, 7.0);
    EXPECT_EQ(outputs.begin()->second.time, 7.0);
}

TEST_F(RasterDelegateTest, SyncsHeightfieldEditsRemovalsAndTextures) {
    Scene scene = TerrainScene(RollingTerrain(33, 17, 2.0));
    RasterDelegate delegate(*context_);
    delegate.Sync(scene);
    EXPECT_EQ(delegate.Terrain().Width(), 33u);
    EXPECT_EQ(delegate.Terrain().Height(), 17u);

    ernie::Terrain(scene, "/World/Terrain")
        .SetHeights(HeightsOf(RollingTerrain(9, 5, 2.0)));
    delegate.Sync(scene);
    EXPECT_EQ(delegate.Terrain().Width(), 9u);

    ernie::Material(scene, "/Looks/Rock")
        .SetDetailAlbedo(Tensor::Zeros(DType::UInt8, {8, 8, 3}))
        .SetDetailNormal(Tensor::Zeros(DType::Float32, {8, 8, 4}))
        .SetAlbedoMap(Tensor::Zeros(DType::Float32, {4, 4}));
    delegate.Sync(scene);
    ASSERT_TRUE(delegate.MaterialId("/Looks/Rock").has_value());
    const ernie::RasterMaterial& rock = delegate.Renderer().Material(
        *delegate.MaterialId("/Looks/Rock"));
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

    ernie::Terrain(scene, "/World/Terrain")
        .SetVisibility(tokens::kInvisible);
    delegate.Sync(scene);
    EXPECT_FALSE(delegate.Terrain().HasHeightfield());
    ernie::Terrain(scene, "/World/Terrain")
        .SetVisibility(tokens::kInherited);
    delegate.Sync(scene);
    EXPECT_TRUE(delegate.Terrain().HasHeightfield());
}

TEST_F(RasterDelegateTest,
       ReportsBadTensorsOnceWithThePrimAndAttribute) {
    Scene scene = TerrainScene(RollingTerrain(9, 9, 2.0));
    ernie::Terrain(scene, "/World/Terrain")
        .SetHeights(Tensor::Zeros(DType::Float64, {9, 9}));
    RasterDelegate delegate(*context_);
    try {
        delegate.Sync(scene);
        FAIL() << "expected an upload error";
    } catch (const std::invalid_argument& error) {
        EXPECT_NE(std::string(error.what())
                      .find("heights of '/World/Terrain'"),
                  std::string::npos)
            << error.what();
    }
    ernie::Terrain(scene, "/World/Terrain").SetExaggeration(2.0);
    EXPECT_NO_THROW(delegate.Sync(scene))
        << "not retried until the heights change";

    ernie::Material(scene, "/Looks/Rock")
        .SetDetailNormal(Tensor::Zeros(DType::Int32, {2, 2}));
    EXPECT_THROW(delegate.Sync(scene), std::invalid_argument);
    ernie::Material(scene, "/Looks/Rock")
        .SetAlbedoMap(Tensor::Zeros(DType::Float32, {4}));
    EXPECT_THROW(delegate.Sync(scene), std::invalid_argument);
}

TEST_F(RasterDelegateTest, SwitchingScenesResyncsEverything) {
    Scene first = TerrainScene(RollingTerrain(33, 17, 2.0));
    for (int i = 0; i < 20; ++i) {
        ernie::Terrain(first, "/World/Terrain").SetExaggeration(1.0);
    }
    RasterDelegate delegate(*context_);
    delegate.Sync(first);
    EXPECT_EQ(delegate.Terrain().Width(), 33u);

    Scene second = TerrainScene(RollingTerrain(9, 5, 2.0));
    ASSERT_LT(second.Version(), first.Version())
        << "an older cursor must not hide it";
    delegate.Sync(second);
    EXPECT_EQ(delegate.Terrain().Width(), 9u);

    Scene empty;
    delegate.Sync(empty);
    EXPECT_FALSE(delegate.Terrain().HasHeightfield());
    delegate.Sync(second);
    EXPECT_TRUE(delegate.Terrain().HasHeightfield());
}

TEST_F(RasterDelegateTest,
       DrawsMeshesAndInstancersWithTheirMaterials) {
    Scene scene = MeshScene();
    RasterDelegate delegate(*context_);
    Tensor image = delegate.Render(scene);
    EXPECT_EQ(delegate.Meshes().MeshCount(), 2u);
    EXPECT_EQ(delegate.Meshes().InstancerCount(), 1u);
    const std::uint8_t* quad = TexelAt(scene, image, {-12, 0, 0});
    EXPECT_GT(quad[0], 200);
    EXPECT_LT(quad[1] + quad[2], quad[0] / 2)
        << "red from its material";
    for (const float y : {-10.0f, 0.0f, 10.0f}) {
        const std::uint8_t* rock = TexelAt(scene, image, {12, y, 0});
        EXPECT_GT(rock[1], 200)
            << "green from displayColor, instance at y " << y;
        EXPECT_LT(rock[0] + rock[2], rock[1] / 2);
    }
    const std::uint8_t* prototype = TexelAt(scene, image, {0, 0, 0});
    EXPECT_EQ(prototype[0] + prototype[1] + prototype[2], 0)
        << "prototypes draw only as instances";
    EXPECT_TRUE(delegate.Renderer().LightShadowViews().size() > 0)
        << "meshes alone enable shadows";
}

TEST_F(RasterDelegateTest,
       MinimalShadingOfNoColourDrawsNoShadowMaps) {
    Scene scene = MeshScene();
    RasterDelegate delegate(*context_);
    scene.Set("/Render/Settings", tokens::kMinimalMode,
              std::string(tokens::kConstantDiffuse));
    delegate.Render(scene);
    ASSERT_FALSE(delegate.Renderer().LightShadowViews().empty())
        << "lit minimal shading has its sun's shadows";
    scene.Set("/Render/Settings", tokens::kMinimalMode,
              std::string(tokens::kNone));
    delegate.Render(scene);
    EXPECT_TRUE(delegate.Renderer().LightShadowViews().empty())
        << "nothing lit, nothing to shadow";
}

TEST_F(RasterDelegateTest, MovesInstancesAndDeformsMeshesInPlace) {
    Scene scene = MeshScene();
    RasterDelegate delegate(*context_);
    delegate.Render(scene);

    const std::array<float, 9> moved{0, -10, 0, 12, 0, 0, 12, 10, 0};
    ernie::PointInstancer(scene, "/World/Rocks")
        .SetPositions(Tensor::Copy<float>(moved, {3, 3}));
    const std::array<float, 12> points{-2, 6,  0, 2,  6,  0,
                                       2,  10, 0, -2, 10, 0};
    ernie::Mesh(scene, "/World/Quad")
        .SetPoints(Tensor::Copy<float>(points, {4, 3}));
    const Tensor image = delegate.Render(scene);
    EXPECT_EQ(delegate.Meshes().MeshCount(), 2u);
    EXPECT_GT(TexelAt(scene, image, {0, -10, 0})[1], 200)
        << "the instance moved";
    EXPECT_EQ(TexelAt(scene, image, {12, -10, 0})[1], 0);
    EXPECT_GT(TexelAt(scene, image, {-12, 8, 0})[0], 200)
        << "the mesh deformed";
    EXPECT_EQ(TexelAt(scene, image, {-12, 0, 0})[0], 0);

    scene.Remove("/World/Rocks");
    const Tensor removed = delegate.Render(scene);
    EXPECT_EQ(delegate.Meshes().InstancerCount(), 0u);
    EXPECT_EQ(TexelAt(scene, removed, {12, 0, 0})[1], 0);
    EXPECT_EQ(delegate.Meshes().MeshCount(), 1u)
        << "the prototype went with its instancer";
}

TEST_F(RasterDelegateTest, ABadMeshDoesNotBlockTheOthers) {
    Scene scene = MeshScene();
    DefineQuad(scene, "/World/Broken");
    const std::array<std::int64_t, 4> bad{0, 1, 2, 9};
    ernie::Mesh(scene, "/World/Broken")
        .SetFaceVertexIndices(Tensor::Copy<std::int64_t>(bad, {4}));
    RasterDelegate delegate(*context_);
    try {
        delegate.Sync(scene);
        FAIL() << "expected an upload error";
    } catch (const std::invalid_argument& error) {
        EXPECT_NE(
            std::string(error.what()).find("'/World/Broken' mesh"),
            std::string::npos)
            << error.what();
    }
    const Tensor image = delegate.Render(scene);
    EXPECT_GT(TexelAt(scene, image, {-12, 0, 0})[0], 200)
        << "the good quad still draws";
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
                 cuda->Upload(*device.Read("/World/Rocks",
                                           tokens::kPositions)));
    device.Set(
        "/World/Rocks", tokens::kExtent,
        std::vector<float>{-2.0f, -12.0f, -1.0f, 14.0f, 12.0f, 1.0f});
    device.Write(
        "/World/Quad", tokens::kPoints,
        cuda->Upload(*device.Read("/World/Quad", tokens::kPoints)));
    const Tensor image = delegate.Render(device);
    ASSERT_EQ(image.Elements(), expected.Elements());
    EXPECT_EQ(std::memcmp(image.Data(), expected.Data(),
                          static_cast<std::size_t>(image.Elements())),
              0)
        << "CUDA inputs draw the same image";

    const Tensor first = delegate.Render(
        device, {}, {ernie::DeviceKind::Cuda, cuda->Device()});
    EXPECT_EQ(first.Location().kind, ernie::DeviceKind::Cuda);
    EXPECT_EQ(first.Shape(), expected.Shape());
    const Tensor second = delegate.Render(
        device, {}, {ernie::DeviceKind::Cuda, cuda->Device()});
    EXPECT_NE(first.Data(), second.Data())
        << "a live result is not overwritten";
    const Tensor downloaded = cuda->Download(first);
    EXPECT_EQ(
        std::memcmp(downloaded.Data(), expected.Data(),
                    static_cast<std::size_t>(expected.Elements())),
        0)
        << "the CUDA result holds the render";

    const std::array<float, 9> moved{12, -10, 0, 0, 0, 0, 12, 10, 0};
    device.Write("/World/Rocks", tokens::kPositions,
                 cuda->Upload(Tensor::Copy<float>(moved, {3, 3})));
    const Tensor after = delegate.Render(device);
    EXPECT_GT(TexelAt(device, after, {0, 0, 0})[1], 200)
        << "a moved CUDA instance";

    EXPECT_THROW(delegate.Render(
                     device, {},
                     {ernie::DeviceKind::Cuda, cuda->Device() + 7}),
                 std::invalid_argument);
}

TEST_F(RasterDelegateTest, InstancesDrawInTheirOwnColours) {
    Scene scene = MeshScene();
    const std::array<float, 9> colors{1, 0, 0, 0, 1, 0, 0, 0, 1};
    ernie::PointInstancer(scene, "/World/Rocks")
        .SetDisplayColor(Tensor::Copy<float>(colors, {3, 3}));
    RasterDelegate delegate(*context_);
    const Tensor image = delegate.Render(scene);
    // The instances at y = -10, 0 and 10 in red, green and blue, in
    // place of the prototype's green.
    const auto expectColours = [&](const Tensor& drawn,
                                   const char* what) {
        for (int i = 0; i < 3; ++i) {
            const std::uint8_t* texel =
                TexelAt(scene, drawn, {12.0, -10.0 + 10.0 * i, 0.0});
            for (int c = 0; c < 3; ++c) {
                if (c == i) {
                    EXPECT_GT(texel[c], 200) << what << " " << i;
                } else {
                    EXPECT_LT(texel[c], 50) << what << " " << i;
                }
            }
        }
    };
    expectColours(image, "host colours");
    // The same instances keep their colours through GPU culling's
    // compacted ids.
    for (const std::string_view mode :
         {tokens::kOff, tokens::kCpu, tokens::kGpu}) {
        ernie::RenderSettings(scene, "/Render/Settings")
            .SetCulling(mode);
        expectColours(delegate.Render(scene), mode.data());
    }

    scene.Clear("/World/Rocks", tokens::kDisplayColor);
    const Tensor plain = delegate.Render(scene);
    EXPECT_GT(TexelAt(scene, plain, {12, -10, 0})[1], 200)
        << "without them, the prototype's green again";
    EXPECT_LT(TexelAt(scene, plain, {12, -10, 0})[0], 50);

    std::string why;
    ernie::ComputeInterop* cuda = delegate.Compute(&why);
    if (cuda == nullptr) {
        GTEST_SKIP() << "No CUDA interop: " << why;
    }
    // Colours alone on the GPU, positions on the host.
    scene.Write("/World/Rocks", tokens::kDisplayColor,
                cuda->Upload(Tensor::Copy<float>(colors, {3, 3})));
    const Tensor device = delegate.Render(scene);
    expectColours(device, "CUDA colours");
    EXPECT_EQ(std::memcmp(device.Data(), image.Data(),
                          static_cast<std::size_t>(image.Elements())),
              0)
        << "CUDA colours draw as host ones";

    // Written in place: blue, red and green.
    const Tensor mapped = delegate.MapArray(scene, "/World/Rocks",
                                            tokens::kDisplayColor);
    EXPECT_EQ(mapped.Shape(), (std::vector<std::int64_t>{3, 3}));
    const std::array<float, 9> rotated{0, 0, 1, 1, 0, 0, 0, 1, 0};
    cuda->Copy(cuda->Upload(Tensor::Copy<float>(rotated, {3, 3})),
               mapped);
    scene.Unmap("/World/Rocks", tokens::kDisplayColor);
    const Tensor moved = delegate.Render(scene);
    EXPECT_GT(TexelAt(scene, moved, {12, -10, 0})[2], 200);
    EXPECT_GT(TexelAt(scene, moved, {12, 0, 0})[0], 200);
    EXPECT_GT(TexelAt(scene, moved, {12, 10, 0})[1], 200);
}

TEST_F(RasterDelegateTest, CudaTrianglesDrawAndChangeInPlace) {
    RasterDelegate delegate(*context_);
    std::string why;
    ernie::ComputeInterop* cuda = delegate.Compute(&why);
    if (cuda == nullptr) {
        GTEST_SKIP() << "No CUDA interop: " << why;
    }
    Scene host = MeshScene();
    RasterDelegate reference(*context_);
    const Tensor expected = reference.Render(host);

    // The quad's fan as triangles on the GPU, without counts.
    Scene scene = MeshScene();
    scene.Clear("/World/Quad", tokens::kFaceVertexCounts);
    const std::array<std::int32_t, 6> fan{0, 1, 2, 0, 2, 3};
    scene.Write("/World/Quad", tokens::kFaceVertexIndices,
                cuda->Upload(Tensor::Copy<std::int32_t>(fan, {6})));
    const Tensor image = delegate.Render(scene);
    ASSERT_EQ(image.Elements(), expected.Elements());
    EXPECT_EQ(std::memcmp(image.Data(), expected.Data(),
                          static_cast<std::size_t>(image.Elements())),
              0)
        << "CUDA triangles draw as the host's";

    // The second triangle over the first: the upper left half goes.
    const std::size_t meshes = delegate.Meshes().MeshCount();
    const std::array<std::int32_t, 6> half{0, 1, 2, 0, 1, 2};
    scene.Write("/World/Quad", tokens::kFaceVertexIndices,
                cuda->Upload(Tensor::Copy<std::int32_t>(half, {6})));
    const Tensor halved = delegate.Render(scene);
    EXPECT_GT(TexelAt(scene, halved, {-11, -1, 0})[0], 200)
        << "the lower right half stays";
    EXPECT_EQ(TexelAt(scene, halved, {-13, 1, 0})[0], 0)
        << "the upper left half went";
    EXPECT_EQ(delegate.Meshes().MeshCount(), meshes);

    // Counts that are not all 3 cannot go with GPU indices.
    const std::array<std::int64_t, 1> quad{4};
    scene.Write("/World/Quad", tokens::kFaceVertexCounts,
                Tensor::Copy<std::int64_t>(quad, {1}));
    EXPECT_THROW(delegate.Sync(scene), std::invalid_argument);
}

TEST_F(RasterDelegateTest, ActiveCountsDrawTheTrianglesInUse) {
    // Drawn directly (CPU culling) and by the GPU's culling jobs:
    // either way only the triangles in use reach the vertex shader.
    for (const std::string& culling :
         {std::string(tokens::kCpu), std::string(tokens::kGpu)}) {
        SCOPED_TRACE(culling);
        RasterDelegate delegate(*context_);
        std::string why;
        ernie::ComputeInterop* cuda = delegate.Compute(&why);
        if (cuda == nullptr) {
            GTEST_SKIP() << "No CUDA interop: " << why;
        }
        Scene scene = MeshScene();
        scene.Set("/Render/Settings", tokens::kCulling, culling);
        scene.Clear("/World/Quad", tokens::kFaceVertexCounts);
        const std::array<std::int32_t, 6> fan{0, 1, 2, 0, 2, 3};
        scene.Write(
            "/World/Quad", tokens::kFaceVertexIndices,
            cuda->Upload(Tensor::Copy<std::int32_t>(fan, {6})));
        const auto expectHalves = [&](bool upper, const char* what) {
            const Tensor image = delegate.Render(scene);
            EXPECT_GT(TexelAt(scene, image, {-11, -1, 0})[0], 200)
                << what;
            if (upper) {
                EXPECT_GT(TexelAt(scene, image, {-13, 1, 0})[0], 200)
                    << what;
            } else {
                EXPECT_EQ(TexelAt(scene, image, {-13, 1, 0})[0], 0)
                    << what;
            }
        };
        expectHalves(true, "no counts");
        const std::size_t meshes = delegate.Meshes().MeshCount();

        // One triangle of two in use (3 indices of 6), from the host,
        // then from the GPU.
        const std::array<std::int32_t, 2> one{4, 3};
        const std::array<std::int32_t, 2> both{4, 6};
        scene.Write("/World/Quad", tokens::kActiveCounts,
                    Tensor::Copy<std::int32_t>(one, {2}));
        expectHalves(false, "host counts: the first triangle");
        scene.Write("/World/Quad", tokens::kActiveCounts,
                    Tensor::Copy<std::int32_t>(both, {2}));
        expectHalves(true, "host counts: both");
        scene.Write(
            "/World/Quad", tokens::kActiveCounts,
            cuda->Upload(Tensor::Copy<std::int32_t>(one, {2})));
        expectHalves(false, "CUDA counts: the first triangle");
        EXPECT_EQ(delegate.Meshes().MeshCount(), meshes)
            << "counts change in place";

        // As a simulator counts them: its overflow flag after the
        // counts, and only whole triangles of the indices drawn.
        const std::array<std::int32_t, 3> flagged{4, 5, 0};
        scene.Write(
            "/World/Quad", tokens::kActiveCounts,
            cuda->Upload(Tensor::Copy<std::int32_t>(flagged, {3})));
        expectHalves(false, "CUDA counts with a flag: one triangle");

        // Without them, every triangle again.
        scene.Clear("/World/Quad", tokens::kActiveCounts);
        expectHalves(true, "no counts");

        const std::array<std::int32_t, 1> wrong{4};
        scene.Write("/World/Quad", tokens::kActiveCounts,
                    Tensor::Copy<std::int32_t>(wrong, {1}));
        EXPECT_THROW(delegate.Sync(scene), std::invalid_argument);
    }
}

TEST_F(RasterDelegateTest, MappedTrianglesAreWrittenInPlace) {
    RasterDelegate delegate(*context_);
    std::string why;
    ernie::ComputeInterop* cuda = delegate.Compute(&why);
    if (cuda == nullptr) {
        GTEST_SKIP() << "No CUDA interop: " << why;
    }
    Scene scene = MeshScene();
    scene.Clear("/World/Quad", tokens::kFaceVertexCounts);
    const std::array<std::int32_t, 6> fan{0, 1, 2, 0, 2, 3};
    scene.Write("/World/Quad", tokens::kFaceVertexIndices,
                cuda->Upload(Tensor::Copy<std::int32_t>(fan, {6})));
    delegate.Render(scene);

    // The view starts as the triangles written.
    const Tensor mapped = delegate.MapArray(
        scene, "/World/Quad", tokens::kFaceVertexIndices);
    EXPECT_EQ(mapped.Shape(), (std::vector<std::int64_t>{6}));
    EXPECT_EQ(mapped.Type(), ernie::DType::Int32);
    const Tensor back = cuda->Download(mapped);
    EXPECT_TRUE(
        std::equal(fan.begin(), fan.end(), back.As<std::int32_t>()));

    const std::size_t meshes = delegate.Meshes().MeshCount();
    const std::array<std::int32_t, 6> half{0, 1, 2, 0, 1, 2};
    cuda->Copy(cuda->Upload(Tensor::Copy<std::int32_t>(half, {6})),
               mapped);
    scene.Unmap("/World/Quad", tokens::kFaceVertexIndices);
    const Tensor halved = delegate.Render(scene);
    EXPECT_GT(TexelAt(scene, halved, {-11, -1, 0})[0], 200);
    EXPECT_EQ(TexelAt(scene, halved, {-13, 1, 0})[0], 0)
        << "the triangles written in place";
    EXPECT_EQ(delegate.Meshes().MeshCount(), meshes);

    // Host triangles do not map.
    EXPECT_THROW(delegate.MapArray(scene, "/World/Rocks/Rock",
                                   tokens::kFaceVertexIndices),
                 std::invalid_argument);
}

TEST_F(RasterDelegateTest,
       CudaPointsWithoutBoundsAreBoundedOnTheGpu) {
    RasterDelegate delegate(*context_);
    std::string why;
    ernie::ComputeInterop* cuda = delegate.Compute(&why);
    if (cuda == nullptr) {
        GTEST_SKIP() << "No CUDA interop: " << why;
    }
    // A fifth point far above the quad, out of use.
    Scene scene = MeshScene();
    const Tensor* quad = scene.Read("/World/Quad", tokens::kPoints);
    std::vector<float> points(quad->As<float>(),
                              quad->As<float>() + quad->Elements());
    points.insert(points.end(), {0.0f, 0.0f, 500.0f});
    scene.Write("/World/Quad", tokens::kPoints,
                cuda->Upload(Tensor::Copy<float>(points, {5, 3})));
    const std::array<std::int32_t, 2> used{4, 6};
    scene.Write("/World/Quad", tokens::kActiveCounts,
                Tensor::Copy<std::int32_t>(used, {2}));
    delegate.Render(scene);
    ASSERT_TRUE(delegate.Meshes().Bounds());
    EXPECT_GT(delegate.Meshes().Bounds()->second.z, 400.0f)
        << "downloaded at first: every point";

    // Measured on the GPU frames later: the points in use alone.
    for (int frame = 0; frame < 4; ++frame) {
        scene.Write(
            "/World/Quad", tokens::kPoints,
            cuda->Upload(Tensor::Copy<float>(points, {5, 3})));
        delegate.Render(scene);
    }
    ASSERT_TRUE(delegate.Meshes().Bounds());
    EXPECT_LT(delegate.Meshes().Bounds()->second.z, 10.0f);
    const Tensor image = delegate.Render(scene);
    EXPECT_GT(TexelAt(scene, image, {-12, 0, 0})[0], 200)
        << "still drawn";
}

/// A compute shader writing MeshScene's quad, the fan's first
/// `triangles` triangles in use, through a mesh writer's pointers.
constexpr const char* kQuadWriterSlang = R"(
struct Push {
    float* points;
    uint* indices;
    int* counts;
    int triangles;
}
[[vk::push_constant]] ConstantBuffer<Push> push;

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
    let corners = float2[4](float2(-10, -10), float2(10, -10), float2(10, 10), float2(-10, 10));
    for (int i = 0; i < 4; ++i) {
        push.points[3 * i] = corners[i].x;
        push.points[3 * i + 1] = corners[i].y;
        push.points[3 * i + 2] = 0.0;
    }
    let fan = uint[6](0, 1, 2, 0, 2, 3);
    for (int i = 0; i < 6; ++i) {
        push.indices[i] = fan[i];
    }
    push.counts[0] = 4;
    push.counts[1] = 3 * push.triangles;
}
)";

TEST_F(RasterDelegateTest, MeshWritersDrawWhatTheyWrite) {
    const ernie::ShaderCompiler compiler;
    const std::array<ernie::ShaderModuleSource, 1> modules{
        {{"quad_writer", kQuadWriterSlang}}};
    const std::vector<std::uint32_t> spirv =
        compiler.CompileSpirv(modules, "main");
    const ernie::PipelineLayout layout(*context_, {}, 32);
    const ernie::ComputePipeline pipeline(*context_, layout, spirv);

    // The quad as placeholders: room for 8 points and 4 triangles.
    Scene scene = MeshScene();
    const std::vector<float> points(24, 0.0f);
    const std::vector<std::int32_t> counts(4, 3);
    const std::vector<std::int32_t> indices(12, 0);
    ernie::Mesh(scene, "/World/Quad")
        .SetPoints(Tensor::Copy<float>(points, {8, 3}))
        .SetFaceVertexCounts(Tensor::Copy<std::int32_t>(counts, {4}))
        .SetFaceVertexIndices(
            Tensor::Copy<std::int32_t>(indices, {12}));
    RasterDelegate delegate(*context_);
    int triangles = 2;
    ernie::MeshWriteTarget written;
    delegate.SetMeshWriter(
        "/World/Quad", [&](VkCommandBuffer commands,
                           const ernie::MeshWriteTarget& target) {
            written = target;
            vkCmdBindPipeline(commands,
                              VK_PIPELINE_BIND_POINT_COMPUTE,
                              pipeline.Handle());
            struct {
                VkDeviceAddress points, indices, counts;
                std::int32_t triangles, pad;
            } push{target.points, target.indices, target.counts,
                   triangles, 0};
            layout.PushConstants(commands, &push, sizeof(push));
            vkCmdDispatch(commands, 1, 1, 1);
        });
    const auto red = [&](glm::vec3 p) {
        return TexelAt(scene, delegate.Render(scene), p)[0];
    };
    EXPECT_GT(red({-11, -1, 0}), 200) << "the lower right half";
    EXPECT_GT(red({-13, 1, 0}), 200) << "the upper left half";
    EXPECT_EQ(written.pointCapacity, 8u);
    EXPECT_EQ(written.triangleCapacity, 4u);
    EXPECT_EQ(written.normals, 0u) << "the quad has no normals";

    triangles = 1;
    EXPECT_GT(red({-11, -1, 0}), 200);
    EXPECT_EQ(red({-13, 1, 0}), 0) << "one triangle in use";

    // The points written, measured on the GPU frames later.
    for (int frame = 0; frame < 3; ++frame) {
        delegate.Render(scene);
    }
    const std::optional<ernie::Bounds3> bounds =
        delegate.Meshes().Bounds();
    ASSERT_TRUE(bounds);
    EXPECT_NEAR(bounds->first.x, -22.0f, 0.01f) << "the quad's left";

    // Without the writer, the placeholders: nothing.
    delegate.SetMeshWriter("/World/Quad", nullptr);
    EXPECT_EQ(red({-11, -1, 0}), 0);
}

// A ridge of two slopes without normals, lit from the right: smooth
// normals, computed on the GPU from the triangles around each point,
// shade it alike just either side of the ridge, where flat ones would
// jump from the sunlit slope's to the other's; its feet still differ.
TEST_F(RasterDelegateTest, MeshesWithoutNormalsShadeSmooth) {
    Scene scene = MeshScene();
    scene.Set("/World/Sun", tokens::kTransform,
              ernie::DistantLightTransform({1.0, 0.0, 1.0}));
    const std::array<float, 18> points{-4, -4, 0,    0,  -4, 1.5f,
                                       0,  4,  1.5f, -4, 4,  0,
                                       4,  -4, 0,    4,  4,  0};
    const std::array<std::int32_t, 2> counts{4, 4};
    const std::array<std::int32_t, 8> indices{0, 1, 2, 3, 1, 4, 5, 2};
    ernie::Mesh::Define(scene, "/World/Ridge")
        .SetPoints(Tensor::Copy<float>(points, {6, 3}))
        .SetFaceVertexCounts(Tensor::Copy<std::int32_t>(counts, {2}))
        .SetFaceVertexIndices(
            Tensor::Copy<std::int32_t>(indices, {8}))
        .SetMaterialBinding("/Looks/Red");
    RasterDelegate delegate(*context_);
    const Tensor image = delegate.Render(scene);
    const auto red = [&](float x) {
        const float z = 1.5f * (1.0f - std::abs(x) / 4.0f);
        return static_cast<int>(TexelAt(scene, image, {x, 0, z})[0]);
    };
    EXPECT_NEAR(red(-0.3f), red(0.3f), 12) << "no edge at the ridge";
    EXPECT_GT(red(3.5f), red(-3.5f) + 30)
        << "the slopes still differ";
}

// The ridge's six points and four triangles, written on the GPU.
constexpr const char* kRidgeWriterSlang = R"(
struct Push {
    float* points;
    uint* indices;
    int* counts;
}
[[vk::push_constant]] ConstantBuffer<Push> push;

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
    let corners = float3[6](float3(-4, -4, 0), float3(0, -4, 1.5), float3(0, 4, 1.5),
                            float3(-4, 4, 0), float3(4, -4, 0), float3(4, 4, 0));
    for (int i = 0; i < 6; ++i) {
        push.points[3 * i] = corners[i].x;
        push.points[3 * i + 1] = corners[i].y;
        push.points[3 * i + 2] = corners[i].z;
    }
    let triangles = uint[12](0, 1, 2, 0, 2, 3, 1, 4, 5, 1, 5, 2);
    for (int i = 0; i < 12; ++i) {
        push.indices[i] = triangles[i];
    }
    push.counts[0] = 6;
    push.counts[1] = 12;
}
)";

// The ridge again, its triangles written on the GPU by a writer, so
// no list of the triangles around its points is known on the host:
// its normals summed on the GPU each frame still shade it smooth.
TEST_F(RasterDelegateTest, MeshesWrittenOnTheGpuShadeSmooth) {
    const ernie::ShaderCompiler compiler;
    const std::array<ernie::ShaderModuleSource, 1> modules{
        {{"ridge_writer", kRidgeWriterSlang}}};
    const ernie::PipelineLayout layout(*context_, {}, 32);
    const ernie::ComputePipeline pipeline(
        *context_, layout, compiler.CompileSpirv(modules, "main"));
    Scene scene = MeshScene();
    scene.Set("/World/Sun", tokens::kTransform,
              ernie::DistantLightTransform({1.0, 0.0, 1.0}));
    const std::vector<float> points(18, 0.0f);
    const std::vector<std::int32_t> counts(4, 3);
    const std::vector<std::int32_t> indices(12, 0);
    ernie::Mesh::Define(scene, "/World/Ridge")
        .SetPoints(Tensor::Copy<float>(points, {6, 3}))
        .SetFaceVertexCounts(Tensor::Copy<std::int32_t>(counts, {4}))
        .SetFaceVertexIndices(
            Tensor::Copy<std::int32_t>(indices, {12}))
        .SetMaterialBinding("/Looks/Red");
    RasterDelegate delegate(*context_);
    ernie::MeshWriteTarget written;
    delegate.SetMeshWriter(
        "/World/Ridge", [&](VkCommandBuffer commands,
                            const ernie::MeshWriteTarget& target) {
            written = target;
            vkCmdBindPipeline(commands,
                              VK_PIPELINE_BIND_POINT_COMPUTE,
                              pipeline.Handle());
            struct {
                VkDeviceAddress points, indices, counts;
            } push{target.points, target.indices, target.counts};
            layout.PushConstants(commands, &push, sizeof(push));
            vkCmdDispatch(commands, 1, 1, 1);
        });
    const auto red = [&](const Tensor& image, float x) {
        const float z = 1.5f * (1.0f - std::abs(x) / 4.0f);
        return static_cast<int>(TexelAt(scene, image, {x, 0, z})[0]);
    };
    const Tensor image = delegate.Render(scene);
    EXPECT_EQ(written.normals, 0u)
        << "its normals are the renderer's";
    EXPECT_NEAR(red(image, -0.3f), red(image, 0.3f), 12)
        << "no edge at the ridge";
    EXPECT_GT(red(image, 3.5f), red(image, -3.5f) + 30)
        << "the slopes still differ";
}

// The ridge with a normal per face (uniform): a hard edge, each slope
// as flat as its foot; then the same as an indexed face-varying
// primvar, which takes the place of `normals`.
TEST_F(RasterDelegateTest,
       UniformAndFaceVaryingNormalsMakeHardEdges) {
    Scene scene = MeshScene();
    scene.Set("/World/Sun", tokens::kTransform,
              ernie::DistantLightTransform({1.0, 0.0, 1.0}));
    const std::array<float, 18> points{-4, -4, 0,    0,  -4, 1.5f,
                                       0,  4,  1.5f, -4, 4,  0,
                                       4,  -4, 0,    4,  4,  0};
    const std::array<std::int32_t, 2> counts{4, 4};
    const std::array<std::int32_t, 8> indices{0, 1, 2, 3, 1, 4, 5, 2};
    const float slope = 1.5f / std::hypot(1.5f, 4.0f);
    const float rise = 4.0f / std::hypot(1.5f, 4.0f);
    const std::array<float, 6> faces{-slope, 0, rise, slope, 0, rise};
    ernie::Mesh::Define(scene, "/World/Ridge")
        .SetPoints(Tensor::Copy<float>(points, {6, 3}))
        .SetNormals(Tensor::Copy<float>(faces, {2, 3}))
        .SetFaceVertexCounts(Tensor::Copy<std::int32_t>(counts, {2}))
        .SetFaceVertexIndices(
            Tensor::Copy<std::int32_t>(indices, {8}))
        .SetMaterialBinding("/Looks/Red");
    RasterDelegate delegate(*context_);
    const auto red = [&](const Tensor& image, float x) {
        const float z = 1.5f * (1.0f - std::abs(x) / 4.0f);
        return static_cast<int>(TexelAt(scene, image, {x, 0, z})[0]);
    };
    const Tensor uniform = delegate.Render(scene);
    EXPECT_GT(red(uniform, 0.3f), red(uniform, -0.3f) + 30)
        << "a hard edge";
    EXPECT_NEAR(red(uniform, -0.3f), red(uniform, -3.5f), 4);
    EXPECT_NEAR(red(uniform, 0.3f), red(uniform, 3.5f), 4);

    scene.Clear("/World/Ridge", tokens::kNormals);
    const std::string primvar = "primvars:normals";
    const std::array<float, 6> swapped{slope,  0, rise,
                                       -slope, 0, rise};
    const std::array<std::int32_t, 8> which{1, 1, 1, 1, 0, 0, 0, 0};
    scene.Write("/World/Ridge", primvar,
                Tensor::Copy<float>(swapped, {2, 3}));
    scene.Write("/World/Ridge", primvar + ":indices",
                Tensor::Copy<std::int32_t>(which, {8}));
    scene.Set("/World/Ridge", primvar + ":interpolation",
              std::string("faceVarying"));
    const Tensor faceVarying = delegate.Render(scene);
    for (const float x : {-3.5f, -0.3f, 0.3f, 3.5f}) {
        EXPECT_NEAR(red(faceVarying, x), red(uniform, x), 3)
            << "face-varying as uniform at " << x;
    }
    // One normal for the mesh (constant), straight up: both slopes
    // lit alike, as no triangle's own normal would have them.
    scene.Clear("/World/Ridge", primvar + ":indices");
    scene.Clear("/World/Ridge", primvar + ":interpolation");
    const std::array<float, 3> up{0, 0, 1};
    scene.Write("/World/Ridge", primvar,
                Tensor::Copy<float>(up, {1, 3}));
    const Tensor constant = delegate.Render(scene);
    EXPECT_NEAR(red(constant, -3.5f), red(constant, 3.5f), 3);
}

TEST_F(RasterDelegateTest,
       CudaPointsOfDetailedPrototypesDrawAsHostOnes) {
    RasterDelegate delegate(*context_);
    std::string why;
    ernie::ComputeInterop* cuda = delegate.Compute(&why);
    if (cuda == nullptr) {
        GTEST_SKIP() << "No CUDA interop: " << why;
    }
    // MeshScene's rocks as bumpy grids of 16 x 16 cells (512
    // triangles, enough for levels of detail), four times as large,
    // drawn through GPU culling: a coarser level shades differently.
    const auto gridScene = [] {
        Scene scene = MeshScene();
        constexpr int kCells = 16;
        std::vector<float> points;
        std::vector<std::int32_t> indices;
        for (int y = 0; y <= kCells; ++y) {
            for (int x = 0; x <= kCells; ++x) {
                const float u =
                    -1.5f + 3.0f * static_cast<float>(x) / kCells;
                const float v =
                    -1.5f + 3.0f * static_cast<float>(y) / kCells;
                points.insert(
                    points.end(),
                    {u, v,
                     0.6f * std::sin(4.0f * u) * std::cos(4.0f * v)});
            }
        }
        const std::array<float, 9> scales{4, 4, 4, 4, 4, 4, 4, 4, 4};
        ernie::PointInstancer(scene, "/World/Rocks")
            .SetScales(Tensor::Copy<float>(scales, {3, 3}));
        for (int y = 0; y < kCells; ++y) {
            for (int x = 0; x < kCells; ++x) {
                const int a = y * (kCells + 1) + x;
                const int b = a + 1;
                const int c = a + kCells + 1;
                indices.insert(indices.end(),
                               {a, b, c + 1, a, c + 1, c});
            }
        }
        const std::vector<std::int32_t> counts(indices.size() / 3, 3);
        ernie::Mesh(scene, "/World/Rocks/Rock")
            .SetPoints(Tensor::Copy<float>(
                points,
                {static_cast<std::int64_t>(points.size() / 3), 3}))
            .SetFaceVertexCounts(Tensor::Copy<std::int32_t>(
                counts, {static_cast<std::int64_t>(counts.size())}))
            .SetFaceVertexIndices(Tensor::Copy<std::int32_t>(
                indices,
                {static_cast<std::int64_t>(indices.size())}));
        ernie::RenderSettings(scene, "/Render/Settings")
            .SetCulling(tokens::kGpu);
        return scene;
    };
    Scene host = gridScene();
    RasterDelegate reference(*context_);
    const Tensor expected = reference.Render(host);
    EXPECT_GT(TexelAt(host, expected, {12, 0, 0})[1], 100)
        << "the green grid drawn";

    Scene device = gridScene();
    device.Write("/World/Rocks/Rock", tokens::kPoints,
                 cuda->Upload(*device.Read("/World/Rocks/Rock",
                                           tokens::kPoints)));
    const Tensor image = delegate.Render(device);
    EXPECT_EQ(std::memcmp(image.Data(), expected.Data(),
                          static_cast<std::size_t>(image.Elements())),
              0)
        << "levels of detail from the device's points, not zeros";
}

TEST_F(RasterDelegateTest,
       CudaInstancesWithoutExtentAreBoundedOnTheGpu) {
    RasterDelegate delegate(*context_);
    std::string why;
    ernie::ComputeInterop* cuda = delegate.Compute(&why);
    if (cuda == nullptr) {
        GTEST_SKIP() << "No CUDA interop: " << why;
    }
    // MeshScene's rocks from CUDA without an extent, moved each
    // frame: their bounds measured on the GPU a frame or two late,
    // not copied to the host, draw them as host positions do.
    const std::array<float, 9> moved{12, -10, 0, 6, 4, 0, 12, 10, 0};
    Scene host = MeshScene();
    ernie::PointInstancer(host, "/World/Rocks")
        .SetPositions(Tensor::Copy<float>(moved, {3, 3}));
    RasterDelegate reference(*context_);
    const Tensor expected = reference.Render(host);

    Scene device = MeshScene();
    const Tensor start = cuda->Upload(
        *device.Read("/World/Rocks", tokens::kPositions));
    const Tensor end =
        cuda->Upload(Tensor::Copy<float>(moved, {3, 3}));
    for (int frame = 0; frame < 4; ++frame) {
        device.Write("/World/Rocks", tokens::kPositions,
                     frame < 2 ? start : end);
        delegate.Render(device);
    }
    const Tensor image = delegate.Render(device);
    EXPECT_EQ(std::memcmp(image.Data(), expected.Data(),
                          static_cast<std::size_t>(image.Elements())),
              0);
    EXPECT_GT(TexelAt(device, image, {6, 4, 0})[1], 200)
        << "the moved instance";
}

TEST(ExtentOf, ReadsValuesAndArrays) {
    Scene scene;
    scene.Define("/A", tokens::kMesh)
        .Set(tokens::kExtent, std::vector<float>{0, 1, 2, 3, 4, 5});
    const auto extent = ernie::ExtentOf(scene, "/A");
    ASSERT_TRUE(extent.has_value());
    EXPECT_EQ(extent->second, glm::vec3(3.0f, 4.0f, 5.0f));
    const std::array<double, 6> values{-1, -1, -1, 1, 1, 1};
    scene.Write("/A", tokens::kExtent,
                Tensor::Copy<double>(values, {2, 3}));
    EXPECT_EQ(ernie::ExtentOf(scene, "/A")->first, glm::vec3(-1.0f));
    scene.Set("/A", tokens::kExtent, std::vector<float>{1, 2});
    EXPECT_FALSE(ernie::ExtentOf(scene, "/A").has_value());
    EXPECT_FALSE(ernie::ExtentOf(scene, "/Missing").has_value());
}

TEST_F(RasterDelegateTest, DrawsUsdGeomPrimitivesAndInstancesThem) {
    Scene scene = MeshScene();
    scene.Remove("/World/Quad");
    scene.Remove("/World/Rocks/Rock");
    ernie::Sphere::Define(scene, "/World/Ball")
        .SetRadius(3.0)
        .SetMaterialBinding("/Looks/Red");
    scene.Set(
        "/World/Ball", tokens::kTransform,
        glm::translate(glm::dmat4(1.0), glm::dvec3(-12.0, 0.0, 0.0)));
    ernie::Cube::Define(scene, "/World/Rocks/Rock")
        .SetSize(3.0)
        .SetDisplayColor(glm::dvec3(0.0, 1.0, 0.0));
    RasterDelegate delegate(*context_);
    const Tensor image = delegate.Render(scene);
    EXPECT_EQ(delegate.Meshes().MeshCount(), 2u);
    const std::uint8_t* ball = TexelAt(scene, image, {-12, 0, 3});
    EXPECT_GT(ball[0], 150);
    EXPECT_LT(ball[1] + ball[2], ball[0] / 2)
        << "the sphere, red from its material";
    for (const float y : {-10.0f, 0.0f, 10.0f}) {
        const std::uint8_t* rock =
            TexelAt(scene, image, {12, y, 1.5f});
        EXPECT_GT(rock[1], 150)
            << "a cube prototype instanced at y " << y;
    }
    const std::uint8_t* origin = TexelAt(scene, image, {0, 0, 0});
    EXPECT_EQ(origin[0] + origin[1] + origin[2], 0)
        << "the prototype draws only as instances";

    ernie::Sphere(scene, "/World/Ball").SetRadius(0.5);
    const Tensor smaller = delegate.Render(scene);
    const std::uint8_t* rim = TexelAt(scene, smaller, {-10, 0, 0});
    EXPECT_EQ(rim[0], 0) << "editing the radius regenerates the mesh";
}

namespace {

/// FlatScene with its terrain replaced by a grey GroundPlane at
/// `height`.
Scene GroundScene(double height) {
    Scene scene = FlatScene();
    scene.Remove("/World/Terrain");
    ernie::GroundPlane::Define(scene, "/World/Ground")
        .SetHeight(height)
        .SetMaterialBinding("/Looks/Grey");
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
    EXPECT_NEAR(At<float>(vars.at("DistanceToImagePlane"), c, c),
                90.0f, 0.5f);
    EXPECT_NEAR(At<float>(vars.at("DistanceToImagePlane"), 0, 0),
                90.0f, 0.5f)
        << "a flat plane under a nadir camera";
    EXPECT_NEAR(At<float>(vars.at("Normal"), c, c, 2), 1.0f, 1e-4f);
    EXPECT_NEAR(At<float>(vars.at("HdrColor"), c, c, 1), 0.5f, 0.02f)
        << "albedo 0.5 under the sun";
    EXPECT_EQ(
        delegate.PrimPath(At<std::uint32_t>(vars.at("PrimId"), 0, 0)),
        Path("/World/Ground"));

    Scene withTerrain = FlatScene();
    ernie::GroundPlane::Define(withTerrain, "/World/Ground")
        .SetHeight(-5.0);
    ernie::RenderProductSetOutputs both = delegate.Step(withTerrain);
    vars = MapVars(both);
    EXPECT_EQ(
        delegate.PrimPath(At<std::uint32_t>(vars.at("PrimId"), c, c)),
        Path("/World/Terrain"))
        << "the terrain is in front of a plane below it";
    EXPECT_NEAR(At<float>(vars.at("DistanceToImagePlane"), c, c),
                100.0f, 0.5f);

    ernie::GroundPlane(withTerrain, "/World/Ground")
        .SetVisibility(tokens::kInvisible);
    delegate.Step(withTerrain);
    EXPECT_FALSE(delegate.Ground().Enabled());
}

TEST_F(RasterDelegateTest,
       NormalMapsFinerThanAPixelShadeAsRoughness) {
    // A normal map of slopes +-0.5 along x in a checker, tiled every
    // centimetre and seen from 90 m: each pixel covers thousands of
    // tiles, flat on average, with slopes deviating by sqrt(0.125)
    // per axis. Under a low sun the plane shades as Hapke's theta-bar
    // widened by that variance (`widenedMeanSlope`), not as the flat
    // average normal.
    Scene scene = GroundScene(10.0);
    scene.Set("/World/Sun", tokens::kTransform,
              ernie::DistantLightTransform({1.0, 0.0, 0.35}));
    ernie::RenderSettings::Define(scene, "/Render/Settings")
        .SetAmbient(0.0);
    ernie::Material(scene, "/Looks/Grey")
        .SetBrdf("hapke")
        .SetDetailTileSize(0.01)
        .SetDetailHexTiling(false);
    scene.Set("/Looks/Grey", "inputs:theta_bar", 15.0);
    std::vector<float> normals;
    for (int row = 0; row < 8; ++row) {
        for (int col = 0; col < 8; ++col) {
            const float slope = (row + col) % 2 == 0 ? 0.5f : -0.5f;
            const glm::vec3 n =
                glm::normalize(glm::vec3(-slope, 0.0f, 1.0f));
            normals.insert(normals.end(),
                           {0.5f * n.x + 0.5f, 0.5f * n.y + 0.5f,
                            0.5f * n.z + 0.5f, 1.0f});
        }
    }
    const auto meanGreen = [&](RasterDelegate& delegate) {
        ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
        const auto vars = MapVars(outputs);
        double sum = 0.0;
        for (std::size_t row = 0; row < kVarSize; ++row) {
            for (std::size_t col = 0; col < kVarSize; ++col) {
                sum += static_cast<double>(
                    At<float>(vars.at("HdrColor"), row, col, 1));
            }
        }
        return sum / static_cast<double>(kVarSize * kVarSize);
    };
    RasterDelegate delegate(*context_);
    const double smooth = meanGreen(delegate);
    ernie::Material(scene, "/Looks/Grey")
        .SetDetailNormal(Tensor::Copy<float>(normals, {8, 8, 4}));
    const double mapped = meanGreen(delegate);

    scene.Clear("/Looks/Grey", tokens::kDetailNormal);
    const double tanT = std::tan(15.0 * std::numbers::pi / 180.0);
    scene.Set("/Looks/Grey", "inputs:theta_bar",
              std::atan(std::sqrt(tanT * tanT +
                                  2.0 * 0.125 / std::numbers::pi)) *
                  180.0 / std::numbers::pi);
    const double widened = meanGreen(delegate);
    EXPECT_GT(smooth - widened, 0.02 * smooth)
        << "the widened roughness shades the plane darker";
    EXPECT_NEAR(mapped, widened, 0.005 * widened);
}

TEST_F(RasterDelegateTest,
       GroundPlanesReachTheHorizonWithACheckerboard) {
    Scene scene = GroundScene(0.0);
    ernie::GroundPlane(scene, "/World/Ground")
        .SetChecker(true)
        .SetCheckerSize(4.0)
        .SetCheckerColor1(glm::dvec3(0.0));
    RasterDelegate delegate(*context_);
    ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
    const auto vars = MapVars(outputs);
    int light = 0;
    int dark = 0;
    for (std::size_t row = 0; row < kVarSize; ++row) {
        for (std::size_t col = 0; col < kVarSize; ++col) {
            const float value =
                At<float>(vars.at("HdrColor"), row, col, 1);
            light += value > 0.4f ? 1 : 0;
            dark += value < 0.1f ? 1 : 0;
        }
    }
    EXPECT_GT(light, 300)
        << "4 m squares at about 1 m per pixel are crisp";
    EXPECT_GT(dark, 300);

    scene.Set("/World/Camera", tokens::kTransform,
              ernie::LookAtTransform({0.0, 0.0, 2.0},
                                     {0.0, 1000.0, 1.0},
                                     {0.0, 0.0, 1.0}));
    ernie::RenderProductSetOutputs horizon = delegate.Step(scene);
    const auto far = MapVars(horizon);
    const std::size_t c = kVarSize / 2;
    const std::uint32_t top =
        At<std::uint32_t>(far.at("PrimId"), 0, c);
    const std::uint32_t bottom =
        At<std::uint32_t>(far.at("PrimId"), kVarSize - 1, c);
    EXPECT_EQ(top, 0u) << "sky above the horizon";
    EXPECT_EQ(delegate.PrimPath(bottom), Path("/World/Ground"));
    std::size_t lastGround = 0;
    for (std::size_t row = 0; row < kVarSize; ++row) {
        if (At<std::uint32_t>(far.at("PrimId"), row, c) != 0) {
            lastGround = row;
            break;
        }
    }
    EXPECT_GT(lastGround, kVarSize / 2 - 3)
        << "the plane reaches up to the horizon";
    EXPECT_LT(lastGround, kVarSize / 2 + 3);
    const float distant =
        At<float>(far.at("HdrColor"), lastGround, c, 1);
    EXPECT_NEAR(distant, 0.25f, 0.1f)
        << "the filtered checker fades to its mean far away";
}

TEST_F(RasterDelegateTest, PrimitivesStandOnTheGroundAndShadowIt) {
    Scene scene = GroundScene(0.0);
    ernie::DistantLight(scene, "/World/Sun").SetShadowEnable(true);
    scene.Set("/World/Sun", tokens::kTransform,
              ernie::DistantLightTransform({1.0, 0.0, 1.0}));
    ernie::Cube::Define(scene, "/World/Block").SetSize(6.0);
    scene.Set(
        "/World/Block", tokens::kTransform,
        glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 0.0, 3.0)));
    RasterDelegate delegate(*context_);
    ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
    const auto vars = MapVars(outputs);
    const auto [row, col] = PixelOf(scene, 0.0f, 0.0f);
    EXPECT_EQ(delegate.PrimPath(
                  At<std::uint32_t>(vars.at("PrimId"), row, col)),
              Path("/World/Block"));
    const auto [shadowRow, shadowCol] = PixelOf(scene, -5.0f, 0.0f);
    const auto [litRow, litCol] = PixelOf(scene, 5.0f, 5.0f);
    EXPECT_EQ(delegate.PrimPath(At<std::uint32_t>(
                  vars.at("PrimId"), shadowRow, shadowCol)),
              Path("/World/Ground"));
    EXPECT_LT(At<float>(vars.at("HdrColor"), shadowRow, shadowCol, 1),
              0.2f *
                  At<float>(vars.at("HdrColor"), litRow, litCol, 1))
        << "the cube shadows the ground away from the sun";
}

TEST_F(RasterDelegateTest,
       FloatingCastersShadowTheGroundFarBelowThem) {
    // A block 8 above the ground: its shadow lands below the level of
    // its own bottom, where cascades fitted to the casters alone
    // would not reach.
    Scene scene = GroundScene(0.0);
    ernie::DistantLight(scene, "/World/Sun").SetShadowEnable(true);
    scene.Set("/World/Sun", tokens::kTransform,
              ernie::DistantLightTransform({1.0, 0.0, 1.0}));
    ernie::Cube::Define(scene, "/World/Block").SetSize(4.0);
    scene.Set(
        "/World/Block", tokens::kTransform,
        glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 0.0, 8.0)));
    RasterDelegate delegate(*context_);
    ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
    const auto vars = MapVars(outputs);
    // Under the sun at 45 degrees, the shadow of the block's centre
    // falls 8 to the -x side.
    const auto [shadowRow, shadowCol] = PixelOf(scene, -8.0f, 0.0f);
    const auto [litRow, litCol] = PixelOf(scene, 5.0f, 5.0f);
    EXPECT_EQ(delegate.PrimPath(At<std::uint32_t>(
                  vars.at("PrimId"), shadowRow, shadowCol)),
              Path("/World/Ground"));
    EXPECT_LT(At<float>(vars.at("HdrColor"), shadowRow, shadowCol, 1),
              0.2f *
                  At<float>(vars.at("HdrColor"), litRow, litCol, 1));
}

TEST_F(RasterDelegateTest,
       FloatingCastersShadowTheGroundInBatchesToo) {
    // As above, through a batch of two views posed alike: its one
    // shadow map, fitted to the casters with no far split, reaches
    // the ground below the block.
    Scene scene = GroundScene(0.0);
    ernie::DistantLight(scene, "/World/Sun").SetShadowEnable(true);
    scene.Set("/World/Sun", tokens::kTransform,
              ernie::DistantLightTransform({1.0, 0.0, 1.0}));
    ernie::Cube::Define(scene, "/World/Block").SetSize(4.0);
    scene.Set(
        "/World/Block", tokens::kTransform,
        glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 0.0, 8.0)));
    const glm::mat4 pose(scene.WorldTransform("/World/Camera"));
    std::vector<float> poses;
    for (int v = 0; v < 2; ++v) {
        poses.insert(poses.end(), glm::value_ptr(pose),
                     glm::value_ptr(pose) + 16);
    }
    ernie::Camera(scene, "/World/Camera")
        .SetPoses(Tensor::Copy<float>(poses, {2, 4, 4}));
    RasterDelegate delegate(*context_);
    ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
    const auto vars = MapVars(outputs);
    const Tensor& hdr = vars.at("HdrColor");
    ASSERT_EQ(hdr.Shape().front(), 2);
    const auto green =
        [&](std::size_t view,
            std::pair<std::size_t, std::size_t> pixel) {
            return hdr.As<float>()[((view * kVarSize + pixel.first) *
                                        kVarSize +
                                    pixel.second) *
                                       4 +
                                   1];
        };
    for (std::size_t view = 0; view < 2; ++view) {
        EXPECT_LT(green(view, PixelOf(scene, -8.0f, 0.0f)),
                  0.2f * green(view, PixelOf(scene, 5.0f, 5.0f)))
            << "view " << view;
    }
}

namespace {

/// MeshScene under /World moved by `offset`, with a checkered ground
/// plane 1 below the meshes (2 unit squares, so offsets in multiples
/// of 4 keep the pattern in place), a lamp and a terrain beside them.
Scene MovedScene(const glm::dvec3& offset) {
    Scene scene = MeshScene();
    scene.Define("/World", tokens::kXform)
        .Set(tokens::kTransform,
             glm::translate(glm::dmat4(1.0), offset));
    ernie::GroundPlane::Define(scene, "/World/Ground")
        .SetHeight(offset.z - 1.0)
        .SetChecker(true)
        .SetCheckerSize(2.0);
    ernie::SphereLight::Define(scene, "/World/Lamp")
        .SetIntensity(30.0);
    scene.Set(
        "/World/Lamp", tokens::kTransform,
        glm::translate(glm::dmat4(1.0), glm::dvec3(4.0, 4.0, 6.0)));
    return scene;
}

}  // namespace

TEST_F(RasterDelegateTest,
       ScenesFarFromTheWorldOriginRenderAsNearIt) {
    // About 2000 km out and 3 km up, where float positions are 0.25 m
    // apart: rendered relative to an origin near the camera, the
    // image matches the scene's at the world origin.
    RasterDelegate delegate(*context_);
    Scene near = MovedScene(glm::dvec3(0.0));
    const std::vector<std::uint8_t> expected =
        Bytes(delegate.Render(near));
    Scene far =
        MovedScene(glm::dvec3(1'234'564.0, -1'765'432.0, 3000.0));
    const std::vector<std::uint8_t> moved =
        Bytes(delegate.Render(far));
    EXPECT_LE(BytesDiffering(moved, expected), expected.size() / 1000)
        << "the origin moved with the camera, a rebase";
    // And back near the world origin: the origin follows again.
    EXPECT_LE(BytesDiffering(Bytes(delegate.Render(near)), expected),
              expected.size() / 1000);
}

namespace {

/// How far the sun's cascades reach along the view: their last split.
float ShadowDistance(RasterDelegate& delegate) {
    const std::array<float, 4> splits =
        delegate.Renderer().ShadowSplits();
    return *std::max_element(splits.begin(), splits.end());
}

}  // namespace

TEST_F(RasterDelegateTest, ShadowDistancesFitTheDepthsTheViewSees) {
    // MeshScene's meshes lie 60 below its camera, and a quad 360
    // below out of its view: once frames have measured what the view
    // sees, the sun's cascades reach a little past the meshes rather
    // than as far as the quad.
    RasterDelegate delegate(*context_);
    Scene scene = MeshScene();
    ernie::testing::DefineQuad(scene, "/World/Unseen");
    scene.Set("/World/Unseen", tokens::kTransform,
              glm::translate(glm::dmat4(1.0),
                             glm::dvec3(500.0, 0.0, -300.0)));
    delegate.Render(scene);
    const float sceneDistance = ShadowDistance(delegate);
    EXPECT_GT(sceneDistance, 360.0f);
    for (int i = 0; i < 6; ++i) {
        delegate.Render(scene);
    }
    const float fitted = ShadowDistance(delegate);
    EXPECT_GT(fitted, 60.0f);
    EXPECT_LT(fitted, 80.0f);
    // A view far from the one measured falls back to the scene's
    // reach until it measures again, as a first frame there does.
    scene.Set("/World/Camera", tokens::kTransform,
              ernie::LookAtTransform({0.0, 0.0, 30.0},
                                     {0.0, 0.0, 0.0},
                                     {0.0, 1.0, 0.0}));
    delegate.Render(scene);
    RasterDelegate fresh(*context_);
    fresh.Render(scene);
    EXPECT_EQ(ShadowDistance(delegate), ShadowDistance(fresh));
}

TEST_F(RasterDelegateTest, ShadowDistancesFitWhatACubeCameraSees) {
    // MeshScene through an equidistant fisheye reaching 80 degrees
    // off axis, on cube faces: its faces measure the depths they see,
    // and its cascades, which surround the eye to hold every face,
    // reach a little past the meshes 60 below rather than to a quad
    // 340 above, which the fisheye does not see.
    RasterDelegate delegate(*context_);
    Scene scene = MeshScene();
    ernie::testing::DefineQuad(scene, "/World/Unseen");
    scene.Set(
        "/World/Unseen", tokens::kTransform,
        glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 0.0, 400.0)));
    const double aperture = 20.0;
    const double corner = std::sqrt(2.0) * 0.5 * kSize;
    ernie::Camera(scene, "/World/Camera")
        .SetLensProjection(tokens::kFisheye)
        .SetVerticalAperture(aperture)
        .SetHorizontalAperture(aperture)
        .SetFocalLength(aperture * corner / (80.0 * M_PI / 180.0) /
                        kSize);
    ASSERT_EQ(ernie::CubeFacesOf(ernie::CameraParamsOf(
                  scene, "/World/Camera", kSize, kSize)),
              ernie::kCubeFaces);
    delegate.Render(scene);
    const std::span<const ernie::ShadowCascade> views =
        delegate.Renderer().LightShadowViews();
    const std::vector<ernie::ShadowCascade> cascades(views.begin(),
                                                     views.end());
    ASSERT_FALSE(cascades.empty())
        << "a cube camera's view has cascades of its own";
    EXPECT_GT(ShadowDistance(delegate), 340.0f);
    for (int i = 0; i < 6; ++i) {
        delegate.Render(scene);
    }
    // The faces' depths along their own axes: the meshes' corners
    // are farther than 60 along the side faces.
    const float fitted = ShadowDistance(delegate);
    EXPECT_GT(fitted, 60.0f);
    EXPECT_LT(fitted, 200.0f);
}

TEST_F(RasterDelegateTest, EditsRedrawOnlyTheirMeshesAsAFullSyncWould) {
    Scene scene = MeshScene();
    ernie::Xform::Define(scene, "/World/Group");
    for (const char* name : {"A", "B"}) {
        const Path cube = Path("/World/Group").Child(name);
        ernie::Cube::Define(scene, cube).SetSize(6.0).SetDisplayColor(
            {0.2, 0.8, 0.3});
        ernie::SetXform(scene, cube,
                        {.translate = {name[0] == 'A' ? -4.0 : 4.0, 8.0, 0.0}});
    }
    RasterDelegate delegate(*context_);
    delegate.Render(scene);
    // Each edit reaches only some meshes' draws: the renderer kept in
    // step draws as a new one syncing the whole scene does.
    const auto expectFresh = [&](const char* edit) {
        const Tensor kept = delegate.Render(scene);
        RasterDelegate fresh(*context_);
        const Tensor whole = fresh.Render(scene);
        ASSERT_EQ(kept.Elements(), whole.Elements());
        EXPECT_EQ(std::memcmp(kept.Data(), whole.Data(),
                              static_cast<std::size_t>(kept.Elements())),
                  0)
            << edit;
    };
    ernie::Cube(scene, "/World/Group/A").SetDisplayColor({0.9, 0.1, 0.9});
    expectFresh("a mesh's colour");
    ernie::SetXform(scene, "/World/Group/B", {.translate = {6.0, -8.0, 0.0}});
    expectFresh("a mesh's transform");
    ernie::SetXform(scene, "/World/Group", {.translate = {0.0, 4.0, 0.0}});
    expectFresh("its group's transform");
    ernie::Xform(scene, "/World/Group").SetVisibility(tokens::kInvisible);
    expectFresh("its group hidden");
    ernie::Xform(scene, "/World/Group").SetVisibility(tokens::kInherited);
    expectFresh("its group shown");
    scene.Remove("/World/Group/B");
    ernie::Cube::Define(scene, "/World/Group/C").SetSize(3.0);
    expectFresh("a mesh removed and another added");
    // What reaches meshes from other prims: a material they are bound
    // to, gone and back, and their group's world.
    ernie::Cube(scene, "/World/Group/A").SetMaterialBinding("/Looks/Red");
    expectFresh("a mesh bound to a material");
    scene.Remove("/Looks/Red");
    expectFresh("the material removed");
    ernie::Material::Define(scene, "/Looks/Red")
        .SetAlbedo(glm::dvec3(0.0, 0.0, 1.0));
    expectFresh("the material back");
    ernie::Xform(scene, "/World/Group").SetWorld(1);
    expectFresh("its group's world");
}

TEST_F(RasterDelegateTest, LikeShapesShareOneMeshThatEditsKeepApart) {
    Scene scene = MeshScene();
    for (int i = 0; i < 3; ++i) {
        const Path cube("/World/Cube" + std::to_string(i));
        ernie::Cube::Define(scene, cube).SetSize(4.0);
        ernie::SetXform(scene, cube,
                        {.translate = {-8.0 + 8.0 * i, 6.0, 0.0}});
    }
    RasterDelegate delegate(*context_);
    const auto meshes = [&delegate] {
        for (const auto& [name, value] : delegate.Stats().counters) {
            if (name == ernie::raster_stats::kMeshes) {
                return value;
            }
        }
        return std::uint64_t{0};
    };
    const auto expectFresh = [&](const char* edit) {
        const Tensor kept = delegate.Render(scene);
        RasterDelegate fresh(*context_);
        const Tensor whole = fresh.Render(scene);
        ASSERT_EQ(kept.Elements(), whole.Elements());
        EXPECT_EQ(std::memcmp(kept.Data(), whole.Data(),
                              static_cast<std::size_t>(kept.Elements())),
                  0)
            << edit;
    };
    delegate.Render(scene);
    const std::uint64_t alone = meshes();
    // A fourth like cube adds a draw, not a mesh.
    ernie::Cube::Define(scene, "/World/Cube3").SetSize(4.0);
    ernie::SetXform(scene, "/World/Cube3", {.translate = {0.0, -6.0, 0.0}});
    expectFresh("a like cube added");
    EXPECT_EQ(meshes(), alone);
    // One resized takes a mesh of its own; the others keep theirs.
    ernie::Cube(scene, "/World/Cube1").SetSize(2.0);
    expectFresh("a cube resized");
    EXPECT_EQ(meshes(), alone + 1);
    // Back to the others' size, it shares theirs again.
    ernie::Cube(scene, "/World/Cube1").SetSize(4.0);
    expectFresh("a cube resized back");
    EXPECT_EQ(meshes(), alone);
    // The mesh stays while a cube draws it, and goes with the last.
    for (const char* gone : {"/World/Cube0", "/World/Cube1", "/World/Cube2"}) {
        scene.Remove(gone);
        expectFresh("a cube removed");
        EXPECT_EQ(meshes(), alone);
    }
    scene.Remove("/World/Cube3");
    expectFresh("the last cube removed");
    EXPECT_EQ(meshes(), alone - 1);
}

TEST_F(RasterDelegateTest, ColumnsOfHostRowsDrawAsTheirPrimsWouldAlone) {
    const auto scene = [](bool column, float lift) {
        Scene built = MeshScene();
        std::vector<Path> cubes;
        std::vector<float> rows;
        // Red, green and blue: the colours as a column too.
        const std::vector<float> colours = {0.9f, 0.1f, 0.1f, 0.1f, 0.9f,
                                            0.1f, 0.1f, 0.1f, 0.9f};
        for (int i = 0; i < 3; ++i) {
            const Path cube("/World/Cube" + std::to_string(i));
            ernie::Cube::Define(built, cube).SetSize(4.0);
            const glm::dvec3 at(-8.0 + 8.0 * i, -6.0,
                                static_cast<double>(lift) * i);
            ernie::SetXform(built, cube, {.translate = at});
            if (!column) {
                const auto c = static_cast<std::size_t>(3 * i);
                ernie::Cube(built, cube).SetDisplayColor(
                    std::span(colours).subspan(c, 3));
            }
            cubes.push_back(cube);
            rows.insert(rows.end(), {static_cast<float>(at.x),
                                     static_cast<float>(at.y),
                                     static_cast<float>(at.z)});
        }
        if (column) {
            built.GetColumn(ernie::PrimSet(cubes), tokens::kTranslate)
                .Write(Tensor::Copy<float>(rows, {3, 3}));
            built.GetColumn(ernie::PrimSet(cubes), tokens::kDisplayColor)
                .Write(Tensor::Copy<float>(colours, {3, 3}));
        }
        return built;
    };
    const auto near = [](const Tensor& a, const Tensor& b) {
        int worst = 0;
        for (std::int64_t i = 0; i < a.Elements(); ++i) {
            worst = std::max(worst, std::abs(int{a.As<std::uint8_t>()[i]} -
                                             int{b.As<std::uint8_t>()[i]}));
        }
        return worst;
    };
    // Written as a column twice, kept in step with the second.
    Scene columns = scene(true, 0.0f);
    RasterDelegate delegate(*context_);
    delegate.Render(columns);
    std::vector<float> lifted;
    for (int i = 0; i < 3; ++i) {
        lifted.insert(lifted.end(), {-8.0f + 8.0f * static_cast<float>(i),
                                     -6.0f, 2.0f * static_cast<float>(i)});
    }
    columns
        .GetColumn(ernie::PrimSet({"/World/Cube0", "/World/Cube1",
                                   "/World/Cube2"}),
                   tokens::kTranslate)
        .Write(Tensor::Copy<float>(lifted, {3, 3}));
    const std::vector<float> colours = {0.9f, 0.1f, 0.1f, 0.1f, 0.9f,
                                        0.1f, 0.1f, 0.1f, 0.9f};
    columns
        .GetColumn(ernie::PrimSet({"/World/Cube0", "/World/Cube1",
                                   "/World/Cube2"}),
                   tokens::kDisplayColor)
        .Write(Tensor::Copy<float>(colours, {3, 3}));
    const Tensor drawn = delegate.Render(columns);
    RasterDelegate fresh(*context_);
    const Tensor alone = fresh.Render(scene(false, 2.0f));
    ASSERT_EQ(drawn.Elements(), alone.Elements());
    EXPECT_LE(near(drawn, alone), 2);
}

TEST_F(RasterDelegateTest, ColumnsOfVisibilityHideAndShowInstances) {
    const std::vector<Path> cubes = {"/World/Cube0", "/World/Cube1",
                                     "/World/Cube2"};
    const auto scene = [&](bool column, bool middleHidden) {
        Scene built = MeshScene();
        std::vector<float> rows;
        for (int i = 0; i < 3; ++i) {
            const auto cube = cubes[static_cast<std::size_t>(i)];
            ernie::Cube::Define(built, cube).SetSize(4.0);
            const glm::dvec3 at(-8.0 + 8.0 * i, -6.0, 0.0);
            ernie::SetXform(built, cube, {.translate = at});
            rows.insert(rows.end(), {static_cast<float>(at.x),
                                     static_cast<float>(at.y), 0.0f});
        }
        if (column) {
            built.GetColumn(ernie::PrimSet(cubes), tokens::kTranslate)
                .Write(Tensor::Copy<float>(rows, {3, 3}));
            const std::vector<float> shown = {1.0f,
                                              middleHidden ? 0.0f : 1.0f,
                                              1.0f};
            built.GetColumn(ernie::PrimSet(cubes), tokens::kVisibility)
                .Write(Tensor::Copy<float>(shown, {3}));
        } else if (middleHidden) {
            ernie::Cube(built, cubes[1]).SetVisibility(tokens::kInvisible);
        }
        return built;
    };
    const auto worst = [](const Tensor& a, const Tensor& b) {
        int most = 0;
        for (std::int64_t i = 0; i < a.Elements(); ++i) {
            most = std::max(most, std::abs(int{a.As<std::uint8_t>()[i]} -
                                           int{b.As<std::uint8_t>()[i]}));
        }
        return most;
    };
    // Hidden by its row, then shown again by the column written anew.
    Scene columns = scene(true, true);
    RasterDelegate delegate(*context_);
    const Tensor hidden = delegate.Render(columns);
    RasterDelegate alone(*context_);
    EXPECT_LE(worst(hidden, alone.Render(scene(false, true))), 2);
    const std::vector<float> all = {1.0f, 1.0f, 1.0f};
    columns.GetColumn(ernie::PrimSet(cubes), tokens::kVisibility)
        .Write(Tensor::Copy<float>(all, {3}));
    const Tensor shown = delegate.Render(columns);
    RasterDelegate every(*context_);
    EXPECT_LE(worst(shown, every.Render(scene(false, false))), 2);
    EXPECT_GT(worst(hidden, shown), 50) << "the middle cube came back";
}

TEST_F(RasterDelegateTest, EditsBesideGpuPosedGroupsRedrawAsAFullSyncWould) {
    Scene scene = MeshScene();
    const std::vector<Path> bodies = {"/World/Body0", "/World/Body1",
                                      "/World/Body2"};
    std::vector<float> rows;
    for (int i = 0; i < 3; ++i) {
        const glm::dvec3 at(-8.0 + 8.0 * i, -6.0, 0.0);
        ernie::Cube::Define(scene, bodies[static_cast<std::size_t>(i)])
            .SetSize(4.0)
            .SetXform({.translate = at});
        rows.insert(rows.end(), {static_cast<float>(at.x),
                                 static_cast<float>(at.y), 0.0f});
    }
    // Posed as one group, from a column.
    scene.GetColumn(ernie::PrimSet(bodies), tokens::kTranslate)
        .Write(Tensor::Copy<float>(rows, {3, 3}));
    ernie::Cube::Define(scene, "/World/Lone")
        .SetSize(3.0)
        .SetXform({.translate = {0.0, 8.0, 0.0}});
    RasterDelegate delegate(*context_);
    delegate.Render(scene);
    const auto expectFresh = [&](const char* edit) {
        const Tensor kept = delegate.Render(scene);
        RasterDelegate fresh(*context_);
        const Tensor whole = fresh.Render(scene);
        ASSERT_EQ(kept.Elements(), whole.Elements());
        int worst = 0;
        for (std::int64_t i = 0; i < kept.Elements(); ++i) {
            worst = std::max(worst,
                             std::abs(int{kept.As<std::uint8_t>()[i]} -
                                      int{whole.As<std::uint8_t>()[i]}));
        }
        EXPECT_LE(worst, 2) << edit;
    };
    ernie::Cube(scene, "/World/Lone").SetDisplayColor({0.9, 0.2, 0.2});
    expectFresh("a lone cube's colour");
    ernie::SetXform(scene, "/World/Lone", {.translate = {4.0, 8.0, 0.0}});
    expectFresh("a lone cube moved");
    ernie::Cube(scene, bodies[1]).SetDisplayColor({0.2, 0.2, 0.9});
    expectFresh("a posed cube's own colour");
    ernie::Cube::Define(scene, "/World/New").SetSize(2.0);
    expectFresh("a cube added");
    scene.Remove(bodies[2]);
    expectFresh("a posed cube removed, the group posed anew");
}
