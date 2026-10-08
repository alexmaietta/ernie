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
