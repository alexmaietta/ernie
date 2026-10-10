#include "ernie/device/offscreen_target.h"
#include "ernie/geometry/primitives.h"
#include "ernie/raster/delegate.h"
#include "ernie/scene/cahvore.h"
#include "ernie/scene/instruments.h"
#include "ernie/scene/isd.h"
#include "ernie/scene/orbit_controller.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/wide_view.h"
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
#include <filesystem>
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
    ernie::ComputeInterop* cuda = delegate.Compute(&why);
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
    ernie::ComputeInterop* cuda = delegate.Compute(&why);
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
    EXPECT_EQ(specs[0].filters, ernie::StepsOf(std::vector<ernie::SensorFilter>{
                                    ernie::SensorFilter::Cie, ernie::SensorFilter::Exposure}));

    scene.Define("/Render/Vars/Dn", tokens::kRenderVar).Set(tokens::kSourceName, "RawDN");
    scene.Set("/Render/Top", tokens::kOrderedVars, std::vector<Path>{"/Render/Vars/Dn"});
    EXPECT_THROW(ernie::ResolveRenderProducts(scene, top, {}), std::invalid_argument)
        << "the chain has no ADC";
    scene.Set("/Render/Top", tokens::kOrderedVars, std::vector<Path>{"/Render/Vars/Color"});
    // A chain with a type that is not built in may hold a custom integrator standing for cie:
    // the renderer, which knows the custom filters, checks it.
    scene.Set("/Render/Filters/Cie", tokens::kFilterType, "bloom");
    EXPECT_NO_THROW(ernie::ResolveRenderProducts(scene, top, {}));
    scene.Set("/Render/Top", tokens::kFilters, std::vector<Path>{"/World/Camera"});
    EXPECT_THROW(ernie::ResolveRenderProducts(scene, top, {}), std::invalid_argument);
    scene.Define("/Render/Filters/Adc", tokens::kSensorFilter).Set(tokens::kFilterType, "adc");
    scene.Set("/Render/Top", tokens::kFilters, std::vector<Path>{"/Render/Filters/Adc"});
    EXPECT_THROW(ernie::ResolveRenderProducts(scene, top, {}), std::invalid_argument)
        << "adc needs qe";
}

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
    for (double w = 360.5; w < 830.0; w += 1.0) {
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

    // Noise on pixels well below the full well (saturated pixels read the full well, without
    // noise).
    scene.Set("/World/Camera", tokens::kCameraExposure, 0.0f);
    scene.Set("/World/Camera", tokens::kExposureTime, 1e-6);  // its pixels are 0.5 mm, at f/1
    ernie::RenderProductSetOutputs noiseless = delegate.Step(scene);
    const auto cleanVars = MapVars(noiseless);
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
            clean += At<float>(cleanVars.at("Electrons"), row, col, 1);
            differing += At<float>(a, row, col, 1) != At<float>(b, row, col, 1) ? 1 : 0;
        }
    }
    const ernie::Camera camera = ernie::CameraOf(scene, "/World/Camera", kVarSize, kVarSize);
    ASSERT_LT(clean / (kVarSize * kVarSize), 0.5 * camera.sensor.fullWellElectrons)
        << "unsaturated";
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

TEST_F(RasterDelegateTest, OrthographicSeesItsApertureInTheScene) {
    // Looking straight down from 50 over terrain spanning x in [-65, 65]: an aperture of 2000
    // tenths of a scene unit spans 200 units, 6.25 per pixel, so the terrain covers the middle
    // 65% of the image, every pixel of it at the same distance.
    Scene scene = FlatScene();
    scene.Set("/World/Camera", tokens::kTransform,
              ernie::LookAtTransform({0.0, 0.0, 50.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}));
    scene.Set("/World/Camera", tokens::kProjection, std::string(tokens::kOrthographic));
    scene.Set("/World/Camera", tokens::kVerticalAperture, 2000.0f);
    scene.Set("/World/Camera", tokens::kClippingRange, glm::dvec2(1.0, 1000.0));
    RasterDelegate delegate(*context_);
    ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
    const auto vars = MapVars(outputs);
    const std::size_t c = kVarSize / 2;
    EXPECT_EQ(At<std::uint32_t>(vars.at("PrimId"), c, 4), 0u) << "x = -71.9, past the terrain";
    for (const std::size_t col : {std::size_t{6}, c, std::size_t{25}}) {
        EXPECT_NE(At<std::uint32_t>(vars.at("PrimId"), c, col), 0u) << "column " << col;
        EXPECT_NEAR(At<float>(vars.at("DistanceToImagePlane"), c, col), 50.0f, 0.05f);
        EXPECT_NEAR(At<float>(vars.at("DistanceToCamera"), c, col), 50.0f, 0.05f)
            << "rays start on the image plane";
        EXPECT_NEAR(At<float>(vars.at("HdrColor"), c, col), 0.5f, 0.02f)
            << "the sun overhead lights it as a perspective view does";
    }
    EXPECT_EQ(At<std::uint32_t>(vars.at("PrimId"), c, 27), 0u) << "x = 71.9";
}

TEST_F(RasterDelegateTest, PushbroomScansItsRowsAlongItsMotion) {
    // Looking straight down from 50, moving 32 along +y over one frame: row r sees the ground
    // under the camera (r + 0.5) / 32 of the way along, y = r + 0.5 - 16, one unit per row, and
    // across it a pinhole's view of x. A 2-unit cube at y = 5 shows on rows 20 and 21 only.
    Scene scene = FlatScene();
    scene.Define("/World/Box", tokens::kCube)
        .Set(tokens::kSize, 2.0)
        .Set(tokens::kTransform,
             glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 5.0, 1.0)));
    const auto at = [](double y) {
        return ernie::LookAtTransform({0.0, y, 50.0}, {0.0, y, 0.0}, {0.0, 1.0, 0.0});
    };
    scene.Set("/World/Camera", tokens::kCameraScan, std::string(tokens::kPushbroom));
    scene.Set("/World/Camera", tokens::kTransform, at(-16.0));
    RasterDelegate delegate(*context_);
    delegate.Step(scene);  // the strip's start
    scene.Set("/World/Camera", tokens::kTransform, at(16.0));
    ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
    const auto vars = MapVars(outputs);
    const std::size_t c = kVarSize / 2;
    const std::uint32_t terrain = At<std::uint32_t>(vars.at("PrimId"), 5, c);
    ASSERT_EQ(delegate.PrimPath(terrain), Path("/World/Terrain"));
    for (std::size_t row = 0; row < kVarSize; ++row) {
        const std::uint32_t id = At<std::uint32_t>(vars.at("PrimId"), row, c);
        const bool box = row == 20 || row == 21;
        EXPECT_EQ(delegate.PrimPath(id), Path(box ? "/World/Box" : "/World/Terrain"))
            << "row " << row;
    }
    const ernie::Camera camera = ernie::CameraOf(scene, "/World/Camera", kVarSize, kVarSize);
    const double x = (0.5 - camera.PrincipalPoint().x) / camera.FocalLengthPixels();
    for (const std::size_t row : {std::size_t{2}, std::size_t{12}, std::size_t{30}}) {
        EXPECT_NEAR(At<float>(vars.at("DistanceToImagePlane"), row, 0), 50.0f, 0.05f);
        EXPECT_NEAR(At<float>(vars.at("DistanceToCamera"), row, 0), 50.0 * std::sqrt(1.0 + x * x),
                    0.05)
            << "across the row as a pinhole sees, from the row's own pose";
        EXPECT_NEAR(At<float>(vars.at("HdrColor"), row, c), 0.5f, 0.02f) << "lit as a frame";
    }
}

TEST_F(RasterDelegateTest, AnIsdStripSeesTheGroundWhereUsgscsmPutsIt) {
    // usgscsm's synthetic orbital line scanner (16 x 16 pixels of about 100 m, 50 km above a
    // 1000 km body) rendered as one strip: balls of 60 m radius where the camera model puts two
    // pixels' rays on the ground show on exactly those pixels.
    const ernie::IsdLineScanner scanner(ernie::ReadIsdFile(
        (std::filesystem::path(ERNIE_TEST_FIXTURES) / "isd" / "orbitalLineScan.json").string()));
    const ernie::IsdLineScanner::Strip strip = scanner.StripOf(0, 16);
    Scene scene = FlatScene();
    scene.Remove("/World/Terrain");
    scene.Set("/Render/Settings", tokens::kResolution, glm::ivec2(16, 16));
    const std::pair<std::size_t, std::size_t> pixels[] = {{8, 8}, {3, 12}, {13, 2}};
    for (std::size_t i = 0; i < std::size(pixels); ++i) {
        const auto [row, column] = pixels[i];
        const glm::dvec3 ground = scanner.ImageToGround(row + 0.5, column + 0.5);
        scene.Define(Path("/World/Ball" + std::to_string(i)), tokens::kSphere)
            .Set(tokens::kGeomRadius, 60.0)
            .Set(tokens::kTransform, glm::translate(glm::dmat4(1.0), ground));
    }
    ernie::Camera camera = strip.camera;
    camera.transform = strip.from;
    ernie::SetCamera(scene, "/World/Camera", camera);
    scene.Set("/World/Camera", tokens::kSensorNoise, false);
    RasterDelegate delegate(*context_);
    delegate.Step(scene);  // the strip's start
    scene.Set("/World/Camera", tokens::kTransform, strip.to);
    ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
    const Tensor ids = outputs.at("/Render/Product").vars.at("PrimId").Map(ernie::MapDevice::Cpu).tensor;
    const auto at = [&](std::size_t row, std::size_t column) {
        return delegate.PrimPath(ids.As<std::uint32_t>()[row * 16 + column]);
    };
    for (std::size_t i = 0; i < std::size(pixels); ++i) {
        const auto [row, column] = pixels[i];
        const Path ball("/World/Ball" + std::to_string(i));
        EXPECT_EQ(at(row, column), ball) << "row " << row << ", column " << column;
        EXPECT_NE(at(row, column + 1), ball) << "one pixel across";
        EXPECT_NE(at(row + 1, column), ball) << "one line along";
        EXPECT_NE(at(row - 1, column), ball);
    }
}

TEST_F(RasterDelegateTest, AnLroNacStripSeesThroughItsLensDistortion) {
    // LRO's NAC, 2 x 2 summed: pixels about 0.72 m across the track and 20 m along it, 36 km
    // above the Moon, the lens moving the detector's ends by many pixels. Balls of 0.6 m radius
    // where usgscsm's model (with the ISD's own distortion) puts pixels' rays on the ground show
    // on those pixels of a strip rendered through the fitted distortion.
    const ernie::IsdLineScanner nac(ernie::ReadIsdFile(
        (std::filesystem::path(ERNIE_TEST_FIXTURES) / "isd" / "toughLroNacLineScan.json")
            .string()));
    constexpr std::uint32_t kFirst = 4096;
    constexpr std::uint32_t kLines = 64;
    const ernie::IsdLineScanner::Strip strip = nac.StripOf(kFirst, kLines);
    ASSERT_NE(strip.camera.lens.k1, 0.0);
    Scene scene = FlatScene();
    scene.Remove("/World/Terrain");
    scene.Set("/Render/Settings", tokens::kResolution,
              glm::ivec2(static_cast<int>(nac.Isd().samples), static_cast<int>(kLines)));
    const std::pair<std::size_t, std::size_t> pixels[] = {{10, 12}, {40, 1266}, {25, 2515}};
    for (std::size_t i = 0; i < std::size(pixels); ++i) {
        const auto [row, column] = pixels[i];
        const glm::dvec3 ground = nac.ImageToGround(kFirst + row + 0.5, column + 0.5);
        scene.Define(Path("/World/Ball" + std::to_string(i)), tokens::kSphere)
            .Set(tokens::kGeomRadius, 0.6)
            .Set(tokens::kTransform, glm::translate(glm::dmat4(1.0), ground));
    }
    ernie::Camera camera = strip.camera;
    camera.transform = strip.from;
    ernie::SetCamera(scene, "/World/Camera", camera);
    scene.Set("/World/Camera", tokens::kSensorNoise, false);
    scene.Set("/World/Camera", tokens::kLensDistortion,
              std::vector<float>{static_cast<float>(camera.lens.k1),
                                 static_cast<float>(camera.lens.k2),
                                 static_cast<float>(camera.lens.k3), 0.0f, 0.0f});
    RasterDelegate delegate(*context_);
    delegate.Step(scene);  // the strip's start
    scene.Set("/World/Camera", tokens::kTransform, strip.to);
    ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
    const Tensor ids =
        outputs.at("/Render/Product").vars.at("PrimId").Map(ernie::MapDevice::Cpu).tensor;
    const std::size_t width = nac.Isd().samples;
    const auto at = [&](std::size_t row, std::size_t column) {
        return delegate.PrimPath(ids.As<std::uint32_t>()[row * width + column]);
    };
    // Each ball on its row alone, centred on its column within half a pixel: PrimId reads the
    // raster pixel nearest each sensor pixel's ray, and the raster is wider than the sensor by
    // the overscan, so ids land within half a pixel.
    for (std::size_t i = 0; i < std::size(pixels); ++i) {
        const auto [row, column] = pixels[i];
        const Path ball("/World/Ball" + std::to_string(i));
        double sum = 0.0;
        int hits = 0;
        for (std::size_t r = 0; r < kLines; ++r) {
            for (std::size_t c = 0; c < width; ++c) {
                if (at(r, c) == ball) {
                    EXPECT_EQ(r, row) << ball.String() << " on another line";
                    sum += static_cast<double>(c);
                    ++hits;
                }
            }
        }
        ASSERT_GT(hits, 0) << ball.String();
        EXPECT_LE(hits, 3) << ball.String();
        EXPECT_NEAR(sum / hits, static_cast<double>(column), 0.5)
            << ball.String() << " at row " << row << ", column " << column;
    }
}

TEST_F(RasterDelegateTest, AnLroWacFrameletSeesTheGroundWhereUsgscsmPutsIt) {
    // LRO's WAC, a visible band's framelet of 1024 x 14 pixels about 75 m across, rendered as a
    // frame camera through the fitted distortion (its constant term in the focal length): balls
    // where usgscsm's push-frame model puts pixels' rays on the ground show on those pixels.
    const ernie::IsdPushFrame wac(ernie::ReadIsdFile(
        (std::filesystem::path(ERNIE_TEST_FIXTURES) / "isd" / "lroWacPushFrame.json").string()));
    const ernie::IsdPushFrame::Framelet framelet = wac.FrameletOf(182);
    const std::size_t width = wac.Isd().samples;
    const std::size_t height = wac.FrameletLines();
    Scene scene = FlatScene();
    scene.Remove("/World/Terrain");
    scene.Set("/Render/Settings", tokens::kResolution,
              glm::ivec2(static_cast<int>(width), static_cast<int>(height)));
    const std::pair<std::size_t, std::size_t> pixels[] = {{2, 10}, {7, 512}, {11, 1013}};
    for (std::size_t i = 0; i < std::size(pixels); ++i) {
        const auto [row, column] = pixels[i];
        const double line = framelet.firstLine + row + 0.5;
        const glm::dvec3 ground = wac.ImageToGround(line, column + 0.5);
        const double across = glm::length(wac.ImageToGround(line, column + 1.5) - ground);
        scene.Define(Path("/World/Ball" + std::to_string(i)), tokens::kSphere)
            .Set(tokens::kGeomRadius, 0.4 * across)
            .Set(tokens::kTransform, glm::translate(glm::dmat4(1.0), ground));
    }
    const ernie::Camera& camera = framelet.camera;
    ernie::SetCamera(scene, "/World/Camera", camera);
    scene.Set("/World/Camera", tokens::kSensorNoise, false);
    scene.Set("/World/Camera", tokens::kLensDistortion,
              std::vector<float>{static_cast<float>(camera.lens.k1),
                                 static_cast<float>(camera.lens.k2),
                                 static_cast<float>(camera.lens.k3), 0.0f, 0.0f});
    RasterDelegate delegate(*context_);
    ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
    const Tensor ids =
        outputs.at("/Render/Product").vars.at("PrimId").Map(ernie::MapDevice::Cpu).tensor;
    for (std::size_t i = 0; i < std::size(pixels); ++i) {
        const auto [row, column] = pixels[i];
        const Path ball("/World/Ball" + std::to_string(i));
        glm::dvec2 sum(0.0);
        int hits = 0;
        for (std::size_t r = 0; r < height; ++r) {
            for (std::size_t c = 0; c < width; ++c) {
                if (delegate.PrimPath(ids.As<std::uint32_t>()[r * width + c]) == ball) {
                    // The framelet's rows, flipped back if Ernie's run the other way.
                    const std::size_t framelet_row = framelet.rowsReversed ? height - 1 - r : r;
                    sum += glm::dvec2(framelet_row, c);
                    ++hits;
                }
            }
        }
        ASSERT_GT(hits, 0) << ball.String();
        EXPECT_LE(hits, 4) << ball.String();
        EXPECT_NEAR(sum.x / hits, static_cast<double>(row), 0.5) << ball.String();
        EXPECT_NEAR(sum.y / hits, static_cast<double>(column), 0.5) << ball.String();
    }
}

TEST_F(RasterDelegateTest, TheLrocNacPresetSendsEightBitNumbers) {
    // NAC-L's readout (Robinson et al. 2010, Table 2 and Appendix A): electrons over 90.5 per DN
    // into 12 bits, then the nominal table into 8.
    Scene scene = FlatScene();
    ernie::AuthorLrocNac(scene, "/World/Camera", "/Render/Product");
    scene.Set("/World/Camera", tokens::kSensorNoise, false);
    scene.Set("/World/Camera", tokens::kExposureTime, 0.004);
    RasterDelegate delegate(*context_);
    ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
    const auto vars = MapVars(outputs);
    const auto table = [](double x) {
        return x < 32 ? std::floor(x / 2) : x < 136 ? std::floor(x / 4 + 8)
             : x < 543 ? std::floor(x / 8 + 25) : x < 2207 ? std::floor(x / 16 + 59)
                       : std::floor(x / 32 + 128);
    };
    int checked = 0;
    for (std::size_t row = 0; row < kVarSize; ++row) {
        for (std::size_t col = 0; col < kVarSize; ++col) {
            const double e = std::min<double>(At<float>(vars.at("Electrons"), row, col), 334000.0);
            const double twelve = std::clamp(std::floor(e / 90.5 + 0.5), 0.0, 4095.0);
            EXPECT_EQ(At<std::uint16_t>(vars.at("RawDN"), row, col), table(twelve))
                << row << ", " << col << ": " << e << " electrons";
            checked += e > 1000.0 ? 1 : 0;
        }
    }
    EXPECT_GT(checked, 100) << "lit pixels";
}

TEST_F(RasterDelegateTest, TheMars2020NavcamAndHazcamPairsSeeInStereo) {
    // Perseverance's Navcams and front Hazcams (Maki et al. 2020, Table 2; the Hazcams on cube
    // faces), binned to 64 x 48, over a 1 m cube: each eye sends 12-bit demosaiced numbers and
    // an 8-bit JPEG view, and the cube sits further right in the left eye's image, by the
    // baseline's disparity (42.4 cm from 5 m up, 24.8 cm from 3 m).
    struct Pair {
        const char* rig;
        double baseline;
        double height;
        bool hazcam;
    };
    for (const Pair pair :
         {Pair{"/World/Navcam", 0.424, 5.0, false}, Pair{"/World/Hazcam", 0.248, 3.0, true}}) {
        Scene scene = FlatScene();
        scene.Define("/World/Cube", tokens::kCube)
            .Set(tokens::kSize, 1.0)
            .Set(tokens::kTransform, glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 0.0, 0.5)));
        const std::vector<Path> vars =
            *scene.Get("/Render/Product", tokens::kOrderedVars)->Get<std::vector<Path>>();
        for (const char* eye : {"Left", "Right"}) {
            scene.Define(Path("/Render").Child(eye), tokens::kRenderProduct)
                .Set(tokens::kOrderedVars, vars)
                .Set(tokens::kCameraRel, Path(pair.rig).Child(eye))
                .Set(tokens::kResolution, glm::ivec2(64, 48));
        }
        scene.Set("/Render/Settings", tokens::kProducts,
                  std::vector<Path>{"/Render/Left", "/Render/Right"});
        if (pair.hazcam) {
            ernie::AuthorMars2020HazcamPair(scene, pair.rig, "/Render/Left", "/Render/Right");
        } else {
            ernie::AuthorMars2020NavcamPair(scene, pair.rig, "/Render/Left", "/Render/Right");
        }
        scene.Set(
            pair.rig, tokens::kTransform,
            ernie::LookAtTransform({0.0, 0.0, pair.height}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}));
        for (const char* eye : {"Left", "Right"}) {
            scene.Set(Path(pair.rig).Child(eye), tokens::kExposureTime, 1e-4);
        }
        RasterDelegate delegate(*context_);
        ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
        double cubeColumn[2] = {0.0, 0.0};
        for (int e = 0; e < 2; ++e) {
            auto& product = outputs.at(e == 0 ? "/Render/Left" : "/Render/Right");
            const Tensor raw = product.vars.at("RawDN").Map(ernie::MapDevice::Cpu).tensor;
            const Tensor ldr = product.vars.at("LdrColor").Map(ernie::MapDevice::Cpu).tensor;
            const Tensor ids = product.vars.at("PrimId").Map(ernie::MapDevice::Cpu).tensor;
            EXPECT_EQ(ldr.Type(), DType::UInt8);
            ASSERT_EQ(raw.Type(), DType::UInt16);
            int between = 0;
            for (std::int64_t i = 0; i < raw.Elements(); ++i) {
                const std::uint16_t dn = raw.As<std::uint16_t>()[i];
                ASSERT_LE(dn, 4095) << "12 bits";
                between += dn > 0 && dn < 4095 ? 1 : 0;
            }
            EXPECT_GT(between, raw.Elements() / 2) << "exposed within the ADC's range";
            double sum = 0.0;
            int hits = 0;
            for (std::size_t row = 0; row < 48; ++row) {
                for (std::size_t col = 0; col < 64; ++col) {
                    if (delegate.PrimPath(ids.As<std::uint32_t>()[row * 64 + col]) ==
                        Path("/World/Cube")) {
                        sum += static_cast<double>(col);
                        ++hits;
                    }
                }
            }
            ASSERT_GT(hits, 0);
            cubeColumn[e] = sum / hits;
        }
        // The cube's top, 1 m below the camera's height, at f / pitch pixels per unit of x / z.
        const ernie::Camera left = ernie::CameraOf(scene, Path(pair.rig).Child("Left"), 64, 48);
        EXPECT_EQ(ernie::NeedsCubeFaces(left), pair.hazcam);
        const double disparity = cubeColumn[0] - cubeColumn[1];
        EXPECT_NEAR(disparity, pair.baseline * left.FocalLengthPixels() / (pair.height - 1.0), 1.5)
            << pair.rig;
    }
}

TEST_F(RasterDelegateTest, MastcamZsNearInfraredFiltersSeeTheSunsBlackBody) {
    // Mastcam-Z's right camera (Bell et al. 2021, Table 3) through R1 (800 +/- 9 nm) and R6
    // (1022 +/- 19 nm) over grey ground under a black-body sun at Mars's irradiance: both see it,
    // R6 the dimmer, silicon nearly blind at 1 um despite its band twice as wide.
    const auto meanElectrons = [&](int filter) {
        Scene scene = FlatScene();
        scene.Set("/World/Sun", tokens::kLightIrradiance, 590.0);
        ernie::AuthorMastcamZ(scene, "/World/Camera", "/Render/Product", ernie::MastcamZEye::Right,
                              filter);
        scene.Set("/World/Camera", tokens::kSensorNoise, false);
        // The 32 x 32 image bins the detector's pixels about 37 x 37: a short exposure.
        scene.Set("/World/Camera", tokens::kExposureTime, 2e-6);
        // A narrow band's sensor numbers; it has no colour to show.
        scene.Set("/Render/Product", tokens::kOrderedVars,
                  std::vector<Path>{"/Render/Vars/Electrons", "/Render/Vars/RawDN"});
        RasterDelegate delegate(*context_);
        ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
        const auto vars = MapVars(outputs);
        double sum = 0.0;
        for (std::size_t row = 0; row < kVarSize; ++row) {
            for (std::size_t col = 0; col < kVarSize; ++col) {
                const float e = At<float>(vars.at("Electrons"), row, col);
                EXPECT_LT(e, 21818.0f) << "below the full well";
                sum += e;
            }
        }
        return sum / (kVarSize * kVarSize);
    };
    const double r1 = meanElectrons(1);
    const double r6 = meanElectrons(6);
    EXPECT_GT(r1, 100.0);
    EXPECT_GT(r6, 0.0);
    EXPECT_LT(r6, 0.5 * r1);
}

TEST_F(RasterDelegateTest, AFisheyeWiderThan180DegreesSeesPastItsHorizonOnCubeFaces) {
    // An equidistant fisheye 5 m over flat ground looking straight down, its corners 2 radians
    // (115 degrees) off axis, drawn on cube faces: the ground at 5 / cos(theta) for each pixel's
    // own ray, a cube 95 degrees off axis (above the horizon) where the lens puts it, and nothing
    // past the horizon elsewhere.
    Scene scene = FlatScene();
    scene.Set("/World/Camera", tokens::kTransform,
              ernie::LookAtTransform({0.0, 0.0, 5.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}));
    scene.Set("/World/Camera", tokens::kLensProjection, std::string(tokens::kFisheye));
    const double corner = std::sqrt(2.0) * 0.5 * kVarSize;  // pixels from the centre
    const double focalPixels = corner / 2.0;
    const double aperture = 20.0;
    scene.Set("/World/Camera", tokens::kVerticalAperture, aperture);
    scene.Set("/World/Camera", tokens::kHorizontalAperture, aperture);
    scene.Set("/World/Camera", tokens::kFocalLength, aperture * focalPixels / kVarSize);
    // 95 degrees off axis along the image's diagonal (camera x and y = world x and y here).
    const double theta = 95.0 * M_PI / 180.0;
    const glm::dvec3 towards(std::sin(theta) / std::sqrt(2.0), std::sin(theta) / std::sqrt(2.0),
                             -std::cos(theta));
    scene.Define("/World/Cube", tokens::kCube)
        .Set(tokens::kSize, 2.0)
        .Set(tokens::kTransform,
             glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 0.0, 5.0) + 12.0 * towards));
    const ernie::Camera camera = ernie::CameraOf(scene, "/World/Camera", kVarSize, kVarSize);
    ASSERT_TRUE(ernie::NeedsCubeFaces(camera));
    RasterDelegate delegate(*context_);
    ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
    const auto vars = MapVars(outputs);
    const glm::dvec2 centre = camera.PrincipalPoint();
    const double f = camera.FocalLengthPixels();
    glm::dvec2 cubeSum(0.0);
    int cubeHits = 0;
    int ground = 0;
    for (std::size_t row = 0; row < kVarSize; ++row) {
        for (std::size_t col = 0; col < kVarSize; ++col) {
            const glm::dvec2 distorted((col + 0.5 - centre.x) / f, (centre.y - (row + 0.5)) / f);
            const glm::dvec3 ray = ernie::LensRay(camera.lens, distorted);
            const double offAxis = std::acos(-ray.z);
            const Path prim = delegate.PrimPath(At<std::uint32_t>(vars.at("PrimId"), row, col));
            if (prim == Path("/World/Cube")) {
                cubeSum += glm::dvec2(col + 0.5, row + 0.5);
                ++cubeHits;
                continue;
            }
            if (offAxis < 60.0 * M_PI / 180.0) {
                ASSERT_EQ(prim, Path("/World/Terrain")) << row << ", " << col;
                EXPECT_NEAR(At<float>(vars.at("DistanceToCamera"), row, col), 5.0 / -ray.z,
                            0.05 * 5.0 / -ray.z)
                    << row << ", " << col;
                ++ground;
            } else if (offAxis > 92.0 * M_PI / 180.0) {
                EXPECT_EQ(At<std::uint32_t>(vars.at("PrimId"), row, col), 0u)
                    << "past the horizon at " << row << ", " << col;
            }
        }
    }
    EXPECT_GT(ground, 100);
    ASSERT_GT(cubeHits, 0) << "the cube above the horizon";
    // Where the lens puts its centre: radius theta f along the diagonal (up and right).
    const glm::dvec2 expected = centre + theta * f * glm::dvec2(1.0, -1.0) / std::sqrt(2.0);
    EXPECT_LT(glm::distance(cubeSum / double(cubeHits), expected), 1.5);
}

TEST_F(RasterDelegateTest, CahvoreCamerasSeeThroughTheirRayTablesExactly) {
    // Perseverance's Navcam (58 degrees off axis, one pinhole raster) and Hazcam (79, cube
    // faces), scaled to a quarter of their 5120 x 3840 (H and V scaled, image coordinates with
    // them), each rendered through its CAHVORE model's ray table: spheres along chosen pixels'
    // model rays land on those pixels, where the fitted lens alone is off by a pixel.
    const auto models = ernie::ReadCahvoreModelsFile(
        (std::filesystem::path(ERNIE_TEST_FIXTURES) / "cahvore" / "m2020_camera_models.json")
            .string());
    for (const char* name : {"NAVL", "HAZFLA"}) {
        ernie::CahvoreModel model;
        for (const ernie::NamedCahvore& named : models) {
            model = named.name == name ? named.model : model;
        }
        constexpr double kScale = 0.25;
        model.h *= kScale;
        model.v *= kScale;
        model.width = static_cast<std::uint32_t>(model.width * kScale);
        model.height = static_cast<std::uint32_t>(model.height * kScale);
        const ernie::CahvoreModel::ErnieCamera fit = model.ToCamera(6.4 / kScale);
        const ernie::LensRayTable table = model.RayTable(4);

        Scene scene = FlatScene();
        scene.Remove("/World/Terrain");
        scene.Set("/Render/Settings", tokens::kResolution,
                  glm::ivec2(static_cast<int>(model.width), static_cast<int>(model.height)));
        scene.Set("/Render/Product", tokens::kOrderedVars,
                  std::vector<Path>{"/Render/Vars/PrimId"});
        ernie::Camera camera = fit.camera;
        camera.bands.count = 3;
        ernie::SetCamera(scene, "/World/Camera", camera);
        scene.Set("/World/Camera", tokens::kLensProjection, std::string(tokens::kFisheye));
        scene.Set("/World/Camera", tokens::kLensFisheye,
                  std::vector<float>{float(camera.lens.fisheye[0]), float(camera.lens.fisheye[1]),
                                     float(camera.lens.fisheye[2]), float(camera.lens.fisheye[3])});
        std::vector<float> rays;
        for (const glm::vec3& ray : table.rays) {
            rays.insert(rays.end(), {ray.x, ray.y, ray.z});
        }
        scene.Write("/World/Camera", tokens::kLensRays,
                    Tensor::Copy(std::span<const float>(rays),
                                 {std::int64_t(table.rows), std::int64_t(table.columns), 3}));
        scene.Set("/World/Camera", tokens::kLensRaysStep, 4);
        // Spheres 30 m along the model's rays of pixels across the image, 1.5 pixels in radius:
        // far enough that the CAHVORE entrance pupil's movement (up to 12.5 mm, which Ernie's
        // central camera leaves out) moves them by under a fifth of a pixel.
        constexpr double kDistance = 30.0;
        // At pixel centres, so the nearest raster texels the ids come from fall evenly about them.
        const std::vector<glm::dvec2> pixels = {{0.5 * model.width + 0.5, 0.5 * model.height + 0.5},
                                                {40.5, 30.5},
                                                {model.width - 40.5, 50.5},
                                                {60.5, model.height - 40.5},
                                                {model.width - 30.5, model.height - 30.5},
                                                {0.5 * model.width + 0.5, 25.5}};
        const double pixelAngle = 1.0 / fit.camera.FocalLengthPixels();
        for (std::size_t i = 0; i < pixels.size(); ++i) {
            const ernie::CahvoreModel::Ray ray = model.ImageToRay(pixels[i] - 0.5);
            scene.Define(Path("/World/Ball" + std::to_string(i)), tokens::kSphere)
                .Set(tokens::kGeomRadius, kDistance * 1.5 * pixelAngle)
                .Set(tokens::kTransform,
                     glm::translate(glm::dmat4(1.0), ray.origin + kDistance * ray.direction));
        }
        const ernie::Camera rendered =
            ernie::CameraOf(scene, "/World/Camera", model.width, model.height);
        ASSERT_NE(rendered.lens.rays, nullptr);
        EXPECT_EQ(ernie::NeedsCubeFaces(rendered), std::string(name) == "HAZFLA");
        RasterDelegate delegate(*context_);
        ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
        const Tensor ids =
            outputs.at("/Render/Product").vars.at("PrimId").Map(ernie::MapDevice::Cpu).tensor;
        for (std::size_t i = 0; i < pixels.size(); ++i) {
            const Path ball("/World/Ball" + std::to_string(i));
            glm::dvec2 sum(0.0);
            int hits = 0;
            for (std::uint32_t row = 0; row < model.height; ++row) {
                for (std::uint32_t col = 0; col < model.width; ++col) {
                    if (delegate.PrimPath(ids.As<std::uint32_t>()[row * model.width + col]) ==
                        ball) {
                        sum += glm::dvec2(col + 0.5, row + 0.5);
                        ++hits;
                    }
                }
            }
            ASSERT_GT(hits, 0) << name << " " << ball.String();
            // PrimId comes from the nearest raster texel (within half a pixel); the fitted lens
            // alone is off by a pixel at the edges here.
            EXPECT_LT(glm::distance(sum / double(hits), pixels[i]), 0.6)
                << name << " " << ball.String() << " at " << pixels[i].x << ", " << pixels[i].y;
        }
    }
}

TEST_F(RasterDelegateTest, LateralColourScalesEachBandsImageAboutThePrincipalPoint) {
    // A lens whose image is 10% smaller at 400 nm and 10% larger at 700 than its reference: a
    // cube off the axis lands, in each band's spectral radiance, its offset from the principal
    // point times the band's mean scale.
    const auto centroids = [&](bool fringed) {
        Scene scene = FlatScene();
        scene.Remove("/World/Terrain");
        scene.Set("/Render/Settings", tokens::kResolution, glm::ivec2(96, 96));
        scene.Set("/Render/Product", tokens::kOrderedVars,
                  std::vector<Path>{"/Render/Vars/SpectralRadiance"});
        scene.Define("/World/Cube", tokens::kCube)
            .Set(tokens::kSize, 4.0)
            .Set(tokens::kTransform, glm::translate(glm::dmat4(1.0), glm::dvec3(9.0, 5.0, 2.0)));
        if (fringed) {
            scene.Set("/World/Camera", tokens::kLensChromaticScale,
                      std::vector<float>{400.0f, 0.9f, 700.0f, 1.1f});
        }
        RasterDelegate delegate(*context_);
        ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
        const Tensor spectral = outputs.at("/Render/Product")
                                    .vars.at("SpectralRadiance")
                                    .Map(ernie::MapDevice::Cpu)
                                    .tensor;
        const auto bands = static_cast<std::size_t>(spectral.Shape()[0]);
        const ernie::Camera camera = ernie::CameraOf(scene, "/World/Camera", 96, 96);
        std::vector<std::pair<glm::dvec2, double>> result;  // offset, the band's scale
        for (std::size_t b = 0; b < bands; ++b) {
            glm::dvec2 sum(0.0);
            double total = 0.0;
            for (std::size_t row = 0; row < 96; ++row) {
                for (std::size_t col = 0; col < 96; ++col) {
                    const double value = spectral.As<float>()[(b * 96 + row) * 96 + col];
                    sum += value * glm::dvec2(col + 0.5, row + 0.5);
                    total += value;
                }
            }
            const double low = camera.bands.Lower(static_cast<int>(b));
            const double scale =
                fringed ? camera.lens.chromaticScale.Average(low, low + camera.bands.Width()) : 1.0;
            result.emplace_back(sum / total - camera.PrincipalPoint(), scale);
        }
        return result;
    };
    const auto plain = centroids(false);
    const auto fringed = centroids(true);
    ASSERT_EQ(plain.size(), fringed.size());
    EXPECT_GT(glm::length(plain.front().first), 10.0) << "well off the axis";
    for (std::size_t b = 0; b < plain.size(); ++b) {
        EXPECT_NEAR(glm::distance(plain[b].first, plain.front().first), 0.0, 1e-3)
            << "every band alike without it";
        EXPECT_NEAR(glm::distance(fringed[b].first, fringed[b].second * plain[b].first), 0.0, 0.15)
            << "band " << b << ", scale " << fringed[b].second;
    }
    EXPECT_LT(fringed.front().second, 0.95);
    EXPECT_GT(fringed.back().second, 1.05);
}

TEST_F(RasterDelegateTest, AxialColourDefocusesTheBandsAwayFromTheReference) {
    // A fast lens (f/1.4) focused on the ground 100 m below, its focal length 4 mm longer at 400
    // and 700 nm than through the greens: blurred by the defocus filter into discs about six
    // pixels across, red and blue lose the cubes' sharp edges and green keeps them.
    const auto edges = [&](bool coloured) {
        Scene scene = FlatScene();
        scene.Set("/Render/Product", tokens::kOrderedVars,
                  std::vector<Path>{"/Render/Vars/HdrColor"});
        for (int i = 0; i < 3; ++i) {
            scene.Define(Path("/World/Cube" + std::to_string(i)), tokens::kCube)
                .Set(tokens::kSize, 6.0)
                .Set(tokens::kTransform,
                     glm::translate(glm::dmat4(1.0), glm::dvec3(10.0 * i - 10.0, 2.0, 0.0)));
        }
        scene.Set("/World/Camera", tokens::kFStop, 1.4);
        scene.Set("/World/Camera", tokens::kFocusDistance, 100.0);
        if (coloured) {
            scene.Set("/World/Camera", tokens::kLensChromaticFocus,
                      std::vector<float>{400.0f, 4.0f, 480.0f, 0.0f, 580.0f, 0.0f, 700.0f, 4.0f});
        }
        const Path defocus("/Render/Filters/Defocus");
        scene.Define(defocus, tokens::kSensorFilter)
            .Set(tokens::kFilterType, std::string("defocus"))
            .Set("inputs:maxRadius", 4.0);
        const Path cie("/Render/Filters/Cie");
        scene.Define(cie, tokens::kSensorFilter).Set(tokens::kFilterType, std::string("cie"));
        scene.Set("/Render/Product", tokens::kFilters, std::vector<Path>{defocus, cie});
        RasterDelegate delegate(*context_);
        ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
        const Tensor hdr =
            outputs.at("/Render/Product").vars.at("HdrColor").Map(ernie::MapDevice::Cpu).tensor;
        // Each channel's total variation along the rows.
        std::array<double, 3> variation{};
        for (std::size_t row = 0; row < kVarSize; ++row) {
            for (std::size_t col = 1; col < kVarSize; ++col) {
                for (std::size_t c = 0; c < 3; ++c) {
                    variation[c] +=
                        std::abs(At<float>(hdr, row, col, c) - At<float>(hdr, row, col - 1, c));
                }
            }
        }
        return variation;
    };
    const std::array<double, 3> sharp = edges(false);
    const std::array<double, 3> coloured = edges(true);
    EXPECT_LT(coloured[0], 0.8 * sharp[0]) << "red blurs";
    EXPECT_LT(coloured[2], 0.8 * sharp[2]) << "blue blurs";
    EXPECT_GT(coloured[1], 0.85 * sharp[1]) << "green stays sharp";
}

TEST_F(RasterDelegateTest, FisheyeSeesFieldAnglesInProportionToRadius) {
    // 50 above flat ground looking straight down: a ray at field angle theta meets it
    // 50 / cos(theta) away. Equidistant, a pixel's theta is its radius over the focal length.
    Scene scene = FlatScene();
    scene.Set("/World/Camera", tokens::kTransform,
              ernie::LookAtTransform({0.0, 0.0, 50.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}));
    scene.Set("/World/Camera", tokens::kFocalLength, 12.0f);
    scene.Set("/World/Camera", tokens::kLensProjection, std::string(tokens::kFisheye));
    RasterDelegate delegate(*context_);
    ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
    const auto vars = MapVars(outputs);
    const ernie::Camera camera = ernie::CameraOf(scene, "/World/Camera", kVarSize, kVarSize);
    const std::size_t c = kVarSize / 2;
    for (const std::size_t col : {std::size_t{0}, std::size_t{4}, c}) {
        const double x = (col + 0.5 - camera.PrincipalPoint().x) / camera.FocalLengthPixels();
        const double y = (camera.PrincipalPoint().y - (c + 0.5)) / camera.FocalLengthPixels();
        const double theta = std::hypot(x, y);
        // The raster image is sampled at its nearest texel, a pixel of a 32-pixel image.
        EXPECT_NEAR(At<float>(vars.at("DistanceToCamera"), c, col), 50.0 / std::cos(theta), 0.8)
            << "column " << col;
    }
    const double edge = std::abs((0.5 - camera.PrincipalPoint().x) / camera.FocalLengthPixels());
    EXPECT_GT(50.0 / std::cos(edge) - 50.0 * std::sqrt(1.0 + edge * edge), 1.6)
        << "a pinhole would see the edge nearer";
}

namespace {

/// The Electrons var of `product` in `outputs`, on the CPU.
std::vector<float> ElectronsOf(ernie::RenderProductSetOutputs& outputs, const Path& product) {
    const Tensor tensor =
        outputs.at(product).vars.at("Electrons").Map(ernie::MapDevice::Cpu).tensor;
    const auto* data = static_cast<const float*>(tensor.Data());
    return {data, data + tensor.Elements()};
}

/// FlatScene with sensor noise, exposed well below the full well, and a second product
/// `/Render/Other` of the same camera.
Scene NoisyScene() {
    Scene scene = FlatScene();
    scene.Set("/World/Camera", tokens::kSensorNoise, true);
    scene.Set("/World/Camera", tokens::kExposureTime, 1e-6);
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

TEST_F(RasterDelegateTest, MonochromeSensorsMakeGreyPhotographicImages) {
    RasterDelegate delegate(*context_);
    Scene scene = MeshScene();
    const glm::vec3 quad(-12.0f, 0.0f, 0.0f);
    const Tensor colour = delegate.Render(scene);
    const std::uint8_t* red = TexelAt(scene, colour, quad);
    EXPECT_GT(red[0], red[1] + 50) << "the red quad through an RGB sensor";
    scene.Set("/World/Camera", tokens::kSensorType, std::string("mono"));
    const Tensor grey = delegate.Render(scene);
    const std::uint8_t* mono = TexelAt(scene, grey, quad);
    EXPECT_EQ(mono[0], mono[1]);
    EXPECT_EQ(mono[1], mono[2]);
    EXPECT_GT(mono[0], 10) << "silicon sees red light";
    const std::uint8_t* rock = TexelAt(scene, grey, {12.0f, 0.0f, 0.0f});
    EXPECT_EQ(rock[0], rock[2]);
}
