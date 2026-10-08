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
