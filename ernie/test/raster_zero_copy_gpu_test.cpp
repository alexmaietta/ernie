// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/compute_interop.h"
#include "ernie/raster/delegate.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/terrain_schemas.h"
#include "ernie/scene/tokens.h"
#include "support/delegate_scenes.h"
#include "support/test_raster.h"
#include "support/vulkan_fixture.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using ernie::Path;
using ernie::RasterDelegate;
using ernie::Scene;
using ernie::Tensor;
using namespace ernie::testing;
namespace tokens = ernie::tokens;

namespace {

/// The number of bytes of `a` and `b` differing by more than `tolerance`.
std::size_t Differing(const Tensor& a, const Tensor& b, int tolerance = 0) {
    std::size_t count = 0;
    for (std::int64_t i = 0; i < a.Elements(); ++i) {
        count += std::abs(int(a.As<std::uint8_t>()[i]) - int(b.As<std::uint8_t>()[i])) > tolerance
                     ? 1u
                     : 0u;
    }
    return count;
}

/// `values` downloaded from a mapped array.
std::vector<float> Floats(ernie::ComputeInterop& cuda, const Tensor& device) {
    const Tensor host = cuda.Download(device);
    return {host.As<float>(), host.As<float>() + host.Elements()};
}

/// Writes `values` into mapped array `view` on the legacy stream, as a kernel would.
void WriteInto(ernie::ComputeInterop& cuda, const Tensor& view, const std::vector<float>& values) {
    cuda.Copy(cuda.Upload(Tensor::Copy<float>(values, view.Shape())), view);
}

}  // namespace

TEST_F(RasterDelegateTest, MappedMeshInstancerAndPointsArraysDrawWithoutCopies) {
    RasterDelegate delegate(*context_);
    std::string why;
    ernie::ComputeInterop* cuda = delegate.Compute(&why);
    if (cuda == nullptr) {
        GTEST_SKIP() << "No CUDA interop: " << why;
    }
    Scene scene = MeshScene();
    const std::array<float, 9> grains{-4, -12, 0, 0, -12, 0, 4, -12, 0};
    ernie::Points::Define(scene, "/World/Grains")
        .SetPoints(Tensor::Copy<float>(grains, {3, 3}))
        .SetWidths(2.0);
    scene.Set("/World/Quad", tokens::kExtent,
              std::vector<float>{-14.0f, -4.0f, -1.0f, -10.0f, 4.0f, 1.0f});
    scene.Set("/World/Rocks", tokens::kExtent,
              std::vector<float>{-2.0f, -12.0f, -1.0f, 14.0f, 12.0f, 1.0f});
    scene.Set("/World/Grains", tokens::kExtent,
              std::vector<float>{-6.0f, -14.0f, -2.0f, 6.0f, -10.0f, 2.0f});
    delegate.Render(scene);

    // Each map holds the values as they are, and the scene holds the map.
    Tensor quad = delegate.MapArray(scene, "/World/Quad", tokens::kPoints);
    EXPECT_EQ(quad.Location().kind, ernie::DeviceKind::Cuda);
    EXPECT_EQ(Floats(*cuda, quad), (std::vector<float>{-2, -2, 0, 2, -2, 0, 2, 2, 0, -2, 2, 0}));
    EXPECT_EQ(scene.Read("/World/Quad", tokens::kPoints)->Data(), quad.Data());
    Tensor rocks = delegate.MapArray(scene, "/World/Rocks", tokens::kPositions);
    EXPECT_EQ(Floats(*cuda, rocks), (std::vector<float>{12, -10, 0, 12, 0, 0, 12, 10, 0}));
    Tensor points = delegate.MapArray(scene, "/World/Grains", tokens::kPoints);
    EXPECT_EQ(Floats(*cuda, points), std::vector<float>(grains.begin(), grains.end()));

    // New values written in place draw as the host draws them written as arrays.
    const std::vector<float> wideQuad{-3, -3, 0, 3, -3, 0, 3, 3, 0, -3, 3, 0};
    const std::vector<float> movedRocks{12, -10, 0, 6, 4, 0, 12, 10, 0};
    const std::vector<float> raisedGrains{-4, -12, 1, 0, -11, 0, 4, -12, -1};
    WriteInto(*cuda, quad, wideQuad);
    WriteInto(*cuda, rocks, movedRocks);
    WriteInto(*cuda, points, raisedGrains);
    scene.Unmap("/World/Quad", tokens::kPoints);
    scene.Unmap("/World/Rocks", tokens::kPositions);
    scene.Unmap("/World/Grains", tokens::kPoints);
    const Tensor image = delegate.Render(scene);

    Scene host = MeshScene();
    ernie::Points::Define(host, "/World/Grains")
        .SetPoints(Tensor::Copy<float>(raisedGrains, {3, 3}))
        .SetWidths(2.0);
    for (const auto& [path, extent] :
         {std::pair{"/World/Quad", std::vector<float>{-14.0f, -4.0f, -1.0f, -10.0f, 4.0f, 1.0f}},
          std::pair{"/World/Rocks", std::vector<float>{-2.0f, -12.0f, -1.0f, 14.0f, 12.0f, 1.0f}},
          std::pair{"/World/Grains",
                    std::vector<float>{-6.0f, -14.0f, -2.0f, 6.0f, -10.0f, 2.0f}}}) {
        host.Set(path, tokens::kExtent, extent);
    }
    ernie::Mesh(host, "/World/Quad").SetPoints(cuda->Upload(Tensor::Copy<float>(wideQuad, {4, 3})));
    ernie::PointInstancer(host, "/World/Rocks")
        .SetPositions(cuda->Upload(Tensor::Copy<float>(movedRocks, {3, 3})));
    ernie::Points(host, "/World/Grains")
        .SetPoints(cuda->Upload(Tensor::Copy<float>(raisedGrains, {3, 3})));
    RasterDelegate reference(*context_);
    reference.Compute();
    EXPECT_EQ(Differing(image, reference.Render(host)), 0u);

    // The next map is the other half, holding the values the last one wrote.
    const Tensor again = delegate.MapArray(scene, "/World/Quad", tokens::kPoints);
    EXPECT_NE(again.Data(), quad.Data());
    EXPECT_EQ(Floats(*cuda, again), wideQuad);
    scene.Unmap("/World/Quad", tokens::kPoints);
    EXPECT_EQ(Differing(delegate.Render(scene), image), 0u);

    EXPECT_THROW(delegate.MapArray(scene, "/World/Quad", tokens::kNormals), std::invalid_argument);
    EXPECT_THROW(delegate.MapArray(scene, "/World/Nothing", tokens::kPoints),
                 std::invalid_argument);
}

TEST_F(RasterDelegateTest, MappedHeightsDrawWhereDirtyWithoutCopies) {
    RasterDelegate delegate(*context_);
    std::string why;
    ernie::ComputeInterop* cuda = delegate.Compute(&why);
    if (cuda == nullptr) {
        GTEST_SKIP() << "No CUDA interop: " << why;
    }
    const TestRaster raster = RollingTerrain(65, 65, 2.0);
    Scene scene = TerrainScene(raster);
    ernie::SetCamera(scene, "/World/Camera",
                     [] {
                         ernie::CameraParams camera;
                         camera.width = kSize;
                         camera.height = kSize;
                         camera.transform = ernie::LookAtTransform(
                             {0.0, -150.0, 120.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0});
                         return camera;
                     }());
    EXPECT_THROW(delegate.MapArray(scene, "/World/Terrain", tokens::kHeights),
                 std::invalid_argument)
        << "heights on the host";
    ernie::Terrain(scene, "/World/Terrain").SetHeights(cuda->Upload(HeightsOf(raster)));
    const Tensor before = delegate.Render(scene);

    Tensor heights = delegate.MapArray(scene, "/World/Terrain", tokens::kHeights);
    EXPECT_EQ(heights.Shape(), (std::vector<std::int64_t>{65, 65}));
    std::vector<float> values = Floats(*cuda, heights);
    EXPECT_EQ(values, raster.values);
    for (std::size_t row = 20; row < 40; ++row) {
        for (std::size_t col = 20; col < 40; ++col) {
            values[row * 65 + col] += 30.0f;
        }
    }
    WriteInto(*cuda, heights, values);
    scene.Set("/World/Terrain", tokens::kHeightsDirty, glm::vec4(20.0f, 20.0f, 20.0f, 20.0f));
    scene.Unmap("/World/Terrain", tokens::kHeights);
    // The LOD bounds and the colormap's height range come back from the GPU two frames later.
    delegate.Render(scene);
    delegate.Render(scene);
    const Tensor after = delegate.Render(scene);
    EXPECT_GT(Differing(after, before, 2), 50u);

    TestRaster raised = raster;
    raised.values = values;
    Scene host = TerrainScene(raised);
    ernie::SetCamera(host, "/World/Camera",
                     ernie::CameraParamsOf(scene, "/World/Camera", kSize, kSize));
    RasterDelegate reference(*context_);
    EXPECT_LT(Differing(after, reference.Render(host), 2), 10u);
}
