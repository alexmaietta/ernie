// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/compute_interop.h"
#include "ernie/geometry/lod_quadtree.h"
#include "ernie/raster/delegate.h"
#include "ernie/scene/orbit_controller.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"
#include "support/delegate_scenes.h"
#include "support/test_raster.h"
#include "support/vulkan_fixture.h"

#include <glm/gtc/packing.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
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
std::size_t Differing(const Tensor& a, const Tensor& b, int tolerance) {
    std::size_t count = 0;
    for (std::int64_t i = 0; i < a.Elements(); ++i) {
        count += std::abs(int(a.As<std::uint8_t>()[i]) - int(b.As<std::uint8_t>()[i])) > tolerance
                     ? 1u
                     : 0u;
    }
    return count;
}

/// TerrainScene of `raster` seen from 0.8 rad above the horizon, framed on the terrain.
Scene FramedTerrain(const TestRaster& raster) {
    Scene scene = TerrainScene(raster);
    ernie::OrbitController orbit;
    orbit.pitch = 0.8f;
    const glm::vec3 low(
        static_cast<float>(raster.originX),
        static_cast<float>(raster.originY - static_cast<double>(raster.height) * raster.cellSize),
        -60.0f);
    const glm::vec3 high(
        static_cast<float>(raster.originX + static_cast<double>(raster.width) * raster.cellSize),
        static_cast<float>(raster.originY), 60.0f);
    orbit.Frame(low, high);
    ernie::SetCamera(scene, "/World/Camera", orbit.MakeCamera(kSize, kSize));
    return scene;
}

/// `raster` with a bump of height `rise` over cells [col, col + size) x [row, row + size).
TestRaster Bumped(TestRaster raster, std::size_t col, std::size_t row, std::size_t size,
                  float rise) {
    for (std::size_t r = row; r < row + size; ++r) {
        for (std::size_t c = col; c < col + size; ++c) {
            raster.values[r * raster.width + c] += rise;
        }
    }
    return raster;
}

/// Mip `mip` of an R16G16 float image, read back as floats.
std::vector<float> GradientsOf(const ernie::VulkanContext& context, const ernie::GpuImage& source,
                               std::uint32_t mip) {
    auto& image = const_cast<ernie::GpuImage&>(source);  // only moved between layouts
    const std::uint32_t w = std::max(image.Desc().width >> mip, 1u);
    const std::uint32_t h = std::max(image.Desc().height >> mip, 1u);
    ernie::GpuBuffer buffer(context, VkDeviceSize{w} * h * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            ernie::BufferMemory::Readback);
    const VkImageLayout layout = image.Layout();
    context.Submit([&](VkCommandBuffer commands) {
        image.Transition(commands, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0, 1};
        region.imageExtent = {w, h, 1};
        vkCmdCopyImageToBuffer(commands, image.Handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               buffer.Handle(), 1, &region);
        image.Transition(commands, layout);
    });
    std::vector<std::uint32_t> packed(std::size_t{w} * h);
    buffer.Read(std::as_writable_bytes(std::span(packed)));
    std::vector<float> values;
    for (const std::uint32_t p : packed) {
        const glm::vec2 v = glm::unpackHalf2x16(p);
        values.insert(values.end(), {v.x, v.y});
    }
    return values;
}

/// The largest difference between two delegates' terrain gradients over every mip, relative to
/// gradients above 1.
float GradientError(const ernie::VulkanContext& context, const RasterDelegate& a,
                    const RasterDelegate& b) {
    float worst = 0.0f;
    for (std::uint32_t mip = 0; mip < a.Terrain().Gradients()->Desc().mips; ++mip) {
        const std::vector<float> x = GradientsOf(context, *a.Terrain().Gradients(), mip);
        const std::vector<float> y = GradientsOf(context, *b.Terrain().Gradients(), mip);
        for (std::size_t i = 0; i < x.size(); ++i) {
            worst = std::max(worst, std::abs(x[i] - y[i]) / std::max(1.0f, std::abs(y[i])));
        }
    }
    return worst;
}

}  // namespace

TEST(LodQuadtree, LeavesFromElsewhereBoundAsTheRaster) {
    const TestRaster raster = RollingTerrain(200, 150, 2.0);
    const ernie::LodQuadtree built(raster.View(), 16);
    ernie::LodQuadtree later(200, 150, 2.0, 16);
    EXPECT_GT(later.HeightRange().x, later.HeightRange().y) << "no bounds yet";
    const std::uint32_t side = later.LeavesPerSide();
    std::vector<glm::vec2> leaves;
    for (std::uint32_t y = 0; y < side; ++y) {
        for (std::uint32_t x = 0; x < side; ++x) {
            const auto [low, high] = built.NodeBounds(0, x, y, 1.0f);
            leaves.emplace_back(low.z, high.z);
        }
    }
    later.SetLeaves(leaves);
    for (std::uint32_t level = 0; level < built.Levels(); ++level) {
        const std::uint32_t nodes = side >> level;
        for (std::uint32_t y = 0; y < nodes; ++y) {
            for (std::uint32_t x = 0; x < nodes; ++x) {
                const auto a = built.NodeBounds(level, x, y, 1.0f);
                const auto b = later.NodeBounds(level, x, y, 1.0f);
                EXPECT_EQ(a.first.z, b.first.z);
                EXPECT_EQ(a.second.z, b.second.z);
            }
        }
    }
    EXPECT_EQ(built.MinFirstRange(1.0f, 0.67f), later.MinFirstRange(1.0f, 0.67f));
    EXPECT_THROW(later.SetLeaves(std::span(leaves).subspan(1)), std::invalid_argument);
}

TEST_F(RasterDelegateTest, HeightsFromCudaDrawAsTheHostsAndUpdateWhereDirty) {
    RasterDelegate delegate(*context_);
    std::string why;
    ernie::ComputeInterop* cuda = delegate.Compute(&why);
    if (cuda == nullptr) {
        GTEST_SKIP() << "No CUDA interop: " << why;
    }
    const TestRaster raster = RollingTerrain(161, 129, 2.0);
    RasterDelegate reference(*context_);
    Scene host = FramedTerrain(raster);
    const Tensor expected = reference.Render(host);

    Scene device = FramedTerrain(raster);
    device.Write("/World/Terrain", tokens::kHeights, cuda->Upload(HeightsOf(raster)));
    const Tensor image = delegate.Render(device);
    ASSERT_EQ(image.Elements(), expected.Elements());
    // The gradient mips the GPU builds differ from the host's in half-float rounding only.
    EXPECT_LT(GradientError(*context_, delegate, reference), 2e-3f);
    EXPECT_LT(Differing(image, expected, 2), 10u) << "CUDA heights draw as the host's";

    // A bump written on the GPU with its dirty rectangle, as the host draws the bumped raster.
    const TestRaster bumped = Bumped(raster, 60, 40, 30, 25.0f);
    device.Set("/World/Terrain", tokens::kHeightsDirty, glm::vec4(60.0f, 40.0f, 30.0f, 30.0f));
    device.Write("/World/Terrain", tokens::kHeights, cuda->Upload(HeightsOf(bumped)));
    host.Write("/World/Terrain", tokens::kHeights, HeightsOf(bumped));
    const Tensor bumpedExpected = reference.Render(host);
    EXPECT_GT(Differing(bumpedExpected, expected, 2), 50u) << "the bump shows";
    delegate.Render(device);
    delegate.Render(device);  // its LOD bounds back on the host
    EXPECT_LT(Differing(delegate.Render(device), bumpedExpected, 2), 10u);
    EXPECT_LT(GradientError(*context_, delegate, reference), 2e-3f) << "updated where dirty";
    EXPECT_FLOAT_EQ(delegate.Terrain().Bounds(1.0f).second.z,
                    reference.Terrain().Bounds(1.0f).second.z);

    device.Set("/World/Terrain", tokens::kHeightsDirty, glm::vec4(150.0f, 0.0f, 30.0f, 1.0f));
    device.Write("/World/Terrain", tokens::kHeights, cuda->Upload(HeightsOf(bumped)));
    EXPECT_THROW(delegate.Render(device), std::invalid_argument) << "outside the raster";
}
