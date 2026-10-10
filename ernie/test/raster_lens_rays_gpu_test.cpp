#include "ernie/device/buffer.h"
#include "ernie/device/pipeline.h"
#include "ernie/device/shader_compiler.h"
#include "ernie/kernel/embedded_shaders.h"
#include "ernie/scene/wide_view.h"
#include "support/vulkan_fixture.h"

#include <glm/glm.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <span>
#include <vector>

namespace {

// For each case (sensor x, y): the fisheye's and Brown-Conrady's rays (lens.slang), and the
// fisheye ray's cube face and its plane coordinates (cube_faces.slang).
constexpr char kHarness[] = R"(import lens;
import cube_faces;

[[vk::binding(0, 0)]] StructuredBuffer<float4> cases;
[[vk::binding(1, 0)]] StructuredBuffer<float4> params;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> results;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint count;
    uint stride;
    cases.GetDimensions(count, stride);
    if (id.x >= count) {
        return;
    }
    let sensor = cases[id.x].xy;
    let fisheye = Fisheye.rayOf(sensor, params[0], 0.0);
    let brown = BrownConrady.rayOf(0.25 * sensor, params[1], params[2].x);
    let face = cubeFaceOf(fisheye);
    let plane = cubeFacePlane(face, fisheye);
    results[3 * id.x] = float4(fisheye, 0.0);
    results[3 * id.x + 1] = float4(brown, 0.0);
    results[3 * id.x + 2] = float4(float(face), plane, 0.0);
}
)";

class LensRaysGpuTest : public ernie::testing::VulkanTest {
  protected:
    ernie::ShaderCompiler compiler_;
};

}  // namespace

TEST_F(LensRaysGpuTest, TheSensorKernelsRaysAndFacesMatchTheCpus) {
    // A fisheye reaching 120 degrees off axis, and a barrel-distorted perspective lens.
    ernie::LensModel fisheye;
    fisheye.projection = ernie::LensProjection::Fisheye;
    fisheye.fisheye = {0.02, -0.004, 0.0, 0.0};
    ernie::LensModel brown;
    brown.k1 = -0.08;
    brown.k2 = 0.01;
    brown.p1 = 0.002;
    brown.p2 = -0.001;
    std::vector<float> cases;
    std::vector<glm::dvec2> points;
    std::mt19937 random(3);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    while (points.size() < 512) {
        const glm::dvec2 p(unit(random), unit(random));
        if (glm::length(p) <= 1.0) {
            points.push_back(2.0 * p);  // fisheye radii to 2: field angles near 115 degrees
        }
    }
    for (const glm::dvec2 p : points) {
        cases.insert(cases.end(), {float(p.x), float(p.y), 0.0f, 0.0f});
    }
    const std::vector<float> params = {float(fisheye.fisheye[0]),
                                       float(fisheye.fisheye[1]),
                                       float(fisheye.fisheye[2]),
                                       float(fisheye.fisheye[3]),
                                       float(brown.k1),
                                       float(brown.k2),
                                       float(brown.k3),
                                       float(brown.p1),
                                       float(brown.p2),
                                       0.0f,
                                       0.0f,
                                       0.0f};
    const std::vector<ernie::ShaderModuleSource> modules = {
        {"lens", ernie::shaders::kLensSlang},
        {"cube_faces", ernie::shaders::kCubeFacesSlang},
        {"harness", kHarness}};
    const ernie::ComputePipeline pipeline(*context_, compiler_.CompileSpirv(modules, "main"), 3);
    ernie::GpuBuffer caseBuffer(*context_, cases.size() * sizeof(float),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, ernie::BufferMemory::Device);
    ernie::GpuBuffer paramBuffer(*context_, params.size() * sizeof(float),
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, ernie::BufferMemory::Device);
    ernie::GpuBuffer resultBuffer(*context_, points.size() * 12 * sizeof(float),
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                  ernie::BufferMemory::Readback);
    caseBuffer.Write(std::as_bytes(std::span(cases)));
    paramBuffer.Write(std::as_bytes(std::span(params)));
    const std::array<const ernie::GpuBuffer*, 3> buffers = {&caseBuffer, &paramBuffer,
                                                            &resultBuffer};
    context_->Submit([&](VkCommandBuffer commands) {
        pipeline.Dispatch(commands, buffers, static_cast<std::uint32_t>((points.size() + 63) / 64));
    });
    std::vector<float> raw(points.size() * 12);
    resultBuffer.Read(std::as_writable_bytes(std::span(raw)));
    for (std::size_t i = 0; i < points.size(); ++i) {
        const float* r = &raw[12 * i];
        const glm::dvec3 fisheyeRay = ernie::LensRay(fisheye, points[i]);
        EXPECT_NEAR(glm::distance(glm::dvec3(r[0], r[1], r[2]), fisheyeRay), 0.0, 1e-5) << i;
        // Brown-Conrady over a perspective lens's moderate field, radii to 0.5.
        const glm::dvec3 brownRay = ernie::LensRay(brown, 0.25 * points[i]);
        EXPECT_NEAR(glm::distance(glm::dvec3(r[4], r[5], r[6]), brownRay), 0.0, 1e-5) << i;
        // The face, but for rays all but on a seam, where float and double may pick either.
        std::array<double, 5> forward = {-fisheyeRay.z, fisheyeRay.x, -fisheyeRay.x, fisheyeRay.y,
                                         -fisheyeRay.y};
        std::sort(forward.begin(), forward.end());
        if (forward[4] - forward[3] < 1e-4) {
            continue;
        }
        const ernie::CubeFace face = ernie::CubeFaceOf(fisheyeRay);
        const glm::dvec2 plane = ernie::CubeFacePlane(face, fisheyeRay);
        EXPECT_EQ(static_cast<int>(r[8]), static_cast<int>(face)) << i;
        EXPECT_NEAR(r[9], plane.x, 1e-4) << i;
        EXPECT_NEAR(r[10], plane.y, 1e-4) << i;
    }
}
