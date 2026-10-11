// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/compute_interop.h"
#include "ernie/geometry/primitives.h"
#include "ernie/raster/delegate.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"
#include "support/delegate_scenes.h"
#include "support/vulkan_fixture.h"

#include <glm/glm.hpp>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using ernie::MeshData;
using ernie::Path;
using ernie::RasterDelegate;
using ernie::Scene;
using ernie::Tensor;
using namespace ernie::testing;
namespace tokens = ernie::tokens;

namespace {

/// MeshScene with a sphere of per-vertex colours and normals at
/// /World/Ball.
Scene BallScene(const MeshData& ball) {
    Scene scene = MeshScene();
    const auto n = static_cast<std::int64_t>(ball.points.size() / 3);
    const auto triangles =
        static_cast<std::int64_t>(ball.indices.size() / 3);
    std::vector<std::int32_t> counts(
        static_cast<std::size_t>(triangles), 3);
    std::vector<std::int32_t> indices(ball.indices.begin(),
                                      ball.indices.end());
    std::vector<float> colors;
    for (std::size_t i = 0; i < ball.points.size(); i += 3) {
        const float t = 0.5f + ball.points[i + 2] / 16.0f;
        colors.insert(colors.end(), {t, 0.2f, 1.0f - t});
    }
    ernie::Mesh::Define(scene, "/World/Ball")
        .SetPoints(Tensor::Copy<float>(ball.points, {n, 3}))
        .SetNormals(Tensor::Copy<float>(ball.normals, {n, 3}))
        .SetFaceVertexCounts(
            Tensor::Copy<std::int32_t>(counts, {triangles}))
        .SetFaceVertexIndices(Tensor::Copy<std::int32_t>(
            indices, {static_cast<std::int64_t>(indices.size())}));
    // Colours per vertex are an array, not the schema's single
    // displayColor.
    scene.Write("/World/Ball", tokens::kDisplayColor,
                Tensor::Copy<float>(colors, {n, 3}));
    return scene;
}

/// Area-weighted vertex normals of `mesh`, as mesh_normals.slang
/// computes them.
std::vector<float> AreaWeightedNormals(const MeshData& mesh) {
    std::vector<glm::vec3> sums(mesh.points.size() / 3,
                                glm::vec3(0.0f));
    const auto at = [&](std::uint32_t i) {
        return glm::vec3(mesh.points[3 * i], mesh.points[3 * i + 1],
                         mesh.points[3 * i + 2]);
    };
    for (std::size_t t = 0; t < mesh.indices.size(); t += 3) {
        const std::uint32_t a = mesh.indices[t];
        const std::uint32_t b = mesh.indices[t + 1];
        const std::uint32_t c = mesh.indices[t + 2];
        const glm::vec3 n = glm::cross(at(b) - at(a), at(c) - at(a));
        sums[a] += n;
        sums[b] += n;
        sums[c] += n;
    }
    std::vector<float> normals;
    for (const glm::vec3& sum : sums) {
        const glm::vec3 n = glm::length(sum) > 0.0f
                                ? glm::normalize(sum)
                                : glm::vec3(0.0f);
        normals.insert(normals.end(), {n.x, n.y, n.z});
    }
    return normals;
}

/// The number of bytes of `a` and `b` differing by more than
/// `tolerance`.
std::size_t Differing(const Tensor& a, const Tensor& b,
                      int tolerance) {
    std::size_t count = 0;
    for (std::int64_t i = 0; i < a.Elements(); ++i) {
        count += std::abs(int(a.As<std::uint8_t>()[i]) -
                          int(b.As<std::uint8_t>()[i])) > tolerance
                     ? 1u
                     : 0u;
    }
    return count;
}

}  // namespace

TEST_F(RasterDelegateTest,
       MeshNormalsAndColoursFromCudaMatchTheHostPath) {
    RasterDelegate delegate(*context_);
    std::string why;
    ernie::ComputeInterop* cuda = delegate.Compute(&why);
    if (cuda == nullptr) {
        GTEST_SKIP() << "No CUDA interop: " << why;
    }
    const MeshData ball = ernie::SphereMesh(8.0f, 32, 16);
    Scene host = BallScene(ball);
    RasterDelegate reference(*context_);
    const Tensor expected = reference.Render(host);

    Scene device = BallScene(ball);
    for (const std::string_view attribute :
         {tokens::kPoints, tokens::kNormals, tokens::kDisplayColor}) {
        device.Write(
            "/World/Ball", attribute,
            cuda->Upload(*device.Read("/World/Ball", attribute)));
    }
    const Tensor image = delegate.Render(device);
    ASSERT_EQ(image.Elements(), expected.Elements());
    EXPECT_EQ(Differing(image, expected, 0), 0u)
        << "CUDA normals and colours draw the same image";

    // New normals and colours on the device update in place.
    std::vector<float> flipped(ball.normals.size());
    for (std::size_t i = 0; i < flipped.size(); ++i) {
        flipped[i] = i % 3 == 2 ? ball.normals[i] : -ball.normals[i];
    }
    const auto n = static_cast<std::int64_t>(ball.points.size() / 3);
    device.Write("/World/Ball", tokens::kNormals,
                 cuda->Upload(Tensor::Copy<float>(flipped, {n, 3})));
    host.Write("/World/Ball", tokens::kNormals,
               Tensor::Copy<float>(flipped, {n, 3}));
    std::vector<float> green(ball.points.size(), 0.0f);
    for (std::size_t i = 1; i < green.size(); i += 3) {
        green[i] = 1.0f;
    }
    device.Write("/World/Ball", tokens::kDisplayColor,
                 cuda->Upload(Tensor::Copy<float>(green, {n, 3})));
    host.Write("/World/Ball", tokens::kDisplayColor,
               Tensor::Copy<float>(green, {n, 3}));
    const Tensor updated = delegate.Render(device);
    EXPECT_GT(Differing(updated, image, 2), 100u);
    EXPECT_EQ(Differing(updated, reference.Render(host), 0), 0u);
}

TEST_F(RasterDelegateTest, NormalsFollowPointsMovedOnTheGpu) {
    RasterDelegate delegate(*context_);
    std::string why;
    ernie::ComputeInterop* cuda = delegate.Compute(&why);
    if (cuda == nullptr) {
        GTEST_SKIP() << "No CUDA interop: " << why;
    }
    const MeshData ball = ernie::SphereMesh(8.0f, 32, 16);
    // Squashed and sheared on the GPU: the authored normals no longer
    // fit the points.
    MeshData squashed = ball;
    for (std::size_t i = 0; i < squashed.points.size(); i += 3) {
        squashed.points[i] += 0.5f * squashed.points[i + 2];
        squashed.points[i + 2] *= 0.4f;
    }
    const auto n = static_cast<std::int64_t>(ball.points.size() / 3);
    Scene device = BallScene(ball);
    device.Set(
        "/World/Ball", tokens::kExtent,
        std::vector<float>{-12.0f, -8.0f, -4.0f, 12.0f, 8.0f, 4.0f});
    delegate.Render(device);
    device.Write(
        "/World/Ball", tokens::kPoints,
        cuda->Upload(Tensor::Copy<float>(squashed.points, {n, 3})));
    const Tensor image = delegate.Render(device);

    // As the host draws the squashed points with their area-weighted
    // normals.
    Scene host = BallScene(ball);
    host.Write("/World/Ball", tokens::kPoints,
               Tensor::Copy<float>(squashed.points, {n, 3}));
    host.Write(
        "/World/Ball", tokens::kNormals,
        Tensor::Copy<float>(AreaWeightedNormals(squashed), {n, 3}));
    RasterDelegate reference(*context_);
    EXPECT_LT(Differing(image, reference.Render(host), 2), 8u);
    // Not as the stale authored normals draw them.
    Scene stale = BallScene(ball);
    stale.Write("/World/Ball", tokens::kPoints,
                Tensor::Copy<float>(squashed.points, {n, 3}));
    EXPECT_GT(Differing(image, reference.Render(stale), 2), 100u);
}
