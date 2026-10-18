// SPDX-License-Identifier: BSD-3-Clause
/// @file ray_tracer_test.cpp
/// @brief Tests of `RayTracer` through the device scene: rays at each unit shape, sized, turned and
/// placed as prims, and at a mesh, against their analytic distances and normals; and a planet's
/// sphere staying round.
#include "ernie/render/device_scene.h"
#include "ernie/scene/xform.h"

#include "tests/device/common.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

namespace ernie {
namespace {

using test::ExpectClean;

/// Traces each ray, writing its hit distance and world normal, or -1 and zero.
const ShaderModule kTrace{"trace", R"(
import ray_tracer;
struct Ray { float4 origin; float4 direction; };
struct Parameters { TracerParameters tracer; Ray* rays; float4* hits; uint count; };
[[vk::push_constant]] ConstantBuffer<Parameters> parameters;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= parameters.count) {
        return;
    }
    Ray r = parameters.rays[id.x];
    RayDesc ray = { r.origin.xyz, 0, r.direction.xyz, 1e30 };
    RayTracer tracer = RayTracer(parameters.tracer);
    Hit hit = tracer.TraceClosest(ray, 0xff);
    if (hit.found != tracer.TraceAny(ray, 0xff)) {
        parameters.hits[id.x] = float4(-2, 0, 0, 0);
        return;
    }
    float4 result = float4(-1, 0, 0, 0);
    if (hit.found) {
        result = float4(hit.t, tracer.WorldNormal(hit));
    }
    parameters.hits[id.x] = result;
}
)"};

struct Ray {
    glm::vec4 origin;
    glm::vec4 direction;
};

struct TraceParameters {
    TracerParameters tracer;
    VkDeviceAddress rays, hits;
    std::uint32_t count, pad;
};

/// The hit a ray should make: its distance and world normal, or a miss.
struct Expected {
    const char* what;
    glm::vec3 origin;
    glm::vec3 direction;
    float t;
    glm::vec3 normal{0.0f};
};

constexpr float kMiss = -1.0f;

class RayTracerScene : public ::testing::Test {
  protected:
    void SetUp() override {
        context_ = test::MakeContext();
        if (!context_) {
            GTEST_SKIP() << "No Vulkan device";
        }
        if (!context_->Capabilities().rayQuery) {
            GTEST_SKIP() << context_->Info().name << " has no ray queries";
        }
        const ShaderCompiler compiler;
        std::vector<ShaderModule> modules(RayTracer::Modules().begin(), RayTracer::Modules().end());
        modules.push_back(kTrace);
        trace_ = std::make_unique<ComputeKernel>(*context_, compiler.Spirv(modules, "main"),
                                                 "trace", std::uint32_t{sizeof(TraceParameters)});
        arenas_ = std::make_unique<Arenas>(*context_, MemoryUse::Device, "scene", 1 << 20);
        rays_ = std::make_unique<DeviceColumn>(*arenas_, "rays", DType::Float32,
                                               std::vector<std::int64_t>{8}, 64);
        hits_ = std::make_unique<DeviceColumn>(*arenas_, "hits", DType::Float32,
                                               std::vector<std::int64_t>{4}, 64);
    }
    void TearDown() override {
        if (context_) {
            context_->WaitIdle();
            ExpectClean(*context_);
        }
    }

    /// @return Each ray's hit after a frame of the device scene: distance, then world normal.
    std::vector<glm::vec4> Frame(const std::vector<Ray>& rays) {
        if (!device_) {
            device_ = std::make_unique<DeviceScene>(*context_, *arenas_, scene_, ShaderCompiler());
        }
        Uploader uploader(*context_, context_->Main());
        uploader.Write(rays_->Storage(), rays_->Range().offset, std::span<const Ray>(rays));
        const Ticket uploaded = uploader.Flush();
        const auto count = static_cast<std::uint32_t>(rays.size());
        const Ticket done = device_->Execute(
            [&](graph::Graph& graph) {
                const TraceScene& scene = graph.Board().Get<TraceScene>();
                const graph::Buffer from = graph.Import(*rays_, "rays");
                graph::Buffer to = graph.Import(*hits_, "hits");
                graph.Add(
                    "Trace", graph::Kind::Compute,
                    [&](graph::PassBuilder& pass) {
                        scene.Read(pass);
                        pass.ReadAddress(from);
                        to = pass.WriteAddress(to, graph::Use::StorageWrite);
                    },
                    [this, from, to, count, tracer = scene.parameters](graph::PassContext& pass) {
                        trace_->Dispatch(pass.commands, {1, 1, 1},
                                         TraceParameters{tracer, pass.Get(from).Address(),
                                                         pass.Get(to).Address(), count, 0});
                    });
            },
            std::span(&uploaded, 1));
        Readback readback(*context_, context_->Main(), hits_->Storage(), hits_->Range().offset,
                          rays.size() * sizeof(glm::vec4), std::span(&done, 1));
        std::vector<glm::vec4> hits(rays.size());
        std::memcpy(hits.data(), readback.Bytes().data(), hits.size() * sizeof(glm::vec4));
        return hits;
    }

    /// @brief Traces each expected ray and checks its distance within `tolerance` and its normal.
    void ExpectHits(const std::vector<Expected>& expected, float tolerance = 1e-4f) {
        std::vector<Ray> rays;
        for (const Expected& ray : expected) {
            rays.push_back({glm::vec4(ray.origin, 1), glm::vec4(ray.direction, 0)});
        }
        const std::vector<glm::vec4> hits = Frame(rays);
        for (std::size_t i = 0; i < expected.size(); ++i) {
            const Expected& want = expected[i];
            ASSERT_NE(hits[i].x, -2.0f) << want.what << ": TraceAny disagrees with TraceClosest";
            if (want.t == kMiss) {
                EXPECT_EQ(hits[i].x, kMiss) << want.what;
                continue;
            }
            EXPECT_NEAR(hits[i].x, want.t, tolerance) << want.what;
            const glm::vec3 normal(hits[i].y, hits[i].z, hits[i].w);
            EXPECT_NEAR(glm::length(normal - glm::normalize(want.normal)), 0.0f, 1e-4f)
                << want.what << ": normal (" << normal.x << ", " << normal.y << ", " << normal.z
                << ")";
        }
    }

    /// @brief A 200 m square floor at `height`, wound to face +z.
    void Floor(float height) {
        const PrimId floor = scene_.Define("/Floor", "Mesh");
        const float h = height;
        const std::vector<float> points{-100, -100, h, 100, -100, h, 100, 100, h, -100, 100, h};
        scene_.Write(floor, "points", Tensor::Copy(std::span<const float>(points), {4, 3}));
        scene_.Write(
            floor, "faceVertexCounts",
            Tensor::Copy(std::span<const std::int32_t>(std::vector<std::int32_t>{4}), {1}));
        scene_.Write(
            floor, "faceVertexIndices",
            Tensor::Copy(std::span<const std::int32_t>(std::vector<std::int32_t>{0, 1, 2, 3}),
                         {4}));
    }

    std::unique_ptr<DeviceContext> context_;
    std::unique_ptr<ComputeKernel> trace_;
    std::unique_ptr<Arenas> arenas_;
    std::unique_ptr<DeviceColumn> rays_;
    std::unique_ptr<DeviceColumn> hits_;
    Scene scene_;
    std::unique_ptr<DeviceScene> device_;
};

const glm::vec3 kDown(0, 0, -1);
const glm::vec3 kUp(0, 0, 1);
const glm::vec3 kNorth(0, 1, 0);
const glm::vec3 kEast(1, 0, 0);

TEST_F(RayTracerScene, HitsEachShapeWhereItsSurfaceIs) {
    Floor(-20);
    // Each shape at its own x, so rays straight down or along y meet only it and the floor.
    const PrimId sphere = scene_.Define("/Sphere", "Sphere");
    scene_.Set(sphere, "radius", 2.0);

    scene_.Define("/Cube", "Cube");
    SetXform(scene_, "/Cube", {.translate = glm::dvec3(10, 0, 0), .rotate = glm::dvec3(0, 0, 45)});

    const PrimId cylinder = scene_.Define("/Cylinder", "Cylinder");
    scene_.Set(cylinder, "height", 4.0);
    scene_.Set(cylinder, "axis", std::string("X"));
    SetXform(scene_, "/Cylinder", {.translate = glm::dvec3(20, 0, 0)});

    scene_.Define("/Cone", "Cone");
    SetXform(scene_, "/Cone", {.translate = glm::dvec3(30, 0, 0)});

    const PrimId capsule = scene_.Define("/Capsule", "Capsule");
    scene_.Set(capsule, "radius", 0.5);
    scene_.Set(capsule, "height", 1.0);
    SetXform(scene_, "/Capsule", {.translate = glm::dvec3(40, 0, 0)});

    const PrimId plane = scene_.Define("/Plane", "Plane");
    scene_.Set(plane, "width", 4.0);
    scene_.Set(plane, "length", 2.0);
    SetXform(scene_, "/Plane", {.translate = glm::dvec3(50, 0, 0)});

    const float root3 = std::sqrt(3.0f);
    const float root2 = std::sqrt(2.0f);
    // The capsule's top hemisphere about z = 0.5 with radius 0.5, met at z = 0.75.
    const float capY = std::sqrt(0.25f - 0.0625f);
    ExpectHits({
        {"the sphere's top", {0, 0, 10}, kDown, 8, kUp},
        // x^2 + z^2 = 4 at x = 1.
        {"the sphere off centre", {1, 0, 10}, kDown, 10 - root3, {1, 0, root3}},
        {"the sphere from inside", {0, 0, 0}, kNorth, 2, kNorth},
        // Turned 45 degrees about z, its face (x - y) / sqrt 2 = 1 meets x = 0.3 at y = 0.3 -
        // sqrt 2.
        {"the turned cube's face", {10.3f, -10, 0}, kNorth, 10.3f - root2, {1, -1, 0}},
        {"over the turned cube's top", {10.3f, 0, 10}, kDown, 9, kUp},
        {"the cylinder's side, along x", {20, 0, 10}, kDown, 9, kUp},
        {"the cylinder's cap at x = 18", {12, 0, 0}, kEast, 6, -kEast},
        {"beyond the cylinder's end", {22.5f, 0, 10}, kDown, 30, kUp},
        {"the cone's base", {30, 0, -10}, kUp, 9, -kUp},
        // At z = 0 its radius is (1 - z) / 2; the normal is (x, y, (1 - z) / 4) there.
        {"the cone's side", {30, -10, 0}, kNorth, 9.5f, {0, -0.5f, 0.25f}},
        {"the capsule's top", {40, 0, 10}, kDown, 9, kUp},
        {"the capsule's side", {40, -10, 0}, kNorth, 9.5f, -kNorth},
        {"the capsule's cap", {40, -10, 0.75f}, kNorth, 10 - capY, {0, -capY, 0.25f}},
        {"the plane, 4 m wide along x", {51.5f, 0, 10}, kDown, 10, kUp},
        {"past the plane's 2 m length", {50, 1.5f, 10}, kDown, 30, kUp},
        {"the floor's winding", {60, 0, 10}, kDown, 30, kUp},
        {"beyond the floor", {150, 0, 10}, kDown, kMiss},
    });
}

TEST_F(RayTracerScene, ScaledShapesKeepTheirNormalsPerpendicular) {
    // A sphere squashed to an ellipsoid with semi-axes (4, 1, 1): at x = 2 its surface is at
    // z = sqrt(1 - 1/4), and its normal is the gradient (x / 16, 0, z).
    scene_.Define("/Ellipsoid", "Sphere");
    SetXform(scene_, "/Ellipsoid", {.scale = glm::dvec3(4, 1, 1)});
    const float z = std::sqrt(0.75f);
    ExpectHits({{"the ellipsoid", {2, 0, 10}, kDown, 10 - z, {2.0f / 16.0f, 0, z}}});
}

TEST_F(RayTracerScene, APlanetsSphereStaysRound) {
    // The Earth's radius, its centre below the origin so the surface passes through it. Each ray
    // drops from 100 m above the true surface. A faceted sphere would be metres off between
    // vertices; the implicit sphere is off only by its float transform's rounding, the centre
    // 6371 km away held to a quarter metre, until hits are solved against a double-float origin.
    constexpr double kRadius = 6.371e6;
    const PrimId earth = scene_.Define("/Earth", "Sphere");
    scene_.Set(earth, "radius", kRadius);
    SetXform(scene_, "/Earth", {.translate = glm::dvec3(0, 0, -kRadius)});
    std::vector<Expected> expected;
    for (const double x : {0.0, 1e3, 1e4, 1e5}) {
        const double surface = std::sqrt(kRadius * kRadius - x * x) - kRadius;
        expected.push_back({"100 m above the surface", glm::vec3(x, 0, surface + 100), kDown, 100,
                            glm::vec3(x, 0, surface + kRadius)});
    }
    ExpectHits(expected, 0.5f);
}

}  // namespace
}  // namespace ernie
