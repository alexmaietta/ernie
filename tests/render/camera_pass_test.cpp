// SPDX-License-Identifier: BSD-3-Clause
/// @file camera_pass_test.cpp
/// @brief Tests of the camera pass under validation: a sphere ahead of a pinhole camera, seen in
/// each output where it should be, near the origin and a million kilometres from it.
#include "ernie/device/bindless.h"
#include "ernie/render/rt/camera_pass.h"
#include "ernie/scene/xform.h"

#include "tests/render/common.h"

#include <gtest/gtest.h>

#include <glm/gtc/matrix_transform.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace ernie {
namespace {

/// An odd size, so the middle pixel looks straight ahead.
constexpr glm::uvec2 kSize{33, 33};
constexpr std::size_t kPixels = std::size_t{kSize.x} * kSize.y;
constexpr std::size_t kMiddle = kPixels / 2;
/// The background, and its RGBA8 bytes.
constexpr glm::vec3 kBackground{0.25f, 0.5f, 0.75f};
constexpr std::array<std::uint8_t, 4> kBackgroundBytes{64, 128, 191, 255};

/// What the camera pass wrote for a frame.
struct Outputs {
    std::vector<float> distances;
    std::vector<glm::vec3> normals;
    std::vector<std::uint32_t> instances;
    std::vector<glm::vec3> albedos;
    std::vector<std::array<std::uint8_t, 4>> ldr;
};

class CameraPassTest : public test::RayTracingTest {
  protected:
    void SetUp() override {
        test::RayTracingTest::SetUp();
        if (IsSkipped()) {
            return;
        }
        heap_ = std::make_unique<BindlessHeap>(*context_);
        camera_ = std::make_unique<CameraPass>(*context_, compiler_, *heap_);
        const auto column = [&](const char* name, DType type, std::int64_t width) {
            return std::make_unique<DeviceColumn>(*arenas_, name, type,
                                                  width == 1 ? std::vector<std::int64_t>{}
                                                             : std::vector<std::int64_t>{width},
                                                  static_cast<std::int64_t>(kPixels));
        };
        distances_ = column("distances", DType::Float32, 1);
        normals_ = column("normals", DType::Float32, 3);
        instances_ = column("instances", DType::Int32, 1);
        albedos_ = column("albedos", DType::Float32, CameraPass::kBands);
        ldr_ = column("ldr", DType::UInt8, 4);
        colour_ = std::make_unique<DeviceImage>(
            *context_,
            ImageDescription{.format = ImageFormat::Rgba8Unorm,
                             .extent = {kSize, 1},
                             .usage = ImageUsage::Sampled | ImageUsage::Storage},
            "colour");
        colourIndex_ = heap_->AddStorageImage(*colour_);
    }

    /// @return The outputs of a frame looking from `eye` at `target`.
    Outputs Frame(const glm::dvec3& eye, const glm::dvec3& target) {
        // As the viewer does, keeping the origin near the eye.
        Device().Follow(eye);
        const CameraView view{.cameraToWorld =
                                  glm::inverse(glm::lookAtRH(eye, target, glm::dvec3(0, 0, 1))),
                              .size = kSize,
                              .lens = MakePinhole(0.5, kSize)};
        const Ticket ready = colour_->Ready();
        const Ticket done = Device().Execute(
            [&](graph::Graph& graph) {
                CameraOutputs outputs{
                    .colour = graph.Import(*colour_, "colour"),
                    .colourIndex = colourIndex_,
                    .background = kBackground,
                    .distances = graph.Import(*distances_),
                    .normals = graph.Import(*normals_),
                    .instances = graph.Import(*instances_),
                    .albedos = graph.Import(*albedos_),
                    .ldr = graph.Import(*ldr_),
                };
                camera_->Add(graph, view, outputs);
            },
            std::span(&ready, 1));
        return {.distances = Read<float>(*distances_, kPixels, done),
                .normals = Read<glm::vec3>(*normals_, kPixels, done),
                .instances = Read<std::uint32_t>(*instances_, kPixels, done),
                .albedos = Read<glm::vec3>(*albedos_, kPixels, done),
                .ldr = Read<std::array<std::uint8_t, 4>>(*ldr_, kPixels, done)};
    }

    /// @brief Puts a sphere of radius 1 at `centre`.
    PrimId Sphere(const glm::dvec3& centre) {
        const PrimId sphere = scene_.Define("/Ball", "Sphere");
        SetXform(scene_, "/Ball", {.translate = centre});
        return sphere;
    }

    /// @brief Expects the middle pixel to see the sphere's near side 4 m off along +y, of albedo
    /// `colour`, and the corners to see nothing.
    void ExpectTheSphereAhead(const Outputs& outputs, PrimId sphere,
                              glm::vec3 colour = glm::vec3(0.5f)) {
        EXPECT_NEAR(outputs.distances[kMiddle], 4.0f, 1e-4f);
        EXPECT_NEAR(glm::length(outputs.normals[kMiddle] - glm::vec3(0, -1, 0)), 0.0f, 1e-4f);
        EXPECT_EQ(outputs.instances[kMiddle], *Device().Instances().Row(sphere));
        EXPECT_EQ(outputs.albedos[kMiddle], colour);
        for (const std::size_t corner : {std::size_t{0}, std::size_t{kSize.x} - 1, kPixels - 1}) {
            EXPECT_EQ(outputs.distances[corner], std::numeric_limits<float>::infinity()) << corner;
            EXPECT_EQ(outputs.normals[corner], glm::vec3(0)) << corner;
            EXPECT_EQ(outputs.instances[corner], CameraPass::kNoInstance) << corner;
            EXPECT_EQ(outputs.albedos[corner], glm::vec3(0)) << corner;
            EXPECT_EQ(outputs.ldr[corner], kBackgroundBytes) << corner;
        }
    }

    std::unique_ptr<BindlessHeap> heap_;
    std::unique_ptr<CameraPass> camera_;
    std::unique_ptr<DeviceColumn> distances_;
    std::unique_ptr<DeviceColumn> normals_;
    std::unique_ptr<DeviceColumn> instances_;
    std::unique_ptr<DeviceColumn> albedos_;
    std::unique_ptr<DeviceColumn> ldr_;
    std::unique_ptr<DeviceImage> colour_;
    BindlessIndex colourIndex_;
};

TEST_F(CameraPassTest, SeesASphereAheadInEveryOutput) {
    const PrimId sphere = Sphere({0, 5, 0});
    const Outputs outputs = Frame({0, 0, 0}, {0, 1, 0});
    ExpectTheSphereAhead(outputs, sphere);
    // Albedo 0.5 lit head on, display encoded: 0.5^(1 / 2.2) of 255.
    EXPECT_EQ(outputs.ldr[kMiddle], (std::array<std::uint8_t, 4>{186, 186, 186, 255}));
    EXPECT_EQ(Device().Transforms().Origin(), glm::dvec3(0)) << "the eye is at the origin";
}

TEST_F(CameraPassTest, SeesTheSameAMillionKilometresOutOnceTheOriginFollows) {
    // Floats spaced 128 m apart there: only an origin near the eye keeps a 1 m sphere whole.
    const glm::dvec3 far(1e9, -2e9, 3e8);
    const PrimId sphere = Sphere(far + glm::dvec3(0, 5, 0));
    ExpectTheSphereAhead(Frame(far, far + glm::dvec3(0, 1, 0)), sphere);
    const glm::dvec3 origin = Device().Transforms().Origin();
    EXPECT_LE(glm::length(origin - far), DeviceScene::kRebaseDistance);
    EXPECT_EQ(glm::mod(origin, glm::dvec3(DeviceScene::kRebaseDistance)), glm::dvec3(0))
        << "on the rebase grid";
}

TEST_F(CameraPassTest, KeepsSeeingWhatItFliesTo) {
    const glm::dvec3 far(4e7, 0, 0);
    const PrimId sphere = Sphere(far + glm::dvec3(0, 5, 0));
    Frame({0, 0, 0}, {0, 1, 0});
    EXPECT_FALSE(Device().Follow({500, 0, 0})) << "within the rebase distance";
    ExpectTheSphereAhead(Frame(far, far + glm::dvec3(0, 1, 0)), sphere);
}

TEST_F(CameraPassTest, AlbedoFollowsTheDisplayColourWithoutRebuilding) {
    const PrimId sphere = Sphere({0, 5, 0});
    scene_.Set(sphere, "primvars:displayColor", glm::dvec3(0.25, 0.5, 0.75));
    ExpectTheSphereAhead(Frame({0, 0, 0}, {0, 1, 0}), sphere, {0.25f, 0.5f, 0.75f});
    scene_.Set(sphere, "primvars:displayColor", glm::dvec3(1, 0, 0));
    ExpectTheSphereAhead(Frame({0, 0, 0}, {0, 1, 0}), sphere, {1, 0, 0});
    const BottomLevels::Statistics& totals = Device().Tracer()->Bottoms().Totals();
    EXPECT_EQ(totals.built, 1u) << "the unit sphere, once";
    EXPECT_EQ(totals.refitted, 0u);
}

TEST_F(CameraPassTest, StacksViewsOfDifferentSizesFromOneDispatch) {
    const PrimId sphere = Sphere({0, 5, 0});
    const glm::uvec2 small{17, 9};
    const std::size_t smallPixels = std::size_t{small.x} * small.y;
    const DeviceColumn distances(*arenas_, "distances", DType::Float32, {}, kPixels + smallPixels);
    const DeviceColumn prims(*arenas_, "prims", DType::UInt32, {}, kPixels + smallPixels);
    const auto looking = [](const glm::dvec3& target) {
        return glm::inverse(glm::lookAtRH(glm::dvec3(0), target, glm::dvec3(0, 0, 1)));
    };
    const std::array views{
        CameraView{
            .cameraToWorld = looking({0, 1, 0}), .size = kSize, .lens = MakePinhole(0.5, kSize)},
        CameraView{
            .cameraToWorld = looking({0, -1, 0}), .size = small, .lens = MakePinhole(0.8, small)},
    };
    const Ticket done = Device().Execute([&](graph::Graph& graph) {
        CameraOutputs outputs{.distances = graph.Import(distances), .prims = graph.Import(prims)};
        camera_->Add(graph, views, outputs);
    });
    const std::vector<float> seen = Read<float>(distances, kPixels + smallPixels, done);
    EXPECT_NEAR(seen[kMiddle], 4.0f, 1e-4f);
    EXPECT_EQ(Read<std::uint32_t>(prims, kPixels, done)[kMiddle], sphere.slot);
    for (std::size_t pixel = kPixels; pixel < seen.size(); ++pixel) {
        ASSERT_EQ(seen[pixel], std::numeric_limits<float>::infinity()) << "nothing behind";
    }
}

TEST_F(CameraPassTest, OrthographicRaysRunParallel) {
    Sphere({0, 5, 0});
    // 4 m across 33 pixels: the 2 m sphere covers the middle half, at the same depth everywhere
    // a straight-on ray would see it.
    const auto density = static_cast<float>(kSize.x) / 4.0f;
    const CameraView view{.cameraToWorld = glm::inverse(glm::lookAtRH(
                              glm::dvec3(0), glm::dvec3(0, 1, 0), glm::dvec3(0, 0, 1))),
                          .size = kSize,
                          .lens = {.kind = shaders::kOrthographic,
                                   .focal = glm::vec2(density),
                                   .principal = 0.5f * glm::vec2(kSize)}};
    const Ticket done = Device().Execute([&](graph::Graph& graph) {
        CameraOutputs outputs{.distances = graph.Import(*distances_)};
        camera_->Add(graph, view, outputs);
    });
    const std::vector<float> distances = Read<float>(*distances_, kPixels, done);
    EXPECT_NEAR(distances[kMiddle], 4.0f, 1e-4f);
    // Four pixels right of the axis the ray runs x metres off it, meeting the sphere sqrt(1 - x^2)
    // nearer than its centre.
    const float x = 4.0f / density;
    EXPECT_NEAR(distances[kMiddle + 4], 5.0f - std::sqrt(1.0f - x * x), 1e-4f);
    EXPECT_EQ(distances[0], std::numeric_limits<float>::infinity()) << "the corner, off the sphere";
}

}  // namespace
}  // namespace ernie
