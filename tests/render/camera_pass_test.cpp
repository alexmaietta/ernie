// SPDX-License-Identifier: BSD-3-Clause
/// @file camera_pass_test.cpp
/// @brief Tests of the camera pass under validation: a sphere ahead of a pinhole camera, seen in
/// each output where it should be, near the origin and a million kilometres from it.
#include "ernie/device/bindless.h"
#include "ernie/render/device_scene.h"
#include "ernie/render/rt/camera_pass.h"
#include "ernie/scene/xform.h"

#include "tests/device/common.h"

#include <gtest/gtest.h>

#include <glm/gtc/matrix_transform.hpp>

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace ernie {
namespace {

using test::ExpectClean;

/// An odd size, so the middle pixel looks straight ahead.
constexpr glm::uvec2 kSize{33, 33};
constexpr std::size_t kPixels = std::size_t{kSize.x} * kSize.y;
constexpr std::size_t kMiddle = kPixels / 2;

/// What the camera pass wrote for a frame.
struct Outputs {
    std::vector<float> distances;
    std::vector<glm::vec3> normals;
    std::vector<std::uint32_t> instances;
    std::vector<glm::vec3> albedos;
};

class CameraPassTest : public ::testing::Test {
  protected:
    void SetUp() override {
        context_ = test::MakeContext();
        if (!context_) {
            GTEST_SKIP() << "No Vulkan device";
        }
        if (!context_->Capabilities().rayQuery) {
            GTEST_SKIP() << context_->Info().name << " has no ray queries";
        }
        heap_ = std::make_unique<BindlessHeap>(*context_);
        camera_ = std::make_unique<CameraPass>(*context_, ShaderCompiler(), *heap_);
        arenas_ = std::make_unique<Arenas>(*context_, MemoryUse::Device, "scene", 1 << 20);
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
        colour_ = std::make_unique<DeviceImage>(
            *context_,
            ImageDescription{.format = ImageFormat::Rgba8Unorm,
                             .extent = {kSize, 1},
                             .usage = ImageUsage::Sampled | ImageUsage::Storage},
            "colour");
        colourIndex_ = heap_->AddStorageImage(*colour_);
    }
    void TearDown() override {
        if (context_) {
            context_->WaitIdle();
            ExpectClean(*context_);
        }
    }

    /// @return The outputs of a frame looking from `eye` at `target`.
    Outputs Frame(const glm::dvec3& eye, const glm::dvec3& target) {
        if (!device_) {
            device_ = std::make_unique<DeviceScene>(*context_, *arenas_, scene_, ShaderCompiler());
        }
        // As the viewer does, keeping the origin near the eye.
        device_->Follow(eye);
        const PinholeView view{.cameraToWorld =
                                   glm::inverse(glm::lookAtRH(eye, target, glm::dvec3(0, 0, 1))),
                               .verticalFov = 0.5,
                               .size = kSize};
        const Ticket ready = colour_->Ready();
        const Ticket done = device_->Execute(
            [&](graph::Graph& graph) {
                CameraOutputs outputs{
                    .colour = graph.Import(*colour_, "colour"),
                    .colourIndex = colourIndex_,
                    .distances = graph.Import(*distances_, "distances"),
                    .normals = graph.Import(*normals_, "normals"),
                    .instances = graph.Import(*instances_, "instances"),
                    .albedos = graph.Import(*albedos_, "albedos"),
                };
                camera_->Add(graph, view, outputs);
            },
            std::span(&ready, 1));
        Outputs outputs;
        outputs.distances = Read<float>(*distances_, kPixels, done);
        outputs.normals = Read<glm::vec3>(*normals_, kPixels, done);
        outputs.instances = Read<std::uint32_t>(*instances_, kPixels, done);
        outputs.albedos = Read<glm::vec3>(*albedos_, kPixels, done);
        return outputs;
    }

    template <typename T>
    std::vector<T> Read(const DeviceColumn& column, std::size_t count, const Ticket& done) {
        Readback readback(*context_, context_->Main(), column.Storage(), column.Range().offset,
                          count * sizeof(T), std::span(&done, 1));
        std::vector<T> values(count);
        std::memcpy(values.data(), readback.Bytes().data(), count * sizeof(T));
        return values;
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
        EXPECT_EQ(outputs.instances[kMiddle], *device_->Instances().Row(sphere));
        EXPECT_EQ(outputs.albedos[kMiddle], colour);
        for (const std::size_t corner : {std::size_t{0}, std::size_t{kSize.x} - 1, kPixels - 1}) {
            EXPECT_EQ(outputs.distances[corner], std::numeric_limits<float>::infinity()) << corner;
            EXPECT_EQ(outputs.normals[corner], glm::vec3(0)) << corner;
            EXPECT_EQ(outputs.instances[corner], CameraPass::kNoInstance) << corner;
            EXPECT_EQ(outputs.albedos[corner], glm::vec3(0)) << corner;
        }
    }

    std::unique_ptr<DeviceContext> context_;
    std::unique_ptr<BindlessHeap> heap_;
    std::unique_ptr<CameraPass> camera_;
    std::unique_ptr<Arenas> arenas_;
    std::unique_ptr<DeviceColumn> distances_;
    std::unique_ptr<DeviceColumn> normals_;
    std::unique_ptr<DeviceColumn> instances_;
    std::unique_ptr<DeviceColumn> albedos_;
    std::unique_ptr<DeviceImage> colour_;
    BindlessIndex colourIndex_;
    Scene scene_;
    std::unique_ptr<DeviceScene> device_;
};

TEST_F(CameraPassTest, SeesASphereAheadInEveryOutput) {
    const PrimId sphere = Sphere({0, 5, 0});
    ExpectTheSphereAhead(Frame({0, 0, 0}, {0, 1, 0}), sphere);
    EXPECT_EQ(device_->Transforms().Origin(), glm::dvec3(0)) << "the eye is at the origin";
}

TEST_F(CameraPassTest, SeesTheSameAMillionKilometresOutOnceTheOriginFollows) {
    // Floats spaced 128 m apart there: only an origin near the eye keeps a 1 m sphere whole.
    const glm::dvec3 far(1e9, -2e9, 3e8);
    const PrimId sphere = Sphere(far + glm::dvec3(0, 5, 0));
    ExpectTheSphereAhead(Frame(far, far + glm::dvec3(0, 1, 0)), sphere);
    const glm::dvec3 origin = device_->Transforms().Origin();
    EXPECT_LE(glm::length(origin - far), DeviceScene::kRebaseDistance);
    EXPECT_EQ(glm::mod(origin, glm::dvec3(DeviceScene::kRebaseDistance)), glm::dvec3(0))
        << "on the rebase grid";
}

TEST_F(CameraPassTest, KeepsSeeingWhatItFliesTo) {
    const glm::dvec3 far(4e7, 0, 0);
    const PrimId sphere = Sphere(far + glm::dvec3(0, 5, 0));
    Frame({0, 0, 0}, {0, 1, 0});
    EXPECT_FALSE(device_->Follow({500, 0, 0})) << "within the rebase distance";
    ExpectTheSphereAhead(Frame(far, far + glm::dvec3(0, 1, 0)), sphere);
}

TEST_F(CameraPassTest, AlbedoFollowsTheDisplayColourWithoutRebuilding) {
    const PrimId sphere = Sphere({0, 5, 0});
    scene_.Set(sphere, "primvars:displayColor", glm::dvec3(0.25, 0.5, 0.75));
    ExpectTheSphereAhead(Frame({0, 0, 0}, {0, 1, 0}), sphere, {0.25f, 0.5f, 0.75f});
    scene_.Set(sphere, "primvars:displayColor", glm::dvec3(1, 0, 0));
    ExpectTheSphereAhead(Frame({0, 0, 0}, {0, 1, 0}), sphere, {1, 0, 0});
    const BottomLevels::Statistics& totals = device_->Tracer()->Bottoms().Totals();
    EXPECT_EQ(totals.built, 1u) << "the unit sphere, once";
    EXPECT_EQ(totals.refitted, 0u);
}

TEST_F(CameraPassTest, TracesViewsOfDifferentSizesInOneDispatch) {
    const PrimId sphere = Sphere({0, 5, 0});
    const glm::uvec2 small{17, 9};
    const DeviceColumn behind(*arenas_, "behind", DType::Float32, {}, small.x * small.y);
    device_ = std::make_unique<DeviceScene>(*context_, *arenas_, scene_, ShaderCompiler());
    const auto looking = [](const glm::dvec3& target) {
        return glm::inverse(glm::lookAtRH(glm::dvec3(0), target, glm::dvec3(0, 0, 1)));
    };
    const Ticket done = device_->Execute([&](graph::Graph& graph) {
        std::array views{
            CameraView{
                .view = {.cameraToWorld = looking({0, 1, 0}), .verticalFov = 0.5, .size = kSize},
                .outputs = {.distances = graph.Import(*distances_, "distances"),
                            .instances = graph.Import(*instances_, "instances")}},
            CameraView{.view = {.cameraToWorld = looking({0, -1, 0}), .size = small},
                       .outputs = {.distances = graph.Import(behind, "behind")}},
        };
        camera_->Add(graph, views);
    });
    const std::vector<float> ahead = Read<float>(*distances_, kPixels, done);
    EXPECT_NEAR(ahead[kMiddle], 4.0f, 1e-4f);
    EXPECT_EQ(Read<std::uint32_t>(*instances_, kPixels, done)[kMiddle],
              *device_->Instances().Row(sphere));
    for (const float distance : Read<float>(behind, std::size_t{small.x} * small.y, done)) {
        ASSERT_EQ(distance, std::numeric_limits<float>::infinity()) << "nothing behind";
    }
}

}  // namespace
}  // namespace ernie
