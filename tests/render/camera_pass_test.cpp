// SPDX-License-Identifier: BSD-3-Clause
/// @file camera_pass_test.cpp
/// @brief Tests of the camera pass under validation: a sphere ahead of a pinhole camera, seen in
/// each output where it should be, near the origin and a million kilometres from it.
#include "ernie/device/bindless.h"
#include "ernie/render/camera_pass.h"
#include "ernie/render/device_scene.h"
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
            device_->Transforms().SetOrigin(origin_);
        }
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
                };
                camera_->Add(graph, view, outputs);
            },
            std::span(&ready, 1));
        Outputs outputs;
        outputs.distances = Read<float>(*distances_, kPixels, done);
        outputs.normals = Read<glm::vec3>(*normals_, kPixels, done);
        outputs.instances = Read<std::uint32_t>(*instances_, kPixels, done);
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

    /// @brief Expects the middle pixel to see the sphere's near side 4 m off along +y, and the
    /// corners to see nothing.
    void ExpectTheSphereAhead(const Outputs& outputs, PrimId sphere) {
        EXPECT_NEAR(outputs.distances[kMiddle], 4.0f, 1e-4f);
        EXPECT_NEAR(glm::length(outputs.normals[kMiddle] - glm::vec3(0, -1, 0)), 0.0f, 1e-4f);
        EXPECT_EQ(outputs.instances[kMiddle], *device_->Instances().Row(sphere));
        for (const std::size_t corner : {std::size_t{0}, std::size_t{kSize.x} - 1, kPixels - 1}) {
            EXPECT_EQ(outputs.distances[corner], std::numeric_limits<float>::infinity()) << corner;
            EXPECT_EQ(outputs.normals[corner], glm::vec3(0)) << corner;
            EXPECT_EQ(outputs.instances[corner], CameraPass::kNoInstance) << corner;
        }
    }

    std::unique_ptr<DeviceContext> context_;
    std::unique_ptr<BindlessHeap> heap_;
    std::unique_ptr<CameraPass> camera_;
    std::unique_ptr<Arenas> arenas_;
    std::unique_ptr<DeviceColumn> distances_;
    std::unique_ptr<DeviceColumn> normals_;
    std::unique_ptr<DeviceColumn> instances_;
    std::unique_ptr<DeviceImage> colour_;
    BindlessIndex colourIndex_;
    Scene scene_;
    glm::dvec3 origin_{0.0};
    std::unique_ptr<DeviceScene> device_;
};

TEST_F(CameraPassTest, SeesASphereAheadInEveryOutput) {
    const PrimId sphere = Sphere({0, 5, 0});
    ExpectTheSphereAhead(Frame({0, 0, 0}, {0, 1, 0}), sphere);
}

TEST_F(CameraPassTest, SeesTheSameAMillionKilometresOutAboutTheOrigin) {
    // Floats spaced 128 m apart there: only the origin keeps a 1 m sphere whole.
    const glm::dvec3 far(1e9, -2e9, 3e8);
    origin_ = far;
    const PrimId sphere = Sphere(far + glm::dvec3(0, 5, 0));
    ExpectTheSphereAhead(Frame(far, far + glm::dvec3(0, 1, 0)), sphere);
}

}  // namespace
}  // namespace ernie
