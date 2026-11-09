// SPDX-License-Identifier: BSD-3-Clause
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/render/render_output.h"
#include "ernie/render/renderer.h"
#include "ernie/scene/scene.h"
#include "support/devices.h"

namespace ernie {
namespace {

using device::Device;
using test::ForEachDevice;

constexpr uint32_t kEmitters = 16;
constexpr uint32_t kFirings = 64;
const std::vector<std::string> kProducts = {"/Render/Scan"};
const std::vector<std::string> kChannels = {"Coordinates", "Distance", "Intensity"};

struct Cloud {
    uint32_t count = 0;
    std::vector<float> coordinates;
    std::vector<float> distance;
    std::vector<float> intensity;
};

class Bench {
  public:
    Bench(Device& device, FrameOfReference frame, Float3 position = {0, 0, 0})
        : m_scene(Scene::Create(device, {.waitForKernels = true})) {
        Scene& scene = *m_scene;
        scene.Define<Plane>("/Wall", {.translate = {0, 0, -5}, .width = 0, .length = 0});
        std::vector<float> elevations;
        for (uint32_t emitter = 0; emitter < kEmitters; ++emitter) {
            elevations.push_back(-15 + 30 * float(emitter) / (kEmitters - 1));
        }
        scene.Define<Lidar>("/Lidar", {.translate = {position.x, position.y, position.z},
                                       .pipelineBinding = {"/Pipe"},
                                       .scanType = ScanType::Rotary,
                                       .azimuthRange = {-30, 30},
                                       .firingsPerScan = kFirings,
                                       .emitterElevations = elevations});
        m_pipe = scene.Define<SensorPipeline>("/Pipe");
        m_last = scene.Define<Shader>("/Pipe/Returns", {.id = "LidarReturns"});
        scene.Define<Scope>("/Render");
        scene.Define<RenderVar>(
            "/Render/Plain",
            {.sourceName = "PointCloud", .channels = kChannels, .frameOfReference = frame});
        scene.Define<RenderVar>(
            "/Render/Piped",
            {.sourceName = "cloud", .channels = kChannels, .frameOfReference = frame});
        scene.Define<RenderProduct>(
            "/Render/Scan",
            {.camera = {"/Lidar"}, .orderedVars = {"/Render/Plain", "/Render/Piped"}});
    }

    Scene& GetScene() {
        return *m_scene;
    }

    Prim Add(const std::string& name, const char* id) {
        const Prim node = m_scene->Define<Shader>("/Pipe/" + name, {.id = id});
        node.Connect("returns", m_last, "returns");
        m_last = node;
        return node;
    }

    std::pair<Cloud, Cloud> Render() {
        if (!m_renderer) {
            const Prim cloud = m_scene->Define<Shader>("/Pipe/Cloud", {.id = "ToPointCloud"});
            cloud.Connect("returns", m_last, "returns");
            m_pipe.Connect("cloud", cloud, "cloud");
            m_renderer.emplace(m_scene);
        }
        RenderOutput output = Step();
        const auto& frame = output["/Render/Scan"].frames.at(0);
        const auto read = [&](const char* source) {
            const RenderVarOutput& var = frame.Source(source);
            Cloud cloud{.count = var.channels.at("Counts").ToHost<uint32_t>().at(0),
                        .coordinates = var.channels.at("Coordinates").ToHost<float>(),
                        .distance = var.channels.at("Distance").ToHost<float>(),
                        .intensity = var.channels.at("Intensity").ToHost<float>()};
            cloud.coordinates.resize(3 * cloud.count);
            cloud.distance.resize(cloud.count);
            cloud.intensity.resize(cloud.count);
            return cloud;
        };
        return {read("PointCloud"), read("cloud")};
    }

    RenderOutput Step(const std::vector<std::string>& products = kProducts) {
        RenderOutput output = m_renderer->Step(products, 0);
        output.Wait();
        return output;
    }

  private:
    Ref<Scene> m_scene;
    Prim m_pipe;
    Prim m_last;
    std::optional<Renderer> m_renderer;
};

TEST(LidarPipelineTest, ReturnsStraightToACloudMatchTheTracedCloud) {
    ForEachDevice([](Device& device) {
        Bench bench(device, FrameOfReference::Sensor);
        const auto [plain, piped] = bench.Render();
        ASSERT_EQ(plain.count, kEmitters * kFirings);
        EXPECT_EQ(piped.count, plain.count);
        EXPECT_EQ(piped.coordinates, plain.coordinates);
        EXPECT_EQ(piped.distance, plain.distance);
        EXPECT_EQ(piped.intensity, plain.intensity);
    });
}

TEST(LidarPipelineTest, ProductsReadingOnlyTheTracedCloudSkipThePipeline) {
    ForEachDevice([](Device& device) {
        Bench bench(device, FrameOfReference::Sensor);
        bench.Add("Noise", "RangeNoise").Set("sigma", 0.05f);
        Scene& scene = bench.GetScene();
        scene.Define<RenderProduct>("/Render/Traced",
                                    {.camera = {"/Lidar"}, .orderedVars = {"/Render/Plain"}});
        const auto [plain, piped] = bench.Render();
        RenderOutput output = bench.Step({"/Render/Traced", "/Render/Scan"});
        const RenderVarOutput& traced =
            output["/Render/Traced"].frames.at(0).RenderVar("/Render/Plain");
        EXPECT_EQ(traced.channels.at("Counts").ToHost<uint32_t>().at(0), plain.count);
        EXPECT_GT(piped.count, 0u);
        EXPECT_GT(output["/Render/Scan"]
                      .frames.at(0)
                      .RenderVar("/Render/Piped")
                      .channels.at("Counts")
                      .ToHost<uint32_t>()
                      .at(0),
                  0u);
    });
}

TEST(LidarPipelineTest, RangeNoiseHasItsSigmaAndKeepsPointsOnTheirBeams) {
    for (const FrameOfReference frame : {FrameOfReference::Sensor, FrameOfReference::World}) {
        ForEachDevice([&](Device& device) {
            const Float3 position = {1, 2, 3};
            Bench bench(device, frame, position);
            bench.Add("Noise", "RangeNoise").Set("sigma", 0.05f);
            const auto [plain, piped] = bench.Render();
            ASSERT_EQ(piped.count, plain.count);
            ASSERT_GT(plain.count, 0u);
            const Float3 origin = frame == FrameOfReference::World ? position : Float3{0, 0, 0};
            double sum = 0;
            double squares = 0;
            for (uint32_t i = 0; i < plain.count; ++i) {
                const double error = piped.distance[i] - plain.distance[i];
                sum += error;
                squares += error * error;
                const float ratio = piped.distance[i] / plain.distance[i];
                const float axes[] = {origin.x, origin.y, origin.z};
                for (int axis = 0; axis < 3; ++axis) {
                    const float from = plain.coordinates[3 * i + axis] - axes[axis];
                    EXPECT_NEAR(piped.coordinates[3 * i + axis] - axes[axis], from * ratio, 2e-3f)
                        << i << " axis " << axis;
                }
            }
            const double mean = sum / plain.count;
            const double sigma = std::sqrt(squares / plain.count - mean * mean);
            EXPECT_NEAR(mean, 0, 0.005);
            EXPECT_NEAR(sigma, 0.05, 0.005);
        });
    }
}

TEST(LidarPipelineTest, TapsOfReturnStagesAreClouds) {
    ForEachDevice([](Device& device) {
        Bench bench(device, FrameOfReference::Sensor);
        bench.Add("Noise", "RangeNoise").Set("sigma", 0.05f);
        Scene& scene = bench.GetScene();
        for (const auto& [name, node] :
             {std::pair{"Source", "/Pipe/Returns"}, std::pair{"Noisy", "/Pipe/Noise"}}) {
            scene.Define<RenderVar>(std::string("/Render/") + name,
                                    {.sourceName = "Tap", .channels = kChannels, .tap = {node}});
        }
        scene.Find("/Render/Scan")
            ->Set(RenderProduct::kOrderedVars,
                  std::vector<std::string>{"/Render/Plain", "/Render/Piped", "/Render/Source",
                                           "/Render/Noisy"});
        const uint32_t traced = bench.Render().first.count;
        RenderOutput output = bench.Step();
        const auto& frame = output["/Render/Scan"].frames.at(0);
        const auto distances = [&](const char* var) {
            const RenderVarOutput& cloud = frame.RenderVar(var);
            std::vector<float> values = cloud.channels.at("Distance").ToHost<float>();
            values.resize(cloud.channels.at("Counts").ToHost<uint32_t>().at(0));
            return values;
        };
        EXPECT_EQ(distances("/Render/Source"), distances("/Render/Plain"));
        EXPECT_EQ(distances("/Render/Noisy"), distances("/Render/Piped"));
        EXPECT_NE(distances("/Render/Noisy"), distances("/Render/Plain"));
        EXPECT_EQ(distances("/Render/Plain").size(), traced);
    });
}

TEST(LidarPipelineTest, DropoutDropsWeakReturnsAndItsRate) {
    ForEachDevice([](Device& device) {
        Bench bench(device, FrameOfReference::Sensor);
        const Prim dropout = bench.Add("Dropout", "Dropout");
        dropout.Set("rate", 0.3f);
        auto [plain, piped] = bench.Render();
        const double kept = double(piped.count) / plain.count;
        EXPECT_NEAR(kept, 0.7, 0.04);
        std::tie(plain, piped) = bench.Render();
        EXPECT_NEAR(double(piped.count) / plain.count, 0.7, 0.04);

        dropout.Set("rate", 0.0f);
        dropout.Set("threshold", 1e9f);
        std::tie(plain, piped) = bench.Render();
        EXPECT_GT(plain.count, 0u);
        EXPECT_EQ(piped.count, 0u);
    });
}

TEST(LidarPipelineTest, RangeQuantizeSnapsRangesToItsStep) {
    ForEachDevice([](Device& device) {
        Bench bench(device, FrameOfReference::Sensor);
        bench.Add("Quantize", "RangeQuantize").Set("step", 0.25f);
        const auto [plain, piped] = bench.Render();
        ASSERT_EQ(piped.count, plain.count);
        for (uint32_t i = 0; i < piped.count; ++i) {
            EXPECT_NEAR(piped.distance[i], std::round(plain.distance[i] / 0.25f) * 0.25f, 1e-4f);
        }
    });
}

}
}
