// SPDX-License-Identifier: BSD-3-Clause
#include <cmath>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/render/render_output.h"
#include "ernie/render/renderer.h"
#include "ernie/render/standalone.h"
#include "ernie/scene/scene.h"
#include "support/devices.h"

namespace ernie {
namespace {

using device::Device;
using test::ForEachDevice;

const std::vector<std::string> kProducts = {"/Render/Scan"};
const std::vector<std::string> kChannels = {"Coordinates", "Distance", "Intensity",
                                            "RadialVelocity"};

struct Cloud {
    uint32_t count = 0;
    std::vector<float> coordinates;
    std::vector<float> distance;
    std::vector<float> intensity;
    std::vector<float> velocity;
};

class Bench {
  public:
    explicit Bench(Device& device) : m_scene(Scene::Create(device, {.waitForKernels = true})) {
        Scene& scene = *m_scene;
        const Prim wall =
            scene.Define<Plane>("/Wall", {.translate = {0, 0, -20}, .width = 0, .length = 0});
        scene.Apply<MotionAPI>(wall, {.motionVelocity = {0, 0, 3}});
        scene.Define<Radar>(
            "/Radar", {.pipelineBinding = {"/Pipe"}, .fieldOfView = {40, 20}, .rays = {64, 32}});
        m_pipe = scene.Define<SensorPipeline>("/Pipe");
        m_last = scene.Define<Shader>("/Pipe/Returns", {.id = "RadarReturns"});
        scene.Define<Scope>("/Render");
        scene.Define<RenderVar>("/Render/Plain",
                                {.sourceName = "PointCloud", .channels = kChannels});
        scene.Define<RenderVar>("/Render/Piped", {.sourceName = "cloud", .channels = kChannels});
        scene.Define<RenderProduct>(
            "/Render/Scan",
            {.camera = {"/Radar"}, .orderedVars = {"/Render/Plain", "/Render/Piped"}});
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
        RenderOutput output = m_renderer->Step(kProducts, 0);
        output.Wait();
        const auto& frame = output["/Render/Scan"].frames.at(0);
        const auto read = [&](const char* source) {
            const RenderVarOutput& var = frame.Source(source);
            Cloud cloud{.count = var.channels.at("Counts").ToHost<uint32_t>().at(0),
                        .coordinates = var.channels.at("Coordinates").ToHost<float>(),
                        .distance = var.channels.at("Distance").ToHost<float>(),
                        .intensity = var.channels.at("Intensity").ToHost<float>(),
                        .velocity = var.channels.at("RadialVelocity").ToHost<float>()};
            cloud.coordinates.resize(3 * cloud.count);
            cloud.distance.resize(cloud.count);
            cloud.intensity.resize(cloud.count);
            cloud.velocity.resize(cloud.count);
            return cloud;
        };
        return {read("PointCloud"), read("cloud")};
    }

  private:
    Ref<Scene> m_scene;
    Prim m_pipe;
    Prim m_last;
    std::optional<Renderer> m_renderer;
};

TEST(RadarPipelineTest, ReturnsStraightToACloudMatchTheTracedCloud) {
    ForEachDevice([](Device& device) {
        Bench bench(device);
        const auto [plain, piped] = bench.Render();
        ASSERT_EQ(plain.count, 64u * 32u);
        EXPECT_EQ(piped.count, plain.count);
        EXPECT_EQ(piped.coordinates, plain.coordinates);
        EXPECT_EQ(piped.distance, plain.distance);
        EXPECT_EQ(piped.intensity, plain.intensity);
        EXPECT_EQ(piped.velocity, plain.velocity);
    });
}

TEST(RadarPipelineTest, RadarNoiseHasItsSigmas) {
    ForEachDevice([](Device& device) {
        Bench bench(device);
        const Prim noise = bench.Add("Noise", "RadarNoise");
        noise.Set("rangeSigma", 0.1f);
        noise.Set("velocitySigma", 0.2f);
        const auto [plain, piped] = bench.Render();
        ASSERT_EQ(piped.count, plain.count);
        const auto sigma = [&](const std::vector<float>& a, const std::vector<float>& b) {
            double sum = 0;
            double squares = 0;
            for (size_t i = 0; i < a.size(); ++i) {
                sum += a[i] - b[i];
                squares += (a[i] - b[i]) * (a[i] - b[i]);
            }
            const double mean = sum / double(a.size());
            return std::sqrt(squares / double(a.size()) - mean * mean);
        };
        EXPECT_NEAR(sigma(piped.distance, plain.distance), 0.1, 0.01);
        EXPECT_NEAR(sigma(piped.velocity, plain.velocity), 0.2, 0.02);
        EXPECT_GT(std::abs(plain.velocity.front()), 1);
    });
}

TEST(RadarPipelineTest, TracedCfarKeepsReturnsByItsFalseAlarmRate) {
    ForEachDevice([](Device& device) {
        Bench bench(device);
        const Prim cfar = bench.Add("Cfar", "Cfar");
        cfar.Set("falseAlarmRate", 0.5f);
        auto [plain, piped] = bench.Render();
        ASSERT_GT(plain.count, 0u);
        EXPECT_GT(piped.count, plain.count * 9 / 10);
        cfar.Set("falseAlarmRate", 1e-3f);
        std::tie(plain, piped) = bench.Render();
        EXPECT_LT(piped.count, plain.count / 10);
    });
}

TEST(RadarPipelineTest, CfarFindsATargetAndHoldsItsFalseAlarmRateOnNoise) {
    constexpr int64_t kSide = 96;
    constexpr int64_t kTarget = 40;
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        const Prim pipe = scene->Define<SensorPipeline>("/Pipe");
        const Prim returns = scene->Define<Shader>("/Pipe/Returns", {.id = "RadarReturns"});
        const Prim cfar = scene->Define<Shader>("/Pipe/Cfar", {.id = "Cfar"});
        cfar.Connect("returns", returns, "returns");
        cfar.Set("falseAlarmRate", 0.05f);
        pipe.Connect("detected", cfar, "returns");
        const Pipeline pipeline(scene, "/Pipe");

        std::mt19937 random(7);
        std::exponential_distribution<float> noise(1);
        std::vector<float> cells(kSide * kSide * 8, 0);
        for (int64_t cell = 0; cell < kSide * kSide; ++cell) {
            cells[cell * 8] = 1;
            cells[cell * 8 + 1] = 10;
            cells[cell * 8 + 2] = noise(random);
        }
        cells[(kTarget * kSide + kTarget) * 8 + 2] = 1000;
        PipelineRunDesc run;
        run.inputs.emplace("Returns",
                           device::Tensor(device.CreateBuffer({.label = "returns"}, cells), 0,
                                          device::DType::Float32, {1, kSide, kSide, 8}));
        const std::vector<float> detected = pipeline.Run(run).at("detected").ToHost<float>();
        ASSERT_EQ(detected.size(), cells.size());

        EXPECT_EQ(detected[(kTarget * kSide + kTarget) * 8], 1);
        int64_t alarms = 0;
        int64_t tested = 0;
        for (int64_t y = 0; y < kSide; ++y) {
            for (int64_t x = 0; x < kSide; ++x) {
                if (std::abs(x - kTarget) <= 3 && std::abs(y - kTarget) <= 3) {
                    continue;
                }
                alarms += detected[(y * kSide + x) * 8] != 0;
                ++tested;
            }
        }
        EXPECT_NEAR(double(alarms) / double(tested), 0.05, 0.01);
    });
}

}
}
