// SPDX-License-Identifier: BSD-3-Clause
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/sensor_pipeline/node.h"
#include "support/devices.h"
#include "unit/sensor_pipeline/registry.h"
#include "unit/sensor_pipeline/runner.h"

namespace ernie::sensor_pipeline {
namespace {

using test::Frame;
using test::Node;
using test::Pipeline;
using test::RunPipeline;

NodeRegistry Builtins() {
    NodeRegistry registry;
    RegisterBuiltinNodes(registry);
    return registry;
}

CameraRecord Camera() {
    CameraRecord camera{};
    camera.exposureTime = 0.01f;
    camera.iso = 200;
    return camera;
}

DetectorRecord Detector() {
    DetectorRecord detector{};
    detector.fullWell = 1e6f;
    detector.darkCurrentDoubling = 7;
    detector.temperature = 293.15f;
    detector.systemGain = 0.5f;
    detector.bitDepth = 8;
    detector.blackLevel = 10;
    return detector;
}

SensorPipelineDesc Noisy(std::string shot, std::string fixed) {
    return Pipeline({Node(1, "CameraSignal"),
                     Node(2, "Integrate", {{"electronRate", 1, "electronRate"}}),
                     Node(3, "Emva1288", {{"electrons", 2, "electrons"}},
                          {{"shot", shot}, {"fixedPatterns", fixed}})},
                    {{"electrons", 3, "electrons"}});
}

double Mean(const std::vector<float>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) / double(values.size());
}

double Variance(const std::vector<float>& values) {
    const double mean = Mean(values);
    double total = 0;
    for (const float value : values) {
        total += (value - mean) * (value - mean);
    }
    return total / double(values.size() - 1);
}

TEST(SensorPipelineNodesTest, ADeterministicChainFollowsTheCameraModel) {
    ernie::test::ForEachDevice([](device::Device& device) {
        const NodeRegistry registry = Builtins();
        DetectorRecord detector = Detector();
        detector.darkCurrent = 100;
        detector.temperature = 293.15f + 7;
        const SensorPipelineDesc desc = Pipeline(
            {Node(1, "CameraSignal"), Node(2, "Integrate", {{"electronRate", 1, "electronRate"}}),
             Node(3, "Emva1288", {{"electrons", 2, "electrons"}},
                  {{"shot", "off"}, {"fixedPatterns", "false"}}),
             Node(4, "Adc", {{"electrons", 3, "electrons"}}),
             Node(5, "ToColor", {{"digital", 4, "digital"}}),
             Node(6, "Encode", {{"color", 5, "color"}})},
            {{"raw", 4, "digital"}, {"display", 6, "display"}});
        Frame frame;
        for (uint32_t i = 0; i < frame.width * frame.channels; ++i) {
            frame.rates.push_back(1000.0f * float(i * i));
        }
        frame.parameters = {0, 0};
        const auto outputs = RunPipeline(device, registry, desc, Camera(), detector, frame);
        const std::vector<float>& raw = outputs.at("raw");
        const std::vector<float>& display = outputs.at("display");
        for (uint32_t i = 0; i < frame.rates.size(); ++i) {
            const float electrons = frame.rates[i] * 0.01f + 100 * 0.01f * 2;
            const float expected = std::min(std::round(electrons * 0.5f * 2 + 10), 255.0f);
            EXPECT_FLOAT_EQ(raw[i], expected) << i;
            const float linear = std::max(expected - 10, 0.0f) / 245;
            const float encoded = linear <= 0.0031308f
                                      ? 12.92f * linear
                                      : 1.055f * std::pow(linear, 1 / 2.4f) - 0.055f;
            EXPECT_NEAR(display[i], encoded, 1e-5f) << i;
        }
    });
}

TEST(SensorPipelineNodesTest, PoissonShotNoiseHasTheVarianceOfItsMean) {
    ernie::test::ForEachDevice([](device::Device& device) {
        const NodeRegistry registry = Builtins();
        for (const float mean : {4.0f, 60.0f}) {
            Frame frame{.width = 64, .height = 64, .channels = 1};
            frame.rates.assign(64 * 64, mean / 0.01f);
            frame.parameters = {0, 0};
            const std::vector<float> electrons =
                RunPipeline(device, registry, Noisy("poisson", "false"), Camera(), Detector(),
                            frame)
                    .at("electrons");
            EXPECT_NEAR(Mean(electrons), mean, 0.05 * mean) << mean;
            EXPECT_NEAR(Variance(electrons), mean, 0.12 * mean) << mean;
            for (const float value : electrons) {
                ASSERT_EQ(value, std::round(value));
            }
        }
    });
}

TEST(SensorPipelineNodesTest, FixedPatternsStayAcrossCapturesWhileTemporalNoiseChanges) {
    ernie::test::ForEachDevice([](device::Device& device) {
        const NodeRegistry registry = Builtins();
        DetectorRecord detector = Detector();
        detector.prnu = 0.1f;
        detector.dsnu = 1;
        Frame frame{.width = 16, .height = 1, .channels = 1};
        frame.rates.assign(16, 100 / 0.01f);
        frame.parameters = {0, 0};
        const auto run = [&](uint32_t capture, const DetectorRecord& record) {
            frame.capture = capture;
            return RunPipeline(device, registry, Noisy("off", "true"), Camera(), record, frame)
                .at("electrons");
        };
        const std::vector<float> first = run(0, detector);
        EXPECT_EQ(run(1, detector), first);
        EXPECT_GT(Variance(first), 1);
        DetectorRecord other = detector;
        other.seed = 3;
        EXPECT_NE(run(0, other), first);
        detector.readNoise = 2;
        EXPECT_NE(run(1, detector), run(0, detector));
    });
}

TEST(SensorPipelineNodesTest, TheFullWellClipsCollectedCharge) {
    ernie::test::ForEachDevice([](device::Device& device) {
        const NodeRegistry registry = Builtins();
        DetectorRecord detector = Detector();
        detector.fullWell = 50;
        Frame frame{.width = 4, .height = 1, .channels = 1};
        frame.rates = {1000, 4000, 6000, 100000};
        frame.parameters = {0, 0};
        const std::vector<float> electrons =
            RunPipeline(device, registry, Noisy("off", "false"), Camera(), detector, frame)
                .at("electrons");
        EXPECT_EQ(electrons, (std::vector<float>{10, 40, 50, 50}));

        SensorPipelineDesc unclipped = Noisy("off", "false");
        unclipped.nodes[2].uniforms.emplace_back("clipping", "none");
        EXPECT_EQ(
            RunPipeline(device, registry, unclipped, Camera(), detector, frame).at("electrons"),
            (std::vector<float>{10, 40, 60, 1000}));
    });
}

}
}
