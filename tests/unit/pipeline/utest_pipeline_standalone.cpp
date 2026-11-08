// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/render/standalone.h"

#include <cmath>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/pipeline/builtins.h"
#include "ernie/util/error.h"
#include "support/devices.h"
#include "unit/pipeline/registry.h"
#include "unit/pipeline/runner.h"

namespace ernie::pipeline {
namespace {

using test::Frame;
using test::Node;
using test::Pipeline;
using test::RunPipeline;

constexpr uint32_t kWidth = 12;
constexpr uint32_t kHeight = 5;

PipelineDesc Chain() {
    PipelineDesc desc = Pipeline({Node(1, "CameraSignal"),
                                  Node(2, "Integrate", {{"electronRate", 1, "electronRate"}}),
                                  Node(3, "Emva1288", {{"electrons", 2, "electrons"}}),
                                  Node(4, "Adc", {{"electrons", 3, "electrons"}}),
                                  Node(5, "Gain", {{"digital", 4, "digital"}})},
                                 {{"raw", 5, "digital"}});
    desc.nodes[4].values = {{"gain", {2}}};
    return desc;
}

std::vector<float> Rates() {
    std::vector<float> rates;
    for (uint32_t i = 0; i < kWidth * kHeight * 3; ++i) {
        rates.push_back(float(i % 23) * 400);
    }
    return rates;
}

device::Tensor Input(device::Device& device, const std::vector<float>& rates, int64_t channels) {
    return device::Tensor(device.CreateBuffer({.label = "input"}, rates), 0, device::DType::Float32,
                          {1, kHeight, kWidth, channels});
}

DetectorDesc Detector() {
    return {.readNoise = 3, .prnu = 0.05f, .dsnu = 1, .systemGain = 0.5f, .bitDepth = 12};
}

TEST(PipelineStandaloneTest, RunsMatchTheTracedPipelineAndSessionsAdvanceCaptures) {
    ernie::test::ForEachDevice([](device::Device& device) {
        const ernie::Pipeline pipeline(device, Chain());
        const std::vector<float> rates = Rates();
        PipelineRunDesc run{.inputs = {{"/Pipe/CameraSignal1", Input(device, rates, 3)}},
                            .detector = Detector(),
                            .exposureTime = 0.01f,
                            .iso = 100,
                            .capture = 1,
                            .identity = 7};
        const std::vector<float> standalone = pipeline.Run(run).at("raw").ToHost<float>();

        CameraRecord camera{};
        camera.exposureTime = 0.01f;
        camera.iso = 100;
        DetectorRecord detector{};
        detector.channels = 3;
        detector.fullWell = 10000;
        detector.readNoise = 3;
        detector.darkCurrent = 1;
        detector.darkCurrentDoubling = 7;
        detector.temperature = 293.15f;
        detector.prnu = 0.05f;
        detector.dsnu = 1;
        detector.systemGain = 0.5f;
        detector.bitDepth = 12;
        Frame frame{.width = kWidth,
                    .height = kHeight,
                    .rates = rates,
                    .parameters = {0, 0, 2},
                    .capture = 1,
                    .identity = 7};
        const std::vector<float> traced =
            RunPipeline(device, BuiltinNodes(), Chain(), camera, detector, frame).at("raw");
        EXPECT_EQ(standalone, traced);

        PipelineSession session(pipeline);
        run.capture = 99;
        const std::vector<float> first = session.Run(run).at("raw").ToHost<float>();
        const std::vector<float> second = session.Run(run).at("raw").ToHost<float>();
        EXPECT_EQ(session.Captures(), 2u);
        EXPECT_EQ(second, standalone);
        EXPECT_NE(first, second);
    });
}

TEST(PipelineStandaloneTest, AutoExposureSettlesAcrossASessionButNotAcrossRuns) {
    ernie::test::ForEachDevice([](device::Device& device) {
        const ernie::Pipeline pipeline(
            device,
            Pipeline({Node(1, "CameraSignal"),
                      Node(2, "Integrate", {{"electronRate", 1, "electronRate"}}),
                      Node(3, "Adc", {{"electrons", 2, "electrons"}}),
                      Node(4, "ToColor", {{"digital", 3, "digital"}}),
                      Node(5, "Meter", {{"color", 4, "color"}}),
                      Node(6, "AutoExposure", {{"color", 4, "color"}, {"stat", 5, "stat"}})},
                     {{"color", 6, "color"}}));
        const std::vector<float> rates(size_t(kWidth) * kHeight * 3, 2949);
        const PipelineRunDesc run{.inputs = {{"/Pipe/CameraSignal1", Input(device, rates, 3)}},
                                  .detector = {.systemGain = 1, .bitDepth = 16}};
        const float linear = 2949.0f / 65535;
        const float ideal = 0.18f / linear;
        PipelineSession session(pipeline);
        float gain = 1;
        for (int frame = 0; frame < 6; ++frame) {
            const std::vector<float> color = session.Run(run).at("color").ToHost<float>();
            EXPECT_NEAR(color[0] / linear, gain, 1e-3f * gain) << frame;
            EXPECT_NEAR(pipeline.Run(run).at("color").ToHost<float>()[0] / linear, 1, 1e-5f);
            gain *= std::sqrt(ideal / gain);
        }
        EXPECT_NEAR(gain, ideal, 0.1f);
    });
}

TEST(PipelineStandaloneTest, InputsMustBeGivenAndShapedForTheDetector) {
    ernie::test::ForEachDevice([](device::Device& device) {
        const ernie::Pipeline pipeline(device, Chain());
        EXPECT_THROW(pipeline.Run({}), Error);
        PipelineRunDesc run{.inputs = {{"/Pipe/CameraSignal1", Input(device, Rates(), 3)}},
                            .detector = {.pattern = DetectorPattern::Mono}};
        EXPECT_THROW(pipeline.Run(run), Error);
        run.detector.pattern = DetectorPattern::Rgb;
        EXPECT_NO_THROW(pipeline.Run(run));
    });
}

}
}
