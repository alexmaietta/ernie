// SPDX-License-Identifier: BSD-3-Clause
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/pipeline/builtins.h"
#include "ernie/scene/scene.h"
#include "support/devices.h"
#include "unit/pipeline/registry.h"
#include "unit/pipeline/runner.h"

namespace ernie::pipeline {
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

DetectorRecord Detector(uint32_t channels, DetectorPattern pattern) {
    DetectorRecord detector{};
    detector.channels = channels;
    detector.pattern = uint32_t(pattern);
    detector.fullWell = 1e9f;
    detector.systemGain = 1;
    detector.bitDepth = 16;
    return detector;
}

CameraRecord Camera() {
    CameraRecord camera{};
    camera.exposureTime = 1;
    camera.iso = 100;
    return camera;
}

PipelineDesc RawThen(std::vector<NodeDesc> tail, std::initializer_list<test::Link> outputs) {
    std::vector<NodeDesc> nodes = {Node(1, "CameraSignal", {}, {{"layout", "mosaic"}}),
                                   Node(2, "Integrate", {{"electronRate", 1, "electronRate"}}),
                                   Node(3, "Adc", {{"electrons", 2, "electrons"}})};
    nodes.insert(nodes.end(), tail.begin(), tail.end());
    return Pipeline(std::move(nodes), outputs);
}

TEST(PipelineIspTest, DemosaicReconstructsLinearRampsOfABayerMosaic) {
    ernie::test::ForEachDevice([](device::Device& device) {
        const NodeRegistry registry = Builtins();
        constexpr uint32_t kSide = 8;
        const auto truth = [](uint32_t channel, uint32_t x, uint32_t y) {
            return float(100 + 10 * (channel + 1) * x + 7 * (channel + 2) * y);
        };
        Frame frame{.width = kSide,
                    .height = kSide,
                    .channels = 3,
                    .parameters = {0},
                    .pattern = uint32_t(DetectorPattern::Rggb),
                    .mosaic = true};
        for (uint32_t y = 0; y < kSide; ++y) {
            for (uint32_t x = 0; x < kSide; ++x) {
                const uint32_t site = (x & 1) | ((y & 1) << 1);
                const uint32_t channel = site == 0 ? 0 : site == 3 ? 2 : 1;
                frame.rates.push_back(truth(channel, x, y));
            }
        }
        const std::vector<float> full =
            RunPipeline(device, registry,
                        RawThen({Node(4, "Demosaic", {{"digital", 3, "digital"}})},
                                {{"full", 4, "digital"}}),
                        Camera(), Detector(3, DetectorPattern::Rggb), frame)
                .at("full");
        ASSERT_EQ(full.size(), size_t(kSide) * kSide * 3);
        for (uint32_t y = 1; y + 1 < kSide; ++y) {
            for (uint32_t x = 1; x + 1 < kSide; ++x) {
                for (uint32_t channel = 0; channel < 3; ++channel) {
                    EXPECT_FLOAT_EQ(full[(y * kSide + x) * 3 + channel], truth(channel, x, y))
                        << x << ", " << y << ", " << channel;
                }
            }
        }
    });
}

TEST(PipelineIspTest, ThePsfConservesEnergyAndKeepsMosaicChannelsApart) {
    ernie::test::ForEachDevice([](device::Device& device) {
        const NodeRegistry registry = Builtins();
        constexpr uint32_t kSide = 16;
        const PipelineDesc desc =
            Pipeline({Node(1, "CameraSignal"),
                      Node(2, "GaussianPsf", {{"electronRate", 1, "electronRate"}})},
                     {{"blurred", 2, "electronRate"}});
        Frame frame{.width = kSide, .height = kSide, .channels = 1, .parameters = {1.2f}};
        frame.rates.assign(kSide * kSide, 0);
        frame.rates[8 * kSide + 8] = 1000;
        const std::vector<float> blurred =
            RunPipeline(device, registry, desc, Camera(), Detector(1, DetectorPattern::Mono), frame)
                .at("blurred");
        EXPECT_NEAR(std::accumulate(blurred.begin(), blurred.end(), 0.0), 1000, 0.05);
        EXPECT_LT(blurred[8 * kSide + 8], 200);
        EXPECT_GT(blurred[8 * kSide + 9], 50);

        frame.pattern = uint32_t(DetectorPattern::Rggb);
        frame.mosaic = true;
        PipelineDesc mosaic = desc;
        mosaic.nodes[0].uniforms = {{"layout", "mosaic"}};
        const std::vector<float> sameSite = RunPipeline(device, registry, mosaic, Camera(),
                                                        Detector(3, DetectorPattern::Rggb), frame)
                                                .at("blurred");
        EXPECT_EQ(sameSite[8 * kSide + 9], 0);
        EXPECT_EQ(sameSite[9 * kSide + 8], 0);
        EXPECT_GT(sameSite[8 * kSide + 10], 0);
    });
}

TEST(PipelineIspTest, ColorNodesMatchHandValues) {
    ernie::test::ForEachDevice([](device::Device& device) {
        const NodeRegistry registry = Builtins();
        const PipelineDesc desc = Pipeline(
            {Node(1, "CameraSignal"), Node(2, "Integrate", {{"electronRate", 1, "electronRate"}}),
             Node(3, "Adc", {{"electrons", 2, "electrons"}}),
             Node(4, "ToColor", {{"digital", 3, "digital"}}),
             Node(5, "WhiteBalance", {{"color", 4, "color"}}),
             Node(6, "Ccm", {{"color", 5, "color"}}),
             Node(7, "Tone", {{"color", 6, "color"}}, {{"curve", "reinhard"}})},
            {{"color", 7, "color"}});
        Frame frame{.width = 2, .height = 1, .channels = 3};
        frame.rates = {6554, 13107, 32768, 65535, 0, 19661};
        frame.parameters = {0, 2, 1, 0.5f, 1, 0.5f, 0, 0, 1, 0, 0.25f, 0.25f, 0.5f};
        const std::vector<float> color =
            RunPipeline(device, registry, desc, Camera(), Detector(3, DetectorPattern::Rgb), frame)
                .at("color");
        for (uint32_t pixel = 0; pixel < 2; ++pixel) {
            float linear[3];
            for (uint32_t c = 0; c < 3; ++c) {
                linear[c] = std::round(frame.rates[pixel * 3 + c]) / 65535;
            }
            const float balanced[3] = {linear[0] * 2, linear[1], linear[2] * 0.5f};
            const float corrected[3] = {balanced[0] + 0.5f * balanced[1], balanced[1],
                                        0.25f * balanced[0] + 0.25f * balanced[1] +
                                            0.5f * balanced[2]};
            for (uint32_t c = 0; c < 3; ++c) {
                EXPECT_NEAR(color[pixel * 3 + c], corrected[c] / (1 + corrected[c]), 1e-5f)
                    << pixel << ", " << c;
            }
        }
    });
}

TEST(PipelineIspTest, VignetteDarkensCornersAndDenoiseKeepsEdges) {
    ernie::test::ForEachDevice([](device::Device& device) {
        const NodeRegistry registry = Builtins();
        constexpr uint32_t kSide = 8;
        Frame frame{.width = kSide, .height = kSide, .channels = 1, .parameters = {0.5f}};
        frame.rates.assign(kSide * kSide, 100);
        const std::vector<float> vignetted =
            RunPipeline(device, registry,
                        Pipeline({Node(1, "CameraSignal"),
                                  Node(2, "Vignette", {{"electronRate", 1, "electronRate"}})},
                                 {{"out", 2, "electronRate"}}),
                        Camera(), Detector(1, DetectorPattern::Mono), frame)
                .at("out");
        const float corner = 1 - 0.5f * (3.5f * 3.5f * 2) / (4 * 4 * 2);
        EXPECT_NEAR(vignetted[0], 100 * corner, 1e-3f);
        EXPECT_NEAR(vignetted[3 * kSide + 3], 100 * (1 - 0.5f * 0.5f / 32), 1e-3f);

        Frame edge{.width = kSide, .height = kSide, .channels = 3, .parameters = {0, 1, 0.01f}};
        for (uint32_t y = 0; y < kSide; ++y) {
            for (uint32_t x = 0; x < kSide; ++x) {
                for (uint32_t c = 0; c < 3; ++c) {
                    edge.rates.push_back(x < kSide / 2 ? 6554 : 52428);
                }
            }
        }
        const std::vector<float> denoised =
            RunPipeline(device, registry,
                        Pipeline({Node(1, "CameraSignal"),
                                  Node(2, "Integrate", {{"electronRate", 1, "electronRate"}}),
                                  Node(3, "Adc", {{"electrons", 2, "electrons"}}),
                                  Node(4, "ToColor", {{"digital", 3, "digital"}}),
                                  Node(5, "Denoise", {{"color", 4, "color"}})},
                                 {{"out", 5, "color"}}),
                        Camera(), Detector(3, DetectorPattern::Rgb), edge)
                .at("out");
        EXPECT_NEAR(denoised[(3 * kSide + 3) * 3], 6554.0f / 65535, 1e-4f);
        EXPECT_NEAR(denoised[(3 * kSide + 4) * 3], 52428.0f / 65535, 1e-4f);
    });
}

}
}
