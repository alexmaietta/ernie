// SPDX-License-Identifier: BSD-3-Clause
#include <cmath>
#include <cstdint>
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

static_assert(kStatWords == ernie::kStatWords && kReduceGroups == ernie::kReduceGroups);

NodeRegistry Builtins() {
    NodeRegistry registry;
    RegisterBuiltinNodes(registry);
    return registry;
}

DetectorRecord Detector() {
    DetectorRecord detector{};
    detector.channels = 3;
    detector.pattern = uint32_t(DetectorPattern::Rgb);
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

std::vector<NodeDesc> ColorChain() {
    return {Node(1, "CameraSignal"), Node(2, "Integrate", {{"electronRate", 1, "electronRate"}}),
            Node(3, "Adc", {{"electrons", 2, "electrons"}}),
            Node(4, "ToColor", {{"digital", 3, "digital"}})};
}

TEST(PipelineReduceTest, MeterAveragesEachCaptureWhateverItsSize) {
    ernie::test::ForEachDevice([](device::Device& device) {
        const NodeRegistry registry = Builtins();
        constexpr uint32_t kWidth = 100;
        constexpr uint32_t kHeight = 37;
        Frame frame{.width = kWidth, .height = kHeight, .parameters = {0, 0.2f}, .captures = 2};
        double sums[3] = {0, 0, 0};
        for (uint32_t i = 0; i < kWidth * kHeight; ++i) {
            for (uint32_t c = 0; c < 3; ++c) {
                const float dn = float((i * (c + 3)) % 9000);
                frame.rates.push_back(dn);
                sums[c] += dn / 65535;
            }
        }
        for (const std::string mode : {"average", "spot"}) {
            std::vector<NodeDesc> nodes = ColorChain();
            nodes.push_back(Node(5, "Meter", {{"color", 4, "color"}}, {{"mode", mode}}));
            const std::vector<float> stat =
                RunPipeline(device, registry, Pipeline(nodes, {{"stat", 5, "stat"}}), Camera(),
                            Detector(), frame)
                    .at("stat");
            const size_t words = kStatStorageWords;
            ASSERT_EQ(stat.size(), 2 * words);
            for (uint32_t capture = 0; capture < 2; ++capture) {
                for (uint32_t word = 0; word < kStatWords; ++word) {
                    EXPECT_EQ(stat[capture * words + word], stat[word]) << capture << ", " << word;
                }
            }
            if (mode == "average") {
                EXPECT_FLOAT_EQ(stat[kStatWords - 1], float(kWidth * kHeight));
                for (uint32_t c = 0; c < 3; ++c) {
                    EXPECT_NEAR(stat[c], sums[c] / (kWidth * kHeight), 1e-5) << c;
                }
            } else {
                uint32_t inside = 0;
                for (uint32_t y = 0; y < kHeight; ++y) {
                    for (uint32_t x = 0; x < kWidth; ++x) {
                        const float dx = x + 0.5f - kWidth / 2.0f;
                        const float dy = y + 0.5f - kHeight / 2.0f;
                        const float r2 = (dx * dx + dy * dy) /
                                         (kWidth * kWidth / 4.0f + kHeight * kHeight / 4.0f);
                        inside += r2 <= 0.04f ? 1 : 0;
                    }
                }
                EXPECT_FLOAT_EQ(stat[kStatWords - 1], float(inside));
            }
        }
    });
}

TEST(PipelineReduceTest, MeteredWhiteBalanceTurnsACastGray) {
    ernie::test::ForEachDevice([](device::Device& device) {
        const NodeRegistry registry = Builtins();
        std::vector<NodeDesc> nodes = ColorChain();
        nodes.push_back(Node(5, "Meter", {{"color", 4, "color"}}));
        nodes.push_back(Node(6, "WhiteBalance", {{"color", 4, "color"}, {"stat", 5, "stat"}}));
        Frame frame{.width = 16, .height = 9, .parameters = {0, 0, 1, 1, 1}};
        for (uint32_t i = 0; i < 16 * 9; ++i) {
            frame.rates.insert(frame.rates.end(), {8000, 16000, 32000});
        }
        const std::vector<float> balanced =
            RunPipeline(device, registry, Pipeline(nodes, {{"color", 6, "color"}}), Camera(),
                        Detector(), frame)
                .at("color");
        for (uint32_t i = 0; i < 16 * 9; ++i) {
            EXPECT_NEAR(balanced[i * 3], 16000.0f / 65535, 1e-5f) << i;
            EXPECT_NEAR(balanced[i * 3 + 1], 16000.0f / 65535, 1e-5f) << i;
            EXPECT_NEAR(balanced[i * 3 + 2], 16000.0f / 65535, 1e-5f) << i;
        }
    });
}

}
}
