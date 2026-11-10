// SPDX-License-Identifier: BSD-3-Clause
#include <cmath>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/pipeline/node.h"
#include "support/devices.h"
#include "unit/pipeline/registry.h"
#include "unit/pipeline/runner.h"

namespace ernie::pipeline {
namespace {

using test::Frame;
using test::Node;
using test::Pipeline;
using test::RunPipeline;

const char* const kDrawNode = R"(
import ernie.pipeline;

using namespace ernie;

struct TestDraws {
    static void Evaluate(PointContext context, out Electrons draws) {
        NoiseStream first = context.Noise(0);
        NoiseStream second = context.Noise(1);
        NoiseStream fixedStream = context.FixedNoise(0);
        Channels channels = Channels(3);
        channels[0] = first.Normal();
        channels[1] = second.Normal();
        channels[2] = fixedStream.Normal();
        draws = Electrons.Make(channels);
    }
}
)";

constexpr uint32_t kSide = 64;
constexpr uint32_t kElements = kSide * kSide;

std::vector<float> Draws(device::Device& device, uint32_t capture, uint32_t captures) {
    device::RegisterFile("pipeline_test/draws.slang", kDrawNode);
    NodeRegistry registry;
    registry.Add({.id = "Draws",
                  .outputs = {{"draws", SignalType::Electrons}},
                  .layout = LayoutRule::Color,
                  .code = {.module = "pipeline_test.draws", .type = "TestDraws"}});
    Frame frame{.width = kSide, .height = kSide, .capture = capture, .captures = captures};
    return RunPipeline(device, registry, Pipeline({Node(1, "Draws")}, {{"draws", 1, "draws"}}), {},
                       {}, frame)
        .at("draws");
}

double Channel(const std::vector<float>& draws, uint32_t element, uint32_t channel) {
    return draws[size_t(element) * 3 + channel];
}

double Correlation(const std::vector<float>& draws, uint32_t a, uint32_t b) {
    double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
    for (uint32_t e = 0; e < kElements; ++e) {
        const double x = Channel(draws, e, a);
        const double y = Channel(draws, e, b);
        sa += x;
        sb += y;
        saa += x * x;
        sbb += y * y;
        sab += x * y;
    }
    const double n = kElements;
    return (sab / n - sa / n * sb / n) /
           std::sqrt((saa / n - sa / n * sa / n) * (sbb / n - sb / n * sb / n));
}

TEST(PipelineNoiseTest, DrawsAreIndependentReproducibleAndKeyedByCapture) {
    static std::vector<std::vector<float>> perDevice;
    ernie::test::ForEachDevice([](device::Device& device) {
        const std::vector<float> batch = Draws(device, 0, 3);
        const std::vector<float> alone = Draws(device, 1, 1);
        ASSERT_EQ(batch.size(), size_t(kElements) * 3 * 3);
        const std::vector<float> second(batch.begin() + kElements * 3,
                                        batch.begin() + kElements * 6);
        EXPECT_EQ(alone, second);

        const std::vector<float> first(batch.begin(), batch.begin() + kElements * 3);
        for (uint32_t channel = 0; channel < 3; ++channel) {
            double mean = 0, square = 0;
            for (uint32_t e = 0; e < kElements; ++e) {
                mean += Channel(first, e, channel);
                square += Channel(first, e, channel) * Channel(first, e, channel);
            }
            mean /= kElements;
            EXPECT_NEAR(mean, 0, 0.05) << channel;
            EXPECT_NEAR(square / kElements - mean * mean, 1, 0.08) << channel;
        }
        EXPECT_LT(std::abs(Correlation(first, 0, 1)), 0.05);
        EXPECT_LT(std::abs(Correlation(first, 0, 2)), 0.05);
        EXPECT_LT(std::abs(Correlation(first, 1, 2)), 0.05);

        uint32_t sameTemporal = 0;
        for (uint32_t e = 0; e < kElements; ++e) {
            EXPECT_EQ(Channel(first, e, 2), Channel(second, e, 2));
            sameTemporal += Channel(first, e, 0) == Channel(second, e, 0) ? 1 : 0;
        }
        EXPECT_LT(sameTemporal, 4u);
        perDevice.push_back(first);
        for (size_t i = 0; i < first.size(); ++i) {
            ASSERT_NEAR(perDevice.back()[i], perDevice.front()[i], 1e-4f) << i;
        }
    });
}

}
}
