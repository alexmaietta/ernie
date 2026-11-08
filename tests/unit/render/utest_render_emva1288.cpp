// SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/render/renderer.h"
#include "ernie/scene/scene.h"
#include "support/devices.h"

namespace ernie {
namespace {

constexpr uint32_t kSize = 64;
constexpr float kSystemGain = 1;
constexpr float kFullWell = 3000;
constexpr float kReadNoise = 3;
constexpr float kDsnu = 5;
constexpr float kBlackLevel = 100;
const std::vector<std::string> kProducts = {"/Render/View"};

struct Moments {
    double mean = 0;
    double variance = 0;
};

Moments Measure(const std::vector<double>& values) {
    Moments moments;
    for (const double value : values) {
        moments.mean += value;
    }
    moments.mean /= double(values.size());
    for (const double value : values) {
        moments.variance += (value - moments.mean) * (value - moments.mean);
    }
    moments.variance /= double(values.size() - 1);
    return moments;
}

class Bench {
  public:
    explicit Bench(device::Device& device) : m_scene(Scene::Create(device)) {
        Scene& scene = *m_scene;
        scene.Define<DomeLight>("/Sky", {});
        m_camera = scene.Define<Camera>("/Cam", {.focalLength = 50});
        m_camera.Set(DetectorAPI::kDetectorPattern, DetectorPattern::Mono);
        m_camera.Set(DetectorAPI::kDetectorSystemGain, kSystemGain);
        m_camera.Set(DetectorAPI::kDetectorFullWell, kFullWell);
        m_camera.Set(DetectorAPI::kDetectorReadNoise, kReadNoise);
        m_camera.Set(DetectorAPI::kDetectorDsnu, kDsnu);
        m_camera.Set(DetectorAPI::kDetectorDarkCurrent, 0.0f);
        m_camera.Set(DetectorAPI::kDetectorBlackLevel, kBlackLevel);
        m_camera.Set(DetectorAPI::kDetectorBitDepth, 12);
        m_camera.Set(Camera::kPipelineBinding, std::vector<std::string>{"/Raw"});
        const Prim pipe = scene.Define<SensorPipeline>("/Raw");
        const Prim signal = scene.Define<Shader>("/Raw/Signal", {.id = "CameraSignal"});
        const Prim integrate = scene.Define<Shader>("/Raw/Integrate", {.id = "Integrate"});
        integrate.Connect("electronRate", signal, "electronRate");
        const Prim noise = scene.Define<Shader>("/Raw/Noise", {.id = "Emva1288"});
        noise.Connect("electrons", integrate, "electrons");
        const Prim adc = scene.Define<Shader>("/Raw/Adc", {.id = "Adc"});
        adc.Connect("electrons", noise, "electrons");
        pipe.Connect("raw", adc, "digital");
        scene.Define<Scope>("/Render");
        scene.Define<RenderVar>("/Render/Rate", {.sourceName = "ElectronRate"});
        scene.Define<RenderVar>("/Render/Raw", {.sourceName = "raw"});
        scene.Define<RenderProduct>("/Render/View",
                                    {.renderMode = RenderMode::RealTime,
                                     .camera = {"/Cam"},
                                     .resolution = {kSize, kSize},
                                     .orderedVars = {"/Render/Rate", "/Render/Raw"}});
        m_renderer.emplace(m_scene);
    }

    std::vector<double> Capture(float exposureTime, const char* source = "raw") {
        m_camera.Set(Camera::kExposureTime, exposureTime);
        RenderOutput output = m_renderer->Step(kProducts, 0);
        output.Wait();
        const std::vector<float> values =
            output["/Render/View"].frames.at(0).Source(source).tensor.ToHost<float>();
        return {values.begin(), values.end()};
    }

    Moments Temporal(float exposureTime) {
        const std::vector<double> first = Capture(exposureTime);
        const std::vector<double> second = Capture(exposureTime);
        std::vector<double> difference(first.size());
        std::vector<double> sum(first.size());
        for (size_t i = 0; i < first.size(); ++i) {
            difference[i] = first[i] - second[i];
            sum[i] = (first[i] + second[i]) / 2;
        }
        return {Measure(sum).mean, Measure(difference).variance / 2};
    }

  private:
    Ref<Scene> m_scene;
    Prim m_camera;
    std::optional<Renderer> m_renderer;
};

TEST(Emva1288Test, PhotonTransferRecoversTheDetector) {
    test::ForEachDevice([](device::Device& device) {
        Bench bench(device);
        const std::vector<double> rates = bench.Capture(1, "ElectronRate");
        const double brightest = *std::ranges::max_element(rates);
        const double dimmest = *std::ranges::min_element(rates);
        ASSERT_GT(dimmest, 0);
        ASSERT_LT(brightest / dimmest, 1.2);

        std::vector<std::vector<double>> darks;
        for (int frame = 0; frame < 4; ++frame) {
            darks.push_back(bench.Capture(0));
        }
        std::vector<double> difference(darks[0].size());
        std::vector<double> average(darks[0].size());
        for (size_t i = 0; i < average.size(); ++i) {
            difference[i] = darks[0][i] - darks[1][i];
            for (const std::vector<double>& dark : darks) {
                average[i] += dark[i] / double(darks.size());
            }
        }
        const double darkVariance = Measure(difference).variance / 2;
        const Moments dark = Measure(average);
        EXPECT_NEAR(dark.mean, kBlackLevel, 0.5);
        EXPECT_NEAR(std::sqrt(darkVariance - 1.0 / 12) / kSystemGain, kReadNoise,
                    0.05 * kReadNoise);
        const double dsnu =
            std::sqrt(dark.variance - darkVariance / double(darks.size())) / kSystemGain;
        EXPECT_NEAR(dsnu, kDsnu, 0.05 * kDsnu);

        std::vector<double> signals;
        std::vector<double> variances;
        for (const double fraction : {0.1, 0.3, 0.5, 0.7}) {
            const Moments moments = bench.Temporal(float(fraction * kFullWell / brightest));
            signals.push_back(moments.mean - dark.mean);
            variances.push_back(moments.variance - darkVariance);
        }
        double xx = 0;
        double xy = 0;
        for (size_t i = 0; i < signals.size(); ++i) {
            xx += signals[i] * signals[i];
            xy += signals[i] * variances[i];
        }
        EXPECT_NEAR(xy / xx, kSystemGain, 0.03 * kSystemGain);

        const std::vector<double> saturated = bench.Capture(float(3 * kFullWell / dimmest));
        EXPECT_NEAR((Measure(saturated).mean - kBlackLevel) / kSystemGain, kFullWell,
                    0.01 * kFullWell);
    });
}

}
}
