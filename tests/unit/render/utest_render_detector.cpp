// SPDX-License-Identifier: BSD-3-Clause
#include <cmath>
#include <cstdint>
#include <numbers>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/render/render_output.h"
#include "ernie/render/renderer.h"
#include "ernie/scene/scene.h"
#include "ernie/util/error.h"
#include "support/devices.h"

namespace ernie {
namespace {

using device::Device;
using test::ForEachDevice;

constexpr uint32_t kSize = 8;
constexpr float kFocalLength = 10;
constexpr float kFilm = 20.955f;
const std::vector<std::string> kProducts = {"/Render/View"};

struct Rates {
    std::vector<int64_t> shape;
    std::vector<float> values;
    float hdr = 0;
};

Ref<Scene> UniformSky(Device& device) {
    const Ref<Scene> scene = Scene::Create(device);
    scene->Define<DomeLight>("/Sky", {});
    scene->Define<Camera>("/Cam", {.detectorQeRed = {{300, 0.5f}, {900, 0.5f}},
                                   .detectorQeGreen = {{300, 0.25f}, {900, 0.25f}},
                                   .detectorQeBlue = {{300, 1}, {900, 1}},
                                   .detectorQeClear = {{300, 0.8f}, {900, 0.8f}},
                                   .focalLength = kFocalLength});
    scene->Define<Scope>("/Render");
    scene->Define<RenderVar>("/Render/Rate", {.sourceName = RenderVarSource::ElectronRate});
    scene->Define<RenderVar>("/Render/Color", {.sourceName = RenderVarSource::HdrColor});
    scene->Define<RenderProduct>("/Render/View",
                                 {.renderMode = RenderMode::RealTime,
                                  .camera = {"/Cam"},
                                  .resolution = {kSize, kSize},
                                  .orderedVars = {"/Render/Rate", "/Render/Color"}});
    return scene;
}

Rates Render(Renderer& renderer) {
    RenderOutput output = renderer.Step(kProducts, 0);
    output.Wait();
    const auto& frame = output["/Render/View"].frames.at(0);
    const auto& tensor = frame.Source("ElectronRate").tensor;
    Rates rates;
    rates.shape.assign(tensor.Shape().begin(), tensor.Shape().end());
    rates.values = tensor.ToHost<float>();
    rates.hdr =
        frame.Source("HdrColor").tensor.ToHost<float>()[(kSize / 2 * kSize + kSize / 2) * 4];
    return rates;
}

float CenterFalloff() {
    const float film = (0.5f / kSize * 2) * kFilm / 2;
    const float cosine = kFocalLength / std::sqrt(kFocalLength * kFocalLength + 2 * film * film);
    return std::pow(cosine, 4.0f);
}

float BandRate(float nanometers, float width, float qe) {
    const float nit = 1.0f / (683.0f * 106.857f);
    const float photonsPerJoule = nanometers * 1e-9f / (6.62607015e-34f * 2.99792458e8f);
    const float pixel = kFilm / kSize * 1e-3f;
    return qe * nit * width * photonsPerJoule * pixel * pixel * std::numbers::pi_v<float> / 4 *
           CenterFalloff();
}

TEST(DetectorTest, ElectronRateFollowsTheCameraEquation) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = UniformSky(device);
        Prim camera = *scene->Find("/Cam");
        Renderer renderer(scene);
        const uint32_t center = kSize / 2 * kSize + kSize / 2;

        const Rates rgb = Render(renderer);
        ASSERT_NEAR(rgb.hdr, 1, 1e-3f);
        EXPECT_EQ(rgb.shape, (std::vector<int64_t>{1, kSize, kSize, 3}));
        float visible = 0;
        for (uint32_t band = 0; band < 8; ++band) {
            visible += BandRate(380 + 50 * (band + 0.5f), 50, 1);
        }
        const float qes[] = {0.5f, 0.25f, 1};
        for (uint32_t channel = 0; channel < 3; ++channel) {
            EXPECT_NEAR(rgb.values[center * 3 + channel] / (qes[channel] * visible), 1, 1e-3f)
                << channel;
        }
        EXPECT_LT(rgb.values[0], rgb.values[center * 3]);

        camera.Set(Camera::kExposureTime, 0.01f);
        const Rates shorter = Render(renderer);
        EXPECT_NEAR(shorter.hdr, 0.01f, 1e-4f);
        EXPECT_NEAR(shorter.values[center * 3] / rgb.values[center * 3], 1, 1e-4f);

        camera.Set(Camera::kExposureFStop, 2.0f);
        EXPECT_NEAR(Render(renderer).values[center * 3] / rgb.values[center * 3], 0.25f, 1e-4f);
        camera.Set(Camera::kExposureFStop, 1.0f);

        camera.Set(Camera::kDetectorPattern, DetectorPattern::Mono);
        const Rates mono = Render(renderer);
        EXPECT_EQ(mono.shape, (std::vector<int64_t>{1, kSize, kSize}));
        EXPECT_NEAR(mono.values[center] / (0.8f * visible), 1, 1e-3f);

        camera.Set(Camera::kDetectorPattern, DetectorPattern::Bands);
        camera.Set(Camera::kDetectorSpectralRange, Float2{700, 1100});
        const Rates bands = Render(renderer);
        EXPECT_EQ(bands.shape, (std::vector<int64_t>{1, kSize, kSize, 8}));
        for (uint32_t band = 0; band < 8; ++band) {
            const float expected = band < 4 ? BandRate(700 + 50 * (band + 0.5f), 50, 0.8f) : 0;
            EXPECT_NEAR(bands.values[center * 8 + band], expected, 1e-3f * BandRate(725, 50, 1))
                << band;
        }
    });
}

TEST(DetectorTest, StackedCamerasWithDifferentChannelCountsCannotWriteElectronRate) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = UniformSky(device);
        scene->Define<Camera>("/Mono", {.detectorPattern = DetectorPattern::Mono});
        scene->Find("/Render/View")
            ->Set(RenderProduct::kCamera, std::vector<std::string>{"/Cam", "/Mono"});
        Renderer renderer(scene);
        EXPECT_THROW(renderer.Step(kProducts, 0), Error);
    });
}

}  // namespace
}  // namespace ernie
