// SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/render/renderer.h"
#include "ernie/render/sensor_pipeline_runner.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/tables/semantics.h"
#include "ernie/util/profiler.h"
#include "support/devices.h"
#include "unit/sensor_pipeline/registry.h"

namespace ernie {
namespace {

constexpr uint32_t kWidth = 6;
constexpr uint32_t kHeight = 4;
constexpr float kExposureTime = 0.5f;

Prim Node(Scene& scene, const std::string& path, const char* id) {
    return scene.Define<Shader>(path, {.id = id});
}

Ref<Scene> Sky(device::Device& device, DetectorPattern pattern) {
    Ref<Scene> scene = Scene::Create(device);
    scene->Define<DomeLight>("/Sky", {});
    const Prim camera = scene->Define<Camera>("/Cam", {.focalLength = 10});
    camera.Set(Camera::kExposureTime, kExposureTime);
    camera.Set(DetectorAPI::kDetectorPattern, pattern);
    scene->Define<Scope>("/Pipes");
    scene->Define<Scope>("/Render");
    scene->Define<RenderVar>("/Render/Rate", {.sourceName = "ElectronRate"});
    return scene;
}

void DefineRaw(Scene& scene) {
    const Prim pipe = scene.Define<SensorPipeline>("/Pipes/Raw");
    const Prim signal = Node(scene, "/Pipes/Raw/Signal", "CameraSignal");
    const Prim integrate = Node(scene, "/Pipes/Raw/Integrate", "Integrate");
    integrate.Connect("electronRate", signal, "electronRate");
    const Prim emva = Node(scene, "/Pipes/Raw/Noise", "Emva1288");
    emva.Connect("electrons", integrate, "electrons");
    const Prim adc = Node(scene, "/Pipes/Raw/Adc", "Adc");
    adc.Connect("electrons", emva, "electrons");
    const Prim gain = Node(scene, "/Pipes/Raw/Gain", "Gain");
    gain.Set(Gain::kGain, 2.0f);
    gain.Connect("digital", adc, "digital");
    pipe.Connect("raw", gain, "digital");
    scene.Define<RenderVar>("/Render/Raw", {.sourceName = "raw"});
}

sensor_pipeline::SensorPipelineDesc RawDesc() {
    using sensor_pipeline::test::Link;
    sensor_pipeline::SensorPipelineDesc desc{.prim = 1, .path = "/Pipes/Raw"};
    const auto node = [&](uint32_t prim, const char* name, const char* id,
                          std::vector<Link> links) {
        sensor_pipeline::NodeDesc& added = desc.nodes.emplace_back(sensor_pipeline::NodeDesc{
            .prim = prim, .path = std::string("/Pipes/Raw/") + name, .id = id});
        for (const Link& link : links) {
            added.inputs.push_back({link.input, link.source, "", link.output});
        }
    };
    node(2, "Signal", "CameraSignal", {});
    node(3, "Integrate", "Integrate", {{"electronRate", 2, "electronRate"}});
    node(4, "Noise", "Emva1288", {{"electrons", 3, "electrons"}});
    node(5, "Adc", "Adc", {{"electrons", 4, "electrons"}});
    node(6, "Gain", "Gain", {{"digital", 5, "digital"}});
    desc.nodes.back().values = {{"gain", {2}}};
    desc.outputs = {{"raw", 6, "", "digital"}};
    return desc;
}

std::vector<float> Host(const FrameOutput& frame, const std::string& source) {
    return frame.Source(source).tensor.ToHost<float>();
}

TEST(SensorPipelinesTest, TracedRunsMatchStandaloneRunsOfTheirRates) {
    test::ForEachDevice([](device::Device& device) {
        const Ref<Scene> scene = Sky(device, DetectorPattern::Rgb);
        DefineRaw(*scene);
        scene->Find("/Cam")->Set(Camera::kPipelineBinding, std::vector<std::string>{"/Pipes/Raw"});
        scene->Define<RenderProduct>("/Render/View",
                                     {.renderMode = RenderMode::RealTime,
                                      .camera = {"/Cam"},
                                      .resolution = {kWidth, kHeight},
                                      .orderedVars = {"/Render/Rate", "/Render/Raw"}});
        Renderer renderer(scene);
        const SensorPipelineRunner standalone(device, RawDesc());
        const SensorPipelineRunner loaded(scene, "/Pipes/Raw");
        EXPECT_EQ(loaded.Sources(), std::vector<std::string>{"/Pipes/Raw/Signal"});
        EXPECT_EQ(loaded.Outputs(), std::vector<std::string>{"raw"});
        SensorPipelineRunDesc run{.exposureTime = kExposureTime,
                                  .identity = HashPath("/Cam").words[0]};
        std::vector<float> previous;
        for (uint32_t capture = 0; capture < 2; ++capture) {
            RenderOutput output = renderer.Step("/Render/View", 0);
            output.Wait();
            const FrameOutput& frame = output["/Render/View"].frames.at(0);
            const RenderVarOutput& raw = frame.Source("raw");
            EXPECT_EQ(std::vector<int64_t>(raw.tensor.Shape().begin(), raw.tensor.Shape().end()),
                      (std::vector<int64_t>{1, kHeight, kWidth, 3}));
            run.inputs = {{"/Pipes/Raw/Signal", frame.Source("ElectronRate").tensor}};
            run.capture = capture;
            const std::vector<float> traced = raw.tensor.ToHost<float>();
            EXPECT_EQ(traced, standalone.Run(run).at("raw").ToHost<float>()) << capture;
            SensorPipelineRunDesc named = run;
            named.inputs = {{"Signal", run.inputs.begin()->second}};
            EXPECT_EQ(traced, loaded.Run(named).at("raw").ToHost<float>()) << capture;
            EXPECT_NE(traced, previous);
            previous = traced;
        }
    });
}

TEST(SensorPipelinesTest, GlobalNodesRunTracedAsTheyDoStandalone) {
    test::ForEachDevice([](device::Device& device) {
        const Ref<Scene> scene = Sky(device, DetectorPattern::Rgb);
        DefineRaw(*scene);
        const Prim glare = Node(*scene, "/Pipes/Raw/Glare", "Glare");
        glare.Set(Glare::kSigma, 3.0f);
        glare.Set(Glare::kStrength, 0.3f);
        glare.Connect("electronRate", *scene->Find("/Pipes/Raw/Signal"), "electronRate");
        scene->Find("/Pipes/Raw/Integrate")->Connect("electronRate", glare, "electronRate");
        scene->Define<Camera>("/Cam2", {.translate = {0, 0, 1}, .focalLength = 10});
        for (const char* camera : {"/Cam", "/Cam2"}) {
            scene->Find(camera)->Set(Camera::kExposureTime, kExposureTime);
            scene->Find(camera)->Set(Camera::kPipelineBinding,
                                     std::vector<std::string>{"/Pipes/Raw"});
        }
        scene->Define<RenderProduct>("/Render/View",
                                     {.renderMode = RenderMode::RealTime,
                                      .camera = {"/Cam", "/Cam2"},
                                      .resolution = {kWidth, kHeight},
                                      .orderedVars = {"/Render/Rate", "/Render/Raw"}});
        Renderer renderer(scene);
        RenderOutput output = renderer.Step("/Render/View", 0);
        output.Wait();
        const FrameOutput& frame = output["/Render/View"].frames.at(0);
        const SensorPipelineRunner loaded(scene, "/Pipes/Raw");
        for (int64_t layer = 0; layer < 2; ++layer) {
            const char* camera = layer == 0 ? "/Cam" : "/Cam2";
            SensorPipelineRunDesc run{.exposureTime = kExposureTime,
                                      .identity = HashPath(camera).words[0]};
            const std::vector<float> rates =
                frame.Source("ElectronRate").tensor.Select(layer).ToHost<float>();
            run.inputs = {
                {"Signal", device::Tensor(device.CreateBuffer({.label = "rates"}, rates), 0,
                                          device::DType::Float32, {1, kHeight, kWidth, 3})}};
            EXPECT_EQ(frame.Source("raw").tensor.Select(layer).ToHost<float>(),
                      loaded.Run(run).at("raw").ToHost<float>())
                << camera;
        }
    });
}

TEST(SensorPipelinesTest, ValueEditsRerunThePipelineWithoutTracingAgain) {
    test::ForEachDevice([](device::Device& device) {
        const Ref<Scene> scene = Sky(device, DetectorPattern::Rgb);
        DefineRaw(*scene);
        scene->Find("/Cam")->Set(Camera::kPipelineBinding, std::vector<std::string>{"/Pipes/Raw"});
        scene->Define<RenderProduct>("/Render/View",
                                     {.renderMode = RenderMode::RealTime,
                                      .camera = {"/Cam"},
                                      .resolution = {kWidth, kHeight},
                                      .orderedVars = {"/Render/Rate", "/Render/Raw"}});
        Renderer renderer(scene);
        const auto step = [&] {
            RenderOutput output = renderer.Step("/Render/View", 0);
            output.Wait();
            const FrameOutput& frame = output["/Render/View"].frames.at(0);
            return std::pair{frame.Source("ElectronRate").tensor.ToHost<float>(),
                             frame.Source("raw").tensor.ToHost<float>()};
        };
        const auto traces = [&](const std::function<void()>& edit) {
            static const profiler::ZoneId zone = profiler::ZoneId::Named("Render/TraceCaptures");
            profiler::BeginFrame();
            edit();
            step();
            profiler::BeginFrame();
            const std::vector<profiler::Frame> frames = profiler::Frames();
            return std::ranges::count_if(
                frames.back().zones, [](const profiler::Zone& each) { return each.id == zone; });
        };
        profiler::SetEnabled(true);
        profiler::SetHistory(4);
        profiler::BeginFrame();
        step();
        const auto [rates, raw] = step();
        EXPECT_EQ(traces([] {}), 0);
        EXPECT_EQ(traces([&] { scene->Find("/Pipes/Raw/Gain")->Set(Gain::kGain, 4.0f); }), 0);
        const auto [kept, raised] = step();
        EXPECT_EQ(kept, rates);
        EXPECT_NE(raised, raw);
        EXPECT_GT(traces([&] { scene->Find("/Sky")->Set(DomeLight::kIntensity, 2.0f); }), 0);
        profiler::SetEnabled(false);
    });
}

TEST(SensorPipelinesTest, ProductsShareRunsSampleMosaicsAndOverrideTheirSensors) {
    test::ForEachDevice([](device::Device& device) {
        const Ref<Scene> scene = Sky(device, DetectorPattern::Rggb);
        DefineRaw(*scene);
        const Prim pipe = scene->Define<SensorPipeline>("/Pipes/Mosaic");
        const Prim mosaic = Node(*scene, "/Pipes/Mosaic/Mosaic", "CameraSignal");
        mosaic.Set(CameraSignal::kLayout, CameraSignalLayout::Mosaic);
        const Prim sampled = Node(*scene, "/Pipes/Mosaic/Sampled", "Integrate");
        sampled.Connect("electronRate", mosaic, "electronRate");
        const Prim full = Node(*scene, "/Pipes/Mosaic/Full", "CameraSignal");
        const Prim integrate = Node(*scene, "/Pipes/Mosaic/Integrate", "Integrate");
        integrate.Connect("electronRate", full, "electronRate");
        const Prim adc = Node(*scene, "/Pipes/Mosaic/Adc", "Adc");
        adc.Connect("electrons", integrate, "electrons");
        const Prim color = Node(*scene, "/Pipes/Mosaic/Color", "ToColor");
        color.Connect("digital", adc, "digital");
        const Prim meter = Node(*scene, "/Pipes/Mosaic/Meter", "Meter");
        meter.Connect("color", color, "color");
        pipe.Connect("mosaic", sampled, "electrons");
        pipe.Connect("color", color, "color");
        pipe.Connect("stat", meter, "stat");
        scene->Find("/Cam")->Set(Camera::kPipelineBinding,
                                 std::vector<std::string>{"/Pipes/Mosaic"});
        scene->Define<RenderVar>("/Render/Mosaic", {.sourceName = "mosaic"});
        scene->Define<RenderVar>("/Render/Color", {.sourceName = "color"});
        scene->Define<RenderVar>("/Render/Stat", {.sourceName = "stat"});
        scene->Define<RenderVar>("/Render/Electrons",
                                 {.sourceName = "Tap", .tap = {"/Pipes/Mosaic/Integrate"}});
        scene->Define<RenderVar>(
            "/Render/Source",
            {.sourceName = "Tap", .tap = {"/Pipes/Mosaic/Mosaic"}, .tapOutput = "electronRate"});
        const auto product = [&](const char* path, std::vector<std::string> vars,
                                 std::vector<std::string> binding) {
            scene->Define<RenderProduct>(path, {.renderMode = RenderMode::RealTime,
                                                .camera = {"/Cam"},
                                                .resolution = {kWidth, kHeight},
                                                .pipelineBinding = std::move(binding),
                                                .orderedVars = std::move(vars)});
        };
        product("/Render/A",
                {"/Render/Rate", "/Render/Mosaic", "/Render/Color", "/Render/Stat",
                 "/Render/Electrons", "/Render/Source"},
                {});
        product("/Render/B", {"/Render/Mosaic"}, {});
        product("/Render/C", {"/Render/Raw"}, {"/Pipes/Raw"});
        Renderer renderer(scene);
        RenderOutput output =
            renderer.Step(std::vector<std::string>{"/Render/A", "/Render/B", "/Render/C"}, 0);
        output.Wait();
        const FrameOutput& a = output["/Render/A"].frames.at(0);
        const std::vector<float> rates = Host(a, "ElectronRate");
        const std::vector<float> electrons = Host(a, "mosaic");
        ASSERT_EQ(electrons.size(), size_t(kWidth * kHeight));
        for (uint32_t y = 0; y < kHeight; ++y) {
            for (uint32_t x = 0; x < kWidth; ++x) {
                const uint32_t pixel = y * kWidth + x;
                const uint32_t site = std::vector<uint32_t>{0, 1, 1, 2}[(x & 1) | ((y & 1) << 1)];
                EXPECT_FLOAT_EQ(electrons[pixel], rates[pixel * 3 + site] * kExposureTime)
                    << x << ", " << y;
            }
        }
        EXPECT_GT(electrons[0], 0);
        EXPECT_EQ(a.RenderVar("/Render/Source").sourceName, "Tap");
        const std::vector<float> source = a.RenderVar("/Render/Source").tensor.ToHost<float>();
        const std::vector<float> integrated =
            a.RenderVar("/Render/Electrons").tensor.ToHost<float>();
        ASSERT_EQ(source.size(), electrons.size());
        ASSERT_EQ(integrated.size(), rates.size());
        for (size_t i = 0; i < electrons.size(); ++i) {
            EXPECT_FLOAT_EQ(source[i] * kExposureTime, electrons[i]) << i;
        }
        for (size_t i = 0; i < rates.size(); ++i) {
            EXPECT_FLOAT_EQ(integrated[i], rates[i] * kExposureTime) << i;
        }
        EXPECT_EQ(Host(output["/Render/B"].frames.at(0), "mosaic"), electrons);

        const std::vector<float> colors = Host(a, "color");
        const std::vector<float> stat = Host(a, "stat");
        ASSERT_EQ(stat.size(), size_t(sensor_pipeline::kStatWords));
        EXPECT_FLOAT_EQ(stat[sensor_pipeline::kStatWords - 1], float(kWidth * kHeight));
        for (uint32_t channel = 0; channel < 3; ++channel) {
            double sum = 0;
            for (uint32_t pixel = 0; pixel < kWidth * kHeight; ++pixel) {
                sum += colors[pixel * 3 + channel];
            }
            EXPECT_NEAR(stat[channel], sum / (kWidth * kHeight), 1e-5) << channel;
        }

        const FrameOutput& c = output["/Render/C"].frames.at(0);
        EXPECT_EQ(c.Source("raw").tensor.Shape()[3], 3);
        EXPECT_EQ(c.renderVars.size(), 1u);
    });
}

}
}
