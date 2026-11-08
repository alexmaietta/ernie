// SPDX-License-Identifier: BSD-3-Clause
#include <bit>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/scene/scene.h"
#include "ernie/scene/scene_impl.h"
#include "ernie/util/error.h"
#include "support/devices.h"

namespace ernie {
namespace {

struct Rig {
    Ref<Scene> scene;
    Prim camera;
    Prim raw;
    Prim emva;
    Prim gain;
    Prim color;
};

Prim Node(Scene& scene, const std::string& path, const char* id) {
    return scene.Define<Shader>(path, {.id = id});
}

Rig MakeRig(device::Device& device) {
    Rig rig;
    rig.scene = Scene::Create(device);
    Scene& scene = *rig.scene;
    rig.camera = scene.Define<Camera>("/Cam");
    scene.Define<Scope>("/Pipes");
    rig.raw = scene.Define<SensorPipeline>("/Pipes/Raw");
    const Prim signal = Node(scene, "/Pipes/Raw/Signal", "CameraSignal");
    const Prim integrate = Node(scene, "/Pipes/Raw/Integrate", "Integrate");
    integrate.Connect("electronRate", signal, "electronRate");
    rig.emva = Node(scene, "/Pipes/Raw/Noise", "Emva1288");
    rig.emva.Connect("electrons", integrate, "electrons");
    const Prim adc = Node(scene, "/Pipes/Raw/Adc", "Adc");
    adc.Connect("electrons", rig.emva, "electrons");
    rig.gain = Node(scene, "/Pipes/Raw/Gain", "Gain");
    rig.gain.Connect("digital", adc, "digital");
    rig.raw.Connect("raw", rig.gain, "digital");

    rig.color = scene.Define<SensorPipeline>("/Pipes/Color");
    const Prim colorSignal = Node(scene, "/Pipes/Color/Signal", "CameraSignal");
    const Prim colorIntegrate = Node(scene, "/Pipes/Color/Integrate", "Integrate");
    colorIntegrate.Connect("electronRate", colorSignal, "electronRate");
    const Prim colorAdc = Node(scene, "/Pipes/Color/Adc", "Adc");
    colorAdc.Connect("electrons", colorIntegrate, "electrons");
    const Prim toColor = Node(scene, "/Pipes/Color/ToColor", "ToColor");
    toColor.Connect("digital", colorAdc, "digital");
    rig.color.Connect("color", toColor, "color");

    rig.camera.Set(Camera::kPipelineBinding, std::vector<std::string>{"/Pipes/Raw"});
    scene.Define<Scope>("/Render");
    scene.Define<RenderVar>("/Render/Raw", {.sourceName = "raw"});
    scene.Define<RenderVar>("/Render/Color", {.sourceName = "color"});
    scene.Define<RenderProduct>("/Render/A", {.renderMode = RenderMode::RealTime,
                                              .camera = {"/Cam"},
                                              .resolution = {8, 4},
                                              .orderedVars = {"/Render/Raw"}});
    scene.Define<RenderProduct>("/Render/B", {.renderMode = RenderMode::RealTime,
                                              .camera = {"/Cam"},
                                              .resolution = {8, 4},
                                              .orderedVars = {"/Render/Raw"}});
    scene.Define<RenderProduct>("/Render/C", {.renderMode = RenderMode::RealTime,
                                              .camera = {"/Cam"},
                                              .resolution = {8, 4},
                                              .pipelineBinding = {"/Pipes/Color"},
                                              .orderedVars = {"/Render/Color"}});
    scene.Publish().Wait();
    return rig;
}

std::shared_ptr<const PipelineInstance> Instance(const Rig& rig, const Prim& pipeline) {
    return SceneImpl::Get(*rig.scene).GetPipelines().Find(pipeline.Index());
}

TEST(PipelinesTest, ValueEditsRepackAndUniformEditsMakeNewDefinitions) {
    test::ForEachDevice([](device::Device& device) {
        Rig rig = MakeRig(device);
        const auto first = Instance(rig, rig.raw);
        ASSERT_NE(first, nullptr);
        ASSERT_EQ(first->error, "");
        EXPECT_GT(first->definition, 0u);
        EXPECT_EQ(first->words, (std::vector<uint32_t>{0, 0, std::bit_cast<uint32_t>(1.0f)}));

        rig.gain.Set(Gain::kGain, 2.0f);
        rig.scene->Publish().Wait();
        const auto repacked = Instance(rig, rig.raw);
        EXPECT_EQ(repacked->definition, first->definition);
        EXPECT_EQ(repacked->words, (std::vector<uint32_t>{0, 0, std::bit_cast<uint32_t>(2.0f)}));

        rig.emva.Set(Emva1288::kShot, ShotNoise::Gaussian);
        rig.scene->Publish().Wait();
        EXPECT_NE(Instance(rig, rig.raw)->definition, first->definition);
        rig.emva.Set(Emva1288::kShot, ShotNoise::Poisson);
        rig.scene->Publish().Wait();
        EXPECT_EQ(Instance(rig, rig.raw)->definition, first->definition);
    });
}

TEST(PipelinesTest, ProductsReadOutputsOfTheirOwnOrTheirSensorsPipeline) {
    test::ForEachDevice([](device::Device& device) {
        Rig rig = MakeRig(device);
        SceneImpl& impl = SceneImpl::Get(*rig.scene);
        const auto a = impl.GetProducts().Find(rig.scene->Find("/Render/A")->Index());
        ASSERT_EQ(a->error, "");
        EXPECT_EQ(a->pipeline, rig.raw.Index());
        ASSERT_EQ(a->vars.size(), 1u);
        EXPECT_EQ(a->vars[0].output, "raw");
        EXPECT_EQ(a->vars[0].type, device::DType::Float32);
        EXPECT_EQ(a->vars[0].shape, (std::vector<int64_t>{1, 4, 8, 3}));

        const auto c = impl.GetProducts().Find(rig.scene->Find("/Render/C")->Index());
        ASSERT_EQ(c->error, "");
        EXPECT_EQ(c->pipeline, rig.color.Index());
        EXPECT_EQ(c->vars[0].shape, (std::vector<int64_t>{1, 4, 8, 3}));

        const std::vector<std::string> paths = {"/Render/A", "/Render/B", "/Render/C"};
        const auto plan = impl.GetPlans().Plan(paths);
        ASSERT_EQ(plan->pipelines.size(), 2u);
        for (const PlannedPipeline& run : plan->pipelines) {
            EXPECT_EQ(run.sensor, rig.camera.Index());
            const std::vector<uint32_t> expected = run.pipeline == rig.raw.Index()
                                                       ? std::vector<uint32_t>{0, 1}
                                                       : std::vector<uint32_t>{2};
            EXPECT_EQ(run.products, expected);
        }

        rig.scene->Find("/Render/B")->Set(RenderProduct::kResolution, Uint2{4, 2});
        rig.scene->Publish().Wait();
        EXPECT_EQ(impl.GetPlans().Plan(paths)->pipelines.size(), 3u);

        rig.scene->Find("/Render/C")
            ->Set(RenderProduct::kOrderedVars, std::vector<std::string>{"/Render/Raw"});
        rig.scene->Publish().Wait();
        EXPECT_NE(impl.GetProducts().Find(rig.scene->Find("/Render/C")->Index())->error, "");
    });
}

TEST(PipelinesTest, MinimalProductsCantMeasureElectronRates) {
    test::ForEachDevice([](device::Device& device) {
        Rig rig = MakeRig(device);
        const Prim a = *rig.scene->Find("/Render/A");
        a.Set(RenderProduct::kRenderMode, RenderMode::Minimal);
        rig.scene->Publish().Wait();
        const auto product = SceneImpl::Get(*rig.scene).GetProducts().Find(a.Index());
        EXPECT_NE(product->error.find("RealTime"), std::string::npos) << product->error;
    });
}

TEST(PipelinesTest, ResolveErrorsReachTheProducts) {
    test::ForEachDevice([](device::Device& device) {
        Rig rig = MakeRig(device);
        const Prim stray = Node(*rig.scene, "/Pipes/Raw/Stray", "Adc");
        stray.Connect("electrons", *rig.scene->Find("/Pipes/Raw/Signal"), "electronRate");
        rig.raw.Connect("stray", stray, "digital");
        rig.scene->Publish().Wait();
        const auto instance = Instance(rig, rig.raw);
        EXPECT_NE(instance->error.find("takes Electrons"), std::string::npos) << instance->error;
        const auto a =
            SceneImpl::Get(*rig.scene).GetProducts().Find(rig.scene->Find("/Render/A")->Index());
        EXPECT_EQ(a->error, instance->error);
    });
}

}
}
