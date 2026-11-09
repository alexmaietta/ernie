// SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/pipeline/builtins.h"
#include "ernie/render/renderer.h"
#include "ernie/render/shaders.h"
#include "ernie/render/standalone.h"
#include "ernie/scene/scene.h"
#include "ernie/util/error.h"
#include "support/devices.h"
#include "support/support.h"

namespace ernie {
namespace {

constexpr uint32_t kWidth = 6;
constexpr uint32_t kHeight = 4;
const std::vector<std::string> kProducts = {"/Render/View"};

const std::filesystem::path& Scratch() {
    static const test::TemporaryDirectory directory;
    return directory.Path();
}

std::filesystem::path WriteFile(std::string_view name, std::string_view text) {
    const std::filesystem::path path = Scratch() / name;
    std::ofstream(path) << text;
    return path;
}

std::string OffsetNode(std::string_view name, std::string_view sign) {
    return std::format(R"(
import ernie.pipeline;

using namespace ernie;

[Node("Test Offset")]
struct {0} : IPointNode {{
    [Input(0, 100)] float amount = 5;
    int times = 2;
    bool negate = false;
    float3 weights = float3(1, 1, 1);

    Digital Apply(Digital digital, PointContext context) {{
        Channels channels = digital.Get();
        const float step = (negate ? -1 : 1) * {1}amount * float(times);
        [ForceUnroll]
        for (uint i = 0; i < 3; ++i) {{
            channels[i] += step * weights[i];
        }}
        return Digital.Make(channels);
    }}
}}

struct {0}Blur : IStencilNode {{
    static const int kRadius = 1;

    Digital Apply(PointContext context, Neighborhood<Digital> n) {{
        Digital sum = Digital.Zero(n.Center());
        for (int y = -kRadius; y <= kRadius; ++y) {{
            for (int x = -kRadius; x <= kRadius; ++x) {{
                sum = sum.Add(n.At(x, y));
            }}
        }}
        return sum.Scaled(1.0 / 9);
    }}
}}
)",
                       name, sign);
}

class Bench {
  public:
    Bench(device::Device& device, const std::string& node, const std::filesystem::path& file)
        : m_scene(Scene::Create(device, {.waitForKernels = true})) {
        Scene& scene = *m_scene;
        LoadShaders(scene, file);
        scene.Define<DomeLight>("/Sky", {});
        const Prim camera = scene.Define<Camera>("/Cam", {.focalLength = 10});
        camera.Set(Camera::kExposureTime, 1e-7f);
        camera.Set(DetectorAPI::kDetectorPattern, DetectorPattern::Rgb);
        camera.Set(Camera::kPipelineBinding, std::vector<std::string>{"/Raw"});
        const Prim pipe = scene.Define<SensorPipeline>("/Raw");
        const Prim signal = scene.Define<Shader>("/Raw/Signal", {.id = "CameraSignal"});
        const Prim integrate = scene.Define<Shader>("/Raw/Integrate", {.id = "Integrate"});
        integrate.Connect("electronRate", signal, "electronRate");
        const Prim adc = scene.Define<Shader>("/Raw/Adc", {.id = "Adc"});
        adc.Connect("electrons", integrate, "electrons");
        m_offset = scene.Define<Shader>("/Raw/Offset", {.id = node});
        m_offset.Connect("digital", adc, "digital");
        const Prim blur = scene.Define<Shader>("/Raw/Blur", {.id = node + "Blur"});
        blur.Connect("n", adc, "digital");
        pipe.Connect("plain", adc, "digital");
        pipe.Connect("offset", m_offset, "digital");
        pipe.Connect("blurred", blur, "digital");
        scene.Define<Scope>("/Render");
        std::vector<std::string> vars;
        for (const char* name : {"plain", "offset", "blurred"}) {
            vars.push_back(std::string("/Render/") + name);
            scene.Define<RenderVar>(vars.back(), {.sourceName = name});
        }
        scene.Define<RenderProduct>("/Render/View", {.renderMode = RenderMode::RealTime,
                                                     .camera = {"/Cam"},
                                                     .resolution = {kWidth, kHeight},
                                                     .orderedVars = vars});
    }

    Scene& GetScene() {
        return *m_scene;
    }

    Prim& Offset() {
        return m_offset;
    }

    std::vector<std::vector<float>> Render() {
        if (!m_renderer) {
            m_renderer.emplace(m_scene);
        }
        RenderOutput output = m_renderer->Step(kProducts, 0);
        output.Wait();
        const auto& frame = output["/Render/View"].frames.at(0);
        std::vector<std::vector<float>> sources;
        for (const char* name : {"plain", "offset", "blurred"}) {
            sources.push_back(frame.Source(name).tensor.ToHost<float>());
        }
        return sources;
    }

  private:
    Ref<Scene> m_scene;
    Prim m_offset;
    std::optional<Renderer> m_renderer;
};

void ExpectOffset(const std::vector<std::vector<float>>& sources, float step) {
    ASSERT_EQ(sources[0].size(), sources[1].size());
    for (size_t i = 0; i < sources[0].size(); ++i) {
        ASSERT_FLOAT_EQ(sources[1][i], sources[0][i] + step) << i;
    }
}

TEST(PipelineNodeTest, LoadedNodesDescribeTheirKindPortsAndInputs) {
    const std::filesystem::path file =
        WriteFile("ernie_pipeline_nodes_describe.slang", OffsetNode("TestDescribe", ""));
    test::ForEachDevice([&](device::Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        EXPECT_EQ(LoadShaders(*scene, file),
                  (std::vector<std::string>{"TestDescribe", "TestDescribeBlur"}));
        const std::shared_ptr<const pipeline::NodeRegistry> nodes = pipeline::PipelineNodes();
        const pipeline::NodeDef& offset = (*nodes)[nodes->Find("TestDescribe")];
        EXPECT_EQ(offset.kind, pipeline::NodeKind::Point);
        ASSERT_EQ(offset.inputs.size(), 1u);
        EXPECT_EQ(offset.inputs[0].name, "digital");
        EXPECT_EQ(offset.inputs[0].type, pipeline::SignalType::Digital);
        ASSERT_EQ(offset.outputs.size(), 1u);
        EXPECT_EQ(offset.outputs[0].name, "digital");
        EXPECT_EQ(offset.context, 1u);
        ASSERT_EQ(offset.parameters.size(), 4u);
        EXPECT_EQ(offset.parameters[0].name, "amount");
        EXPECT_EQ(offset.parameters[0].fallback, std::vector<float>{5});
        EXPECT_EQ(offset.parameters[1].type, pipeline::ParameterType::Int);
        EXPECT_EQ(offset.parameters[1].fallback, std::vector<float>{2});
        EXPECT_EQ(offset.parameters[2].type, pipeline::ParameterType::Bool);
        EXPECT_EQ(offset.parameters[3].words, 3u);
        EXPECT_EQ(offset.parameters[3].fallback, (std::vector<float>{1, 1, 1}));

        const pipeline::NodeDef& blur = (*nodes)[nodes->Find("TestDescribeBlur")];
        EXPECT_EQ(blur.kind, pipeline::NodeKind::Stencil);
        EXPECT_EQ(blur.radius, 1u);
        ASSERT_EQ(blur.inputs.size(), 1u);
        EXPECT_EQ(blur.inputs[0].name, "n");
        EXPECT_EQ(blur.context, 0u);

        EXPECT_EQ(FindPipelineNode("TestDescribeBlur")->kind, "stencil");
        const Prim prim = scene->Define<Shader>("/Offset", {.id = "TestDescribe"});
        EXPECT_EQ(prim.Get<float>("amount"), 5.0f);
        EXPECT_EQ(prim.Get<int32_t>("times"), 2);
        EXPECT_FALSE(prim.Get<bool>("negate"));
    });
}

TEST(PipelineNodeTest, CustomNodesRunInTheTracedPipeline) {
    const std::filesystem::path file =
        WriteFile("ernie_pipeline_nodes_run.slang", OffsetNode("TestRun", ""));
    test::ForEachDevice([&](device::Device& device) {
        Bench bench(device, "TestRun", file);
        std::vector<std::vector<float>> sources = bench.Render();
        ASSERT_GT(*std::ranges::min_element(sources[0]), 20);
        ASSERT_LT(*std::ranges::max_element(sources[0]), 4000);
        ExpectOffset(sources, 10);

        bench.Offset().Set("negate", true);
        bench.Offset().Set("times", 3);
        sources = bench.Render();
        ExpectOffset(sources, -15);

        const std::vector<float>& plain = sources[0];
        const std::vector<float>& blurred = sources[2];
        for (int y = 0; y < int(kHeight); ++y) {
            for (int x = 0; x < int(kWidth); ++x) {
                for (int c = 0; c < 3; ++c) {
                    float sum = 0;
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            const int sx = std::clamp(x + dx, 0, int(kWidth) - 1);
                            const int sy = std::clamp(y + dy, 0, int(kHeight) - 1);
                            sum += plain[(sy * kWidth + sx) * 3 + c];
                        }
                    }
                    EXPECT_NEAR(blurred[(y * kWidth + x) * 3 + c], sum / 9, 1e-3f * sum)
                        << x << "," << y << "," << c;
                }
            }
        }
    });
}

TEST(PipelineNodeTest, ReloadedFilesChangeTheOutputAndBrokenOnesKeepTheLast) {
    const std::filesystem::path file =
        WriteFile("ernie_pipeline_nodes_hot.slang", OffsetNode("TestHot", ""));
    test::ForEachDevice([&](device::Device& device) {
        WriteFile("ernie_pipeline_nodes_hot.slang", OffsetNode("TestHot", ""));
        Bench bench(device, "TestHot", file);
        ExpectOffset(bench.Render(), 10);
        EXPECT_TRUE(ReloadShaders(bench.GetScene()).empty());

        WriteFile("ernie_pipeline_nodes_hot.slang", OffsetNode("TestHot", "3 * "));
        EXPECT_EQ(ReloadShaders(bench.GetScene()),
                  (std::vector<std::string>{"TestHot", "TestHotBlur"}));
        ExpectOffset(bench.Render(), 30);

        WriteFile("ernie_pipeline_nodes_hot.slang",
                  "import ernie.pipeline;\nstruct TestHot : IPointNode { Digital Apply( }");
        EXPECT_TRUE(ReloadShaders(bench.GetScene()).empty());
        ExpectOffset(bench.Render(), 30);
    });
}

TEST(PipelineNodeTest, ErrorsNameTheUsersFile) {
    const auto expectError = [](Scene& scene, const std::filesystem::path& file,
                                std::string_view expected) {
        try {
            LoadShaders(scene, file);
            ADD_FAILURE() << "loading " << file << " succeeded";
        } catch (const Error& error) {
            const std::string message = error.what();
            EXPECT_NE(message.find(std::filesystem::weakly_canonical(file).string()),
                      std::string::npos)
                << message;
            EXPECT_NE(message.find(expected), std::string::npos) << message;
            EXPECT_EQ(message.find("ernie/pipeline/nodes"), std::string::npos) << message;
        }
    };
    const std::filesystem::path lines = WriteFile("ernie_pipeline_nodes_lines.slang", R"(
import ernie.pipeline;
using namespace ernie;

struct TestLines : IPointNode {
    Digital Apply(Digital digital) {
        return NoSuchFunction(digital);
    }
}
)");
    const std::filesystem::path missing = WriteFile("ernie_pipeline_nodes_missing.slang", R"(
import ernie.pipeline;
using namespace ernie;

struct TestMissing : IPointNode {
    float amount = 1;
}
)");
    const std::filesystem::path radius = WriteFile("ernie_pipeline_nodes_radius.slang", R"(
import ernie.pipeline;
using namespace ernie;

struct TestRadius : IStencilNode {
    Digital Apply(Neighborhood<Digital> n) {
        return n.Center();
    }
}
)");
    const std::filesystem::path signal = WriteFile("ernie_pipeline_nodes_signal.slang", R"(
import ernie.pipeline;
using namespace ernie;

struct TestSignal : IPointNode {
    float Apply(Digital digital) {
        return digital.Get()[0];
    }
}
)");
    const std::filesystem::path builtin = WriteFile("ernie_pipeline_nodes_builtin.slang", R"(
import ernie.pipeline;
using namespace ernie;

struct Gain : IPointNode {
    Digital Apply(Digital digital) {
        return digital;
    }
}
)");
    test::ForEachDevice([&](device::Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        expectError(*scene, lines, std::filesystem::weakly_canonical(lines).string() + ":7");
        expectError(*scene, missing, "needs an Apply");
        expectError(*scene, radius, "kRadius");
        expectError(*scene, signal, "isn't a pipeline signal");
        expectError(*scene, builtin, "built-in");
        EXPECT_EQ(pipeline::PipelineNodes()->Find("TestSignal"), pipeline::kNoDef);
    });
}

}
}
