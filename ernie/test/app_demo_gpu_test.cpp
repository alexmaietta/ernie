// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/demo.h"
#include "ernie/device/vulkan_context.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/xform.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

using namespace ernie;
using namespace ernie::app;

/// A cube named `name` under a sun, seen by a camera; counts its
/// steps and checks.
class CubeDemo : public Demo {
  public:
    CubeDemo(Viewer* viewer, const DemoOptions& options,
             std::string name = "Cube")
        : Demo(viewer, options), name_(std::move(name)) {
        numFrames = 3;
        Scene& scene = GetScene();
        Cube::Define(scene, "/World/" + name_).SetSize(2.0);
        DistantLight::Define(scene, "/World/Sun")
            .SetTransform(DistantLightTransform({0.4, -0.6, 0.7}));
        Camera::Define(scene, "/World/Camera")
            .SetTransform(LookAtTransform(
                {0.0, -8.0, 3.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}));
    }

    void Step() override {
        ++steps;
        if (onStep) {
            onStep(*this);
        }
    }

    void TestFinal() override {
        ++finalChecks;
    }

    int steps = 0;
    int finalChecks = 0;
    std::function<void(CubeDemo&)> onStep;

  private:
    std::string name_;
};

/// A demo without checks.
class UncheckedDemo : public Demo {
  public:
    using Demo::Demo;
};

class DemoTest : public ::testing::Test {
  protected:
    void SetUp() override {
        try {
            VulkanContext::Create();
        } catch (const std::runtime_error& error) {
            GTEST_SKIP() << "No Vulkan device: " << error.what();
        }
    }

    static DemoOptions
    Options(std::initializer_list<std::string> args) {
        DemoOptions options =
            ParseDemoOptions(std::vector<std::string>(args));
        options.validation = true;
        return options;
    }

    static std::filesystem::path TempFile(const std::string& name) {
        return std::filesystem::temp_directory_path() /
               ("ernie_demo_" +
                std::to_string(::testing::UnitTest::GetInstance()
                                   ->random_seed()) +
                "_" + name);
    }
};

}  // namespace

TEST_F(DemoTest, RunsWithoutAViewerForItsFramesAndChecks) {
    DemoRunner runner(Options({"--viewer", "null", "--test"}));
    EXPECT_EQ(runner.GetViewer(), nullptr);
    auto demo = std::make_shared<CubeDemo>(nullptr, runner.Options());
    const std::shared_ptr<Demo> ran = runner.Run(demo);
    EXPECT_EQ(ran, demo);
    EXPECT_EQ(demo->steps, 3);
    EXPECT_EQ(demo->frame, 3u);
    EXPECT_NEAR(demo->time, 3.0 / 60.0, 1e-9);
    EXPECT_EQ(demo->finalChecks, 1);
    ASSERT_FALSE(demo->Image().Empty());
    EXPECT_EQ(demo->Image().Shape().back(), 4);
}

TEST_F(DemoTest, NumFramesOverridesTheDemosOwnCount) {
    DemoRunner runner(
        Options({"--viewer", "null", "--num-frames", "5"}));
    auto demo = std::make_shared<CubeDemo>(nullptr, runner.Options());
    runner.Run(demo);
    EXPECT_EQ(demo->steps, 5);
}

TEST_F(DemoTest, ATestRunOfADemoWithoutChecksFails) {
    DemoRunner runner(
        Options({"--viewer", "null", "--test", "--num-frames", "1"}));
    EXPECT_THROW(runner.Run(std::make_shared<UncheckedDemo>(
                     nullptr, runner.Options())),
                 std::logic_error);
}

TEST_F(DemoTest, WritesTheOutputWithoutAViewer) {
    const std::filesystem::path path = TempFile("output.png");
    DemoRunner runner(Options({"--viewer", "null", "--num-frames",
                               "1", "--output", path.string()}));
    runner.Run(std::make_shared<CubeDemo>(nullptr, runner.Options()));
    EXPECT_GT(std::filesystem::file_size(path), 100u);
    std::filesystem::remove(path);
}

TEST_F(DemoTest, BuildsIntoTheViewersSceneHeadless) {
    DemoRunner runner(Options({"--headless", "--test"}));
    ASSERT_NE(runner.GetViewer(), nullptr);
    runner.SetCatalog({MakeDemoInfo<CubeDemo>("test", "cube")});
    const std::shared_ptr<Demo> demo = runner.Create(0);
    EXPECT_EQ(&demo->GetScene(), &runner.GetViewer()->GetScene());
    runner.Run(demo);
    EXPECT_EQ(demo->frame, 3u);
    EXPECT_TRUE(demo->GetScene().Has("/World/Cube"));
}

TEST_F(DemoTest,
       SwitchingRemovesWhatTheLastDemoAddedAndBuildsTheNext) {
    DemoRunner runner(Options({"--headless", "--num-frames", "2"}));
    Viewer& viewer = *runner.GetViewer();
    const ViewportDisplay display = viewer.GetViewportDisplay();
    const auto make = [&runner](std::string name, bool switches) {
        return DemoInfo{
            "test", name, "",
            [&runner, name, switches](Viewer* v,
                                      const DemoOptions& o) {
                auto demo = std::make_shared<CubeDemo>(v, o, name);
                v->AddPanel(name + " panel", Dock::Right,
                            [](Viewer&) {});
                v->OnFrame([](Viewer&, const FrameInfo&) {});
                v->AddViewportToggle(name + " toggle");
                v->AddMenuItem("Demo", name, [](Viewer&) {});
                v->SetLighting(Lighting::Camera);
                ViewportDisplay hidden = v->GetViewportDisplay();
                hidden.grid = false;
                v->SetViewportDisplay(hidden);
                if (switches) {
                    demo->onStep = [&runner](CubeDemo& self) {
                        if (self.frame == 1) {
                            runner.RequestSwitch(1);
                        }
                    };
                }
                return demo;
            }};
    };
    runner.SetCatalog({make("A", true), make("B", false)});
    const std::shared_ptr<Demo> first = runner.Create(0);
    EXPECT_EQ(viewer.GetLighting(), Lighting::Camera);
    const std::shared_ptr<Demo> last = runner.Run(first);
    ASSERT_NE(last, nullptr);
    EXPECT_NE(last, first);
    EXPECT_EQ(last->frame, 2u);  // B's own frames, from 0
    Scene& scene = viewer.GetScene();
    EXPECT_TRUE(scene.Has("/World/B"));
    EXPECT_FALSE(scene.Has("/World/A"));
    EXPECT_TRUE(scene.Has(Viewer::PerspectiveCamera()));
    EXPECT_EQ(viewer.FindPanel("A panel"), nullptr);
    // B's registrations go when the run ends, as they hold B.
    EXPECT_EQ(viewer.FindPanel("B panel"), nullptr);
    // B's lighting was its own choice, kept over the session's reset;
    // the display was restored before B hid the grid again.
    EXPECT_EQ(viewer.GetLighting(), Lighting::Camera);
    EXPECT_FALSE(viewer.GetViewportDisplay().grid);
    EXPECT_TRUE(display.grid);
}

TEST_F(DemoTest, ViewerScopesRemoveWhatWasAddedInThem) {
    Scene scene;
    Cube::Define(scene, "/World/Cube");
    ViewerOptions options;
    options.hidden = true;
    options.validation = true;
    Viewer viewer(scene, options);
    int calls = 0;
    viewer.OnFrame([&calls](Viewer&, const FrameInfo&) { ++calls; });
    const std::uint64_t scope = viewer.NewScope();
    EXPECT_EQ(viewer.Scope(), 0u);
    viewer.SetScope(scope);
    viewer.OnFrame(
        [&calls](Viewer&, const FrameInfo&) { calls += 10; });
    viewer.AddPanel("Scoped", Dock::Left, [](Viewer&) {});
    const std::uint64_t hud =
        viewer.AddHudItem("Scoped", [](Viewer&) { return "x"; });
    viewer.SetScope(0);
    viewer.Step();
    EXPECT_EQ(calls, 11);
    viewer.RemoveScope(scope);
    viewer.Step();
    EXPECT_EQ(calls, 12);
    EXPECT_EQ(viewer.FindPanel("Scoped"), nullptr);
    EXPECT_THROW(viewer.HudItemShown(hud), std::invalid_argument);
    EXPECT_THROW(viewer.RemoveScope(0), std::invalid_argument);
}

TEST_F(DemoTest,
       ResetSessionKeepsTheCameraChosenSinceTheSceneWasCleared) {
    Scene scene;
    ViewerOptions options;
    options.hidden = true;
    options.validation = true;
    Viewer viewer(scene, options);
    viewer.ClearScene();
    Cube::Define(scene, "/World/Cube");
    Camera::Define(scene, "/World/Camera");
    viewer.SetActiveCamera("/World/Camera");
    viewer.ResetSession();
    EXPECT_EQ(viewer.ActiveCamera(), Path("/World/Camera"));
    EXPECT_EQ(viewer.GetLighting(),
              Lighting::Default);  // no lights, none chosen
    viewer.ClearScene();
    EXPECT_FALSE(scene.Has("/World/Cube"));
    EXPECT_TRUE(scene.Has(Viewer::PerspectiveCamera()));
    EXPECT_EQ(viewer.ActiveCamera(), Viewer::PerspectiveCamera());
    Camera::Define(scene, "/World/Camera");
    viewer.ResetSession();
    EXPECT_EQ(viewer.ActiveCamera(), Viewer::PerspectiveCamera());
}
