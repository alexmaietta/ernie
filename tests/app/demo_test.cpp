// SPDX-License-Identifier: BSD-3-Clause
/// @file demo_test.cpp
/// @brief Tests of the demo registry, and of running each registered demo for a while.
#include "ernie/app/demo.h"
#include "ernie/app/ui.h"

#include <gtest/gtest.h>

namespace ernie {
namespace {

TEST(Demos, AreFoundByIdLabelOrPath) {
    const std::vector<app::DemoInfo> demos = app::RegisteredDemos();
    ASSERT_FALSE(demos.empty());
    const app::DemoInfo* bounce = app::FindDemo(demos, "demo_basics_bounce");
    ASSERT_NE(bounce, nullptr);
    EXPECT_EQ(app::FindDemo(demos, "Basics: Bounce"), bounce);
    EXPECT_EQ(app::FindDemo(demos, "basics/bounce"), bounce);
    EXPECT_EQ(app::FindDemo(demos, "nothing"), nullptr);
}

TEST(Demos, EachBuildsAndStepsItsScene) {
    for (const app::DemoInfo& info : app::RegisteredDemos()) {
        Scene scene;
        const std::unique_ptr<app::Demo> demo = info.create(scene);
        EXPECT_FALSE(scene.Children(Path::Root()).empty()) << info.Label();
        for (int step = 0; step < 240; ++step) {
            demo->Advance();
            demo->TestPostStep();
        }
        EXPECT_NO_THROW(demo->TestFinal()) << info.Label();
        EXPECT_EQ(demo->Frame(), 240u);
        EXPECT_NEAR(demo->Time(), 4.0, 1e-9);
    }
}

TEST(Demos, TheShellRunsResetsAndReplacesTheScene) {
    Scene scene;
    scene.Define("/Before", "Xform");
    app::Shell shell(scene, nullptr);
    const std::vector<app::DemoInfo> demos = app::RegisteredDemos();
    const app::DemoInfo* bounce = app::FindDemo(demos, "demo_basics_bounce");
    ASSERT_NE(bounce, nullptr);
    shell.LoadDemo(*bounce);
    EXPECT_FALSE(scene.Has("/Before")) << "the demo's scene replaces it";
    EXPECT_TRUE(scene.Has("/World/Ball"));
    shell.StepDemo();
    shell.StepDemo();
    EXPECT_EQ(shell.CurrentDemo()->Frame(), 2u);
    shell.ResetDemo();
    EXPECT_EQ(shell.CurrentDemo()->Frame(), 0u);
}

}  // namespace
}  // namespace ernie
