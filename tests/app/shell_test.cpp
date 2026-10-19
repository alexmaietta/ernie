// SPDX-License-Identifier: BSD-3-Clause
/// @file shell_test.cpp
/// @brief Tests of the viewer's UI drawn without a window: the layout, menus, shortcuts, the
/// Stage panel's search and the camera's controls.
#include "ernie/app/imgui_backend.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/schema/geom.h"
#include "ernie/scene/schema/lux.h"
#include "ernie/scene/schema/render.h"
#include "ernie/scene/xform.h"

#include <gtest/gtest.h>
#include <imgui.h>
#include <imgui_internal.h>

#include <cmath>
#include <filesystem>

namespace ernie {
namespace {

using app::Shell;

/// An ImGui context that draws the shell's frames at 1280 x 720 without a window or a device, and
/// takes simulated input.
class UiFrames {
  public:
    UiFrames() : session_(app::Theme()) {
        ImGuiIO& io = ImGui::GetIO();
        io.DisplaySize = {1280.0f, 720.0f};
        // Textures ImGui asks for are left unmade, as no renderer draws these frames.
        io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
    }

    void Draw(Shell& shell) const {
        ImGui::GetIO().DeltaTime = 1.0f / 60.0f;
        ImGui::NewFrame();
        shell.Draw();
        ImGui::Render();
    }

    /// @return Whether the window titled `title` was drawn last frame.
    static bool Drawn(const char* title) {
        const ImGuiWindow* window = ImGui::FindWindowByName(title);
        return window != nullptr && window->Active;
    }

  private:
    app::ImGuiSession session_;
};

/// A sphere of radius 1 at (10, 0, 0) under /World, and a lamp beside it.
void Build(Scene& scene) {
    Xform::Define(scene, "/World");
    SetXform(scene, "/World", {.translate = glm::dvec3(10.0, 0.0, 0.0)});
    Sphere::Define(scene, "/World/Rover/Wheel");
    Xform::Define(scene, "/World/Lamp");
    scene.Update();
}

TEST(Shell, DocksItsPanelsAroundTheViewport) {
    const UiFrames frames;
    Scene scene;
    Build(scene);
    Shell shell(scene, nullptr);
    frames.Draw(shell);
    frames.Draw(shell);
    for (const char* title : {"Stage", "Properties", "Log", "Info", "Devices", "Profiler"}) {
        EXPECT_TRUE(UiFrames::Drawn(title)) << title;
        EXPECT_NE(ImGui::FindWindowByName(title)->DockNode, nullptr) << title << " is docked";
    }
    const auto [origin, size] = shell.Viewport();
    EXPECT_GT(origin.y, 0.0f) << "below the menu bar";
    EXPECT_GT(size.x, 0.5f * 1280.0f) << "the viewport keeps most of the width";
    EXPECT_LT(size.x, 1280.0f);
    EXPECT_LT(size.y, 720.0f);
}

TEST(Shell, TabsTheLogDevicesAndProfilerAtTheBottomOpeningOnTheLog) {
    const UiFrames frames;
    Scene scene;
    Build(scene);
    Shell shell(scene, nullptr);
    frames.Draw(shell);
    frames.Draw(shell);
    const ImGuiWindow* log = ImGui::FindWindowByName("Log");
    for (const char* title : {"Info", "Devices", "Profiler"}) {
        EXPECT_EQ(ImGui::FindWindowByName(title)->DockNode, log->DockNode) << title;
    }
    EXPECT_NE(ImGui::FindWindowByName("Properties")->DockNode, log->DockNode);
    EXPECT_GT(log->Pos.y, shell.Viewport().first.y) << "below the viewport";
    ASSERT_NE(log->DockNode->TabBar, nullptr);
    EXPECT_EQ(log->DockNode->TabBar->SelectedTabId, log->TabId);
    EXPECT_FALSE(ImGui::FindWindowByName("Devices")->DockTabIsVisible);
}

TEST(Shell, ShowsAndHidesPanels) {
    const UiFrames frames;
    Scene scene;
    Build(scene);
    Shell shell(scene, nullptr);
    frames.Draw(shell);
    shell.FindPanel("Profiler")->SetVisible(false);
    shell.FindPanel("Log")->SetVisible(false);
    frames.Draw(shell);
    EXPECT_FALSE(UiFrames::Drawn("Profiler"));
    EXPECT_FALSE(UiFrames::Drawn("Log"));
    EXPECT_TRUE(UiFrames::Drawn("Info"));
    EXPECT_EQ(shell.FindPanel("Nonesuch"), nullptr);
}

TEST(Shell, ShowsTheSelectedPrimsProperties) {
    const UiFrames frames;
    Scene scene;
    Build(scene);
    Shell shell(scene, nullptr);
    ASSERT_NE(shell.FindPanel("Properties"), nullptr);
    ASSERT_NE(shell.FindPanel("Info"), nullptr);
    frames.Draw(shell);
    shell.Selected().Set("/World/Rover/Wheel");
    frames.Draw(shell);
    shell.Selected().Set("/World/Gone");
    frames.Draw(shell);
    EXPECT_TRUE(UiFrames::Drawn("Properties")) << "draws a selection the scene lacks";
}

TEST(Shell, QuitsOnControlQ) {
    const UiFrames frames;
    Scene scene;
    Build(scene);
    Shell shell(scene, nullptr);
    frames.Draw(shell);
    ImGuiIO& io = ImGui::GetIO();
    // On macOS shortcuts' Ctrl is the Command key, so Cmd+Q quits there.
    io.AddKeyEvent(io.ConfigMacOSXBehaviors ? ImGuiMod_Super : ImGuiMod_Ctrl, true);
    io.AddKeyEvent(ImGuiKey_Q, true);
    frames.Draw(shell);
    EXPECT_FALSE(shell.Open());
}

TEST(Shell, FramesTheSceneOrTheSelection) {
    Scene scene;
    Build(scene);
    Shell shell(scene, nullptr);
    shell.Frame();
    EXPECT_NEAR(shell.Camera().target.x, 10.0, 1e-9) << "the whole scene: the one sphere";
    const double distance = shell.Camera().distance;
    EXPECT_NEAR(distance, 1.1 * std::sqrt(3.0) / std::sin(0.5 * shell.Camera().verticalFov), 1e-9);

    shell.Selected().Set("/World/Lamp");
    shell.Frame();
    EXPECT_EQ(shell.Camera().distance, distance) << "a selection without bounds moves nothing";
}

TEST(Shell, ZoomsWithTheWheelOverTheViewport) {
    const UiFrames frames;
    Scene scene;
    Build(scene);
    Shell shell(scene, nullptr);
    frames.Draw(shell);
    const auto [origin, size] = shell.Viewport();
    ImGuiIO& io = ImGui::GetIO();
    io.AddMousePosEvent(origin.x + 0.5f * size.x, origin.y + 0.5f * size.y);
    frames.Draw(shell);
    const double before = shell.Camera().distance;
    io.AddMouseWheelEvent(0.0f, 1.0f);
    frames.Draw(shell);
    EXPECT_LT(shell.Camera().distance, before);
}

TEST(Shell, CreatesUnderTheSelectedGroupAndUndoesAsOneStep) {
    Scene scene;
    Build(scene);
    Shell shell(scene, nullptr);
    shell.Selected().Set("/World");
    const Path sphere = shell.Create(Sphere::kName);
    EXPECT_EQ(sphere, Path("/World/Sphere"));
    EXPECT_EQ(shell.Selected().Paths(), std::vector<Path>{sphere});
    EXPECT_EQ(shell.Create(Sphere::kName), Path("/World/Sphere1")) << "numbered when taken";
    // Into a fresh scene, its default parent made too: undone in one step.
    shell.Selected().Clear();
    const Path light = shell.Create(DistantLight::kName);
    EXPECT_EQ(light, Path("/World/Sun"));
    shell.Selected().Clear();
    const Path settings = shell.Create(RenderSettings::kName);
    EXPECT_TRUE(scene.Has("/Render"));
    shell.History().Undo(scene);
    EXPECT_FALSE(scene.Has(settings));
    EXPECT_FALSE(scene.Has("/Render")) << "the parent it made, in the same step";
    EXPECT_THROW(shell.Create("Mystery"), std::invalid_argument);
}

TEST(Shell, DuplicatesAndDeletesTheSelection) {
    Scene scene;
    Build(scene);
    Shell shell(scene, nullptr);
    shell.Selected().Set("/World/Rover");
    shell.DuplicateSelection();
    EXPECT_TRUE(scene.Has("/World/Rover1/Wheel")) << "with its descendants";
    EXPECT_EQ(shell.Selected().Paths(), std::vector<Path>{Path("/World/Rover1")});
    shell.DeleteSelection();
    EXPECT_FALSE(scene.Has("/World/Rover1"));
    EXPECT_TRUE(shell.Selected().Empty());
    shell.History().Undo(scene);
    EXPECT_TRUE(scene.Has("/World/Rover1/Wheel"));
}

TEST(Shell, OpensAndSavesSceneFiles) {
    Scene scene;
    Build(scene);
    Shell shell(scene, nullptr);
    const std::filesystem::path file =
        std::filesystem::temp_directory_path() / "ernie_shell_test_scene.json";
    shell.SaveScene(file);
    shell.Selected().Set("/World/Lamp");
    shell.History().Remove(scene, "/World/Lamp");
    shell.OpenScene(file);
    EXPECT_TRUE(scene.Has("/World/Lamp"));
    EXPECT_FALSE(shell.History().CanUndo()) << "a new scene starts its own history";
    EXPECT_TRUE(shell.Selected().Empty());
    EXPECT_THROW(shell.OpenScene(file.string() + ".missing"), std::exception);
    EXPECT_TRUE(scene.Has("/World/Lamp")) << "a bad file leaves the scene as it was";
    std::filesystem::remove(file);
}

TEST(StagePanel, SearchKeepsMatchesAndTheirAncestors) {
    Scene scene;
    Build(scene);
    EXPECT_TRUE(app::StageMatches(scene, "").empty()) << "no search lists everything";
    const std::unordered_set<Path> matches = app::StageMatches(scene, "WHEEL");
    EXPECT_EQ(matches, (std::unordered_set<Path>{"/World", "/World/Rover", "/World/Rover/Wheel"}));
    EXPECT_TRUE(app::StageMatches(scene, "nothing").empty());
}

}  // namespace
}  // namespace ernie
