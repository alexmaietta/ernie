// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/app.h"
#include "ernie/device/memory_stats.h"
#include "ernie/device/vulkan_context.h"
#include "ernie/raster/delegate.h"
#include "ernie/scene/camera.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/stage_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/util/profiler.h"

#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <imgui_internal.h>
#include <stb_image.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ernie::app {
namespace {

namespace tokens = ernie::tokens;

/// A white cube at the origin lit by a sun, with render settings.
Scene CubeScene() {
    Scene scene;
    Cube::Define(scene, "/World/Cube").SetSize(2.0);
    scene.Define("/World/Sun", tokens::kDistantLight)
        .Set(tokens::kTransform,
             DistantLightTransform({0.4, -0.6, 1.0}));
    RenderSettings::Define(scene, "/Render/Settings");
    return scene;
}

/// A hidden viewer with validation, skipping without a Vulkan device
/// and failing on validation messages.
class ViewerTest : public ::testing::Test {
  protected:
    void SetUp() override {
        try {
            VulkanContext::Create();
        } catch (const std::runtime_error& error) {
            GTEST_SKIP() << "No Vulkan device: " << error.what();
        }
        scene_ = CubeScene();
        ViewerOptions options;
        options.hidden = true;
        options.validation = true;
        options.width = 640;
        options.height = 400;
        viewer_ = std::make_unique<Viewer>(scene_, options);
    }

    void TearDown() override {
        if (viewer_) {
            for (const std::string& message :
                 viewer_->Context().TakeValidationMessages()) {
                ADD_FAILURE() << "Vulkan validation: " << message;
            }
        }
    }

    std::filesystem::path TempFile(const std::string& name) const {
        return std::filesystem::temp_directory_path() /
               ("ernie_app_" +
                std::to_string(::testing::UnitTest::GetInstance()
                                   ->random_seed()) +
                "_" + name);
    }

    Scene scene_;
    std::unique_ptr<Viewer> viewer_;
};

}  // namespace
}  // namespace ernie::app

using namespace ernie;
using namespace ernie::app;

TEST_F(ViewerTest, DefinesItsPerspectiveCameraAndLooksThroughIt) {
    EXPECT_TRUE(scene_.Has(Viewer::PerspectiveCamera()));
    EXPECT_EQ(scene_.TypeOf(Viewer::PerspectiveCamera()),
              tokens::kCamera);
    EXPECT_EQ(viewer_->ActiveCamera(), Viewer::PerspectiveCamera());
    ASSERT_FALSE(viewer_->Cameras().empty());
    EXPECT_EQ(viewer_->Cameras().front(),
              Viewer::PerspectiveCamera());
    EXPECT_TRUE(
        Viewer::PerspectiveCamera().HasPrefix(Viewer::SessionRoot()));
    EXPECT_EQ(viewer_->Settings(), Path("/Render/Settings"));
}

TEST_F(ViewerTest, OpensFramingTheScene) {
    const CameraParams camera =
        CameraParamsOf(scene_, Viewer::PerspectiveCamera(), 64, 64);
    const glm::dvec3 toCube = glm::normalize(-camera.Eye());
    EXPECT_GT(glm::dot(toCube, camera.Forward()), 0.999);
}

TEST_F(ViewerTest, PausesItsSensorClockAndStepsIt) {
    viewer_->Step();
    viewer_->Step();
    const double running = viewer_->Delegate().Time();
    EXPECT_GT(running, 0.0) << "the clock runs with the frames";
    viewer_->SetClockPaused(true);
    EXPECT_TRUE(viewer_->ClockPaused());
    viewer_->Step();
    const double paused = viewer_->Delegate().Time();
    viewer_->Step();
    viewer_->Step();
    EXPECT_EQ(viewer_->Delegate().Time(), paused);
    viewer_->StepClock(0.25);
    viewer_->Step();
    EXPECT_DOUBLE_EQ(viewer_->Delegate().Time(), paused + 0.25);
    viewer_->Step();
    EXPECT_DOUBLE_EQ(viewer_->Delegate().Time(), paused + 0.25)
        << "one step only";
}

TEST_F(ViewerTest, RendersTheSceneIntoTheViewport) {
    for (int i = 0; i < 3; ++i) {
        ASSERT_TRUE(viewer_->Step());
    }
    const std::filesystem::path path = TempFile("capture.png");
    viewer_->Capture(path);
    ASSERT_TRUE(viewer_->Step());
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* texels = stbi_load(path.string().c_str(), &width,
                                      &height, &channels, 4);
    ASSERT_NE(texels, nullptr);
    EXPECT_GT(width, 100);
    EXPECT_GT(height, 100);
    const std::size_t rowLength = static_cast<std::size_t>(width);
    const unsigned char* centre =
        texels +
        4 * (static_cast<std::size_t>(height / 2) * rowLength +
             rowLength / 2);
    const unsigned char* corner = texels;
    EXPECT_GT(centre[0] + centre[1] + centre[2],
              corner[0] + corner[1] + corner[2] + 60);
    stbi_image_free(texels);
    std::filesystem::remove(path);
}

TEST_F(ViewerTest, ShowsEveryRenderVarTheBackendCanDraw) {
    const std::vector<std::string>& vars =
        viewer_->Delegate().Info().viewVars;
    EXPECT_EQ(vars.size(), kRenderVarNames.size());
    ASSERT_TRUE(viewer_->Step());
    // The centre pixel (the cube) and a corner (background) of a
    // capture of `var`. Without the grid and gizmos over the image.
    ViewportDisplay display;
    display.grid = display.lights = display.cameras = false;
    viewer_->SetViewportDisplay(display);
    const auto sample = [&](const std::string& var) {
        const std::filesystem::path path = TempFile("var.png");
        viewer_->Capture(path, var);
        EXPECT_TRUE(viewer_->Step());
        int width = 0;
        int height = 0;
        int channels = 0;
        unsigned char* texels = stbi_load(
            path.string().c_str(), &width, &height, &channels, 4);
        EXPECT_NE(texels, nullptr) << var;
        const std::size_t row = static_cast<std::size_t>(width);
        const unsigned char* centre =
            texels + 4 * (static_cast<std::size_t>(height / 2) * row +
                          row / 2);
        const std::array<std::array<int, 3>, 2> pixels{
            {{centre[0], centre[1], centre[2]},
             {texels[0], texels[1], texels[2]}}};
        stbi_image_free(texels);
        std::filesystem::remove(path);
        return pixels;
    };
    const auto sum = [](const std::array<int, 3>& rgb) {
        return rgb[0] + rgb[1] + rgb[2];
    };
    for (const std::string& var : vars) {
        const auto [centre, corner] = sample(var);
        // The cube is no instance (id 0), stands still (no motion)
        // and reflects its light unpolarized: black.
        if (var != "InstanceId" && var != "MotionVector" &&
            var != "DegreeOfLinearPolarization" &&
            var != "LinearStokes") {
            EXPECT_GT(sum(centre), 0) << var << " shows the cube";
        }
        // Colour shows the clear colour, sensor vars the
        // background's noise, and an angle of polarization 0 a hue.
        if (var != "LdrColor" && var != "HdrColor" &&
            var != "Electrons" && var != "RawDN" &&
            var != "AngleOfLinearPolarization") {
            EXPECT_EQ(sum(corner), 0)
                << var << " shows the background black";
        }
    }
    EXPECT_NE(sample("Normal")[0], sample("PrimId")[0]);
    EXPECT_EQ(viewer_->ViewVar(), "LdrColor")
        << "captures leave the shown var";

    // SpectralRadiance in grey, the mean over the bands or one of
    // them.
    const auto mean = sample("SpectralRadiance")[0];
    EXPECT_EQ(mean[0], mean[2]) << "grey";
    viewer_->SetViewBand(8);
    const auto band = sample("SpectralRadiance")[0];
    EXPECT_GT(sum(band), 0);
    EXPECT_EQ(viewer_->ViewBand(), 8);
    viewer_->SetViewBand(-5);
    EXPECT_EQ(viewer_->ViewBand(), -1);

    viewer_->SetViewVar("DistanceToCamera");
    ASSERT_TRUE(viewer_->Step());
    EXPECT_EQ(viewer_->ViewVar(), "DistanceToCamera");
}

TEST_F(ViewerTest, SwitchesCamerasAndFallsBackWhenOneIsRemoved) {
    CameraParams top;
    top.transform = glm::inverse(
        glm::lookAtRH(glm::dvec3(0.0, 0.0, 20.0), glm::dvec3(0.0),
                      glm::dvec3(0.0, 1.0, 0.0)));
    SetCamera(scene_, Path("/Cameras/Top"), top);
    const std::vector<Path> cameras = viewer_->Cameras();
    ASSERT_EQ(cameras.size(), 2u);
    EXPECT_EQ(cameras[1], Path("/Cameras/Top"));
    viewer_->SetActiveCamera(Path("/Cameras/Top"));
    EXPECT_EQ(viewer_->ActiveCamera(), Path("/Cameras/Top"));
    ASSERT_TRUE(viewer_->Step());
    EXPECT_EQ(scene_.Get(Path("/__session/RenderSettings"),
                         tokens::kCameraRel, Path()),
              Path("/Cameras/Top"));

    scene_.Remove(Path("/Cameras"));
    ASSERT_TRUE(viewer_->Step());
    EXPECT_EQ(viewer_->ActiveCamera(), Viewer::PerspectiveCamera());
    EXPECT_THROW(viewer_->SetActiveCamera(Path("/World/Cube")),
                 std::invalid_argument);
    EXPECT_THROW(viewer_->SetActiveCamera(Path("/Missing")),
                 std::invalid_argument);
}

TEST_F(ViewerTest, RecreatesItsCameraIfTheSceneRemovesIt) {
    scene_.Remove(Viewer::SessionRoot());
    ASSERT_TRUE(viewer_->Step());
    EXPECT_TRUE(scene_.Has(Viewer::PerspectiveCamera()));
    EXPECT_EQ(viewer_->ActiveCamera(), Viewer::PerspectiveCamera());
}

TEST_F(ViewerTest, FrameAimsAtTheGivenPrims) {
    Cube::Define(scene_, "/World/Far").SetSize(1.0);
    scene_.Set(
        "/World/Far", tokens::kTransform,
        glm::translate(glm::dmat4(1.0), glm::dvec3(30.0, 0.0, 0.0)));
    scene_.Set(
        "/World/Far", tokens::kExtent,
        std::vector<float>{-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f});
    const std::vector<Path> far{Path("/World/Far")};
    viewer_->Frame(far);
    const CameraParams camera =
        CameraParamsOf(scene_, Viewer::PerspectiveCamera(), 64, 64);
    const glm::dvec3 toFar =
        glm::normalize(glm::dvec3(30.0, 0.0, 0.0) - camera.Eye());
    EXPECT_GT(glm::dot(toFar, camera.Forward()), 0.999);
}

TEST_F(ViewerTest, LookAtPlacesTheActiveCamera) {
    // The viewer's own camera, and a scene camera once active, far
    // from the origin as a planet's view is.
    const glm::dvec3 eye(2.0e6, -1.0e6, 5.0e5);
    const glm::dvec3 target(0.0, 0.0, 0.0);
    for (const Path& path :
         {Viewer::PerspectiveCamera(), Path("/Cameras/Orbit")}) {
        if (path != Viewer::PerspectiveCamera()) {
            SetCamera(scene_, path, CameraParams{});
            viewer_->SetActiveCamera(path);
        }
        viewer_->LookAt(eye, target);
        const CameraParams camera =
            CameraParamsOf(scene_, path, 64, 64);
        // To a part in 10^6 of the distance: the controller's angles
        // are floats.
        EXPECT_NEAR(glm::length(camera.Eye() - eye), 0.0,
                    1e-6 * glm::length(eye - target))
            << path.String();
        EXPECT_GT(
            glm::dot(glm::normalize(target - eye), camera.Forward()),
            0.999999)
            << path.String();
    }
    EXPECT_THROW(viewer_->LookAt(eye, eye), std::invalid_argument);
}

TEST_F(ViewerTest, CameraFromViewCopiesTheActiveCamera) {
    const Path made = viewer_->CameraFromView(Path("/Cameras/Saved"));
    EXPECT_EQ(made, Path("/Cameras/Saved"));
    EXPECT_EQ(scene_.TypeOf(made), tokens::kCamera);
    const CameraParams saved = CameraParamsOf(scene_, made, 64, 64);
    const CameraParams view =
        CameraParamsOf(scene_, Viewer::PerspectiveCamera(), 64, 64);
    EXPECT_NEAR(glm::length(saved.Eye() - view.Eye()), 0.0, 1e-6);
    EXPECT_THROW(viewer_->CameraFromView(made),
                 std::invalid_argument);
    EXPECT_THROW(viewer_->CameraFromView(Path("/__session/Other")),
                 std::invalid_argument);
}

TEST_F(ViewerTest, FrameCallbacksRunEveryStepUntilRemoved) {
    std::vector<FrameInfo> seen;
    const std::uint64_t id =
        viewer_->OnFrame([&](Viewer&, const FrameInfo& info) {
            seen.push_back(info);
        });
    viewer_->Step();
    viewer_->Step();
    ASSERT_EQ(seen.size(), 2u);
    EXPECT_EQ(seen[1].frame, seen[0].frame + 1);
    EXPECT_GE(seen[1].time, seen[0].time);
    viewer_->RemoveCallback(id);
    viewer_->Step();
    EXPECT_EQ(seen.size(), 2u);
}

TEST_F(ViewerTest, FrameCallbackEditsShowInTheSameFrame) {
    viewer_->OnFrame([](Viewer& viewer, const FrameInfo&) {
        viewer.GetScene().Set(Path("/Render/Settings"),
                              "ernie:wireframe", true);
    });
    viewer_->Step();
    EXPECT_TRUE(scene_.Get(Path("/__session/RenderSettings"),
                           "ernie:wireframe", false));
}

TEST_F(ViewerTest, CallbacksRunInFrameUiPreRenderPostRenderOrder) {
    std::vector<std::string> order;
    viewer_->OnPostRender(
        [&](Viewer&, const FrameInfo&) { order.push_back("post"); });
    viewer_->OnPreRender(
        [&](Viewer&, const FrameInfo&) { order.push_back("pre"); });
    viewer_->OnUi([&](Viewer&, const FrameInfo&) {
        EXPECT_NE(ImGui::GetCurrentContext(), nullptr);
        ImGui::Begin("Overlay");
        ImGui::TextUnformatted("inside the ImGui frame");
        ImGui::End();
        order.push_back("ui");
    });
    viewer_->OnFrame(
        [&](Viewer&, const FrameInfo&) { order.push_back("frame"); });
    viewer_->Step();
    EXPECT_EQ(order, (std::vector<std::string>{"frame", "ui", "pre",
                                               "post"}));
}

TEST_F(ViewerTest,
       PreRenderEditsShowInTheFrameAndPostRenderEditsInTheNext) {
    viewer_->OnPreRender([](Viewer& viewer, const FrameInfo& info) {
        viewer.GetScene().Set(Path("/Render/Settings"), "test:pre",
                              static_cast<int>(info.frame));
    });
    viewer_->OnPostRender([](Viewer& viewer, const FrameInfo& info) {
        viewer.GetScene().Set(Path("/Render/Settings"), "test:post",
                              static_cast<int>(info.frame));
    });
    const Path session("/__session/RenderSettings");
    viewer_->Step();
    EXPECT_EQ(scene_.Get(session, "test:pre", -1), 0)
        << "mirrored into this frame's settings";
    EXPECT_EQ(scene_.Get(session, "test:post", -1), -1)
        << "set after the frame rendered";
    viewer_->Step();
    EXPECT_EQ(scene_.Get(session, "test:pre", -1), 1);
    EXPECT_EQ(scene_.Get(session, "test:post", -1), 0);
}

TEST_F(ViewerTest,
       PreAndPostRenderExceptionsPropagateAndTheViewerKeepsWorking) {
    bool failPre = true;
    bool failPost = false;
    viewer_->OnPreRender([&](Viewer&, const FrameInfo&) {
        if (failPre) {
            throw std::runtime_error("pre failed");
        }
    });
    viewer_->OnPostRender([&](Viewer&, const FrameInfo&) {
        if (failPost) {
            throw std::runtime_error("post failed");
        }
    });
    EXPECT_THROW(viewer_->Step(), std::runtime_error);
    failPre = false;
    failPost = true;
    EXPECT_THROW(viewer_->Step(), std::runtime_error);
    failPost = false;
    EXPECT_TRUE(viewer_->Step());
    EXPECT_TRUE(viewer_->Step());
}

TEST_F(ViewerTest, RemoveCallbackRemovesEveryKind) {
    int calls = 0;
    const auto count = [&](Viewer&, const FrameInfo&) { ++calls; };
    const std::vector<std::uint64_t> ids{
        viewer_->OnFrame(count), viewer_->OnUi(count),
        viewer_->OnPreRender(count), viewer_->OnPostRender(count)};
    viewer_->Step();
    EXPECT_EQ(calls, 4);
    for (const std::uint64_t id : ids) {
        viewer_->RemoveCallback(id);
    }
    viewer_->RemoveCallback(12345);
    viewer_->Step();
    EXPECT_EQ(calls, 4);
}

/// A panel docked on the left that counts its draws.
class CountingPanel final : public Panel {
  public:
    CountingPanel(std::string title, Dock dock, int& draws)
        : Panel(std::move(title), dock), draws_(draws) {}
    void Draw(Viewer&) override {
        ++draws_;
        ImGui::TextUnformatted("counted");
    }

  private:
    int& draws_;
};

TEST_F(ViewerTest, PanelsDockWhereTheyAsk) {
    int draws = 0;
    // Alone in its dock, as tabs hidden behind the built-in bottom
    // panels would not draw.
    for (const char* title : {"Log", "Info", "Overlays", "Capture"}) {
        viewer_->RemovePanel(title);
    }
    CountingPanel& left = viewer_->AddPanel<CountingPanel>(
        "Left Panel", Dock::Left, draws);
    viewer_->AddPanel<CountingPanel>("Bottom Panel", Dock::Bottom,
                                     draws);
    viewer_->AddPanel<CountingPanel>("Floating Panel", Dock::Floating,
                                     draws);
    EXPECT_EQ(left.GetDock(), Dock::Left);
    viewer_->Step();
    viewer_->Step();
    EXPECT_GE(draws, 6);
    const std::filesystem::path path = TempFile("docks.ini");
    viewer_->SaveLayout(path);
    std::ifstream file(path);
    const std::string text{std::istreambuf_iterator<char>(file),
                           std::istreambuf_iterator<char>()};
    file.close();  // Windows removes no file that is open.
    const auto docked = [&](const std::string& title) {
        const std::size_t at = text.find("[Window][" + title + "]");
        if (at == std::string::npos) {
            return false;
        }
        const std::size_t end = text.find("\n\n", at);
        return text.substr(at, end - at).find("DockId=") !=
               std::string::npos;
    };
    EXPECT_TRUE(docked("Left Panel"));
    EXPECT_TRUE(docked("Bottom Panel"));
    EXPECT_TRUE(docked("Properties"));
    EXPECT_FALSE(docked("Floating Panel"));
    std::filesystem::remove(path);
}

TEST_F(ViewerTest, ToolbarMenusCanBeHidden) {
    ViewportDisplay display = viewer_->GetViewportDisplay();
    EXPECT_TRUE(display.toolbarRenderer && display.toolbarVar &&
                display.toolbarCamera && display.toolbarLighting);
    for (bool ViewportDisplay::* menu :
         {&ViewportDisplay::toolbarRenderer,
          &ViewportDisplay::toolbarVar,
          &ViewportDisplay::toolbarCamera,
          &ViewportDisplay::toolbarLighting}) {
        display.*menu = false;
        viewer_->SetViewportDisplay(display);
        EXPECT_TRUE(viewer_->Step());
        EXPECT_EQ(viewer_->GetViewportDisplay(), display);
    }
    const std::filesystem::path path = TempFile("toolbar.png");
    viewer_->CaptureWindow(path);
    EXPECT_TRUE(std::filesystem::exists(path));
    std::filesystem::remove(path);
}

TEST_F(ViewerTest, DockSizesSetTheDefaultLayout) {
    // The height of the Log's dock as a fraction of the window's,
    // read during the UI.
    const auto logShare = [](Viewer& viewer) {
        float share = 0.0f;
        const std::uint64_t id =
            viewer.OnUi([&](Viewer&, const FrameInfo&) {
                if (const ImGuiWindow* log =
                        ImGui::FindWindowByName("Log")) {
                    share = log->Size.y /
                            ImGui::GetMainViewport()->WorkSize.y;
                }
            });
        for (int frame = 0; frame < 3; ++frame) {
            viewer.Step();
        }
        viewer.RemoveCallback(id);
        return share;
    };
    const float usual = logShare(*viewer_);
    viewer_.reset();
    ViewerOptions options;
    options.hidden = true;
    options.validation = true;
    options.width = 640;
    options.height = 400;
    options.docks.bottom = 0.1f;
    viewer_ = std::make_unique<Viewer>(scene_, options);
    const float small = logShare(*viewer_);
    EXPECT_NEAR(usual, 0.25f, 0.05f);
    EXPECT_NEAR(small, 0.1f, 0.05f);
}

TEST_F(ViewerTest, FunctionPanelsDrawAndPanelsAreFoundByTitle) {
    int draws = 0;
    Panel& panel = viewer_->AddPanel(
        "Function", Dock::Bottom, [&](Viewer& viewer) {
            EXPECT_EQ(&viewer, viewer_.get());
            ImGui::TextUnformatted("drawn by a function");
            ++draws;
        });
    EXPECT_EQ(panel.GetDock(), Dock::Bottom);
    EXPECT_EQ(viewer_->FindPanel("Function"), &panel);
    ASSERT_NE(viewer_->FindPanel("Info"), nullptr);
    EXPECT_EQ(viewer_->FindPanel("Missing"), nullptr);
    viewer_->Step();
    EXPECT_EQ(draws, 1);
    panel.SetVisible(false);
    viewer_->Step();
    EXPECT_EQ(draws, 1);
}

TEST_F(ViewerTest, FocusPanelBringsADockedTabToTheFront) {
    // Whether each title's window is the selected tab of its dock,
    // read during the UI.
    std::map<std::string, bool> front;
    viewer_->OnUi([&](Viewer&, const FrameInfo&) {
        for (const char* title : {"Info", "Log"}) {
            const ImGuiWindow* window =
                ImGui::FindWindowByName(title);
            front[title] =
                window != nullptr && window->DockNode != nullptr &&
                window->DockNode->SelectedTabId == window->TabId;
        }
    });
    viewer_->FindPanel("Log")->SetVisible(false);
    viewer_->FocusPanel("Log");
    EXPECT_TRUE(viewer_->FindPanel("Log")->IsVisible())
        << "focusing shows the panel";
    for (int frame = 0; frame < 4; ++frame) {
        viewer_->Step();
    }
    EXPECT_TRUE(front["Log"]);
    EXPECT_FALSE(front["Info"]);

    viewer_->FocusPanel("Info");
    for (int frame = 0; frame < 2; ++frame) {
        viewer_->Step();
    }
    EXPECT_TRUE(front["Info"]);
    EXPECT_FALSE(front["Log"]);
    EXPECT_THROW(viewer_->FocusPanel("Missing"),
                 std::invalid_argument);

    viewer_->AddPanel("Extra", Dock::Bottom, [](Viewer&) {});
    viewer_->FocusPanel("Extra");
    viewer_->RemovePanel("Extra");
    EXPECT_TRUE(viewer_->Step())
        << "a removed panel's focus request is dropped";
}

TEST_F(ViewerTest, FileDialogsDrawModallyAndCallBackOnTheViewer) {
    const std::filesystem::path dir = TempFile("dialog");
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "terrain.json") << "{}";
    std::filesystem::path chosen;
    Viewer* calledOn = nullptr;
    viewer_->OpenFileDialog(
        {"Open Terrain", false, {{"Terrains", {".json"}}}, dir},
        [&](Viewer& viewer, const std::filesystem::path& path) {
            calledOn = &viewer;
            chosen = path;
        });
    ASSERT_TRUE(viewer_->GetFileDialog().IsOpen());
    for (int frame = 0; frame < 3; ++frame) {
        viewer_->Step();
    }
    EXPECT_TRUE(viewer_->GetFileDialog().IsOpen())
        << "drawing keeps it open";
    viewer_->CaptureWindow(TempFile("dialog.png"));
    EXPECT_TRUE(viewer_->GetFileDialog().Accept("terrain.json"));
    EXPECT_EQ(calledOn, viewer_.get());
    EXPECT_EQ(chosen, dir / "terrain.json");
    viewer_->Step();
    EXPECT_FALSE(viewer_->GetFileDialog().IsOpen());
}

TEST_F(ViewerTest, OpensAndSavesScenesAsJson) {
    const std::filesystem::path saved = TempFile("saved_scene.json");
    viewer_->SaveScene(saved);
    std::ifstream savedFile(saved);
    const std::string text(
        (std::istreambuf_iterator<char>(savedFile)),
        std::istreambuf_iterator<char>());
    savedFile.close();  // Windows removes no file that is open.
    EXPECT_NE(text.find("\"/World\""), std::string::npos) << text;
    EXPECT_EQ(text.find(Viewer::SessionRoot().String()),
              std::string::npos)
        << "the viewer's own prims are not saved";

    const std::filesystem::path other = TempFile("other_scene.json");
    std::ofstream(other) << R"({"/Stage": {"type": "Xform",
        "Ball": {"type": "Sphere", "radius": 0.5, "translate": [0, 0, 1]},
        "Lamp": {"type": "SphereLight", "radius": 0.1}},
      "/Render/Other": {"type": "RenderSettings"}})";
    viewer_->SetSelection({Path("/World/Cube")});
    scene_.GetOverlay("user_marks");
    viewer_->OpenScene(other);
    EXPECT_FALSE(scene_.Has(Path("/World")))
        << "the stage's prims are replaced";
    EXPECT_EQ(Sphere(scene_, "/Stage/Ball").GetRadius(), 0.5);
    EXPECT_TRUE(scene_.Has(Viewer::PerspectiveCamera()))
        << "the viewer's prims stay";
    EXPECT_EQ(scene_.FindOverlay("user_marks"), nullptr);
    EXPECT_TRUE(viewer_->Selection().empty());
    EXPECT_EQ(viewer_->Settings(), Path("/Render/Other"));
    EXPECT_EQ(viewer_->ActiveCamera(), Viewer::PerspectiveCamera());
    EXPECT_TRUE(viewer_->Step());

    // A file that does not load leaves the stage alone.
    const std::filesystem::path broken =
        TempFile("broken_scene.json");
    std::ofstream(broken)
        << R"({"/Bad": {"type": "Sphere", "radus": 1}})";
    EXPECT_THROW(viewer_->OpenScene(broken), std::invalid_argument);
    EXPECT_THROW(viewer_->OpenScene(TempFile("no_such_scene.json")),
                 std::invalid_argument);
    EXPECT_TRUE(scene_.Has(Path("/Stage/Ball")));
    EXPECT_FALSE(scene_.Has(Path("/Bad")));

    // Back to the saved scene, through File > Open Scene's dialog.
    viewer_->OpenFileDialog(
        {"Open Scene", false, {{"Scenes", {".json"}}}, saved},
        [](Viewer& viewer, const std::filesystem::path& path) {
            viewer.OpenScene(path);
        });
    EXPECT_TRUE(
        viewer_->GetFileDialog().Accept(saved.filename().string()));
    EXPECT_EQ(Cube(scene_, "/World/Cube").GetSize(), 2.0);
    EXPECT_FALSE(scene_.Has(Path("/Stage")));
    EXPECT_TRUE(viewer_->Step());
    for (const std::filesystem::path& path : {saved, other, broken}) {
        std::filesystem::remove(path);
    }
}

TEST_F(ViewerTest, RendererIsAViewerOptionAndSwitchable) {
    const std::vector<std::string> renderers = Viewer::Renderers();
    ASSERT_FALSE(renderers.empty());
    EXPECT_EQ(viewer_->Renderer(), "raster");
    EXPECT_NE(std::find(renderers.begin(), renderers.end(), "raster"),
              renderers.end());
    EXPECT_THROW(viewer_->SetRenderer("no_such_backend"),
                 std::invalid_argument);
    EXPECT_EQ(viewer_->Renderer(), "raster")
        << "a failed switch keeps the current backend";
    viewer_->SetRenderer("raster");
    EXPECT_TRUE(viewer_->Step());
    EXPECT_GT(viewer_->Stats().triangles, 0u);
    EXPECT_NE(viewer_->Brdfs().Find("lambert"), nullptr);
}

TEST_F(ViewerTest, SyncReportsDataTheRendererCannotUse) {
    EXPECT_NO_THROW(viewer_->Sync());
    Cube::Define(scene_, "/World/Broken").SetSize(-1.0);
    EXPECT_THROW(viewer_->Sync(), std::invalid_argument);
}

TEST(ViewerOptionsTest, RejectsAnUnknownRenderer) {
    try {
        VulkanContext::Create();
    } catch (const std::runtime_error& error) {
        GTEST_SKIP() << "No Vulkan device: " << error.what();
    }
    Scene scene = CubeScene();
    ViewerOptions options;
    options.hidden = true;
    options.renderer = "no_such_backend";
    EXPECT_THROW(Viewer(scene, options), std::invalid_argument);
}

TEST_F(ViewerTest, SelectionCallbacksFireOnlyOnChange) {
    int calls = 0;
    std::vector<Path> last;
    viewer_->OnSelection([&](Viewer&, std::span<const Path> paths) {
        ++calls;
        last.assign(paths.begin(), paths.end());
    });
    viewer_->SetSelection({Path("/World/Cube")});
    viewer_->SetSelection({Path("/World/Cube")});
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(last, std::vector<Path>{Path("/World/Cube")});
    EXPECT_EQ(viewer_->Selection(), last);
}

TEST_F(ViewerTest, CloseEndsRun) {
    int frames = 0;
    viewer_->OnFrame([&](Viewer& viewer, const FrameInfo&) {
        if (++frames == 3) {
            viewer.Close();
        }
    });
    viewer_->Run();
    EXPECT_EQ(frames, 3);
    EXPECT_FALSE(viewer_->IsOpen());
    EXPECT_FALSE(viewer_->Step());
}

TEST_F(ViewerTest,
       CallbackExceptionsPropagateAndTheViewerKeepsWorking) {
    bool fail = true;
    viewer_->OnFrame([&](Viewer&, const FrameInfo&) {
        if (fail) {
            throw std::runtime_error("callback failed");
        }
    });
    EXPECT_THROW(viewer_->Step(), std::runtime_error);
    fail = false;
    EXPECT_TRUE(viewer_->Step());
}

/// A panel that throws while drawing.
class ThrowingPanel final : public Panel {
  public:
    ThrowingPanel() : Panel("Throwing") {}
    void Draw(Viewer&) override {
        ImGuiThrow();
    }

  private:
    static void ImGuiThrow() {
        throw std::runtime_error("panel failed");
    }
};

TEST_F(ViewerTest, PanelExceptionsPropagateAndTheViewerRecovers) {
    viewer_->AddPanel(std::make_unique<ThrowingPanel>());
    EXPECT_THROW(viewer_->Step(), std::runtime_error);
    viewer_->RemovePanel("Throwing");
    EXPECT_TRUE(viewer_->Step());
}

TEST_F(ViewerTest, PanelsAndMenuItemsAreManagedByTitle) {
    WidgetPanel& panel = viewer_->AddWidgetPanel("Controls");
    panel.Slider("Sun", 0.0, 90.0, 10.0, nullptr);
    panel.Plot("History",
               [] { return std::vector<float>{1.0f, 3.0f, 2.0f}; });
    viewer_->AddMenuItem("Tools", "Reset", [](Viewer&) {});
    EXPECT_TRUE(viewer_->Step());
    EXPECT_THROW(viewer_->AddWidgetPanel("Controls"),
                 std::invalid_argument);
    EXPECT_THROW(viewer_->AddWidgetPanel("Viewport"),
                 std::invalid_argument);
    EXPECT_THROW(viewer_->AddPanel(nullptr), std::invalid_argument);
    EXPECT_THROW(viewer_->AddMenuItem("", "x", nullptr),
                 std::invalid_argument);
    viewer_->RemovePanel("Controls");
    EXPECT_THROW(viewer_->RemovePanel("Controls"),
                 std::invalid_argument);
    viewer_->RemovePanel("Info");
    EXPECT_NO_THROW(viewer_->RemovePanel("Info"));
    EXPECT_TRUE(viewer_->Step());
}

TEST_F(ViewerTest, ViewportTogglesShowOverlaysAndHudItemsList) {
    const Path session("/__session/RenderSettings");
    const auto shown = [&] {
        std::istringstream words(
            GetToken(scene_, session, tokens::kOverlays, ""));
        return std::set<std::string>(
            std::istream_iterator<std::string>(words), {});
    };
    scene_.GetOverlay("contacts")
        .Line({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0});
    scene_.GetOverlay("labels").Line({0.0, 0.0, 0.0},
                                     {0.0, 1.0, 0.0});
    std::vector<bool> switched;
    const std::uint64_t toggle = viewer_->AddViewportToggle(
        "Contacts", false, "contacts",
        [&](Viewer&, bool on) { switched.push_back(on); });
    int asked = 0;
    const std::uint64_t hud =
        viewer_->AddHudItem("Physics", [&](Viewer&) {
            ++asked;
            return std::string("1 ms");
        });
    viewer_->AddViewportMenuItem(
        "Pause", [](Viewer&) {}, [](const Viewer&) { return true; });
    // The grid, off by default.
    ViewportDisplay withGrid = viewer_->GetViewportDisplay();
    withGrid.grid = true;
    viewer_->SetViewportDisplay(withGrid);
    viewer_->Step();
    EXPECT_FALSE(shown().contains("contacts"))
        << "its toggle hides it";
    EXPECT_TRUE(shown().contains("labels"));
    EXPECT_TRUE(shown().contains("__viewport"));
    EXPECT_GT(asked, 0);
    EXPECT_FALSE(scene_.GetOverlay("__viewport").Empty())
        << "the grid";
    // A layer hidden on its own, the others still shown.
    withGrid.hiddenOverlays = {"labels"};
    viewer_->SetViewportDisplay(withGrid);
    viewer_->Step();
    EXPECT_FALSE(shown().contains("labels"));
    EXPECT_TRUE(shown().contains("__viewport"));
    withGrid.hiddenOverlays.clear();
    viewer_->SetViewportDisplay(withGrid);
    viewer_->Step();
    EXPECT_TRUE(shown().contains("labels"));

    viewer_->SetViewportToggleShown(toggle, true);
    ViewportDisplay display = viewer_->GetViewportDisplay();
    display.overlays = false;
    display.grid = false;
    display.lights = false;
    display.cameras = false;
    viewer_->SetViewportDisplay(display);
    viewer_->Step();
    EXPECT_TRUE(viewer_->ViewportToggleShown(toggle));
    EXPECT_EQ(switched, std::vector<bool>{true});
    EXPECT_TRUE(shown().contains("contacts"))
        << "toggled overlays follow their toggle alone";
    EXPECT_FALSE(shown().contains("labels"));
    EXPECT_TRUE(scene_.GetOverlay("__viewport").Empty());

    viewer_->SetHudItemShown(hud, false);
    EXPECT_FALSE(viewer_->HudItemShown(hud));
    viewer_->RemoveViewportItem(toggle);
    viewer_->RemoveViewportItem(hud);
    EXPECT_THROW(viewer_->ViewportToggleShown(toggle),
                 std::invalid_argument);
    EXPECT_THROW(viewer_->AddViewportToggle(""),
                 std::invalid_argument);
    EXPECT_THROW(viewer_->AddHudItem("x", nullptr),
                 std::invalid_argument);
    EXPECT_TRUE(viewer_->Step());
}

TEST_F(ViewerTest, DuplicateAndRenameAreSingleUndoSteps) {
    Sphere::Define(scene_, "/World/Cube/Child").SetRadius(0.5);
    const std::vector<Path> copies =
        viewer_->Duplicate(std::vector<Path>{Path("/World/Cube")});
    ASSERT_EQ(copies, std::vector<Path>{Path("/World/Cube_01")});
    EXPECT_EQ(scene_.TypeOf("/World/Cube_01"), tokens::kCube);
    EXPECT_DOUBLE_EQ(
        GetNumber(scene_, "/World/Cube_01", tokens::kSize, 0.0), 2.0);
    EXPECT_TRUE(scene_.Has("/World/Cube_01/Child"))
        << "with its descendants";
    EXPECT_EQ(viewer_->Selection(), copies);
    EXPECT_EQ(
        viewer_->Duplicate(std::vector<Path>{Path("/World/Cube")})
            .front(),
        Path("/World/Cube_02"));

    const Path renamed = viewer_->Rename("/World/Cube_01", "Box");
    EXPECT_EQ(renamed, Path("/World/Box"));
    EXPECT_FALSE(scene_.Has("/World/Cube_01"));
    EXPECT_TRUE(scene_.Has("/World/Box/Child"));
    EXPECT_THROW(viewer_->Rename("/World/Box", "Cube"),
                 std::invalid_argument)
        << "taken";
    EXPECT_THROW(viewer_->Rename("/World/Box", "bad name"),
                 std::invalid_argument);
    ASSERT_TRUE(viewer_->Undo());
    EXPECT_TRUE(scene_.Has("/World/Cube_01/Child"))
        << "one step undoes the rename";
    EXPECT_FALSE(scene_.Has("/World/Box"));
    EXPECT_TRUE(viewer_->Step());
}

TEST_F(ViewerTest, OutlinesTheSelectionInTheViewport) {
    // Pixels of the outline colour, Kit's orange.
    const auto outlined = [&] {
        const std::filesystem::path path = TempFile("outline.png");
        viewer_->Capture(path);
        EXPECT_TRUE(viewer_->Step());
        int width = 0;
        int height = 0;
        int channels = 0;
        unsigned char* texels = stbi_load(
            path.string().c_str(), &width, &height, &channels, 4);
        std::size_t count = 0;
        const glm::vec4 colour = viewer_->GetTheme().outline;
        for (int i = 0; i < width * height; ++i) {
            const unsigned char* p = texels + 4 * i;
            if (std::abs(p[0] - colour.r * 255.0f) < 12.0f &&
                std::abs(p[1] - colour.g * 255.0f) < 12.0f &&
                std::abs(p[2] - colour.b * 255.0f) < 12.0f) {
                ++count;
            }
        }
        stbi_image_free(texels);
        std::filesystem::remove(path);
        return count;
    };
    viewer_->Step();
    EXPECT_LT(outlined(), 5u) << "nothing selected";
    viewer_->SetSelection({Path("/World/Cube")});
    EXPECT_GT(outlined(), 100u) << "the cube's silhouette";
    ViewportDisplay display = viewer_->GetViewportDisplay();
    display.selection = false;
    viewer_->SetViewportDisplay(display);
    EXPECT_LT(outlined(), 5u) << "the eye menu hides it";
}

TEST_F(ViewerTest, LayoutsRoundTripThroughFiles) {
    viewer_->Step();
    const std::filesystem::path path = TempFile("layout.ini");
    viewer_->SaveLayout(path);
    std::ifstream file(path);
    const std::string text{std::istreambuf_iterator<char>(file),
                           std::istreambuf_iterator<char>()};
    file.close();  // Windows removes no file that is open.
    EXPECT_NE(text.find("[Docking]"), std::string::npos);
    EXPECT_NE(text.find("Viewport"), std::string::npos);
    viewer_->LoadLayout(path);
    EXPECT_TRUE(viewer_->Step());
    std::filesystem::remove(path);
    EXPECT_THROW(viewer_->LoadLayout(TempFile("missing.ini")),
                 std::runtime_error);
    EXPECT_THROW(
        viewer_->LoadLayout(std::filesystem::temp_directory_path()),
        std::runtime_error);
}

TEST_F(ViewerTest, ShowsRenderErrorsWithoutThrowing) {
    Cube::Define(scene_, "/World/Broken").SetSize(-1.0);
    EXPECT_NO_THROW(viewer_->Step());
    scene_.Remove(Path("/World/Broken"));
    EXPECT_TRUE(viewer_->Step());
}

TEST_F(ViewerTest,
       RendersWithItsSessionSettingsWhileTheSceneHasNone) {
    // Without the scene's, the viewer's own.
    scene_.Remove(Path("/Render"));
    EXPECT_TRUE(viewer_->Step());
    EXPECT_EQ(viewer_->Settings(), Viewer::SessionSettings());
    EXPECT_EQ(scene_.TypeOf(Viewer::SessionSettings()),
              tokens::kRenderSettings);
    // Edited as a scene's are, and rendered with.
    viewer_->Edit(Viewer::SessionSettings(), tokens::kWireframe,
                  std::string(tokens::kOverlay));
    EXPECT_TRUE(viewer_->Step());
    EXPECT_EQ(*scene_.Get(Path("/__session/RenderSettings"),
                          tokens::kWireframe),
              Value(std::string(tokens::kOverlay)))
        << "mirrored into what the viewport renders with";

    // Settings the scene gains take over; the session's stay to
    // choose again.
    RenderSettings::Define(scene_, "/Render/Settings");
    EXPECT_TRUE(viewer_->Step());
    EXPECT_EQ(viewer_->Settings(), Path("/Render/Settings"));
    viewer_->SetSettings(Viewer::SessionSettings());
    EXPECT_EQ(viewer_->Settings(), Viewer::SessionSettings());
}

TEST_F(ViewerTest, PlaysTheTimelineBeforeTheFrameCallbacks) {
    Stage(scene_).SetStartTimeCode(0.0).SetEndTimeCode(48.0);
    Timeline& timeline = viewer_->GetTimeline();
    timeline.SetPlayEveryFrame(true);
    std::vector<double> seen;
    viewer_->OnFrame([&seen](Viewer&, const FrameInfo& info) {
        seen.push_back(info.timeCode);
    });
    EXPECT_TRUE(viewer_->Step());
    timeline.Play();
    EXPECT_TRUE(viewer_->Step());
    EXPECT_TRUE(viewer_->Step());
    ASSERT_EQ(seen.size(), 3u);
    EXPECT_EQ(seen[0], 0.0) << "held while paused";
    EXPECT_EQ(seen[1], 1.0) << "a frame a step";
    EXPECT_EQ(seen[2], 2.0);
    EXPECT_EQ(Stage(scene_).GetTimeCode(), 2.0);
    timeline.Stop(scene_);
    EXPECT_TRUE(viewer_->Step());
    EXPECT_EQ(Stage(scene_).GetTimeCode(), 0.0);
}

TEST_F(ViewerTest, ScreenshotsAndSequencesLandInTheCaptureFolder) {
    const std::filesystem::path dir = TempFile("captures");
    std::filesystem::remove_all(dir);
    viewer_->GetCaptureOptions().folder = dir;
    viewer_->GetCaptureOptions().name = "shot";
    const std::filesystem::path shot = viewer_->Screenshot();
    EXPECT_EQ(shot, dir / "shot.png");
    EXPECT_TRUE(viewer_->Step());
    EXPECT_TRUE(std::filesystem::exists(shot))
        << "the folder is made";
    ASSERT_FALSE(viewer_->RecentCaptures().empty());
    EXPECT_EQ(viewer_->RecentCaptures().back(), shot);
    EXPECT_EQ(viewer_->Screenshot(), dir / "shot_1.png")
        << "never over another";
    EXPECT_TRUE(viewer_->Step());

    // Every other frame of 0 to 4: a frame a step at 0, 2 and 4.
    Stage(scene_).SetStartTimeCode(0.0).SetEndTimeCode(4.0);
    std::vector<double> recorded;
    viewer_->OnFrame(
        [&recorded](Viewer& viewer, const FrameInfo& info) {
            if (viewer.Recording()) {
                recorded.push_back(info.timeCode);
            }
        });
    SequenceOptions sequence;
    sequence.every = 2;
    sequence.video = VideoFormat::None;
    viewer_->RecordSequence(sequence);
    EXPECT_EQ(viewer_->RecordingProgress().second, 3);
    for (int i = 0; i < 10 && viewer_->Recording(); ++i) {
        EXPECT_TRUE(viewer_->Step());
    }
    EXPECT_FALSE(viewer_->Recording());
    EXPECT_EQ(recorded, (std::vector<double>{0.0, 2.0, 4.0}));
    for (const char* frame :
         {"shot_0000.png", "shot_0001.png", "shot_0002.png"}) {
        EXPECT_TRUE(std::filesystem::exists(dir / "shot" / frame))
            << frame;
    }
    EXPECT_FALSE(
        std::filesystem::exists(dir / "shot" / "shot_0003.png"));

    if (!FindFfmpeg().empty()) {
        // Into an MP4, without its frames.
        sequence.every = 1;
        sequence.video = VideoFormat::Mp4;
        sequence.keepFrames = false;
        viewer_->RecordSequence(sequence);
        for (int i = 0; i < 2000 &&
                        (viewer_->Recording() || viewer_->Encoding());
             ++i) {
            EXPECT_TRUE(viewer_->Step());
        }
        EXPECT_TRUE(std::filesystem::exists(dir / "shot_1.mp4"));
        EXPECT_FALSE(std::filesystem::exists(dir / "shot_1"))
            << "its frames removed";
        EXPECT_EQ(viewer_->RecentCaptures().back(),
                  dir / "shot_1.mp4");
    }
    std::filesystem::remove_all(dir);
}

TEST_F(ViewerTest, CapturesRenderVarsAnewAtAScaleOfTheViewport) {
    const std::filesystem::path dir = TempFile("var_captures");
    std::filesystem::remove_all(dir);
    CaptureOptions& options = viewer_->GetCaptureOptions();
    options.folder = dir;
    options.name = "vars";
    // The layout settles over the first frames.
    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(viewer_->Step());
    }
    // The viewport as shown, for its size.
    const std::filesystem::path shown = viewer_->Screenshot();
    EXPECT_TRUE(viewer_->Step());
    const auto size = [](const std::filesystem::path& path) {
        int width = 0;
        int height = 0;
        int channels = 0;
        EXPECT_TRUE(stbi_info(path.string().c_str(), &width, &height,
                              &channels));
        return std::pair{width, height};
    };
    const auto [width, height] = size(shown);

    options.name = "rendered";
    options.vars = {"LdrColor", "DistanceToCamera"};
    options.scale = 2.0;
    viewer_->Screenshot();
    EXPECT_TRUE(viewer_->Step());
    const std::vector<std::filesystem::path>& written =
        viewer_->RecentCaptures();
    ASSERT_EQ(written.size(), 3u);
    EXPECT_EQ(written[1], dir / "rendered.png");
    EXPECT_EQ(written[2], dir / "rendered_DistanceToCamera.exr");
    EXPECT_EQ(size(dir / "rendered.png"),
              std::pair(2 * width, 2 * height))
        << "twice the viewport's render resolution";
    EXPECT_TRUE(std::filesystem::exists(
        dir / "rendered_DistanceToCamera.exr"));
    std::filesystem::remove_all(dir);
}

TEST_F(ViewerTest, HidesItsViewportUi) {
    ViewportDisplay display = viewer_->GetViewportDisplay();
    EXPECT_TRUE(display.ui);
    display.ui = false;
    viewer_->SetViewportDisplay(display);
    EXPECT_TRUE(viewer_->Step());
    EXPECT_FALSE(viewer_->GetViewportDisplay().ui);
    display.ui = true;
    viewer_->SetViewportDisplay(display);
    EXPECT_TRUE(viewer_->Step());
}

TEST_F(ViewerTest, RejectsVarsAndSettingsItCannotShowYet) {
    EXPECT_EQ(viewer_->ViewVar(), "LdrColor");
    EXPECT_THROW(viewer_->SetViewVar("NotAVar"),
                 std::invalid_argument);
    EXPECT_THROW(viewer_->SetViewVar("Depth"), std::invalid_argument);
    EXPECT_NO_THROW(viewer_->SetViewVar("SpectralRadiance"))
        << "as grey, by band or mean";
    EXPECT_NO_THROW(viewer_->SetViewVar("LdrColor"));
    EXPECT_THROW(viewer_->SetSettings(Path("/World/Cube")),
                 std::invalid_argument);
    EXPECT_THROW(
        viewer_->SetSettings(Path("/__session/RenderSettings")),
        std::invalid_argument);
    RenderSettings::Define(scene_, "/Render/Other");
    viewer_->SetSettings(Path("/Render/Other"));
    EXPECT_EQ(viewer_->Settings(), Path("/Render/Other"));
    EXPECT_THROW(viewer_->Capture(TempFile("x.png"), "Depth"),
                 std::invalid_argument);
}

TEST_F(ViewerTest, LightingModesChooseTheSessionLights) {
    EXPECT_EQ(viewer_->GetLighting(), Lighting::Stage)
        << "the scene has a sun";
    const Path session("/__session/RenderSettings");
    viewer_->Step();
    EXPECT_EQ(scene_.Get(session, tokens::kLightsRel), nullptr);

    viewer_->SetLighting(Lighting::Camera);
    viewer_->Step();
    EXPECT_EQ(
        scene_.Get(session, tokens::kLightsRel, std::vector<Path>{}),
        std::vector<Path>{Path("/__session/Lights/Camera")});
    const CameraParams view =
        CameraParamsOf(scene_, viewer_->ActiveCamera(), 64, 64);
    EXPECT_NEAR(
        glm::dot(
            TowardsLight(scene_, Path("/__session/Lights/Camera")),
            -view.Forward()),
        1.0, 1e-6);

    viewer_->SetLighting(Lighting::Default);
    viewer_->Step();
    EXPECT_EQ(
        scene_.Get(session, tokens::kLightsRel, std::vector<Path>{}),
        std::vector<Path>{Path("/__session/Lights/Default")});
    EXPECT_EQ(viewer_->GetLighting(), Lighting::Default);

    viewer_->SetLighting(Lighting::Stage);
    viewer_->Step();
    EXPECT_EQ(scene_.Get(session, tokens::kLightsRel), nullptr);
}

TEST_F(ViewerTest,
       WireframeModesOverrideTheSettingsInTheViewportOnly) {
    const Path session("/__session/RenderSettings");
    scene_.Set(Path("/Render/Settings"), tokens::kWireframe, true);
    viewer_->Step();
    EXPECT_FALSE(viewer_->GetWireframe().has_value());
    EXPECT_EQ(WireframeOf(scene_, session), Wireframe::Overlay)
        << "follows the settings";

    viewer_->SetWireframe(Wireframe::HiddenLine);
    viewer_->Step();
    EXPECT_EQ(WireframeOf(scene_, session), Wireframe::HiddenLine);
    EXPECT_EQ(WireframeOf(scene_, Path("/Render/Settings")),
              Wireframe::Overlay)
        << "the scene's settings are untouched";

    viewer_->SetWireframe(std::nullopt);
    viewer_->Step();
    EXPECT_EQ(WireframeOf(scene_, session), Wireframe::Overlay);
}

TEST_F(ViewerTest, HeadlessRendersOfTheSceneIgnoreTheSessionPrims) {
    viewer_->SetLighting(Lighting::Camera);
    viewer_->Step();
    const RenderPrims prims = ResolveRenderPrims(scene_, {});
    EXPECT_EQ(prims.settings, Path("/Render/Settings"));
    EXPECT_EQ(prims.lights, std::vector<Path>{Path("/World/Sun")});
}

TEST(ViewerLightingTest,
     AnUnlitSceneOpensWithTheDefaultLightAndIsLit) {
    try {
        VulkanContext::Create();
    } catch (const std::runtime_error& error) {
        GTEST_SKIP() << "No Vulkan device: " << error.what();
    }
    Scene scene;
    Cube::Define(scene, "/World/Cube").SetSize(2.0);
    ViewerOptions options;
    options.hidden = true;
    options.width = 320;
    options.height = 240;
    Viewer viewer(scene, options);
    EXPECT_EQ(viewer.GetLighting(), Lighting::Default);
    viewer.Step();
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        "ernie_app_unlit_capture.png";
    viewer.Capture(path);
    viewer.Step();
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* texels = stbi_load(path.string().c_str(), &width,
                                      &height, &channels, 4);
    ASSERT_NE(texels, nullptr);
    long brightest = 0;
    for (int i = 0; i < width * height; ++i) {
        const unsigned char* texel =
            texels + 4 * static_cast<std::size_t>(i);
        brightest =
            std::max<long>(brightest, texel[0] + texel[1] + texel[2]);
    }
    EXPECT_GT(brightest, 200);
    stbi_image_free(texels);
    std::filesystem::remove(path);
}

TEST_F(ViewerTest, RenderResolutionAndScaleSetTheViewportImage) {
    const auto captured = [&]() {
        for (int frame = 0; frame < 3; ++frame) {
            EXPECT_TRUE(viewer_->Step());
        }
        const std::filesystem::path path = TempFile("resolution.png");
        viewer_->Capture(path);
        EXPECT_TRUE(viewer_->Step());
        int width = 0;
        int height = 0;
        int channels = 0;
        unsigned char* texels = stbi_load(
            path.string().c_str(), &width, &height, &channels, 4);
        stbi_image_free(texels);
        std::filesystem::remove(path);
        return std::pair(width, height);
    };
    viewer_->SetRenderResolution(std::pair(320u, 200u));
    EXPECT_EQ(captured(), std::pair(320, 200));
    viewer_->SetRenderResolution(std::pair(320u, 200u), 0.5f, false);
    EXPECT_EQ(captured(), std::pair(160, 100));
    viewer_->SetRenderResolution(std::nullopt);
    EXPECT_NE(captured(), std::pair(320, 200)) << "the panel's own";
    EXPECT_THROW(viewer_->SetRenderResolution(std::nullopt, 0.0f),
                 std::invalid_argument);
}

TEST_F(ViewerTest, ThemesApplyAtRunTime) {
    Theme theme;
    theme.panel = {0.5f, 0.5f, 0.5f, 1.0f};
    viewer_->SetTheme(theme);
    EXPECT_EQ(viewer_->GetTheme().panel, theme.panel);
    EXPECT_TRUE(viewer_->Step());
}

TEST_F(ViewerTest, TheProfilerPanelRecordsGpuZonesWhileShown) {
    profiler::Clear();
    viewer_->FindPanel("Profiler")->SetVisible(true);
    for (int i = 0; i < 6; ++i) {
        ASSERT_TRUE(viewer_->Step());
    }
    std::set<std::string> gpu;
    std::size_t cpu = 0;
    for (const profiler::Frame& frame : profiler::Frames()) {
        for (const profiler::Zone& zone : frame.gpu) {
            gpu.insert(zone.name);
        }
        cpu += frame.cpu.size();
    }
    EXPECT_TRUE(gpu.count("Shading") == 1 || gpu.count("UI") == 1);
    EXPECT_GT(cpu, 0u);
    EXPECT_FALSE(memory::Live().empty())
        << "the renderer's targets are tracked";

    // Behind another tab it is not drawn, and stops recording.
    viewer_->FocusPanel("Log");
    for (int i = 0; i < 3; ++i) {
        ASSERT_TRUE(viewer_->Step());
    }
    EXPECT_FALSE(profiler::Enabled());
    viewer_->FocusPanel("Profiler");
    for (int i = 0; i < 3; ++i) {
        ASSERT_TRUE(viewer_->Step());
    }
    EXPECT_TRUE(profiler::Enabled());

    viewer_->FindPanel("Profiler")->SetVisible(false);
    ASSERT_TRUE(viewer_->Step());
    EXPECT_FALSE(profiler::Enabled());
}

TEST_F(ViewerTest, TheLogIsTheBottomDocksOpeningTab) {
    bool front = false;
    viewer_->OnUi([&](Viewer&, const FrameInfo&) {
        const ImGuiWindow* window = ImGui::FindWindowByName("Log");
        front = window != nullptr && window->DockNode != nullptr &&
                window->DockNode->SelectedTabId == window->TabId;
    });
    for (int frame = 0; frame < 3; ++frame) {
        ASSERT_TRUE(viewer_->Step());
    }
    EXPECT_TRUE(front);
}

TEST_F(ViewerTest, TheInfoPanelCountsTheScene) {
    viewer_->FocusPanel("Info");
    EXPECT_TRUE(viewer_->Step());
    EXPECT_TRUE(viewer_->Step());
}

TEST(ViewerOptionsTest, RejectsAnEmptyWindowAndUnknownSettings) {
    try {
        VulkanContext::Create();
    } catch (const std::runtime_error& error) {
        GTEST_SKIP() << "No Vulkan device: " << error.what();
    }
    Scene scene = CubeScene();
    ViewerOptions options;
    options.hidden = true;
    options.width = 0;
    EXPECT_THROW(Viewer(scene, options), std::invalid_argument);
    options.width = 64;
    options.settings = Path("/World/Cube");
    EXPECT_THROW(Viewer(scene, options), std::invalid_argument);
}
