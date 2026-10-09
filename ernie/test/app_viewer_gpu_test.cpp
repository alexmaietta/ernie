#include "ernie/app/app.h"
#include "ernie/device/vulkan_context.h"
#include "ernie/raster/delegate.h"
#include "ernie/scene/camera.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"

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
    scene.Define("/World/Cube", tokens::kCube).Set(tokens::kSize, 2.0);
    scene.Define("/World/Sun", tokens::kDistantLight)
        .Set(tokens::kTransform, DistantLightTransform({0.4, -0.6, 1.0}));
    scene.Define("/Render/Settings", tokens::kRenderSettings);
    return scene;
}

/// A hidden viewer with validation, skipping without a Vulkan device and failing on validation
/// messages.
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
            for (const std::string& message : viewer_->Context().TakeValidationMessages()) {
                ADD_FAILURE() << "Vulkan validation: " << message;
            }
        }
    }

    std::filesystem::path TempFile(const std::string& name) const {
        return std::filesystem::temp_directory_path() /
               ("ernie_app_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()) +
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
    EXPECT_EQ(scene_.TypeOf(Viewer::PerspectiveCamera()), tokens::kCamera);
    EXPECT_EQ(viewer_->ActiveCamera(), Viewer::PerspectiveCamera());
    ASSERT_FALSE(viewer_->Cameras().empty());
    EXPECT_EQ(viewer_->Cameras().front(), Viewer::PerspectiveCamera());
    EXPECT_TRUE(Viewer::PerspectiveCamera().HasPrefix(Viewer::SessionRoot()));
    EXPECT_EQ(viewer_->Settings(), Path("/Render/Settings"));
}

TEST_F(ViewerTest, OpensFramingTheScene) {
    const Camera camera = CameraOf(scene_, Viewer::PerspectiveCamera(), 64, 64);
    const glm::dvec3 toCube = glm::normalize(-camera.Eye());
    EXPECT_GT(glm::dot(toCube, camera.Forward()), 0.999);
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
    unsigned char* texels = stbi_load(path.string().c_str(), &width, &height, &channels, 4);
    ASSERT_NE(texels, nullptr);
    EXPECT_GT(width, 100);
    EXPECT_GT(height, 100);
    const std::size_t rowLength = static_cast<std::size_t>(width);
    const unsigned char* centre =
        texels + 4 * (static_cast<std::size_t>(height / 2) * rowLength + rowLength / 2);
    const unsigned char* corner = texels;
    EXPECT_GT(centre[0] + centre[1] + centre[2], corner[0] + corner[1] + corner[2] + 60);
    stbi_image_free(texels);
    std::filesystem::remove(path);
}

TEST_F(ViewerTest, ShowsEveryRenderVarTheBackendCanDraw) {
    const std::vector<std::string>& vars = viewer_->Delegate().Info().viewVars;
    EXPECT_EQ(vars.size(), kRenderVarNames.size() - 1) << "every one but the opt-in MotionVector";
    ASSERT_TRUE(viewer_->Step());
    // The centre pixel (the cube) and a corner (background) of a capture of `var`.
    // Without the grid and gizmos over the image.
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
        unsigned char* texels = stbi_load(path.string().c_str(), &width, &height, &channels, 4);
        EXPECT_NE(texels, nullptr) << var;
        const std::size_t row = static_cast<std::size_t>(width);
        const unsigned char* centre =
            texels + 4 * (static_cast<std::size_t>(height / 2) * row + row / 2);
        const std::array<std::array<int, 3>, 2> pixels{
            {{centre[0], centre[1], centre[2]}, {texels[0], texels[1], texels[2]}}};
        stbi_image_free(texels);
        std::filesystem::remove(path);
        return pixels;
    };
    const auto sum = [](const std::array<int, 3>& rgb) { return rgb[0] + rgb[1] + rgb[2]; };
    for (const std::string& var : vars) {
        const auto [centre, corner] = sample(var);
        if (var != "InstanceId") {  // The cube is no instance: id 0, black.
            EXPECT_GT(sum(centre), 0) << var << " shows the cube";
        }
        // Colour shows the clear colour, and sensor vars the background's noise.
        if (var != "LdrColor" && var != "HdrColor" && var != "Electrons" && var != "RawDN") {
            EXPECT_EQ(sum(corner), 0) << var << " shows the background black";
        }
    }
    EXPECT_NE(sample("Normal")[0], sample("PrimId")[0]);
    EXPECT_EQ(viewer_->ViewVar(), "LdrColor") << "captures leave the shown var";

    // SpectralRadiance in grey, the mean over the bands or one of them.
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
    Camera top;
    top.transform = glm::inverse(
        glm::lookAtRH(glm::dvec3(0.0, 0.0, 20.0), glm::dvec3(0.0), glm::dvec3(0.0, 1.0, 0.0)));
    SetCamera(scene_, Path("/Cameras/Top"), top);
    const std::vector<Path> cameras = viewer_->Cameras();
    ASSERT_EQ(cameras.size(), 2u);
    EXPECT_EQ(cameras[1], Path("/Cameras/Top"));
    viewer_->SetActiveCamera(Path("/Cameras/Top"));
    EXPECT_EQ(viewer_->ActiveCamera(), Path("/Cameras/Top"));
    ASSERT_TRUE(viewer_->Step());
    EXPECT_EQ(scene_.Get(Path("/__session/RenderSettings"), tokens::kCameraRel, Path()),
              Path("/Cameras/Top"));

    scene_.Remove(Path("/Cameras"));
    ASSERT_TRUE(viewer_->Step());
    EXPECT_EQ(viewer_->ActiveCamera(), Viewer::PerspectiveCamera());
    EXPECT_THROW(viewer_->SetActiveCamera(Path("/World/Cube")), std::invalid_argument);
    EXPECT_THROW(viewer_->SetActiveCamera(Path("/Missing")), std::invalid_argument);
}

TEST_F(ViewerTest, RecreatesItsCameraIfTheSceneRemovesIt) {
    scene_.Remove(Viewer::SessionRoot());
    ASSERT_TRUE(viewer_->Step());
    EXPECT_TRUE(scene_.Has(Viewer::PerspectiveCamera()));
    EXPECT_EQ(viewer_->ActiveCamera(), Viewer::PerspectiveCamera());
}

TEST_F(ViewerTest, FrameAimsAtTheGivenPrims) {
    scene_.Define("/World/Far", tokens::kCube)
        .Set(tokens::kSize, 1.0)
        .Set(tokens::kTransform, glm::translate(glm::dmat4(1.0), glm::dvec3(30.0, 0.0, 0.0)))
        .Set(tokens::kExtent, std::vector<float>{-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f});
    const std::vector<Path> far{Path("/World/Far")};
    viewer_->Frame(far);
    const Camera camera = CameraOf(scene_, Viewer::PerspectiveCamera(), 64, 64);
    const glm::dvec3 toFar = glm::normalize(glm::dvec3(30.0, 0.0, 0.0) - camera.Eye());
    EXPECT_GT(glm::dot(toFar, camera.Forward()), 0.999);
}

TEST_F(ViewerTest, CameraFromViewCopiesTheActiveCamera) {
    const Path made = viewer_->CameraFromView(Path("/Cameras/Saved"));
    EXPECT_EQ(made, Path("/Cameras/Saved"));
    EXPECT_EQ(scene_.TypeOf(made), tokens::kCamera);
    const Camera saved = CameraOf(scene_, made, 64, 64);
    const Camera view = CameraOf(scene_, Viewer::PerspectiveCamera(), 64, 64);
    EXPECT_NEAR(glm::length(saved.Eye() - view.Eye()), 0.0, 1e-6);
    EXPECT_THROW(viewer_->CameraFromView(made), std::invalid_argument);
    EXPECT_THROW(viewer_->CameraFromView(Path("/__session/Other")), std::invalid_argument);
}

TEST_F(ViewerTest, FrameCallbacksRunEveryStepUntilRemoved) {
    std::vector<FrameInfo> seen;
    const std::uint64_t id =
        viewer_->OnFrame([&](Viewer&, const FrameInfo& info) { seen.push_back(info); });
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
        viewer.GetScene().Set(Path("/Render/Settings"), "ernie:wireframe", true);
    });
    viewer_->Step();
    EXPECT_TRUE(scene_.Get(Path("/__session/RenderSettings"), "ernie:wireframe", false));
}

TEST_F(ViewerTest, CallbacksRunInFrameUiPreRenderPostRenderOrder) {
    std::vector<std::string> order;
    viewer_->OnPostRender([&](Viewer&, const FrameInfo&) { order.push_back("post"); });
    viewer_->OnPreRender([&](Viewer&, const FrameInfo&) { order.push_back("pre"); });
    viewer_->OnUi([&](Viewer&, const FrameInfo&) {
        EXPECT_NE(ImGui::GetCurrentContext(), nullptr);
        ImGui::Begin("Overlay");
        ImGui::TextUnformatted("inside the ImGui frame");
        ImGui::End();
        order.push_back("ui");
    });
    viewer_->OnFrame([&](Viewer&, const FrameInfo&) { order.push_back("frame"); });
    viewer_->Step();
    EXPECT_EQ(order, (std::vector<std::string>{"frame", "ui", "pre", "post"}));
}

TEST_F(ViewerTest, PreRenderEditsShowInTheFrameAndPostRenderEditsInTheNext) {
    viewer_->OnPreRender([](Viewer& viewer, const FrameInfo& info) {
        viewer.GetScene().Set(Path("/Render/Settings"), "test:pre", static_cast<int>(info.frame));
    });
    viewer_->OnPostRender([](Viewer& viewer, const FrameInfo& info) {
        viewer.GetScene().Set(Path("/Render/Settings"), "test:post", static_cast<int>(info.frame));
    });
    const Path session("/__session/RenderSettings");
    viewer_->Step();
    EXPECT_EQ(scene_.Get(session, "test:pre", -1), 0) << "mirrored into this frame's settings";
    EXPECT_EQ(scene_.Get(session, "test:post", -1), -1) << "set after the frame rendered";
    viewer_->Step();
    EXPECT_EQ(scene_.Get(session, "test:pre", -1), 1);
    EXPECT_EQ(scene_.Get(session, "test:post", -1), 0);
}

TEST_F(ViewerTest, PreAndPostRenderExceptionsPropagateAndTheViewerKeepsWorking) {
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
    const std::vector<std::uint64_t> ids{viewer_->OnFrame(count), viewer_->OnUi(count),
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
    // Alone in its dock, as tabs hidden behind the built-in bottom panels would not draw.
    for (const char* title : {"Log", "Stats", "Overlays"}) {
        viewer_->RemovePanel(title);
    }
    CountingPanel& left = viewer_->AddPanel<CountingPanel>("Left Panel", Dock::Left, draws);
    viewer_->AddPanel<CountingPanel>("Bottom Panel", Dock::Bottom, draws);
    viewer_->AddPanel<CountingPanel>("Floating Panel", Dock::Floating, draws);
    EXPECT_EQ(left.GetDock(), Dock::Left);
    viewer_->Step();
    viewer_->Step();
    EXPECT_GE(draws, 6);
    const std::filesystem::path path = TempFile("docks.ini");
    viewer_->SaveLayout(path);
    std::ifstream file(path);
    const std::string text{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    const auto docked = [&](const std::string& title) {
        const std::size_t at = text.find("[Window][" + title + "]");
        if (at == std::string::npos) {
            return false;
        }
        const std::size_t end = text.find("\n\n", at);
        return text.substr(at, end - at).find("DockId=") != std::string::npos;
    };
    EXPECT_TRUE(docked("Left Panel"));
    EXPECT_TRUE(docked("Bottom Panel"));
    EXPECT_TRUE(docked("Properties"));
    EXPECT_FALSE(docked("Floating Panel"));
    std::filesystem::remove(path);
}

TEST_F(ViewerTest, FunctionPanelsDrawAndPanelsAreFoundByTitle) {
    int draws = 0;
    Panel& panel = viewer_->AddPanel("Function", Dock::Bottom, [&](Viewer& viewer) {
        EXPECT_EQ(&viewer, viewer_.get());
        ImGui::TextUnformatted("drawn by a function");
        ++draws;
    });
    EXPECT_EQ(panel.GetDock(), Dock::Bottom);
    EXPECT_EQ(viewer_->FindPanel("Function"), &panel);
    ASSERT_NE(viewer_->FindPanel("Stats"), nullptr);
    EXPECT_EQ(viewer_->FindPanel("Missing"), nullptr);
    viewer_->Step();
    EXPECT_EQ(draws, 1);
    panel.SetVisible(false);
    viewer_->Step();
    EXPECT_EQ(draws, 1);
}

TEST_F(ViewerTest, RendererIsAViewerOptionAndSwitchable) {
TEST_F(ViewerTest, FocusPanelBringsADockedTabToTheFront) {
    // Whether each title's window is the selected tab of its dock, read during the UI.
    std::map<std::string, bool> front;
    viewer_->OnUi([&](Viewer&, const FrameInfo&) {
        for (const char* title : {"Stats", "Log"}) {
            const ImGuiWindow* window = ImGui::FindWindowByName(title);
            front[title] = window != nullptr && window->DockNode != nullptr &&
                           window->DockNode->SelectedTabId == window->TabId;
        }
    });
    viewer_->FindPanel("Log")->SetVisible(false);
    viewer_->FocusPanel("Log");
    EXPECT_TRUE(viewer_->FindPanel("Log")->IsVisible()) << "focusing shows the panel";
    for (int frame = 0; frame < 4; ++frame) {
        viewer_->Step();
    }
    EXPECT_TRUE(front["Log"]);
    EXPECT_FALSE(front["Stats"]);

    viewer_->FocusPanel("Stats");
    for (int frame = 0; frame < 2; ++frame) {
        viewer_->Step();
    }
    EXPECT_TRUE(front["Stats"]);
    EXPECT_FALSE(front["Log"]);
    EXPECT_THROW(viewer_->FocusPanel("Missing"), std::invalid_argument);

    viewer_->AddPanel("Extra", Dock::Bottom, [](Viewer&) {});
    viewer_->FocusPanel("Extra");
    viewer_->RemovePanel("Extra");
    EXPECT_TRUE(viewer_->Step()) << "a removed panel's focus request is dropped";
}

    const std::vector<std::string> renderers = Viewer::Renderers();
    ASSERT_FALSE(renderers.empty());
    EXPECT_EQ(viewer_->Renderer(), "raster");
    EXPECT_NE(std::find(renderers.begin(), renderers.end(), "raster"), renderers.end());
    EXPECT_THROW(viewer_->SetRenderer("no_such_backend"), std::invalid_argument);
    EXPECT_EQ(viewer_->Renderer(), "raster") << "a failed switch keeps the current backend";
    viewer_->SetRenderer("raster");
    EXPECT_TRUE(viewer_->Step());
    EXPECT_GT(viewer_->Stats().triangles, 0u);
    EXPECT_NE(viewer_->Brdfs().Find("lambert"), nullptr);
}

TEST_F(ViewerTest, SyncReportsDataTheRendererCannotUse) {
    EXPECT_NO_THROW(viewer_->Sync());
    scene_.Define("/World/Broken", tokens::kCube).Set(tokens::kSize, -1.0);
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

TEST_F(ViewerTest, CallbackExceptionsPropagateAndTheViewerKeepsWorking) {
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
    panel.Plot("History", [] { return std::vector<float>{1.0f, 3.0f, 2.0f}; });
    viewer_->AddMenuItem("Tools", "Reset", [](Viewer&) {});
    EXPECT_TRUE(viewer_->Step());
    EXPECT_THROW(viewer_->AddWidgetPanel("Controls"), std::invalid_argument);
    EXPECT_THROW(viewer_->AddWidgetPanel("Viewport"), std::invalid_argument);
    EXPECT_THROW(viewer_->AddPanel(nullptr), std::invalid_argument);
    EXPECT_THROW(viewer_->AddMenuItem("", "x", nullptr), std::invalid_argument);
    viewer_->RemovePanel("Controls");
    EXPECT_THROW(viewer_->RemovePanel("Controls"), std::invalid_argument);
    viewer_->RemovePanel("Stats");
    EXPECT_NO_THROW(viewer_->RemovePanel("Stats"));
    EXPECT_TRUE(viewer_->Step());
}

TEST_F(ViewerTest, ViewportTogglesShowOverlaysAndHudItemsList) {
    const Path session("/__session/RenderSettings");
    const auto shown = [&] {
        std::istringstream words(GetToken(scene_, session, tokens::kOverlays, ""));
        return std::set<std::string>(std::istream_iterator<std::string>(words), {});
    };
    scene_.GetOverlay("contacts").Line({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0});
    scene_.GetOverlay("labels").Line({0.0, 0.0, 0.0}, {0.0, 1.0, 0.0});
    std::vector<bool> switched;
    const std::uint64_t toggle = viewer_->AddViewportToggle(
        "Contacts", false, "contacts", [&](Viewer&, bool on) { switched.push_back(on); });
    int asked = 0;
    const std::uint64_t hud = viewer_->AddHudItem("Physics", [&](Viewer&) {
        ++asked;
        return std::string("1 ms");
    });
    viewer_->AddViewportMenuItem("Pause", [](Viewer&) {}, [](const Viewer&) { return true; });
    viewer_->Step();
    EXPECT_FALSE(shown().contains("contacts")) << "its toggle hides it";
    EXPECT_TRUE(shown().contains("labels"));
    EXPECT_TRUE(shown().contains("__viewport"));
    EXPECT_GT(asked, 0);
    EXPECT_FALSE(scene_.GetOverlay("__viewport").Empty()) << "the grid";

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
    EXPECT_TRUE(shown().contains("contacts")) << "toggled overlays follow their toggle alone";
    EXPECT_FALSE(shown().contains("labels"));
    EXPECT_TRUE(scene_.GetOverlay("__viewport").Empty());

    viewer_->SetHudItemShown(hud, false);
    EXPECT_FALSE(viewer_->HudItemShown(hud));
    viewer_->RemoveViewportItem(toggle);
    viewer_->RemoveViewportItem(hud);
    EXPECT_THROW(viewer_->ViewportToggleShown(toggle), std::invalid_argument);
    EXPECT_THROW(viewer_->AddViewportToggle(""), std::invalid_argument);
    EXPECT_THROW(viewer_->AddHudItem("x", nullptr), std::invalid_argument);
    EXPECT_TRUE(viewer_->Step());
}

TEST_F(ViewerTest, DuplicateAndRenameAreSingleUndoSteps) {
    scene_.Define("/World/Cube/Child", tokens::kSphere).Set(tokens::kGeomRadius, 0.5);
    const std::vector<Path> copies = viewer_->Duplicate(std::vector<Path>{Path("/World/Cube")});
    ASSERT_EQ(copies, std::vector<Path>{Path("/World/Cube_01")});
    EXPECT_EQ(scene_.TypeOf("/World/Cube_01"), tokens::kCube);
    EXPECT_DOUBLE_EQ(GetNumber(scene_, "/World/Cube_01", tokens::kSize, 0.0), 2.0);
    EXPECT_TRUE(scene_.Has("/World/Cube_01/Child")) << "with its descendants";
    EXPECT_EQ(viewer_->Selection(), copies);
    EXPECT_EQ(viewer_->Duplicate(std::vector<Path>{Path("/World/Cube")}).front(),
              Path("/World/Cube_02"));

    const Path renamed = viewer_->Rename("/World/Cube_01", "Box");
    EXPECT_EQ(renamed, Path("/World/Box"));
    EXPECT_FALSE(scene_.Has("/World/Cube_01"));
    EXPECT_TRUE(scene_.Has("/World/Box/Child"));
    EXPECT_THROW(viewer_->Rename("/World/Box", "Cube"), std::invalid_argument) << "taken";
    EXPECT_THROW(viewer_->Rename("/World/Box", "bad name"), std::invalid_argument);
    ASSERT_TRUE(viewer_->Undo());
    EXPECT_TRUE(scene_.Has("/World/Cube_01/Child")) << "one step undoes the rename";
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
        unsigned char* texels = stbi_load(path.string().c_str(), &width, &height, &channels, 4);
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
    const std::string text{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    EXPECT_NE(text.find("[Docking]"), std::string::npos);
    EXPECT_NE(text.find("Viewport"), std::string::npos);
    viewer_->LoadLayout(path);
    EXPECT_TRUE(viewer_->Step());
    std::filesystem::remove(path);
    EXPECT_THROW(viewer_->LoadLayout(TempFile("missing.ini")), std::runtime_error);
    EXPECT_THROW(viewer_->LoadLayout(std::filesystem::temp_directory_path()), std::runtime_error);
}

TEST_F(ViewerTest, ShowsRenderErrorsWithoutThrowing) {
    scene_.Define("/World/Broken", tokens::kCube).Set(tokens::kSize, -1.0);
    EXPECT_NO_THROW(viewer_->Step());
    scene_.Remove(Path("/World/Broken"));
    EXPECT_TRUE(viewer_->Step());
}

TEST_F(ViewerTest, RejectsVarsAndSettingsItCannotShowYet) {
    EXPECT_EQ(viewer_->ViewVar(), "LdrColor");
    EXPECT_THROW(viewer_->SetViewVar("NotAVar"), std::invalid_argument);
    EXPECT_THROW(viewer_->SetViewVar("Depth"), std::invalid_argument);
    EXPECT_NO_THROW(viewer_->SetViewVar("SpectralRadiance")) << "as grey, by band or mean";
    EXPECT_NO_THROW(viewer_->SetViewVar("LdrColor"));
    EXPECT_THROW(viewer_->SetSettings(Path("/World/Cube")), std::invalid_argument);
    EXPECT_THROW(viewer_->SetSettings(Path("/__session/RenderSettings")), std::invalid_argument);
    scene_.Define("/Render/Other", tokens::kRenderSettings);
    viewer_->SetSettings(Path("/Render/Other"));
    EXPECT_EQ(viewer_->Settings(), Path("/Render/Other"));
    EXPECT_THROW(viewer_->Capture(TempFile("x.png"), "Depth"), std::invalid_argument);
}

TEST_F(ViewerTest, LightingModesChooseTheSessionLights) {
    EXPECT_EQ(viewer_->GetLighting(), Lighting::Stage) << "the scene has a sun";
    const Path session("/__session/RenderSettings");
    viewer_->Step();
    EXPECT_EQ(scene_.Get(session, tokens::kLightsRel), nullptr);

    viewer_->SetLighting(Lighting::Camera);
    viewer_->Step();
    EXPECT_EQ(scene_.Get(session, tokens::kLightsRel, std::vector<Path>{}),
              std::vector<Path>{Path("/__session/Lights/Camera")});
    const Camera view = CameraOf(scene_, viewer_->ActiveCamera(), 64, 64);
    EXPECT_NEAR(glm::dot(TowardsLight(scene_, Path("/__session/Lights/Camera")), -view.Forward()),
                1.0, 1e-6);

    viewer_->SetLighting(Lighting::Default);
    viewer_->Step();
    EXPECT_EQ(scene_.Get(session, tokens::kLightsRel, std::vector<Path>{}),
              std::vector<Path>{Path("/__session/Lights/Default")});
    EXPECT_EQ(viewer_->GetLighting(), Lighting::Default);

    viewer_->SetLighting(Lighting::Stage);
    viewer_->Step();
    EXPECT_EQ(scene_.Get(session, tokens::kLightsRel), nullptr);
}

TEST_F(ViewerTest, WireframeModesOverrideTheSettingsInTheViewportOnly) {
    const Path session("/__session/RenderSettings");
    scene_.Set(Path("/Render/Settings"), tokens::kWireframe, true);
    viewer_->Step();
    EXPECT_FALSE(viewer_->GetWireframe().has_value());
    EXPECT_EQ(WireframeOf(scene_, session), Wireframe::Overlay) << "follows the settings";

    viewer_->SetWireframe(Wireframe::HiddenLine);
    viewer_->Step();
    EXPECT_EQ(WireframeOf(scene_, session), Wireframe::HiddenLine);
    EXPECT_EQ(WireframeOf(scene_, Path("/Render/Settings")), Wireframe::Overlay)
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
    EXPECT_EQ(prims.light, Path("/World/Sun"));
    EXPECT_TRUE(prims.lights.empty());
}

TEST(ViewerLightingTest, AnUnlitSceneOpensWithTheDefaultLightAndIsLit) {
    try {
        VulkanContext::Create();
    } catch (const std::runtime_error& error) {
        GTEST_SKIP() << "No Vulkan device: " << error.what();
    }
    Scene scene;
    scene.Define("/World/Cube", tokens::kCube).Set(tokens::kSize, 2.0);
    ViewerOptions options;
    options.hidden = true;
    options.width = 320;
    options.height = 240;
    Viewer viewer(scene, options);
    EXPECT_EQ(viewer.GetLighting(), Lighting::Default);
    viewer.Step();
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "ernie_app_unlit_capture.png";
    viewer.Capture(path);
    viewer.Step();
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* texels = stbi_load(path.string().c_str(), &width, &height, &channels, 4);
    ASSERT_NE(texels, nullptr);
    long brightest = 0;
    for (int i = 0; i < width * height; ++i) {
        const unsigned char* texel = texels + 4 * static_cast<std::size_t>(i);
        brightest = std::max<long>(brightest, texel[0] + texel[1] + texel[2]);
    }
    EXPECT_GT(brightest, 200);
    stbi_image_free(texels);
    std::filesystem::remove(path);
}

TEST_F(ViewerTest, ThemesApplyAtRunTime) {
    viewer_->SetTheme(LightTheme());
    EXPECT_EQ(viewer_->GetTheme().panel, LightTheme().panel);
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
