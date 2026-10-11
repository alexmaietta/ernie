// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/cli.h"

#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/scene_json.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/xform.h"

#include <cstdio>
#include <exception>
#include <filesystem>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace ernie::app {

namespace {

constexpr std::string_view kUsage = R"(Usage: ernie [scene.json] [options]

Opens Ernie's viewer on a scene file, or on a default scene, with a Demos panel to switch to
any of Ernie's demos. With --viewer null or --output, renders without a window.

  ernie                               the viewer on the default scene
  ernie scene.json                    the viewer on a scene file (JSON, as ernie.save_scene writes)
  ernie scene.json --output out.png   renders the scene file to a PNG
  ernie --demo lighting/illuminants   the viewer with a demo running
  ernie --list                        the demos' names

Options:
)";

/// A scene to look at rather than a demo: the default scene or a scene file, still, for the
/// Demos panel's first entry and renders of scene files.
class SceneDemo : public Demo {
  public:
    SceneDemo(Viewer* viewer, const DemoOptions& options, const std::filesystem::path& file)
        : Demo(viewer, options) {
        numFrames = 1;
        if (file.empty()) {
            description = "Ernie's default scene: a cube, a sphere and a cylinder under a sun.";
            DefineDefaultScene(GetScene());
        } else {
            description = "The scene file " + file.string() + ".";
            LoadSceneFile(GetScene(), file);
        }
    }

    // A scene has no checks of its own beyond rendering.
    void TestFinal() override {
        RenderImage();
    }
};

}  // namespace

void DefineDefaultScene(Scene& scene) {
    GroundPlane::Define(scene, "/World/Ground");
    Cube::Define(scene, "/World/Cube").SetSize(2.0).SetDisplayColor({0.8, 0.3, 0.2});
    SetXform(scene, "/World/Cube", {.translate = glm::dvec3(-3.0, 0.0, 1.0)});
    Sphere::Define(scene, "/World/Sphere").SetRadius(1.0).SetDisplayColor({0.2, 0.5, 0.8});
    SetXform(scene, "/World/Sphere", {.translate = glm::dvec3(0.0, 0.0, 1.0)});
    Cylinder::Define(scene, "/World/Cylinder").SetDisplayColor({0.3, 0.7, 0.3});
    SetXform(scene, "/World/Cylinder", {.translate = glm::dvec3(3.0, 0.0, 1.0)});
    DistantLight::Define(scene, "/World/Sun").SetTransform(DistantLightTransform({0.4, -0.6, 0.7}));
    Camera::Define(scene, "/World/Camera")
        .SetFocalLength(24.0)
        .SetTransform(LookAtTransform({0.0, -12.0, 5.0}, {0.0, 0.0, 0.8}, {0.0, 0.0, 1.0}));
}

int ErnieMain(std::span<const std::string> args, std::string_view version) {
    for (const std::string& arg : args) {
        if (arg == "--version") {
            std::printf("ernie %.*s\n", static_cast<int>(version.size()), version.data());
            return 0;
        }
    }
    DemoOptions options;
    try {
        options = ParseDemoOptions(args);
    } catch (const std::invalid_argument& error) {
        std::fprintf(stderr, "ernie: %s\n\n%s%s", error.what(), std::string(kUsage).c_str(),
                     std::string(DemoOptionsHelp()).c_str());
        return 2;
    }
    if (options.help) {
        std::printf("%s%s  --version              show Ernie's version and exit\n",
                    std::string(kUsage).c_str(), std::string(DemoOptionsHelp()).c_str());
        return 0;
    }
    // The first argument that is no option is a scene file.
    std::filesystem::path file;
    for (auto it = options.extra.begin(); it != options.extra.end(); ++it) {
        if (!it->starts_with("-")) {
            file = *it;
            options.extra.erase(it);
            break;
        }
    }
    std::vector<DemoInfo> demos = RegisteredDemos();
    if (options.list) {
        for (const DemoInfo& info : demos) {
            std::printf("%s\n", info.Id().c_str());
        }
        return 0;
    }
    if (!file.empty() && !std::filesystem::exists(file)) {
        std::fprintf(stderr, "ernie: no file %s\n", file.string().c_str());
        return 2;
    }
    // The scene first, then the demos; --demo starts on one of those.
    DemoInfo scene{"scene", file.empty() ? "default" : file.filename().string(), "",
                   [file](Viewer* viewer, const DemoOptions& o) -> std::shared_ptr<Demo> {
                       return std::make_shared<SceneDemo>(viewer, o, file);
                   }};
    demos.insert(demos.begin(), std::move(scene));
    std::size_t start = 0;
    if (!options.demo.empty()) {
        const std::optional<std::size_t> found = FindDemo(demos, options.demo);
        if (!found) {
            std::fprintf(stderr, "ernie: no demo named '%s' (ernie --list lists them)\n",
                         options.demo.c_str());
            return 2;
        }
        start = *found;
    }
    // Writing a file renders without a window, unless headless is asked for.
    if (!options.output.empty() && options.Windowed()) {
        options.viewer = false;
    }
    try {
        DemoRunner runner(options, "Ernie");
        runner.SetCatalog(std::move(demos), start);
        runner.Run(runner.Create(start));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ernie: %s\n", error.what());
        return 1;
    }
    return 0;
}

}  // namespace ernie::app
