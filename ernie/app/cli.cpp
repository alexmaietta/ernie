// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/cli.h"

#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/scene_json.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/xform.h"
#include "ernie/util/texture_bake.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace ernie::app {

namespace {

constexpr std::string_view kUsage =
    R"(Usage: ernie [scene.json] [options]

Opens Ernie's viewer on a scene file, or on a default scene, with a Demos panel to switch to
any of Ernie's demos. With --viewer null or --output, renders without a window.

  ernie                               the viewer on the default scene
  ernie scene.json                    the viewer on a scene file (JSON, as ernie.save_scene writes)
  ernie scene.json --output out.png   renders the scene file to a PNG
  ernie --demo demo_lighting_illuminants
                                      the viewer with a demo running
  ernie --list                        the demos' names
  ernie bake-texture in.png out.dds   bakes a texture (ernie bake-texture --help)

Options:
)";

/// A scene to look at rather than a demo: the default scene or a
/// scene file, still, for the Demos panel's first entry and renders
/// of scene files.
class SceneDemo : public Demo {
  public:
    SceneDemo(Viewer* viewer, const DemoOptions& options,
              const std::filesystem::path& file)
        : Demo(viewer, options) {
        numFrames = 1;
        if (file.empty()) {
            description = "Ernie's default scene: a cube, a sphere "
                          "and a cylinder under a sun.";
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
    Cube::Define(scene, "/World/Cube")
        .SetSize(2.0)
        .SetDisplayColor({0.8, 0.3, 0.2});
    SetXform(scene, "/World/Cube",
             {.translate = glm::dvec3(-3.0, 0.0, 1.0)});
    Sphere::Define(scene, "/World/Sphere")
        .SetRadius(1.0)
        .SetDisplayColor({0.2, 0.5, 0.8});
    SetXform(scene, "/World/Sphere",
             {.translate = glm::dvec3(0.0, 0.0, 1.0)});
    Cylinder::Define(scene, "/World/Cylinder")
        .SetDisplayColor({0.3, 0.7, 0.3});
    SetXform(scene, "/World/Cylinder",
             {.translate = glm::dvec3(3.0, 0.0, 1.0)});
    DistantLight::Define(scene, "/World/Sun")
        .SetTransform(DistantLightTransform({0.4, -0.6, 0.7}));
    Camera::Define(scene, "/World/Camera")
        .SetFocalLength(24.0)
        .SetTransform(LookAtTransform(
            {0.0, -12.0, 5.0}, {0.0, 0.0, 0.8}, {0.0, 0.0, 1.0}));
}

int BakeTextureMain(std::span<const std::string> args) {
    constexpr std::string_view kBakeUsage =
        R"(Usage: ernie bake-texture input output.dds [options]
       ernie bake-texture --mosaic 4x2 tile1.png ... tile8.png output.dds [options]

Bakes a PNG, JPEG or EXR image into a DDS file for Ernie's textures: every mip, filtered as its
role says, then block-compressed (BC7 for colour and data, BC5 for normal maps), as virtual
textures (ernie:virtualTexture) need.

  --role colour|normal|data    what the image holds (default colour: sRGB colour)
  --format bc1|bc3|bc4|bc5|bc7|rgba8
                               the stored format, in place of the role's
  --linear                     colour stored linear rather than sRGB
  --quality 0-4                BC7's effort (default 1)
  --mosaic COLUMNSxROWS        the inputs are one image's tiles of one size, row by row from the
                               top left (a global mosaic's), baked a tile at a time
)";
    std::vector<std::string> files;
    BakeOptions options;
    std::uint32_t columns = 0;
    std::uint32_t rows = 0;
    try {
        for (std::size_t i = 0; i < args.size(); ++i) {
            const std::string& arg = args[i];
            const auto value = [&]() -> const std::string& {
                if (i + 1 >= args.size()) {
                    throw std::invalid_argument(arg +
                                                " needs a value");
                }
                return args[++i];
            };
            if (arg == "--help" || arg == "-h") {
                std::printf("%s", std::string(kBakeUsage).c_str());
                return 0;
            } else if (arg == "--role") {
                const std::string& role = value();
                if (role == "colour" || role == "color") {
                    options.role = TextureRole::Colour;
                } else if (role == "normal") {
                    options.role = TextureRole::Normal;
                } else if (role == "data") {
                    options.role = TextureRole::Data;
                } else {
                    throw std::invalid_argument("unknown role " +
                                                role);
                }
            } else if (arg == "--format") {
                const std::string& format = value();
                const std::pair<const char*, TextureFormat>
                    kFormats[] = {{"bc1", TextureFormat::Bc1},
                                  {"bc3", TextureFormat::Bc3},
                                  {"bc4", TextureFormat::Bc4},
                                  {"bc5", TextureFormat::Bc5},
                                  {"bc7", TextureFormat::Bc7},
                                  {"rgba8", TextureFormat::Rgba8}};
                const auto it = std::find_if(
                    std::begin(kFormats), std::end(kFormats),
                    [&](const auto& entry) {
                        return format == entry.first;
                    });
                if (it == std::end(kFormats)) {
                    throw std::invalid_argument("unknown format " +
                                                format);
                }
                options.format = it->second;
            } else if (arg == "--linear") {
                options.srgb = false;
            } else if (arg == "--quality") {
                options.quality = std::stoi(value());
            } else if (arg == "--mosaic") {
                const std::string& grid = value();
                const std::size_t x = grid.find('x');
                if (x == std::string::npos) {
                    throw std::invalid_argument(
                        "--mosaic takes COLUMNSxROWS, not " + grid);
                }
                columns = static_cast<std::uint32_t>(
                    std::stoul(grid.substr(0, x)));
                rows = static_cast<std::uint32_t>(
                    std::stoul(grid.substr(x + 1)));
            } else if (arg.starts_with("--")) {
                throw std::invalid_argument("unknown option " + arg);
            } else {
                files.push_back(arg);
            }
        }
        const std::size_t inputs =
            columns > 0 ? std::size_t{columns} * rows : 1;
        if (files.size() != inputs + 1) {
            throw std::invalid_argument(
                "needs " + std::to_string(inputs) +
                (inputs == 1 ? " input" : " inputs") +
                " and an output file");
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ernie bake-texture: %s\n\n%s",
                     error.what(), std::string(kBakeUsage).c_str());
        return 2;
    }
    try {
        const std::filesystem::path output = files.back();
        files.pop_back();
        const TextureData texture =
            columns > 0
                ? BakeMosaic(std::vector<std::filesystem::path>(
                                 files.begin(), files.end()),
                             columns, output, options)
                : BakeTexture(files[0], output, options);
        std::printf("%s: %u x %u, %u mips, %llu bytes\n",
                    output.string().c_str(), texture.width,
                    texture.height, texture.mips,
                    static_cast<unsigned long long>(
                        std::filesystem::file_size(output)));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ernie bake-texture: %s\n",
                     error.what());
        return 1;
    }
    return 0;
}

int ErnieMain(std::span<const std::string> args,
              std::string_view version) {
    if (!args.empty() && args.front() == "bake-texture") {
        return BakeTextureMain(args.subspan(1));
    }
    for (const std::string& arg : args) {
        if (arg == "--version") {
            std::printf("ernie %.*s\n",
                        static_cast<int>(version.size()),
                        version.data());
            return 0;
        }
    }
    DemoOptions options;
    try {
        options = ParseDemoOptions(args);
    } catch (const std::invalid_argument& error) {
        std::fprintf(stderr, "ernie: %s\n\n%s%s", error.what(),
                     std::string(kUsage).c_str(),
                     std::string(DemoOptionsHelp()).c_str());
        return 2;
    }
    if (options.help) {
        std::printf("%s%s  --version              show Ernie's "
                    "version and exit\n",
                    std::string(kUsage).c_str(),
                    std::string(DemoOptionsHelp()).c_str());
        return 0;
    }
    // Without --demo, a first argument that is no option is a scene
    // file; the other arguments are the demo's own (such as --views
    // 64).
    std::filesystem::path file;
    if (options.demo.empty() && !options.extra.empty() &&
        !options.extra.front().starts_with("-")) {
        file = options.extra.front();
        options.extra.erase(options.extra.begin());
    }
    std::vector<DemoInfo> demos = RegisteredDemos();
    if (options.list) {
        for (const DemoInfo& info : demos) {
            std::printf("%s\n", info.Id().c_str());
        }
        return 0;
    }
    if (!file.empty() && !std::filesystem::exists(file)) {
        std::fprintf(stderr, "ernie: no file %s\n",
                     file.string().c_str());
        return 2;
    }
    // The scene first, then the demos; --demo starts on one of those.
    DemoInfo scene{
        "scene", file.empty() ? "default" : file.filename().string(),
        "",
        [file](Viewer* viewer,
               const DemoOptions& o) -> std::shared_ptr<Demo> {
            return std::make_shared<SceneDemo>(viewer, o, file);
        }};
    demos.insert(demos.begin(), std::move(scene));
    std::size_t start = 0;
    if (!options.demo.empty()) {
        const std::optional<std::size_t> found =
            FindDemo(demos, options.demo);
        if (!found) {
            std::fprintf(stderr,
                         "ernie: no demo named '%s' (ernie --list "
                         "lists them)\n",
                         options.demo.c_str());
            return 2;
        }
        start = *found;
    }
    // Writing a file renders without a window, unless headless is
    // asked for.
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
