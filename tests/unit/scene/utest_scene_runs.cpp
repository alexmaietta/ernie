// SPDX-License-Identifier: BSD-3-Clause
#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/device/device.h"
#include "ernie/scene/scene.h"
#include "support/devices.h"

namespace ernie {
namespace {

constexpr std::string_view kTables[] = {"hierarchy", "geometry",  "bounds",   "tangents",
                                        "textures",  "semantics", "sensors",  "products",
                                        "materials", "instances", "surfaces", "lights"};

using Tables = std::set<std::string_view>;

Ref<Scene> Stage(device::Device& device) {
    const Ref<Scene> scene = Scene::Create(device);
    scene->Define<Scope>("/Looks");
    scene->Define<Material>(
        "/Looks/Red", {.id = "UsdPreviewSurface", .inputs = {{"diffuseColor", Float3{1, 0, 0}}}});
    scene->Define<World>("/Env0");
    const Prim ball = scene->Define<Sphere>("/Env0/Ball", {.materialBinding = {"/Looks/Red"}});
    scene->Apply<SemanticsLabelsAPI>(ball, "class", {.semanticsLabels = {"rock"}});
    scene->Define<Mesh>("/Env0/Quad", {.points = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}},
                                       .faceVertexIndices = {0, 1, 2, 3},
                                       .faceVertexCounts = {4}});
    scene->Define<Camera>("/Env0/Cam", {.translate = {0, 0, 5}});
    scene->Define<SphereLight>("/Env0/Lamp", {.translate = {0, 2, 2}});
    scene->Clone(*scene->Find("/Env0"), std::vector<std::string>{"/Env1"});
    scene->Define<DistantLight>("/Sun");
    scene->Define<Scope>("/Render");
    scene->Define<RenderVar>("/Render/Color", {.sourceName = RenderVarSource::LdrColor});
    scene->Define<RenderVar>("/Render/Ids", {.sourceName = RenderVarSource::SemanticSegmentation});
    scene->Define<RenderProduct>("/Render/View", {.camera = {"/Env0/Cam"},
                                                  .resolution = {8, 8},
                                                  .orderedVars = {"/Render/Color", "/Render/Ids"}});
    const std::vector<Token> taxonomy = {scene->Token("class")};
    scene->Require(taxonomy);
    scene->Publish({.wait = true}).Wait();
    return scene;
}

std::vector<uint64_t> Versions(Scene& scene) {
    const Snapshot snapshot = scene.Acquire(Stream::Host());
    std::vector<uint64_t> versions;
    for (const std::string_view table : kTables) {
        versions.push_back(snapshot.Version(table));
    }
    scene.Release(snapshot);
    return versions;
}

struct Case {
    const char* name;
    std::function<void(Scene&)> edit;
    Tables ran;
};

TEST(RunsTest, EachEditRunsOnlyTheTablesThatReadIt) {
    const Case cases[] = {
        {"publishing nothing", [](Scene&) {}, {}},
        {"moving a ball",
         [](Scene& s) { s.Find("/Env0/Ball")->Set(Xformable::kTranslate, Double3{1, 0, 0}); },
         {"hierarchy", "bounds", "instances", "lights"}},
        {"recoloring a ball",
         [](Scene& s) { s.Find("/Env0/Ball")->Set(Gprim::kDisplayColor, Float3{0, 1, 0}); },
         {"surfaces"}},
        {"moving cloned points",
         [](Scene& s) {
             s.Find("/Env0/Quad")
                 ->Set(PointBased::kPoints,
                       std::vector<Float3>{{0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}});
         },
         {"geometry", "bounds", "tangents", "instances", "surfaces"}},
        {"editing a material",
         [](Scene& s) { s.Find("/Looks/Red")->Set(UsdPreviewSurface::kRoughness, 0.2f); },
         {"materials"}},
        {"dimming a lamp",
         [](Scene& s) { s.Find("/Env0/Lamp")->Set(LightAPI::kIntensity, 3.0f); },
         {"lights"}},
        {"moving the camera",
         [](Scene& s) { s.Find("/Env0/Cam")->Set(Xformable::kTranslate, Double3{0, 1, 5}); },
         {"hierarchy", "bounds", "instances", "lights"}},
        {"moving owned points in place",
         [](Scene& s) {
             const std::vector<Float3> up = {{0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
             s.Find("/Env0/Quad")->Set(PointBased::kPoints, up);
             s.Publish({.wait = true}).Wait();
             const std::vector<Float3> down = {{0, 0, 2}, {1, 0, 2}, {1, 1, 2}, {0, 1, 2}};
             s.Find("/Env0/Quad")->Set(PointBased::kPoints, down);
         },
         {"geometry", "bounds", "tangents", "instances"}},
        {"zooming the camera",
         [](Scene& s) { s.Find("/Env0/Cam")->Set(Camera::kFocalLength, 70.0f); },
         {"sensors"}},
        {"resizing the product",
         [](Scene& s) { s.Find("/Render/View")->Set(RenderProduct::kResolution, Uint2{16, 16}); },
         {"products"}},
        {"relabeling a ball",
         [](Scene& s) {
             s.Find("/Env1/Ball")->Set(SemanticsLabelsAPI::kSemanticsLabels, "class", {"stone"});
         },
         {"semantics"}},
        {"defining a cube", [](Scene& s) { s.Define<Cube>("/Env0/Box"); },
         Tables(std::begin(kTables), std::end(kTables))},
        {"advancing the clock", [](Scene& s) { s.Publish({.dt = 0.1}).Wait(); }, {}},
        {"removing a ball", [](Scene& s) { s.Remove(*s.Find("/Env1/Ball")); },
         Tables(std::begin(kTables), std::end(kTables))},
    };
    test::ForEachDevice([&](device::Device& device) {
        const Ref<Scene> scene = Stage(device);
        for (const Case& c : cases) {
            const std::vector<uint64_t> before = Versions(*scene);
            c.edit(*scene);
            scene->Publish({.wait = true}).Wait();
            const std::vector<uint64_t> after = Versions(*scene);
            Tables ran;
            for (size_t i = 0; i < std::size(kTables); ++i) {
                if (after[i] != before[i]) {
                    ran.insert(kTables[i]);
                }
            }
            EXPECT_EQ(ran, c.ran) << c.name;
        }
    });
}

}
}
