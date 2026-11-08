// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/import/importer.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/scene/scene.h"
#include "ernie/util/error.h"
#include "support/devices.h"

namespace ernie {
namespace {

using device::Device;
using test::ForEachDevice;

constexpr std::string_view kObj = R"(mtllib shapes.mtl
o Quad
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
vn 0 0 1
vt 0 0
vt 1 0
vt 1 1
vt 0 1
usemtl red
f 1/1/1 2/2/1 3/3/1 4/4/1
o Tri
v 0 0 1
v 2 0 1
v 0 2 1
f -3 -2 -1
)";

constexpr std::string_view kMtl = R"(newmtl red
Kd 1 0 0
map_Kd missing.png
)";

FileBytes Bytes(std::string_view text) {
    auto bytes = std::make_shared<std::vector<std::byte>>(text.size());
    std::transform(text.begin(), text.end(), bytes->begin(), [](char c) { return std::byte(c); });
    return bytes;
}

ImportSource Shapes(std::string_view obj = kObj) {
    return {.file = "shapes.obj",
            .files = {{"shapes.obj", Bytes(obj)}, {"shapes.mtl", Bytes(kMtl)}}};
}

std::vector<Prim> Meshes(const Prim& root) {
    std::vector<Prim> meshes;
    std::vector<Prim> pending = {root};
    while (!pending.empty()) {
        const Prim prim = pending.back();
        pending.pop_back();
        if (prim.IsA<Mesh>()) {
            meshes.push_back(prim);
        }
        for (const Prim& child : prim.Children()) {
            pending.push_back(child);
        }
    }
    std::sort(meshes.begin(), meshes.end(),
              [](const Prim& a, const Prim& b) { return a.Path() < b.Path(); });
    return meshes;
}

Quatd Orient(Scene& scene, const Prim& prim) {
    return scene.Select(prim).Get<Quatd>("orient").Get().at(0);
}

Float3 Scale(Scene& scene, const Prim& prim) {
    return scene.Select(prim).Get<Float3>("scale").Get().at(0);
}

TEST(ImportTest, ReadsObjMeshesTurnedZUpAndScaled) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        const Prim root = scene->Import("/Shapes", Shapes(), {.synchronous = true}).Wait();
        EXPECT_EQ(root.Path(), "/Shapes");
        const std::vector<Prim> meshes = Meshes(root);
        ASSERT_EQ(meshes.size(), 2u);
        EXPECT_EQ(scene->Select(meshes[0]).GetArray<Float3>("points").Get(0).size(), 4u);
        EXPECT_EQ(scene->Select(meshes[1]).GetArray<Float3>("points").Get(0).size(), 3u);
        EXPECT_EQ(scene->Select(meshes[0]).GetArray<Float3>("points").Get(0)[2], (Float3{1, 1, 0}));
        const double half = std::sqrt(0.5);
        EXPECT_EQ(Orient(*scene, root), (Quatd{half, 0, 0, half}));
        EXPECT_EQ(Scale(*scene, root), (Float3{1, 1, 1}));

        const Prim zUp =
            scene->Import("/ZUp", Shapes(), {.synchronous = true, .up = UpAxis::Z}).Wait();
        EXPECT_EQ(Orient(*scene, zUp), Quatd{});

        const Prim scaled =
            scene->Import("/Scaled", Shapes(), {.synchronous = true, .metersPerUnit = 0.01f})
                .Wait();
        EXPECT_EQ(Scale(*scene, scaled), (Float3{0.01f, 0.01f, 0.01f}));
    });
}

TEST(ImportTest, ObjMaterialsWithinAShapeBecomeGeomSubsets) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        constexpr std::string_view kTwoLooks = R"(mtllib shapes.mtl
o Strip
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 2 0 0
v 2 1 0
usemtl red
f 1 2 3 4
usemtl blue
f 2 5 6 3
usemtl none
f 1 3 4
)";
        ImportSource source = Shapes(kTwoLooks);
        source.files["shapes.mtl"] = Bytes("newmtl red\nKd 1 0 0\nnewmtl blue\nKd 0 0 1\n");
        const Prim root = scene->Import("/Strip", source, {.synchronous = true}).Wait();
        const std::vector<Prim> meshes = Meshes(root);
        ASSERT_EQ(meshes.size(), 1u);
        EXPECT_EQ(scene->Select(meshes[0]).GetArray<Float3>("points").Get(0).size(), 6u);
        EXPECT_TRUE(MeshPrim(meshes[0]).MaterialBinding().empty());
        std::vector<GeomSubsetPrim> subsets;
        for (const Prim& child : meshes[0].Children()) {
            if (child.IsA<GeomSubset>()) {
                subsets.emplace_back(child);
            }
        }
        ASSERT_EQ(subsets.size(), 2u);
        for (const GeomSubsetPrim& subset : subsets) {
            EXPECT_EQ(subset.FamilyName(), "materialBind");
            ASSERT_EQ(subset.MaterialBinding().size(), 1u);
            const bool red = subset.MaterialBinding()[0].ends_with("red");
            EXPECT_EQ(subset.Indices(), (std::vector<int32_t>{red ? 0 : 1}));
        }
    });
}

TEST(ImportTest, RejectsBrokenAndUnknownFiles) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        EXPECT_THROW(
            scene->Import("/Bad", Shapes("v 0 0 0\nf 1 2 9\n"), {.synchronous = true}).Wait(),
            Error);
        ImportSource unknown = Shapes();
        unknown.file = "shapes.xyz";
        EXPECT_THROW(scene->Import("/Unknown", unknown, {.synchronous = true}).Wait(), Error);
        EXPECT_FALSE(scene->Find("/Bad"));
    });
}

TEST(ImportTest, CustomReadersReportProgressAndCanBeCancelled) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        std::vector<uint64_t> reported;
        const ImportReader reader{.name = "boxes", .read = [](ImportContext& context) {
                                      ImportPlan plan;
                                      plan.Define<Xform>("/Boxes");
                                      for (uint64_t i = 0; i < 3; ++i) {
                                          context.Report("boxes", i + 1, 3);
                                          plan.Define<Cube>("/Boxes/Box" + std::to_string(i));
                                      }
                                      plan.AddFile("notes.txt", Bytes("hello"));
                                      EXPECT_EQ(plan.PrimCount(), 4u);
                                      return plan;
                                  }};
        const Prim root = scene
                              ->Import("/Boxes", reader,
                                       {.progress =
                                            [&](const ImportProgress& progress) {
                                                reported.push_back(progress.done);
                                            },
                                        .synchronous = true})
                              .Wait();
        EXPECT_EQ(root.Children().size(), 3u);
        EXPECT_TRUE(scene->Find("/Boxes/Box2"));
        EXPECT_FALSE(reported.empty());

        std::atomic<bool> started = false;
        const ImportReader waiting{.name = "waiting", .read = [&](ImportContext& context) {
                                       started = true;
                                       while (!context.Cancelled()) {
                                       }
                                       ImportPlan partial;
                                       partial.Define<Xform>("/Waiting");
                                       return partial;
                                   }};
        ImportJob job = scene->Import("/Waiting", waiting, {});
        while (!started) {
        }
        job.Cancel();
        EXPECT_THROW(job.Wait(), CancelledError);
        EXPECT_FALSE(scene->Find("/Waiting"));
        EXPECT_TRUE(job.Ready());
    });
}

TEST(ImportTest, RegistersImportersByExtension) {
    const std::vector<std::string> extensions = ImporterExtensions();
    EXPECT_NE(std::find(extensions.begin(), extensions.end(), ".obj"), extensions.end());
    EXPECT_NE(FindImporter(".OBJ"), nullptr);
    EXPECT_EQ(FindImporter(".nothing"), nullptr);

    class Nothing final : public Importer {
      public:
        std::string Name() const override {
            return "nothing";
        }

        std::vector<std::string> Extensions() const override {
            return {".nothing"};
        }

        ImportPlan Read(std::string_view, const ImportSource&, ImportContext&) override {
            return {};
        }
    };

    RegisterImporter(std::make_shared<Nothing>());
    ASSERT_NE(FindImporter(".nothing"), nullptr);
    EXPECT_EQ(FindImporter(".nothing")->Name(), "nothing");
    UnregisterImporter(".nothing");
    EXPECT_EQ(FindImporter(".nothing"), nullptr);
}

}
}
